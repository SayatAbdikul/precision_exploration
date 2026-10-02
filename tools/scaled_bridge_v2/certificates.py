"""Per-output-channel prefix certificates and lossless accumulator widths.

For a MAC node with input code range [a_lo, a_hi] (a_lo <= 0 <= a_hi) and
weight grid integers w, every term a*w lies in [min(a_lo*w, a_hi*w),
max(a_lo*w, a_hi*w)], an interval that contains zero. Any reduction prefix
(in any order, with any padded positions contributing zero) therefore lies in
the channel interval [lo_c, hi_c] obtained by summing the term intervals.

Three widths are reported per layer:
  signed_bits_absolute    v1 definition: bit_length(max|a| * sum|w|) + 1
  signed_bits_range       smallest two's-complement width holding [lo_c, hi_c]
                          for the input tensor's full code range
  signed_bits_structural  the same after intersecting the input range with
                          [0, inf) when the producer is a ReLU-class operator
                          whose stored codebook contains zero
  signed_bits_closure     (contract 2.2) the same for the input range obtained
                          by interval propagation through the graph (`closure`)
The engine relies on the absolute bound (kernel choice, binary64 exactness)
and checks the structural and the closure input range on every call.
Tensor products (mul, contract 2.2) get a product certificate (`certify_products`).
"""
from __future__ import annotations
import numpy as np
from .common import array_hash
from .codebooks import Codebook, quantize
from .native import operand_width

NONNEGATIVE_PRODUCERS = ('relu', 'relu6', 'hardsigmoid')
PASSTHROUGH = ('identity', 'flatten')
INF = float('inf')


def _hardsigmoid(r):
    """Contract 2.2 hard-sigmoid of one binary64 value (monotone non-decreasing)."""
    if r == -INF:
        return 0.
    if r == INF:
        return 1.
    t = min(max(float(np.float64(r) + np.float64(3.)), 0.), 6.)
    return float(np.float64(t) / np.float64(6.)) + 0.


def _hardswish(r):
    """Contract 2.2 hard-swish of one binary64 value (non-increasing below -1.5, non-decreasing above)."""
    if r == -INF:
        return 0.
    if r == INF:
        return INF
    t = min(max(float(np.float64(r) + np.float64(3.)), 0.), 6.)
    return float((np.float64(r) * np.float64(t)) / np.float64(6.)) + 0.


def _add(a, b):
    with np.errstate(invalid='ignore', over='ignore'):
        return float(np.float64(a) + np.float64(b))


def closure(export):
    """Interval bounds of every node's state (contract 2.2 `closure_certificate`).

    Stored state: ('units', lo, hi, codebook id, scale). Unstored value: ('real', lo, hi) with binary64 bounds,
    possibly infinite. Every map is monotone or is evaluated at its extreme points, so the bounds hold for every
    input image.
    """
    books = {k: Codebook(v) for k, v in export['codebooks'].items()}
    state = {}

    def real(s):
        if s[0] == 'real':
            return s[1], s[2]
        book = books[s[3]]
        return tuple(float(np.ldexp(np.float64(u), -book.shift) * np.float64(s[4])) for u in (s[1], s[2]))

    def store(bounds, spec):
        book = books[spec['codebook']]; units = []
        for value in bounds:
            if np.isfinite(value):
                units.append(int(quantize(np.array([value]), spec['scale'], book)[1][0]))
            else:
                units.append(book.lo if value < 0 else book.hi)
        return ('units', min(units), max(units), book.id, float(spec['scale']))

    for node in export['nodes']:
        key, op = node['name'], node['op']; args = [state[n] for n in node['inputs']]
        if op == 'output':
            continue
        if op in PASSTHROUGH or (op == 'maxpool' and node.get('store') is None and args[0][0] == 'units'):
            state[key] = args[0]; continue
        if op in ('input', 'conv', 'linear'):
            bounds = (-INF, INF)
        elif op in ('relu', 'relu6'):
            lo, hi = real(args[0]); top = 6. if op == 'relu6' else INF
            bounds = (min(max(lo, 0.), top), min(max(hi, 0.), top))
        elif op == 'hardsigmoid':
            lo, hi = real(args[0]); bounds = (_hardsigmoid(lo), _hardsigmoid(hi))
        elif op == 'hardswish':
            lo, hi = real(args[0])
            if lo >= -1.5:
                bounds = (_hardswish(lo), _hardswish(hi))
            elif hi <= -1.5:
                bounds = (_hardswish(hi), _hardswish(lo))
            else:
                bounds = (-0.375, max(_hardswish(lo), _hardswish(hi)))
        elif op in ('maxpool', 'avgpool'):
            bounds = real(args[0])
        elif op == 'add':
            (la, ua), (lb, ub) = real(args[0]), real(args[1])
            bounds = (_add(la, lb), _add(ua, ub))
        elif op == 'mul':
            a, b = args
            if a[0] == 'units' and b[0] == 'units':
                ba, bb = books[a[3]], books[b[3]]
                corners = [p * q for p in (a[1], a[2]) for q in (b[1], b[2])]
                def value(p):
                    return float((np.ldexp(np.float64(p), -(ba.shift + bb.shift)) * np.float64(a[4])) * np.float64(b[4]))
                bounds = (value(min(corners)), value(max(corners)))
            else:
                with np.errstate(invalid='ignore', over='ignore'):
                    corners = [float(np.float64(p) * np.float64(q)) for p in real(a) for q in real(b)]
                bounds = (-INF, INF) if any(np.isnan(corners)) else (min(corners), max(corners))
        else:
            raise ValueError(f'{key}: operator {op} has no closure rule')
        bounds = tuple(v + 0. for v in bounds)
        state[key] = store(bounds, node['store']) if node.get('store') is not None else ('real',) + bounds
    return state


def certify_products(export):
    """Product certificates of tensor multiplies whose operands are both stored (contract 2.2 `mul_certificate`)."""
    nodes = {n['name']: n for n in export['nodes']}
    books = {k: Codebook(v) for k, v in export['codebooks'].items()}
    result = {}
    for node in export['nodes']:
        if node['op'] != 'mul':
            continue
        sources = [producer(nodes, name) for name in node['inputs']]
        if any(s.get('store') is None for s in sources):
            result[node['name']] = {'node': node['name'], 'stored_operands': False,
                                    'rule': 'RNE64(r_a * r_b) of the binary64 operand values'}
            continue
        a, b = (books[s['store']['codebook']] for s in sources)
        bound = max(abs(a.lo), abs(a.hi)) * max(abs(b.lo), abs(b.hi))
        if bound >= 2**53:
            raise ValueError(f'{node["name"]}: code product bound exceeds the exact binary64 integers')
        result[node['name']] = {'node': node['name'], 'stored_operands': True, 'input_codebooks': [a.id, b.id],
                                'input_shifts': [a.shift, b.shift], 'product_shift': a.shift + b.shift,
                                'max_abs_product_units': bound, 'signed_bits_product': bound.bit_length() + 1,
                                'binary64_exact_product': True}
    return result


def signed_bits(lo, hi):
    """Smallest two's-complement width W with -2^(W-1) <= lo and hi <= 2^(W-1)-1."""
    lo, hi = int(lo), int(hi)
    if lo > 0 or hi < 0:
        raise ValueError('prefix interval must contain zero')
    return max(hi.bit_length(), (-lo - 1).bit_length() if lo < 0 else 0) + 1


def producer(nodes, name):
    """The node whose stored codes reach `name`: looks through reshapes and code-forwarding max-pools."""
    node = nodes[name]
    while node['op'] in PASSTHROUGH or (node['op'] == 'maxpool' and node.get('store') is None):
        node = nodes[node['inputs'][0]]
    return node


def certify(export, arrays):
    nodes = {n['name']: n for n in export['nodes']}
    books = {k: Codebook(v) for k, v in export['codebooks'].items()}
    bounds = closure(export)
    result = {}
    for node in export['nodes']:
        if node['op'] not in ('conv', 'linear'):
            continue
        key = node['name']
        source = producer(nodes, node['inputs'][0])
        if source.get('store') is None:
            raise ValueError(f'{key}: MAC input is not a stored code tensor')
        book_in = books[source['store']['codebook']]
        book_w = books[node['mac']['weight_codebook']]
        w = arrays[node['mac']['weight_units']]
        if w.dtype != np.int64 or w.shape[0] != len(node['mac']['weight_scales']):
            raise ValueError(f'{key}: malformed weight export')
        flat = w.reshape(w.shape[0], -1)
        if flat.size and (int(flat.min()) < book_w.lo or int(flat.max()) > book_w.hi):
            raise ValueError(f'{key}: weight units outside their codebook')
        if not np.isin(flat, book_w.units).all():
            raise ValueError(f'{key}: weight units are not codebook levels')
        max_w = max(abs(book_w.lo), abs(book_w.hi))
        if flat.shape[1] * max_w >= 2**62:
            raise ValueError(f'{key}: channel weight sums exceed the certificate arithmetic')
        positive = [int(v) for v in np.where(flat > 0, flat, 0).sum(1)]
        negative = [int(v) for v in np.where(flat < 0, flat, 0).sum(1)]
        a_lo, a_hi = book_in.lo, book_in.hi
        if a_lo > 0 or a_hi < 0:
            raise ValueError(f'{key}: input code range must contain zero')
        structural_lo = 0 if (source['op'] in NONNEGATIVE_PRODUCERS and book_in.has_zero) else a_lo
        max_a = max(abs(a_lo), abs(a_hi))
        absolute = [max_a * (p - n) for p, n in zip(positive, negative)]
        hi = [p * a_hi + n * a_lo for p, n in zip(positive, negative)]
        lo = [p * a_lo + n * a_hi for p, n in zip(positive, negative)]
        s_hi = [p * a_hi + n * structural_lo for p, n in zip(positive, negative)]
        s_lo = [p * structural_lo + n * a_hi for p, n in zip(positive, negative)]
        reach = bounds[node['inputs'][0]]
        if reach[0] != 'units' or reach[3] != book_in.id:
            raise ValueError(f'{key}: closure state of the MAC input is not the stored code tensor')
        c_lo, c_hi = min(reach[1], 0), max(reach[2], 0)
        if c_lo < structural_lo or c_hi > a_hi:
            raise ValueError(f'{key}: closure range is not inside the structural range')
        k_hi = [p * c_hi + n * c_lo for p, n in zip(positive, negative)]
        k_lo = [p * c_lo + n * c_hi for p, n in zip(positive, negative)]
        bound = max(absolute)
        operand = operand_width(max_a, max_w, bound)
        if operand is None:
            raise ValueError(f'{key}: reduction exceeds every certified exact accumulator')
        result[key] = {
            'node': key, 'K': int(flat.shape[1]), 'shape': list(w.shape),
            'input_codebook': book_in.id, 'weight_codebook': book_w.id,
            'input_shift': book_in.shift, 'weight_shift': book_w.shift,
            'product_shift': book_in.shift + book_w.shift,
            'input_range_units': [a_lo, a_hi], 'structural_input_range_units': [structural_lo, a_hi],
            'structural_basis': source['op'] if structural_lo != a_lo else 'codebook range only',
            'per_channel_abs_prefix_units': absolute, 'per_channel_range_units': [lo, hi],
            'per_channel_structural_range_units': [s_lo, s_hi],
            'max_abs_prefix_units': bound,
            'signed_bits_absolute': bound.bit_length() + 1,
            'signed_bits_range': max(signed_bits(a, b) for a, b in zip(lo, hi)),
            'signed_bits_structural': max(signed_bits(a, b) for a, b in zip(s_lo, s_hi)),
            'closure_input_range_units': [c_lo, c_hi], 'closure_basis': source['op'],
            'signed_bits_closure': max(signed_bits(a, b) for a, b in zip(k_lo, k_hi)),
            'groups': int((node.get('attrs') or {}).get('groups', 1)),
            'binary64_exact_dot': bound < 2**53,
            'fp32_all_prefix_exact_sufficient': bound < 2**24,
            'kernel_operand_bits': operand,
            'exact_accumulator': 'int64' if operand == 32 else 'two 64-bit limbs',
            'weight_units_sha256': array_hash(w)}
    return result
