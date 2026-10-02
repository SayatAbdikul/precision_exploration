"""Data-driven scaled code-domain engine (contract v2)."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import time
import numpy as np
from .common import array_hash, digest
from .codebooks import Codebook, quantize, reconstructed
from .certificates import certify, certify_products, PASSTHROUGH
from .accumulators import resolve
from .native import Native
from .oracles import expected_dot


@dataclass
class Coded:
    codes: np.ndarray
    units: np.ndarray
    scale: float
    book: Codebook


@dataclass
class Raw:
    values: np.ndarray


def real(state):
    """Binary64 value of a state: RNE64(code level * scale) for stored codes."""
    return reconstructed(state.units, state.scale, state.book) if isinstance(state, Coded) else state.values


def window_max(values, attrs, pad):
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = attrs['kernel_size']; sh, sw = attrs['stride']; ph, pw = attrs['padding']
    if attrs['dilation'] != [1, 1]:
        raise ValueError('unsupported maxpool dilation')
    padded = np.pad(values, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=pad)
    windows = sliding_window_view(padded, (kh, kw), axis=(-2, -1))
    return windows[:, :, ::sh, ::sw].max(axis=(-2, -1))


def stable_top5(logits):
    """Contract ordering: descending value, ties by ascending class index."""
    return [int(i) for i in np.argsort(-logits, kind='stable')[:5]]


def oracle_dots(x, w, attrs, actual, shift_a, shift_w, policy, events=None, certificate=None):
    """Independent rational checks of fixed output dots of the first image.

    Positions: first, middle and last flat index, one interior position (centre
    channel, centre row and column) and, for a parameterised policy, the first
    output that reports an accumulator event. Value and event word must match.
    """
    n, ci, h, wi = x.shape; co, cg, kh, kw = w.shape
    sh, sw = attrs.get('stride', [1, 1]); ph, pw = attrs.get('padding', [0, 0])
    dh, dw = attrs.get('dilation', [1, 1]); g = attrs.get('groups', 1)
    interior = int(np.ravel_multi_index((0, co // 2, actual.shape[2] // 2, actual.shape[3] // 2), actual.shape))
    selected = {0, actual.size // 2, actual.size - 1, interior}
    if events is not None and events.any():
        selected.add(int(np.flatnonzero(events.ravel())[0]))
    for flat in sorted(selected):
        bn, oc, oy, ox = np.unravel_index(flat, actual.shape)
        pairs = []
        for c in range(cg):
            for ky in range(kh):
                for kx in range(kw):
                    iy = oy * sh - ph + ky * dh; ix = ox * sw - pw + kx * dw
                    a = int(x[bn, (oc // (co // g)) * cg + c, iy, ix]) if 0 <= iy < h and 0 <= ix < wi else 0
                    pairs.append((a, int(w[oc, c, ky, kx])))
        expected, word = expected_dot(pairs, shift_a, shift_w, policy, certificate)
        if np.float64(expected).tobytes() != actual.flat[flat].tobytes():
            raise ValueError(f'independent rational dot mismatch {resolve(policy).name} output {flat}')
        if word is not None and int(events.flat[flat]) != word:
            raise ValueError(f'independent accumulator event mismatch {resolve(policy).name} output {flat}')
    return len(selected)


OPERATORS = {'input': set(), 'output': set(), 'identity': set(), 'flatten': set(), 'add': set(), 'relu': set(),
             'avgpool': set(), 'linear': set(), 'conv': {'stride', 'padding', 'dilation', 'groups'},
             'maxpool': {'kernel_size', 'stride', 'padding', 'dilation'},
             # contract 2.2
             'relu6': set(), 'hardsigmoid': set(), 'hardswish': set(), 'mul': set()}
BINARY = ('add', 'mul')


def hard_gate(r):
    """Contract 2.2: t = min(max(RNE64(r + 3), +0), 6)."""
    return np.minimum(np.maximum(r + np.float64(3.), 0.), np.float64(6.))


def broadcast_shape(a, b, name):
    """Contract 2.2 mul operands: equal shapes, or (N, C, 1, 1) broadcast over the positions of (N, C, H, W)."""
    a, b = tuple(a), tuple(b)
    if a == b:
        return a
    small, big = sorted((a, b), key=lambda s: int(np.prod(s)))
    if len(big) != 4 or len(small) != 4 or small[:2] != big[:2] or small[2:] != (1, 1):
        raise ValueError(f'{name}: tensor-product operands {a} and {b} are not admitted')
    return big


UNARY_2_2 = ('relu6', 'hardsigmoid', 'hardswish')


def unary(op, s):
    """Contract 2.2 relu6, hardsigmoid and hardswish of a stored (Coded) or unstored (Raw) state."""
    if op == 'relu6':
        r = reconstructed(np.maximum(s.units, 0), s.scale, s.book) if isinstance(s, Coded) else np.maximum(s.values, 0.0)
        return np.minimum(r, np.float64(6.)) + 0.0
    r = real(s)
    if op == 'hardsigmoid':
        return hard_gate(r) / np.float64(6.) + 0.0
    if op == 'hardswish':
        return (r * hard_gate(r)) / np.float64(6.) + 0.0
    raise ValueError(f'{op} is not a contract 2.2 unary operator')


def tensor_product(a, b, cert, name):
    """Contract 2.2 mul: exact code product then the MAC post-operation, or RNE64 of the operand values."""
    shape = broadcast_shape(a.units.shape if isinstance(a, Coded) else a.values.shape,
                            b.units.shape if isinstance(b, Coded) else b.values.shape, name)
    if isinstance(a, Coded) and isinstance(b, Coded):
        if not cert['stored_operands'] or cert['input_codebooks'] != [a.book.id, b.book.id]:
            raise ValueError(f'{name}: tensor-product operands are not the certified code tensors')
        product = (a.units * b.units).astype(np.float64)       # exact: |p| < 2^53 by the product certificate
        value = np.ldexp(product, -(a.book.shift + b.book.shift)) * np.float64(a.scale)
        value = value * np.float64(b.scale)
    else:
        if cert['stored_operands']:
            raise ValueError(f'{name}: tensor-product operands are not the certified code tensors')
        value = real(a) * real(b)
    if value.shape != shape:
        raise ValueError(f'{name}: tensor-product shape')
    return value + 0.0


def validate(nodes):
    """Reject operators, attributes and pooling geometry the contract does not define."""
    for node in nodes:
        op = node['op']; attrs = node.get('attrs') or {}
        if op not in OPERATORS:
            raise ValueError(f'{node["name"]}: operator {op} is not admitted by this contract revision')
        if set(attrs) - OPERATORS[op] or (op == 'maxpool' and set(attrs) != OPERATORS[op]):
            raise ValueError(f'{node["name"]}: unsupported attributes {sorted(set(attrs) ^ OPERATORS[op])}')
        if op == 'maxpool':
            if attrs['dilation'] != [1, 1] or any(2 * p > k for p, k in zip(attrs['padding'], attrs['kernel_size'])) \
                    or min(attrs['kernel_size'] + attrs['stride']) < 1 or min(attrs['padding']) < 0:
                raise ValueError(f'{node["name"]}: unsupported max-pool geometry')
        if len(node['inputs']) != {'input': 0, 'add': 2, 'mul': 2}.get(op, 1):
            raise ValueError(f'{node["name"]}: wrong number of inputs')
        if op in PASSTHROUGH + ('output',) and node.get('store') is not None:
            raise ValueError(f'{node["name"]}: this operator cannot store')


class Engine:
    def __init__(self, export, arrays, backend, policy):
        self.export, self.policy = export, resolve(policy)
        self.nodes = export['nodes']
        validate(self.nodes)
        self.books = {k: Codebook(v) for k, v in export['codebooks'].items()}
        self.certificates = certify(export, arrays)
        self.products = certify_products(export)
        self.native = Native(backend)
        self.mac = {}
        for node in self.nodes:
            if node['op'] not in ('conv', 'linear'):
                continue
            key = node['name']; cert = self.certificates[key]; spec = node['mac']
            self.policy.precondition(cert, self.books[cert['input_codebook']], self.books[cert['weight_codebook']])
            params, info = self.policy.node(cert['product_shift'], cert)
            kind = np.int32 if cert['kernel_operand_bits'] == 32 else np.int64
            w = arrays[spec['weight_units']]
            if node['op'] == 'linear':
                w = w[:, :, None, None]
            scales = np.asarray(spec['weight_scales'], dtype=np.float64)
            bias = np.asarray(arrays[spec['bias']], dtype=np.float64)
            if scales.shape != (w.shape[0],) or bias.shape != (w.shape[0],):
                raise ValueError(f'{key}: weight scale or bias shape does not match the output channels')
            if not (np.isfinite(scales).all() and (scales > 0).all() and np.isfinite(bias).all()):
                raise ValueError(f'{key}: invalid weight scale or bias')
            self.mac[key] = {'cert': cert, 'kind': kind, 'units': w, 'params': params, 'info': info,
                             'w': np.ascontiguousarray(w, dtype=kind),
                             'scales': scales[None, :, None, None], 'bias': bias[None, :, None, None]}

    def _mac(self, node, state, oracle, timing):
        key = node['name']; m = self.mac[key]; cert = m['cert']
        if not isinstance(state, Coded) or state.book.id != cert['input_codebook']:
            raise ValueError(f'{key}: MAC input is not the certified stored code tensor')
        x = state.units
        if node['op'] == 'linear':
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        lo, hi = cert['structural_input_range_units']
        if int(x.min()) < lo or int(x.max()) > hi:
            raise ValueError(f'{key}: input units outside the certified range')
        lo, hi = cert['closure_input_range_units']
        if int(x.min()) < lo or int(x.max()) > hi:
            raise ValueError(f'{key}: input units outside the certified closure range')
        tick = time.perf_counter()
        dot, events = self.native.run(np.ascontiguousarray(x, dtype=m['kind']), m['w'], node['attrs'],
                                      cert['product_shift'], self.policy, cert['kernel_operand_bits'], m['params'])
        timing['mac'] += time.perf_counter() - tick
        checks = 0
        if oracle:
            tick = time.perf_counter()
            checks = oracle_dots(x[:1], m['units'], node['attrs'], dot[:1], cert['input_shift'],
                                 cert['weight_shift'], self.policy, None if events is None else events[:1], cert)
            timing['oracle'] += time.perf_counter() - tick
        n = dot.shape[0]; stats = None; bad = np.zeros(n, dtype=bool); used = dot
        if self.policy.kind == 'sat':
            e = events.reshape(n, -1)
            stats = [{**m['info'], 'elements': int(e.shape[1]), 'saturated_elements': int((e[i] != 0).sum()),
                      'high_clamps': int((e[i] & 0xFFFF).sum(dtype=np.int64)),
                      'low_clamps': int((e[i] >> 16).sum(dtype=np.int64))} for i in range(n)]
        elif self.policy.kind == 'float':
            finite = np.isfinite(dot).reshape(n, -1); e = events.reshape(n, -1)
            bad = ~finite.all(1)
            stats = [{**m['info'], 'elements': int(e.shape[1]), 'nonfinite_elements': int((~finite[i]).sum()),
                      'nonfinite_steps': int(e[i].sum(dtype=np.int64))} for i in range(n)]
            if bad.any():
                # A non-finite dot ends the image (recorded failure). Its slot
                # is zeroed only so that the rest of the batch can continue.
                used = dot.copy(); used[bad] = 0.0
        raw = used * np.float64(state.scale)
        raw = raw * m['scales']
        raw = raw + m['bias']
        if node['op'] == 'linear':
            raw = raw[:, :, 0, 0]
        return raw, dot, checks, stats, bad

    def _value(self, node, args):
        op = node['op']; s = args[0] if args else None
        if op == 'add':
            a, b = real(args[0]), real(args[1])
            if a.shape != b.shape:
                raise ValueError(f'{node["name"]}: residual operands differ in shape')
            return a + b
        if op == 'relu':
            if isinstance(s, Coded):
                return reconstructed(np.maximum(s.units, 0), s.scale, s.book)
            return np.maximum(s.values, 0.0) + 0.0
        if op in UNARY_2_2:
            return unary(op, s)
        if op == 'mul':
            return tensor_product(args[0], args[1], self.products[node['name']], node['name'])
        if op == 'maxpool':
            if isinstance(s, Coded):
                return reconstructed(window_max(s.units, node['attrs'], np.iinfo(np.int64).min), s.scale, s.book)
            return window_max(s.values, node['attrs'], -np.inf) + 0.0
        if op == 'avgpool':
            if not isinstance(s, Coded):
                raise ValueError('average pooling of an unstored tensor is outside the contract')
            count = s.units.shape[-2] * s.units.shape[-1]
            if count * max(abs(s.book.lo), abs(s.book.hi)) >= 2**53:
                raise ValueError('average-pool code sum exceeds the exact binary64 grid')
            summed = s.units.sum(axis=(-2, -1), keepdims=True, dtype=np.int64)
            raw = np.ldexp(summed.astype(np.float64), -s.book.shift)
            raw = raw / np.float64(count)
            return raw * np.float64(s.scale)
        raise ValueError(f'operator {op} is not admitted by this contract revision')

    def run(self, inputs, *, oracle=False, trace='full', capture=None):
        """Run a batch. Returns one record per image; timing covers the batch.

        capture: optional dict that receives the stored code arrays (per storing node) and the output array.
        """
        if trace not in ('full', 'codes', 'none'):
            raise ValueError('unknown trace level')
        inputs = np.asarray(inputs)
        n = inputs.shape[0]
        values = {}; layers = [{} for _ in range(n)]; accumulator = [{} for _ in range(n)]
        alive = np.ones(n, dtype=bool); failure = [None] * n
        timing = {'mac': 0., 'nonmac': 0., 'quantization_trace': 0., 'oracle': 0.}
        oracle_count = 0; start = time.perf_counter(); output = None

        def note(key, entries):
            for i in range(n):
                if alive[i]:
                    layers[i][key] = {k: (v[i] if isinstance(v, list) else v) for k, v in entries.items()}

        def hashes(array):
            return [array_hash(array[i:i + 1]) for i in range(n)] if array is not None else None

        for node in self.nodes:
            key, op = node['name'], node['op']
            args = [values[name] for name in node['inputs']]
            tick = time.perf_counter(); before = timing['mac'] + timing['oracle']
            if op == 'output':
                output = real(args[0]) + 0.0
                break
            passthrough = op in PASSTHROUGH
            pooled_codes = op == 'maxpool' and node.get('store') is None and isinstance(args[0], Coded)
            if passthrough or pooled_codes:
                s = args[0]
                if pooled_codes:
                    # A max-pool that does not quantise forwards codes: the
                    # maximum code level of a window is itself a level.
                    units = window_max(s.units, node['attrs'], np.iinfo(np.int64).min)
                    state = Coded(s.book.codes[np.searchsorted(s.book.units, units)], units, s.scale, s.book)
                    diagnostics = {'code_passthrough': True}
                else:
                    shape = (n, -1) if op == 'flatten' else None
                    if isinstance(s, Coded):
                        state = Coded(s.codes.reshape(shape) if shape else s.codes,
                                      s.units.reshape(shape) if shape else s.units, s.scale, s.book)
                    else:
                        state = Raw(s.values.reshape(shape) if shape else s.values)
                    diagnostics = {'skipped': True}
                values[key] = state
                if trace == 'full':
                    note(key, {'codes': hashes(state.codes) if isinstance(state, Coded) else None,
                               'state': hashes(real(state)), 'raw': None, 'mac': None, 'diagnostics': diagnostics})
                elif trace == 'codes':
                    note(key, {'codes': hashes(state.codes) if isinstance(state, Coded) else None,
                               'diagnostics': diagnostics})
                timing['nonmac'] += time.perf_counter() - tick
                continue
            dot = None
            if op == 'input':
                raw = np.asarray(inputs, dtype=np.float64)
            elif op in ('conv', 'linear'):
                raw, dot, checks, stats, bad = self._mac(node, args[0], oracle and bool(alive[0]), timing)
                oracle_count += checks
                if stats is not None:
                    for i in range(n):
                        if alive[i]:
                            accumulator[i][key] = stats[i]
                for i in np.flatnonzero(bad & alive):
                    failure[i] = {'kind': 'nonfinite accumulator result', 'node': key, **stats[i],
                                  'positive_infinite': int(np.isposinf(dot[i]).sum()),
                                  'negative_infinite': int(np.isneginf(dot[i]).sum())}
                    layers[i][key] = {'codes': None, 'state': None, 'raw': None, 'mac': array_hash(dot[i:i + 1]),
                                      'diagnostics': {'nonfinite_accumulator': True}}
                    alive[i] = False
            else:
                raw = self._value(node, args)
            if not np.isfinite(raw).all():
                raise ValueError(f'{key}: nonfinite raw value')
            timing['nonmac'] += time.perf_counter() - tick - (timing['mac'] + timing['oracle'] - before)
            tick = time.perf_counter()
            store = node.get('store')
            if store is None:
                raw = raw + 0.0
                state = Raw(raw); diagnostics = {'unstored': True}; codes = None
            else:
                book = self.books[store['codebook']]
                codes, units, diagnostics = quantize(raw, store['scale'], book)
                state = Coded(codes, units, float(store['scale']), book)
                if capture is not None:
                    capture[key] = codes
            values[key] = state
            if trace == 'full':
                note(key, {'codes': hashes(codes), 'state': hashes(real(state)), 'raw': hashes(raw),
                           'mac': hashes(dot), 'diagnostics': diagnostics})
            elif trace == 'codes':
                note(key, {'codes': hashes(codes), 'diagnostics': diagnostics})
            timing['quantization_trace'] += time.perf_counter() - tick
        if output is None or output.ndim != 2 or not np.isfinite(output).all():
            raise ValueError('missing or nonfinite output')
        timing['execution'] = time.perf_counter() - start
        if capture is not None:
            capture['output'] = output
        timing['batch_images'] = n
        records = []
        for i in range(n):
            logits = output[i]
            record = {'layers': layers[i], 'output': array_hash(output[i:i + 1]),
                      'top5': stable_top5(logits),
                      'top1_tie_count': int(np.sum(logits == logits.max())),
                      'top1_tied_classes': [int(c) for c in np.flatnonzero(logits == logits.max())],
                      'diagnostic_signature': digest({k: v['diagnostics'] for k, v in layers[i].items()}),
                      'timing': timing, 'oracle_dot_checks': oracle_count if i == 0 else 0}
            if self.policy.parameterised:
                record['accumulator'] = accumulator[i]; record['failure'] = failure[i]
                if failure[i] is not None:
                    # No prediction exists for an image whose accumulator left the finite range.
                    record.update(output=None, top5=None, top1_tie_count=None, top1_tied_classes=None)
            records.append(record)
        return records


def numerical(record):
    keys = ('layers', 'output', 'top5', 'top1_tie_count', 'top1_tied_classes', 'diagnostic_signature')
    return {k: record[k] for k in keys + (('accumulator', 'failure') if 'accumulator' in record else ())}
