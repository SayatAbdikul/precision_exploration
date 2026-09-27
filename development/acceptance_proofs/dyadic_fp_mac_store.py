"""Finite-code MAC output-store proof for saturating dyadic FP formats.

The proof compares the exact-real sum plus original bias with the declared
ordered FP64 Model C store. It assumes identical finite stored input operands.
It does not issue graph acceptance or assert native backend conformance.
"""
from collections import Counter
from fractions import Fraction
from math import prod

from public.analysis.phase3.finite_fp64_bounds import finite_codebook
from public.analysis.phase3.finite_fp64_nonmac import exact_grid
from public.analysis.phase3.fixed_mac_bounds import quantum
from public.analysis.phase3.fp64_bounds import error_bound
from public.inference.reference.arithmetic import encode, format_named
from public.inference.tensor import Encoding, parse_encoding


FORMATS = {'fp4_e2m1', 'fp5_e2m2', 'fp6_e2m3', 'fp6_e3m2', 'fp7_e3m3', 'fp8_e4m3fn'}
MACS = {'linear', 'conv2d', 'depthwise_conv2d'}
ACCUMULATOR = 'fp64_e11m52_accumulator'


def _domain(document, name):
    encoding = parse_encoding(document)
    if (not isinstance(encoding, Encoding) or encoding.format != name or
            encoding.scales != (Fraction(1),) or encoding.axis is not None or
            encoding.block_size is not None):
        raise ValueError('proof requires an unscaled scalar dyadic FP encoding')
    fmt = format_named(name)
    if fmt.manifest['rounding'] != 'rne' or fmt.manifest['overflow'] != 'saturate':
        raise ValueError('proof requires a saturating RNE output format')
    values = finite_codebook(encoding)
    if not values or any(v.denominator & (v.denominator - 1) for v in values.values()):
        raise ValueError('proof requires a nonempty exact dyadic finite codebook')
    return values


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _floor(value):
    return value.numerator // value.denominator


def prove_graph(graph, name):
    if name not in FORMATS:
        raise ValueError('format is outside the saturating dyadic FP proof domain')
    fmt, accumulator = format_named(name), format_named(ACCUMULATOR)
    domains, rows = dict(graph['inputs']), []
    for node in graph['nodes']:
        attrs, op = node['attrs'], node['op']
        source = domains[node['inputs'][0]]
        domains[node['name']] = attrs.get('output', source)
        if op not in MACS:
            continue
        if attrs.get('accumulator') != ACCUMULATOR:
            raise ValueError('MAC does not use the declared FP64 accumulator')
        x_values, out_values = _domain(source, name), _domain(attrs['output'], name)
        weight = graph['constants'][node['inputs'][1]]
        w_values = _domain(weight['encoding'], name)
        if not set(weight['codes']) <= set(w_values):
            raise ValueError('encoded weight contains a nonfinite or invalid code')
        k, channels = prod(weight['shape'][1:]), weight['shape'][0]
        if len(weight['codes']) != k*channels:
            raise ValueError('encoded weight population mismatch')
        biases = attrs.get('bias', ['0']*channels)
        if len(biases) != channels:
            raise ValueError('bias population mismatch')
        q = quantum(x_values.values()) * quantum(w_values.values())
        ordered = sorted(set(out_values.values()))
        # Distinct numeric values omit the two signed-zero codes. Zero is an
        # additional code boundary because the output sign bit can change there.
        thresholds = sorted({(a+b)/2 for a, b in zip(ordered, ordered[1:])} | {Fraction(0)})
        residues = {t % q for t in thresholds}
        maximum_x = max(abs(value) for value in x_values.values())
        methods, pending = Counter(), []
        for channel in range(channels):
            counts = Counter(weight['codes'][channel*k:(channel+1)*k])
            maximum_dot = maximum_x * sum((abs(w_values[code])*count for code, count in counts.items()), Fraction(0))
            b = Fraction(biases[channel])
            bound = error_bound(maximum_dot, k, absolute_bias=abs(b))
            e = bound['absolute_error_bound']
            if not bound['overflow_excluded'] or not exact_grid(q, maximum_dot) or 2*e >= q:
                pending.append({'channel': channel, 'reason': 'exact-prefix, overflow or narrow-error precondition failed'})
                continue
            distance = min(min((r-b) % q, (b-r) % q) for r in residues)
            if distance > e:
                methods['strict_threshold_margin'] += 1
                continue
            # Only exact dot-product lattice sites within the conservative
            # Model C error radius of a store threshold can change codes.
            # Enumerate those sites, including signed zero, and compare the
            # actual bias store and final FP64 addition with the exact value.
            stored_bias = accumulator.rounded(b)
            if not isinstance(stored_bias, Fraction):
                pending.append({'channel': channel, 'reason': 'nonfinite or signed-zero bias store'})
                continue
            sensitive = set()
            for threshold in thresholds:
                lo = _ceil((threshold-b-e)/q)
                hi = _floor((threshold-b+e)/q)
                for index in range(lo, hi+1):
                    s = index*q
                    if abs(s) <= maximum_dot:
                        sensitive.add(s)
            mismatches = []
            for s in sorted(sensitive):
                ideal = s+b
                rounded = accumulator.rounded(s+stored_bias)
                if encode(fmt, ideal) != encode(fmt, rounded):
                    mismatches.append({'dot': str(s), 'exact': str(ideal), 'fp64': str(rounded),
                                       'reference_code': encode(fmt, ideal), 'model_c_code': encode(fmt, rounded)})
                    if len(mismatches) == 4:
                        break
            if mismatches:
                pending.append({'channel': channel, 'reason': 'output code differs at a sensitive lattice site',
                                'examples': mismatches})
            else:
                methods['all_sensitive_lattice_sites_equal'] += 1
        rows.append({'node': node['name'], 'channels': channels, 'product_quantum': str(q),
                     'threshold_residue_count': len(residues), 'methods': dict(methods),
                     'pending_channels': pending})
    if not rows:
        raise ValueError('graph contains no supported MACs')
    return {'status': 'conditional_output_codes_proven' if all(not row['pending_channels'] for row in rows) else 'pending',
            'mac_nodes': len(rows), 'channels': sum(row['channels'] for row in rows),
            'methods': dict(sum((Counter(row['methods']) for row in rows), Counter())),
            'pending_channels': sum(len(row['pending_channels']) for row in rows), 'records': rows,
            'scope': 'identical finite stored operands, exact FP64 MAC prefixes, original bias and saturating RNE output store',
            'excludes': ['nonfinite activation codes', 'other graph operators', 'native C++/CUDA pilot', 'graph acceptance']}
