"""Conditional all-code MAC output-store proof for frozen Q1.6/FP64 graphs.

This is arithmetic evidence, not graph acceptance or native pilot evidence.
"""
from collections import Counter
from fractions import Fraction
from math import prod

from public.analysis.phase3.fixed_mac_bounds import quantum
from public.analysis.phase3.finite_fp64_nonmac import exact_grid
from public.analysis.phase3.fp64_bounds import error_bound
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, parse_encoding


MACS = {'linear', 'conv2d', 'depthwise_conv2d'}
ACCUMULATOR = 'fp64_e11m52_accumulator'
FORMAT = 'q1_6'


def _finite_domain(document):
    encoding = parse_encoding(document)
    if (not isinstance(encoding, Encoding) or encoding.format != FORMAT or
            encoding.scales != (Fraction(1),) or encoding.axis is not None or
            encoding.block_size is not None):
        raise ValueError('proof requires unscaled scalar Q1.6 encoding')
    fmt = format_named(encoding.format)
    values = {code: Fraction(fmt.decode(code)) for code in range(1 << fmt.bits)}
    if fmt.manifest['rounding'] != 'rne' or fmt.manifest['overflow'] != 'saturate' or len(set(values.values())) != 1 << fmt.bits:
        raise ValueError('proof requires finite, unique, saturating RNE Q1.6 codes')
    return values


def _threshold_residues(values, q):
    ordered = sorted(values.values())
    return {(a + b) / 2 % q for a, b in zip(ordered, ordered[1:])}


def _distance_to_thresholds(bias, q, residues):
    return min(min((r - bias) % q, (bias - r) % q) for r in residues)


def prove_graph(graph):
    """Prove identical MAC output codes for every finite stored Q1.6 input.

    For each output channel, every exact dot product lies on the product grid.
    The retained Model C error bound covers all ordered FP64 reductions, bias
    storage and final addition. If the shifted grid is farther from every Q1.6
    output threshold than that bound, output codes agree for all input codes.
    Exact ties are admitted only when the entire biased lattice is itself
    exactly representable in FP64, so Model C cannot move the tie.
    """
    domains, rows = dict(graph['inputs']), []
    for node in graph['nodes']:
        attrs, op = node['attrs'], node['op']
        source = domains[node['inputs'][0]]
        domains[node['name']] = attrs.get('output', source)
        if op not in MACS:
            continue
        if attrs.get('accumulator') != ACCUMULATOR:
            raise ValueError('Q1.6 MAC uses a different accumulator')
        inputs = _finite_domain(source)
        outputs = _finite_domain(attrs['output'])
        weight = graph['constants'][node['inputs'][1]]
        weights = _finite_domain(weight['encoding'])
        k, channels = prod(weight['shape'][1:]), weight['shape'][0]
        if len(weight['codes']) != channels * k or not set(weight['codes']) <= set(weights):
            raise ValueError('Q1.6 weight population or code mismatch')
        bias = attrs.get('bias', ['0'] * channels)
        if len(bias) != channels:
            raise ValueError('Q1.6 bias population mismatch')
        q = quantum(inputs.values()) * quantum(weights.values())
        residues = _threshold_residues(outputs, q)
        maximum_input = max(abs(value) for value in inputs.values())
        proofs = Counter()
        pending = []
        for channel in range(channels):
            counts = Counter(weight['codes'][channel*k:(channel+1)*k])
            maximum_dot = maximum_input * sum((abs(weights[code]) * n for code, n in counts.items()), Fraction(0))
            b = Fraction(bias[channel])
            bound = error_bound(maximum_dot, k, absolute_bias=abs(b))
            if not bound['overflow_excluded']:
                raise ValueError('FP64 overflow was not excluded')
            margin = _distance_to_thresholds(b, q, residues)
            if margin > bound['absolute_error_bound']:
                proofs['strict_threshold_margin'] += 1
            elif b % q == 0 and exact_grid(q, maximum_dot + abs(b)):
                proofs['exact_biased_lattice_including_ties'] += 1
            else:
                pending.append({'channel': channel, 'bias': str(b), 'threshold_margin': str(margin),
                                'maximum_error_bound': str(bound['absolute_error_bound'])})
        rows.append({'node': node['name'], 'channels': channels,
                     'product_quantum': str(q), 'threshold_residues': [str(v) for v in sorted(residues)],
                     'proof_methods': dict(proofs), 'pending_channels': pending})
    if not rows:
        raise ValueError('graph contains no Q1.6 MACs')
    return {'status': 'conditional_output_codes_proven' if all(not row['pending_channels'] for row in rows) else 'pending',
            'mac_nodes': len(rows), 'channels': sum(row['channels'] for row in rows),
            'methods': dict(sum((Counter(row['proof_methods']) for row in rows), Counter())),
            'pending_channels': sum(len(row['pending_channels']) for row in rows), 'records': rows,
            'scope': 'same finite stored Q1.6 inputs and retained weights; every ordered FP64 MAC, bias store/add and output RNE code',
            'excludes': ['other graph operators', 'C++/CUDA image conformance', 'graph acceptance and screening']}
