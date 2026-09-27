"""Conservative FP64 mean/store proof for scaled binary and ternary.

This is development evidence. It never admits a graph or bypasses native pilots.
All arithmetic in the error bound and output threshold distances is rational.
"""
from fractions import Fraction
from math import prod

from public.analysis.phase3.fixed_mac_bounds import quantum
from public.analysis.phase3.fixed_nonmac import finite_values
from public.analysis.phase3.fp64_bounds import error_bound, UNIT_ROUNDOFF, HALF_MIN_SUBNORMAL, MAX_FINITE
from public.inference.reference.arithmetic import format_named
from public.inference.reference.operators import pool2d
from public.inference.tensor import Encoding, Tensor, parse_encoding


def prove_mean(source, output, count):
    if type(count) is not int or count < 1:
        raise ValueError('mean count must be a positive integer')
    for domain in (source, output):
        if (not isinstance(domain, Encoding) or domain.axis is not None or domain.block_size is not None
                or domain.format not in {'binary_pm1', 'ternary'}):
            raise ValueError('proof requires scalar mapped binary/ternary domains')
        fmt = format_named(domain.format)
        if fmt.manifest['rounding'] != 'rne' or fmt.manifest['overflow'] != 'saturate':
            raise ValueError('proof requires saturating nearest output stores')
    values, outputs = finite_values(source), finite_values(output)
    # Keep the offset: an odd number of +/-s cannot sum to zero.
    origin = count*min(values)
    step = quantum([v-min(values) for v in values])
    last = (count*max(values)-origin)/step
    if last.denominator != 1:
        raise ValueError('sum domain is not on its declared lattice')
    last = int(last)
    maximum = max(abs(v) for v in values)
    summation = error_bound(count*maximum, count)
    mean_error = summation['absolute_error_bound']/count
    bound = mean_error+UNIT_ROUNDOFF*(maximum+mean_error)+HALF_MIN_SUBNORMAL
    margin, ties = None, 0
    for a, b in zip(outputs, outputs[1:]):
        threshold = (a+b)/2
        position = (count*threshold-origin)/step
        floor = position.numerator//position.denominator
        for i in {max(0, min(last, j)) for j in (floor-1, floor, floor+1, floor+2)}:
            distance = abs((origin+i*step)/count-threshold)
            if distance == 0:
                ties += 1
            elif margin is None or distance < margin:
                margin = distance
    no_overflow = summation['overflow_excluded'] and maximum+bound <= MAX_FINITE
    safe = no_overflow and ties == 0 and margin is not None and bound < margin
    return {'status': 'exact_output_codes' if safe else 'pending', 'count': count,
            'sum_lattice_origin': str(origin), 'sum_lattice_step': str(step),
            'sum_error_bound': str(summation['absolute_error_bound']), 'mean_error_bound': str(bound),
            'minimum_nonzero_margin': str(margin), 'possible_exact_threshold_ties': ties,
            'overflow_excluded': no_overflow,
            'scope': 'every order of finite stored operands; one FP64 rounding per addition and one after exact division'}


def cancellation_witness(source, output, count):
    if source.format != 'binary_pm1' or count % 2:
        return None
    values = finite_values(source)
    for sequence in (list(values[:1])*(count//2)+list(values[-1:])*(count//2),
                     list(values[-1:])*(count//2)+list(values[:1])*(count//2)):
        tensor = Tensor.quantize(sequence, (1, 1, 1, count), source)
        exact = Tensor.quantize([sum(sequence, Fraction(0))/count], (1, 1, 1, 1), output)
        actual = pool2d(tensor, kind='average', kernel_size=(1, count), output=output,
                        accumulator='fp64_e11m52_accumulator')
        if actual.codes != exact.codes:
            return {'input': tensor.document(), 'exact_output': exact.document(), 'FP64_output': actual.document(),
                    'scope': 'reference-operator witness; no native pilot claimed'}
    return None


def prove_graph_means(graph, shapes):
    domains, rows = dict(graph['inputs']), []
    for node in graph['nodes']:
        attrs, source = node['attrs'], node['inputs'][0]
        domain = domains[source]
        domains[node['name']] = attrs.get('output', domain)
        if node['op'] != 'adaptive_average_pool2d':
            continue
        if attrs.get('accumulator') != 'fp64_e11m52_accumulator' or attrs.get('output_size', 1) not in (1, [1, 1]):
            raise ValueError('proof requires the original FP64 global average policy')
        x, y = parse_encoding(domain), parse_encoding(attrs['output'])
        count = prod(shapes[source][-2:])
        proof = prove_mean(x, y, count)
        rows.append({'node': node['name'], 'source': domain, 'output': attrs['output'], 'proof': proof,
                     'counterexample': cancellation_witness(x, y, count) if proof['status'] == 'pending' else None})
    return {'status': 'mean_stores_proven' if rows and all(r['proof']['status'] == 'exact_output_codes' for r in rows) else 'pending',
            'records': rows, 'graph_acceptance': False,
            'excludes': ['other operators', 'native CPP/CUDA pilot', 'whole-graph admission']}
