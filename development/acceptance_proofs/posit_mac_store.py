"""Conditional all-finite-code posit/quire MAC output-store certificate."""
from collections import Counter
from fractions import Fraction
from math import prod

from public.analysis.phase3.finite_fp64_bounds import finite_codebook
from public.analysis.phase3.fixed_mac_bounds import fixed_mac_bounds, quantum
from public.inference.reference.arithmetic import encode, format_named
from public.inference.tensor import Encoding, parse_encoding


FORMATS = {'posit4_es0', 'posit6_es1', 'posit8_es1'}
MACS = {'linear', 'conv2d', 'depthwise_conv2d'}


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _floor(value):
    return value.numerator // value.denominator


def _domain(document, name):
    encoding = parse_encoding(document)
    if (not isinstance(encoding, Encoding) or encoding.format != name or
            encoding.scales != (Fraction(1),) or encoding.axis is not None or
            encoding.block_size is not None):
        raise ValueError('proof requires an unscaled scalar posit encoding')
    fmt = format_named(name)
    if fmt.manifest['rounding'] != 'rne' or fmt.manifest['overflow'] != 'saturate':
        raise ValueError('proof requires saturating RNE posit output')
    return finite_codebook(encoding)


def prove_graph(graph, name):
    if name not in FORMATS:
        raise ValueError('format is outside the posit proof domain')
    fixed = fixed_mac_bounds(graph)
    if fixed['pending_nodes'] or any(row['status'] != 'exact_products_and_sums_after_bias_store' for row in fixed['records']):
        raise ValueError('fixed MAC grid and headroom proof is incomplete')
    fixed_rows = {row['node']: row for row in fixed['records']}
    fmt = format_named(name)
    domains, rows = dict(graph['inputs']), []
    for node in graph['nodes']:
        attrs, op = node['attrs'], node['op']
        source = domains[node['inputs'][0]]
        domains[node['name']] = attrs.get('output', source)
        if op not in MACS:
            continue
        x_values, out_values = _domain(source, name), _domain(attrs['output'], name)
        weight = graph['constants'][node['inputs'][1]]
        w_values = _domain(weight['encoding'], name)
        if not set(weight['codes']) <= set(w_values):
            raise ValueError('encoded weight contains a NaR or invalid code')
        k, channels = prod(weight['shape'][1:]), weight['shape'][0]
        if len(weight['codes']) != k*channels or len(fixed_rows[node['name']]['channels']) != channels:
            raise ValueError('fixed MAC channel population mismatch')
        biases = attrs.get('bias', ['0']*channels)
        if len(biases) != channels:
            raise ValueError('bias population mismatch')
        q = quantum(x_values.values()) * quantum(w_values.values())
        ordered = sorted(set(out_values.values()))
        thresholds = sorted({(a+b)/2 for a, b in zip(ordered, ordered[1:])})
        residues = {t % q for t in thresholds}
        maximum_x = max(abs(value) for value in x_values.values())
        acc = format_named(attrs['accumulator'])
        methods, pending = Counter(), []
        for channel in range(channels):
            if not fixed_rows[node['name']]['channels'][channel]['products_on_grid'] or not fixed_rows[node['name']]['channels'][channel]['overflow_excluded']:
                raise ValueError('fixed MAC channel lacks exact grid/headroom')
            counts = Counter(weight['codes'][channel*k:(channel+1)*k])
            maximum_dot = maximum_x * sum((abs(w_values[code])*count for code, count in counts.items()), Fraction(0))
            b = Fraction(biases[channel])
            stored_bias = acc.rounded(b)
            if not isinstance(stored_bias, Fraction):
                pending.append({'channel': channel, 'reason': 'nonfinite bias store'})
                continue
            error = abs(b-stored_bias)
            if not error:
                methods['exact_bias_store'] += 1
                continue
            distance = min(min((r-b) % q, (b-r) % q) for r in residues)
            if distance > error:
                methods['strict_threshold_margin'] += 1
                continue
            if 2*error > 16*q:
                pending.append({'channel': channel, 'reason': 'bias error spans too many lattice sites'})
                continue
            sensitive = set()
            for threshold in thresholds:
                lo = _ceil((threshold-b-error)/q)
                hi = _floor((threshold-b+error)/q)
                for index in range(lo, hi+1):
                    s = index*q
                    if abs(s) <= maximum_dot:
                        sensitive.add(s)
            mismatches = []
            for s in sorted(sensitive):
                exact, model = s+b, s+stored_bias
                if encode(fmt, exact) != encode(fmt, model):
                    mismatches.append({'dot': str(s), 'exact': str(exact), 'quire': str(model),
                                       'reference_code': encode(fmt, exact), 'model_c_code': encode(fmt, model)})
                    if len(mismatches) == 4:
                        break
            if mismatches:
                pending.append({'channel': channel, 'reason': 'stored-bias boundary changes output code',
                                'examples': mismatches})
            else:
                methods['all_sensitive_lattice_sites_equal'] += 1
        rows.append({'node': node['name'], 'channels': channels, 'product_quantum': str(q),
                     'threshold_residue_count': len(residues), 'methods': dict(methods),
                     'pending_channels': pending})
    if len(rows) != len(fixed_rows):
        raise ValueError('fixed MAC proof omitted or duplicated a graph node')
    return {'status': 'conditional_output_codes_proven' if all(not row['pending_channels'] for row in rows) else 'pending',
            'mac_nodes': len(rows), 'channels': sum(row['channels'] for row in rows),
            'methods': dict(sum((Counter(row['methods']) for row in rows), Counter())),
            'pending_channels': sum(len(row['pending_channels']) for row in rows), 'records': rows,
            'scope': 'identical finite posit codes, exact quire products/prefixes and stored bias, saturating RNE output code',
            'excludes': ['NaR inputs', 'other graph operators', 'native C++/CUDA pilot', 'graph acceptance']}
