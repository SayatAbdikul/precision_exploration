"""Finite-code FP64 MAC/store threshold proof for scaled binary and ternary."""
from collections import Counter
from fractions import Fraction
from math import prod

from public.analysis.phase3.finite_fp64_bounds import finite_codebook
from public.analysis.phase3.fixed_mac_bounds import quantum
from public.analysis.phase3.fp64_bounds import error_bound
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, parse_encoding


FORMATS = {'binary_pm1', 'ternary'}
MACS = {'linear', 'conv2d', 'depthwise_conv2d'}


def _encoding(document, name, *, weight=False):
    encoding = parse_encoding(document)
    if (not isinstance(encoding, Encoding) or encoding.format != name or encoding.block_size is not None or
            encoding.axis not in ((None, 0) if weight else (None,))):
        raise ValueError('scaled binary/ternary proof requires ordinary scalar or channel encoding')
    fmt = format_named(name)
    if fmt.manifest['rounding'] != 'rne' or fmt.manifest['overflow'] != 'saturate':
        raise ValueError('proof requires saturating RNE output')
    values = finite_codebook(encoding)
    if len(values) < 2:
        raise ValueError('mapped finite codebook is empty')
    return encoding, values


def prove_graph(graph, name):
    if name not in FORMATS:
        raise ValueError('format is outside the mapped proof domain')
    domains, rows = dict(graph['inputs']), []
    for node in graph['nodes']:
        attrs, op = node['attrs'], node['op']
        source = domains[node['inputs'][0]]
        domains[node['name']] = attrs.get('output', source)
        if op not in MACS:
            continue
        if attrs.get('accumulator') != 'fp64_e11m52_accumulator':
            raise ValueError('mapped MAC does not use FP64')
        x, xv = _encoding(source, name)
        out, ov = _encoding(attrs['output'], name)
        weight = graph['constants'][node['inputs'][1]]
        w, wv = _encoding(weight['encoding'], name, weight=True)
        k, channels = prod(weight['shape'][1:]), weight['shape'][0]
        if len(weight['codes']) != k*channels or not set(weight['codes']) <= set(wv):
            raise ValueError('mapped weight population mismatch')
        if w.axis == 0 and len(w.scales) != channels:
            raise ValueError('mapped weight channel-scale population mismatch')
        biases = attrs.get('bias', ['0']*channels)
        if len(biases) != channels:
            raise ValueError('mapped bias population mismatch')
        ordered = sorted(set(value*out.scales[0] for value in ov.values()))
        thresholds = [(a+b)/2 for a, b in zip(ordered, ordered[1:])]
        maximum_x = max(abs(value) for value in xv.values())*x.scales[0]
        xq, wq = quantum(xv.values())*x.scales[0], quantum(wv.values())
        passed, pending = 0, []
        for channel in range(channels):
            scale = w.scales[channel if w.axis == 0 else 0]
            q = xq*wq*scale
            residues = {threshold % q for threshold in thresholds}
            counts = Counter(weight['codes'][channel*k:(channel+1)*k])
            maximum_dot = maximum_x*scale*sum((abs(wv[code])*n for code, n in counts.items()), Fraction(0))
            b = Fraction(biases[channel])
            bound = error_bound(maximum_dot, k, absolute_bias=abs(b))
            if not bound['overflow_excluded']:
                raise ValueError('mapped FP64 MAC overflow is not excluded')
            margin = min(min((residue-b) % q, (b-residue) % q) for residue in residues)
            if margin > bound['absolute_error_bound']:
                passed += 1
            else:
                pending.append({'channel': channel, 'threshold_margin': str(margin),
                                'maximum_error_bound': str(bound['absolute_error_bound'])})
        rows.append({'node': node['name'], 'channels': channels, 'strict_threshold_margin_channels': passed,
                     'pending_channels': pending})
    if not rows:
        raise ValueError('graph contains no mapped MACs')
    return {'status': 'conditional_output_codes_proven' if all(not row['pending_channels'] for row in rows) else 'pending',
            'mac_nodes': len(rows), 'channels': sum(row['channels'] for row in rows),
            'strict_threshold_margin_channels': sum(row['strict_threshold_margin_channels'] for row in rows),
            'pending_channels': sum(len(row['pending_channels']) for row in rows), 'records': rows,
            'scope': 'identical finite scaled binary/ternary operands, every ordered FP64 MAC, original bias and output RNE store',
            'excludes': ['other graph operators', 'native C++/CUDA pilot', 'graph acceptance']}
