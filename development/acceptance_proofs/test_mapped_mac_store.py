"""Oracle comparisons for mapped binary/ternary MAC threshold bounds."""
import json
from fractions import Fraction
from itertools import product

from public.inference.reference.arithmetic import encode, format_named, model_c
from development.acceptance_proofs.mapped_mac_store import prove_graph


def small_graph(name):
    x = {'format': name, 'scales': ['0.75'], 'axis': None, 'block_size': None}
    w = {'format': name, 'scales': ['0.375'], 'axis': 0, 'block_size': None}
    out = {'format': name, 'scales': ['0.5'], 'axis': None, 'block_size': None}
    return {'inputs': {'x': x},
            'constants': {'w': {'shape': [1, 2], 'encoding': w, 'codes': [0, 1]}},
            'nodes': [{'name': 'fc', 'op': 'linear', 'inputs': ['x', 'w'],
                       'attrs': {'accumulator': 'fp64_e11m52_accumulator', 'output': out,
                                 'bias': ['0.003']}}]}


def test_small_mapped_domains_agree_with_model_c():
    for name in ('binary_pm1', 'ternary'):
        graph = small_graph(name)
        proof = prove_graph(graph, name)
        assert proof['status'] == 'conditional_output_codes_proven'
        fmt, accumulator = format_named(name), format_named('fp64_e11m52_accumulator')
        x_codes = [code for code in range(1 << fmt.bits) if fmt.decode(code).is_finite()]
        weights = [Fraction(fmt.decode(code))*Fraction('0.375') for code in (0, 1)]
        for codes in product(x_codes, repeat=2):
            values = [Fraction(fmt.decode(code))*Fraction('0.75') for code in codes]
            exact = sum((a*b for a, b in zip(values, weights)), Fraction('0.003'))
            state = model_c(zip(values, weights), accumulator, bias='0.003')
            assert encode(fmt, exact, scale=Fraction('0.5')) == encode(fmt, accumulator.decode(state), scale=Fraction('0.5'))


def test_retained_mobilenet_v2_sensitive_channels_remain_pending():
    inventory = json.load(open('results/summaries/phase3-gate-inventory.json'))
    for name in ('binary_pm1', 'ternary'):
        row = next(row for row in inventory['records'] if row['configuration'] == 'mobilenet_v2/'+name)
        graph = json.load(open(row['graph']['path']))
        proof = prove_graph(graph, name)
        assert proof['status'] == 'pending'
        assert proof['pending_channels'] > 0
