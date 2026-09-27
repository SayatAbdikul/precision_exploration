"""Model C boundary checks for saturating dyadic FP MAC certificates."""
import json
from fractions import Fraction
from itertools import product

import pytest

from public.inference.reference.arithmetic import encode, format_named, model_c
from development.acceptance_proofs.dyadic_fp_mac_store import prove_graph


def small_graph(name, bias='0.003'):
    domain = {'format': name, 'scales': ['1'], 'axis': None, 'block_size': None}
    return {'inputs': {'x': domain},
            'constants': {'w': {'shape': [1, 2], 'encoding': domain, 'codes': [1, 2]}},
            'nodes': [{'name': 'fc', 'op': 'linear', 'inputs': ['x', 'w'],
                       'attrs': {'accumulator': 'fp64_e11m52_accumulator',
                                 'output': domain, 'bias': [bias]}}]}


def test_certificate_matches_independent_short_model_c():
    graph = small_graph('fp4_e2m1')
    proof = prove_graph(graph, 'fp4_e2m1')
    assert proof['status'] == 'conditional_output_codes_proven'
    fmt, accumulator = format_named('fp4_e2m1'), format_named('fp64_e11m52_accumulator')
    weights = [Fraction(fmt.decode(code)) for code in (1, 2)]
    finite = [code for code in range(16) if fmt.decode(code).is_finite()]
    for codes in product(finite, repeat=2):
        values = [Fraction(fmt.decode(code)) for code in codes]
        exact = sum((a*b for a, b in zip(values, weights)), Fraction('0.003'))
        state = model_c(zip(values, weights), accumulator, bias='0.003')
        assert encode(fmt, exact) == encode(fmt, accumulator.decode(state))


def test_nonfinite_weight_fails_closed():
    graph = small_graph('fp8_e4m3fn')
    fmt = format_named('fp8_e4m3fn')
    graph['constants']['w']['codes'][0] = next(code for code in range(256) if not fmt.decode(code).is_finite())
    with pytest.raises(ValueError, match='nonfinite'):
        prove_graph(graph, 'fp8_e4m3fn')


def test_retained_fp8_boundary_discrepancy_stays_pending():
    inventory = json.load(open('results/summaries/phase3-gate-inventory.json'))
    row = next(row for row in inventory['records'] if row['configuration'] == 'resnet18/fp8_e4m3fn')
    graph = json.load(open(row['graph']['path']))
    proof = prove_graph(graph, 'fp8_e4m3fn')
    assert proof['status'] == 'pending'
    assert proof['pending_channels'] > 0
    assert any(item['reason'] == 'output code differs at a sensitive lattice site'
               for row in proof['records'] for item in row['pending_channels'])
