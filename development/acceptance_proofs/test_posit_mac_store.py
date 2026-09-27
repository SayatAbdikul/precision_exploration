"""Boundary checks for conditional posit/quire MAC output certificates."""
import json
from fractions import Fraction
from itertools import product

import pytest

from public.inference.reference.arithmetic import encode, format_named, model_c
from development.acceptance_proofs.posit_mac_store import prove_graph


def small_graph(name, bias='0.003'):
    domain = {'format': name, 'scales': ['1'], 'axis': None, 'block_size': None}
    return {'inputs': {'x': domain},
            'constants': {'w': {'shape': [1, 2], 'encoding': domain, 'codes': [1, 2]}},
            'nodes': [{'name': 'fc', 'op': 'linear', 'inputs': ['x', 'w'],
                       'attrs': {'accumulator': 'posit8_es1_quire64_accumulator',
                                 'output': domain, 'bias': [bias]}}]}


def test_small_posit4_certificate_agrees_with_model_c():
    graph = small_graph('posit4_es0')
    proof = prove_graph(graph, 'posit4_es0')
    assert proof['status'] == 'conditional_output_codes_proven'
    fmt, accumulator = format_named('posit4_es0'), format_named('posit8_es1_quire64_accumulator')
    weights = [Fraction(fmt.decode(code)) for code in (1, 2)]
    finite = [code for code in range(16) if fmt.decode(code).is_finite()]
    for codes in product(finite, repeat=2):
        values = [Fraction(fmt.decode(code)) for code in codes]
        exact = sum((a*b for a, b in zip(values, weights)), Fraction('0.003'))
        state = model_c(zip(values, weights), accumulator, bias='0.003')
        assert encode(fmt, exact) == encode(fmt, accumulator.decode(state))


def test_nar_weight_fails_closed():
    graph = small_graph('posit4_es0')
    fmt = format_named('posit4_es0')
    graph['constants']['w']['codes'][0] = next(code for code in range(16) if not fmt.decode(code).is_finite())
    with pytest.raises(ValueError, match='incomplete'):
        prove_graph(graph, 'posit4_es0')


def test_retained_posit8_bias_witness_stays_pending():
    inventory = json.load(open('results/summaries/phase3-gate-inventory.json'))
    row = next(row for row in inventory['records'] if row['configuration'] == 'resnet18/posit8_es1')
    graph = json.load(open(row['graph']['path']))
    proof = prove_graph(graph, 'posit8_es1')
    assert proof['status'] == 'pending'
    assert proof['pending_channels'] > 0
    assert any(item['reason'] == 'stored-bias boundary changes output code'
               for row in proof['records'] for item in row['pending_channels'])
