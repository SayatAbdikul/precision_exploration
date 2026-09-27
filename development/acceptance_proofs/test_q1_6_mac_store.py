"""Boundary tests for the Q1.6 all-code MAC/store certificate."""
from fractions import Fraction
from itertools import product

from public.inference.reference.arithmetic import encode, format_named, model_c
from development.acceptance_proofs.q1_6_mac_store import prove_graph


def small_graph(bias):
    domain = {'format': 'q1_6', 'scales': ['1'], 'axis': None, 'block_size': None}
    return {'inputs': {'x': domain},
            'constants': {'w': {'shape': [1, 2], 'encoding': domain, 'codes': [1, 64]}},
            'nodes': [{'name': 'fc', 'op': 'linear', 'inputs': ['x', 'w'],
                       'attrs': {'accumulator': 'fp64_e11m52_accumulator', 'output': domain,
                                 'bias': [bias]}}]}


def test_exact_tie_has_exact_biased_lattice():
    proof = prove_graph(small_graph('0'))
    assert proof['status'] == 'conditional_output_codes_proven'
    assert proof['methods'] == {'exact_biased_lattice_including_ties': 1}


def test_near_threshold_fails_closed():
    proof = prove_graph(small_graph('0.0078125000000000001'))
    assert proof['status'] == 'pending'
    assert proof['pending_channels'] == 1


def test_margin_certificate_matches_exhaustive_short_model_c():
    bias = '0.003'
    proof = prove_graph(small_graph(bias))
    assert proof['methods'] == {'strict_threshold_margin': 1}
    fmt, acc = format_named('q1_6'), format_named('fp64_e11m52_accumulator')
    weight = [Fraction(fmt.decode(code)) for code in (1, 64)]
    for codes in product(range(256), (0, 1, 64, 127, 128, 255)):
        values = [Fraction(fmt.decode(code)) for code in codes]
        exact = sum((a*b for a, b in zip(values, weight)), Fraction(bias))
        state = model_c(zip(values, weight), acc, bias=bias)
        assert encode(fmt, exact) == encode(fmt, acc.decode(state))
