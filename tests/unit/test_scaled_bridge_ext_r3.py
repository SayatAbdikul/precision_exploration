"""Lane E1 reference models: hand examples for the statements added to the contract 2.3 draft in revision r3.

Each test pins one sentence of docs/analysis/scaled-bridge-contract-2.3-draft-2026-10-02.md (section in the name).
CPU, exact, no image.
"""
from fractions import Fraction as Q
import math
import pytest
from tools.scaled_bridge_ext import blocks as B
from tools.scaled_bridge_ext.policies import resolve
from tools.scaled_bridge_ext.reference import dot

FLOAT_ZERO = {'nonfinite_steps': 0, 'overflow_steps': 0, 'inexact_steps': 0, 'absorbed_steps': 0}


def test_a23_scale_exponent_sign_is_that_of_21():
    # the accumulator holds the code-level value times 2^e: fp16.x-12 reaches 65,504 * 2^12 code level
    assert 65504 * 2 ** 12 == 268304384
    assert dot([(2 ** 14, 2 ** 13)], 'flt.e5m10.x-12', 0, 0) == (2.0 ** 27, FLOAT_ZERO)
    assert math.isinf(dot([(2 ** 14, 2 ** 13)], 'flt.e5m10', 0, 0)[0])
    assert math.isinf(dot([(2 ** 14, 2 ** 14)], 'flt.e5m10.x-12', 0, 0)[0])       # 2^28 > 65,504 * 2^12


def test_a23_product_scale_has_the_same_sign():
    # p becomes R'(p * 2^e') * 2^-e': E4M3 saturates at 240 * 2^-e' code level
    assert dot([(200, 100)], 'wide.pr-e4m3-x-3', 0, 0)[0] == 240.0 * 8
    assert dot([(200, 100)], 'wide.pr-e4m3-x3', 0, 0)[0] == 240.0 / 8
    assert dot([(200, 100)], 'flt.e5m10.x-12.pr-e4m3', 0, 0)[0] == 240.0           # never in accumulator units


def test_a24_blu24_name_saturates():
    name = 'ch16_flt.e4m7.x3.ftz.rz.pr-e4m7-x5-rz-ftz_flt.e4m7.x3.ftz.rz'
    assert resolve(name).name() == name
    v, e = dot([(3, 5)] * 40, name, 2, 2)                                         # 40 * 15/16 = 37.5 > 255 / 8
    assert v == 255 / 8 and e['outer_overflow_steps'] == 1 and e['chunks'] == 3
    rne = 'ch16_flt.e4m7.x3.ftz.ofsat.pr-e4m7-x5-ftz_flt.e4m7.x3.ftz.ofsat'
    assert resolve(rne).name() == rne


def test_a23_product_rounded_to_zero_is_a_skipped_term():
    # state 1.0, product 0.125: fused, the add absorbs it; .nf rounds the product to 0 first (E2M1 spacing 0.5)
    v, e = dot([(8, 1), (1, 1)], 'flt.e2m1', 3, 0)
    assert v == 1.0 and e['inexact_steps'] == 1 and e['absorbed_steps'] == 1
    assert dot([(8, 1), (1, 1)], 'flt.e2m1.nf', 3, 0) == (1.0, FLOAT_ZERO)


def test_a23_ftz_flush_against_zero_state_is_absorbed():
    v, e = dot([(61, 1)], 'flt.e4m3.ftz', 12, 0)                                  # 61/4096 < 1/64: +0
    assert v == 0.0 and e == {'nonfinite_steps': 0, 'overflow_steps': 0, 'inexact_steps': 1, 'absorbed_steps': 1}
    assert dot([(61, 1)], 'flt.e4m3', 12, 0)[0] == 1 / 64                         # gradual: rounds up to 1/64


def test_a31_element_ties_go_to_the_even_code():
    book = B.element_book('mxfp4_e2m1')                                           # levels 0 .5 1 1.5 2 3 4 6
    ties = {Q(1, 4): 0, Q(3, 4): 1, Q(5, 4): 1, Q(7, 4): 2, Q(5, 2): 2, Q(7, 2): 4, Q(5): 4}
    for y, level in ties.items():
        assert Q(book.nearest(y), 1 << book.shift) == level, y
        assert Q(book.nearest(-y), 1 << book.shift) == -level, -y


def test_a33_elt_is_a_name_modifier_of_single_or_inner_stages():
    for name in ('sat.abs-0.elt', 'ch32_flt.e8m10.elt_flt.e8m23', 'sat.w20.l-4.rne', 'sat.w20.rne'):
        assert resolve(name).name() == name
    for bad in ('ch2_wide_flt.e8m23.elt', 'flt.e8m23.ord-tree.elt', 'sat.w20.l0'):
        with pytest.raises(ValueError):
            resolve(bad)
