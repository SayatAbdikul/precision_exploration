"""Lane E1 reference models: hand examples for the findings of review 2 (revision r4 of the contract 2.3 draft).

Each test pins one sentence of docs/analysis/scaled-bridge-contract-2.3-draft-2026-10-02.md (section in the name).
CPU, exact, no image.
"""
from fractions import Fraction as Q
import math
import random
import pytest
from tools.scaled_bridge_ext import blocks as B
from tools.scaled_bridge_ext.certify import NodeCertificate
from tools.scaled_bridge_ext.policies import resolve
from tools.scaled_bridge_ext.reference import dot

ZERO_PAD = [0] * 31


# -- B11 (A2.3, A2.5): a tree adder rounds its output also when one operand is zero ------------------------------
def test_a25_tree_adder_with_a_zero_leaf_rounds():
    # flt.e3m2: P = 3, numbers in [8, 16) are 8, 10, 12, 14. Leaves 9, 0, 1, 0 (exact products; 9 is not representable).
    # adders: R(9 + 0) = 8 (tie to even, inexact), R(1 + 0) = 1, R(8 + 1) = R(9) = 8 (inexact, absorbed)
    v, e = dot([(9, 1), (0, 1), (1, 1), (0, 1)], 'flt.e3m2.ord-tree', 0, 0)
    assert v == 8.0 and e == {'nonfinite_steps': 0, 'overflow_steps': 0, 'inexact_steps': 2, 'absorbed_steps': 1}
    # skipping the zero leaves would carry 9 unrounded and give R(9 + 1) = 10; the same taps without the zeros do
    assert dot([(9, 1), (1, 1)], 'flt.e3m2.ord-tree', 0, 0)[0] == 10.0
    # a padded tap is a zero leaf, so padding decides values under .ord-tree (not under a sequential order)
    assert dot([(9, 1), (0, 1), (1, 1), (0, 1)], 'flt.e3m2', 0, 0)[0] == dot([(9, 1), (1, 1)], 'flt.e3m2', 0, 0)[0]


def test_a23_tree_flush_against_a_zero_leaf_is_absorbed():
    # E4M3 smallest normal 2^-6 = 64/4096: the adder 61/4096 + 0 flushes to +0 (tininess before rounding)
    v, e = dot([(61, 1), (0, 1)], 'flt.e4m3.ftz.ord-tree', 12, 0)
    assert v == 0.0 and e == {'nonfinite_steps': 0, 'overflow_steps': 0, 'inexact_steps': 1, 'absorbed_steps': 1}


# -- B12 (A2.6, A3.2, A3.3): bias and anchor on shared-exponent nodes ---------------------------------------------------
def _one_term_node():
    a = [(0, [1] + ZERO_PAD)]
    w = [(-3, [1] + ZERO_PAD)]                                       # block term 1/8 element product units
    cert = NodeCertificate.blocks([w], max_a_units=1, e_a_hi=2, length=1, bias=[Q(5, 16)])
    assert cert.anchor == -1
    return a, w, cert


def test_a26_wide_bias_is_exact_on_block_nodes():
    a, w, cert = _one_term_node()
    exact = 0.4375                                                   # 1/8 + 5/16
    for kwargs in ({}, {'anchor': 0}, {'anchor': -1}, {'anchor': 7}, {'certificate': cert}):
        for spec in ('wide.bias-post', 'wide.bias-pre', 'flt.e8m23.bias-post', 'ch1_wide_wide.bias-post'):
            assert B.block_dot(a, w, spec, 0, 0, bias_units=Q(5, 16), **kwargs)[0] == exact, (spec, kwargs)
    # rounded float stages and a rounded bias: the value does not depend on the anchor either
    a3, w3 = [(0, [3, 5, 7] + [0] * 29)], [(-3, [1, 3, 1] + [0] * 29)]          # exact dot 25/8, bias 11/32
    for spec, want in (('flt.e3m2.bias-post', 3.5), ('flt.e3m2.rz.bias-post', 3.0), ('flt.e2m3.ftz.bias-pre', 3.0),
                       ('ch2_flt.e3m2.elt_flt.e4m2.bias-post', 3.5)):
        assert {B.block_dot(a3, w3, spec, 0, 0, anchor=an, bias_units=Q(11, 32), length=3)[0]
                for an in (-9, -3, 0, 4, 20)} == {want}, spec


def test_a33_integer_register_needs_the_anchor():
    a, w, cert = _one_term_node()
    for spec in ('sat.w20', 'wrap.w20.bias-post', 'ch2_sat.w8_wide', 'ch2_wide_sat.w20'):
        with pytest.raises(ValueError):
            B.block_dot(a, w, spec, 0, 0, bias_units=Q(5, 16) if 'bias' in spec else None)
    # anchor -1: grid 1/2; term floor(1/4) = 0, bias RNE(5/8) = 1 -> 1 * 1/2
    assert B.block_dot(a, w, 'sat.w20.bias-post', 0, 0, certificate=cert, bias_units=Q(5, 16))[0] == 0.5
    assert B.block_dot(a, w, 'sat.w20.bias-post', 0, 0, anchor=-1, bias_units=Q(5, 16))[0] == 0.5
    with pytest.raises(ValueError):
        B.block_dot(a, w, 'sat.w20', 0, 0, anchor=0, certificate=cert)      # anchor differs from the certificate
    with pytest.raises(ValueError):
        B.block_dot(a, w, 'sat.abs-0', 0, 0, certificate={'signed_bits_absolute': 8})   # 2.1 dict: scalar only
    scalar = NodeCertificate.scalar([[1]], -1, 1)
    with pytest.raises(ValueError):
        B.block_dot(a, w, 'sat.abs-0', 0, 0, certificate=scalar)


def test_a32_alignment_includes_an_exact_wide_bias():
    a, w, _ = _one_term_node()
    r = B.alignment(a, w, bias_units=Q(5, 16))                       # 1 * 2^-3 and 5 * 2^-4
    assert r['e_min'] == -4 and r['aligned'] == [2, 5] and r['bound'] == 7 and r['max_prefix'] == 7
    assert B.alignment(a, w)['e_min'] == -3
    assert B.dyadic(Q(-12, 1)) == (-3, 2)
    with pytest.raises(ValueError):
        B.dyadic(Q(1, 3))


# -- must-fix: .nf on the outer stage of ch is refused --------------------------------------------------------
def test_a24_outer_stage_refuses_nf():
    for bad in ('ch2_wide_flt.e2m1.nf', 'ch2_flt.e2m1_flt.e2m1.nf', 'ch4_sat.w8_flt.e8m23.nf.ord-tree'):
        with pytest.raises(ValueError):
            resolve(bad)
    assert resolve('ch2_flt.e2m1.nf_flt.e2m1').inner.nonfused                   # the inner stage keeps it


# -- must-fix: .elt needs K; the padding of the last block is never a term -------------------------------------
def test_a33_elt_needs_the_length():
    a = [(0, [2, 2, 2] + [0] * 29)]
    w = [(0, [2, 2, 2] + [0] * 29)]
    with pytest.raises(ValueError):
        B.block_dot(a, w, 'ch2_flt.e2m1.elt_wide', 0, 0)
    with pytest.raises(ValueError):
        B.block_terms(a, w, 'element')
    for bad in (0, 33):
        with pytest.raises(ValueError):
            B.block_dot(a, w, 'ch2_flt.e2m1.elt_wide', 0, 0, length=bad)
    v, e = B.block_dot(a, w, 'ch2_flt.e2m1.elt_wide', 0, 0, length=3)
    # chunks [4, 4] and [4]; e2m1's largest finite value is 3: each chunk overflows on its first add
    assert math.isinf(v) and e['chunks'] == 2 and e['inner_overflow_steps'] == 2 and e['inner_nonfinite_steps'] == 3
    assert B.block_terms(a, w, 'element', 3) == [4, 4, 4]


# -- non-blocking: .elt with .ord-hwc fails at the parser; second names ----------------------------------------
def test_a33_elt_with_hwc_is_refused_by_the_parser():
    for bad in ('sat.w8.elt.ord-hwc', 'flt.e8m7.elt.ord-hwc', 'ch4_sat.w8.elt.ord-hwc_wide'):
        with pytest.raises(ValueError):
            resolve(bad)
    assert resolve('sat.w8.ord-hwc').order == 'hwc'                            # still a scalar-node name


def test_a0_second_names_are_the_same_arithmetic():
    rng = random.Random(41)
    book = B.element_book('mxfp6_e3m2')
    for _ in range(20):
        k = rng.randrange(1, 80)
        a = B.blocks([Q(rng.randrange(-99, 100), 16) for _ in range(k)], book, 'maxabs')
        w = B.blocks([Q(rng.randrange(-99, 100), 64) for _ in range(k)], book, 'maxabs')
        assert B.block_dot(a, w, 'wide.elt', book.shift, book.shift, length=k) == \
            B.block_dot(a, w, 'wide', book.shift, book.shift)
        pairs = [(rng.randrange(-40, 41), rng.randrange(-40, 41)) for _ in range(k)]
        assert dot(pairs, 'sat.w8.rne', 3, 4) == dot(pairs, 'sat.w8', 3, 4)
        assert dot(pairs, 'wrap.w9.rne', 3, 4) == dot(pairs, 'wrap.w9', 3, 4)
