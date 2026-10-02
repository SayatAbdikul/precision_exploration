"""Lane E1 reference models: accumulator policies of the contract 2.3 draft (CPU, exact)."""
from fractions import Fraction as Q
import math
import random
import pytest
from tools.scaled_bridge_ext import vectors
from tools.scaled_bridge_ext.orders import permutation, tree
from tools.scaled_bridge_ext.policies import resolve
from tools.scaled_bridge_ext.reference import dot, dot_terms, legacy_word, reduce_terms
from tools.scaled_bridge_ext.certify import register_widths
from tools.scaled_bridge_ext.rounding import FloatFormat, round_int
from tools.scaled_bridge_v2.oracles import expected_dot


# -- 1. stored vectors of the existing oracles ----------------------------------------------------------------
def test_stored_vectors_reproduce_existing_oracles():
    data = vectors.load()
    assert data['oracle_sources_sha256'] == vectors.oracle_hashes(), 'an existing oracle source changed'
    assert len(data['cases']) == 601
    for case in data['cases']:
        pairs = list(zip(case['a'], case['w']))
        value, events = dot(pairs, case['policy'], case['shift_a'], case['shift_w'])
        assert float(value).hex() == case['value_hex'], case['policy']
        assert legacy_word(case['policy'], events) == case['word'], case['policy']


def test_stored_vectors_still_match_live_oracle():
    for case in vectors.load()['cases'][::7]:
        value, word = expected_dot(list(zip(case['a'], case['w'])), case['shift_a'], case['shift_w'], case['policy'])
        assert float(value).hex() == case['value_hex'] and word == case['word']


def test_aliases_and_control_equal_generic_float():
    rng = random.Random(5)
    for _ in range(60):
        k = rng.randrange(1, 40)
        pairs = [(rng.randrange(-2**12, 2**12), rng.randrange(-2**10, 2**10)) for _ in range(k)]
        sa, sw = rng.randrange(0, 12), rng.randrange(0, 6)
        for old, new in (('fp16', 'flt.e5m10'), ('f21.x-7', 'flt.e8m12.x-7'), ('control', 'flt.e8m23')):
            assert dot(pairs, old, sa, sw) == dot(pairs, new, sa, sw)
    assert resolve('fp16').canonical == 'flt.e5m10' and resolve('f21.x3').canonical == 'flt.e8m12.x3'


# -- 2. names fail closed -------------------------------------------------------------------------------------
@pytest.mark.parametrize('bad', ['sat.w1', 'sat.w128', 'wrap.w08', 'sat.w16.rne.l2', 'sat.w16.l0', 'flt.e1m3',
                                 'flt.e5m10.x0', 'flt.e5m10.nf.ftz', 'wide.ord-tree', 'control.bias-pre',
                                 'sat.w16.ord-chw', 'sat.w16.ord-tree.bias-pre', 'ch0_sat.w16_wide',
                                 'ch8_sat.w16.bias-pre_wide', 'ch8_flt.e5m10_sat.w32', 'fp32', 'sat.w16.x3',
                                 'flt.e5m4.ofsat.rz', 'flt.e5m4.rz.ofsat', 'ch4_wide_wide.pr-e4m3', 'sat.w8.rz',
                                 'wide.pr-e4m3.pr-e5m2', 'control.pr-e4m3'])
def test_invalid_names_refused(bad):
    with pytest.raises(ValueError):
        resolve(bad)


def test_valid_names_round_trip():
    for name in ('wrap.w20', 'sat.w24.l4', 'sat.w24.l4.rne', 'wrap.struct-3.l2', 'flt.e5m4.x-9.ftz.ofsat.nf',
                 'flt.e8m7.ord-tree', 'sat.w20.ord-hwc.bias-post', 'ch64_sat.w16_wide', 'ch32_flt.e8m7.ord-tree_flt.e8m23.bias-pre',
                 'wide.bias-pre', 'sat.w127', 'flt.e4m7.ftz.rz.pr-e5m6', 'ch16_flt.e4m7.ftz.rz.pr-e5m6_wide',
                 'wide.pr-e4m3', 'ch4_wide_flt.e8m23', 'sat.w20.l2.pr-e4m3.ord-tree'):
        assert resolve(name).name() == name


# -- 3. hand-computed examples --------------------------------------------------------------------------------
def test_round_int_modes():
    assert [round_int(Q(n, 2), 'floor') for n in (-3, -1, 1, 3)] == [-2, -1, 0, 1]
    assert [round_int(Q(n, 2), 'rne') for n in (-3, -1, 1, 3, 5)] == [-2, 0, 0, 2, 2]


def test_wrap_hand():
    v, e = dot([(7, 1), (1, 1)], 'wrap.w4', 0, 0)
    assert v == -8.0 and e['high'] == 1 and e['low'] == 0
    v, e = dot([(7, 1), (1, 1), (-1, 1)], 'wrap.w4', 0, 0)          # wraps up, then back: final = exact 7
    assert v == 7.0 and e['high'] == 1 and e['low'] == 1
    v, e = dot([(7, 1), (1, 1), (-1, 1)], 'sat.w4', 0, 0)           # saturation does not come back
    assert v == 6.0 and e['high'] == 1


def test_low_bits_hand():
    v, e = dot([(3, 1), (3, 1), (-3, 1)], 'sat.w4.l1', 0, 0)       # floor: 1 + 1 - 2 = 0
    assert v == 0.0 and e['inexact_terms'] == 3
    v, _ = dot([(3, 1), (3, 1), (-3, 1)], 'sat.w4.l1.rne', 0, 0)   # rne: 2 + 2 - 2 = 2 -> 4 units
    assert v == 4.0
    v, _ = dot([(2, 1), (6, 1), (10, 1)], 'sat.w8.l2.rne', 0, 0)   # 0.5 -> 0, 1.5 -> 2, 2.5 -> 2: 4 * 4
    assert v == 16.0
    v, _ = dot([(2, 1), (6, 1), (10, 1)], 'sat.w8.l2', 0, 1)       # floor 0 + 1 + 2 = 3 -> 12 units, shift 1
    assert v == 6.0


def test_tiny_float_hand():
    f = FloatFormat.ieee(2, 1)                                     # values 0, 0.5, 1, 1.5, 2, 3
    assert (f.P, f.emin, f.emax, f.max_finite) == (2, -1, 1, 3)
    assert [f.round(Q(x)) for x in ('0.25', '0.75', '2.5', '3.4')] == [0, 1, 2, 3]
    assert f.round(Q('3.5')) == math.inf and f.round(Q('-3.5')) == -math.inf     # tie to even 4 -> overflow
    v, e = dot([(1, 1)] * 4, 'flt.e2m1', 0, 0)
    assert v == math.inf and e['nonfinite_steps'] == 1
    v, e = dot([(1, 1)] * 5, 'flt.e2m1.ofsat', 0, 0)
    assert v == 3.0 and e['nonfinite_steps'] == 0 and e['absorbed_steps'] == 2
    assert dot([(1, 1)], 'flt.e2m1.ftz', 1, 0)[0] == 0.0 and dot([(1, 1)], 'flt.e2m1', 1, 0)[0] == 0.5
    assert dot([(6, 1), (5, 1)], 'flt.e2m1', 2, 0)[0] == 3.0       # fused: RNE(1.5 + 1.25) = 3
    assert dot([(6, 1), (5, 1)], 'flt.e2m1.nf', 2, 0)[0] == 2.0    # not fused: RNE(1.5 + RNE(1.25)=1) = 2 (tie)
    v, e = dot([(3, 1), (-3, 1), (2, 1), (-2, 1)], 'flt.e2m1', 0, 0)
    assert v == 0.0 and math.copysign(1, v) == 1


def test_infinities_meet_nan():
    v, e = dot_terms([Q(3)] * 2 + [Q(-3)] * 4, 'ch2_flt.e2m1_flt.e2m1', 0)
    assert math.isnan(v)


def test_order_hand():
    assert permutation('hwc', (2, 1, 2)) == [0, 2, 1, 3] and permutation('rev', 3) == [2, 1, 0]
    assert tree([1, 2, 3, 4, 5], lambda a, b: f'({a}+{b})') == '(((1+2)+(3+4))+5)'
    terms = [(4, 1)] + [(1, 1)] * 4                                # code level 4 then four 0.5 (shift 1)
    assert dot([(8, 1)] + [(1, 1)] * 4, 'flt.e5m2', 1, 0)[0] == 4.0            # swamped
    assert dot([(8, 1)] + [(1, 1)] * 4, 'flt.e5m2.ord-tree', 1, 0)[0] == 6.0   # 4, 1, .5 -> 5.5 -> 6
    assert dot(terms, 'wide', 0, 0)[0] == 8.0


def test_chunk_and_bias_hand():
    pairs = [(7, 1), (7, 1), (-7, 1), (-7, 1)]
    assert dot(pairs, 'sat.w4', 0, 0)[0] == -7.0
    v, e = dot(pairs, 'ch2_sat.w4_wide', 0, 0)                      # chunks 7 (clamped 14) and -8 (clamped -14)
    assert v == -1.0 and e['inner_high'] == 1 and e['inner_low'] == 1 and e['chunks'] == 2
    assert dot([(3, 1), (-3, 1)], 'sat.w4.bias-pre', 0, 0, bias_units=Q(6))[0] == 4.0    # 6+3 -> 7, 7-3 = 4
    assert dot([(3, 1), (-3, 1)], 'sat.w4.bias-post', 0, 0, bias_units=Q(6))[0] == 6.0
    assert dot([], 'wide.bias-pre', 0, 0, bias_units=Q(5, 2))[0] == 2.0                  # bias RNE onto the grid
    assert dot([], 'wide.bias-post', 0, 0, bias_units=Q(7, 2))[0] == 4.0
    assert dot([(1, 1)], 'sat.w8.l2.bias-pre', 0, 0, bias_units=Q(6))[0] == 8.0          # 6/4 -> 2 (rne), 1/4 -> 0


# -- 4. properties --------------------------------------------------------------------------------------------
def same(x, y):
    return (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)) or x == y


def _random_pairs(rng, k, span):
    return [(rng.randrange(-(1 << span), 1 << span), rng.randrange(-(1 << span), 1 << span)) for _ in range(k)]


def test_registers_equal_wide_without_events():
    rng = random.Random(11)
    for _ in range(300):
        pairs = _random_pairs(rng, rng.randrange(1, 50), rng.randrange(1, 12))
        exact = dot(pairs, 'wide', 2, 1)[0]
        width = rng.randrange(2, 30)
        for kind in ('sat', 'wrap'):
            v, e = dot(pairs, f'{kind}.w{width}', 2, 1)
            if e['high'] == e['low'] == 0:
                assert v == exact
        need = max(abs(sum(a * b for a, b in pairs[:i])) for i in range(len(pairs) + 1))
        w = need.bit_length() + 1
        assert dot(pairs, f'sat.w{max(w, 2)}', 2, 1)[0] == exact == dot(pairs, f'wrap.w{max(w, 2)}.ord-tree', 2, 1)[0] or w > 127


def test_wrap_is_exact_when_the_final_sum_fits():
    rng = random.Random(12)
    for _ in range(200):
        pairs = _random_pairs(rng, rng.randrange(2, 60), 8)
        total = sum(a * b for a, b in pairs)
        width = max(total.bit_length() + 1, 2)
        v, e = dot(pairs, f'wrap.w{width}', 0, 0)
        assert v == float(total)                       # whatever happened in between


def test_enough_mantissa_equals_wide():
    rng = random.Random(13)
    for _ in range(150):
        pairs = _random_pairs(rng, rng.randrange(1, 50), 20)
        exact = dot(pairs, 'wide', 3, 4)[0]
        for spec in ('flt.e11m52', 'flt.e11m52.ord-tree', 'flt.e11m52.nf', 'ch7_flt.e11m52_flt.e11m52'):
            v, e = dot(pairs, spec, 3, 4)
            assert v == exact, spec


def test_one_chunk_equals_plain_and_identity_orders():
    rng = random.Random(14)
    for _ in range(150):
        k = rng.randrange(1, 40)
        pairs = _random_pairs(rng, k, rng.randrange(2, 14))
        for inner in ('sat.w12', 'wrap.w10.l2', 'flt.e5m6.x-3', 'flt.e4m3.ord-tree', 'sat.w14.l3.rne.ord-rev'):
            plain = dot(pairs, inner, 1, 2)
            assert same(dot(pairs, f'ch{k + rng.randrange(0, 5)}_{inner}_wide', 1, 2)[0], plain[0])
            if not inner.startswith('flt'):
                continue
            outer = inner.split('.ord-')[0]
            assert same(dot(pairs, f'ch{k}_{inner}_{outer}', 1, 2)[0], plain[0])
        terms = [Q(a * b) for a, b in pairs]
        for spec in ('sat.w12', 'flt.e5m4', 'wrap.w9.l1.rne'):
            p = resolve(spec)
            width = register_widths(p)[0]
            a, b = reduce_terms([terms[i] for i in permutation('chw', k)], p, 3, width), reduce_terms(terms, p, 3, width)
            assert same(a[0], b[0]) and a[1] == b[1]
            for taps in ((1, k, 1), (k, 1, 1)):                    # cg = 1, or a 1x1 kernel: hwc is the identity
                a, b = dot(pairs, spec + '.ord-hwc', 1, 2, taps=taps), dot(pairs, spec, 1, 2)
                assert same(a[0], b[0]) and a[1] == b[1]


def test_float_reference_matches_public_manifests():
    from public.inference.reference.arithmetic import format_named
    rng = random.Random(15)
    for name, (E, M) in (('fp16_e5m10_accumulator', (5, 10)), ('fp32_e8m23_accumulator', (8, 23))):
        public, mine = format_named(name), FloatFormat.ieee(E, M)
        for _ in range(3000):
            x = Q(rng.randrange(-2**40, 2**40), 1 << rng.randrange(0, 200)) * rng.choice([1, 2**rng.randrange(0, 140)])
            a = public.decode(public.encode(x))
            b = mine.round(x)
            if isinstance(b, float):
                assert not a.is_finite() and (a.is_signed() == (b < 0))
            else:
                assert (Q(0) if not isinstance(a, Q) else a) == b


def test_round_toward_zero_and_product_rounding_hand():
    f = FloatFormat.ieee(2, 1, mode='rz')                          # 0, 0.5, 1, 1.5, 2, 3
    assert [f.round(Q(x)) for x in ('0.75', '-0.75', '2.9', '3.5', '-100')] == [Q(1, 2), Q(-1, 2), 2, 3, -3]
    assert dot([(1, 1)] * 5, 'flt.e2m1.rz', 0, 0)[0] == 3.0       # never infinite under rz
    assert dot([(9, 1)], 'wide.pr-e2m1', 0, 0)[0] == 3.0          # 9 rounds to the largest finite 3 (saturating)
    assert dot([(5, 1), (5, 1)], 'wide.pr-e2m1', 1, 0)[0] == 4.0  # 2.5 -> 2 (tie to even) twice
    assert dot([(5, 1), (5, 1)], 'wide', 1, 0)[0] == 5.0
    assert dot([(1, 1)] * 4, 'ch4_wide_flt.e2m1', 0, 0)[0] == math.inf   # exact 4, one rounding: overflow
    assert dot([(3, 1), (1, 2)], 'ch2_wide_flt.e2m1', 1, 0)[0] == 2.0    # 1.5 + 1 = 2.5 -> 2, one rounding
    assert dot([(3, 1), (1, 2)], 'flt.e2m1', 1, 0)[0] == 2.0
