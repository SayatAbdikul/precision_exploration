"""Lane E1 reference models: the findings of review 1 (blocking B1-B10, must-fix, test gaps). CPU, exact."""
from fractions import Fraction as Q
import math
import random
import pytest
from tools.scaled_bridge_ext import blocks as B
from tools.scaled_bridge_ext import vectors
from tools.scaled_bridge_ext.certify import NodeCertificate, certified_widths, register_widths
from tools.scaled_bridge_ext.orders import permutation
from tools.scaled_bridge_ext.policies import resolve
from tools.scaled_bridge_ext.reference import dot, dot_terms, legacy_word
from tools.scaled_bridge_ext.rounding import FloatFormat, round_int
from tools.scaled_bridge_v2.certificates import signed_bits as signed_bits_21


def same(x, y):
    return (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)) or x == y


# -- B1: lossless float condition; .ftz needs the smallest normal at or below one grid unit ------------------
def test_ftz_counterexample_of_review():
    # five products of 1 unit at shift 13; E4M7 smallest normal 2^-6 code level = 2^7 units > 1 unit
    v, e = dot([(1, 1)] * 5, 'flt.e4m7.ftz', 13, 0)
    assert v == 0.0 and e['absorbed_steps'] == 5
    assert dot([(1, 1)] * 5, 'wide', 13, 0)[0] == 5 * 2.0 ** -13 == dot([(1, 1)] * 5, 'flt.e4m7', 13, 0)[0]
    assert dot([(1, 1)] * 5, 'flt.e4m7.ftz', 6, 0)[0] == 5 * 2.0 ** -6     # smallest normal = 1 unit: exact


def lossless(fmt_units, bound, ftz):
    """The corrected A2.3 condition on a format already in product-grid units."""
    return (bound < (1 << fmt_units.P) and fmt_units.emin <= 0 and fmt_units.max_finite >= bound
            and (not ftz or fmt_units.min_normal <= 1))


def test_lossless_condition_holds_for_every_variant():
    rng = random.Random(31)
    checked = 0
    while checked < 300:
        k = rng.randrange(1, 30)
        pairs = [(rng.randrange(-200, 201), rng.randrange(-60, 61)) for _ in range(k)]
        bound = sum(abs(a * w) for a, w in pairs)
        E, M, x = rng.randrange(3, 12), rng.randrange(1, 30), rng.randrange(-8, 9)
        sa, sw = rng.randrange(0, 8), rng.randrange(0, 8)
        mods = rng.choice(['', '.ftz', '.rz', '.nf', '.ftz.nf.rz', '.ofsat', '.ftz.ofsat.nf'])
        name = f'flt.e{E}m{M}' + (f'.x{x}' if x else '') + mods
        fmt = FloatFormat.ieee(E, M).scaled(x - sa - sw)
        if not lossless(fmt, bound, '.ftz' in mods):
            continue
        exact = dot(pairs, 'wide', sa, sw)[0]
        outer = name.replace('.nf', '')                                  # r4: .nf is refused on an outer stage
        for spec in (name, name + '.ord-tree', name + '.ord-rev', f'ch3_{name}_{outer}'):
            v, e = dot(pairs, spec, sa, sw)
            assert v == exact, spec
            assert sum(c for key, c in e.items() if 'inexact' in key or 'overflow' in key or 'nonfinite' in key) == 0
        checked += 1


# -- B2/B3: one definition of certified widths, applied to every relative name -------------------------------
def test_floor_entry_widens_the_certificate():
    # nine terms of -1/4 grid unit (shared-exponent terms below the anchor): floor enters each as -1
    cert = NodeCertificate({'abs': [[(-Q(1, 4), Q(1, 4))] * 9]}, exact=False)
    assert certified_widths('sat.abs-0', cert) == (5,)                     # [-11, 2]: entered prefix down to -9
    v, e = dot_terms([-Q(1, 4)] * 9, 'sat.abs-0', 0, cert)
    assert v == -9 and e['high'] == e['low'] == 0 and e['inexact_terms'] == 9
    assert certified_widths('sat.abs-0.rne', cert) == (4,)                 # [-7, 6]
    assert dot_terms([-Q(1, 4)] * 9, 'sat.abs-0.rne', 0, cert)[0] == 0


def test_plain_widths_equal_the_21_certificate():
    rng = random.Random(32)
    for _ in range(100):
        cout, k = rng.randrange(1, 5), rng.randrange(1, 40)
        weights = [[rng.randrange(-100, 101) for _ in range(k)] for _ in range(cout)]
        a_lo, a_hi = -rng.randrange(0, 300), rng.randrange(1, 300)
        s_lo = rng.choice([a_lo, 0])
        cert = NodeCertificate.scalar(weights, a_lo, a_hi, s_lo)
        max_a = max(-a_lo, a_hi)
        b = max(max_a * sum(abs(w) for w in row) for row in weights)
        struct = max(signed_bits_21(sum(w * (s_lo if w > 0 else a_hi) for w in row),
                                    sum(w * (a_hi if w > 0 else s_lo) for w in row)) for row in weights)
        assert certified_widths('sat.abs-0', cert) == (b.bit_length() + 1,)
        assert certified_widths('wrap.struct-2', cert) == (struct,)
        legacy = {'signed_bits_absolute': b.bit_length() + 1, 'signed_bits_structural': struct}
        assert register_widths('sat.abs-3', legacy) == register_widths('sat.abs-3', cert)
        with pytest.raises(ValueError):
            register_widths('sat.abs-0.l2', legacy)            # the 2.1 dict defines plain registers only
        with pytest.raises(ValueError):
            register_widths('ch4_sat.abs-0_wide', legacy)


RELATIVE = ('sat.abs-0', 'wrap.abs-0', 'sat.struct-0', 'sat.abs-0.l3', 'sat.abs-0.l3.rne', 'wrap.struct-0.l5',
            'sat.abs-0.l-2', 'sat.abs-0.ord-tree', 'sat.abs-0.l2.ord-rev', 'sat.abs-0.bias-pre',
            'wrap.abs-0.l1.bias-post', 'sat.struct-0.ord-hwc.bias-post', 'ch4_sat.abs-0_sat.abs-0',
            'ch3_sat.abs-0.l2_sat.abs-0.l1.bias-post', 'ch5_sat.abs-0.l1_sat.abs-0.l3', 'ch2_wide_sat.abs-0.l2',
            'ch4_sat.w4_sat.abs-0', 'ch3_sat.abs-0.ord-hwc_wrap.abs-0.l1.ord-tree.bias-post',
            'ch4_sat.abs-0.l2.rne.ord-tree_sat.abs-0.bias-pre')


def test_relative_zero_never_has_an_event_and_matches_its_grid():
    rng = random.Random(33)
    for _ in range(200):
        cg, kh, kw = rng.randrange(1, 4), rng.randrange(1, 4), rng.randrange(1, 4)
        k = cg * kh * kw
        cout = rng.randrange(1, 3)
        weights = [[rng.randrange(-40, 41) for _ in range(k)] for _ in range(cout)]
        a_lo, a_hi = -rng.randrange(0, 60), rng.randrange(1, 60)
        s_lo = rng.choice([a_lo, 0])
        bias = [Q(rng.randrange(-5000, 5000), rng.randrange(1, 9)) for _ in range(cout)]
        cert = NodeCertificate.scalar(weights, a_lo, a_hi, s_lo, bias, (cg, kh, kw))
        c = rng.randrange(cout)
        acts = [rng.randrange(s_lo, a_hi + 1) for _ in range(k)]
        pairs = list(zip(acts, weights[c]))
        for spec in RELATIVE:
            if 'struct' in spec and rng.random() < .5:
                continue
            p = resolve(spec)
            b = bias[c] if p.stages[-1].bias else None
            v, e = dot(pairs, spec, 2, 1, cert, bias_units=b)
            keys = ['high', 'low'] + [f'{role}_{side}' for role, stage in zip(('inner', 'outer'), p.stages)
                                      if p.kind == 'chunk' and stage.kind == 'int' and stage.reference != 'global'
                                      for side in ('high', 'low')]
            assert all(e.get(key, 0) == 0 for key in keys), spec
            if p.kind == 'int' and not p.bias and p.low_bits >= 0:      # the value is the sum of the entered terms
                g = 1 << p.low_bits
                assert v == sum(round_int(Q(a * w, g), p.low_mode) for a, w in pairs) * g / 8, spec


def test_relative_width_is_narrower_with_dropped_bits():
    cert = NodeCertificate.scalar([[100] * 64], -128, 127)
    full = certified_widths('sat.abs-0', cert)[0]                  # 128 * 6400 = 819200: 21 bits
    # floor entry on 2^4: [floor(-51200) - 64 + 1, 51200] = [-51263, 51200]: 17 bits
    assert full == 21 and certified_widths('sat.abs-0.l4', cert) == (17,)
    assert register_widths('sat.abs-2.l4', cert) == (15,)


def test_block_certificate_has_no_event_at_abs_0():
    rng = random.Random(34)
    for name in ('mxfp4_e2m1', 'mxfp8_e4m3', 'bfp6'):
        book = B.element_book(name)
        top = max(abs(u) for u in book.units)
        for _ in range(25):
            k = rng.randrange(1, 100)
            nb = (k + 31) // 32
            w_blocks = [[(rng.randrange(-6, 3), [rng.choice(book.units) for _ in range(32)]) for _ in range(nb)]
                        for _ in range(2)]
            for row in w_blocks:                                     # zero padding of the last block
                e, u = row[-1]
                row[-1] = (e, u[:k - 32 * (nb - 1)] + [0] * (32 * nb - k))
            e_hi = rng.randrange(-3, 4)
            bias = [Q(rng.randrange(-999, 999), 7), Q(rng.randrange(-999, 999), 3)]
            cert = NodeCertificate.blocks(w_blocks, top, e_hi, k, bias)
            a_blocks = [(e_hi - rng.randrange(0, 12), [rng.choice(book.units) for _ in range(32)]) for _ in range(nb)]
            for c in range(2):
                for spec in ('sat.abs-0', 'wrap.abs-0', 'sat.abs-0.l2', 'sat.abs-0.rne', 'sat.abs-0.l-3',
                             'sat.abs-0.elt', 'sat.abs-0.l1.rne.elt', 'ch2_sat.abs-0_sat.abs-0',
                             'ch8_sat.abs-0.elt_sat.abs-0.l1', 'sat.abs-0.bias-post', 'sat.abs-0.ord-tree'):
                    b = bias[c] if 'bias' in spec else None
                    _, e = B.block_dot(a_blocks, w_blocks[c], spec, book.shift, book.shift, certificate=cert,
                                       bias_units=b, length=k)
                    assert all(e.get(key, 0) == 0 for key in ('high', 'low', 'inner_high', 'inner_low',
                                                               'outer_high', 'outer_low')), (name, spec)
            with pytest.raises(ValueError):
                certified_widths('sat.struct-0', cert)


# -- B4: the outer stage takes no order other than tree ------------------------------------------------------
@pytest.mark.parametrize('bad', ['ch1_wide_flt.e5m2.ord-rev', 'ch1_wide_flt.e5m2.ord-hwc', 'ch4_sat.w8_sat.w9.ord-rev',
                                 'ch4_wide_sat.w16.elt', 'ch4_wide_wide.pr-e4m3'])
def test_outer_stage_modifiers_refused(bad):
    with pytest.raises(ValueError):
        resolve(bad)


def test_outer_tree_is_a_tree_over_chunks():
    terms = [Q(8), Q(8), Q(5), Q(16), Q(3), Q(16), Q(5)]
    assert dot_terms(terms, 'ch1_wide_flt.e5m2.ord-tree', 0)[0] == dot_terms(terms, 'flt.e5m2.ord-tree', 0)[0] == 56


# -- B5: one-term reductions under .ord-tree -----------------------------------------------------------------
def test_lone_root_is_rounded_like_one_sequential_step():
    v, e = dot([(4, 1)], 'flt.e2m1.ord-tree', 0, 0)
    assert v == math.inf and e['nonfinite_steps'] == 1 and e['overflow_steps'] == 1
    assert dot([(4, 1)], 'flt.e2m1', 0, 0) == (v, e)
    # one tap 1.25 (shift 2), bias 0.5: R(0 + 1.25) = 1 (tie to even), then R(1 + 0.5) = 1.5, in both orders
    assert dot([(5, 1)], 'flt.e2m1.ord-tree.bias-post', 2, 0, bias_units=Q(2)) == \
        dot([(5, 1)], 'flt.e2m1.bias-post', 2, 0, bias_units=Q(2)) and \
        dot([(5, 1)], 'flt.e2m1.bias-post', 2, 0, bias_units=Q(2))[0] == 1.5
    rng = random.Random(35)
    for _ in range(300):
        pair = [(rng.randrange(-2**12, 2**12), rng.randrange(-2**8, 2**8))]
        base, mods = rng.choice([('flt.e2m1', ''), ('flt.e3m2', '.ftz'), ('flt.e5m3', '.rz'), ('flt.e4m2', '.nf'),
                                 ('flt.e2m2', '.ofsat')])
        spec = base + rng.choice(['', '.x-5', '.x7']) + mods
        b = Q(rng.randrange(-2**14, 2**14), rng.randrange(1, 9))
        sa = rng.randrange(0, 10)
        assert same(*[dot(pair, spec + o, sa, 0)[0] for o in ('', '.ord-tree')])
        assert dot(pair, spec, sa, 0)[1] == dot(pair, spec + '.ord-tree', sa, 0)[1]
        x, y = (dot(pair, spec + o + '.bias-post', sa, 0, bias_units=b) for o in ('', '.ord-tree'))
        assert same(x[0], y[0]) and x[1] == y[1]
    # a one-tap last chunk (K = 9 in chunks of 2) is rounded into the inner format like a sequential chunk
    terms = [Q(1)] * 8 + [Q(13)]
    v, e = dot_terms(terms, 'ch2_flt.e5m2.ord-tree_wide', 0)
    assert v == 8 + 12 and e['inner_inexact_steps'] == 1


# -- B6: product rounding format -----------------------------------------------------------------------------
def test_product_format_units_rounding_and_subnormals():
    # (i) the product format is in code-level units (times 2^x'); the accumulator's .x<e> does not act on it
    assert dot([(200, 100)], 'flt.e8m23.x-12.pr-e4m3', 0, 0)[0] == 240.0          # saturates at E4M3's 240
    assert dot([(200, 100)], 'flt.e8m23.pr-e4m3-x-7', 0, 0)[0] == 20480.0         # 156.25 * 2^7 -> 160 * 2^7
    # (ii) .rz of the accumulator does not act on the product; -rz does
    assert dot([(7, 1)], 'flt.e8m23.rz.pr-e2m1', 2, 0)[0] == 2.0                  # 1.75 -> 2 (tie to even)
    assert dot([(7, 1)], 'flt.e8m23.rz.pr-e2m1-rz', 2, 0)[0] == 1.5
    # (iii) subnormals of the product format (E2M1: 0.5 is subnormal, smallest normal 1)
    assert [dot([(n, 1)], 'wide.pr-e2m1', 3, 0)[0] for n in (2, 3, 5)] == [0.0, 0.5, 0.5]
    assert [dot([(n, 1)], 'wide.pr-e2m1-ftz', 3, 0)[0] for n in (3, 7, 8)] == [0.0, 0.0, 1.0]
    blu24 = 'ch16_flt.e4m7.x3.ftz.rz.pr-e4m7-x5-rz-ftz_flt.e4m7.x3.ftz.rz'
    assert resolve(blu24).name() == blu24
    for bad in ('wide.pr-e4m3-x0', 'wide.pr-e4m3-x65', 'wide.pr-e4m3-ftz-rz', 'control.pr-e4m3'):
        with pytest.raises(ValueError):
            resolve(bad)


# -- B7: flush to zero detects tininess before rounding ------------------------------------------------------
def test_ftz_is_tininess_before_rounding():
    e4m3 = FloatFormat.ieee(4, 3)
    assert e4m3.min_normal == Q(1, 64)
    assert e4m3.round(Q(61, 4096)) == Q(1, 64)                                    # gradual: rounds up to 1/64
    assert FloatFormat.ieee(4, 3, subnormals=False).round(Q(61, 4096)) == 0       # flush: 61/4096 < 1/64
    assert FloatFormat.ieee(4, 3, subnormals=False).round(Q(65, 4096)) == Q(1, 64)
    assert FloatFormat.ieee(4, 3, subnormals=False, mode='rz').round(Q(127, 4096)) == Q(30, 1024)


# -- B8: event fields ----------------------------------------------------------------------------------------
def test_event_fields_hand():
    v, e = dot([(5, 1)], 'flt.e2m1', 2, 0)                                       # 1.25 -> 1
    assert v == 1.0 and e == {'nonfinite_steps': 0, 'overflow_steps': 0, 'inexact_steps': 1, 'absorbed_steps': 0}
    v, e = dot([(8, 1), (1, 1), (1, 2), (8, 1)], 'flt.e5m2', 1, 0)             # 4, 4.5->4, 5, 9->8
    assert v == 8.0 and e['inexact_steps'] == 2 and e['absorbed_steps'] == 1
    v, e = dot([(1, 1), (16, 1)], 'flt.e5m2', 1, 0)                              # 0.5 + 8 = 8.5 -> 8: the state is lost
    assert v == 8.0 and e['absorbed_steps'] == 1
    v, e = dot([(1, 1)] * 5, 'flt.e2m1.ofsat', 0, 0)
    assert v == 3.0 and e == {'nonfinite_steps': 0, 'overflow_steps': 2, 'inexact_steps': 2, 'absorbed_steps': 2}
    v, e = dot([(3, 1)] * 4, 'flt.e2m1.ord-tree', 0, 0)                         # 6 -> inf twice, inf + inf
    assert v == math.inf and e['nonfinite_steps'] == 3 and e['overflow_steps'] == 2
    v, e = dot([(4, 1), (1, 1)], 'flt.e2m1.nf', 2, 0)                           # 1, then 0.25 rounds to +0
    assert v == 1.0 and e['absorbed_steps'] == 0 and e['inexact_steps'] == 0
    v, e = dot([(4, 1), (0, 1), (0, 1)], 'flt.e2m1.bias-post', 0, 0, bias_units=Q(1, 2))
    assert v == math.inf and e['nonfinite_steps'] == 4 and legacy_word('flt.e2m1', e) == 4
    v, e = dot([(4, 1)], 'flt.e2m1.bias-post', 0, 0, bias_units=Q(-9))           # +inf + -inf
    assert math.isnan(v)
    v, e = dot([(20, 1)], 'sat.w4.ord-tree', 0, 0)
    assert v == 7.0 and e['high'] == 1
    v, e = dot([(1, 1)], 'sat.w4.bias-pre', 0, 0, bias_units=Q(-30))
    assert v == -7.0 and e['low'] == 1


# -- B9: block names, negative L, RNE at L = 0, hwc refused, chunks of terms ---------------------------------
def test_block_grammar():
    for name in ('sat.w16.elt', 'control.elt', 'ch4_sat.w16.elt_wide', 'sat.w16.l-3', 'sat.w16.rne', 'sat.struct-0',
                 'wrap.w20.l-126.rne.elt.ord-tree.bias-post'):
        assert resolve(name).name() == name
    for bad in ('sat.w16.l127', 'sat.w16.l-127', 'sat.w16.l500', 'sat.w16.l-0', 'sat.w16.elt.l2', 'sat.w16.l03'):
        with pytest.raises(ValueError):
            resolve(bad)
    with pytest.raises(ValueError):
        dot([(1, 1)], 'sat.w16.elt', 0, 0)                                      # .elt is for block nodes
    a = [(0, [3] + [0] * 31), (0, [1] + [0] * 31), (0, [1] + [0] * 31)]
    w = [(-2, [1] + [0] * 31), (0, [1] + [0] * 31), (0, [1] + [0] * 31)]       # block terms 3/4, 1, 1 (anchor 0)
    assert B.block_dot(a, w, 'sat.w20', 0, 0, anchor=0)[0] == 2.0                        # floor(3/4) = 0
    assert B.block_dot(a, w, 'sat.w20.rne', 0, 0, anchor=0)[0] == 3.0                    # RNE at L = 0: 1
    assert B.block_dot(a, w, 'sat.w20.l-2', 0, 0, anchor=0)[0] == 2.75                   # two bits below the anchor
    assert B.block_dot(a, w, 'ch2_wide_wide', 0, 0)[1]['chunks'] == 2          # 3 block terms in chunks of 2
    with pytest.raises(ValueError):
        B.block_dot(a, w, 'sat.w20.ord-hwc', 0, 0, anchor=0)


# -- non-blocking test gaps ----------------------------------------------------------------------------------
def test_hwc_with_rows_and_columns():
    assert permutation('hwc', (2, 2, 3)) == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
    t = [1, 1, 1, 8, 8, 1, 2, 5]                        # cg 2, kh 2, kw 2: hwc sequence 1, 8, 1, 1, 1, 2, 8, 5
    pairs = [(v, 1) for v in t]
    assert dot(pairs, 'flt.e5m2.ord-hwc', 0, 0, taps=(2, 2, 2))[0] == 20.0      # row/column swapped would give 24
    assert dot(pairs, 'flt.e5m2', 0, 0)[0] == 28.0
    with pytest.raises(ValueError):
        dot(pairs, 'flt.e5m2.ord-hwc', 0, 0)                                    # no geometry: refused


def test_tree_shape_at_the_reference_level():
    # levels: (16, 21->20, 19->20, 5) -> (36->32, 25->24) -> 56; pairing from the right would give 64
    assert dot([(v, 1) for v in (8, 8, 5, 16, 3, 16, 5)], 'flt.e5m2.ord-tree', 0, 0)[0] == 56.0


def test_scale_exponent_on_a_generic_float():
    assert dot([(100, 1)], 'flt.e3m2', 0, 0)[0] == math.inf                     # largest finite 28
    assert dot([(100, 1)], 'flt.e3m2.x-3', 0, 0)[0] == 96.0                    # holds 12.5 -> 12, times 8


def test_mse_search_reaches_every_candidate_it_could_choose():
    # the search compares d = 0..6 exactly; d >= 4 was never the minimum in 12,000 random blocks of the four
    # tables (lane E1 r2 scratch search), so the cut at d <= 3 is not observable on these tables; what is
    # checked here is that the chosen exponent is the exact argmin over all seven candidates
    rng = random.Random(36)
    for name in B.SHARED:
        book = B.element_book(name)
        for _ in range(20):
            vals = [Q(rng.randrange(-64, 65)) * Q(2) ** rng.randrange(-10, 1) for _ in range(32)]
            vals[rng.randrange(32)] = Q(rng.randrange(1, 64)) * Q(2) ** rng.randrange(0, 6)
            e0 = B.exponent_maxabs(vals, book)
            errs = [(B.squared_error(vals, max(-127, e0 - d), book), d) for d in range(7)]
            assert B.exponent(vals, book, 'mse') == max(-127, e0 - min(errs)[1])


def test_supplement_vectors_reproduce_the_21_oracle():
    data = vectors.load_supplement()
    assert data['oracle_sources_sha256'] == vectors.oracle_hashes()
    words = []
    for case in data['cases']:
        value, events = dot(list(zip(case['a'], case['w'])), case['policy'], case['shift_a'], case['shift_w'],
                            case['certificate'])
        assert float(value).hex() == case['value_hex'] and legacy_word(case['policy'], events) == case['word']
        words.append(case['word'])
    assert 0xFFFF in words and 0xFFFF0000 in words and 0xFFFE0000 in words
