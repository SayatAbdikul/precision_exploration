"""Lane E1 reference models: shared-exponent block semantics against the simulator (CPU, exact)."""
from fractions import Fraction as Q
import random
import numpy as np
import pytest
import torch
from torch import nn
from tools.scaled_bridge_ext import blocks as B
from tools.scaled_bridge_ext.reference import dot
from tools.experiment_b_ext.shared import BlockQuantizer, block_conv2d, codebook, quantize_axis
from tools.experiment_b2.blocks import B2BlockQuantizer


def dyadic(rng, n, bits=6, lo=-6, hi=3):
    """Values exact in binary32 with few significant bits (so the simulator's binary32 arithmetic is exact)."""
    return [Q(rng.randrange(-(1 << bits), 1 << bits)) * Q(2) ** rng.randrange(lo, hi) for _ in range(n)]


@pytest.mark.parametrize('name', B.SHARED)
def test_element_table_midpoints_equal_simulator_boundaries(name):
    book = B.element_book(name)
    levels, boundaries, ties = codebook(name)
    assert [Q(float(v)) for v in levels] == [Q(u, 1 << book.shift) for u in book.units]
    assert [Q(float(v)) for v in boundaries] == book.midpoints      # the binary32 midpoints are exact
    assert list(ties) == list(book.ties)


def test_exponent_rule_hand():
    book = B.element_book('mxfp4_e2m1')                                # L_max = 6
    assert [B.exponent_maxabs([Q(v)], book) for v in (6, Q(13, 2), 3, Q(3, 1000), 0)] == [0, 1, -1, -10, -127]
    assert B.exponent_maxabs([6 * Q(2) ** 130], book) == 127 and B.quantize([6 * Q(2) ** 130], 127, book) == [12]
    assert B.quantize([Q(5, 4), Q(-7, 4), Q(5)], 0, book) == [2, -4, 8]   # 1.25 -> 1.0 (tie up? no: even code), 5 -> 4
    assert B.blocks([Q(1)] * 33, book, 'maxabs')[1] == (-2, [8] + [0] * 31)   # partial block, zero padding


@pytest.mark.parametrize('name', B.SHARED)
def test_maxabs_blocks_equal_simulator(name):
    rng = random.Random(21)
    book = B.element_book(name)
    sim = BlockQuantizer(name, 'maxabs', 'cpu')
    for _ in range(40):
        n = rng.randrange(1, 100)
        values = dyadic(rng, n, bits=rng.randrange(1, 10), lo=-12, hi=6)
        if rng.random() < 0.2:
            values[rng.randrange(n)] = Q(0)
        mine = B.reconstruct(B.blocks(values, book, 'maxabs'), book, n)
        theirs = sim(torch.tensor([float(v) for v in values], dtype=torch.float32))
        assert mine == [Q(float(v)) for v in theirs.tolist()]


@pytest.mark.parametrize('name', B.SHARED)
def test_mse_blocks_equal_simulator(name):
    rng = random.Random(22)
    book = B.element_book(name)
    sim = B2BlockQuantizer(name, 'mse', 'cpu')
    disagreements = 0
    for _ in range(30):
        n = 32 * rng.randrange(1, 3)
        values = dyadic(rng, n, bits=rng.randrange(2, 8), lo=-4, hi=2)
        mine = B.reconstruct(B.blocks(values, book, 'mse'), book, n)
        theirs = sim.core(torch.tensor([[float(v) for v in values]], dtype=torch.float32), search=True)[0]
        disagreements += mine != [Q(float(v)) for v in theirs.tolist()]
    assert disagreements == 0


def test_stored_channel_blocks_equal_simulator():
    rng = random.Random(23)
    name = 'mxfp6_e3m2'; book = B.element_book(name)
    n, c, h, w = 2, 40, 3, 2
    x = [[[[dyadic(rng, 1, 7, -8, 4)[0] for _ in range(w)] for _ in range(h)] for _ in range(c)] for _ in range(n)]
    theirs = quantize_axis(torch.tensor([[[[float(v) for v in r] for r in p] for p in im] for im in x], dtype=torch.float32),
                           BlockQuantizer(name, 'maxabs', 'cpu'), 1)
    for i in range(n):
        for y in range(h):
            for z in range(w):
                channels = [x[i][k][y][z] for k in range(c)]
                mine = B.reconstruct(B.blocks(channels, book, 'maxabs'), book, c)
                assert mine == [Q(float(theirs[i, k, y, z])) for k in range(c)]


def test_convolution_reblocking_equals_simulator():
    """Patch re-blocking along K (chw order, padding zeros inside blocks) and the exact block dot."""
    rng = random.Random(24)
    name = 'mxfp4_e2m1'; book = B.element_book(name)
    cin, cout, kh, kw, h, w = 5, 3, 3, 3, 5, 4                       # K = 45: one full and one partial block
    conv = nn.Conv2d(cin, cout, (kh, kw), stride=(2, 1), padding=(1, 2), dilation=(1, 2), bias=False)
    x_raw = [[[Q(rng.randrange(-12, 13), 4) for _ in range(w)] for _ in range(h)] for _ in range(cin)]
    stored = [[[None] * w for _ in range(h)] for _ in range(cin)]
    for y in range(h):
        for z in range(w):
            q = B.reconstruct(B.blocks([x_raw[c][y][z] for c in range(cin)], book, 'maxabs'), book, cin)
            for c in range(cin):
                stored[c][y][z] = q[c]
    k = cin * kh * kw
    w_blocks = [B.blocks([Q(rng.randrange(-12, 13), 8) for _ in range(k)], book, 'maxabs') for _ in range(cout)]
    weights = torch.tensor([[float(v) for v in B.reconstruct(wb, book, k)] for wb in w_blocks]).reshape(cout, cin, kh, kw)
    inputs = torch.tensor([[[[float(v) for v in r] for r in p] for p in stored]], dtype=torch.float32)
    theirs = block_conv2d(inputs, conv, BlockQuantizer(name, 'maxabs', 'cpu'), weights)
    oh, ow = theirs.shape[2:]
    for oc in range(cout):
        for oy in range(oh):
            for ox in range(ow):
                taps = B.patch(stored, 0, cin, kh, kw, oy, ox, (2, 1), (1, 2), (1, 2))
                a_blocks = B.blocks(taps, book, 'maxabs')
                value, _ = B.block_dot(a_blocks, w_blocks[oc], 'wide', book.shift, book.shift)
                assert value == float(theirs[0, oc, oy, ox])


def test_equal_exponents_equal_scalar_dot():
    rng = random.Random(25)
    book = B.element_book('mxfp8_e4m3')
    for _ in range(30):
        k = rng.randrange(1, 100); e = rng.randrange(-5, 5)
        ua = [rng.choice(book.units) for _ in range(k)]; uw = [rng.choice(book.units) for _ in range(k)]
        pad = (-k) % 32
        a_blocks = [(e, (ua + [0] * pad)[i:i + 32]) for i in range(0, k + pad, 32)]
        w_blocks = [(e, (uw + [0] * pad)[i:i + 32]) for i in range(0, k + pad, 32)]
        s = book.shift
        pairs = list(zip(ua, uw))
        assert B.block_dot(a_blocks, w_blocks, 'wide', s, s)[0] == dot(pairs, 'wide', s, s)[0] * 2.0 ** (2 * e) \
            == dot(pairs, 'wide', s - e, s - e)[0]
        for spec, elt in (('sat.w30', 'sat.w30.elt'), ('wrap.w26.l3', 'wrap.w26.l3.elt'), ('flt.e8m7', 'flt.e8m7.elt'),
                          ('flt.e5m3.ord-tree', 'flt.e5m3.elt.ord-tree'), ('ch16_sat.w20_wide', 'ch16_sat.w20.elt_wide')):
            # the register grid anchored at the common block exponent: identical to the scalar engine
            mine = B.block_dot(a_blocks, w_blocks, elt, s, s, anchor=2 * e, length=k)
            assert mine == dot(pairs, spec, s - e, s - e)
        assert B.block_dot(a_blocks, w_blocks, 'flt.e8m7', s, s)[0] == \
            B.block_dot(a_blocks, w_blocks, 'flt.e8m7', s, s, anchor=5)[0]       # floats do not see the anchor


def test_alignment_and_static_width_hand():
    a = [(2, [3, 0]), (0, [1, 1]), (-127, [0, 0])]
    w = [(2, [1, 5]), (0, [1, 0]), (5, [7, 7])]                       # sums 3 (E 4), 1 (E 0), 0 (ignored)
    r = B.alignment(a, w)
    assert r['e_min'] == 0 and r['aligned'] == [48, 1] and r['max_prefix'] == 49 and r['prefix_bits'] == 7
    s = B.static_width([(2, [1, -5]), (0, [3, 0])], max_a_units=4, e_a_lo=-3, e_a_hi=1)
    # anchor 0 - 3 = -3; terms 4*6 << (2+1+3) = 1536, 4*3 << (0+1+3) = 192 -> 1728, 12 signed bits
    assert s == {'anchor': -3, 'bound': 1728, 'signed_bits': 12}
