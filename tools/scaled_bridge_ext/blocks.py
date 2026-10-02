"""Shared-exponent (MXFP4/6/8, BFP6) block semantics, exact.

The structure is the simulator's (tools/experiment_b_ext/shared.py, tools/experiment_b2/blocks.py; imported for the
element tables only, never edited):
  * one E8M0 exponent e in [-127, 127] per block of 32 elements; element values u * 2^(e - s) with u an integer
    unit of the element table on the 2^-s grid (the table's binary32 levels, which are dyadic);
  * weights: blocks of 32 consecutive taps along K = C_in/groups * kh * kw (chw order) per output channel;
  * stored activations: blocks of 32 consecutive channels at every (n, y, x) position;
  * convolution: the patch of an output position (chw tap order, padded taps are zeros) is re-blocked along K into
    blocks of 32 with the activation rule and re-quantised; a linear layer reads its input's channel blocks, which
    are already K blocks;
  * the last block of an axis is zero-padded to 32 (padded elements are zeros, not stored).
Rules (the arithmetic is exact here; the simulator evaluates the same rule in binary32/binary64):
  maxabs  e = smallest integer with max|v| <= L_max * 2^e, clamped to [-127, 127]; an all-zero block has e = -127
  mse     d in 0..6, e = clamp(e_maxabs - d); minimise the exact sum of squared errors; ties keep the smallest d
Elements: nearest table level to v / 2^e, exact midpoints, the table's tie preference, clipping at the end points.
Dot: S_b = sum of u_a * u_w over block b (an exact integer), term T_b = S_b * 2^(e_a,b + e_w,b) in element product
units 2^-(s_a + s_w); the exact dot is sum_b T_b.
"""
from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from .rounding import floor_log2, pow2, signed_bits

BLOCK, E_LO, E_HI, CLIP_OCTAVES = 32, -127, 127, 6
SHARED = ('bfp6', 'mxfp4_e2m1', 'mxfp6_e3m2', 'mxfp8_e4m3')


@dataclass(frozen=True)
class ElementBook:
    name: str
    shift: int
    units: tuple
    ties: tuple

    @property
    def lmax(self):
        return Q(self.units[-1], 1 << self.shift)

    @property
    def midpoints(self):
        return [Q(a + b, 1 << (self.shift + 1)) for a, b in zip(self.units[:-1], self.units[1:])]

    def nearest(self, y):
        """Unit of the level nearest to y (exact): midpoint ties go up where the table prefers the upper code."""
        mids = self.midpoints
        pos = bisect_left(mids, y)
        if pos < len(mids) and mids[pos] == y and self.ties[pos]:
            pos += 1
        return self.units[pos]


@lru_cache(None)
def element_book(name):
    """The simulator's element table (binary32 levels, ascending; tie preference) as exact units."""
    from tools.experiment_b_ext.shared import codebook
    if name not in SHARED:
        raise ValueError(f'not a shared-exponent format: {name}')
    levels, _, ties = codebook(name)
    q = [Q(float(v)) for v in levels]
    shift = max(v.denominator.bit_length() - 1 for v in q)
    return ElementBook(name, shift, tuple(int(v * (1 << shift)) for v in q), tuple(bool(t) for t in ties))


def exponent_maxabs(values, book):
    m = max((abs(Q(v)) for v in values), default=Q(0))
    if m == 0:
        return E_LO
    r = m / book.lmax
    t = floor_log2(r)
    return max(E_LO, min(E_HI, t if pow2(t) == r else t + 1))


def quantize(values, e, book):
    scale = pow2(e)
    return [book.nearest(Q(v) / scale) for v in values]


def squared_error(values, e, book):
    return sum((Q(u, 1 << book.shift) * pow2(e) - Q(v)) ** 2 for u, v in zip(quantize(values, e, book), values))


def exponent(values, book, rule):
    e0 = exponent_maxabs(values, book)
    if rule == 'maxabs':
        return e0
    if rule != 'mse':
        raise ValueError(f'block rule {rule!r} is not admitted (maxabs, mse)')
    best = None
    for d in range(CLIP_OCTAVES + 1):
        e = max(E_LO, min(E_HI, e0 - d))
        err = squared_error(values, e, book)
        if best is None or err < best[0]:
            best = (err, e)
    return best[1]


def blocks(values, book, rule):
    """[(e, units)] for consecutive blocks of 32 of a 1-D sequence; the last block is zero-padded."""
    values = [Q(v) for v in values]
    values += [Q(0)] * ((-len(values)) % BLOCK)
    out = []
    for i in range(0, len(values), BLOCK):
        block = values[i:i + BLOCK]
        e = exponent(block, book, rule)
        out.append((e, quantize(block, e, book)))
    return out


def reconstruct(blocked, book, length=None):
    """Element values of a blocked sequence (padding dropped when length is given)."""
    values = [Q(u, 1 << book.shift) * pow2(e) for e, units in blocked for u in units]
    return values[:length] if length is not None else values


def check_length(n_blocks, length):
    """K must lie in the last block: 32 (n - 1) < K <= 32 n (K cannot be recovered from the blocks)."""
    if not (BLOCK * (n_blocks - 1) < length <= BLOCK * n_blocks):
        raise ValueError(f'length {length} does not fit {n_blocks} blocks of {BLOCK}')


def block_terms(a_blocks, w_blocks, granularity='block', length=None):
    """Terms in element product units: per block S_b * 2^(e_a + e_w), or per element in K order (the zero padding
    of the last block is not a tap: element terms stop at `length` = K, which is required for them)."""
    if len(a_blocks) != len(w_blocks):
        raise ValueError('block structures differ')
    if granularity == 'element' and length is None:
        raise ValueError('element terms need length = K (the padding of the last block is not a tap)')
    if length is not None:
        check_length(len(a_blocks), length)
    terms = []
    for (ea, ua), (ew, uw) in zip(a_blocks, w_blocks):
        scale = pow2(ea + ew)
        if granularity == 'block':
            terms.append(sum(a * w for a, w in zip(ua, uw)) * scale)
        elif granularity == 'element':
            terms.extend(a * w * scale for a, w in zip(ua, uw))
        else:
            raise ValueError('granularity is block or element')
    return terms[:length] if granularity == 'element' and length is not None else terms


def dyadic(value):
    """(odd integer m, exponent e) with value = m * 2^e for a nonzero dyadic rational (a binary64 bias)."""
    value = Q(value)
    den = value.denominator
    if den & (den - 1):
        raise ValueError('not a dyadic rational')
    m, e = value.numerator, -(den.bit_length() - 1)
    while m % 2 == 0:
        m, e = m // 2, e + 1
    return m, e


def alignment(a_blocks, w_blocks, bias_units=None):
    """Exact two-limb feasibility of one dot: block sums aligned to the smallest exponent of a non-zero block sum.

    Returns E_min, the aligned integers, the largest |prefix| in block order and its signed width, and the
    order-free bound sum_b |S_b| 2^(E_b - E_min) with its width (what a kernel checks before a two-limb sum).
    bias_units (element product units, a dyadic rational) is one more aligned term after the blocks: the exact
    bias of `wide.bias-pre/post` on a shared-exponent node.
    """
    sums = [(sum(a * w for a, w in zip(ua, uw)), ea + ew) for (ea, ua), (ew, uw) in zip(a_blocks, w_blocks)]
    if bias_units:
        sums.append(dyadic(bias_units))
    live = [(s, e) for s, e in sums if s]
    if not live:
        return {'e_min': None, 'aligned': [], 'max_prefix': 0, 'prefix_bits': 1, 'bound': 0, 'bound_bits': 1}
    e_min = min(e for _, e in live)
    aligned = [s << (e - e_min) for s, e in live]
    prefix, worst = 0, 0
    for v in aligned:
        prefix += v
        worst = max(worst, abs(prefix))
    bound = sum(abs(v) for v in aligned)
    return {'e_min': e_min, 'aligned': aligned, 'max_prefix': worst, 'prefix_bits': signed_bits(worst),
            'bound': bound, 'bound_bits': signed_bits(bound)}


def static_width(w_blocks, max_a_units, e_a_lo, e_a_hi):
    """Certified width of an exact fixed-point register for one output channel of a node.

    The register is anchored at the finest reachable grid 2^(min_b e_w,b + e_a_lo) element product units and must
    hold sum_b max|u_a| * sum|u_w,b| * 2^(e_w,b + e_a_hi) (every prefix, any order). e_a_lo/e_a_hi bound the
    activation block exponents (e_a_lo is -127 unless a closure bound is proved).
    """
    floor = min(ew for ew, _ in w_blocks) + e_a_lo
    total = sum(max_a_units * sum(abs(u) for u in uw) << (ew + e_a_hi - floor) for ew, uw in w_blocks)
    return {'anchor': floor, 'bound': total, 'signed_bits': signed_bits(total)}


def patch(x, c_first, cg, kh, kw, oy, ox, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """Tap values of one output position in chw order; x[c][y][x] are element values, padded taps are 0."""
    out = []
    height, width = len(x[0]), len(x[0][0])
    for c in range(cg):
        for ky in range(kh):
            for kx in range(kw):
                iy = oy * stride[0] - padding[0] + ky * dilation[0]
                ix = ox * stride[1] - padding[1] + kx * dilation[1]
                out.append(Q(x[c_first + c][iy][ix]) if 0 <= iy < height and 0 <= ix < width else Q(0))
    return out


def block_dot(a_blocks, w_blocks, spec, shift_a, shift_w, anchor=None, certificate=None, bias_units=None,
              length=None):
    """(binary64 code-level dot, events) of one block-scaled dot under a 2.3 policy.

    The policy acts on the block terms in block order (one term per block sum, the exact block dot of an MX unit)
    or, when its (inner) stage carries `.elt`, on the element products in K order; `length` = K is then required
    (it drops the zero padding of the last block). A chunk of ch<C> is C terms. An integer register's grid is
    2^(anchor + L) element product units, with the node's anchor (certify.NodeCertificate.blocks: max e_w + e_a,hi):
    a policy with an integer stage needs `anchor` or the certificate, nothing defaults it. Float formats are
    defined in code-level units as for scalar formats and do not see the anchor; a `wide` stage is exact and its
    bias is added exactly (no fixed grid), so neither needs it. `.ord-hwc` is refused (no tap geometry on block
    terms). bias_units is in element product units. certificate: a 2.3 NodeCertificate of this node built by
    NodeCertificate.blocks (a 2.1 dict or a scalar certificate is refused).
    """
    from .policies import resolve
    from .reference import dot_terms, to_binary64
    policy = resolve(spec)
    if certificate is not None:
        if isinstance(certificate, dict) or certificate.exact:
            raise ValueError('a shared-exponent node needs its NodeCertificate.blocks certificate')
        if anchor is not None and anchor != certificate.anchor:
            raise ValueError('the anchor differs from the certificate anchor')
        anchor = certificate.anchor
    if anchor is None:
        if any(stage.kind == 'int' for stage in policy.stages):
            raise ValueError('an integer register on a shared-exponent node needs the node anchor (or its certificate)')
        anchor = 0                                                   # exact rescaling: no value depends on it
    if policy.elementwise and length is None:
        raise ValueError('.elt needs length = K (the padding of the last block is not a tap)')
    granularity = 'element' if policy.elementwise else 'block'
    terms = [t / pow2(anchor) for t in block_terms(a_blocks, w_blocks, granularity, length)]
    shift = shift_a + shift_w - anchor
    bias = None if bias_units is None else Q(bias_units) / pow2(anchor)
    value, events = dot_terms(terms, policy, shift, certificate, None, bias, wide_grid=False)
    return to_binary64(value, shift), events
