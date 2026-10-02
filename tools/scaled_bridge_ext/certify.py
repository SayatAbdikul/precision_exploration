"""Certified register widths of the contract 2.3 draft (A2.1, A2.2, A2.4, A2.6, A3.3), exact integers.

One definition for every integer stage, so that a relative name (`abs-<d>`, `struct-<d>`) has one meaning:

  W_cert(stage) = max over output channels c (and over chunks j for an inner stage) of signed_range_bits(R_c,j),

where R_c,j is a closed integer range of the REGISTER CONTENT (units of the register grid 2^L) that contains every
state the register can hold when no overflow event occurs:

  1. Term ranges. The node certificate gives one exact range [lo_k, hi_k] (containing 0) per term k and channel,
     in the units the stage receives: product-grid units on scalar nodes (tap k: [-max|a|*|w_k|, max|a|*|w_k|]
     for `abs`; [w_k*a_lo', w_k*a_hi] or [w_k*a_hi, w_k*a_lo'] for `struct`, a_lo' the 2.1 structural input
     minimum), anchored element-product units on shared-exponent nodes (block term b:
     +-max|u_a| * sum_{k in b}|u_w,k| * 2^(e_w,b + e_a,hi - anchor); element term k: +-max|u_a|*|u_w,k|*2^(...)).
     Summing them over all K taps gives exactly the 2.1 per-channel ranges.
  2. Sum. lo = sum of lo_k, hi = sum of hi_k over the n terms the stage receives (all terms; or one chunk). Every
     prefix and every subset sum (any order, any tree node) lies in [lo, hi] because every term range contains 0.
  3. Entry onto the grid g = 2^L (registers.py):
       exact entry (every term is an integer multiple of g: scalar terms with L <= 0; chunk values of an integer
       inner stage whose grid is a multiple of the outer grid):        [ceil(lo/g), floor(hi/g)]
       floor entry otherwise (each term loses less than one unit):      [floor(lo/g) - n + 1, floor(hi/g)]
       rne entry otherwise (each term moves by at most half a unit):    [ceil(lo/g - n/2), floor(hi/g + n/2)]
  4. Bias (`.bias-pre`, `.bias-post`): B_c = RNE(b_c / g), the integer the register receives; the range becomes
     [lo' + min(B_c, 0), hi' + max(B_c, 0)] (covers the preloaded bias, every prefix with or without it, and the
     final add).
  5. Chunked outer stage: the terms are the chunk values. An integer inner chunk (c, j) contributes its range
     from steps 2-3 times g_in when the inner register is at least that wide (no inner event is possible), else
     the inner register's whole range [-2^(W_in-1), 2^(W_in-1)-1] * g_in (a saturated or wrapped value stays
     inside it); a `wide` inner chunk contributes [lo, hi] of its terms. Then steps 3-4 with n = number of chunks.

For a plain register (L = 0, no bias, not chunked) on a scalar node, step 3 is the exact entry and W_cert equals
the 2.1 fields `signed_bits_absolute` / `signed_bits_structural` (the 2.1 certificate dict is accepted for those
names only). A relative name resolves to W = max(2, W_cert - d).
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
import math
from .orders import permutation
from .policies import resolve
from .rounding import pow2, round_int


def signed_range_bits(lo, hi):
    """Smallest two's-complement width W with -2^(W-1) <= lo and hi <= 2^(W-1) - 1 (the 2.1 definition)."""
    lo, hi = int(lo), int(hi)
    if lo > 0 or hi < 0:
        raise ValueError('a register range must contain zero')
    return max(hi.bit_length(), (-lo - 1).bit_length() if lo < 0 else 0) + 1


def entered_range(lo, hi, n, low_bits, mode, exact):
    """Step 3: integer range of the register content for terms summing to within [lo, hi]."""
    g = pow2(low_bits)
    lo, hi = Q(lo) / g, Q(hi) / g
    if exact:
        return math.ceil(lo), math.floor(hi)
    if mode == 'floor':
        return math.floor(lo) - n + 1, math.floor(hi)
    if mode == 'rne':
        return math.ceil(lo - Q(n, 2)), math.floor(hi + Q(n, 2))
    raise ValueError(f'unknown entry mode {mode!r}')


@dataclass
class NodeCertificate:
    """Per-term ranges of one MAC node (step 1).

    terms   {'abs': [[(lo, hi) per term] per channel], 'struct': ...} in the units the first stage receives,
            terms in the engine's order (chw taps; blocks; element products with '.elt' keys on block nodes)
    exact   True on scalar nodes: every term is an integer of the product grid
    bias    per-channel node bias in the same units (exact rationals), for bias policies
    anchor  shared-exponent nodes: the anchor (units are element products / 2^anchor)
    taps    (cg, kh, kw) of a convolution (needed for hwc)
    """
    terms: dict
    exact: bool = True
    bias: list = None
    anchor: int = 0
    taps: tuple = None

    @classmethod
    def scalar(cls, weights, a_lo, a_hi, structural_lo=None, bias=None, taps=None):
        """weights: per output channel, the integer weight units in chw tap order; [a_lo, a_hi] the input code range
        (contains 0); structural_lo the 2.1 structural input minimum (default a_lo)."""
        s_lo = a_lo if structural_lo is None else structural_lo
        if not (a_lo <= s_lo <= 0 <= a_hi):
            raise ValueError('input ranges must contain zero and nest')
        max_a = max(-a_lo, a_hi)
        absolute = [[(-max_a * abs(w), max_a * abs(w)) for w in row] for row in weights]
        structural = [[(w * s_lo, w * a_hi) if w >= 0 else (w * a_hi, w * s_lo) for w in row] for row in weights]
        return cls({'abs': absolute, 'struct': structural}, True, bias, 0, taps)

    @classmethod
    def blocks(cls, w_blocks, max_a_units, e_a_hi, length, bias=None):
        """w_blocks: per output channel, [(e_w, units)] blocks along K (blocks.py); max_a_units the largest |u| of
        the activation element table; e_a_hi an upper bound of the activation block exponents; length = K."""
        anchor = max(ew for row in w_blocks for ew, _ in row) + e_a_hi
        block, element = [], []
        for row in w_blocks:
            scale = [pow2(ew + e_a_hi - anchor) for ew, _ in row]
            block.append([(-m, m) for m in (max_a_units * sum(abs(u) for u in uw) * s for (_, uw), s in zip(row, scale))])
            element.append([(-m, m) for m in (max_a_units * abs(u) * s for (_, uw), s in zip(row, scale) for u in uw)][:length])
        bias = None if bias is None else [Q(b) / pow2(anchor) for b in bias]
        return cls({'abs': block, 'abs.elt': element}, False, bias, anchor, None)

    def ranges(self, reference, element=False):
        key = reference + ('.elt' if element else '')
        if key not in self.terms:
            raise ValueError(f'no {key!r} term ranges on this node (struct is not defined on shared-exponent nodes)')
        return self.terms[key]


def _stage_range(stage, lo, hi, n, exact, bias):
    """Steps 3 and 4 for one integer stage."""
    lo2, hi2 = entered_range(lo, hi, n, stage.low_bits, stage.low_mode, exact)
    if stage.bias:
        if bias is None:
            raise ValueError('a bias policy needs the node bias in its certificate')
        b = round_int(Q(bias) / pow2(stage.low_bits), 'rne')
        lo2, hi2 = lo2 + min(b, 0), hi2 + max(b, 0)
    return lo2, hi2


def _sequence(ranges, policy, taps):
    order = policy.order if policy.kind in ('int', 'float') else 'chw'
    if order == 'hwc' and taps is None:
        raise ValueError('hwc needs the convolution tap geometry')
    return [ranges[k] for k in permutation(order, taps if order == 'hwc' else len(ranges))]


def _plain(policy):
    return policy.kind == 'int' and policy.low_bits == 0 and not policy.bias


def certified_widths(spec, certificate):
    """Per stage: W_cert (int) of each integer stage whose name is relative; None for every other stage."""
    policy = resolve(spec)
    stages = policy.stages
    needs = [s.kind == 'int' and s.reference != 'global' for s in stages]
    if not any(needs):
        return tuple(None for _ in stages)
    if certificate is None:
        raise ValueError('a relative register width needs a node certificate')
    if isinstance(certificate, dict):                                # the 2.1 certificate of a scalar node
        if policy.kind != 'int' or not _plain(policy):
            raise ValueError('the 2.1 certificate defines plain registers only; pass a 2.3 NodeCertificate')
        return (certificate['signed_bits_absolute' if policy.reference == 'abs' else 'signed_bits_structural'],)
    first = stages[0]
    reference = next(s.reference for s, need in zip(stages, needs) if need)
    if any(need and s.reference != reference for s, need in zip(stages, needs)):
        raise ValueError('inner and outer relative widths must use the same reference (abs or struct)')
    channels = certificate.ranges(reference, policy.elementwise)
    bias = certificate.bias or [None] * len(channels)
    if policy.kind != 'chunk':
        if policy.kind != 'int':
            return (None,)
        width = 0
        for row, b in zip(channels, bias):
            lo, hi = sum(r[0] for r in row), sum(r[1] for r in row)
            exact = certificate.exact and policy.low_bits <= 0
            width = max(width, signed_range_bits(*_stage_range(policy, lo, hi, len(row), exact, b)))
        return (width,)
    inner, outer = stages
    w_inner = w_outer = 0
    in_exact = certificate.exact and inner.kind == 'int' and inner.low_bits <= 0
    chunk_values = []
    for row in channels:
        seq = _sequence(row, inner, certificate.taps)
        values = []
        for i in range(0, len(seq), policy.chunk):
            part = seq[i:i + policy.chunk]
            lo, hi = sum(r[0] for r in part), sum(r[1] for r in part)
            if inner.kind == 'int':
                rlo, rhi = _stage_range(inner, lo, hi, len(part), in_exact, None)
                bits = signed_range_bits(rlo, rhi)
                w_inner = max(w_inner, bits)
                values.append((bits, rlo, rhi))
            else:
                values.append((None, lo, hi))
        chunk_values.append(values)
    if needs[0]:
        w_inner_reg = inner.width(w_inner)
    else:
        w_inner_reg = inner.amount if inner.kind == 'int' else None
    if outer.kind == 'int':
        if inner.kind == 'int':
            g_in, exact_out = pow2(inner.low_bits), inner.low_bits >= outer.low_bits
        else:                                                         # wide inner: exact sums of the terms
            g_in, exact_out = Q(1), certificate.exact and outer.low_bits <= 0
        for values, b in zip(chunk_values, bias):
            lo = hi = Q(0)
            for bits, rlo, rhi in values:
                if bits is not None and bits > w_inner_reg:                  # inner events possible
                    rlo, rhi = -(1 << (w_inner_reg - 1)), (1 << (w_inner_reg - 1)) - 1
                lo, hi = lo + rlo * g_in, hi + rhi * g_in
            w_outer = max(w_outer, signed_range_bits(*_stage_range(outer, lo, hi, len(values), exact_out, b)))
    return (w_inner if needs[0] else None, w_outer if needs[1] else None)


def register_widths(spec, certificate=None):
    """Per stage: the register width W of each integer stage (None for other stages)."""
    policy = resolve(spec)
    certified = certified_widths(policy, certificate)
    return tuple(s.width(c) if s.kind == 'int' else None for s, c in zip(policy.stages, certified))
