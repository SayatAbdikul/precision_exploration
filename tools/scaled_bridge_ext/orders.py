"""Summation orders of one reduction.

Taps of a convolution output are indexed k = c*kh*kw + ky*kw + kx (input channel of the group c, kernel row ky,
kernel column kx); padded positions are taps whose product is zero. The engine's order is k = 0, 1, ..., K-1.

  chw   k ascending (the engine's order; a sequential MAC iterating input channel, then kernel row, then column)
  rev   k descending
  hwc   kernel row, kernel column, then input channel innermost (an NHWC / channel-last sequential MAC)
  tree  pairwise adder tree over the chw sequence: at every level elements (0,1), (2,3), ... are added and an odd
        last element moves up unchanged, until one value is left
"""
from __future__ import annotations

ORDERS = ('chw', 'rev', 'hwc', 'tree')


def permutation(name, taps):
    """Tap indices in summation order for a sequential order; taps = (cg, kh, kw) or K for a linear layer."""
    cg, kh, kw = (taps, 1, 1) if isinstance(taps, int) else taps
    k = cg * kh * kw
    if name in ('chw', 'tree'):
        return list(range(k))
    if name == 'rev':
        return list(range(k - 1, -1, -1))
    if name == 'hwc':
        return [c * kh * kw + ky * kw + kx for ky in range(kh) for kx in range(kw) for c in range(cg)]
    raise ValueError(f'unknown order {name!r}')


def tree(values, combine):
    """Pairwise reduction; combine(left, right) is applied at every internal node, left before right."""
    level = list(values)
    if not level:
        raise ValueError('empty reduction')
    while len(level) > 1:
        nxt = [combine(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]
