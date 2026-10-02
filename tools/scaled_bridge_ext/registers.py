"""W-bit two's-complement integer register on a 2^L grid of term units (-126 <= L <= 126).

Each term (an exact rational in product-grid units) enters the register as the integer round(term / 2^L) under the
low-bit mode: floor (toward minus infinity, the arithmetic shift) or rne (nearest, ties to the even integer). The
entry is exact when the term is a multiple of 2^L (a scalar node's integer products with L <= 0). After every add the exact integer sum is brought into
[-2^(W-1), 2^(W-1) - 1]:
  sat   clamped to the nearer end point (never wraps)
  wrap  reduced modulo 2^W into that range (two's-complement wrap-around)
The register value is R * 2^L units. Counters: `high`/`low` = adds whose exact sum lay above/below the range,
`inexact` = terms that lost low bits on entry.
"""
from __future__ import annotations
from .rounding import pow2, round_int, LOW_MODES

MIN_WIDTH, MAX_WIDTH = 2, 127
MAX_LOW_BITS = 126


class Register:
    def __init__(self, width, low_bits=0, low_mode='floor', overflow='sat'):
        if not MIN_WIDTH <= width <= MAX_WIDTH:
            raise ValueError(f'register width {width} outside {MIN_WIDTH}..{MAX_WIDTH}')
        if abs(low_bits) > MAX_LOW_BITS:
            raise ValueError(f'register grid exponent {low_bits} outside -{MAX_LOW_BITS}..{MAX_LOW_BITS}')
        if low_mode not in LOW_MODES or overflow not in ('sat', 'wrap'):
            raise ValueError('unknown register mode')
        self.width, self.low_bits, self.low_mode, self.overflow = width, low_bits, low_mode, overflow
        self.hi = (1 << (width - 1)) - 1
        self.lo = -(1 << (width - 1))
        self.high = self.low = self.inexact = 0

    def enter(self, term):
        """Integer of one term on the register grid (low bits dropped or rounded)."""
        scaled = term / pow2(self.low_bits)
        value = round_int(scaled, self.low_mode)
        if value != scaled:
            self.inexact += 1
        return value

    def fit(self, total):
        """Exact integer sum -> register content, counting the event."""
        if self.lo <= total <= self.hi:
            return total
        if total > self.hi:
            self.high += 1
        else:
            self.low += 1
        if self.overflow == 'sat':
            return self.hi if total > self.hi else self.lo
        return (total - self.lo) % (1 << self.width) + self.lo

    def add(self, content, term):
        return self.fit(content + self.enter(term))

    def value(self, content):
        return content * pow2(self.low_bits)
