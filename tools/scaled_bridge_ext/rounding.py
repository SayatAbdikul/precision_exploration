"""Exact rounding of rationals: integer grids and generic binary floating-point formats.

Values are fractions.Fraction; infinities and NaN are the Python floats inf, -inf and nan. No host floating-point
arithmetic is used for any rounding decision.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
import math

INF = math.inf
LOW_MODES = ('floor', 'rne')


def pow2(e):
    return Q(1 << e) if e >= 0 else Q(1, 1 << -e)


def round_int(x, mode):
    """Integer nearest to the rational x under `mode`.

    floor  toward minus infinity: what dropping low bits of a two's-complement number does (an arithmetic shift)
    rne    to nearest, ties to the even integer
    """
    x = Q(x)
    q, r = divmod(x.numerator, x.denominator)          # Python floors
    if mode == 'floor' or r == 0:
        return q
    if mode == 'rne':
        c = 2 * r - x.denominator
        return q + 1 if c > 0 or (c == 0 and q & 1) else q
    raise ValueError(f'unknown rounding mode {mode!r}')


def floor_log2(m):
    """floor(log2 m) for a positive rational m."""
    t = m.numerator.bit_length() - m.denominator.bit_length()
    return t - 1 if pow2(t) > m else t


def signed_bits(value):
    """Smallest two's-complement width holding the integer value (sign bit included)."""
    value = int(value)
    return (value.bit_length() if value >= 0 else (-value - 1).bit_length()) + 1


@dataclass(frozen=True)
class FloatFormat:
    """Binary floating point with P significand bits.

    emin  exponent of the subnormal spacing (the smallest positive value is 2^emin)
    emax  exponent of the leading bit of the largest binade (largest finite value (2^P - 1) * 2^(emax - P + 1))
    subnormals  True: gradual underflow (round on the 2^emin grid below the smallest normal).
                False (flush to zero, contract 2.3 `.ftz`): tininess is detected BEFORE rounding: an exact value
                whose magnitude is below the smallest normal 2^(emin + P - 1) becomes +0; every other value is
                rounded to P significant bits (it is normal, so the subnormal grid never acts).
    overflow    'inf': IEEE (a value whose rounding with unbounded exponent exceeds the largest finite value is
                +-infinity); 'saturate': it becomes the largest finite value of its sign.
    Rounding is round-to-nearest-even unless mode = 'rz' (toward zero: the magnitude is floored, and an overflow
    gives the largest finite value of its sign, as IEEE round-toward-zero does). Zero is canonical +0.
    """
    P: int
    emin: int
    emax: int
    subnormals: bool = True
    overflow: str = 'inf'
    mode: str = 'rne'            # 'rne' or 'rz' (toward zero: the magnitude is floored; overflow gives the largest
                                 # finite value, as IEEE round-toward-zero does)

    @classmethod
    def ieee(cls, E, M, **kw):
        """IEEE-style layout: sign, E exponent bits (bias 2^(E-1) - 1, top code reserved for inf/NaN), M fraction bits."""
        if not (2 <= E <= 11 and 1 <= M <= 52):
            raise ValueError('float layout outside E 2..11, M 1..52')
        return cls(M + 1, 2 - (1 << (E - 1)) - M, (1 << (E - 1)) - 1, **kw)

    def scaled(self, k):
        """The same format on a grid 2^k times finer (exponents shifted by -k)."""
        return FloatFormat(self.P, self.emin - k, self.emax - k, self.subnormals, self.overflow, self.mode)

    @property
    def max_finite(self):
        return ((1 << self.P) - 1) * pow2(self.emax - self.P + 1)

    @property
    def min_normal(self):
        return pow2(self.emin + self.P - 1)

    def round_overflow(self, x):
        """(R(x), overflowed): overflowed is True when x is finite and its rounding with unbounded exponent exceeds
        the largest finite value (whatever the overflow rule then returns)."""
        if isinstance(x, float):
            return x, False
        x = Q(x)
        if x == 0:
            return Q(0), False
        sign = -1 if x < 0 else 1
        m = abs(x)
        if not self.subnormals and m < self.min_normal:
            return Q(0), False                                     # flush: tininess before rounding
        q = max(floor_log2(m) - self.P + 1, self.emin)           # exponent of the last kept bit
        value = round_int(m / pow2(q), 'rne' if self.mode == 'rne' else 'floor') * pow2(q)   # a carry is exact
        if value > self.max_finite:
            return (sign * INF if self.overflow == 'inf' and self.mode == 'rne' else sign * self.max_finite), True
        return (sign * value if value else Q(0)), False

    def round(self, x):
        """R(x) in this format; x is a Fraction or an infinity/NaN float (returned unchanged)."""
        return self.round_overflow(x)[0]

    def representable(self, x):
        return isinstance(x, float) or self.round(x) == x
