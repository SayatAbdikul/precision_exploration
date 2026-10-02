"""Codebooks as data, exact-dyadic stores and reconstruction.

A codebook is a plain dictionary: ascending integer `units` on a 2^-shift
grid, the stored `codes`, and the midpoint tie preference. Nothing here knows
format names except `scalar_codebook`, which derives the data from an accepted
manifest. Recipes may also supply their own codebooks (for example unsigned
activation ranges).
"""
from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
import json
import numpy as np
from .common import ROOT, reference


class NoDyadicGrid(ValueError):
    """The format has finite levels that are not dyadic rationals."""


@lru_cache(None)
def _scalar(name):
    from public.formats.oracle.number_format import NumberFormat
    path = ROOT / f'public/formats/manifests/accepted/{name}.json'
    manifest = json.loads(path.read_text())
    if manifest['scaling']['mode'] == 'intrinsic_shared':
        raise ValueError('shared-exponent format needs the block contract, not a scalar codebook')
    fmt = NumberFormat(manifest)
    by_value = {}
    for code in range(1 << fmt.bits):
        value = fmt.decode(code)
        if value.is_finite():
            q = Fraction(value)
            by_value[q] = min(code, by_value.get(q, code))
    values = sorted(by_value)
    if any(q.denominator & (q.denominator - 1) for q in values):
        raise NoDyadicGrid(f'{name}: finite levels are not all dyadic')
    shift = max(q.denominator.bit_length() - 1 for q in values)
    codes = [by_value[q] for q in values]
    return {'id': name, 'bits': fmt.bits, 'shift': shift,
            'units': [int(q * (1 << shift)) for q in values], 'codes': codes,
            'choose_upper_tie': [(b & 1, b) < (a & 1, a) for a, b in zip(codes[:-1], codes[1:])],
            'origin': 'accepted manifest finite levels; lowest code per value; code-parity then code-order midpoint ties',
            'manifest': reference(path)}


def scalar_codebook(name):
    return json.loads(json.dumps(_scalar(name)))


class Codebook:
    def __init__(self, data):
        units = [int(u) for u in data['units']]
        codes = [int(c) for c in data['codes']]
        ties = [bool(t) for t in data['choose_upper_tie']]
        shift = int(data['shift'])
        if len(units) < 2 or len(codes) != len(units) or len(ties) != len(units) - 1 or not 0 <= shift <= 60:
            raise ValueError('malformed codebook')
        if any(b <= a for a, b in zip(units[:-1], units[1:])) or min(codes) < 0 or max(codes) > 65535:
            raise ValueError('codebook units must increase strictly and codes fit 16 bits')
        # Levels and adjacent midpoints must be exact binary64 values.
        if max(abs(u) for u in units) >= 2**52:
            raise ValueError('codebook units exceed the exact binary64 midpoint domain')
        self.data, self.id, self.shift = data, str(data['id']), shift
        self.units = np.array(units, dtype=np.int64)
        self.levels = np.ldexp(self.units.astype(np.float64), -shift)
        self.bounds = np.ldexp((self.units[:-1] + self.units[1:]).astype(np.float64), -shift - 1)
        self.codes = np.array(codes, dtype=np.uint8 if max(codes) < 256 else np.uint16)
        self.ties = np.array(ties, dtype=bool)
        self.lo, self.hi = units[0], units[-1]
        self.has_zero = 0 in units
        self.float32_exact = all(int(np.float32(u)) == u for u in units)
        for u, level in zip(units, self.levels):
            if Fraction(float(level)) != Fraction(u, 1 << shift):
                raise ValueError('codebook level is not exact in binary64')
        for a, b, bound in zip(units[:-1], units[1:], self.bounds):
            if Fraction(float(bound)) != Fraction(a + b, 1 << (shift + 1)):
                raise ValueError('codebook midpoint is not exact in binary64')


def quantize(raw, scale, book, *, b_weight=False):
    """Store rule. Returns codes, units and one diagnostics dict per leading index.

    b_weight=True reproduces B's FP32 normalisation for weight tensors (and for
    tracing B's own FP32 states); the contract store uses binary64.
    """
    kind = np.float32 if b_weight else np.float64
    raw = np.asarray(raw, dtype=kind)
    scale = np.asarray(scale, dtype=kind)
    if not np.isfinite(raw).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('nonfinite bridge input/scale')
    with np.errstate(over='ignore'):
        y = raw / scale
    pos = np.searchsorted(book.bounds, y, side='left')
    adjacent = np.minimum(pos, len(book.bounds) - 1)
    tie = y == book.bounds[adjacent]
    pos += tie & book.ties[adjacent]
    codes, units = book.codes[pos], book.units[pos]
    n = raw.shape[0] if raw.ndim else 1
    def count(mask):
        return np.asarray(mask).reshape(n, -1).sum(1)
    low, high, ties = count(y < book.levels[0]), count(y > book.levels[-1]), count(tie)
    zeros, overflow = count(units == 0), count(~np.isfinite(y))
    size = int(raw.size // n)
    diagnostics = [{'elements': size, 'clipped_low': int(low[i]), 'clipped_high': int(high[i]),
                    'ties': int(ties[i]), 'zeros': int(zeros[i]),
                    'normalization_overflow': int(overflow[i])} for i in range(n)]
    return codes, units, diagnostics


def reconstructed(units, scale, book):
    return np.ldexp(np.asarray(units, dtype=np.float64), -book.shift) * np.asarray(scale, dtype=np.float64)
