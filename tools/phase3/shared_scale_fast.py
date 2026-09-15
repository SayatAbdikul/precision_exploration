"""Exact integer-grid evaluation of all 255 shared scales for admitted FP32 blocks.

For finite FP32 values in [2^-100, 2^100] (plus zero), the oracle's
200-significant-digit Decimal conversions/divisions are exact at every legal
scale. The largest denominator after division is 2^250, with at most 24
significand bits: fewer than 184 decimal significant digits. All reconstructions
also fit that precision. A common 2^-149 grid therefore reproduces the exact
squared errors. Tied nearest codes have the same squared error, regardless of
their RNE code parity. Scales are visited in increasing order, retaining the
first minimum. Inputs outside this deliberately bounded domain use the oracle.
"""
from bisect import bisect_left
from fractions import Fraction
from functools import lru_cache

from public.formats.oracle.manifest import manifest_sha256
from public.inference.reference.arithmetic import decimal, format_named, pow2, real
from public.quantization.calibration.mse import mse_scale as oracle_mse_scale

FORMATS = frozenset({"bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"})
GRID = 1 << 149


def admitted(value):
    if not isinstance(value, Fraction):
        return False
    if value == 0:
        return True
    if not Fraction(1, 1 << 100) <= abs(value) <= (1 << 100):
        return False
    if value.denominator & (value.denominator-1):
        return False
    numerator = abs(value.numerator)
    odd = numerator // (numerator & -numerator)
    return odd.bit_length() <= 24


@lru_cache(maxsize=8)
def tables(name, identity):
    fmt = format_named(name)
    if (manifest_sha256(fmt.manifest) != identity or name not in FORMATS
            or fmt.manifest["rounding"] != "rne" or fmt.manifest["overflow"] != "saturate"
            or fmt.manifest["block"]["shared_scale_format"] != "e8m0"):
        raise ValueError("unsupported integer-grid shared format")
    finite = sorted({Fraction(v) for code in range(1 << fmt.bits) if (v := fmt.decode(code)).is_finite()})
    smallest = [v*(1 << 22) for v in finite]  # 2^-127 scale on the 2^-149 grid.
    if any(v.denominator != 1 for v in smallest):
        raise ValueError("shared reconstructions do not lie on the admitted grid")
    return tuple(tuple(int(v) << exponent for v in smallest) for exponent in range(255))


def mse_scale_fp32(values, format_name, *, coarse_candidates=100, fine_candidates=50, fallback=oracle_mse_scale):
    values = tuple(values)
    normalized = tuple(real(value) for value in values)
    normalized = tuple(Fraction(value) if not isinstance(value, Fraction) and value.is_finite() else value for value in normalized)
    if (format_name not in FORMATS or not 1 <= len(values) <= 32 or not all(admitted(v) for v in normalized)
            or any(type(n) is not int or n < 2 for n in (coarse_candidates, fine_candidates))):
        return fallback(values, format_name, coarse_candidates=coarse_candidates, fine_candidates=fine_candidates)
    fmt = format_named(format_name)
    candidates = tables(format_name, manifest_sha256(fmt.manifest))
    integers = tuple(int(value*GRID) for value in normalized)
    best_error, best_index = None, None
    for index, levels in enumerate(candidates):
        error = 0
        for value in integers:
            at = bisect_left(levels, value)
            if at == 0:
                distance = levels[0]-value
            elif at == len(levels):
                distance = value-levels[-1]
            else:
                distance = min(value-levels[at-1], levels[at]-value)
            error += distance*distance
        if best_error is None or error < best_error:
            best_error, best_index = error, index
    return {"schema_version": "2.0.0", "format": format_name, "scale": str(decimal(pow2(best_index-127))),
            "mse": str(decimal(Fraction(best_error, len(integers)*GRID*GRID))), "candidate_count": 255,
            "policy": "intrinsic_or_required_only", "tie_break": "smallest_scale"}
