"""Exact shared-scale search for a bounded 53-bit dyadic input domain.

Nonzero admitted values have magnitude in [2^-80, 2^80] and at most 53
significand bits. Their denominators are at most 2^132. Division by any E8M0
scale gives a denominator at most 2^259 and at most 197 significant decimal
digits, within the existing oracle's 200-digit exact-conversion domain. Both
inputs and all shared reconstructions lie on the existing 2^-149 grid.

This is an isolated accelerator prototype. Existing frozen execution does not
install it. Outside this domain, the verified FP32 helper or original oracle
retains the existing behavior.
"""
from bisect import bisect_left
from fractions import Fraction

from public.formats.oracle.manifest import manifest_sha256
from public.inference.reference.arithmetic import decimal, format_named, pow2, real
from tools.phase3.shared_scale_fast import FORMATS, GRID, mse_scale_fp32, tables


def admitted(value):
    if not isinstance(value, Fraction):
        return False
    if value == 0:
        return True
    if not Fraction(1, 1 << 80) <= abs(value) <= 1 << 80 or value.denominator & (value.denominator-1):
        return False
    numerator = abs(value.numerator)
    return (numerator//(numerator & -numerator)).bit_length() <= 53


def mse_scale_dyadic(values, format_name, *, coarse_candidates=100, fine_candidates=50, fallback=mse_scale_fp32):
    values = tuple(values)
    normalized = tuple(real(v) for v in values)
    normalized = tuple(Fraction(v) if not isinstance(v, Fraction) and v.is_finite() else v for v in normalized)
    if (format_name not in FORMATS or not 1 <= len(values) <= 32 or not all(admitted(v) for v in normalized)
            or any(type(n) is not int or n < 2 for n in (coarse_candidates, fine_candidates))):
        return fallback(values, format_name, coarse_candidates=coarse_candidates, fine_candidates=fine_candidates)
    fmt = format_named(format_name)
    integers = tuple(int(v*GRID) for v in normalized)
    candidates = tables(format_name, manifest_sha256(fmt.manifest))
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
