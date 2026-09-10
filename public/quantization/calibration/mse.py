"""Experiment A scale search. No optional external scaling is inferred."""
from __future__ import annotations

from fractions import Fraction
from decimal import Decimal

from public.inference.reference.arithmetic import decode, encode, format_named, pow2, real, decimal


def mse_scale(values, format_name, *, coarse_candidates=100, fine_candidates=50):
    values = tuple(real(v) for v in values)
    values = tuple(Fraction(v) if not isinstance(v, Fraction) and v.is_finite() else v for v in values)
    if not values or any(not isinstance(v, Fraction) for v in values):
        raise ValueError("calibration requires nonempty finite values")
    if any(type(n) is not int or n < 2 for n in (coarse_candidates, fine_candidates)):
        raise ValueError("scale search needs at least two candidates per stage")
    fmt = format_named(format_name)
    mode = fmt.manifest["scaling"]["mode"]

    def score(scale):
        errors = []
        for value in values:
            reconstructed = decode(fmt, encode(fmt, value, scale), scale)
            if isinstance(reconstructed, Decimal):
                if not reconstructed.is_finite():
                    return Decimal("Infinity")
                reconstructed = Fraction(reconstructed)
            errors.append((value - reconstructed) ** 2)
        return sum(errors, Fraction(0)) / len(values)

    if mode == "none":
        best, searched = Fraction(1), 1
    elif mode == "intrinsic_shared":
        # E8M0 is a discrete format: exhaustive legal scales avoid a hidden
        # external optimizer or rounded host logarithm at exponent boundaries.
        if fmt.manifest["block"]["shared_scale_format"] != "e8m0":
            raise ValueError("unsupported shared scale format")
        if len(values) > fmt.manifest["block"]["block_size"]:
            raise ValueError("shared scale search consumes one valid K block at a time")
        candidates = [pow2(exponent) for exponent in range(-127, 128)]
        best, searched = min(candidates, key=lambda s: (score(s), s)), len(candidates)
    elif mode == "required_mapping":
        decoded = [decode(fmt, code) for code in range(1 << fmt.bits)]
        maximum = max(Fraction(value) for value in decoded if isinstance(value, Fraction) or value.is_finite())
        span = max(abs(v) for v in values)
        if not span:
            best, searched = Fraction(1), 1
        else:
            step = span / maximum / coarse_candidates
            candidates = [i * step for i in range(1, coarse_candidates + 1)]
            coarse = min(candidates, key=lambda s: (score(s), s))
            lower, upper = max(step / fine_candidates, coarse - step), coarse + step
            candidates += [lower + (upper - lower) * i / (fine_candidates - 1) for i in range(fine_candidates)]
            best, searched = min(candidates, key=lambda s: (score(s), s)), len(candidates)
    else:
        raise ValueError("Experiment A forbids optional external scaling")
    # Freeze the decimal scale that execution will actually consume.
    stored = str(decimal(best))
    return {"schema_version": "2.0.0", "format": format_name, "scale": stored,
            "mse": str(decimal(score(real(stored)))), "candidate_count": searched,
            "policy": "intrinsic_or_required_only", "tie_break": "smallest_scale"}
