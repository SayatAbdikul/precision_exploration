"""Exact scale-search pruning for the already verified bounded dyadic domain.

For these codebooks, doubling a code value either leaves the finite range or
produces another existing value. Once a scale contains every input, larger
scales cannot reduce nearest-value squared error: inside that range their
levels are a subset; outside it the existing endpoint is at least as close.
Smaller scales are evaluated in descending order until endpoint clipping alone
exceeds the best error. That bound is monotone as the range shrinks. Equal-error
smaller scales are still evaluated and selected, preserving the original tie.
"""
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
from functools import lru_cache

from public.formats.oracle.manifest import manifest_sha256
from public.inference.reference.arithmetic import decimal, format_named, pow2, real
from tools.phase3.shared_scale_dyadic import admitted as admitted_53, mse_scale_dyadic
from tools.phase3.shared_scale_fast import FORMATS, GRID, admitted as admitted_24, tables


@lru_cache(maxsize=4)
def prunable_tables(name, identity):
    candidates = tables(name, identity)
    base = candidates[0]
    values = set(base)
    if 0 not in values or not base[0] < 0 < base[-1] or any(2*v not in values for v in base if base[0] <= 2*v <= base[-1]):
        raise ValueError("codebook does not satisfy exact power-of-two nesting")
    return candidates


def covering_index(value, endpoint):
    if value <= endpoint:
        return 0
    index = max(0, value.bit_length()-endpoint.bit_length())
    if endpoint << index < value:
        index += 1
    return index


def score(levels, counts):
    total = 0
    for value, count in counts.items():
        at = bisect_left(levels, value)
        if at == 0:
            distance = levels[0]-value
        elif at == len(levels):
            distance = value-levels[-1]
        else:
            distance = min(value-levels[at-1], levels[at]-value)
        total += count*distance*distance
    return total


def mse_scale_pruned(values, format_name, *, coarse_candidates=100, fine_candidates=50, diagnostics=None, fallback=mse_scale_dyadic):
    values = tuple(values)
    normalized = tuple(real(v) for v in values)
    normalized = tuple(Fraction(v) if not isinstance(v, Fraction) and v.is_finite() else v for v in normalized)
    if (format_name not in FORMATS or not 1 <= len(values) <= 32
            or not all(admitted_53(v) or admitted_24(v) for v in normalized)
            or any(type(n) is not int or n < 2 for n in (coarse_candidates, fine_candidates))):
        if diagnostics is not None:
            diagnostics.update(fallback=True)
        return fallback(values, format_name, coarse_candidates=coarse_candidates, fine_candidates=fine_candidates)
    fmt = format_named(format_name)
    candidates = prunable_tables(format_name, manifest_sha256(fmt.manifest))
    counts = Counter(int(v*GRID) for v in normalized)
    minimum, maximum = min(counts), max(counts)
    first = max(covering_index(max(0, -minimum), -candidates[0][0]), covering_index(max(0, maximum), candidates[0][-1]))
    if first > 254:
        raise ValueError("admitted input exceeds the largest legal shared scale")
    best_index, best_error = first, score(candidates[first], counts)
    evaluated, bounded_smaller = 1, 0
    for index in range(first-1, -1, -1):
        levels = candidates[index]
        left_error = max(0, levels[0]-minimum)
        right_error = max(0, maximum-levels[-1])
        lower_bound = counts[minimum]*left_error**2+counts[maximum]*right_error**2
        if lower_bound > best_error:
            bounded_smaller = index+1
            break
        error = score(levels, counts)
        evaluated += 1
        if error <= best_error:
            best_error, best_index = error, index
    if diagnostics is not None:
        diagnostics.update(fallback=False, legal_scales=255, evaluated_scales=evaluated,
                           larger_scales_excluded_by_nesting=254-first,
                           smaller_scales_excluded_by_clipping=bounded_smaller, first_covering_index=first)
    return {"schema_version": "2.0.0", "format": format_name, "scale": str(decimal(pow2(best_index-127))),
            "mse": str(decimal(Fraction(best_error, len(normalized)*GRID*GRID))), "candidate_count": 255,
            "policy": "intrinsic_or_required_only", "tie_break": "smallest_scale"}
