"""Rational reference accumulators (exact Python integers and Fractions).

Nothing here shares code with the native kernels. Rounding to binary64,
binary32, binary16 and the 21-bit float is done by the project's manifest
driven reference encoder (public/inference/reference/arithmetic.py).
"""
from __future__ import annotations
from decimal import Decimal
from fractions import Fraction as Q
from functools import lru_cache
import json
from public.inference.reference.arithmetic import Accumulator, format_named, model_c
from .accumulators import FLOATS, resolve


@lru_cache(None)
def float_reference(name):
    return Accumulator(json.loads(FLOATS[name]['manifest'].read_text()))


def saturating(pairs, width):
    """(value in grid units, high clamps, low clamps) of the W-bit saturating accumulator."""
    hi = (1 << (width - 1)) - 1; lo = -hi - 1; total = up = down = 0
    for a, b in pairs:
        total += a * b
        if total > hi:
            total = hi; up += 1
        elif total < lo:
            total = lo; down += 1
    return total, up, down


def floating(pairs, shift, name, exponent):
    """(code-level value as a float or +-inf, number of steps that end infinite)."""
    fmt = float_reference(name); unit = Q(2) ** (exponent - shift)
    state = Q(0); infinite = 0; bad = 0
    for a, b in pairs:
        if infinite:
            bad += 1; continue
        value = fmt.decode(fmt.encode(state + a * b * unit))
        if isinstance(value, Decimal):
            if value.is_nan():
                raise ValueError('NaN is unreachable in an accumulation of finite products')
            if value.is_infinite():
                infinite = -1 if value.is_signed() else 1; bad += 1; continue
            value = Q(0)
        state = value
    if infinite:
        return infinite * float('inf'), bad
    return float(state / Q(2) ** exponent), bad


def expected_dot(pairs, shift_a, shift_w, policy, certificate=None):
    """(binary64 code-level dot, event word) a kernel must return for these (a, w) grid-integer pairs."""
    policy = resolve(policy); shift = shift_a + shift_w
    if policy.kind == 'wide':
        return float(format_named('fp64_e11m52_accumulator').rounded(Q(sum(a * b for a, b in pairs), 1 << shift))), None
    if policy.kind == 'control':
        fmt = format_named('fp32_e8m23_accumulator')
        return float(fmt.decode(model_c(((Q(a, 1 << shift_a), Q(b, 1 << shift_w)) for a, b in pairs), fmt))), None
    if policy.kind == 'sat':
        total, up, down = saturating(pairs, policy.width(certificate))
        value = float(format_named('fp64_e11m52_accumulator').rounded(Q(total, 1 << shift)))
        return value, min(up, 65535) | (min(down, 65535) << 16)
    value, bad = floating(pairs, shift, policy.float_format, policy.scale_exponent)
    return value, bad
