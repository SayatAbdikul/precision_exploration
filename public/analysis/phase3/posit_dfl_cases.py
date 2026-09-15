"""Finite-code DFL cases for posit domains; never sample their NaR code."""
from fractions import Fraction
import random

from public.inference.reference.arithmetic import format_named


def selected_vectors(encoding, bins, *, seed=310915, random_cases=128):
    fmt = format_named(encoding.format)
    if fmt.family != "posit" or encoding.axis is not None or type(bins) is not int or bins < 2:
        raise ValueError("posit DFL cases require a scalar posit domain and at least two bins")
    finite = sorted((Fraction(v), c) for c in range(1 << fmt.bits) if (v := fmt.decode(c)).is_finite())
    codes = [c for _,c in finite]
    zero = next(c for value,c in finite if value == 0)
    cases = {(c,)*bins for c in codes}
    for code in codes:
        for count in sorted({1, bins//2, bins-1}):
            row = (zero,)*count+(code,)*(bins-count)
            cases.update((row, row[::-1]))
    for code in (codes[0], codes[-1], next(c for v,c in finite if v > 0)):
        for peak in range(bins):
            cases.add(tuple(code if i == peak else zero for i in range(bins)))
    rng = random.Random(seed)
    cases.update(tuple(rng.choice(codes) for _ in range(bins)) for _ in range(random_cases))
    return sorted(cases)
