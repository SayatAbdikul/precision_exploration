"""Conservative output-boundary proof for prescribed fixed-grid bias storage."""
from fractions import Fraction
from functools import lru_cache
from math import lcm

from public.inference.reference.arithmetic import format_named, round_integer


@lru_cache(maxsize=8)
def output_boundaries(name):
    fmt = format_named(name)
    if fmt.family != "posit":
        raise ValueError("fixed bias boundary proof requires a scalar posit output")
    values = sorted((Fraction(v), c) for c in range(1 << fmt.bits) if (v := fmt.decode(c)).is_finite())
    rows = []
    for (a, ca), (b, cb) in zip(values, values[1:]):
        if a == b:
            raise ValueError("duplicate finite values need a separate tie proof")
        preferred = min((ca, cb), key=lambda c: (c & 1, c))
        rows.append(((a+b)/2, ca, cb, -1 if preferred == ca else 1))
    return tuple(rows)


@lru_cache(maxsize=512)
def integer_boundaries(name, denominator):
    return tuple((int(t*denominator), a, b, tie) for t, a, b, tie in output_boundaries(name))


def bias_boundary_check(dot_quantum, maximum_dot, bias, accumulator_step, output_name):
    q, maximum, bias, step = map(Fraction, (dot_quantum, maximum_dot, bias, accumulator_step))
    if q < 0 or maximum < 0 or step <= 0:
        raise ValueError("invalid fixed bias domain")
    stored = round_integer(bias/step)*step
    error = abs(stored-bias)
    if error == 0:
        return {"status": "exact_output_codes", "bias_storage_error": "0", "method": "bias is already exactly stored"}
    fmt = format_named(output_name)
    if fmt.family != "posit":
        raise ValueError("fixed bias boundary proof requires posit output")
    if q == 0:
        from public.inference.reference.arithmetic import encode
        equal = encode(fmt, bias) == encode(fmt, stored)
        return {"status": "exact_output_codes" if equal else "pending", "bias_storage_error": str(error),
                "method": "constant zero-weight channel", "constant_outputs_equal": equal}
    if (q/step).denominator != 1 or (maximum/q).denominator != 1:
        raise ValueError("fixed bias proof requires an exact product-sum grid and lattice-aligned bound")
    boundaries = output_boundaries(output_name)
    denominator = lcm(q.denominator, bias.denominator, stored.denominator, *(t.denominator for t, *_ in boundaries))
    Q, B, S = int(q*denominator), int(bias*denominator), int(stored*denominator)
    limit = int(maximum/q)
    mismatches, examples = 0, []
    for threshold, left_code, right_code, tie_side in integer_boundaries(output_name, denominator):
        floor = (threshold-B)//Q
        for n in {max(-limit, min(limit, floor)), max(-limit, min(limit, floor+1))}:
            original, candidate = n*Q+B, n*Q+S
            a = -1 if original < threshold else 1 if original > threshold else tie_side
            b = -1 if candidate < threshold else 1 if candidate > threshold else tie_side
            if a != b:
                mismatches += 1
                if len(examples) < 4:
                    examples.append({"sum_lattice_index": n, "exact_sum_plus_bias": str(Fraction(original, denominator)),
                                     "sum_plus_stored_bias": str(Fraction(candidate, denominator)),
                                     "threshold": str(Fraction(threshold, denominator)), "left_code": left_code,
                                     "right_code": right_code, "tie_side": "left" if tie_side < 0 else "right"})
    return {"status": "exact_output_codes" if not mismatches else "pending", "bias_storage_error": str(error),
            "dot_quantum": str(q), "maximum_absolute_dot": str(maximum), "boundaries": len(boundaries),
            "possible_boundary_discrepancies": mismatches, "examples": examples,
            "method": "all threshold-adjacent integer sums in a conservative product-sum lattice",
            "limit": "a lattice counterexample may not be reachable by an actual activation-code vector"}
