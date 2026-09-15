"""Deterministic DFL sensitivity cases, separate from workload acceptance."""
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import random

from public.inference.reference.arithmetic import format_named, model_c
from public.inference.reference.detector import softmax
from public.inference.tensor import Tensor


def vectors(bits, bins, *, seed=310914, random_cases=128):
    if type(bits) is not int or not 2 <= bits <= 8 or type(bins) is not int or bins < 2:
        raise ValueError("bounded DFL probe needs 2–8 bit logits and at least two bins")
    low, high = -(1 << (bits-1)), (1 << (bits-1))-1
    cases = {(value,)*bins for value in (low, 0, high)}
    for gap in range(1, high-low+1):
        for count in sorted({1, bins//2, bins-1}):
            row = (high,)*count+(high-gap,)*(bins-count)
            cases.update((row, row[::-1]))
    for gap in sorted({1, high, high-low}):
        for index in range(bins):
            cases.add(tuple(high if i == index else high-gap for i in range(bins)))
    rng = random.Random(seed)
    cases.update(tuple(rng.randint(low, high) for _ in range(bins)) for _ in range(random_cases))
    return [tuple(value % (1 << bits) for value in row) for row in sorted(cases)]


@lru_cache(maxsize=4096)
def exponential(value, precision):
    with localcontext() as context:
        context.prec = precision
        return Fraction((Decimal(value.numerator)/Decimal(value.denominator)).exp())


def wide_probabilities(inputs, output, precision):
    values = inputs.values()
    maximum = max(values)
    exp = [exponential(value-maximum, precision) for value in values]
    total = sum(exp, Fraction(0))
    return Tensor.quantize([value/total for value in exp], inputs.shape, output)


def check_vectors(cases, source, output, weights, accumulator):
    bins = len(weights.codes)
    acc = format_named(accumulator)
    weight_values = weights.values()
    probability_mismatches = projection_mismatches = precision_unstable = 0
    examples, results = [], []
    for index, codes in enumerate(cases):
        if len(codes) != bins:
            raise ValueError("DFL probe vector/projection length mismatch")
        inputs = Tensor((1, bins), tuple(codes), source)
        probabilities = softmax(inputs, axis=1, accumulator=accumulator, output=output)
        wide = wide_probabilities(inputs, output, 400)
        wider = wide_probabilities(inputs, output, 800)
        stable = wide.codes == wider.codes
        actual_value = acc.decode(model_c(zip(probabilities.values(), weight_values), acc))
        actual = Tensor.quantize([actual_value], (1,), output).codes[0]
        # Isolate projection rounding on identical stored probabilities.
        exact_projection = Tensor.quantize([sum((p*w for p, w in zip(probabilities.values(), weight_values)), Fraction(0))], (1,), output).codes[0]
        wide_projection = Tensor.quantize([sum((p*w for p, w in zip(wide.values(), weight_values)), Fraction(0))], (1,), output).codes[0]
        prob_diff = sum(a != b for a, b in zip(probabilities.codes, wide.codes))
        probability_mismatches += prob_diff
        projection_mismatches += actual != exact_projection
        precision_unstable += not stable
        result = {"probability_codes": list(probabilities.codes), "wide_probability_codes": list(wide.codes),
                  "wide_precision_stable": stable, "output_code": actual,
                  "exact_projection_same_probabilities": exact_projection, "wide_output_code": wide_projection}
        results.append(result)
        if (prob_diff or actual != exact_projection or not stable) and len(examples) < 8:
            examples.append({"case": index, "input_codes": list(codes), **result})
    return {"vectors": len(cases), "probability_codes_tested": len(cases)*bins,
            "probability_code_mismatches": probability_mismatches,
            "projection_output_mismatches_on_same_probabilities": projection_mismatches,
            "end_to_end_output_mismatches": sum(r["output_code"] != r["wide_output_code"] for r in results),
            "wide_reference_unstable_vectors": precision_unstable, "examples": examples, "results": results,
            "status": "bounded_cases_match" if not (probability_mismatches or projection_mismatches or precision_unstable)
                      else "requires_diagnosis"}
