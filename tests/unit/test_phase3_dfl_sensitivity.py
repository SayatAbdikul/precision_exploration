from fractions import Fraction

from public.analysis.phase3.dfl_sensitivity import vectors, check_vectors, wide_probabilities
from public.inference.reference.detector import dfl
from public.inference.tensor import Encoding, Tensor


def test_dfl_cases_cover_all_gaps_orders_and_reproduce_seed():
    cases = vectors(4, 4, random_cases=10)
    assert cases == vectors(4, 4, random_cases=10)
    assert len(cases) == len(set(cases))
    for gap in range(1, 16):
        for count in (1, 2, 3):
            row = (7,)*count+((7-gap) % 16,)*(4-count)
            assert row in cases and row[::-1] in cases


def test_wide_softmax_is_uniform_and_shift_invariant():
    source, output = Encoding("int8", (Fraction(1, 10),)), Encoding("int8", (Fraction(1, 127),))
    a = Tensor((1, 4), (0, 1, 2, 3), source)
    b = Tensor((1, 4), (10, 11, 12, 13), source)
    assert wide_probabilities(a, output, 400).codes == wide_probabilities(b, output, 800).codes
    uniform = Tensor((1, 4), (0,)*4, source)
    assert wide_probabilities(uniform, output, 400).codes == Tensor.quantize([Fraction(1, 4)]*4, uniform.shape, output).codes


def test_dfl_probe_exposes_integer_normalization_failure():
    source, output = Encoding("int8"), Encoding("int8", (Fraction(1, 127),))
    weights = Tensor((1, 4), (0, 1, 2, 3), source)
    failed = check_vectors([(0,)*4], source, output, weights, "int32_accumulator")
    assert failed["probability_code_mismatches"] == 4
    assert failed["status"] == "requires_diagnosis"
    passed = check_vectors([(0,)*4], source, output, weights, "fp64_e11m52_accumulator")
    assert passed["status"] == "bounded_cases_match"
    assert passed["wide_reference_unstable_vectors"] == 0
    # Check the probe's composed softmax/projection against the production DFL
    # entry point, including its side/bin layout and output store.
    cases = [(0, 1, 2, 3), (127, 128, 0, 1)]
    result = check_vectors(cases, source, output, weights, "fp64_e11m52_accumulator")
    tensor = Tensor((1, 16, 1, 2), tuple(row[bin_index] for _ in range(4) for bin_index in range(4) for row in cases), source)
    actual = dfl(tensor, weights, bins=4, accumulator="fp64_e11m52_accumulator", output=output)
    assert actual.codes == tuple(r["output_code"] for _ in range(4) for r in result["results"])
