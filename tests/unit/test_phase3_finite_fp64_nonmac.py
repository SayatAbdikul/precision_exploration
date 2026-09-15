from fractions import Fraction
from itertools import product

from public.analysis.phase3.finite_fp64_nonmac import exact_grid, hard_check, mean_check, residual_check
from public.analysis.phase3.fixed_nonmac import finite_values
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, Tensor


def test_exact_grid_rejects_non_dyadic_and_overwide_lattices():
    assert exact_grid(Fraction(3, 8), 12)
    assert exact_grid(Fraction(1, 2**1074), Fraction(1, 2**1022))
    assert not exact_grid(Fraction(1, 3), 1)
    assert not exact_grid(Fraction(1, 2**1075), 1)
    assert not exact_grid(Fraction(1, 8), 2**53)


def test_global_mean_proof_preserves_all_small_fp4_cases_including_ties():
    encoding = Encoding("fp4_e2m1")
    values = finite_values(encoding)
    acc = format_named("fp64_e11m52_accumulator")
    for count in (2, 3):
        proof = mean_check(encoding, encoding, count)
        assert proof["status"] == "exact_output_codes"
        exact, actual = [], []
        for sequence in product(values, repeat=count):
            total = Fraction(0)
            for value in sequence:
                total = Fraction(acc.rounded(total+value))
            exact.append(sum(sequence)/count)
            actual.append(Fraction(acc.rounded(total/count)))
        assert Tensor.quantize(actual, (len(actual),), encoding).codes == Tensor.quantize(exact, (len(exact),), encoding).codes
    assert mean_check(encoding, encoding, 2)["exact_ties"] > 0


def test_residual_checks_cover_non_dyadic_codebook_and_dyadic_lattice():
    assert residual_check(Encoding("fp8_e5m2"), Encoding("fp8_e5m2"))["status"] == "exact_output_codes"
    result = residual_check(Encoding("nf4"), Encoding("nf4"))
    assert result["pairs"] == 136
    assert result["status"] == "exact_output_codes"


def test_non_dyadic_global_mean_remains_pending():
    result = mean_check(Encoding("nf4"), Encoding("nf4"), 49)
    assert result["status"] == "pending"


def test_hard_swish_reference_retains_negative_zero():
    result = hard_check(Encoding("fp4_e2m1"), Encoding("fp4_e2m1"), "hard_swish")
    assert result["finite_codes_tested"] == 14  # Includes both signed zero codes.
    assert result["mismatches"] == 0


def test_logarithmic_residual_boundary_discrepancy_is_retained():
    result = residual_check(Encoding("log6"), Encoding("log6"))
    assert result["status"] == "pending"
    assert result["mismatches"] > 0
    example = result["examples"][0]
    assert example["exact_output_code"] != example["fp64_output_code"]
    assert Fraction(example["left"])+Fraction(example["right"]) == Fraction(example["exact_sum"])
