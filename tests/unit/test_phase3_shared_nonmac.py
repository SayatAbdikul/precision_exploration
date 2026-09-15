from fractions import Fraction
from itertools import product

import pytest

from public.analysis.phase3.shared_nonmac import aligned_sum_proof
from public.inference.reference.arithmetic import format_named


@pytest.mark.parametrize("name", ["bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3"])
def test_common_scale_residual_proof_covers_every_e8m0_scale(name):
    proof = aligned_sum_proof(name)
    assert proof["status"] == "exact_output_codes"
    assert proof["scales_checked"] == 255


def test_all_small_shared_pairs_sum_exactly_at_extreme_common_scales():
    fmt = format_named("mxfp4_e2m1")
    values = [Fraction(v) for c in range(16) if (v := fmt.decode(c)).is_finite()]
    acc = format_named("fp64_e11m52_accumulator")
    for scale in (Fraction(1, 2**127), Fraction(2**127)):
        for a, b in product(values, repeat=2):
            exact = (a+b)*scale
            assert Fraction(acc.rounded(exact)) == exact


def test_independently_scaled_addition_does_not_have_the_common_scale_guarantee():
    acc = format_named("fp64_e11m52_accumulator")
    exact = Fraction(2**127)+Fraction(1, 2**127)
    assert Fraction(acc.rounded(exact)) != exact
