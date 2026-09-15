from fractions import Fraction

from public.analysis.phase3.fixed_bias_sensitivity import bias_boundary_check
from public.inference.reference.arithmetic import encode, format_named, round_integer


def test_bias_boundary_proof_matches_exhaustive_small_sum_lattices_and_ties():
    fmt = format_named("posit4_es0")
    q, step = Fraction(1, 16), Fraction(1, 64)
    for bias in (Fraction(0), Fraction(1, 64), Fraction(1, 1000), Fraction(-1, 1000), Fraction(1, 7), Fraction(-1, 7)):
        stored = round_integer(bias/step)*step
        for limit in (0, 1, 5, 32):
            actual = all(encode(fmt, n*q+bias) == encode(fmt, n*q+stored) for n in range(-limit, limit+1))
            proof = bias_boundary_check(q, limit*q, bias, step, fmt.name)
            assert (proof["status"] == "exact_output_codes") == actual


def test_zero_weight_channel_is_checked_as_a_constant_output():
    assert bias_boundary_check(0, 0, Fraction(1, 1000), Fraction(1, 64), "posit4_es0")["status"] == "exact_output_codes"


def test_tiny_bias_can_change_a_threshold_tie_on_the_conservative_lattice():
    result = bias_boundary_check(Fraction(1, 16), 1, Fraction(1, 1000), Fraction(1, 64), "posit4_es0")
    assert result["status"] == "pending"
    assert result["possible_boundary_discrepancies"] > 0
