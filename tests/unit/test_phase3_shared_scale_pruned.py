from fractions import Fraction
from random import Random

import pytest

from public.quantization.calibration.mse import mse_scale
from tools.phase3.shared_scale_pruned import mse_scale_pruned


@pytest.mark.parametrize("name", ["bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3"])
def test_pruning_matches_oracle_on_asymmetric_ranges_ties_and_mixed_precision(name):
    rng = Random(31)
    cases = [[Fraction(0)]*32, [Fraction(-32), Fraction(31)], [Fraction(1)],
             [Fraction(2**52+1, 2**132), Fraction(1, 2**100), Fraction(2**100), -Fraction(2**53-1, 2**54)],
             [Fraction(rng.randrange(-1024, 1025), 128) for _ in range(32)],
             [Fraction(1, 2**80)]*31+[Fraction(2**80)]]
    for values in cases:
        details = {}
        actual = mse_scale_pruned(values, name, diagnostics=details)
        assert actual == mse_scale(values, name)
        assert not details["fallback"]
        assert (details["evaluated_scales"]+details["larger_scales_excluded_by_nesting"]+
                details["smaller_scales_excluded_by_clipping"]) == 255
        assert details["evaluated_scales"] < 255


def test_equal_errors_choose_the_smallest_scale_including_zero_plateau():
    result = mse_scale_pruned([Fraction(0)], "bfp6")
    assert Fraction(result["scale"]) == Fraction(1, 2**127)
    result = mse_scale_pruned([Fraction(1)], "bfp6")
    assert Fraction(result["scale"]) == 2  # BFP6 payload is fractional; 2, 4, ... store 1 exactly.


def test_unadmitted_values_keep_fallback_behavior():
    calls = []
    def fallback(values, name, **kwargs):
        calls.append((values, name))
        return {"fallback": True}
    details = {}
    value = Fraction(1, 3)
    assert mse_scale_pruned([value], "bfp6", diagnostics=details, fallback=fallback) == {"fallback": True}
    assert calls == [((value,), "bfp6")]
    assert details == {"fallback": True}
