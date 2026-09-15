from fractions import Fraction

import pytest

from public.quantization.calibration.mse import mse_scale
from tools.phase3.shared_scale_dyadic import admitted, mse_scale_dyadic


@pytest.mark.parametrize("name", ["bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3"])
def test_53_bit_search_matches_complete_oracle_response_at_domain_edges(name):
    values = [Fraction(0), Fraction(2**52+1, 2**53), -Fraction(2**52+3, 2**132),
              Fraction(2**52+1, 2**52)*2**79, Fraction(1, 2**80)]
    assert all(admitted(v) for v in values)
    assert mse_scale_dyadic(values, name) == mse_scale(values, name)


def test_wider_significands_and_out_of_range_values_delegate_without_coercion():
    calls = []
    def fallback(values, name, **options):
        calls.append((values, name, options))
        return {"delegated": True}
    for value in (Fraction(2**53+1, 2**54), Fraction(1, 2**81), Fraction(2**81), Fraction(1, 3)):
        assert not admitted(value)
        assert mse_scale_dyadic([value], "bfp6", fallback=fallback) == {"delegated": True}
        assert calls[-1][0] == (value,)


def test_zero_block_retains_smallest_scale_tie_rule():
    result = mse_scale_dyadic([Fraction(0)]*32, "bfp6")
    assert result == mse_scale([Fraction(0)]*32, "bfp6")
    assert Fraction(result["scale"]) == Fraction(1, 2**127)
