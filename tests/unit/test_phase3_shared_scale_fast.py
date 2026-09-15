from fractions import Fraction
import random

import pytest

from public.inference.reference.arithmetic import format_named
from public.quantization.calibration.mse import mse_scale
from tools.phase3.shared_scale_fast import admitted, mse_scale_fp32


@pytest.mark.parametrize("name", ["bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"])
def test_integer_grid_reproduces_entire_oracle_response_at_ties_edges_and_random_blocks(name):
    rng = random.Random(310913)
    fmt = format_named(name)
    levels = sorted({Fraction(v) for c in range(1 << fmt.bits) if (v := fmt.decode(c)).is_finite()})
    midpoints = [(a+b)/2 for a, b in zip(levels, levels[1:])]
    blocks = [[Fraction(0)]*32, [Fraction(1, 1 << 100), Fraction(-(1 << 100))],
              [Fraction(1)], [Fraction(-1)], midpoints[:32], midpoints[-32:]]
    for _ in range(8):
        blocks.append([Fraction(rng.randrange(-(1 << 23), 1 << 23), 1 << rng.randrange(15, 70)) for _ in range(32)])
    for block in blocks:
        assert all(admitted(v) for v in block)
        assert mse_scale_fp32(block, name) == mse_scale(block, name)


@pytest.mark.parametrize("value", [Fraction(1, 3), Fraction(1, 1 << 101), Fraction(1 << 101), Fraction((1 << 24)+1)])
def test_unsupported_inputs_fall_back_without_rounding_them(value):
    calls = []
    def fallback(values, name, **kwargs):
        calls.append(values)
        return {"sentinel": "oracle"}
    assert not admitted(value)
    assert mse_scale_fp32([value], "bfp6", fallback=fallback) == {"sentinel": "oracle"}
    assert calls == [(value,)]
