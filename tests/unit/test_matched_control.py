from fractions import Fraction

import numpy as np
import pytest

from public.inference.reference import operators as ref
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, Tensor
from tools.breadth_study.matched_control import FP32, FP32Native, fp32_sum, pool, residual


def test_control_detects_cancellation_that_exact_sum_preserves():
    assert sum((Fraction(2**24), Fraction(1), Fraction(-2**24))) == 1
    assert fp32_sum((Fraction(2**24), Fraction(1), Fraction(-2**24))) == 0
    assert fp32_sum((Fraction(2**24, 3), Fraction(1, 3), Fraction(-2**24, 3)), Fraction(1, 3)) == 0


@pytest.mark.parametrize('name', ['int32_accumulator', 'int64_accumulator', 'fp64_e11m52_accumulator'])
def test_native_state_interface_matches_scalar_oracle_at_boundaries(name):
    fp, original = format_named(FP32), format_named(name)
    values = [0, -0.0, 1, -1, 2**24, -2**31, 2**31, -(2**63), 2**63, Fraction(1, 2)]
    states = tuple(fp.encode(value) for value in values)
    assert FP32Native.cast_states(states, name) == tuple(original.encode(fp.decode(s)) for s in states)


def test_fp32_integer_state_fast_path_preserves_signed_codes():
    states = tuple(np.array([-2**31, -17, 0, 19, 2**24], dtype=np.float32).view(np.uint32).tolist())
    for name in ('int32_accumulator', 'int64_accumulator'):
        fp, original = format_named(FP32), format_named(name)
        assert FP32Native.cast_states(states, name) == tuple(original.encode(fp.decode(s)) for s in states)


def test_integer_pool_keeps_exact_scale_and_fractional_division():
    inputs = Tensor((1, 1, 2, 3), (1, 2, 3, 4, 5, 6), Encoding('int4', (Fraction('1/3'),)))
    attrs = dict(kind='average', kernel_size=(2, 3), accumulator='int32_accumulator', output=Encoding('int4', (Fraction('1/6'),)))
    assert pool(inputs, **attrs) == ref.pool2d(inputs, **attrs)


def test_residual_keeps_alignment_quantizer_and_output_rounding():
    left = Tensor((1, 16), tuple(range(16)), Encoding('int4', (Fraction('1/3'),)))
    right = Tensor(left.shape, left.codes[::-1], Encoding('int4', (Fraction('2/7'),)))
    attrs = dict(operation='add', output=Encoding('int4', (Fraction('1/5'),)), accumulator='int32_accumulator',
                 alignment=Encoding('int4', (Fraction('1/7'),)))
    assert residual(left, right, **attrs) == ref.elementwise(left, right, **attrs)
