import os
from decimal import Decimal
from fractions import Fraction

import pytest

from public.inference.native import NativeBackend, build, int32_reduction_bound, prepare_tensor
from public.inference.reference.arithmetic import format_named, model_c, real
from public.inference.reference import operators as reference
from public.inference.operators.dispatch import Operators
from public.inference.tensor import Encoding, Tensor


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    name = os.environ.get("PRECISION_TEST_BACKEND", "cpp")
    return NativeBackend(name, build(name, tmp_path_factory.mktemp("wide_integer_"+name)))


@pytest.mark.parametrize("left,right", [
    ([127, -128, 10], [-128, 127, -1]),
    ([2**31-1, 2**31-1, -(2**31-1)], [1, 1, 1]),
    ([2**63-1, 1, -(2**63-1)], [1, 1, 1]),
    ([-2**63, -1, 2**63-1], [1, 1, 1]),
    ([Fraction(1, 2), Fraction(3, 2), -2], [1, 1, 1]),
])
def test_int64_preserves_model_c_order_rounding_and_saturation(backend, left, right):
    acc = format_named("int64_accumulator")
    result = backend.flex_gemm(left, right, batch=1, channels=1, k=len(left), accumulator=acc.name)
    assert result == (model_c(zip(map(real, left), map(real, right)), acc),)
    if int32_reduction_bound(left, right, len(left)):
        assert backend.last_strategy == "int64_proven_int32_reduction"
    else:
        assert backend.last_strategy.startswith("rational_grid_")


def test_proven_shortcut_handles_predecoded_negative_codes(backend):
    tensor = Tensor.quantize([-128, 127, 2], (1, 3), Encoding("int8"))
    values = prepare_tensor(tensor, unscaled=True)
    assert int32_reduction_bound(values, values, 3)
    assert backend.flex_gemm(values, values, batch=1, channels=1, k=3, accumulator="int64_accumulator") == (32517,)
    assert backend.last_strategy == "int64_proven_int32_reduction"


def test_width_proof_rejects_fractional_nonfinite_and_unsafe_prefixes():
    assert not int32_reduction_bound([Fraction(1, 2)], [1], 1)
    assert not int32_reduction_bound([Decimal("NaN")], [1], 1)
    assert not int32_reduction_bound([2**31-1, 1, -1], [1, 1, 1], 3)


def test_zero_product_does_not_bypass_native_operand_range(backend):
    assert backend.flex_gemm([2**100], [0], batch=1, channels=1, k=1, accumulator="int64_accumulator") == (0,)
    assert backend.last_strategy.startswith("rational_grid_")


def test_wide_bias_is_not_truncated_by_proven_narrow_reduction(backend):
    x = Tensor.quantize([0.001], (1, 1), Encoding("int8", ("0.001",)))
    w = Tensor.quantize([0.001], (1, 1), Encoding("int8", ("0.001",)))
    args = {"accumulator": "int64_accumulator", "output": Encoding("int8", ("30",)), "bias": ["3000"]}
    ops = Operators(backend.backend, backend.library._name)
    assert ops.linear(x, w, **args) == reference.linear(x, w, **args)
    assert ops.linear(x, w, **args).values() == (Fraction(3000),)
    assert reference.linear(x, w, **{**args, "accumulator": "int32_accumulator"}).codes != ops.linear(x, w, **args).codes


def test_convolution_uses_the_same_width_proof(backend):
    x = Tensor.quantize([-1, 2, 3, -4], (1, 1, 2, 2), Encoding("int8"))
    w = Tensor.quantize([-2], (1, 1, 1, 1), Encoding("int8"))
    args = {"accumulator": "int64_accumulator", "output": Encoding("int8")}
    ops = Operators(backend.backend, backend.library._name)
    assert ops.conv2d(x, w, **args) == reference.conv2d(x, w, **args)
    assert ops.native.last_strategy == "int64_proven_int32_reduction"
