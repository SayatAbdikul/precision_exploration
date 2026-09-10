"""Native conformance is runnable on a clean checkout without model payloads."""
import os
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from public.inference.native import NativeBackend, build
from public.inference.reference.arithmetic import format_named, model_c, real


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    name = os.environ.get("PRECISION_TEST_BACKEND", "cpp")
    directory = tmp_path_factory.mktemp(name)
    library = build(name, directory)
    return NativeBackend(name, library)


@pytest.mark.parametrize("name", ["fp6_e3m2", "int8", "posit6_es1", "mxfp4_e2m1", "ternary"])
def test_exhaustive_operand_pairs_model_c(backend, name):
    fmt = format_named(name)
    values = [real(fmt.decode(i)) for i in range(1 << fmt.bits)]
    acc_name = "fp16_e5m10_accumulator"
    acc = format_named(acc_name)
    outputs = backend.gemm(values, values, batch=len(values), channels=len(values), k=1, accumulator=acc_name)
    expected = tuple(model_c([(a, b)], acc) for a in values for b in values)
    assert outputs == expected


@pytest.mark.parametrize("k", [3, 9, 27, 64, 128, 256, 1024])
@pytest.mark.parametrize("acc_name", ["fp16_e5m10_accumulator", "fp32_e8m23_accumulator", "int32_accumulator"])
def test_dot_lengths_and_independent_outputs(backend, k, acc_name):
    rng = random.Random(240 + k)
    fmt = format_named("int8" if acc_name == "int32_accumulator" else "fp6_e3m2")
    finite = [real(fmt.decode(i)) for i in range(1 << fmt.bits) if fmt.decode(i).is_finite()]
    left = [rng.choice(finite) for _ in range(2 * k)]
    right = [rng.choice(finite) for _ in range(3 * k)]
    actual = backend.gemm(left, right, batch=2, channels=3, k=k, accumulator=acc_name)
    acc = format_named(acc_name)
    expected = tuple(model_c(zip(left[n*k:(n+1)*k], right[c*k:(c+1)*k]), acc) for n in range(2) for c in range(3))
    assert actual == expected


def test_native_rejects_uncertified_domains(backend):
    with pytest.raises(ValueError, match="dyadic"):
        backend.gemm([Fraction(1, 3)], [1], batch=1, channels=1, k=1, accumulator="fp16_e5m10_accumulator")
    with pytest.raises(ValueError, match="INT32"):
        backend.gemm([Decimal("NaN")], [1], batch=1, channels=1, k=1, accumulator="int32_accumulator")


def test_specials_and_negative_zero(backend):
    left = [Decimal("-0"), Decimal("Infinity"), Decimal("-Infinity"), Decimal("NaN"), 1]
    right = [0, 1, -1, Decimal("Infinity")]
    acc = format_named("fp32_e8m23_accumulator")
    expected = tuple(model_c([(real(a), real(b))], acc) for a in left for b in right)
    assert backend.gemm(left, right, batch=5, channels=4, k=1, accumulator=acc.name) == expected


@pytest.mark.parametrize("groups,weights_shape", [(1, (3, 2, 3, 3)), (2, (4, 1, 3, 3))])
def test_full_convolution_and_dedicated_depthwise(backend, groups, weights_shape):
    from public.inference.operators.dispatch import Operators
    from public.inference.tensor import Tensor, Encoding
    from public.inference.reference import operators as ref
    from math import prod
    rng = random.Random(991)
    encoding = Encoding("fp6_e3m2")
    x = Tensor.quantize([rng.choice([-2, -1, 0, 1, 2]) for _ in range(2*2*5*6)], (2, 2, 5, 6), encoding)
    w = Tensor.quantize([rng.choice([-1, 0, 1]) for _ in range(prod(weights_shape))], weights_shape, encoding)
    options = dict(groups=groups, padding=1, stride=(2, 1), accumulator="fp16_e5m10_accumulator",
                   output=encoding, bias=[Fraction(1, 2)]*weights_shape[0], activation="relu6")
    expected = ref.conv2d(x, w, **options)
    dispatch = Operators()
    dispatch.backend, dispatch.native = backend.backend, backend
    assert dispatch.conv2d(x, w, **options) == expected


def test_int8_scaled_linear_and_native_model_c_witness(backend):
    from public.inference.operators.dispatch import Operators
    from public.inference.tensor import Tensor, Encoding
    from public.inference.reference import operators as ref
    x = Tensor.quantize([1, 2, 3, 4], (2, 2), Encoding("int8", scales=("0.1",)))
    w = Tensor.quantize([1, 2, 2, 1], (2, 2), Encoding("int8", scales=("0.3", "0.7"), axis=0))
    options = dict(accumulator="int32_accumulator", output=Encoding("fp6_e3m2"), bias=[-1, 1])
    dispatch = Operators()
    dispatch.backend, dispatch.native = backend.backend, backend
    assert dispatch.linear(x, w, **options) == ref.linear(x, w, **options)
    assert backend.gemm([Fraction(5,4), Fraction(-3,2)], [Fraction(5,4), 1], batch=1, channels=1, k=2,
                        accumulator="fp16_e5m10_accumulator") == (format_named("fp16_e5m10_accumulator").encode(Fraction(1,16)),)


@pytest.mark.parametrize("name", ["fp6_e3m2", "int8"])
def test_algorithmic_lookup_and_predecoded_are_bit_identical(backend, name):
    fmt = format_named(name)
    codes = list(range(1 << fmt.bits))
    values = [real(fmt.decode(code)) for code in codes]
    args = dict(batch=len(codes), channels=len(codes), k=1, accumulator="fp16_e5m10_accumulator")
    expected = backend.gemm(values, values, **args)
    for strategy in ("algorithmic", "lookup"):
        actual = backend.encoded_gemm(codes, codes, activation_format=name, weight_format=name, strategy=strategy, **args)
        assert actual == expected


def test_weight_activation_formats_are_independent(backend):
    from public.inference.tensor import Encoding, Tensor
    from public.inference.operators.dispatch import Operators
    from public.inference.reference import operators as ref
    inputs = Tensor.quantize([1.25,-0.5,2.5,0.125], (2,2), Encoding("fp6_e3m2"))
    weights = Tensor.quantize([1.5,-1,2,0.5], (2,2), Encoding("fp8_e4m3fn"))
    args = dict(accumulator="fp32_e8m23_accumulator",output=Encoding("fp5_e2m2"),bias=[0.125,-0.25])
    dispatch=Operators()
    dispatch.backend,dispatch.native=backend.backend,backend
    assert dispatch.linear(inputs,weights,**args) == ref.linear(inputs,weights,**args)


def test_native_integer_saturation_and_floating_rounding_order(backend):
    acc=format_named("int32_accumulator")
    left=[2**30,2**30,-2**30]
    assert backend.gemm(left,[1,1,1],batch=1,channels=1,k=3,accumulator=acc.name) == (model_c([(real(a),real(1)) for a in left],acc),)
    acc=format_named("fp16_e5m10_accumulator")
    left=[2048,1,-2048]
    assert backend.gemm(left,[1,1,1],batch=1,channels=1,k=3,accumulator=acc.name) == (acc.encode(0),)


def test_shared_patch_convolution_matches_reference(backend):
    from public.inference.tensor import Tensor, Encoding, SharedEncoding
    from public.inference.reference.blocks import block_conv2d
    inputs = Tensor.quantize([1]*32+[Fraction(1,16)],(1,33,1,1),Encoding("fp6_e3m2"))
    weights = Tensor.quantize([1]+[0]*31+[1],(1,33),SharedEncoding("mxfp4_e2m1",axis=1))
    args = dict(kernel_size=1,patch_format="mxfp4_e2m1",accumulator="fp32_e8m23_accumulator",output=Encoding("fp6_e3m2"),bias=[Fraction(1,4)])
    assert block_conv2d(inputs,weights,native=backend,**args) == block_conv2d(inputs,weights,**args)
