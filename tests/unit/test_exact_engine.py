from decimal import Decimal, localcontext
from fractions import Fraction
import random
import struct

import pytest

from public.formats.oracle import NumberFormat
from public.inference.reference.arithmetic import Accumulator, format_named, model_c, real, pow2
from public.inference.tensor import Encoding, Tensor
from public.inference.reference import operators as ops

FP16 = "fp16_e5m10_accumulator"
FP32 = "fp32_e8m23_accumulator"


def tensor(values, shape, name="fp6_e3m2", **kwargs):
    return Tensor.quantize(values, shape, Encoding(name, **kwargs))


def test_wide_accumulator_constant_memory_and_integer_bounds():
    acc = format_named("int32_accumulator")
    assert not hasattr(acc, "_values")
    for value in (-2**31, -1, 0, 1, 2**31 - 1):
        assert acc.decode(acc.encode(value)) == value
    assert acc.decode(acc.encode(2**40)) == 2**31 - 1
    assert acc.decode(acc.encode(-2**40)) == -2**31
    assert acc.decode(acc.encode(Fraction(7, 8), scale=Fraction(1, 4)), scale=Fraction(1, 4)) == 1
    with pytest.raises(ValueError, match="NaN"):
        acc.encode(Decimal("NaN"))


def test_fp16_all_encodings_agree_with_existing_oracle():
    acc = format_named(FP16)
    oracle = NumberFormat(acc.manifest)
    for code in range(65536):
        value = oracle.decode(code)
        actual = acc.decode(code)
        if value.is_nan():
            assert actual.is_nan()
        else:
            assert actual == value
            assert acc.encode(actual) == oracle.encode(value)


def test_fp32_rounding_against_struct_and_adversarial_ties():
    acc = format_named(FP32)
    rng = random.Random(183)
    for _ in range(1000):
        value = rng.uniform(-1e30, 1e30)
        expected = struct.unpack("<I", struct.pack("<f", value))[0]
        assert acc.encode(value) == expected
    assert acc.encode(1 + pow2(-24)) == acc.encode(1)
    assert acc.encode(1 + pow2(-24) + pow2(-100)) == acc.encode(1 + pow2(-23))
    assert acc.encode(pow2(-150)) == 0
    assert acc.encode(pow2(-150) + pow2(-200)) == 1
    assert acc.encode(Decimal("-0")) == 0x80000000


def test_model_c_retains_product_and_rounds_each_step_in_order():
    acc = format_named(FP16)
    # 1.25^2 is not representable in FP6 E3M2; rounding the product gives 1.5.
    state = model_c([(real("1.25"), real("1.25")), (real("-1.5"), real(1))], acc)
    assert acc.decode(state) == Fraction(1, 16)
    forward = [(real(2048), real(1)), (real(1), real(1)), (real(-2048), real(1))]
    assert acc.decode(model_c(forward, acc)) == 0
    assert acc.decode(model_c([forward[0], forward[2], forward[1]], acc)) == 1
    assert acc.decode(model_c([], acc, bias="0.5")) == Fraction(1, 2)


def test_bias_is_quantized_before_it_is_added_and_decimal_context_is_irrelevant():
    acc = format_named(FP16)
    pairs = [(real(1), real(1))]
    expected = acc.encode(acc.decode(acc.encode(1)) + acc.decode(acc.encode("0.0004885196685791015625")))
    with localcontext() as context:
        context.prec = 3
        assert model_c(pairs, acc, bias="0.0004885196685791015625") == expected


def test_scale_metadata_validation_and_partial_shared_blocks():
    with pytest.raises(ValueError, match="external"):
        Encoding("fp6_e3m2", scales=("2",))
    encoding = Encoding("mxfp6_e3m2", scales=("1", "2", "4", "8"), axis=1, block_size=32)
    value = Tensor.quantize([1] * 66, (2, 33), encoding)
    assert value.encoding.scale_at(value.shape, (0, 32)) == 2
    assert value.encoding.scale_at(value.shape, (1, 0)) == 4
    assert Tensor.from_document(value.document()) == value
    with pytest.raises(ValueError, match="scale count"):
        Tensor.quantize([1] * 33, (1, 33), Encoding("mxfp6_e3m2", axis=1, block_size=32))


def test_int8_per_channel_bias_domain_and_independent_output_format():
    x = tensor([1, 2], (1, 2), "int8", scales=("0.5",))
    w = tensor([1, 2, 2, 1], (2, 2), "int8", scales=("0.25", "0.5"), axis=0)
    result = ops.linear(x, w, accumulator="int32_accumulator", output=Encoding("fp6_e3m2"), bias=[-1, 1])
    assert result.values() == (4, 5)


def test_binary_padding_is_true_zero_and_depthwise_does_not_mix_channels():
    x = tensor([1], (1, 1, 1, 1), "binary_pm1")
    w = tensor([1] * 9, (1, 1, 3, 3), "binary_pm1")
    result = ops.conv2d(x, w, padding=1, accumulator=FP16, output=Encoding("fp6_e3m2"))
    assert result.values() == (1,)
    x = tensor([1, 2, 3, 4, 5, 6, 7, 8], (1, 2, 2, 2))
    w = tensor([1, 2], (2, 1, 1, 1))
    result = ops.depthwise_conv2d(x, w, accumulator=FP16, output=Encoding("fp8_e4m3fn"))
    assert result.values() == (1, 2, 3, 4, 10, 12, 14, 16)


def test_residual_alignment_pooling_and_nonlinear_semantics():
    x = tensor([1, 2, 3, 4], (1, 1, 2, 2))
    aligned = Encoding("int8", scales=("0.5",))
    result = ops.elementwise(x, x, operation="add", accumulator=FP16, alignment=aligned, output=x.encoding)
    assert result.values() == (2, 4, 6, 8)
    assert ops.pool2d(x, kind="average", kernel_size=2, accumulator=FP16, output=x.encoding).values() == (Fraction(5, 2),)
    assert ops.pool2d(x, kind="max", kernel_size=2, output=x.encoding).values() == (4,)
    values = tensor([-4, -3, 0, 3, 4], (5,))
    assert ops.activation(values, function="hard_swish", accumulator=FP16, output=values.encoding).values() == (0, 0, 0, 3, 4)
    lut = ops.nonlinear_lut(x.encoding, x.encoding, "sigmoid")
    zero = tensor([0], (1,))
    assert ops.apply_lut(zero, table=lut, output=zero.encoding).values() == (Fraction(1, 2),)


def test_integer_average_pool_keeps_fractional_requantization():
    x = tensor([1,2], (1,1,1,2), "int8")
    output = Encoding("int8", scales=("0.5",))
    result = ops.pool2d(x, kind="average", kernel_size=(1,2), accumulator="int32_accumulator", output=output)
    assert result.values() == (Fraction(3,2),)


def test_all_accepted_families_have_reference_dot_coverage():
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[2]
    rows = json.loads((root / "public/formats/manifests/accepted/index.json").read_text())["manifests"]
    acc = format_named("fp64_e11m52_accumulator")
    for row in rows:
        fmt = format_named(row["name"])
        value = real(fmt.decode(fmt.encode(1)))
        assert acc.decode(model_c([(value, value)], acc)) == acc.rounded(value * value)
def test_unused_special_codes_do_not_break_finite_activation_or_pool():
    from public.inference.tensor import Encoding,Tensor
    from public.inference.reference.operators import activation,pool2d
    inputs = Tensor.quantize([1],(1,1,1,1),Encoding("fp8_e5m2"))
    output = Encoding("int4")
    assert activation(inputs,function="relu",output=output).values() == (1,)
    assert pool2d(inputs,kind="max",kernel_size=1,output=output).values() == (1,)
