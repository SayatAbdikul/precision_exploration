import os
from decimal import Decimal
from fractions import Fraction
import random
import json
from pathlib import Path

import pytest

from public.inference.native import NativeBackend, build
from public.inference.reference.arithmetic import format_named, model_c, real, pow2

ACCEPTED = json.loads((Path(__file__).resolve().parents[2]/"public/formats/manifests/accepted/index.json").read_text())["manifests"]


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    name = os.environ.get("PRECISION_TEST_BACKEND","cpp")
    return NativeBackend(name,build(name,tmp_path_factory.mktemp("rational_"+name)))


@pytest.mark.parametrize("name",[row["name"] for row in ACCEPTED])
def test_all_rational_operand_pairs(backend,name):
    fmt = format_named(name)
    values = [real(fmt.decode(code)) for code in range(1<<fmt.bits)]
    acc = format_named("fp32_e8m23_accumulator")
    actual = backend.rational_gemm(values,values,batch=len(values),channels=len(values),k=1,accumulator=acc.name)
    assert actual == tuple(model_c([(a,b)],acc) for a in values for b in values)


@pytest.mark.parametrize("acc_name",["fp16_e5m10_accumulator","fp32_e8m23_accumulator","fp64_e11m52_accumulator","int32_accumulator","posit8_es1_quire64_accumulator"])
def test_rational_reductions_and_wide_accumulators(backend,acc_name):
    rng = random.Random(991)
    fmt = format_named("nf4")
    choices = [real(fmt.decode(c)) for c in range(16)]
    left = [rng.choice(choices) for _ in range(2*33)]
    right = [rng.choice(choices) for _ in range(3*33)]
    acc = format_named(acc_name)
    actual = backend.rational_gemm(left,right,batch=2,channels=3,k=33,accumulator=acc_name)
    assert actual == tuple(model_c(zip(left[n*33:(n+1)*33],right[c*33:(c+1)*33]),acc) for n in range(2) for c in range(3))


def test_rational_extreme_binary_exponents_specials_and_ties(backend):
    acc = format_named("fp64_e11m52_accumulator")
    left = [pow2(-1074),pow2(-1075),pow2(-1075)+pow2(-1200),pow2(1023),Decimal("Infinity"),Decimal("-0")]
    right = [1,2,Decimal("-Infinity"),0]
    actual = backend.rational_gemm(left,right,batch=len(left),channels=len(right),k=1,accumulator=acc.name)
    assert actual == tuple(model_c([(real(a),real(b))],acc) for a in left for b in right)


def test_rational_mapping_scales_and_1024_steps(backend):
    acc = format_named("fp32_e8m23_accumulator")
    a = Fraction(1,3)
    left,right = [a]*1024,[a,-a]*512
    actual = backend.rational_gemm(left,right,batch=1,channels=1,k=1024,accumulator=acc.name)
    assert actual == (model_c(zip(left,right),acc),)
    # This uses the largest template and the precision retained by calibration
    # scale serialization; no decimal-to-binary approximation is admitted.
    tiny = Fraction(10**1100+1,10**1100)
    actual = backend.rational_gemm([tiny],[tiny],batch=1,channels=1,k=1,accumulator=acc.name)
    assert actual == (model_c([(tiny,tiny)],acc),)


@pytest.mark.parametrize("name",[row["name"] for row in ACCEPTED])
def test_every_accepted_format_across_reduction_lengths(backend,name):
    rng = random.Random(812)
    fmt = format_named(name)
    choices = [real(fmt.decode(c)) for c in range(1<<fmt.bits)]
    choices = [v for v in choices if not isinstance(v,Decimal) or v.is_finite()]
    acc = format_named("fp64_e11m52_accumulator")
    for k in (3,9,27,64,128,256,1024):
        left = [rng.choice(choices) for _ in range(k)]
        right = [rng.choice(choices) for _ in range(k)]
        actual = backend.flex_gemm(left,right,batch=1,channels=1,k=k,accumulator=acc.name)
        assert actual == (model_c(zip(left,right),acc),), (name,k)
