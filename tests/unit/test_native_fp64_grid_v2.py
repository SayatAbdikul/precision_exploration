"""Graph-bound exact-grid kernels against the unchanged Model C oracle."""
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import random
import time

import numpy as np
import pytest

from public.inference.native import NativeValues, dyadic, prepare_tensor
from public.inference.native_fp64_grid_v2 import (
    ACCUMULATOR, CertifiedGridBackend, GraphGridCertificate, _grid_units,
    build_grid_v2,
)
from public.inference.reference.arithmetic import format_named, model_c
from public.inference.tensor import Tensor, parse_encoding


ROOT = Path(__file__).resolve().parents[2]
GRAPHS = {
    "fp6_e2m3": "a715f60e9ed2eac4c5eec6356f6c7bd445ba95f2247ba9963b07e1adf6cdf607",
    "fp6_e3m2": "2f553a7d3a9ff0bf4365d19789696425711375a56a7b1ed1d4188b982ba28f89",
    "fp7_e3m3": "78f57e0d6ff82fdf4bb7cc326262864e5bcdefef0ccf3f023249b04e9067700b",
}
EXPECTED_BOUNDS = {"fp6_e2m3": 419160, "fp6_e3m2": 5824128, "fp7_e3m3": 27199872}
_BACKENDS = {}


@pytest.fixture(scope="module", params=tuple(GRAPHS))
def case(request):
    fmt = request.param
    graph = json.loads((ROOT / "artifacts/phase3/configurations" / GRAPHS[fmt] / "graph.json").read_text())
    return fmt, graph, GraphGridCertificate(graph)


def backend(kind, graph):
    key = (kind, graph["inputs"]["x"]["format"])
    if key in _BACKENDS:
        return _BACKENDS[key]
    path = build_grid_v2(kind)
    try:
        result = CertifiedGridBackend(kind, graph, grid_library=path)
        _BACKENDS[key] = result
        return result
    except RuntimeError as error:
        if kind == "cuda" and "unavailable" in str(error).lower():
            pytest.skip("CUDA device is unavailable in this test environment")
        raise


def test_actual_weight_prefix_certificate(case):
    fmt, _graph, cert = case
    assert cert.format == fmt
    assert len(cert.records) == 21
    assert max(record.k for record in cert.records) == 4608
    assert max(record.maximum_prefix_units for record in cert.records) == EXPECTED_BOUNDS[fmt]
    assert all(record.maximum_prefix_units == max(record.per_channel_prefix_units)
               and record.maximum_prefix_units <= (1 << 31) - 1 for record in cert.records)


@pytest.mark.parametrize("kind", ["cpp", "cuda"])
def test_every_finite_code_pair_matches_oracle(case, kind):
    fmt, graph, cert = case
    native = backend(kind, graph)
    f = format_named(fmt)
    finite_codes = [code for code in range(1 << f.bits) if
                    not (hasattr(f.decode(code), "is_finite") and not f.decode(code).is_finite())]
    values = [f.decode(code) for code in finite_codes]
    units = np.ascontiguousarray(cert.code_units[finite_codes], dtype=np.int16)
    actual = native._call_gemm(units, units, len(units), len(units), 1, grid=True, shift=cert.shift)
    acc = format_named(ACCUMULATOR)
    for i, left in enumerate(values):
        for j, right in enumerate(values):
            assert actual[i*len(values)+j] == model_c([(left, right)], acc)


@pytest.mark.parametrize("kind", ["cpp", "cuda"])
def test_direct_convolution_geometry_matches_oracle(case, kind):
    fmt, graph, cert = case
    native = backend(kind, graph)
    f, acc = format_named(fmt), format_named(ACCUMULATOR)
    finite_codes = [code for code in range(1 << f.bits) if
                    not (hasattr(f.decode(code), "is_finite") and not f.decode(code).is_finite())]
    rng = random.Random(51)
    xcodes = [rng.choice(finite_codes) for _ in range(4*5*5)]
    wcodes = [rng.choice(finite_codes) for _ in range(4*2*2*2)]
    x, w = cert.code_units[xcodes], cert.code_units[wcodes]
    geometry = dict(n=1, ci=4, h=5, w=5, co=4, kh=2, kw=2, oh=3, ow=4,
                    sh=2, sw=1, ph=1, pw=0, dh=2, dw=1, groups=2)
    actual = native._call_conv(np.ascontiguousarray(x), np.ascontiguousarray(w), geometry,
                               grid=True, shift=cert.shift)
    for oc in range(4):
        for oy in range(3):
            for ox in range(4):
                pairs = []
                for ic in range(2):
                    for kr in range(2):
                        for kc in range(2):
                            iy, ix = oy*2-1+kr*2, ox+kc
                            xcode = xcodes[((oc//2)*2+ic)*25+iy*5+ix] if 0 <= iy < 5 and 0 <= ix < 5 else 0
                            wcode = wcodes[((oc*2+ic)*2+kr)*2+kc]
                            pairs.append((f.decode(xcode), f.decode(wcode)))
                assert actual[(oc*3+oy)*4+ox] == model_c(pairs, acc)


@pytest.mark.parametrize("kind", ["cpp", "cuda"])
def test_selected_fc_exact_states_and_dispatch(case, kind):
    fmt, graph, cert = case
    native = backend(kind, graph)
    f, acc = format_named(fmt), format_named(ACCUMULATOR)
    finite_codes = [code for code in range(1 << f.bits) if
                    not (hasattr(f.decode(code), "is_finite") and not f.decode(code).is_finite())]
    rng = random.Random(131)
    codes = tuple(rng.choice(finite_codes) for _ in range(512))
    x = Tensor((1, 512), codes, parse_encoding(graph["inputs"]["x"]))
    w = Tensor.from_document(graph["constants"]["fc_weight"])
    result = native.flex_gemm(prepare_tensor(x), prepare_tensor(w), batch=1, channels=1000, k=512,
                              accumulator=ACCUMULATOR)
    assert native.last_strategy == "certified_fp64_grid_v2"
    for channel in (0, 1, 77, 999):
        pairs = ((f.decode(codes[i]), f.decode(w.codes[channel*512+i])) for i in range(512))
        assert result[channel] == model_c(pairs, acc)


def test_dispatch_rejects_unsupported_domain_and_graph(case):
    fmt, graph, cert = case
    native = backend("cpp", graph)
    f = format_named(fmt)
    nonfinite = next(code for code in range(1 << f.bits) if
                     hasattr(f.decode(code), "is_finite") and not f.decode(code).is_finite())
    x = Tensor((1, 512), (nonfinite,) + (0,)*511, parse_encoding(graph["inputs"]["x"]))
    assert _grid_units(prepare_tensor(x), cert) is None
    bad = json.loads(json.dumps(graph))
    bad["nodes"][0]["attrs"]["stride"] = [1, 1]
    with pytest.raises(ValueError, match="outside the frozen"):
        GraphGridCertificate(bad)
    # Bad dimensions are rejected before a native pointer can be dereferenced.
    with pytest.raises(ValueError, match="payload or dimensions"):
        native.flex_gemm(prepare_tensor(x), prepare_tensor(x), batch=1, channels=10, k=512,
                         accumulator=ACCUMULATOR)


@pytest.mark.parametrize("kind", ["cpp", "cuda"])
def test_finite_sequential_fma_matches_model_c(case, kind):
    _fmt, graph, _cert = case
    native = backend(kind, graph)
    rng = random.Random(93)
    left = [Fraction(rng.randrange(-(1<<29)+1, (1<<29)-1), 1<<27) for _ in range(67)]
    right = [Fraction(rng.randrange(-(1<<29)+1, (1<<29)-1), 1<<29) for _ in range(67)]
    dtype = np.dtype([("mantissa", "<i8"), ("exponent", "<i4"), ("kind", "<i4")], align=True)
    def prepared(values):
        rows = [dyadic(v) for v in values]
        return NativeValues(np.array([(v.mantissa, v.exponent, v.kind) for v in rows], dtype=dtype))
    actual = native.sequential_gemm(prepared(left), prepared(right), batch=1, channels=1, k=67)
    assert actual == (model_c(zip(left, right), format_named(ACCUMULATOR)),)


@pytest.mark.parametrize("kind", ["cpp", "cuda"])
def test_sequential_rounding_ties_cancellation_and_signed_zero(kind):
    graph = json.loads((ROOT / "artifacts/phase3/configurations" / GRAPHS["fp6_e2m3"] / "graph.json").read_text())
    native = backend(kind, graph)
    dtype = np.dtype([("mantissa", "<i8"), ("exponent", "<i4"), ("kind", "<i4")], align=True)
    def prepared(values):
        rows = [dyadic(v) for v in values]
        return NativeValues(np.array([(v.mantissa, v.exponent, v.kind) for v in rows], dtype=dtype))
    examples = [
        ([Fraction(1), Fraction(1, 1<<26), Fraction(1, 1<<27)],
         [Fraction(1), Fraction(1, 1<<26), Fraction(1, 1<<26)]),
        ([Fraction(1), Fraction(1), Fraction(1, 1<<27)],
         [Fraction(1), Fraction(-1), Fraction(1, 1<<26)]),
        ([Decimal("-0"), Fraction(0), Decimal("-0")],
         [Fraction(1), Fraction(-1), Decimal("-0")]),
    ]
    for left, right in examples:
        actual = native.sequential_gemm(prepared(left), prepared(right), batch=1, channels=1, k=len(left))
        assert actual == (model_c(zip(left, right), format_named(ACCUMULATOR)),)


def test_nonfinite_and_changed_weight_route_to_fallback():
    graph = json.loads((ROOT / "artifacts/phase3/configurations" / GRAPHS["fp6_e2m3"] / "graph.json").read_text())
    native = backend("cpp", graph)
    f = format_named("fp6_e2m3")
    bad_code = next(code for code in range(64) if not f.decode(code).is_finite())
    x = Tensor((1, 512), (bad_code,) + (0,)*511, parse_encoding(graph["inputs"]["x"]))
    w = Tensor.from_document(graph["constants"]["fc_weight"])
    original = native.fallback.flex_gemm
    try:
        native.fallback.flex_gemm = lambda *_args, **_kwargs: (0x1234,)
        native.fallback.last_strategy = "sentinel_fallback"
        assert native.flex_gemm(prepare_tensor(x), prepare_tensor(w), batch=1, channels=1000, k=512,
                                accumulator=ACCUMULATOR) == (0x1234,)
        assert native.last_strategy == "sentinel_fallback"
        good = Tensor((1, 512), (0,)*512, parse_encoding(graph["inputs"]["x"]))
        prepared_w = prepare_tensor(w)
        changed = prepared_w.array.copy()
        location = int(np.flatnonzero((changed["kind"] == 0) & (changed["mantissa"] != 0))[0])
        changed["mantissa"][location] = 0
        assert native.flex_gemm(prepare_tensor(good), NativeValues(changed), batch=1, channels=1000, k=512,
                                accumulator=ACCUMULATOR) == (0x1234,)
        assert native.last_strategy == "sentinel_fallback"
    finally:
        native.fallback.flex_gemm = original


@pytest.mark.parametrize("kind", ["cpp", "cuda"])
def test_selected_first_conv_matches_oracle_samples(kind):
    fmt = "fp6_e2m3"
    graph = json.loads((ROOT / "artifacts/phase3/configurations" / GRAPHS[fmt] / "graph.json").read_text())
    native = backend(kind, graph)
    f, acc = format_named(fmt), format_named(ACCUMULATOR)
    finite = [code for code in range(64) if f.decode(code).is_finite()]
    rng = random.Random(22)
    codes = tuple(rng.choice(finite) for _ in range(3*224*224))
    x = Tensor((1, 3, 224, 224), codes, parse_encoding(graph["inputs"]["x"]))
    w = Tensor.from_document(graph["constants"]["conv1_weight"])
    geometry = dict(n=1, ci=3, h=224, w=224, co=64, kh=7, kw=7, oh=112, ow=112,
                    sh=2, sw=2, ph=3, pw=3, dh=1, dw=1, groups=1)
    prepared_x, prepared_w = prepare_tensor(x), prepare_tensor(w)
    started = time.monotonic()
    states = native.flex_conv2d(prepared_x, prepared_w, geometry=geometry,
                                accumulator=ACCUMULATOR)
    print(f"{kind} selected first-conv dispatch seconds: {time.monotonic()-started:.6f}")
    assert native.last_strategy == "certified_fp64_grid_v2"
    assert len(states) == 64*112*112
    for oc, oy, ox in ((0, 0, 0), (1, 55, 55), (63, 111, 111)):
        pairs = []
        for ic in range(3):
            for kr in range(7):
                for kc in range(7):
                    iy, ix = oy*2-3+kr, ox*2-3+kc
                    xcode = codes[(ic*224+iy)*224+ix] if 0 <= iy < 224 and 0 <= ix < 224 else 0
                    wcode = w.codes[((oc*3+ic)*7+kr)*7+kc]
                    pairs.append((f.decode(xcode), f.decode(wcode)))
        assert states[(oc*112+oy)*112+ox] == model_c(pairs, acc)
