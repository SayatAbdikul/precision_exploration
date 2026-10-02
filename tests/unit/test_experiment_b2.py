"""Hand-checkable tests of every B2 recipe component and the v1 regression."""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tools.experiment_b.classifier import ObservingInterpreter, prepare_qdq  # noqa: E402
from tools.experiment_b.quantizer import numpy_qdq as v1_numpy_qdq, table  # noqa: E402
from tools.experiment_b2 import codebook  # noqa: E402
from tools.experiment_b2.boundaries import analyze  # noqa: E402
from tools.experiment_b2.codebook import TableQuantizer  # noqa: E402
from tools.experiment_b2.engine import audit_summary, prepare_b2  # noqa: E402
from tools.experiment_b2.recipe import NAMED, Recipe, V1_MAXABS, V1_PERCENTILE, hardware_semantics, named  # noqa: E402
from tools.experiment_b2.scales import mse_search, scale_for_top  # noqa: E402


class Tiny(nn.Module):
    """Every operator kind the classifiers use, small enough to reason about."""

    def __init__(self):
        super().__init__()
        self.c1, self.i1, self.r1 = nn.Conv2d(3, 8, 3, padding=1), nn.Identity(), nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.c2, self.i2, self.hs = nn.Conv2d(8, 8, 3, padding=1), nn.Identity(), nn.Hardswish()
        self.se_pool, self.fc, self.hsig = nn.AdaptiveAvgPool2d(1), nn.Conv2d(8, 8, 1), nn.Hardsigmoid()
        self.c3, self.r2 = nn.Conv2d(8, 8, 1), nn.ReLU6()
        self.avg, self.drop, self.lin = nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(), nn.Linear(8, 5)

    def forward(self, x):
        a = self.pool(self.r1(self.i1(self.c1(x))))
        b = self.hs(self.i2(self.c2(a)))
        b = self.hsig(self.fc(self.se_pool(b))) * b
        d = self.r2(self.c3(b) + a)
        return self.lin(self.drop(torch.flatten(self.avg(d), 1)))


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(7)
    graph = torch.fx.symbolic_trace(Tiny().eval())
    inputs = torch.randn(16, 3, 8, 8)
    observer = ObservingInterpreter(graph)
    with torch.inference_mode():
        observer.run(inputs)
    return graph, inputs, dict(observer.samples), dict(observer.maxima)


def test_recipe_identities_and_validation():
    assert V1_MAXABS.v1_equivalent() == "maxabs" and V1_PERCENTILE.v1_equivalent() == "percentile_99_9"
    assert all(recipe.v1_equivalent() is None for name, recipe in NAMED.items() if not name.startswith("v1_"))
    assert dataclasses.replace(V1_MAXABS, weight_range="percentile_99_9").v1_equivalent() is None
    with pytest.raises(ValueError):
        Recipe(boundaries="sometimes")
    with pytest.raises(ValueError):
        Recipe(unsigned=1)
    text = hardware_semantics(named("cum5_bias_correction"))
    assert "once after" in text["requantization_points"] and "unsigned" in text["signedness"]


def test_unsigned_integer_codebook_levels_ties_and_clipping():
    levels, boundaries, ties = codebook.unsigned_table("int4")
    np.testing.assert_array_equal(levels, np.arange(16))
    np.testing.assert_array_equal(boundaries, np.arange(15) + 0.5)
    x = np.array([-3, 0.4, 0.5, 1.5, 2.5, 14.5, 15.5, 300], dtype=np.float32)
    # Ties go to the even code; negatives clip to 0 and overflow clips to 15.
    np.testing.assert_array_equal(codebook.numpy_qdq(x, "int4", "unsigned", 1), [0, 0, 0, 2, 2, 14, 15, 15])
    np.testing.assert_array_equal(codebook.numpy_qdq(x, "int4", "unsigned", 2), [0, 0, 0, 2, 2, 14, 16, 30])
    q = TableQuantizer("int4", "unsigned", "cpu")
    np.testing.assert_array_equal(q(torch.tensor(x), torch.tensor(2.0)).numpy(), codebook.numpy_qdq(x, "int4", "unsigned", 2))
    assert q.top == 15 and q.min_positive == 1
    with pytest.raises(ValueError, match="integer formats only"):
        codebook.unsigned_table("fp8_e4m3fn")
    # Signed path is the v1 table and the v1 arithmetic.
    y = np.linspace(-9, 9, 73, dtype=np.float32)
    np.testing.assert_array_equal(codebook.numpy_qdq(y, "int4", "signed", 0.75), v1_numpy_qdq(y, "int4", 0.75))
    assert codebook.describe("int8", "unsigned")["codes"][-1] == 255
    signed = codebook.describe("int4", "signed")
    assert signed["levels"][0] == -8 and signed["codes"][0] == 8 and signed["codes"][-1] == 7


def test_boundary_analysis(tiny):
    graph = tiny[0]
    v1 = analyze(graph, V1_MAXABS, "int8")
    quantizing = {k for k, v in v1.items() if v["quantizes"]}
    assert quantizing == {"x", "c1", "r1", "pool", "c2", "hs", "se_pool", "fc", "hsig", "mul", "c3", "add", "r2", "avg", "lin"}
    assert all(v["signedness"] == "signed" for v in v1.values() if v["quantizes"])
    relu = analyze(graph, named("a_fused_relu"), "int8")
    assert relu["c1"]["reason"] == "fused_into:r1" and relu["add"]["reason"] == "fused_into:r2"
    assert relu["pool"]["reason"] == "maxpool_passthrough"
    # Hardswish and Hardsigmoid inputs still quantize under fused_relu; c3 feeds an add, so it quantizes.
    assert relu["c2"]["quantizes"] and relu["fc"]["quantizes"] and relu["c3"]["quantizes"]
    every = analyze(graph, named("a_fused_all"), "int8")
    assert every["c2"]["reason"] == "fused_into:hs" and every["fc"]["reason"] == "fused_into:hsig"
    assert every["hs"]["quantizes"] and every["hsig"]["quantizes"]
    unsigned = analyze(graph, named("b_unsigned"), "int8")
    expected_nonnegative = {"r1", "pool", "hsig", "r2", "avg", "drop", "flatten"}
    assert {k for k, v in unsigned.items() if v["nonnegative"]} == expected_nonnegative
    assert {k for k, v in unsigned.items() if v["signedness"] == "unsigned"} == {"r1", "pool", "hsig", "r2", "avg"}
    # Hardswish output can be negative, a product with it is not provably non-negative.
    assert not unsigned["hs"]["nonnegative"] and not unsigned["mul"]["nonnegative"]
    # Unsigned codes are an integer-only identity: other families stay signed.
    assert all(v["signedness"] == "signed" for v in analyze(graph, named("b_unsigned"), "fp8_e4m3fn").values() if v["quantizes"])
    exempt = analyze(graph, named("g_no_input_no_logits"), "int8")
    assert exempt["x"]["reason"] == "input_exempt" and exempt["lin"]["reason"] == "logits_exempt"


def test_mse_search_exact_grid_keeps_maxabs_scale():
    q = TableQuantizer("int4", "signed", "cpu")
    values = torch.tensor(0.5 * np.arange(-7, 8, dtype=np.float32)).reshape(1, -1)
    scales, info = mse_search(values, q)
    assert scales[0] == np.float32(0.5) and info["mse"][0] == 0 and info["ratio_to_maxabs_scale"][0] == 1


def test_mse_search_clips_an_outlier_and_rows_are_independent():
    q = TableQuantizer("int4", "signed", "cpu")
    rng = np.random.default_rng(3)
    bulk = rng.uniform(-1, 1, 4096).astype(np.float32)
    row = np.concatenate([bulk, [10]]).astype(np.float32)
    plain = np.concatenate([bulk, [1]]).astype(np.float32)
    scales, info = mse_search(torch.tensor(np.stack([row, plain])), q)
    # Max-abs would use 10/7 and round most of the bulk to zero (error about 0.2 per element); clipping the
    # single outlier costs at most 81/4097 per element, so the search must clip.
    assert scales[0] < 10 / 7 / 4 and info["mse"][0] < info["mse_at_maxabs_scale"][0]
    assert 0.1 < scales[1] <= np.float32(1 / 7) * 1.01
    alone, _ = mse_search(torch.tensor(row).reshape(1, -1), q)
    assert alone[0] == scales[0]
    # The returned scale really is the reported optimum of the actual QDQ function.
    error = ((codebook.numpy_qdq(row, "int4", "signed", scales[0]) - row).astype(np.float64) ** 2).mean()
    assert error == pytest.approx(info["mse"][0], rel=1e-12)
    for other in (scales[0] * 0.9, scales[0] * 1.1):
        assert ((codebook.numpy_qdq(row, "int4", "signed", np.float32(other)) - row).astype(np.float64) ** 2).mean() >= error


def test_mse_search_moves_tapered_codebook_into_its_dense_region():
    q = TableQuantizer("posit8_es1", "signed", "cpu")
    rng = np.random.default_rng(5)
    values = torch.tensor(rng.normal(0, 1, 8192).astype(np.float32)).reshape(1, -1)
    scales, info = mse_search(values, q)
    # Max-abs maps the largest value to 4096 where posit8 has no fraction bits.
    assert info["ratio_to_maxabs_scale"][0] > 64 and info["mse"][0] < info["mse_at_maxabs_scale"][0] / 100
    zero, _ = mse_search(torch.zeros(2, 9), q)
    np.testing.assert_array_equal(zero, [1, 1])
    assert scale_for_top(0.0, 255) == 1.0 and scale_for_top(51.0, 255) == float(np.float32(0.2))


FORMATS = ("int8", "int4", "fp8_e4m3fn", "fp6_e2m3", "posit8_es1", "log8", "nf4")


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("recipe", ("v1_maxabs", "v1_percentile_99_9"))
def test_b2_with_all_switches_off_is_bit_identical_to_v1(tiny, name, recipe):
    graph, inputs, arrays, maxima = tiny
    with torch.inference_mode():
        old, old_scales = prepare_qdq(graph, name, named(recipe).v1_equivalent(), arrays, maxima, "cpu")
        new, new_scales = prepare_b2(graph, name, named(recipe), arrays, maxima, "cpu")
        expected, actual = old.run(inputs), new.run(inputs)
    assert old_scales["activation_scales"] == new_scales["activation_scales"]
    assert old_scales["weight_scales"] == new_scales["weight_scales"]
    assert torch.equal(expected, actual)
    for (key, a), (_, b) in zip(old.module.state_dict().items(), new.module.state_dict().items()):
        assert torch.equal(a, b), key


def _trace(engine, inputs, names):
    """Outputs of selected modules while the engine runs (conv outputs are pre-activation)."""
    seen, hooks = {}, []
    for name in names:
        hooks.append(engine.module.get_submodule(name).register_forward_hook(
            lambda _m, _i, out, name=name: seen.__setitem__(name, out.detach().clone())))
    with torch.inference_mode():
        out = engine.run(inputs) if hasattr(engine, "run") else engine(inputs)
    for hook in hooks:
        hook.remove()
    return out, seen


def test_bias_correction_matches_fp32_channel_means(tiny):
    graph, inputs, arrays, maxima = tiny
    layers = ("c1", "c2", "fc", "c3", "lin")
    recipe = dataclasses.replace(named("cum1_fused"), bias_correction="empirical")
    with torch.inference_mode():
        plain, _ = prepare_b2(graph, "int4", named("cum1_fused"), arrays, maxima, "cpu")
        fixed, meta = prepare_b2(graph, "int4", recipe, arrays, maxima, "cpu", bias_inputs=inputs.numpy())
    _, reference = _trace(torch.fx.Interpreter(graph), inputs, layers)
    _, before = _trace(plain, inputs, layers)
    _, after = _trace(fixed, inputs, layers)

    def gap(seen, name):
        dims = [0] + list(range(2, seen[name].ndim))
        return float((seen[name].mean(dims) - reference[name].mean(dims)).abs().max())
    for name in layers:
        assert gap(after, name) < 2e-5, name
    assert max(gap(before, name) for name in layers) > 1e-2
    assert set(meta["bias_correction"]) == set(layers)
    # Only biases change: quantized weights are the same with and without the correction.
    for name in layers:
        assert torch.equal(plain.module.get_submodule(name).weight, fixed.module.get_submodule(name).weight)
        assert not torch.equal(plain.module.get_submodule(name).bias, fixed.module.get_submodule(name).bias)
    with pytest.raises(ValueError, match="needs calibration inputs"):
        prepare_b2(graph, "int4", recipe, arrays, maxima, "cpu")


def test_fused_unsigned_and_exemption_semantics(tiny):
    graph, inputs, arrays, maxima = tiny
    with torch.inference_mode():
        engine, meta = prepare_b2(graph, "int4", named("cum2_unsigned"), arrays, maxima, "cpu", audit=True)
        out, seen = _trace(engine, inputs, ("c1", "lin"))
        fp_out, fp_seen = _trace(torch.fx.Interpreter(graph), inputs, ("c1",))
    # Unsigned boundary: scale is max/15 (not max/7) and outputs are non-negative multiples of the scale.
    assert meta["activation_scales"]["r1"] == scale_for_top(maxima["r1"], 15)
    assert meta["activation_scales"]["add"] if "add" in meta["activation_scales"] else True
    assert "c1" not in meta["activation_scales"] and "pool" not in meta["activation_scales"]
    audit = audit_summary(engine)
    assert audit["r1"]["levels_total"] == 16 and audit["x"]["levels_total"] == 16 and audit["r1"]["sqnr_db"] > 0
    # Logit exemption returns the raw linear output; with it quantized the output lies on the int4 grid.
    with torch.inference_mode():
        free, _ = prepare_b2(graph, "int4", named("g_no_logits"), arrays, maxima, "cpu")
        raw, traced = _trace(free, inputs, ("lin",))
    assert torch.equal(raw, traced["lin"])
    scale = np.float32(meta["activation_scales"]["lin"])
    np.testing.assert_array_equal(out.numpy(), codebook.numpy_qdq(seen["lin"].numpy(), "int4", "signed", scale))


def test_mse_recipe_lowers_reconstruction_error_of_every_weight(tiny):
    graph, inputs, arrays, maxima = tiny
    with torch.inference_mode():
        a, _ = prepare_b2(graph, "int4", named("cum1_fused"), arrays, maxima, "cpu")
        b, meta = prepare_b2(graph, "int4", dataclasses.replace(named("cum1_fused"), weight_range="mse"), arrays, maxima, "cpu")
    for name in ("c1", "c2", "fc", "c3", "lin"):
        w = graph.get_submodule(name).weight
        assert (b.module.get_submodule(name).weight - w).pow(2).sum() <= (a.module.get_submodule(name).weight - w).pow(2).sum()
    assert meta["mse_search"]["version"] == "b2_mse_scale_search_v1"


def test_export_reproduces_weights_and_boundaries(tiny, tmp_path, monkeypatch):
    from tools.experiment_b2 import export
    graph, inputs, arrays, maxima = tiny
    recipe = dataclasses.replace(named("cum4_weight_mse"), bias_correction="empirical")
    with torch.inference_mode():
        engine, meta = prepare_b2(graph, "int4", recipe, arrays, maxima, "cpu", bias_inputs=inputs.numpy())
    constants, weights = export.weight_arrays(graph, engine, "int4", meta)
    levels, codes = table("int4")[0], codebook.level_codes("int4", "signed")
    lookup = {int(c): float(l) for c, l in zip(codes, levels)}
    for name in ("c1", "lin"):
        decoded = np.vectorize(lookup.get)(constants[f"{name}.weight_codes"]).astype(np.float32)
        scales = constants[f"{name}.weight_scales"].reshape((-1,) + (1,) * (decoded.ndim - 1))
        np.testing.assert_array_equal(decoded * scales, constants[f"{name}.weight_reconstructed"])
        assert weights[name]["bias_changed"]
    rows = {row["name"]: row for row in export.node_rows(graph, meta)}
    assert rows["r1"]["boundary"]["codebook"] == "int4:unsigned" and rows["r1"]["boundary"]["nonnegative"]
    assert rows["c1"]["boundary"] == {"quantizes": False, "reason": "fused_into:r1", "signedness": None,
                                      "nonnegative": False, "codebook": None}
    scale = rows["r1"]["boundary"]["scale"]
    assert np.frombuffer(bytes.fromhex(rows["r1"]["boundary"]["scale_fp32_hex"]), dtype=">f4")[0] == np.float32(scale)
    assert rows["c1"]["attrs"]["groups"] == 1 and rows["output"]["op"] == "output"


def test_cross_layer_equalization_preserves_function_and_balances_ranges(tiny):
    from tools.experiment_b2.equalization import equalize_graph, find_pairs

    class Chain(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.ia, self.ra = nn.Conv2d(3, 6, 3, padding=1), nn.Identity(), nn.ReLU()
            self.dw, self.rd = nn.Conv2d(6, 6, 3, padding=1, groups=6), nn.ReLU()
            self.p, self.r6, self.q = nn.Conv2d(6, 4, 1), nn.ReLU6(), nn.Conv2d(4, 4, 1)

        def forward(self, x):
            return self.q(self.r6(self.p(self.rd(self.dw(self.ra(self.ia(self.a(x))))))))

    torch.manual_seed(11)
    model = Chain().eval()
    with torch.no_grad():
        model.a.weight[0] *= 40  # one channel with a much larger weight range
    graph = torch.fx.symbolic_trace(model)
    # ReLU pairs only: (a, dw) and (dw, p); the ReLU6 pair (p, q) is not function-preserving and is skipped.
    assert [(a, b, d) for a, _, b, _, d in find_pairs(graph)] == [("a", "dw", True), ("dw", "p", False)]
    equalized, report = equalize_graph(graph)
    x = torch.randn(4, 3, 8, 8)
    with torch.inference_mode():
        torch.testing.assert_close(equalized(x), graph(x), rtol=1e-4, atol=1e-5)
    r1 = equalized.a.weight.abs().flatten(1).amax(1)
    r2 = equalized.dw.weight.abs().flatten(1).amax(1)
    r3 = equalized.p.weight.abs().transpose(0, 1).flatten(1).amax(1)
    torch.testing.assert_close(r1, r2, rtol=1e-3, atol=0)
    torch.testing.assert_close(r2, r3, rtol=1e-3, atol=0)
    assert len(report["pairs"]) == 2 and report["pairs"][0]["scale_max"] > 2
    # The classifier-like fixture has no eligible pair (Hardswish, squeeze-excite, add): nothing changes.
    assert find_pairs(tiny[0]) == []
