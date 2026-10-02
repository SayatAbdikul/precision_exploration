"""Lane Q3: the re-implemented sequential correction equals B2's, and each variant changes only what it says."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from tools.experiment_b2 import frozen  # noqa: F401
from tools.experiment_b2.boundaries import analyze, quantizing_nodes
from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2.engine import bias_correct
from tools.experiment_b2.recipe import named
from tools.experiment_b2_collapse.build import anchored_scale, quantize_weights_by_rule
from tools.experiment_b2_collapse.correct import staged_correct
from tools.experiment_b2_collapse.evaluate import one_class


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        a = self.relu1(self.conv1(x))
        b = self.relu2(self.conv2(a) + a)
        return self.fc(torch.flatten(self.pool(b), 1))


def setup(fmt="int4", recipe_name="default", seed=0):
    torch.manual_seed(seed)
    graph = torch.fx.symbolic_trace(Tiny().eval())
    recipe = named(recipe_name)
    plan = analyze(graph, recipe, fmt)
    quantizers = {"signed": TableQuantizer(fmt, "signed", torch.device("cpu"))}
    if any(row["signedness"] == "unsigned" for row in plan.values()):
        quantizers["unsigned"] = TableQuantizer(fmt, "unsigned", torch.device("cpu"))
    inputs = np.random.default_rng(seed).normal(size=(40, 3, 6, 6)).astype(np.float32)
    # activation scales from FP32 max-abs of each quantizing node
    values = {}
    interp = torch.fx.Interpreter(graph)
    original = interp.run_node

    def record(node):
        out = original(node)
        if isinstance(out, torch.Tensor):
            values[node.name] = float(out.detach().abs().max())
        return out
    interp.run_node = record
    with torch.no_grad():
        interp.run(torch.from_numpy(inputs))
    scales = {n: max(values[n], 1e-3) / quantizers[plan[n]["signedness"]].top for n in quantizing_nodes(plan)}
    q_graph = copy.deepcopy(graph)
    with torch.no_grad():
        for name in ("conv1", "conv2", "fc"):
            module = q_graph.get_submodule(name)
            w = module.weight
            s = w.reshape(w.shape[0], -1).abs().amax(dim=1) / quantizers["signed"].top
            module.weight.copy_(quantizers["signed"](w, s.reshape((-1,) + (1,) * (w.ndim - 1))))
    return graph, q_graph, plan, quantizers, scales, inputs


def biases(graph):
    return {name: graph.get_submodule(name).bias.detach().clone() for name in ("conv1", "conv2", "fc")}


@pytest.mark.parametrize("fmt,recipe_name", [("int4", "default"), ("int6", "default"), ("posit8_es1", "minimal")])
def test_global_is_bit_identical_to_b2(fmt, recipe_name):
    graph, q_graph, plan, quantizers, scales, inputs = setup(fmt, recipe_name)
    reference, mine = copy.deepcopy(q_graph), copy.deepcopy(q_graph)
    with torch.inference_mode():
        bias_correct(graph, reference, plan, quantizers, scales, inputs, chunk=16)
        report = staged_correct(graph, mine, plan, quantizers, scales, inputs, policy="global", chunk=16, store="cpu")
    for name, value in biases(reference).items():
        assert torch.equal(value, biases(mine)[name]), name
    for row in report.values():  # global = local + inherited (RMS triangle)
        assert row["rms_global"] <= row["rms_local"] + row["rms_inherited"] + 1e-6


def test_first_layer_has_no_inherited_part_when_input_is_exact():
    graph, q_graph, plan, quantizers, scales, inputs = setup("int4", "default_no_bias_correction")
    for row in plan.values():
        row["quantizes"] = False  # A32: every boundary exact
    with torch.inference_mode():
        report = staged_correct(graph, q_graph, plan, quantizers, scales, inputs, policy="global", chunk=16)
    first = report["conv1"]
    assert first["rms_inherited"] < 1e-6 * max(1.0, first["rms_global"])
    assert abs(first["rms_analytic_fp32"] - first["rms_local"]) < 1e-6


def test_variants_change_only_what_they_say():
    graph, q_graph, plan, quantizers, scales, inputs = setup("int4", "default")
    plan["x"]["quantizes"] = False  # exact input, so the first layer's local and analytic parts coincide
    del scales["x"]
    before = biases(q_graph)
    results = {}
    for policy in ("global", "executed", "linear_only", "cap_rms", "none", "local_empirical", "analytic_fp32"):
        target = copy.deepcopy(q_graph)
        with torch.inference_mode():
            results[policy] = (staged_correct(graph, target, plan, quantizers, scales, inputs, policy=policy, chunk=16),
                               biases(target))
    # none leaves every bias unchanged
    assert all(torch.equal(results["none"][1][k], before[k]) for k in before)
    # linear_only: rectified layers (conv1 -> relu; conv2 feeds the add, fc feeds the output) unchanged
    report = results["linear_only"][0]
    assert report["conv1"]["consumer"] == "relu" and torch.equal(results["linear_only"][1]["conv1"], before["conv1"])
    assert report["conv2"]["consumer"] == "identity"
    # executed equals global up to float rounding
    for k in before:
        assert torch.allclose(results["executed"][1][k], results["global"][1][k], atol=1e-5)
    # cap: applied RMS never above the FP32 channel-mean RMS
    for row in results["cap_rms"][0].values():
        assert row["rms_applied"] <= row["rms_fp32_channel_mean"] * (1 + 1e-5) + 1e-7
    # the first layer's analytic and local corrections coincide (input left exact above)
    first_local = results["local_empirical"][1]["conv1"] - before["conv1"]
    first_analytic = results["analytic_fp32"][1]["conv1"] - before["conv1"]
    assert torch.allclose(first_local, first_analytic, atol=1e-6)


def test_shared_module_is_refused():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return self.conv(self.conv(x))
    graph = torch.fx.symbolic_trace(Shared().eval())
    plan = analyze(graph, named("default"), "int8")
    quantizers = {"signed": TableQuantizer("int8", "signed", torch.device("cpu"))}
    scales = {n: 0.1 for n in quantizing_nodes(plan)}
    with pytest.raises(ValueError, match="more than one node"):
        staged_correct(graph, copy.deepcopy(graph), plan, quantizers, scales,
                       np.zeros((2, 3, 4, 4), dtype=np.float32), policy="global")


def test_anchor_one_maps_maxabs_to_level_one():
    module = nn.Conv2d(2, 3, 3)
    with torch.no_grad():
        module.weight.copy_(torch.randn_like(module.weight))
    scales, shaped = anchored_scale(module, 1.0)
    maxima = module.weight.reshape(3, -1).abs().amax(dim=1)
    assert torch.allclose(torch.tensor(scales), maxima)
    q = TableQuantizer("posit8_es1", "signed", torch.device("cpu"))
    out = q(module.weight, shaped)
    assert torch.allclose(out.reshape(3, -1).abs().amax(dim=1), maxima)


def test_one_class_share():
    logits = torch.zeros(10, 4)
    logits[:, 2] = 1.0
    logits[0, 3] = 1.0  # one tie
    out = one_class(logits)
    assert out["class"] == 2 and out["share_among_maxima"] == 1.0 and out["collapse"]
    assert abs(out["share_expected"] - 0.95) < 1e-9


@pytest.mark.parametrize("model", ["resnet18", "mobilenet_v2", "mobilenet_v3_large"])
def test_default_and_minimal_differ_only_in_weight_range_and_correction_for_non_integers(model):
    """Mode 3 premise: for non-integer formats the two recipes share boundaries, signedness and activation ranges."""
    from dataclasses import asdict
    from tools.experiment_b.classifier import load_model
    from tools.experiment_b2.blocks import block_plan
    from tools.experiment_b2.codebook import supports_unsigned
    default, minimal = named("default"), named("minimal")
    differing = {k for k, v in asdict(default).items() if asdict(minimal)[k] != v}
    assert differing == {"unsigned", "weight_range", "bias_correction"}
    assert default.activation_range == minimal.activation_range == "mse"
    graph, _, _ = load_model(model, "cpu")
    for fmt in ("fp8_e5m2", "fp6_e3m2", "log6", "posit6_es1", "posit8_es1"):
        assert not supports_unsigned(fmt)
        assert analyze(graph, default, fmt) == analyze(graph, minimal, fmt)
    assert block_plan(graph, default, "mxfp6_e3m2") == block_plan(graph, minimal, "mxfp6_e3m2")
