"""Lane Q4 (agent r3): the low-memory bias correction equals the frozen one bit for bit (CPU, synthetic data)."""
import numpy as np
import pytest
import torch

from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
from tools.experiment_b2.boundaries import analyze, quantizing_nodes
from tools.experiment_b2.engine import bias_correct as frozen_bias_correct
from tools.experiment_b2.recipe import named
from tools.experiment_b2_attrib import lowmem


def _synthetic(graph, fmt, seed=0):
    plan = analyze(graph, named("default"), fmt)
    rng = np.random.default_rng(seed)
    arrays, maxima = {}, {}
    for node in quantizing_nodes(plan):
        values = rng.normal(0, 1, 512).astype(np.float32)
        if plan[node]["signedness"] == "unsigned":
            values = np.abs(values)
        arrays[node], maxima[node] = values, float(np.abs(values).max())
    return plan, arrays, maxima


@pytest.fixture(autouse=True)
def copying_store(monkeypatch):
    # On the CPU ``t.to("cpu")`` returns ``t`` itself; clone so that the test exercises separate copies,
    # as the GPU path does with its host round trips.
    monkeypatch.setattr(lowmem, "_store", lambda t: t.clone())
    monkeypatch.setattr(lowmem, "_load", lambda t, device: t.clone())
    lowmem._FP_CACHE.clear()


@pytest.mark.parametrize("model,fmt,images,chunk", [("mobilenet_v3_large", "int8", 5, 2),
                                                     ("resnet18", "fp6_e2m3", 4, 3)])
def test_lowmem_bias_correction_is_bit_identical(model, fmt, images, chunk):
    import copy
    from tools.experiment_b.classifier import load_model
    from tools.experiment_b2_attrib.engine import Arm, build_arm, prepare_shared
    from tools.experiment_b2_attrib.groups import partition
    from tools.experiment_b2_attrib.arms import parse_arm
    graph = load_model(model, "cpu")[0]
    plan, arrays, maxima = _synthetic(graph, fmt)
    rng = np.random.default_rng(1)
    inputs = rng.normal(0, 1, (images, 3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        shared = prepare_shared(model, graph, fmt, named("default"), arrays, maxima, "cpu", torch.from_numpy(inputs[:1]))
        groups = partition(model, shared.fp_graph, shared.plan)
        arms = ["ref_default", "wide:role.stem" if model == "resnet18" else "wide:role.hswish"]
        for name in arms:
            arm = parse_arm(name, shared.plan, groups)
            base = {k: dict(v) for k, v in shared.plan.items()}
            for node in arm.wide:
                base[node].update(quantizes=False, reason="attrib_wide", signedness=None)
            scales = {n: shared.activation_scales[n] for n in quantizing_nodes(base)}
            a, b = copy.deepcopy(shared.weight_graph), copy.deepcopy(shared.weight_graph)
            report_a = frozen_bias_correct(shared.fp_graph, a, base, shared.quantizers, scales, inputs, chunk=chunk)
            report_b = lowmem.bias_correct(shared.fp_graph, b, base, shared.quantizers, scales, inputs, chunk=chunk)
            assert report_a == report_b
            for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
                assert ka == kb and torch.equal(va, vb)
        # The FP32 pass ran once for both arms (cached per graph and input set).
        assert len(lowmem._FP_CACHE) == 1


def test_install_patches_both_engines():
    import tools.experiment_b2.engine as b2_engine
    import tools.experiment_b2_attrib.engine as attrib_engine
    old = (b2_engine.bias_correct, attrib_engine.bias_correct)
    try:
        lowmem.install()
        assert b2_engine.bias_correct is lowmem.bias_correct and attrib_engine.bias_correct is lowmem.bias_correct
    finally:
        b2_engine.bias_correct, attrib_engine.bias_correct = old


def test_pcf_collect_streams_the_exact_channel_statistics(monkeypatch):
    import math
    from tools.experiment_b.classifier import load_model
    from tools.experiment_b2_attrib import pcfull
    graph = load_model("resnet18", "cpu")[0]
    inputs = np.random.default_rng(2).normal(0, 1, (5, 3, 32, 32)).astype(np.float32)
    monkeypatch.setattr(pcfull, "CHUNK", 2)
    monkeypatch.setattr(pcfull, "MAX_VALUES", 7)
    nodes = ["layer1_0_relu", "layer4_1_relu_1"]
    stats = pcfull.collect(graph, nodes, inputs, "cpu")
    seen = {}

    class Probe(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if node.name in nodes:
                seen.setdefault(node.name, []).append(result.detach().clone())
            return result

    with torch.inference_mode():  # same chunks as collect (CPU kernels depend on the batch size)
        for start in range(0, len(inputs), 2):
            Probe(graph).run(torch.from_numpy(inputs[start:start + 2]))
    seen = {k: torch.cat(v) for k, v in seen.items()}
    for name in nodes:
        full = seen[name].transpose(0, 1).reshape(seen[name].shape[1], -1)
        k = max(1, math.ceil(full.shape[1] / 7))
        assert np.array_equal(stats[name][0], full.abs().amax(dim=1).numpy())
        assert np.array_equal(stats[name][1], full[:, ::k].numpy())


def test_fixbc_control_reproduces_the_default_and_keeps_biases():
    from tools.experiment_b.classifier import load_model
    from tools.experiment_b2_attrib.engine import Arm, build_arm, prepare_shared
    from tools.experiment_b2_attrib.groups import partition
    from tools.experiment_b2_attrib.lmrunner import build_fixbc, parse_lm
    import tools.experiment_b2_attrib.engine as attrib_engine
    graph = load_model("resnet18", "cpu")[0]
    plan, arrays, maxima = _synthetic(graph, "int6")
    inputs = np.random.default_rng(3).normal(0, 1, (4, 3, 32, 32)).astype(np.float32)
    with torch.inference_mode():
        shared = prepare_shared("resnet18", graph, "int6", named("default"), arrays, maxima, "cpu", torch.from_numpy(inputs[:1]))
        groups = partition("resnet18", shared.fp_graph, shared.plan)
        before = attrib_engine.bias_correct
        reference, _ = build_arm(shared, Arm("ref_default", kind="reference"), inputs)
        fixed = {}
        control, meta = build_fixbc(shared, parse_lm("ref_default+fixbc", shared.plan, groups)[0], inputs, fixed)
        assert attrib_engine.bias_correct is before and meta["fixbc"]["version"] == "fixbc_v1"
        for (ka, va), (kb, vb) in zip(reference.module.state_dict().items(), control.module.state_dict().items()):
            assert ka == kb and torch.equal(va, vb)
        wide, _ = build_fixbc(shared, parse_lm("wide:role.stem+fixbc", shared.plan, groups)[0], inputs, fixed)
        biases = {k: v for k, v in wide.module.state_dict().items() if k.endswith("bias")}
        assert all(torch.equal(v, reference.module.state_dict()[k]) for k, v in biases.items())
