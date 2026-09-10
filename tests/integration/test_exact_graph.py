import copy

import pytest

from public.inference.tensor import Encoding, Tensor
from public.quantization.graph.executable import execute, freeze_graph, graph_sha256
from public.inference.conformance_job import SCHEMA, source_identity, run_job, validate_job
from public.experiments.registry import ExperimentRegistry
from public.experiments.scheduler import LocalScheduler
from public.quantization.calibration.mse import mse_scale


def fixture_graph():
    encoding = Encoding("fp6_e3m2")
    weights = Tensor.quantize([1, 0, 0, 1], (2, 2), encoding)
    graph = freeze_graph(inputs={"x": encoding.document()}, constants={"w": weights},
        nodes=[{"name": "fc", "op": "linear", "inputs": ["x", "w"], "attrs": {
                    "accumulator": "fp16_e5m10_accumulator", "output": encoding.document(), "bias": ["0.5", "-0.5"], "activation": "relu"}},
               {"name": "residual", "op": "elementwise", "inputs": ["x", "fc"], "attrs": {
                    "operation": "add", "accumulator": "fp16_e5m10_accumulator", "alignment": encoding.document(), "output": encoding.document()}}],
        outputs=["residual"], provenance={"kind": "synthetic_conformance"})
    inputs = {"x": Tensor.quantize([1, 2, 3, 4], (2, 2), encoding)}
    return graph, inputs


def test_graph_execution_and_manifest_identity():
    graph, inputs = fixture_graph()
    result = execute(graph, inputs)
    assert result["outputs"]["residual"].values() == (2.5, 3.5, 6, 8)
    assert len(result["layers"]) == 2
    changed = copy.deepcopy(graph)
    changed["nodes"][0]["attrs"]["bias"][0] = "1"
    assert graph_sha256(changed) != graph_sha256(graph)
    changed["manifest_hashes"]["fp6_e3m2"] = "0" * 64
    with pytest.raises(ValueError, match="manifest identity"):
        execute(changed, inputs)
    changed = copy.deepcopy(graph)
    changed["nodes"][0]["op"] = "torch_fallback"
    with pytest.raises(ValueError, match="unsupported exact"):
        execute(changed, inputs)


def test_real_executor_through_fenced_registry_and_deduplication(tmp_path):
    graph, inputs = fixture_graph()
    config = {"schema_version": SCHEMA, "purpose": "engine_conformance", "graph": graph,
              "inputs": {name: value.document() for name, value in inputs.items()},
              "runtime": {"backend": "reference", "semantic_version": "2.0.0", "source_sha256": source_identity()}}
    with ExperimentRegistry(tmp_path / "phase2.sqlite") as registry:
        run_id, status = registry.submit(config)
        assert status == "PENDING"
        assert LocalScheduler(registry, worker_id="exact").run_once(run_job)
        assert registry.submit(config) == (run_id, "COMPLETED")
        assert not LocalScheduler(registry, worker_id="repeat").run_once(run_job)
    changed = copy.deepcopy(config)
    changed["runtime"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_job(changed)


def test_calibration_is_deterministic_and_preserves_family_policy():
    int_result = mse_scale([-2, -1, 0, 1, 2], "int4", coarse_candidates=10, fine_candidates=5)
    assert int_result == mse_scale([-2, -1, 0, 1, 2], "int4", coarse_candidates=10, fine_candidates=5)
    assert int_result["candidate_count"] == 15
    assert mse_scale([-1000, 1000], "fp6_e3m2")["scale"] == "1"
    mx = mse_scale([0, 1, 2, 4, 8], "mxfp4_e2m1")
    assert mx["candidate_count"] == 255
    assert mx["mse"] == "0"
    assert mse_scale([-1,0,1], "ternary", coarse_candidates=10, fine_candidates=5)["mse"] == "0"
    assert mse_scale([1e30], "fp8_e5m2")["mse"] == "Infinity"


def test_folded_torchvision_style_graph_lowers_and_runs():
    torch = pytest.importorskip("torch")
    from torch import nn
    from public.quantization.graph.torchvision import lower
    model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2)).eval()
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[1].weight.fill_(1)
        model[1].running_var.fill_(1 - model[1].eps)
        model[5].weight.copy_(torch.eye(2))
        model[5].bias.zero_()
    encoding = Encoding("fp6_e3m2")
    graph = lower(model, input_encoding=encoding, weight_format="fp6_e3m2", accumulator="fp16_e5m10_accumulator",
                  provenance={"kind": "synthetic_conformance"})
    assert all("batch_norm" not in node["op"] for node in graph["nodes"])
    value = Tensor.quantize([1, 2, 3, 4], (1, 1, 2, 2), encoding)
    result = execute(graph, {next(iter(graph["inputs"])): value})
    assert next(iter(result["outputs"].values())).values() == (2.5, 2.5)
