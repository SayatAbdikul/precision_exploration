import copy
import pytest

from tools.phase3.common import digest, read, write, reference
from tools.phase3.screening import checkpoint, save_checkpoint, require_screen_acceptance
from tools.phase3.graphs import accumulator_bounds
from tools.run.phase2_engine import graph_witness


def test_checkpoint_preserves_completed_backend_and_rejects_changed_context(tmp_path):
    path = tmp_path / "image.json"
    context = {"job_sha256": "job", "sample": {"sha256": "sample"}}
    payload = checkpoint(path, context)
    payload["backends"]["cpp"] = {"output_sha256": "result"}
    save_checkpoint(path, payload)
    assert checkpoint(path, context)["backends"]["cpp"]["output_sha256"] == "result"
    with pytest.raises(ValueError, match="provenance"):
        checkpoint(path, {**context, "job_sha256": "different"})
    corrupted = read(path)
    corrupted["backends"]["cpp"]["output_sha256"] = "changed"
    write(path, corrupted)
    with pytest.raises(ValueError, match="hash"):
        checkpoint(path, context)


def test_full_screen_cannot_run_from_prepared_graph_alone(tmp_path):
    with pytest.raises(ValueError, match="full screen blocked"):
        require_screen_acceptance({"configuration_sha256": "prepared"}, tmp_path)


def test_static_bounds_do_not_silently_accept_accumulator_policy():
    graph, _ = graph_witness()
    bounds = accumulator_bounds(graph)
    assert bounds["accepted"] is False
    assert bounds["status"] == "pending_native_sensitivity"


def test_int32_bound_includes_per_channel_mapped_bias_headroom():
    from public.inference.tensor import Tensor, Encoding
    from public.quantization.graph.executable import freeze_graph
    graph = freeze_graph(inputs={"x": Encoding("int8", ("0.001",)).document()},
        constants={"w": Tensor((1, 1), (1,), Encoding("int8", ("0.001",), 0))},
        nodes=[{"name": "linear", "op": "linear", "inputs": ["x", "w"],
                "attrs": {"accumulator": "int32_accumulator", "bias": ["3000"], "output": Encoding("int8").document()}}],
        outputs=["linear"], provenance={"kind": "test"})
    result = accumulator_bounds(graph)
    assert result["status"] == "rejected"
    assert result["reductions"][0]["maximum_stored_bias_codes"] == 3_000_000_000
