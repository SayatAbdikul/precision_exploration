import json

import pytest

from public.quantization.graph.executable import execute
from tools.run.phase2_engine import graph_witness
from tools.run.phase2_fixed_images import compare_backends, load_checkpoint


def test_interrupted_comparison_reuses_cpp_and_still_checks_cuda(tmp_path):
    graph, inputs = graph_witness()
    context = {"job_sha256": "original", "sample": {"relative_path": "image.jpeg"}}
    path = tmp_path / "checkpoint.json"
    calls = []
    def failing(graph, inputs, *, backend):
        calls.append(backend)
        if backend == "cuda":
            raise RuntimeError("interrupted")
        return execute(graph, inputs)
    with pytest.raises(RuntimeError, match="interrupted"):
        compare_backends(graph, inputs, load_checkpoint(path, context), path, runner=failing)
    assert calls == ["cpp", "cuda"]
    calls.clear()
    def resumed(graph, inputs, *, backend):
        calls.append(backend)
        return execute(graph, inputs)
    result = compare_backends(graph, inputs, load_checkpoint(path, context), path, runner=resumed)
    assert calls == ["cuda"]
    assert result["backends"]["cpp"]["outputs"] == result["backends"]["cuda"]["outputs"]
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(path, {**context, "job_sha256": "changed"})
    record = json.loads(path.read_text())
    record["backends"]["cpp"]["outputs"] = {}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, context)
