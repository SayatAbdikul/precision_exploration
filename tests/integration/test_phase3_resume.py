"""Fault injection uses the real image loop, diagnostics, and SQLite lifecycle."""
import fcntl

import pytest
import torch

from public.experiments.registry import ExperimentRegistry
from public.inference.tensor import Encoding
from public.quantization.graph.executable import freeze_graph, execute
from tools.phase3.common import digest, read, reference, write
from tools.phase3.jobs import run_registered
from tools.phase3 import screening


def test_failed_backend_resumes_without_reexecuting_saved_work(tmp_path, monkeypatch):
    root = tmp_path
    graph = freeze_graph(inputs={"x": Encoding("int8").document()}, constants={},
        nodes=[{"name": "relu", "op": "activation", "inputs": ["x"],
                "attrs": {"function": "relu", "accumulator": "int32_accumulator", "output": Encoding("int8").document()}}],
        outputs=["relu"], provenance={"kind": "fault_injection"})
    config = {"model": "resnet18", "formats": {"activation": {"name": "int8"}}}
    write(root / "config.json", config)
    write(root / "graph.json", graph)
    write(root / "prepared.json", {"configuration": reference(root / "config.json", root),
          "configuration_sha256": digest(config), "graph": reference(root / "graph.json", root)})
    population = [{"sha256": str(i)*64, "relative_path": f"{i}.png"} for i in (1, 2)]
    baseline = [{"sample_id": str(i), "ordinal": i-1, "sample_sha256": str(i)*64,
                 "ground_truth": 4, "fp32_prediction": [4, 3, 2, 1, 0]} for i in (1, 2)]
    write(root / "baseline.json", {"records": baseline})
    job = {"prepared": reference(root / "prepared.json", root), "baseline": reference(root / "baseline.json", root),
           "images": 2, "scope": "pilot", "backends": ["reference", "cpp"], "source_sha256": "source", "pipeline_sha256": "pipeline"}
    # Synthetic fixture replaces only external identities/runtime loading. The
    # real execution loop, quantization observer, checkpoint and registry remain.
    monkeypatch.setattr(screening, "validate_job", lambda value, root: value)
    monkeypatch.setattr(screening, "campaign", lambda root: {"diagnostics": {"sample_elements_per_layer_image": 5}})
    monkeypatch.setattr(screening, "screen_rows", lambda *args: (population, root, None))
    monkeypatch.setattr(screening, "source_identity", lambda: "source")
    monkeypatch.setattr(screening, "pipeline_identity", lambda root: "pipeline")
    def observe(value, references):
        references.record("relu", value.relu())
    monkeypatch.setattr(screening, "load_runtime", lambda *args: (
        lambda path: (torch.tensor([[-1., 1., 2., 3., 4.]]), None), observe, {"input_shape": [1, 5]}))
    calls, failed = [], False
    def interrupted(graph, inputs, *, backend, observer):
        nonlocal failed
        calls.append(backend)
        if backend == "cpp" and not failed:
            failed = True
            raise RuntimeError("injected backend interruption")
        return execute(graph, inputs, backend="reference", observer=observer)
    monkeypatch.setattr(screening, "execute", interrupted)
    kwargs = {"root": root, "validator": lambda value: value, "executor": lambda value: screening.run_job(value, root)}
    first = run_registered(job, **kwargs)
    assert first["status"] == "FAILED" and "injected" in first["error"]
    record_path = root / "artifacts/phase3/runs" / digest(job) / ("1"*64 + ".json")
    assert set(read(record_path)["backends"]) == {"reference"}
    second = run_registered(job, **kwargs)
    assert second["status"] == "COMPLETED"
    assert calls == ["reference", "cpp", "cpp", "reference", "cpp"]
    assert len(read(record_path.parent / "paired.json")) == 2
    run_registered(job, **kwargs)
    assert len(calls) == 5  # completed job deduplication
    with ExperimentRegistry(root / "results/databases/phase3.sqlite", validator=lambda value: value) as registry:
        assert [r[0] for r in registry.connection.execute("SELECT status FROM runs ORDER BY run_id")] == ["FAILED", "COMPLETED"]
        assert len(registry.per_image(digest(job))) == 2


def test_expired_lease_recovery_preserves_failed_attempt(tmp_path):
    job = {"test": "recovery"}
    with ExperimentRegistry(tmp_path / "results/databases/phase3.sqlite", validator=lambda value: value) as registry:
        run_id, _ = registry.submit(job)
        registry.claim_next("lost-worker")
        registry.connection.execute("UPDATE runs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE run_id=?", (run_id,))
    row = run_registered(job, root=tmp_path, recover_stale_seconds=300,
                         validator=lambda value: value, executor=lambda value: {"completed": (1, "boolean")})
    assert row["status"] == "COMPLETED" and row["attempt"] == 2
    with ExperimentRegistry(tmp_path / "results/databases/phase3.sqlite", validator=lambda value: value) as registry:
        old = registry.connection.execute("SELECT status,error FROM runs WHERE run_id=?", (run_id,)).fetchone()
        assert tuple(old) == ("FAILED", "worker lease expired")


def test_live_writer_cannot_be_recovered_or_duplicated(tmp_path):
    lock = tmp_path / "artifacts/phase3/locks/native-worker.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="live local writer"):
            run_registered({"test": "lock"}, root=tmp_path, recover_stale_seconds=300, validator=lambda value: value)


def test_cpu_pilot_does_not_claim_pending_gpu_work_or_require_gpu_lock(tmp_path):
    gpu_job, cpu_job = {"backends": ["cuda"], "test": "gpu"}, {"backends": ["cpp"], "test": "cpu"}
    with ExperimentRegistry(tmp_path / "results/databases/phase3.sqlite", validator=lambda value: value) as registry:
        gpu_id, _ = registry.submit(gpu_job)
    lock = tmp_path / "artifacts/phase3/locks/native-worker.lock"
    lock.parent.mkdir(parents=True)
    executed = []
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_registered(cpu_job, root=tmp_path, validator=lambda value: value,
                                executor=lambda value: executed.append(value) or {})
    assert result["status"] == "COMPLETED" and executed == [cpu_job]
    with ExperimentRegistry(tmp_path / "results/databases/phase3.sqlite", validator=lambda value: value) as registry:
        assert registry.connection.execute("SELECT status FROM runs WHERE run_id=?", (gpu_id,)).fetchone()[0] == "PENDING"


def test_cpu_recovery_cannot_expire_another_job_and_cpu_lock_prevents_duplicate(tmp_path):
    gpu_job, cpu_job = {"backends": ["cuda"], "test": "gpu"}, {"backends": ["cpp"], "test": "cpu"}
    with ExperimentRegistry(tmp_path / "results/databases/phase3.sqlite", validator=lambda value: value) as registry:
        gpu_id, _ = registry.submit(gpu_job)
        registry.claim_next("gpu", run_id=gpu_id)
        registry.connection.execute("UPDATE runs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE run_id=?", (gpu_id,))
    result = run_registered(cpu_job, root=tmp_path, recover_stale_seconds=300,
                            validator=lambda value: value, executor=lambda value: {})
    assert result["status"] == "COMPLETED"
    with ExperimentRegistry(tmp_path / "results/databases/phase3.sqlite", validator=lambda value: value) as registry:
        assert registry.connection.execute("SELECT status FROM runs WHERE run_id=?", (gpu_id,)).fetchone()[0] == "RUNNING"
    lock = tmp_path / "artifacts/phase3/locks/cpu-native-worker.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="live local writer"):
            run_registered(cpu_job, root=tmp_path, validator=lambda value: value)
