from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from public.experiments.registry import ArtifactStore, ExperimentRegistry, RegistryError, experiment_sha256
from public.experiments.scheduler import LocalScheduler
from functools import partial

# Lifecycle tests use an explicit small test schema; production validation is
# exercised separately with real frozen inputs and malformed submissions.
ProductionRegistry = ExperimentRegistry

def validate_test_configuration(value):
    if "value" not in value or set(value) - {"value", "schema_version"}:
        raise ValueError("invalid lifecycle fixture")
    return dict(value)

ExperimentRegistry = partial(ProductionRegistry, validator=validate_test_configuration)


def test_scheduler_lifecycle_dedup_artifacts_and_per_image(tmp_path: Path) -> None:
    configuration = {"schema_version": "test", "value": 7}
    experiment_id = experiment_sha256(configuration)
    database = tmp_path / "results.sqlite"
    with ExperimentRegistry(database) as registry:
        run_id, status = registry.submit(configuration)
        assert status == "PENDING"
        duplicate_id, duplicate_status = registry.submit(configuration)
        assert (duplicate_id, duplicate_status) == (run_id, "PENDING")

        scheduler = LocalScheduler(registry, worker_id="test-worker")
        assert scheduler.run_once(lambda _: {"top1": (71.5, "percent")})
        assert not scheduler.run_once(lambda _: {})

        completed_id, completed_status = registry.submit(configuration)
        assert (completed_id, completed_status) == (run_id, "COMPLETED")

        store = ArtifactStore(tmp_path / "artifacts")
        artifact = store.put_bytes(b"truth-table", semantic_type="datatype_truth_table")
        registry.register_artifact(artifact, producer_experiment_id=experiment_id, metadata={"rows": 1})
        registry.attach_artifact(run_id, artifact.sha256, "conformance")
        assert store.get(artifact.sha256) == b"truth-table"
        dependency = store.put_bytes(b"manifest", semantic_type="datatype_manifest")
        registry.register_artifact(dependency, producer_experiment_id=None, metadata={})
        registry.add_artifact_dependency(artifact.sha256, dependency.sha256, "manifest")
        assert registry.artifact_dependents(dependency.sha256) == [artifact.sha256]

        registry.store_per_image(
            experiment_id,
            [
                {
                    "sample_id": "img-2",
                    "ordinal": 1,
                    "ground_truth": 2,
                    "fp32_prediction": 2,
                    "fp32_correct": True,
                },
                {
                    "sample_id": "img-1",
                    "ordinal": 0,
                    "ground_truth": 1,
                    "fp32_prediction": 0,
                    "fp32_correct": False,
                },
            ],
        )
        assert [row["sample_id"] for row in registry.per_image(experiment_id)] == ["img-1", "img-2"]
        with pytest.raises(sqlite3.IntegrityError):
            registry.store_per_image(
                experiment_id,
                [{"sample_id": "img-1", "ordinal": 3, "ground_truth": 1, "fp32_prediction": 1, "fp32_correct": True}],
            )
        snapshot = registry.export_snapshot()
        assert snapshot["runs"][0]["experiment_id"] == experiment_id
        assert snapshot["artifacts"][0]["sha256"] in {artifact.sha256, dependency.sha256}


def test_failed_attempt_can_be_resubmitted(tmp_path: Path) -> None:
    configuration = {"schema_version": "test", "value": 9}
    with ExperimentRegistry(tmp_path / "results.sqlite") as registry:
        run_id, _ = registry.submit(configuration)
        claimed = registry.claim_next("worker")
        assert claimed["run_id"] == run_id
        registry.finish(run_id, "FAILED", lease_token=claimed["lease_token"], error="expected test failure")
        second_id, status = registry.submit(configuration)
        assert second_id != run_id
        assert status == "PENDING"


def test_illegal_finish_transition_is_rejected(tmp_path: Path) -> None:
    with ExperimentRegistry(tmp_path / "results.sqlite") as registry:
        run_id, _ = registry.submit({"value": 1})
        with pytest.raises(RegistryError, match="not RUNNING"):
            registry.finish(run_id, "COMPLETED", lease_token="not-owned")


def test_stale_job_recovery_and_artifact_corruption(tmp_path: Path) -> None:
    with ExperimentRegistry(tmp_path / "results.sqlite") as registry:
        run_id, _ = registry.submit({"value": "stale"})
        registry.claim_next("lost-worker")
        registry.connection.execute(
            "UPDATE runs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE run_id=?", (run_id,)
        )
        assert registry.recover_stale(older_than_seconds=1) == 1
        replacement = registry.claim_next("replacement")
        assert replacement["run_id"] != run_id
        assert replacement["attempt"] == 2

    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put_bytes(b"original", semantic_type="fixture")
    Path(record.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.get(record.sha256)


def test_atomic_claim_allows_exactly_one_worker(tmp_path: Path) -> None:
    database = tmp_path / "concurrent.sqlite"
    with ExperimentRegistry(database) as registry:
        run_id, _ = registry.submit({"value": "one-job"})

    def claim(worker: str) -> int | None:
        with ExperimentRegistry(database) as registry:
            row = registry.claim_next(worker)
            return int(row["run_id"]) if row is not None else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(claim, ("worker-a", "worker-b")))
    assert sorted(value for value in claimed if value is not None) == [run_id]


def test_hardware_ingestion_is_idempotent_and_rejects_collision(tmp_path: Path) -> None:
    identity = {"pdk": "ics55", "report": "same"}
    metrics = {"cell_area": {"value": 10.0, "unit": "library_area_units"}}
    with ExperimentRegistry(tmp_path / "hardware.sqlite") as registry:
        registry.register_hardware_run(
            "a" * 64,
            experiment_id=None,
            identity=identity,
            metrics=metrics,
            evidence_level="logic_synthesis",
            parser_version="1.1.0",
        )
        registry.register_hardware_run(
            "a" * 64,
            experiment_id=None,
            identity=identity,
            metrics=metrics,
            evidence_level="logic_synthesis",
            parser_version="1.1.0",
        )
        count = registry.connection.execute("SELECT COUNT(*) FROM hardware_runs").fetchone()[0]
        assert count == 1
        store = ArtifactStore(tmp_path / "hardware-artifacts")
        report = store.put_bytes(b"report", semantic_type="hardware_tool_report")
        registry.register_artifact(report, producer_experiment_id=None, metadata={})
        registry.attach_hardware_artifact("a" * 64, report.sha256, "source_report")
        registry.attach_hardware_artifact("a" * 64, report.sha256, "source_report")
        attached = registry.connection.execute("SELECT COUNT(*) FROM hardware_run_artifacts").fetchone()[0]
        assert attached == 1
        with pytest.raises(RegistryError, match="collision"):
            registry.register_hardware_run(
                "a" * 64,
                experiment_id=None,
                identity=identity,
                metrics={"cell_area": {"value": 11.0, "unit": "library_area_units"}},
                evidence_level="logic_synthesis",
                parser_version="1.1.0",
            )


def test_production_scheduler_rejects_invalid_submission(tmp_path):
    with ProductionRegistry(tmp_path / "production.sqlite") as registry:
        run_id, status = registry.submit({"invalid": "no schema/model/dataset"})
        assert status == "INVALID"
        assert not LocalScheduler(registry, worker_id="worker").run_once(lambda _: pytest.fail("invalid job executed"))
        assert registry.connection.execute("SELECT error FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]


def test_recovered_worker_cannot_write_or_finish_new_attempt(tmp_path):
    with ExperimentRegistry(tmp_path / "recovery.sqlite") as registry:
        old_id, _ = registry.submit({"value": "recovery"})
        old = registry.claim_next("old")
        registry.connection.execute("UPDATE runs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE run_id=?", (old_id,))
        assert registry.recover_stale(older_than_seconds=1) == 1
        new = registry.claim_next("new")
        for run_id in (old_id, new["run_id"]):
            with pytest.raises(RegistryError):
                registry.finish(run_id, "COMPLETED", lease_token=old["lease_token"])
            with pytest.raises(RegistryError):
                registry.add_metrics(run_id, {"wrong": (1, "count")}, lease_token=old["lease_token"])
        registry.finish(new["run_id"], "COMPLETED", lease_token=new["lease_token"], metrics={"right": (2, "count")})
        assert registry.connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 1
        assert registry.connection.execute("SELECT status FROM runs WHERE run_id=?", (old_id,)).fetchone()[0] == "FAILED"


def test_per_image_batch_is_atomic_and_identical_retries_are_idempotent(tmp_path):
    with ExperimentRegistry(tmp_path / "rows.sqlite") as registry:
        config = {"value": "rows"}
        registry.submit(config)
        identity = experiment_sha256(config)
        a = {"sample_id": "a", "ordinal": 0, "ground_truth": 1, "fp32_prediction": 1}
        registry.store_per_image(identity, [a])
        registry.store_per_image(identity, [a])
        with pytest.raises(sqlite3.IntegrityError):
            registry.store_per_image(identity, [{**a, "sample_id": "b", "ordinal": 1}, {**a, "fp32_prediction": 2}])
        assert [row["sample_id"] for row in registry.per_image(identity)] == ["a"]


def test_dependency_reuse_checks_hashes_and_exact_dependency_set(tmp_path):
    with ExperimentRegistry(tmp_path / "reuse.sqlite") as registry:
        store = ArtifactStore(tmp_path / "artifacts")
        source = store.put_bytes(b"source", semantic_type="manifest")
        output = store.put_bytes(b"result", semantic_type="truth_table")
        for record in (source, output):
            registry.register_artifact(record, producer_experiment_id=None, metadata={})
        registry.add_artifact_dependency(output.sha256, source.sha256, "source")
        assert registry.reuse_artifact(output.sha256, dependencies={"source": source.sha256}) == output
        with pytest.raises(RegistryError):
            registry.reuse_artifact(output.sha256, dependencies={"source": "0" * 64})
        Path(source.path).write_bytes(b"corrupt")
        with pytest.raises(RegistryError):
            registry.reuse_artifact(output.sha256, dependencies={"source": source.sha256})


def test_long_executor_renews_lease(tmp_path):
    import threading
    with ExperimentRegistry(tmp_path / "heartbeat.sqlite") as registry:
        run_id, _ = registry.submit({"value": "long"})
        observed = threading.Event()
        def execute(config):
            before = registry.connection.execute("SELECT heartbeat_at FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
            for _ in range(50):
                observed.wait(0.01)
                after = registry.connection.execute("SELECT heartbeat_at FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
                if after != before:
                    observed.set()
                    break
            assert observed.is_set()
            return {"ok": (1, "count")}
        LocalScheduler(registry, worker_id="renewing", heartbeat_seconds=0.01).run_once(execute)
        assert registry.connection.execute("SELECT status FROM runs").fetchone()[0] == "COMPLETED"


def test_invalidated_completed_config_is_not_reused(tmp_path):
    invalid = False
    def validate(config):
        if invalid: raise ValueError("input changed")
        return config
    with ProductionRegistry(tmp_path / "invalidate.sqlite", validator=validate) as registry:
        config = {"value": "input"}
        registry.submit(config)
        LocalScheduler(registry, worker_id="w").run_once(lambda _: {})
        invalid = True
        _, status = registry.submit(config)
        assert status == "INVALID"
