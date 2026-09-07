from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from public.experiments.registry import ArtifactStore, ExperimentRegistry, RegistryError, experiment_sha256
from public.experiments.scheduler import LocalScheduler


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
        registry.finish(run_id, "FAILED", error="expected test failure")
        second_id, status = registry.submit(configuration)
        assert second_id != run_id
        assert status == "PENDING"


def test_illegal_finish_transition_is_rejected(tmp_path: Path) -> None:
    with ExperimentRegistry(tmp_path / "results.sqlite") as registry:
        run_id, _ = registry.submit({"value": 1})
        with pytest.raises(RegistryError, match="not RUNNING"):
            registry.finish(run_id, "COMPLETED")


def test_stale_job_recovery_and_artifact_corruption(tmp_path: Path) -> None:
    with ExperimentRegistry(tmp_path / "results.sqlite") as registry:
        run_id, _ = registry.submit({"value": "stale"})
        registry.claim_next("lost-worker")
        registry.connection.execute(
            "UPDATE runs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE run_id=?", (run_id,)
        )
        assert registry.recover_stale(older_than_seconds=1) == 1
        assert registry.claim_next("replacement")["run_id"] == run_id

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
