"""SQLite experiment registry and lifecycle implementation."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Callable

from .artifacts import ArtifactRecord
from .identity import canonical_json_bytes, experiment_sha256, validate_submission, IdentityError, content_sha256


VALID_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED", "INVALID"})
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "INVALID"})


class RegistryError(RuntimeError):
    """Raised for an invalid registry transition or conflicting record."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentRegistry:
    def __init__(self, path: str | Path, *, validator: Callable = validate_submission):
        self.validator = validator
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ExperimentRegistry":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS configurations (
                experiment_id TEXT PRIMARY KEY,
                canonical_json BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES configurations(experiment_id),
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETED','FAILED','INVALID')),
                worker_id TEXT,
                lease_token TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                heartbeat_at TEXT,
                finished_at TEXT,
                error TEXT,
                UNIQUE(experiment_id, attempt)
            );
            CREATE INDEX IF NOT EXISTS runs_status_index ON runs(status, run_id);
            CREATE TABLE IF NOT EXISTS artifacts (
                sha256 TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                semantic_type TEXT NOT NULL,
                producer_experiment_id TEXT REFERENCES configurations(experiment_id),
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_artifacts (
                run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
                role TEXT NOT NULL,
                PRIMARY KEY(run_id, sha256, role)
            );
            CREATE TABLE IF NOT EXISTS artifact_dependencies (
                artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
                dependency_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
                dependency_role TEXT NOT NULL,
                PRIMARY KEY(artifact_sha256, dependency_sha256, dependency_role),
                CHECK(artifact_sha256 != dependency_sha256)
            );
            CREATE TABLE IF NOT EXISTS metrics (
                run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                PRIMARY KEY(run_id, name)
            );
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY(run_id, key)
            );
            CREATE TABLE IF NOT EXISTS per_image_results (
                experiment_id TEXT NOT NULL REFERENCES configurations(experiment_id),
                sample_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                ground_truth TEXT NOT NULL,
                fp32_prediction TEXT NOT NULL,
                candidate_prediction TEXT,
                fp32_correct INTEGER CHECK(fp32_correct IN (0,1)),
                candidate_correct INTEGER CHECK(candidate_correct IN (0,1)),
                summary_json TEXT NOT NULL,
                PRIMARY KEY(experiment_id, sample_id),
                UNIQUE(experiment_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS hardware_runs (
                hardware_run_id TEXT PRIMARY KEY,
                experiment_id TEXT REFERENCES configurations(experiment_id),
                identity_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                evidence_level TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hardware_run_artifacts (
                hardware_run_id TEXT NOT NULL REFERENCES hardware_runs(hardware_run_id) ON DELETE CASCADE,
                sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
                role TEXT NOT NULL,
                PRIMARY KEY(hardware_run_id, sha256, role)
            );
            """
        )

        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(runs)")}
        if "lease_token" not in columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN lease_token TEXT")

    def submit(self, configuration: Mapping[str, Any], *, invalid_error: str | None = None) -> tuple[int, str]:
        try:
            configuration = self.validator(configuration)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            invalid_error = str(exc)
        experiment_id = experiment_sha256(configuration)
        canonical = canonical_json_bytes(configuration, derived_field="experiment_id")
        now = _now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute(
                "SELECT canonical_json FROM configurations WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if existing is not None and bytes(existing["canonical_json"]) != canonical:
                raise RegistryError(f"SHA-256 collision or corrupt canonical configuration: {experiment_id}")
            self.connection.execute(
                "INSERT OR IGNORE INTO configurations(experiment_id, canonical_json, created_at) VALUES(?,?,?)",
                (experiment_id, canonical, now),
            )
            completed = self.connection.execute(
                "SELECT run_id FROM runs WHERE experiment_id = ? AND status = 'COMPLETED' ORDER BY attempt LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if completed is not None and invalid_error is None:
                self.connection.execute("COMMIT")
                return int(completed["run_id"]), "COMPLETED"
            active = self.connection.execute(
                "SELECT run_id, status FROM runs WHERE experiment_id = ? AND status IN ('PENDING','RUNNING') ORDER BY attempt DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if active is not None and invalid_error is None:
                self.connection.execute("COMMIT")
                return int(active["run_id"]), str(active["status"])
            next_attempt = self.connection.execute(
                "SELECT COALESCE(MAX(attempt), 0) + 1 FROM runs WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()[0]
            status = "INVALID" if invalid_error else "PENDING"
            cursor = self.connection.execute(
                """INSERT INTO runs(
                    experiment_id, attempt, status, created_at, finished_at, error
                ) VALUES(?,?,?,?,?,?)""",
                (experiment_id, next_attempt, status, now, now if invalid_error else None, invalid_error),
            )
            run_id = int(cursor.lastrowid)
            self.connection.execute("COMMIT")
            return run_id, status
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def claim_next(self, worker_id: str) -> sqlite3.Row | None:
        now = _now()
        lease_token = uuid.uuid4().hex
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT run_id FROM runs WHERE status = 'PENDING' ORDER BY run_id LIMIT 1"
            ).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            run_id = int(row["run_id"])
            changed = self.connection.execute(
                """UPDATE runs SET status='RUNNING', worker_id=?, started_at=?, heartbeat_at=?, lease_token=?
                   WHERE run_id=? AND status='PENDING'""",
                (worker_id, now, now, lease_token, run_id),
            ).rowcount
            if changed != 1:
                raise RegistryError(f"could not atomically claim run {run_id}")
            claimed = self.connection.execute(
                """SELECT r.*, c.canonical_json FROM runs r
                   JOIN configurations c USING(experiment_id) WHERE run_id=?""",
                (run_id,),
            ).fetchone()
            self.connection.execute("COMMIT")
            return claimed
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def heartbeat(self, run_id: int, worker_id: str, *, lease_token: str) -> None:
        changed = self.connection.execute(
            "UPDATE runs SET heartbeat_at=? WHERE run_id=? AND status='RUNNING' AND worker_id=? AND lease_token=?",
            (_now(), run_id, worker_id, lease_token),
        ).rowcount
        if changed != 1:
            raise RegistryError(f"run {run_id} lease is no longer owned by {worker_id}")

    def _owned(self, run_id: int, lease_token: str) -> None:
        row = self.connection.execute(
            "SELECT 1 FROM runs WHERE run_id=? AND status='RUNNING' AND lease_token=?", (run_id, lease_token)
        ).fetchone()
        if row is None:
            raise RegistryError(f"run {run_id} is not RUNNING with this lease")

    def finish(self, run_id: int, status: str, *, lease_token: str, error: str | None = None,
               metrics: Mapping[str, tuple[float, str]] | None = None) -> None:
        if status not in TERMINAL_STATUSES:
            raise RegistryError(f"finish requires a terminal status, got {status}")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._owned(run_id, lease_token)
            if metrics is not None:
                self._write_metrics(run_id, metrics)
            self.connection.execute(
                "UPDATE runs SET status=?, finished_at=?, heartbeat_at=?, error=? WHERE run_id=?",
                (status, _now(), _now(), error, run_id),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def recover_stale(self, *, older_than_seconds: int) -> int:
        if older_than_seconds <= 0:
            raise RegistryError("stale timeout must be positive")
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)).isoformat()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            stale = list(self.connection.execute(
                "SELECT * FROM runs WHERE status='RUNNING' AND heartbeat_at < ?", (cutoff,)))
            for row in stale:
                self.connection.execute(
                    "UPDATE runs SET status='FAILED',finished_at=?,error='worker lease expired' WHERE run_id=?",
                    (_now(), row["run_id"]))
                self.connection.execute(
                    "INSERT INTO runs(experiment_id,attempt,status,created_at) VALUES(?,?,'PENDING',?)",
                    (row["experiment_id"], self.connection.execute(
                        "SELECT MAX(attempt)+1 FROM runs WHERE experiment_id=?", (row["experiment_id"],)).fetchone()[0], _now()))
            self.connection.execute("COMMIT")
            return len(stale)
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def _write_metrics(self, run_id: int, metrics: Mapping[str, tuple[float, str]]) -> None:
        import math
        if any(not math.isfinite(float(value)) for value, _ in metrics.values()):
            raise RegistryError("metrics must be finite")
        self.connection.executemany(
            "INSERT OR REPLACE INTO metrics(run_id,name,value,unit) VALUES(?,?,?,?)",
            ((run_id, name, float(value), unit) for name, (value, unit) in metrics.items()),
        )

    def add_metrics(self, run_id: int, metrics: Mapping[str, tuple[float, str]], *, lease_token: str) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._owned(run_id, lease_token)
            self._write_metrics(run_id, metrics)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def add_run_metadata(self, run_id: int, metadata: Mapping[str, Any]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO run_metadata(run_id,key,value_json) VALUES(?,?,?)",
            (
                (run_id, str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
                for key, value in metadata.items()
            ),
        )

    def register_artifact(
        self,
        record: ArtifactRecord,
        *,
        producer_experiment_id: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        path = Path(record.path)
        if not path.is_file() or path.stat().st_size != record.size_bytes or content_sha256(path) != record.sha256:
            raise RegistryError("artifact publication requires verified bytes and size")
        encoded = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.connection.execute(
            """INSERT INTO artifacts(
                sha256,path,size_bytes,semantic_type,producer_experiment_id,metadata_json,created_at
            ) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(sha256) DO UPDATE SET
                path=excluded.path,
                size_bytes=excluded.size_bytes,
                semantic_type=excluded.semantic_type""",
            (
                record.sha256,
                record.path,
                record.size_bytes,
                record.semantic_type,
                producer_experiment_id,
                encoded,
                _now(),
            ),
        )

    def attach_artifact(self, run_id: int, sha256: str, role: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO run_artifacts(run_id,sha256,role) VALUES(?,?,?)",
            (run_id, sha256, role),
        )

    def add_artifact_dependency(self, artifact_sha256: str, dependency_sha256: str, role: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO artifact_dependencies(artifact_sha256,dependency_sha256,dependency_role) VALUES(?,?,?)",
            (artifact_sha256, dependency_sha256, role),
        )

    def artifact_dependents(self, dependency_sha256: str) -> list[str]:
        return [
            str(row["artifact_sha256"])
            for row in self.connection.execute(
                "SELECT artifact_sha256 FROM artifact_dependencies WHERE dependency_sha256=? ORDER BY artifact_sha256",
                (dependency_sha256,),
            )
        ]

    def store_per_image(self, experiment_id: str, rows: Iterable[Mapping[str, Any]]) -> None:
        encoded_rows = []
        for row in rows:
            encoded_rows.append(
                (
                    experiment_id,
                    str(row["sample_id"]),
                    int(row["ordinal"]),
                    json.dumps(row["ground_truth"], sort_keys=True),
                    json.dumps(row["fp32_prediction"], sort_keys=True),
                    json.dumps(row.get("candidate_prediction"), sort_keys=True)
                    if row.get("candidate_prediction") is not None
                    else None,
                    int(bool(row["fp32_correct"])) if row.get("fp32_correct") is not None else None,
                    int(bool(row["candidate_correct"])) if row.get("candidate_correct") is not None else None,
                    json.dumps(row.get("summary", {}), sort_keys=True, separators=(",", ":")),
                )
            )
        self.connection.execute("SAVEPOINT per_image_batch")
        try:
            for values in encoded_rows:
                existing = self.connection.execute(
                    "SELECT * FROM per_image_results WHERE experiment_id=? AND sample_id=?", values[:2]).fetchone()
                if existing is not None:
                    if tuple(existing) != values:
                        raise sqlite3.IntegrityError("conflicting per-image result")
                    continue
                self.connection.execute(
                    "INSERT INTO per_image_results VALUES(?,?,?,?,?,?,?,?,?)", values)
            self.connection.execute("RELEASE per_image_batch")
        except Exception:
            self.connection.execute("ROLLBACK TO per_image_batch")
            self.connection.execute("RELEASE per_image_batch")
            raise

    def reuse_artifact(self, sha256: str, *, dependencies: Mapping[str, str]) -> ArtifactRecord:
        """Reuse only an exact dependency set, verifying the payload and inputs."""
        rows = list(self.connection.execute(
            "SELECT dependency_role,dependency_sha256 FROM artifact_dependencies WHERE artifact_sha256=?", (sha256,)))
        expected = {(role, digest) for role, digest in dependencies.items()}
        if not expected or {(r[0], r[1]) for r in rows} != expected:
            raise RegistryError("artifact dependencies differ; recomputation required")
        records = []
        for digest in [sha256, *dependencies.values()]:
            row = self.connection.execute("SELECT * FROM artifacts WHERE sha256=?", (digest,)).fetchone()
            if row is None or content_sha256(row["path"]) != digest:
                raise RegistryError("artifact or dependency payload missing/corrupt")
            records.append(row)
        row = records[0]
        return ArtifactRecord(row["sha256"], row["path"], row["size_bytes"], row["semantic_type"])

    def per_image(self, experiment_id: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM per_image_results WHERE experiment_id=? ORDER BY ordinal", (experiment_id,)
            )
        )

    def register_hardware_run(
        self,
        hardware_run_id: str,
        *,
        experiment_id: str | None,
        identity: Mapping[str, Any],
        metrics: Mapping[str, Any],
        evidence_level: str,
        parser_version: str,
    ) -> None:
        identity_json = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
        metrics_json = json.dumps(dict(metrics), sort_keys=True, separators=(",", ":"), allow_nan=False)
        existing = self.connection.execute(
            "SELECT identity_json,metrics_json,experiment_id,evidence_level,parser_version FROM hardware_runs WHERE hardware_run_id=?",
            (hardware_run_id,),
        ).fetchone()
        if existing is not None and (
            existing["identity_json"] != identity_json or existing["metrics_json"] != metrics_json
            or existing["experiment_id"] != experiment_id or existing["evidence_level"] != evidence_level
            or existing["parser_version"] != parser_version
        ):
            raise RegistryError(f"hardware run identity collision: {hardware_run_id}")
        self.connection.execute(
            """INSERT OR IGNORE INTO hardware_runs(
                hardware_run_id,experiment_id,identity_json,metrics_json,evidence_level,parser_version,created_at
            ) VALUES(?,?,?,?,?,?,?)""",
            (
                hardware_run_id,
                experiment_id,
                identity_json,
                metrics_json,
                evidence_level,
                parser_version,
                _now(),
            ),
        )

    def attach_hardware_artifact(self, hardware_run_id: str, sha256: str, role: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO hardware_run_artifacts(hardware_run_id,sha256,role) VALUES(?,?,?)",
            (hardware_run_id, sha256, role),
        )

    def export_snapshot(self) -> dict[str, Any]:
        """Return a deterministic compact export tied to authoritative IDs."""

        def rows(query: str) -> list[dict[str, Any]]:
            return [dict(row) for row in self.connection.execute(query)]

        return {
            "schema_version": "1.0.0",
            "configurations": rows("SELECT experiment_id FROM configurations ORDER BY experiment_id"),
            "runs": rows("SELECT run_id,experiment_id,attempt,status,worker_id,error FROM runs ORDER BY run_id"),
            "artifacts": rows("SELECT sha256,size_bytes,semantic_type,producer_experiment_id FROM artifacts ORDER BY sha256"),
            "run_artifacts": rows("SELECT run_id,sha256,role FROM run_artifacts ORDER BY run_id,sha256,role"),
            "artifact_dependencies": rows("SELECT artifact_sha256,dependency_sha256,dependency_role FROM artifact_dependencies ORDER BY artifact_sha256,dependency_sha256,dependency_role"),
            "metrics": rows("SELECT run_id,name,value,unit FROM metrics ORDER BY run_id,name"),
            "run_metadata": rows("SELECT run_id,key,value_json FROM run_metadata ORDER BY run_id,key"),
            "per_image_counts": rows("SELECT experiment_id,COUNT(*) AS count FROM per_image_results GROUP BY experiment_id ORDER BY experiment_id"),
            "hardware_runs": rows("SELECT hardware_run_id,experiment_id,evidence_level,parser_version FROM hardware_runs ORDER BY hardware_run_id"),
            "hardware_run_artifacts": rows("SELECT hardware_run_id,sha256,role FROM hardware_run_artifacts ORDER BY hardware_run_id,sha256,role"),
        }
