#!/usr/bin/env python3
"""Read-only validation of the complete Phase 1 evidence chain."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from public.experiments.registry.identity import content_sha256, validate_submission
from public.formats.oracle.manifest import manifest_sha256, load_manifest
from public.workloads.datasets.identity import verify_phase1_subsets, verify_payload_record
from public.workloads.models.identity import verify_model_manifest

ROOT = Path(__file__).resolve().parents[2]


def verify() -> dict:
    def read(relative):
        return json.loads((ROOT / relative).read_text())
    import re
    frozen = re.findall(r"\| `([^`]+)` \| `([a-f0-9]{64})` \|", (ROOT / "docs/roadmap/phases/phase-00-contracts.md").read_text())
    assert len(frozen) == 10
    for relative, expected in frozen:
        assert content_sha256(ROOT / relative) == expected
    accepted = read("public/formats/manifests/accepted/index.json")
    for row in accepted["manifests"]:
        assert manifest_sha256(load_manifest(ROOT / row["path"])) == row["sha256"]
    aggregate = hashlib.sha256("".join(f"{r['name']}:{r['sha256']}\n" for r in accepted["manifests"]).encode()).hexdigest()
    assert aggregate == accepted["aggregate_sha256"]
    tables = read("public/formats/conformance/truth-table-index.json")
    assert tables["accepted_manifest_set_sha256"] == aggregate
    for row in tables["records"]:
        path = ROOT / row["artifact_path"]
        assert content_sha256(path) == row["sha256"]
        with path.open() as stream:
            assert sum(1 for _ in stream) == row["row_count"]
        assert row["oracle_version"] == "1.1.0"
    subsets = verify_phase1_subsets(ROOT)
    references = sum(verify_payload_record(ROOT, record) for record in read("data/manifests/index.json")["records"].values())
    for row in read("public/workloads/models/manifests/index.json")["manifests"]:
        verify_model_manifest(ROOT / row["path"], repository_root=ROOT)
    witness = read("public/experiments/configs/phase1_reference_witness.json")
    validate_submission(witness, repository_root=ROOT)
    for folder in ("ics55-pilot", "ics55-pilot-repeat"):
        manifest = read(f"artifacts/ppa/{folder}/run-manifest.json")
        assert content_sha256(ROOT / manifest["config"]["rtl"]) == manifest["rtl_sha256"]
        for name, row in manifest["outputs"].items():
            assert content_sha256(ROOT / f"artifacts/ppa/{folder}/{name}") == row["sha256"]
    database = ROOT / "results/databases/phase1.sqlite"
    con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not list(con.execute("PRAGMA foreign_key_check"))
        for row in con.execute("SELECT * FROM artifacts"):
            path = ROOT / row["path"]
            assert content_sha256(path) == row["sha256"] and path.stat().st_size == row["size_bytes"]
        from public.experiments.registry.identity import experiment_sha256
        for row in con.execute("SELECT * FROM configurations"):
            config = json.loads(row["canonical_json"])
            assert experiment_sha256(config) == row["experiment_id"]
            validate_submission(config, repository_root=ROOT)
        # Reuse the read-only connection to call the deterministic exporter;
        # do not initialize or mutate the authoritative database.
        from public.experiments.registry.database import ExperimentRegistry
        view = object.__new__(ExperimentRegistry)
        view.connection = con
        assert view.export_snapshot() == read("results/summaries/phase1-registry-export.json")["snapshot"]
        per_sample = con.execute("SELECT COUNT(*) FROM per_image_results").fetchone()[0]
        for row in con.execute("SELECT metrics_json FROM hardware_runs"):
            import math
            assert all(math.isfinite(float(m["value"])) for m in json.loads(row["metrics_json"]).values())
    finally:
        con.close()
    return {"status": "passed", "manifest_count": len(accepted["manifests"]), "manifest_set_sha256": aggregate,
            "table_count": len(tables["records"]), "table_rows": sum(r["row_count"] for r in tables["records"]),
            "dataset_references_verified": references, "dataset_counts": subsets["counts"],
            "per_sample_rows": per_sample, "database_integrity": "ok", "export_matches_database": True, "phase0_frozen_hashes_verified": len(frozen)}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True, indent=2))
