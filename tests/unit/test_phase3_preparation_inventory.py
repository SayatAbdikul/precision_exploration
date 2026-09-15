import pytest

from tools.phase3.common import digest, reference, write
from tools.phase3 import preparation_inventory


def prepared(root, name, source):
    config = {"model": "net", "formats": {"activation": {"name": name}}, "runtime": {"source_sha256": source}}
    path = root / f"{name}-{source}.json"
    write(path, config)
    return {"model": "net", "format": name, "configuration": reference(path, root), "configuration_sha256": digest(config)}


def report(root, path, records):
    write(root / path, {"schema_version": "phase3-preparation-1.0.0", "records": records})


def test_combines_workers_and_preserves_current_preparation_over_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation_inventory, "source_identity", lambda: "current")
    stale = prepared(tmp_path, "a", "stale")
    current = prepared(tmp_path, "a", "current")
    other = prepared(tmp_path, "b", "current")
    report(tmp_path, "results/summaries/phase3-preparation.json", {"net/a": stale, "net/b": other})
    report(tmp_path, "artifacts/phase3/preparation/second.json", {"net/a": current, "net/b": {"status": "preparation_failed"}})
    assert preparation_inventory.preparation_records(tmp_path) == {"net/a": current, "net/b": other}


def test_same_configuration_conflicting_evidence_is_not_last_writer_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation_inventory, "source_identity", lambda: "current")
    row = prepared(tmp_path, "a", "current")
    report(tmp_path, "results/summaries/phase3-preparation.json", {"net/a": row})
    report(tmp_path, "artifacts/phase3/preparation/second.json", {"net/a": {**row, "unexpected": "conflict"}})
    with pytest.raises(ValueError, match="conflicting"):
        preparation_inventory.preparation_records(tmp_path)
