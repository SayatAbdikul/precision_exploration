from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_fp32_baseline_metrics_and_prediction_hashes() -> None:
    aggregate = json.loads((ROOT / "results/summaries/phase1-fp32-baselines.json").read_text())
    expected = {
        "resnet18": (1000, 70.1, 89.5, "30cac37513758447c4f1c7c4c0ca76ec82b3c945e408643bef6776094a005ebe"),
        "mobilenet_v2": (1000, 72.1, 91.1, "da9c78e8a080f4f0ff511cf18c13bcd00f1a09c05d88741fbdd2c02c96245e5f"),
        "mobilenet_v3_large": (1000, 75.0, 92.1, "44161f306d2cbc409653e724688b0fc9aad5a014e80ccf1798c830d71e7e2007"),
    }
    for run in aggregate["runs"]:
        if run["model"] in expected:
            count, top1, top5, prediction_hash = expected[run["model"]]
            assert (run["sample_count"], run["top1_percent"], run["top5_percent"], run["prediction_sha256"]) == (
                count, top1, top5, prediction_hash
            )
            path = ROOT / f"artifacts/per_image_predictions/{run['model']}_imagenet1k_screen.jsonl"
            assert sha256(path) == prediction_hash
        else:
            assert run["model"] == "yolov8n"
            assert run["sample_count"] == 5000
            assert run["map50_95"] == 0.3735112134743026
            assert run["map50"] == 0.5256168716683333
            assert sha256(ROOT / "artifacts/per_image_predictions/yolov8n_coco2017_val5k_coco_predictions.json") == run["prediction_sha256"]


def test_truth_table_index_resolves_every_content_hash() -> None:
    index = json.loads((ROOT / "public/formats/conformance/truth-table-index.json").read_text())
    assert index["accepted_manifest_set_sha256"] == "c859204f7e1ec3f1aa4a5b7381d811add705bf2ca6cb89dc1ee7bdadfd922e87"
    assert len(index["records"]) == 725
    assert sum(record["row_count"] for record in index["records"]) == 1_085_846
    for record in index["records"]:
        path = ROOT / record["artifact_path"]
        assert sha256(path) == record["sha256"]
        metadata = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())
        assert metadata["sha256"] == record["sha256"]
        assert metadata["completeness"] == "exhaustive"


def test_ics55_and_registry_evidence_remain_resolvable() -> None:
    hardware = json.loads((ROOT / "results/summaries/ics55-phase1-pilot.json").read_text())
    assert hardware["availability"]["status"] == "public_preview_usable"
    assert hardware["pdk"]["version"] == "v1.10.102"
    assert hardware["repeatability"]["stable_outputs"]["netlist.v"]["identical"] is True
    timing = {run["identity"]["target_clock"]: run["metrics"] for run in hardware["timing_sweep"]}
    assert timing["5ns"]["timing_met"]["value"] == 1
    assert timing["2ns"]["timing_met"]["value"] == 0
    assert timing["10ns"]["fmax"] == {"value": 491.16, "unit": "MHz"}

    export = json.loads((ROOT / "results/summaries/phase1-registry-export.json").read_text())
    snapshot = export["snapshot"]
    assert len(snapshot["configurations"]) == 4
    assert len(snapshot["hardware_runs"]) == 4
    assert sorted(row["count"] for row in snapshot["per_image_counts"]) == [1000, 1000, 1000, 5000]
