"""Verification for the tracked Phase 1 dataset manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


class DatasetManifestError(ValueError):
    pass


def load_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            raise DatasetManifestError(f"missing TSV header: {path}")
        rows = list(reader)
    if not rows:
        raise DatasetManifestError(f"empty dataset manifest: {path}")
    return rows


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique(rows: list[dict[str, str]], key: str, name: str) -> set[str]:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise DatasetManifestError(f"duplicate {key} in {name}")
    return set(values)


def verify_phase1_subsets(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    index_path = root / "data/manifests/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    loaded: dict[str, list[dict[str, str]]] = {}
    for name, record in index["records"].items():
        path = root / record["path"]
        if sha256(path) != record["sha256"]:
            raise DatasetManifestError(f"manifest hash drift: {name}")
        rows = load_tsv(path)
        if len(rows) != record["count"]:
            raise DatasetManifestError(f"manifest count drift: {name}")
        selection = record.get("selection_record")
        if not isinstance(selection, dict):
            raise DatasetManifestError(f"missing selection record: {name}")
        selection_path = root / selection["path"]
        if sha256(selection_path) != selection["sha256"]:
            raise DatasetManifestError(f"selection-record hash drift: {name}")
        selection_value = json.loads(selection_path.read_text(encoding="utf-8"))
        for field in ("dataset", "image_count", "seed", "selection", "split"):
            if field not in selection_value:
                raise DatasetManifestError(f"selection record {name} lacks {field}")
        if name.startswith("coco_") and not {"strata_population", "strata_selection"}.issubset(selection_value):
            raise DatasetManifestError(f"COCO stratification summary is incomplete: {name}")
        loaded[name] = rows

    for name in ("imagenet_calibration_2k", "imagenet_screen_1k", "imagenet_evaluation_10k"):
        _unique(loaded[name], "sha256", name)
        counts = Counter(int(row["label"]) for row in loaded[name])
        expected = {"imagenet_calibration_2k": 2, "imagenet_screen_1k": 1, "imagenet_evaluation_10k": 10}[name]
        if set(counts) != set(range(1000)) or set(counts.values()) != {expected}:
            raise DatasetManifestError(f"ImageNet class balance failure: {name}")
    calibration = {row["sha256"] for row in loaded["imagenet_calibration_2k"]}
    screen = {row["sha256"] for row in loaded["imagenet_screen_1k"]}
    evaluation = {row["sha256"] for row in loaded["imagenet_evaluation_10k"]}
    if calibration & evaluation:
        raise DatasetManifestError("ImageNet calibration/evaluation content overlap")
    if not screen < evaluation:
        raise DatasetManifestError("ImageNet 1k is not a strict subset of 10k")

    for name in ("coco_calibration_2k", "coco_screen_1k", "coco_evaluation_5k"):
        _unique(loaded[name], "image_id", name)
    coco_cal = {row["image_id"] for row in loaded["coco_calibration_2k"]}
    coco_screen = {row["image_id"] for row in loaded["coco_screen_1k"]}
    coco_eval = {row["image_id"] for row in loaded["coco_evaluation_5k"]}
    if coco_cal & coco_eval:
        raise DatasetManifestError("COCO calibration/evaluation overlap")
    if not coco_screen < coco_eval:
        raise DatasetManifestError("COCO screen is not a strict subset of evaluation")
    return {"status": "verified", "counts": {name: len(rows) for name, rows in loaded.items()},
            "imagenet_nested": True, "imagenet_content_disjoint": True,
            "coco_nested": True, "coco_split_disjoint": True}


def verify_payload_record(root: str | Path, record: dict[str, Any]) -> int:
    """Verify every listed image, including nested lists, against frozen bytes."""
    root = Path(root).resolve()
    rows = load_tsv(root / record["path"])
    if sha256(root / record["path"]) != record["sha256"] or len(rows) != record["count"]:
        raise DatasetManifestError("list identity/count mismatch")
    payload_root = root / "data/raw" / record["logical_payload_root"]
    for row in rows:
        path = payload_root / row.get("relative_path", row.get("file_name", ""))
        if not path.resolve().is_relative_to(payload_root.resolve()):
            raise DatasetManifestError("image path escapes payload root")
        if sha256(path) != row["sha256"]:
            raise DatasetManifestError(f"image payload mismatch: {path}")
    return len(rows)
