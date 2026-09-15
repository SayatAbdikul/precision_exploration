"""Frozen paired FP32 predictions on the exact Phase 3 screening populations."""
from collections import defaultdict
import json

from public.workloads.datasets.identity import load_tsv, verify_payload_record
from public.analysis.phase3.statistics import coco_metrics, resampled_coco
from tools.phase3.common import ROOT, read, checked, reference, file_hash, write, digest


def screen_rows(model, plan, root=ROOT, *, verify_images=True):
    name = "coco_screen_1k" if model == "yolov8n" else "imagenet_screen_1k"
    record = read(checked(plan["inputs"]["datasets"], root))["records"][name]
    path = root / record["path"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("screen list hash mismatch")
    rows = sorted(load_tsv(path), key=lambda row: row["sha256"])
    if len(rows) != 1000 or len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Phase 3 needs the complete unique fixed 1k population")
    if verify_images:
        verify_payload_record(root, record)
    return rows, root / "data/raw" / record["logical_payload_root"], record


def baseline(model, plan, root=ROOT):
    rows, _, population = screen_rows(model, plan, root)
    runs = read(checked(plan["inputs"]["baseline"], root))["runs"]
    frozen = next(row for row in runs if row["model"] == model)
    name = "yolov8n_coco2017_val5k_coco_predictions.json" if model == "yolov8n" else f"{model}_imagenet1k_screen.jsonl"
    path = root / "artifacts/per_image_predictions" / name
    if file_hash(path) != frozen["prediction_sha256"]:
        raise ValueError("original FP32 prediction artifact hash mismatch")
    metadata = {"schema_version": "phase3-paired-baseline-1.0.0", "campaign_sha256": digest(plan), "model": model,
                "predictions": reference(path, root), "population": population, "image_order": "sha256_ascending"}
    records = []
    if model == "yolov8n":
        predictions = read(path)
        grouped = defaultdict(list)
        for prediction in predictions:
            grouped[int(prediction["image_id"])].append(prediction)
        annotations = root / "data/raw/coco2017/annotations/instances_val2017.json"
        manifest = read(root / "public/workloads/datasets/coco2017.json")
        if file_hash(annotations) != manifest["annotation_sha256"]["instances_val2017"]:
            raise ValueError("COCO annotation identity mismatch")
        for ordinal, row in enumerate(rows):
            records.append({"sample_id": str(row["image_id"]), "ordinal": ordinal, "sample_sha256": row["sha256"],
                            "fp32_prediction": grouped[int(row["image_id"])], "ground_truth": int(row["image_id"])})
        ids = [int(row["sample_id"]) for row in records]
        subset, (selected,) = resampled_coco(read(annotations), [predictions], ids)
        values = coco_metrics(subset, selected)
        metadata.update(annotations=reference(annotations, root),
                        metrics={"map50_95": float(values[0]), "map50": float(values[1])})
    else:
        original = [json.loads(line) for line in path.read_text().splitlines() if line]
        indexed = {row["sample_id"]: row for row in original}
        if len(indexed) != len(original) or set(indexed) != {row["relative_path"] for row in rows}:
            raise ValueError("FP32 predictions do not cover the exact screen population")
        for ordinal, row in enumerate(rows):
            entry = indexed[row["relative_path"]]
            if entry["ground_truth"] != int(row["label"]):
                raise ValueError("paired ground truth mismatch")
            records.append({"sample_id": row["relative_path"], "ordinal": ordinal, "sample_sha256": row["sha256"],
                            "ground_truth": int(row["label"]), "fp32_prediction": entry["top5_predictions"]})
        metadata["metrics"] = {"top1": sum(row["fp32_prediction"][0] == row["ground_truth"] for row in records)/len(records),
                               "top5": sum(row["ground_truth"] in row["fp32_prediction"] for row in records)/len(records)}
    metadata["records"] = records
    destination = root / "artifacts/phase3/baselines" / f"{model}-{digest(metadata)}.json"
    write(destination, metadata)
    return reference(destination, root)
