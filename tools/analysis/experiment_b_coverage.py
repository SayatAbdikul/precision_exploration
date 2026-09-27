"""Audit both versioned Experiment B components and their 128-image evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, formats, seal, unseal
from tools.experiment_b.common import source_identity as original_source
from tools.experiment_b_ext.runner import source_identity as extension_source


def audit():
    components = (
        ("original", ROOT / "artifacts/experiment_b", original_source()),
        ("extension", ROOT / "artifacts/experiment_b_ext", extension_source()),
    )
    panel = {name: dataset(name)[1][:128] for name in ("imagenet_screen_1k", "coco_screen_1k")}
    matrix, counts, roots = set(), Counter(), {}
    for component, base, source in components:
        status_path = base / "status.json"
        state = json.loads(status_path.read_text())
        if state["source_sha256"] != source:
            raise ValueError(f"{component} source identity changed")
        completed = [task for task in state["tasks"] if task["status"] == "completed"]
        expected = 126 if component == "original" else 74
        if len(completed) != expected:
            raise ValueError(f"{component} completion count is {len(completed)} instead of {expected}")
        if component == "extension" and state["status"] != "extension_finished":
            raise ValueError("extension did not finish")
        for task in completed:
            key = (task["model"], task["format"], task["recipe"])
            if key in matrix:
                raise ValueError("duplicate cross-component configuration")
            matrix.add(key)
            identity = task["configuration_sha256"]
            configuration = unseal(base / "configurations" / f"{identity}.json")
            if digest(configuration) != identity or configuration["source_sha256"] != source:
                raise ValueError("configuration identity mismatch")
            if (configuration["format"], configuration["recipe"]) != key[1:]:
                raise ValueError("configuration key mismatch")
            summary = unseal(base / "summaries" / f"{identity}-128.json")
            if summary["configuration_sha256"] != identity or summary["panel_images"] != 128:
                raise ValueError("summary identity or panel mismatch")
            rows = panel["coco_screen_1k" if key[0] == "yolov8n" else "imagenet_screen_1k"]
            predictions = [unseal(base / "predictions" / identity / (row["sha256"] + ".json")) for row in rows]
            if any(prediction["configuration_sha256"] != identity or prediction["sample"] != row
                   for prediction, row in zip(predictions, rows)):
                raise ValueError("prediction identity/population mismatch")
            if digest(predictions) != summary["prediction_digest"]:
                raise ValueError("summary/prediction digest mismatch")
            counts[component] += 1
            counts["model_images"] += len(predictions)
        roots[component] = {"status_sha256": file_hash(status_path), "source_sha256": source,
                            "completed": len(completed)}
    expected_matrix = {(model, fmt["name"], recipe)
                       for model in ("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n")
                       for fmt in formats() for recipe in ("maxabs", "percentile_99_9")}
    if matrix != expected_matrix or len(matrix) != 200:
        raise ValueError("the 25×4×2 matrix is incomplete")
    return {"version": "experiment-b-128-coverage-audit-1.0.0",
            "status": "completed_exploratory_128_image_matrix", "configurations": len(matrix),
            "datatype_model_pairs": len({(m, f) for m, f, _ in matrix}),
            "datatypes": 25, "models": 4, "recipes": 2,
            "image_inferences": counts["model_images"], "components": roots,
            "interpretation": "development_panel_descriptive_only; no_1k_promotion_or_exact_native_acceptance",
            "audit_source_sha256": file_hash(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.write:
        destination = ROOT / "results/summaries/experiment-b-128-coverage-2026-09-25.json"
        seal(destination, result)
        print(destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
