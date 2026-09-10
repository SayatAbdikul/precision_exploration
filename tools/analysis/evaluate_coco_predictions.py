#!/usr/bin/env python3
"""Normalize YOLO's zero-based JSON categories and run official COCOeval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--category-mapping",choices=("coco80_index","official"),default="coco80_index")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    category_ids = [int(row["id"]) for row in sorted(annotations["categories"], key=lambda row: int(row["id"]))]
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    for prediction in predictions:
        index = int(prediction["category_id"])
        if args.category_mapping == "coco80_index":
            if not 0 <= index < len(category_ids):
                raise ValueError(f"zero-based category outside COCO80 range: {index}")
            prediction["category_id"] = category_ids[index]
        elif index not in category_ids:
            raise ValueError(f"unknown official COCO category: {index}")
    args.normalized.parent.mkdir(parents=True, exist_ok=True)
    args.normalized.write_text(json.dumps(predictions, separators=(",", ":")) + "\n", encoding="utf-8")

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth = COCO(str(args.annotations))
    detected = truth.loadRes(str(args.normalized))
    evaluator = COCOeval(truth, detected, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    summary = {
        "schema_version": "1.0.0", "model": "yolov8n", "dataset": "COCO 2017 val2017",
        "image_count": len(annotations["images"]), "evaluator": "pycocotools.COCOeval",
        "category_mapping": "Ultralytics COCO80 index to official sorted COCO category ID" if args.category_mapping == "coco80_index" else "official COCO category IDs preserved",
        "metrics": {"map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1])},
        "predictions": {"path": str(args.normalized), "sha256": hashlib.sha256(args.normalized.read_bytes()).hexdigest(),
                        "size_bytes": args.normalized.stat().st_size},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
