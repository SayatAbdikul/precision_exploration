"""Pinned Ultralytics YOLOv8n COCO FP32 baseline entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def evaluate(*, checkpoint: Path, dataset: Path, output: Path, summary: Path,
             batch_size: int = 8, workers: int = 0, device: str = "cpu") -> dict:
    from ultralytics import YOLO, __version__

    project = output.parent / "yolo-validation"
    started = time.monotonic()
    result = YOLO(checkpoint).val(
        data=str(dataset.resolve()), split="val", imgsz=640, batch=batch_size,
        device=device, workers=workers, deterministic=True, seed=0,
        plots=False, save_json=True, project=str(project), name="coco2017-val5k",
        exist_ok=True, verbose=False,
    )
    candidates = list((project / "coco2017-val5k").glob("*predictions.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one COCO predictions file, found {len(candidates)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidates[0], output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    evidence = {
        "schema_version": "1.0.0", "model": "yolov8n", "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluator": f"ultralytics-{__version__}", "dataset": "COCO 2017 val2017",
        "image_count": 5000, "input_size": 640, "batch_size": batch_size,
        "workers": workers, "device": device, "seed": 0,
        "metrics": {"map50_95": float(result.box.map), "map50": float(result.box.map50)},
        "predictions": {"path": str(output), "sha256": digest, "size_bytes": output.stat().st_size},
        "elapsed_seconds": time.monotonic() - started,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
