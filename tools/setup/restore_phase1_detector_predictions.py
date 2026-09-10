"""Attempt exact recovery of the frozen detector output using native CPU kernels.

Phase 1 used a macOS CPU. This replay disables Linux oneDNN, preserves the new
outputs separately, and restores the original artifact only on frozen hash
equality. A numerical reproduction with different bytes is reported as such.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from public.workloads.datasets.identity import sha256, verify_payload_record
from public.workloads.models.identity import verify_model_manifest


ROOT = Path(__file__).resolve().parents[2]


def restore(*, threads: int = 4, batch_size: int = 8) -> dict:
    if threads < 1 or batch_size < 1:
        raise ValueError("threads and batch size must be positive")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import torch
    import ultralytics
    import ultralytics.utils.torch_utils as torch_utils
    from public.workloads.models.yolo_eval import evaluate

    manifest = verify_model_manifest(ROOT / "public/workloads/models/manifests/yolov8n.json",
                                     repository_root=ROOT)
    if ultralytics.__version__ != manifest["framework_version"]:
        raise ValueError("frozen Ultralytics version mismatch")
    dataset = json.loads((ROOT / "data/manifests/index.json").read_text())["records"]["coco_evaluation_5k"]
    count = verify_payload_record(ROOT, dataset)
    baseline_path = ROOT / "results/summaries/phase1-fp32-baselines.json"
    expected = next(row for row in json.loads(baseline_path.read_text())["runs"] if row["model"] == "yolov8n")
    # Ultralytics resets torch's CPU thread count during device selection.
    torch_utils.NUM_THREADS = threads
    torch.set_num_threads(threads)
    output_root = ROOT / "artifacts/per_image_predictions/phase2-yolov8n-native"
    raw = output_root / "predictions-raw.json"
    normalized = output_root / "predictions-official.json"
    raw_summary = ROOT / "results/summaries/phase2-yolov8n-native-fp32-raw.json"
    official_summary = ROOT / "results/summaries/phase2-yolov8n-native-fp32-cocoeval.json"
    with torch.backends.mkldnn.flags(enabled=False):
        evaluate(checkpoint=ROOT / manifest["checkpoint_path"],
            dataset=ROOT / "data/raw/coco2017/coco2017-local.yaml", output=raw,
            summary=raw_summary, batch_size=batch_size, workers=0, device="cpu")
    # val2017.txt triggers Ultralytics' COCO category mapping before JSON save.
    subprocess.run([sys.executable, "-m", "tools.analysis.evaluate_coco_predictions",
        "--annotations", "data/raw/coco2017/annotations/instances_val2017.json",
        "--predictions", str(raw.relative_to(ROOT)),
        "--normalized", str(normalized.relative_to(ROOT)),
        "--summary", str(official_summary.relative_to(ROOT)),
        "--category-mapping", "official"], cwd=ROOT, check=True)
    metrics = json.loads(official_summary.read_text())["metrics"]
    match = sha256(normalized) == expected["prediction_sha256"]
    original = ROOT / "artifacts/per_image_predictions/yolov8n_coco2017_val5k_coco_predictions.json"
    if match:
        if original.exists() and sha256(original) != expected["prediction_sha256"]:
            raise ValueError("refusing to replace a conflicting original detector artifact")
        if not original.exists():
            shutil.copyfile(normalized, original)
        if sha256(original) != expected["prediction_sha256"]:
            raise ValueError("restored detector artifact hash mismatch")
    report = {
        "schema_version": "phase2-detector-fp32-native-reproduction-1.0.0",
        "status": "restored_byte_identical" if match else "replayed_with_differences",
        "byte_identical": match, "restored_path": str(original.relative_to(ROOT)) if match else None,
        "dataset": {"path": dataset["path"], "sha256": dataset["sha256"], "verified_images": count},
        "runtime": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                    "threads": threads, "batch_size": batch_size, "mkldnn_enabled": False, "device": "cpu"},
        "metrics": metrics,
        "metric_deltas": {key: metrics[key] - expected[key] for key in metrics},
        "frozen_prediction_sha256": expected["prediction_sha256"],
        "evidence": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                     for path in (raw, normalized, raw_summary, official_summary, baseline_path)],
    }
    (ROOT / "results/summaries/phase2-detector-fp32-native-reproduction.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    report = restore(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
