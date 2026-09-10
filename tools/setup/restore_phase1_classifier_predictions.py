"""Replay frozen classifier baselines and restore only byte-identical artifacts.

The candidate outputs and the new reproduction report belong to Phase 2. The
Phase 1 output paths are populated only after their frozen hashes match; the
original baseline summaries and expected hashes are never changed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from public.workloads.datasets.identity import sha256, verify_payload_record
from public.workloads.models.identity import verify_model_manifest
from public.workloads.models.torchvision_eval import evaluate


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")


def restore(*, threads: int = 4, batch_size: int = 32) -> dict:
    import torch
    import torchvision

    if threads < 1 or batch_size < 1:
        raise ValueError("threads and batch size must be positive")
    torch.set_num_threads(threads)
    index = json.loads((ROOT / "data/manifests/index.json").read_text())
    dataset = index["records"]["imagenet_screen_1k"]
    count = verify_payload_record(ROOT, dataset)
    baseline_path = ROOT / "results/summaries/phase1-fp32-baselines.json"
    frozen = {item["model"]: item for item in json.loads(baseline_path.read_text())["runs"]}
    report = {
        "schema_version": "phase2-classifier-fp32-reproduction-1.0.0",
        "dataset": {"name": "imagenet_screen_1k", "path": dataset["path"],
                    "sha256": dataset["sha256"], "verified_images": count},
        "frozen_baselines": {"path": str(baseline_path.relative_to(ROOT)),
                             "sha256": sha256(baseline_path)},
        "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                    "device": "cpu", "threads": threads, "batch_size": batch_size,
                    "mkldnn_initially_enabled": torch.backends.mkldnn.enabled},
        "runs": [],
    }
    report_path = ROOT / "results/summaries/phase2-classifier-fp32-reproduction.json"
    for name in MODELS:
        manifest = verify_model_manifest(ROOT / f"public/workloads/models/manifests/{name}.json",
                                         repository_root=ROOT)
        if torchvision.__version__.split("+")[0] != manifest["framework_version"]:
            raise ValueError(f"frozen torchvision version mismatch: {name}")
        original = ROOT / f"artifacts/per_image_predictions/{name}_imagenet1k_screen.jsonl"
        candidate = ROOT / f"artifacts/per_image_predictions/phase2-{name}-fp32-imagenet1k-screen.jsonl"
        print(f"Replaying {name} on {count} frozen screen images", flush=True)
        expected = frozen[name]
        def replay(path: Path) -> dict:
            result = evaluate(model_name=name,
                dataset_root=ROOT / "data/raw" / dataset["logical_payload_root"],
                sample_list=ROOT / dataset["path"], output=path,
                checkpoint=ROOT / manifest["checkpoint_path"], batch_size=batch_size,
                workers=0, seed=0, device="cpu")
            result["mkldnn_enabled"] = torch.backends.mkldnn.enabled
            result["predictions"]["path"] = str(path.relative_to(ROOT))
            return result

        result = replay(candidate)
        attempts = [result]
        # Phase 1 ran on macOS without oneDNN. Linux's oneDNN FP32 kernels
        # can change a near-tied top-five order while keeping metrics equal.
        if result["predictions"]["sha256"] != expected["prediction_sha256"] and torch.backends.mkldnn.enabled:
            candidate = ROOT / f"artifacts/per_image_predictions/phase2-{name}-fp32-native-imagenet1k-screen.jsonl"
            with torch.backends.mkldnn.flags(enabled=False):
                result = replay(candidate)
            attempts.append(result)
        match = result["predictions"]["sha256"] == expected["prediction_sha256"]
        metric_match = result["metrics"] == {
            "count": expected["sample_count"], "top1_percent": expected["top1_percent"],
            "top5_percent": expected["top5_percent"]}
        if match and not metric_match:
            raise ValueError(f"byte-identical output has inconsistent metrics: {name}")
        if match:
            if original.exists() and sha256(original) != expected["prediction_sha256"]:
                raise ValueError(f"refusing to replace a conflicting original artifact: {original}")
            if not original.exists():
                shutil.copyfile(candidate, original)
            if sha256(original) != expected["prediction_sha256"]:
                raise ValueError(f"restored artifact hash mismatch: {name}")
        report["runs"].append({
            **result, "predictions": {**result["predictions"], "path": str(candidate.relative_to(ROOT))},
            "attempts": attempts,
            "frozen_prediction_sha256": expected["prediction_sha256"],
            "byte_identical": match, "metrics_identical": metric_match,
            "restored_path": str(original.relative_to(ROOT)) if match else None,
        })
        report["status"] = "in_progress" if len(report["runs"]) < len(MODELS) else (
            "restored_byte_identical" if all(row["byte_identical"] for row in report["runs"])
            else "replayed_with_differences")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"{name}: byte_identical={match}, metrics_identical={metric_match}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = restore(**vars(args))
    print(json.dumps({"status": report["status"], "models": len(report["runs"])}))


if __name__ == "__main__":
    main()
