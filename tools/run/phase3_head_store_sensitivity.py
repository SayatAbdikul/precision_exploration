"""Retain an eight-image FP32 detector study with only its final store quantized."""
import fcntl
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from public.analysis.phase3.head_store import DenseOutput, calibration_positions, store_head
from public.analysis.phase3.coco_cache import paired_coco_cached
from public.inference.conformance_job import source_identity
from public.inference.tensor import parse_encoding
from public.inference.workload_job import detector_predictions
from tools.phase3.baselines import screen_rows
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.phase3.runtime import load_runtime


def main():
    plan, source = campaign(), source_identity()
    prepared = preparation_records()["yolov8n/int8"]
    config = read(checked(prepared["configuration"]))
    if config != configuration("yolov8n", "int8", plan) or config["runtime"]["source_sha256"] != source:
        raise ValueError("head-store study requires current preparation")
    graph = read(checked(prepared["graph"]))
    target = graph["outputs"][0]
    node = next(n for n in graph["nodes"] if n["name"] == target)
    if node["op"] != "concatenate" or node["attrs"]["axis"] != 2:
        raise ValueError("unexpected detector output topology")
    encoding = parse_encoding(node["attrs"]["output"])
    implementations = [source_reference(ROOT / p) for p in (
        "tools/run/phase3_head_store_sensitivity.py", "public/analysis/phase3/head_store.py",
        "tools/phase3/runtime.py", "public/analysis/phase3/coco_cache.py", "public/analysis/phase3/statistics.py")]
    baseline_ref = read(ROOT / "results/summaries/phase3-baselines.json")["models"]["yolov8n"]
    baseline = read(checked(baseline_ref))
    job = {"schema_version": "phase3-head-store-study-1.0.0", "scope": "diagnostic_final_store_only", "source_sha256": source,
           "campaign_sha256": digest(plan), "configuration": prepared["configuration"], "graph": prepared["graph"],
           "target": target, "images": 8, "baseline": baseline_ref, "implementations": implementations}
    identity = digest(job)
    work = ROOT / "artifacts/phase3/head-store-sensitivity" / identity
    lock_path = ROOT / "artifacts/phase3/locks/cpu-head-store.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(work / "job.json", job)
        population, payload, _ = screen_rows("yolov8n", plan, ROOT)
        sample, observe, _ = load_runtime("yolov8n", plan, ROOT)
        records = []
        class Capture:
            value = None
            def record(self, name, value):
                if name == target:
                    self.value = value.detach().cpu().contiguous().numpy().copy()
        for ordinal, row in enumerate(population[:8]):
            frozen = baseline["records"][ordinal]
            if frozen["sample_sha256"] != row["sha256"]:
                raise ValueError("head-store image order differs from frozen population")
            path = work / f"{row['sha256']}.json"
            if path.exists():
                record = read(path)
                if (record["job_sha256"] != identity or record["sample"] != row or
                        digest({k:v for k,v in record.items() if k != "record_sha256"}) != record["record_sha256"]):
                    raise ValueError("head-store checkpoint identity mismatch")
                checked(record["tensors"])
            else:
                started = perf_counter()
                inputs, original_shape = sample(payload / row.get("relative_path", row.get("file_name", "")))
                capture = Capture()
                observe(inputs, capture)
                if capture.value is None:
                    raise ValueError("FP32 observation did not produce the final head")
                stored, codes, diagnostics = store_head(capture.value, encoding)
                fp32 = detector_predictions(DenseOutput(capture.value), original_shape, row["image_id"])
                candidate = detector_predictions(DenseOutput(stored), original_shape, row["image_id"])
                tensors_path = work / f"{row['sha256']}.npz"
                partial = tensors_path.with_suffix(".partial")
                with partial.open("wb") as stream:
                    np.savez_compressed(stream, fp32=capture.value, stored_codes=codes)
                os.replace(partial, tensors_path)
                record = {"job_sha256": identity, "sample": row, "frozen_reference": frozen,
                          "fp32_prediction": fp32, "candidate_prediction": candidate,
                          "frozen_fp32_prediction_match": fp32 == frozen["fp32_prediction"],
                          "original_shape": list(original_shape), "tensors": reference(tensors_path),
                          "diagnostics": diagnostics, "calibration_positions": calibration_positions(capture.value.shape),
                          "seconds": perf_counter()-started}
                record["record_sha256"] = digest(record)
                write(path, record)
                print(f"head-store image {ordinal+1}/8: {len(fp32)} FP32 / {len(candidate)} stored detections", flush=True)
            if source_identity() != source:
                raise ValueError("engine changed during head-store study")
            records.append(record)
        annotations = read(checked(baseline["annotations"]))
        statistics = paired_coco_cached(annotations, [p for r in records for p in r["fp32_prediction"]],
            [p for r in records for p in r["candidate_prediction"]], [int(r["sample"]["image_id"]) for r in records],
            **{k:plan["statistics"][k] for k in ("confidence", "seed")}, resamples=plan["statistics"]["detector_resamples"])
        report = {"schema_version": "phase3-head-store-summary-1.0.0", "status": "completed", "scope": job["scope"],
                  "job_sha256": identity, "job": reference(work / "job.json"), "images": len(records), "statistics": statistics,
                  "image_records": [reference(work / f"{r['sample']['sha256']}.json") for r in records],
                  "frozen_fp32_prediction_matches": sum(r["frozen_fp32_prediction_match"] for r in records),
                  "classification": "DIAGNOSTIC_ONLY", "limits": ["only the final tensor store uses the candidate encoding; all upstream arithmetic is FP32",
                      "paired reference is the same fresh FP32 head, with frozen-reference prediction matches reported separately",
                      "sampling-position coverage describes the frozen observer geometry, not a replacement calibration policy",
                      "no strict baseline change, whole-graph acceptance or D4 ranking"]}
        write(work / "summary.json", report)
        write(ROOT / "results/summaries/phase3-head-store-sensitivity.json", report)
        print(statistics, flush=True)


if __name__ == "__main__":
    main()
