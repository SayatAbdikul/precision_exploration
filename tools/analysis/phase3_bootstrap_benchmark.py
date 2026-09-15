"""Verify cached COCO bootstrap against direct evaluation on the frozen 1k set."""
import time

from public.analysis.phase3.coco_cache import paired_coco_cached
from public.analysis.phase3.statistics import paired_coco
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write


def main():
    plan = campaign()
    baseline_ref = read(ROOT / "results/summaries/phase3-baselines.json")["models"]["yolov8n"]
    baseline = read(checked(baseline_ref))
    if baseline["campaign_sha256"] != digest(plan):
        raise ValueError("detector baseline belongs to another campaign")
    annotations = read(checked(baseline["annotations"]))
    fp32 = [box for row in baseline["records"] for box in row["fp32_prediction"]]
    candidate = fp32[::2]
    ids = [int(row["sample_id"]) for row in baseline["records"]]
    options = {"confidence": plan["statistics"]["confidence"], "seed": plan["statistics"]["seed"], "resamples": 2}
    start = time.perf_counter()
    direct = paired_coco(annotations, fp32, candidate, ids, **options)
    direct_seconds = time.perf_counter()-start
    start = time.perf_counter()
    cached = paired_coco_cached(annotations, fp32, candidate, ids, **options)
    cached_seconds = time.perf_counter()-start
    if direct != cached:
        raise ValueError("cached detector bootstrap differs from direct COCO recomputation")
    report = {"schema_version": "phase3-bootstrap-benchmark-1.0.0", "scope": "engineering_statistics_benchmark",
              "status": "verified_equivalent", "campaign_sha256": digest(plan), "baseline": baseline_ref,
              "images": len(ids), "resamples": options["resamples"], "seed": options["seed"],
              "comparison": "synthetic prediction subset: retain every second FP32 detection; not a candidate model",
              "direct_seconds": direct_seconds, "cached_seconds": cached_seconds, "result": cached,
              "implementations": [reference(ROOT / "public/analysis/phase3/statistics.py"),
                                  reference(ROOT / "public/analysis/phase3/coco_cache.py")],
              "limits": ["small resample count validates equivalence and runtime only; cannot classify a candidate",
                         "production detector analysis retains the frozen 2000 resamples",
                         "wall times overlap authorized screening and preparation work; not an isolated benchmark"]}
    write(ROOT / "results/summaries/phase3-detector-bootstrap-benchmark.json", report)
    print({key: report[key] for key in ("status", "images", "resamples", "direct_seconds", "cached_seconds")})


if __name__ == "__main__":
    main()
