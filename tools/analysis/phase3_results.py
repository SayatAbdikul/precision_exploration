"""Compute paired statistics from completed records; never rank a partial pilot."""
import argparse
from pathlib import Path

from public.analysis.phase3.statistics import paired_classification, screening_label
from public.analysis.phase3.coco_cache import paired_coco_cached
from tools.phase3.common import ROOT, campaign, checked, read, write, reference, digest
from tools.phase3.evidence import verify_complete
from tools.phase3.provenance import source_reference


def analyze(summary_path):
    plan = campaign()
    job, summary, evidence = verify_complete(summary_path)
    prepared, config, _, _, _, _ = evidence
    baseline = read(checked(job["baseline"]))
    rows = read(checked(summary["paired"]))
    expected = baseline["records"][:summary["images"]]
    if len(rows) != len(expected) or any(any(row.get(key) != item[key] for key in ("sample_id", "ordinal", "sample_sha256", "ground_truth", "fp32_prediction"))
                                          for row, item in zip(rows, expected)):
        raise ValueError("candidate rows are not paired with the original frozen reference")
    for item in summary["image_records"]:
        checked(item)
    if summary["scope"] == "screen" and len(rows) != 1000:
        raise ValueError("incomplete fixed-1k screen")
    policy = plan["statistics"]
    options = {key: policy[key] for key in ("confidence", "seed")}
    if baseline["model"] == "yolov8n":
        annotations = read(checked(baseline["annotations"]))
        fp32 = [box for row in rows for box in row["fp32_prediction"]]
        candidate = [box for row in rows for box in row["candidate_prediction"]]
        statistics = paired_coco_cached(annotations, fp32, candidate, [int(row["sample_id"]) for row in rows],
                                 resamples=policy["detector_resamples"], **options)
    else:
        statistics = paired_classification(rows, [row["sample_id"] for row in expected],
                                           resamples=policy["classification_resamples"], **options)
    document = {"schema_version": "phase3-paired-analysis-1.0.0", "job_sha256": summary["job_sha256"],
                "campaign_sha256": digest(plan), "scope": summary["scope"], "statistics": statistics,
                "model": config["model"], "format": config["formats"]["activation"]["name"], "images": summary["images"],
                "configuration_sha256": prepared["configuration_sha256"],
                "analysis_implementation": source_reference(__file__),
                "statistics_implementation": source_reference(ROOT / "public/analysis/phase3" /
                    ("coco_cache.py" if baseline["model"] == "yolov8n" else "statistics.py")),
                "statistics_dependencies": [source_reference(ROOT / "public/analysis/phase3/statistics.py")],
                "evidence": [reference(summary_path), summary["paired"], job["baseline"]]}
    document["classification"] = screening_label(statistics, policy) if summary["scope"] == "screen" else {
        "label": "PILOT_ONLY", "reason": "small native conformance subsets cannot make D4 decisions"}
    destination = ROOT / "results/summaries" / f"phase3-analysis-{summary['job_sha256'][:12]}.json"
    write(destination, document)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.summary))


if __name__ == "__main__":
    main()
