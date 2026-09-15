"""Summarize retained per-layer evidence from a completed native pilot or screen."""
import argparse
from pathlib import Path

from public.analysis.phase3.layer_summary import summarize
from tools.phase3.common import ROOT, reference, write
from tools.phase3.evidence import verify_complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    job, summary, (_, config, _, _, records, _) = verify_complete(args.summary)
    report = {"schema_version": "phase3-layer-summary-1.0.0", "job_sha256": summary["job_sha256"],
              "model": config["model"], "format": config["formats"]["activation"]["name"], "scope": summary["scope"],
              "evidence": reference(args.summary), "implementation": reference(ROOT / "public/analysis/phase3/layer_summary.py"),
              **summarize(records, job["backends"][0])}
    path = ROOT / "results/summaries" / f"phase3-diagnostics-{summary['job_sha256'][:12]}.json"
    write(path, report)
    print(path)


if __name__ == "__main__":
    main()
