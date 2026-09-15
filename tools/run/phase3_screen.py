"""Run a resumable native pilot; full 1k execution requires verified acceptance."""
import argparse
from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, digest, read, reference, write
from tools.phase3.graphs import build
from tools.phase3.screening import validate_job, pipeline_identity
from tools.phase3.jobs import run_registered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--scope", choices=("pilot", "screen"), default="pilot")
    parser.add_argument("--backends", nargs="+", choices=("cpp", "cuda", "reference"), default=["cuda", "cpp"])
    parser.add_argument("--recover-stale-seconds", type=int, help="recover expired leases after at least 300 seconds without a heartbeat")
    args = parser.parse_args()
    plan = campaign()
    prepared = build(args.model, args.format, plan)
    prepared_path = ROOT / prepared["configuration"]["path"]
    submission_prepared = prepared_path.parent / "prepared.json"
    if args.scope == "screen":
        approved = ROOT / "artifacts/phase3/acceptance" / prepared["configuration_sha256"] / "prepared-accepted.json"
        if approved.exists():
            submission_prepared = approved
    baselines = read(ROOT / "results/summaries/phase3-baselines.json")
    job = {"schema_version": "phase3-job-1.0.0", "purpose": "experiment_a_screening",
           "prepared": reference(submission_prepared), "baseline": baselines["models"][args.model],
           "scope": args.scope, "images": args.images, "backends": args.backends,
           "campaign_sha256": digest(plan), "source_sha256": source_identity(), "pipeline_sha256": pipeline_identity()}
    validate_job(job)
    identity = digest(job)
    row = run_registered(job, recover_stale_seconds=args.recover_stale_seconds)
    report = {"job_sha256": identity, "run_id": row["run_id"], "status": row["status"], "error": row["error"],
              "scope": args.scope, "model": args.model, "format": args.format,
              "artifacts": f"artifacts/phase3/runs/{identity}"}
    write(ROOT / f"results/summaries/phase3-run-{identity[:12]}.json", report)
    print(report, flush=True)
    if row["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
