"""Recover publication after interruption, only from complete verified images.

This does no inference and never fills a missing image or backend. Its separate
provenance identifies postprocessing repaired after the original worker failed.
"""
import argparse
from pathlib import Path

from public.experiments.registry import ExperimentRegistry
from tools.phase3.common import ROOT, digest, read, reference, write
from tools.phase3.evidence import read_records, paired_records, verify_complete, verify_pipeline
from tools.phase3.jobs import run_registered
from tools.phase3.screening import validate_job


def publish(job, root=ROOT):
    validate_job(job, root)
    pipeline = verify_pipeline(job, root, current_execution=True)
    prepared, config, _, _, records, image_refs = read_records(job, root=root)
    identity = digest(job)
    directory = root / "artifacts/phase3/runs" / identity
    rows = paired_records(records, config["model"], job["scope"])
    write(directory / "paired.json", rows)
    with ExperimentRegistry(root / "results/databases/phase3.sqlite", validator=lambda value: validate_job(value, root)) as registry:
        registry.store_per_image(identity, rows)
    summary = {"schema_version": "phase3-run-summary-1.0.0", "job_sha256": identity,
               "configuration_sha256": prepared["configuration_sha256"], "source_sha256": job["source_sha256"],
               "status": "completed", "scope": job["scope"], "images": len(records),
               "backend_equality": {"cpp", "cuda"} <= set(job["backends"]), "diagnostics_complete": True,
               "paired": reference(directory / "paired.json", root), "image_records": image_refs,
               "timings_seconds": {backend: [row["backends"][backend]["seconds"] for row in records] for backend in job["backends"]},
               "publication_recovery": {"implementation": reference(__file__, root), "pipeline": pipeline,
                                        "reason": "publish already completed, independently verified image checkpoints"}}
    write(directory / "summary.json", summary)
    verify_complete(directory / "summary.json", root, current_execution=True)
    return {"images": (len(records), "count"), "completed": (1, "boolean")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--recover-stale-seconds", type=int)
    args = parser.parse_args()
    job = read(args.job)
    # Check completeness before creating any publication attempt.
    read_records(job)
    row = run_registered(job, recover_stale_seconds=args.recover_stale_seconds, executor=publish)
    report = {"job_sha256": digest(job), "run_id": row["run_id"], "status": row["status"], "error": row["error"],
              "scope": job["scope"], "artifacts": f"artifacts/phase3/runs/{digest(job)}"}
    write(ROOT / f"results/summaries/phase3-run-{digest(job)[:12]}.json", report)
    print(report)
    if row["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
