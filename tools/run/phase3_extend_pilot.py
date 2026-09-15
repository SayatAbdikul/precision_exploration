"""Seed an extended pilot from verified completed images without rerunning them."""
import argparse
import fcntl
from pathlib import Path

from public.experiments.registry import ExperimentRegistry
from tools.phase3.common import ROOT, checked, digest, read, reference, write
from tools.phase3.evidence import archive_pipeline, verify_complete
from tools.phase3.pilot_reuse import compatible_jobs, merge_record
from tools.phase3.provenance import source_reference
from tools.phase3.screening import validate_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--backends", nargs="+", choices=("cpp", "cuda", "reference"))
    args = parser.parse_args()
    source_job, summary, (_, _, _, _, records, image_refs) = verify_complete(args.summary, current_execution=True)
    target = {**source_job, "images": args.images, "backends": args.backends or source_job["backends"]}
    compatible_jobs(source_job, target)
    validate_job(target)
    identity = digest(target)
    if identity == digest(source_job):
        raise ValueError("target pilot is already the source job")
    cpu_only = set(target["backends"]) <= {"cpp", "reference"}
    lock_path = ROOT / "artifacts/phase3/locks" / ("cpu-native-worker.lock" if cpu_only else "native-worker.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExperimentRegistry(ROOT / "results/databases/phase3.sqlite", validator=lambda value: validate_job(value)) as registry:
            row = registry.connection.execute("SELECT status FROM runs WHERE experiment_id=? ORDER BY attempt DESC LIMIT 1", (identity,)).fetchone()
            if row is not None and row["status"] in {"RUNNING", "COMPLETED"}:
                raise ValueError("cannot seed an active or completed target pilot")
        directory = ROOT / "artifacts/phase3/runs" / identity
        if (directory / "job.json").exists() and read(directory / "job.json") != target:
            raise ValueError("target pilot job identity conflict")
        archive_pipeline()
        write(directory / "job.json", target)
        provenance = {"source_summary": reference(args.summary),
                      "source_job": reference(args.summary.parent / "job.json"),
                      "implementations": [source_reference(ROOT / path) for path in (
                          "tools/run/phase3_extend_pilot.py", "tools/phase3/pilot_reuse.py")]}
        copied = []
        for record, image_ref in zip(records, image_refs):
            checked(image_ref)
            path = directory / f"{record['sample']['sha256']}.json"
            merged, added = merge_record(record, read(path) if path.exists() else None, source_job, target,
                                         {**provenance, "source_image": image_ref})
            if added:
                write(path, merged)
            copied.append({"image": reference(path), "added_backends": added})
        event = {"schema_version": "phase3-pilot-reuse-1.0.0", "source_summary": reference(args.summary),
                 "target_job": reference(directory / "job.json"), "records": copied,
                 "scope": "verified image reuse only; no new inference or pilot completion claimed"}
        write(ROOT / "artifacts/phase3/pilot-reuse" / f"{digest(event)}.json", event)
        print({"target_job_sha256": identity, "requested_images": target["images"],
               "copied_image_backends": sum(len(row["added_backends"]) for row in copied)}, flush=True)


if __name__ == "__main__":
    main()
