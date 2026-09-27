"""Execute one dependency-checked task under an inherited controller lease."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys

from tools.phase3.campaign_inventory import make_job
from tools.phase3.common import ROOT, checked, digest, read, reference, write
from tools.phase3.evidence import verify_complete
from tools.phase3.execution_guard import verify_guard, controller_identity, machine_identity
from tools.phase3.jobs import run_registered
from tools.phase3.pilot_reuse import compatible_jobs, merge_record
from tools.phase3.provenance import source_reference
from tools.phase3.worker_locks import job_lock


def seed_pilot(job, root=ROOT):
    """Import only verified compatible images, retaining their original records."""
    destination = root / "artifacts/phase3/runs" / digest(job)
    if (destination / "job.json").exists() and read(destination / "job.json") != job:
        raise ValueError("existing pilot job identity mismatch")
    if (destination / "summary.json").exists():
        verify_complete(destination / "summary.json", root, current_execution=True)
        return
    write(destination / "job.json", job)
    for path in sorted((root / "artifacts/phase3/runs").glob("*/summary.json")):
        if path.parent == destination:
            continue
        source_job = read(path.parent / "job.json")
        try:
            compatible_jobs(source_job, job)
        except (KeyError, ValueError):
            continue
        _, _, (_, _, _, _, records, refs) = verify_complete(path, root, current_execution=True)
        for record, ref in zip(records, refs):
            target = destination / f"{record['sample']['sha256']}.json"
            merged, added = merge_record(record, read(target) if target.exists() else None, source_job, job,
                                        {"source_summary": reference(path, root), "source_image": ref,
                                         "implementation": source_reference(__file__, root)})
            if added:
                write(target, merged)


@contextmanager
def checkpoint_execution_provenance(compatibility, root=ROOT):
    """Bind every newly saved CUDA image to the validated execution extension."""
    from tools.phase3 import screening
    original = screening.save_checkpoint
    provenance = {"extension": "exact_integer_store_v1", "compatibility": compatibility,
                  "guard_sha256": verify_guard(root), "controller_sha256": controller_identity(root)}
    def save(path, payload):
        payload = {**payload, "execution_provenance": {**payload.get("execution_provenance", {}), "cuda": provenance}}
        original(path, payload)
    screening.save_checkpoint = save
    try:
        yield
    finally:
        screening.save_checkpoint = original


@contextmanager
def rational_admission_guard():
    """Reject the known uncertified fixed-point domain; never change arithmetic.

    The frozen backend remains unchanged so old accepted integer jobs resume.
    This guard admits only calls whose selected scratch size also covers the
    intermediate fractional shift. A revised kernel needs a new engine version.
    """
    from fractions import Fraction
    from math import lcm
    from public.inference import rational
    from public.inference.reference.arithmetic import format_named
    original = rational.gemm
    def guarded(native, inputs, weights, **options):
        acc = format_named(options["accumulator"])
        if acc.family != "float" and acc.manifest["numeric"]["fractional_bits"]:
            left, right = rational.values(inputs), rational.values(weights)
            if any(not isinstance(v, Fraction) for v in left + right):
                raise ValueError("fixed-point rational guard requires finite rational inputs")
            dx = lcm(*(v.denominator for v in left))
            dw = lcm(*(v.denominator for v in right))
            fractional = acc.manifest["numeric"]["fractional_bits"]
            denominator = lcm(dx * dw, 1 << fractional)
            shift = (denominator // (dx * dw)).bit_length() - 1
            xn = max((abs(v.numerator) * (dx // v.denominator) for v in left), default=0)
            wn = max((abs(v.numerator) * (dw // v.denominator) for v in right), default=0)
            intermediate = max((xn * wn).bit_length() + shift, denominator.bit_length() + acc.bits)
            old_required = max(xn.bit_length(), wn.bit_length(), intermediate, denominator.bit_length() + 2) + 4
            width = next((v for v in (1024, 2048, 4096, 10240) if v >= old_required), 0)
            if width < intermediate + fractional + 4:
                raise ValueError("uncertified fixed-point intermediate shift; revised native grid required")
            inputs, weights = left, right
        return original(native, inputs, weights, **options)
    rational.gemm = guarded
    try:
        yield
    finally:
        rational.gemm = original


def execute_task(task, pool_fd, root=ROOT):
    kind = task["kind"]
    if kind == "validate-store":
        from tools.phase3.controller_validation import validate_extension
        return {"compatibility": validate_extension(task, root)}
    if kind in {"screen", "pilot"}:
        job = make_job(task, root)
        if kind == "pilot":
            with job_lock(root, digest(job), cpu_only=False, pool_fd=pool_fd):
                seed_pilot(job, root)
        compatibility = None
        if task.get("accelerated_store"):
            from tools.phase3.controller_validation import verify_certificate
            from tools.phase3.integer_store import accelerated_stores
            compatibility = verify_certificate(task, root)
            extension = accelerated_stores()
        else:
            extension = nullcontext()
        provenance = checkpoint_execution_provenance(compatibility, root) if compatibility else nullcontext()
        with rational_admission_guard(), extension, provenance:
            row = run_registered(job, root=root, recover_stale_seconds=300, pool_fd=pool_fd)
        if row["status"] != "COMPLETED":
            raise RuntimeError(f"native task {row['status']}: {row.get('error')}")
        summary = root / "artifacts/phase3/runs" / digest(job) / "summary.json"
        verify_complete(summary, root, current_execution=True)
        return {"summary": reference(summary, root), **({"compatibility": compatibility} if compatibility else {})}
    if kind == "accept":
        from tools.phase3.acceptance import accept, verify_acceptance
        job = make_job({**task, "kind": "pilot"}, root)
        summary = root / "artifacts/phase3/runs" / digest(job) / "summary.json"
        approved = root / "artifacts/phase3/acceptance" / task["configuration_sha256"] / "prepared-accepted.json"
        if approved.exists():
            verify_acceptance(read(approved), root)
        else:
            result = accept(read(checked(task["prepared"], root)), reference(summary, root), root)
            if result["status"] != "accepted":
                raise ValueError(f"graph acceptance remains pending: {result.get('reasons')}")
        return {"acceptance": reference(approved, root)}
    if kind == "analyze":
        job = make_job({**task, "kind": "screen"}, root)
        summary = root / "artifacts/phase3/runs" / digest(job) / "summary.json"
        verify_complete(summary, root, current_execution=True)
        for module in ("tools.analysis.phase3_results", "tools.analysis.phase3_diagnostics"):
            # Stay in our process group: controller termination also stops these children.
            subprocess.run([sys.executable, "-m", module, "--summary", str(summary)], cwd=root, check=True)
        return {label: reference(root / f"results/summaries/phase3-{label}-{digest(job)[:12]}.json", root)
                for label in ("analysis", "diagnostics")}
    raise ValueError(f"unknown task kind: {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pool-fd", type=int, required=True)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    start = datetime.now(timezone.utc).isoformat()
    guard, implementation = verify_guard(), controller_identity()
    task = read(args.task)
    report = {"task": task, "task_sha256": digest(task), "started_at": start, "machine": machine_identity(),
              "guard_sha256": guard, "controller_sha256": implementation, "pid": os.getpid(),
              "thread_environment": {name: os.environ.get(name) for name in
                                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    exit_code = 0
    try:
        report["outputs"] = execute_task(task, args.pool_fd)
        if verify_guard() != guard or controller_identity() != implementation:
            raise ValueError("execution source changed while task ran")
        report["status"] = "completed"
    except KeyboardInterrupt as error:
        report.update(status="interrupted", error=str(error))
        exit_code = 130
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        exit_code = 1
    report.update(finished_at=datetime.now(timezone.utc).isoformat(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write(args.result, report)
    print(report["status"], report.get("error", ""), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
