"""Single-writer execution and explicit recovery for Phase 3 registry jobs."""
from public.experiments.registry import ExperimentRegistry
from public.experiments.scheduler import LocalScheduler
from tools.phase3.common import ROOT, digest
from tools.phase3.screening import validate_job, run_job
from tools.phase3.worker_locks import job_lock


def run_registered(job, *, root=ROOT, recover_stale_seconds=None, validator=None, executor=None,
                   pool_fd=None, completed_validator=None):
    production_validator = validator is None
    validator = validator or (lambda value: validate_job(value, root))
    executor = executor or (lambda value: run_job(value, root))
    if completed_validator is None and production_validator:
        from tools.phase3.evidence import verify_complete
        completed_validator = lambda value: verify_complete(
            root / "artifacts/phase3/runs" / digest(value) / "summary.json", root, current_execution=True)
    validator(job)
    if production_validator:
        from tools.phase3.evidence import archive_pipeline
        archive_pipeline(root)
    if recover_stale_seconds is not None and recover_stale_seconds < 300:
        raise ValueError("stale recovery requires at least 300 seconds without a heartbeat")
    identity = digest(job)
    # CUDA work keeps the existing global GPU lock. One CPU-only pilot can
    # overlap it; targeted transactional claims prevent cross-resource jobs
    # from accidentally claiming each other's pending entries.
    cpu_only = bool(job.get("backends")) and set(job["backends"]) <= {"cpp", "reference"}
    with job_lock(root, identity, cpu_only=cpu_only, pool_fd=pool_fd):
        with ExperimentRegistry(root / "results/databases/phase3.sqlite", validator=validator) as registry:
            if recover_stale_seconds is not None:
                registry.recover_stale(older_than_seconds=recover_stale_seconds, experiment_id=identity)
            run_id, status = registry.submit(job)
            if status == "COMPLETED" and completed_validator is not None:
                # Corrupt/missing evidence must never be reported as reusable.
                # Keep the historical row intact; repairs require explicit work.
                completed_validator(job)
            if status == "PENDING":
                def checked_executor(value):
                    metrics = executor(value)
                    if completed_validator is not None:
                        completed_validator(value)
                    return metrics
                LocalScheduler(registry, worker_id=f"phase3-{identity[:12]}").run_once(checked_executor, run_id=run_id)
            return dict(registry.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())
