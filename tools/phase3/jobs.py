"""Single-writer execution and explicit recovery for Phase 3 registry jobs."""
import fcntl

from public.experiments.registry import ExperimentRegistry
from public.experiments.scheduler import LocalScheduler
from tools.phase3.common import ROOT, digest
from tools.phase3.screening import validate_job, run_job


def run_registered(job, *, root=ROOT, recover_stale_seconds=None, validator=None, executor=None):
    production_validator = validator is None
    validator = validator or (lambda value: validate_job(value, root))
    executor = executor or (lambda value: run_job(value, root))
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
    lock = root / "artifacts/phase3/locks" / ("cpu-native-worker.lock" if cpu_only else "native-worker.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a Phase 3 native job already has a live local writer") from exc
        with ExperimentRegistry(root / "results/databases/phase3.sqlite", validator=validator) as registry:
            if recover_stale_seconds is not None:
                registry.recover_stale(older_than_seconds=recover_stale_seconds, experiment_id=identity)
            run_id, status = registry.submit(job)
            if status == "PENDING":
                LocalScheduler(registry, worker_id=f"phase3-{identity[:12]}").run_once(executor, run_id=run_id)
            return dict(registry.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())
