"""Real locks, corrupt-completion recovery boundaries and resource admission."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.phase3.common import digest, read, write
from tools.phase3.controller import schedule, validate_tasks
from tools.phase3.controller_resources import GIB, admission
from tools.phase3.controller_worker import rational_admission_guard
from tools.phase3.jobs import run_registered
from tools.phase3.worker_locks import job_lock, pool_locks


def test_completed_job_with_deleted_output_is_not_reported_success(tmp_path):
    output = tmp_path / "output.json"
    job = {"test": "durable-result", "backends": ["cpp"]}
    calls = []
    def execute(_job):
        calls.append(1)
        output.write_text('{"result":1}')
        return {}
    def verify(_job):
        assert json.loads(output.read_text()) == {"result": 1}
    options = dict(root=tmp_path, validator=lambda value: value, executor=execute, completed_validator=verify)
    assert run_registered(job, **options)["status"] == "COMPLETED"
    assert run_registered(job, **options)["status"] == "COMPLETED"
    output.unlink()
    with pytest.raises(FileNotFoundError):
        run_registered(job, **options)
    assert calls == [1]


def test_executor_output_is_verified_before_registry_completion(tmp_path):
    def reject(_job):
        raise ValueError("missing evidence")
    result = run_registered({"case": "incomplete-output"}, root=tmp_path, validator=lambda value: value,
                            executor=lambda value: {}, completed_validator=reject)
    assert result["status"] == "FAILED"
    assert "missing evidence" in result["error"]


def test_pool_excludes_legacy_workers_and_same_job_but_allows_other_jobs(tmp_path):
    with pool_locks(tmp_path) as locks:
        with pytest.raises(RuntimeError, match="live local writer"):
            with job_lock(tmp_path, "a" * 64, cpu_only=False):
                pass
        with job_lock(tmp_path, "a" * 64, cpu_only=False, pool_fd=locks["cuda"]):
            with pytest.raises(RuntimeError, match="live local writer"):
                with job_lock(tmp_path, "a" * 64, cpu_only=False, pool_fd=locks["cuda"]):
                    pass
            with job_lock(tmp_path, "b" * 64, cpu_only=False, pool_fd=locks["cuda"]):
                pass
        # Child contexts must not unlock the shared controller description.
        with pytest.raises(RuntimeError):
            with pool_locks(tmp_path):
                pass
    with pool_locks(tmp_path):
        pass


def test_wrong_pool_descriptor_cannot_bypass_gpu_lock(tmp_path):
    with pool_locks(tmp_path) as locks:
        with pytest.raises(ValueError, match="descriptor"):
            with job_lock(tmp_path, "a" * 64, cpu_only=False, pool_fd=locks["cpu"]):
                pass


def test_inherited_kernel_lock_survives_across_processes(tmp_path):
    with pool_locks(tmp_path) as locks:
        code = "from pathlib import Path; from tools.phase3.worker_locks import job_lock; import sys; " \
               "ctx=job_lock(Path(sys.argv[1]), 'a'*64, cpu_only=False, pool_fd=int(sys.argv[2])); " \
               "ctx.__enter__(); print('locked', flush=True); sys.stdin.readline(); ctx.__exit__(None,None,None)"
        child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(locks["cuda"])],
                                 pass_fds=(locks["cuda"],), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "locked"
            with pytest.raises(RuntimeError):
                with job_lock(tmp_path, "a" * 64, cpu_only=False, pool_fd=locks["cuda"]):
                    pass
            child.communicate("exit\n", timeout=10)
            assert child.returncode == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


def resources(**changes):
    return {"ram_available": 24 * GIB, "disk_free": 80 * GIB, "gpu_free": 10 * GIB, **changes}


def test_admission_reserves_pending_allocations_and_separate_cpu_lane():
    active = [{"resource": "cuda", "rss_bytes": 0}] * 2
    assert admission({"resource": "cuda"}, active, resources(), gpu_workers=2)[0] is False
    assert admission({"resource": "cpu"}, active, resources(), gpu_workers=2)[0] is True
    assert admission({"resource": "cuda"}, active, resources(ram_available=14 * GIB), gpu_workers=4)[0] is False
    assert admission({"resource": "cuda"}, [], resources(gpu_free=None), gpu_workers=1)[0] is False
    assert admission({"resource": "cpu"}, [], resources(disk_free=19 * GIB), gpu_workers=1)[0] is False


def task(name, deps=(), kind="screen", lane="cuda"):
    return {"id": name, "kind": kind, "resource": lane, "model": "test", "format": name, "dependencies": list(deps)}


@pytest.mark.parametrize("tasks", [[task("a"), task("a")], [task("a", ["missing"])], [task("a", ["b"]), task("b", ["a"])]])
def test_dependency_graph_rejects_duplicates_missing_nodes_and_cycles(tasks):
    with pytest.raises(ValueError):
        validate_tasks(tasks)


def fake_launcher(started, *, failures=()):
    class Process:
        pid = 999999999
        def __init__(self, code):
            self.code = code
        def poll(self):
            return self.code
    def launch(command, **options):
        task_path = Path(command[command.index("--task") + 1])
        result_path = Path(command[command.index("--result") + 1])
        value = read(task_path)
        started.append(value["id"])
        failed = value["id"] in failures
        write(result_path, {"task_sha256": digest(value), "status": "failed" if failed else "completed",
                            "guard_sha256": "guard", "controller_sha256": "controller", "outputs": {}, "error": "injected"})
        return Process(1 if failed else 0)
    return launch


def test_scheduler_orders_dependencies_and_reports_scientific_blockers(tmp_path):
    started = []
    inventory = {"campaign_sha256": "campaign", "tasks": [task("pilot"), task("accept", ["pilot"], "accept", "cpu"),
                  task("screen", ["accept"])], "blocked": [{"configuration": "other", "reason": "proof needed"}]}
    code = schedule(inventory, root=tmp_path, gpu_workers=2, poll_seconds=0,
                    launch=fake_launcher(started), probe=resources, guard=lambda: ("guard", "controller"))
    assert started == ["pilot", "accept", "screen"]
    assert code == 2
    state = read(tmp_path / "artifacts/phase3/controller/status.json")
    assert state["phase3_complete"] is False
    assert state["counts"]["completed"] == 3
    assert state["status"] == "blocked"


def test_scheduler_retains_failure_continues_independent_work_and_blocks_dependents(tmp_path):
    started = []
    inventory = {"campaign_sha256": "campaign", "tasks": [task("bad"), task("dependent", ["bad"]), task("good")], "blocked": []}
    code = schedule(inventory, root=tmp_path, gpu_workers=2, poll_seconds=0,
                    launch=fake_launcher(started, failures=["bad"]), probe=resources, guard=lambda: ("guard", "controller"))
    assert code == 1
    assert set(started) == {"bad", "good"}
    state = read(tmp_path / "artifacts/phase3/controller/status.json")
    assert state["tasks"]["dependent"]["status"] == "blocked"
    assert state["tasks"]["good"]["status"] == "completed"


def test_scheduler_does_not_launch_when_memory_unavailable(tmp_path):
    started = []
    inventory = {"campaign_sha256": "campaign", "tasks": [task("screen")], "blocked": []}
    code = schedule(inventory, root=tmp_path, poll_seconds=0, launch=fake_launcher(started),
                    probe=lambda: resources(ram_available=GIB), guard=lambda: ("guard", "controller"))
    assert code == 2 and not started


def test_fixed_point_guard_rejects_silent_overflow_and_preserves_safe_dispatch(monkeypatch):
    from public.inference import rational
    calls = []
    monkeypatch.setattr(rational, "gemm", lambda *a, **kw: calls.append((a, kw)) or (42,))
    with rational_admission_guard():
        with pytest.raises(ValueError, match="uncertified fixed-point"):
            rational.gemm(None, [2**995], [1], batch=1, channels=1, k=1, accumulator="posit8_es1_quire64_accumulator")
        assert rational.gemm(None, [1], [1], batch=1, channels=1, k=1, accumulator="posit8_es1_quire64_accumulator") == (42,)
    assert len(calls) == 1


def test_internal_scheduler_error_is_failure_not_user_interruption(tmp_path):
    def broken_probe():
        raise OSError("resource probe failed")
    code = schedule({"campaign_sha256": "campaign", "tasks": [task("screen")], "blocked": []},
                    root=tmp_path, probe=broken_probe, guard=lambda: ("guard", "controller"))
    assert code == 1
    assert read(tmp_path / "artifacts/phase3/controller/status.json")["status"] == "failed"


def test_seed_pilot_does_not_repair_corrupt_completed_evidence(tmp_path, monkeypatch):
    from tools.phase3 import controller_worker as worker
    job = {"scope": "pilot"}
    directory = tmp_path / "artifacts/phase3/runs" / digest(job)
    write(directory / "job.json", job)
    write(directory / "summary.json", {"status": "completed"})
    def reject(*args, **kwargs):
        raise ValueError("corrupt retained summary")
    monkeypatch.setattr(worker, "verify_complete", reject)
    with pytest.raises(ValueError, match="corrupt retained summary"):
        worker.seed_pilot(job, tmp_path)
    assert read(directory / "summary.json") == {"status": "completed"}
    write(directory / "job.json", {"scope": "different"})
    with pytest.raises(ValueError, match="job identity mismatch"):
        worker.seed_pilot(job, tmp_path)


def test_accelerated_checkpoint_records_provenance_even_without_task_completion(tmp_path, monkeypatch):
    from tools.phase3 import controller_worker as worker, screening
    monkeypatch.setattr(worker, "verify_guard", lambda root: "guard")
    monkeypatch.setattr(worker, "controller_identity", lambda root: "controller")
    original = screening.save_checkpoint
    path = tmp_path / "image.json"
    with pytest.raises(KeyboardInterrupt):
        with worker.checkpoint_execution_provenance({"path": "certificate", "sha256": "cert"}, tmp_path):
            screening.save_checkpoint(path, {"backends": {"cuda": {}}})
            raise KeyboardInterrupt()
    assert screening.save_checkpoint is original
    record = read(path)
    assert record["execution_provenance"]["cuda"]["controller_sha256"] == "controller"
    assert record["record_sha256"] == digest({k: v for k, v in record.items() if k != "record_sha256"})


def test_changed_preprocessing_rejected_by_execution_guard(tmp_path, monkeypatch):
    from tools.phase3 import execution_guard as guard
    write(tmp_path / guard.GUARD_PATH, {"preprocessing_sources": {"input.py": "old"}})
    monkeypatch.setattr(guard, "observed_guard", lambda root: {"preprocessing_sources": {"input.py": "changed"}})
    with pytest.raises(ValueError, match="preprocessing_sources"):
        guard.verify_guard(tmp_path)


def test_terminate_groups_reaps_worker(tmp_path):
    from tools.phase3.controller import terminate_groups
    process = subprocess.Popen([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"],
                               stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        terminate_groups([process], grace_seconds=1)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_checkpoint_extension_requires_retained_matching_certificate(tmp_path):
    from tools.phase3.common import reference
    from tools.phase3.evidence import verify_execution_provenance
    native = tmp_path / "native.json"
    plan = tmp_path / "plan.json"
    evidence = tmp_path / "evidence.json"
    certificate = tmp_path / "certificate.json"
    write(native, {"matched": True})
    write(plan, {"graph": "frozen"})
    identities = {"guard_sha256": "guard", "controller_sha256": "controller"}
    write(evidence, {**identities, "images": 8, "matches_original_layer_codes_outputs_and_quantizer_events": True,
                     "native_evidence": reference(native, tmp_path)})
    write(certificate, {**identities, "configuration_sha256": "config", "images": 8,
                        "plan": reference(plan, tmp_path), "compatibility": reference(evidence, tmp_path)})
    record = {"backends": {"cuda": {}}, "execution_provenance": {"cuda": {
        **identities, "extension": "exact_integer_store_v1", "compatibility": reference(certificate, tmp_path)}}}
    verify_execution_provenance(record, "config", tmp_path)
    with pytest.raises(ValueError, match="certificate mismatch"):
        verify_execution_provenance(record, "different", tmp_path)
    native.unlink()
    with pytest.raises(FileNotFoundError):
        verify_execution_provenance(record, "config", tmp_path)


def test_checkpoint_corruption_cannot_be_reused(tmp_path):
    from tools.phase3.screening import checkpoint, save_checkpoint
    path = tmp_path / "image.json"
    context = {"job_sha256": "job", "graph_sha256": "graph", "sample": {"id": 1}, "paired": {"id": 1}}
    payload = {**context, "backends": {"cuda": {"prediction": [1]}}}
    save_checkpoint(path, payload)
    assert checkpoint(path, context)["backends"] == payload["backends"]
    corrupt = read(path)
    corrupt["backends"]["cuda"]["prediction"] = [2]
    write(path, corrupt)
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        checkpoint(path, context)


def test_termination_reaches_analysis_grandchild_after_leader_exits():
    from tools.phase3.controller import terminate_groups
    code = "import subprocess, sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid, flush=True)"
    leader = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              text=True, start_new_session=True)
    child_pid = int(leader.stdout.readline())
    leader.wait(timeout=5)
    try:
        terminate_groups([leader], grace_seconds=1)
        # A killed orphan can briefly remain as a zombie until init reaps it.
        import time
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            path = Path(f"/proc/{child_pid}/stat")
            if not path.exists() or path.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("analysis grandchild survived process-group termination")
    finally:
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass
