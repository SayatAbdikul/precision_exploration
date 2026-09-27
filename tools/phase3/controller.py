"""Bounded process scheduler over verified Phase 3 task dependencies."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from tools.phase3.common import ROOT, checked, digest, read, reference, write
from tools.phase3.controller_resources import admission, resident_bytes, snapshot, GIB
from tools.phase3.execution_guard import verify_guard, controller_identity
from tools.phase3.worker_locks import exclusive, pool_locks


def worker_environment():
    return {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
            "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", "")}


def task_priority(task):
    progress = task.get("progress", {})
    if progress.get("remaining_images") == 0:
        return (0, task["id"])
    if progress.get("saved_images", 0):
        return (1, task["id"])
    return ({"analyze": 0, "accept": 0, "validate-store": 2, "pilot": 2, "screen": 3}[task["kind"]], task["id"])


def validate_tasks(tasks):
    indexed = {task["id"]: task for task in tasks}
    if len(indexed) != len(tasks):
        raise ValueError("duplicate task identity")
    visited, visiting = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError("cyclic task dependencies")
        if key in visited:
            return
        if key not in indexed:
            raise ValueError("missing dependency")
        visiting.add(key)
        for dep in indexed[key]["dependencies"]:
            visit(dep)
        visiting.remove(key)
        visited.add(key)
    for key in indexed:
        visit(key)


def terminate_groups(processes, *, grace_seconds=15):
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    # Also signal process groups whose leader has exited: analysis children may
    # still be alive and inherited native locks must not survive the controller.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def schedule(inventory, *, gpu_workers=1, ram_per_worker=3 * GIB, root=ROOT,
             poll_seconds=5, launch=None, probe=None, guard=None):
    """Run ready work once; deterministic failures remain retained, not retried."""
    validate_tasks(inventory["tasks"])
    if type(gpu_workers) is not int or not 1 <= gpu_workers <= 6:
        raise ValueError("GPU worker count must be 1–6")
    launch = launch or subprocess.Popen
    probe = probe or (lambda: snapshot(root))
    guard = guard or (lambda: (verify_guard(root), controller_identity(root)))
    expected_guard = guard()
    directory = root / "artifacts/phase3/controller"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    work = directory / "runs" / run_id
    state = {"schema_version": "phase3-controller-status-1.0.0", "run_id": run_id,
             "campaign_sha256": inventory["campaign_sha256"], "phase3_complete": False,
             "guard_sha256": expected_guard[0], "controller_sha256": expected_guard[1],
             "gpu_workers": gpu_workers, "blocked_configurations": inventory["blocked"],
             "tasks": {task["id"]: {"task": task, "status": "pending"} for task in inventory["tasks"]}}
    active, stopped, signal_number = {}, False, None
    old_handlers = {}
    def stop(signum, _frame):
        nonlocal stopped, signal_number
        stopped, signal_number = True, signum
    def persist():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["counts"] = {status: sum(v["status"] == status for v in state["tasks"].values())
                           for status in ("pending", "running", "completed", "failed", "blocked", "interrupted")}
        write(work / "status.json", state)
        write(directory / "status.json", state)
    with exclusive(directory / "controller.lock"), pool_locks(root) as locks:
        work.mkdir(parents=True, exist_ok=True)
        write(work / "inventory.json", inventory)
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop)
        print(f"Controller {run_id}: {gpu_workers} CUDA slots; status: {directory / 'status.json'}", flush=True)
        try:
            while True:
                if stopped:
                    break
                for key, worker in list(active.items()):
                    code = worker["process"].poll()
                    if code is None:
                        continue
                    worker["log"].close()
                    row = state["tasks"][key]
                    try:
                        result = read(worker["result"])
                        if (code != 0 or result["status"] != "completed" or result["task_sha256"] != digest(row["task"]) or
                                result["guard_sha256"] != expected_guard[0] or result["controller_sha256"] != expected_guard[1]):
                            raise ValueError(result.get("error", f"worker exit {code}"))
                        for item in result["outputs"].values():
                            checked(item, root)
                        row.update(status="completed", result=reference(worker["result"], root))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        row.update(status="failed", error=str(error), exit_code=code)
                    print(row["status"], row["task"]["model"], row["task"]["format"], row["task"]["kind"], row.get("error", ""), flush=True)
                    del active[key]
                pending = [row for row in state["tasks"].values() if row["status"] == "pending"]
                for row in pending:
                    if any(state["tasks"][dep]["status"] in {"failed", "blocked", "interrupted"} for dep in row["task"]["dependencies"]):
                        row.update(status="blocked", error="required task did not complete")
                ready = sorted((row for row in pending if row["status"] == "pending" and
                                all(state["tasks"][dep]["status"] == "completed" for dep in row["task"]["dependencies"])),
                               key=lambda row: task_priority(row["task"]))
                resources = probe()
                state["resources"] = resources
                for row in ready:
                    if stopped:
                        break
                    current = [{"resource": state["tasks"][key]["task"]["resource"], "rss_bytes": resident_bytes(v["process"].pid)}
                               for key, v in active.items()]
                    allowed, reason = admission(row["task"], current, resources, gpu_workers=gpu_workers, ram_per_worker=ram_per_worker)
                    if not allowed:
                        row["waiting_for"] = reason
                        continue
                    if guard() != expected_guard:
                        raise ValueError("source or execution dependencies changed during controller run")
                    task = row["task"]
                    key = task["id"]
                    task_path, result_path = work / f"{key}.task.json", work / f"{key}.result.json"
                    write(task_path, task)
                    fd = locks[task["resource"]]
                    log = (work / f"{key}.log").open("a")
                    command = [sys.executable, "-m", "tools.phase3.controller_worker", "--task", str(task_path),
                               "--result", str(result_path), "--pool-fd", str(fd)]
                    try:
                        process = launch(command, cwd=root, env=worker_environment(), stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(fd,), start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    active[key] = {"process": process, "log": log, "result": result_path}
                    row.update(status="running", pid=process.pid, log=str((work / f"{key}.log").relative_to(root)))
                    row.pop("waiting_for", None)
                    print("starting", task["model"], task["format"], task["kind"], flush=True)
                persist()
                if not active:
                    remaining = [row for row in state["tasks"].values() if row["status"] == "pending"]
                    if remaining:
                        for row in remaining:
                            row.update(status="blocked", error=row.get("waiting_for", "unsatisfied dependency"))
                    break
                time.sleep(poll_seconds)
        except BaseException as error:
            stopped = True
            state["error"] = f"{type(error).__name__}: {error}"
        finally:
            if active:
                terminate_groups([worker["process"] for worker in active.values()])
                for key, worker in active.items():
                    worker["log"].close()
                    state["tasks"][key]["status"] = "interrupted"
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            state["status"] = ("failed" if "error" in state else "interrupted" if stopped else "failed" if any(r["status"] == "failed" for r in state["tasks"].values())
                               else "blocked" if inventory["blocked"] or any(r["status"] != "completed" for r in state["tasks"].values())
                               else "execution_complete_D4_review_required")
            persist()
    if state["status"] == "failed":
        return 1
    if stopped:
        return 128 + (signal_number or signal.SIGINT)
    # Execution completion is not the scientific D4 decision.
    return 2
