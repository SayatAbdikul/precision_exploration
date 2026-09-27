"""Measured capacity admission for native workers; no numerical policy changes."""
import os
from pathlib import Path
import shutil
import subprocess

GIB = 1024 ** 3


def snapshot(root):
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    result = {"ram_available": memory["MemAvailable"], "disk_free": shutil.disk_usage(root).free,
              "logical_cpus": len(os.sched_getaffinity(0)), "gpu_free": None, "gpu_error": None}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True)
        rows = [line.strip().split(",") for line in gpu.stdout.strip().splitlines()]
        if len(rows) != 1:
            raise ValueError("this controller requires exactly one visible GPU")
        result.update(gpu_uuid=rows[0][0].strip(), gpu_free=int(rows[0][1]) * 1024 ** 2, gpu_utilization=int(rows[0][2]))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["gpu_error"] = str(error)
    return result


def resident_bytes(pid):
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def admission(task, active, resources, *, gpu_workers, cpu_workers=1, ram_per_worker=3 * GIB,
              ram_reserve=6 * GIB, gpu_reserve=2 * GIB, disk_reserve=20 * GIB):
    kind = task["resource"]
    if sum(row["resource"] == kind for row in active) >= (gpu_workers if kind == "cuda" else cpu_workers):
        return False, f"{kind} worker slots occupied"
    if resources["disk_free"] < disk_reserve:
        return False, "disk reserve reached"
    # Account for workers that have been admitted but have not allocated their
    # model/graph yet, avoiding a burst of overcommitted process launches.
    unallocated = sum(max(0, ram_per_worker - row.get("rss_bytes", 0)) for row in active)
    if resources["ram_available"] - unallocated < ram_reserve + ram_per_worker:
        return False, "RAM reserve reached"
    if kind == "cuda":
        if resources.get("gpu_free") is None:
            return False, "CUDA device is unavailable: " + str(resources.get("gpu_error"))
        pending_cuda = sum(row["resource"] == "cuda" and row.get("rss_bytes", 0) < ram_per_worker / 2 for row in active)
        if resources["gpu_free"] < gpu_reserve + (pending_cuda + 1) * GIB:
            return False, "VRAM reserve reached"
    return True, None
