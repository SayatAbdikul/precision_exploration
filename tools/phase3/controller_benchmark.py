"""Measure concurrent exact native images without changing production checkpoints."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.controller import terminate_groups, worker_environment
from tools.phase3.controller_resources import snapshot, resident_bytes, GIB
from tools.phase3.evidence import verify_complete
from tools.phase3.execution_guard import verify_guard, controller_identity, machine_identity
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.phase3.screening import pipeline_identity
from tools.phase3.worker_locks import exclusive, pool_locks


def benchmark(*, levels=(1, 2, 4), images=2, root=ROOT):
    if not levels or any(type(n) is not int or not 1 <= n <= 6 for n in levels) or len(set(levels)) != len(levels):
        raise ValueError("benchmark needs unique worker counts from 1 to 6")
    if not 1 <= images <= 8:
        raise ValueError("benchmark uses 1–8 accepted images")
    guard, implementation = verify_guard(root), controller_identity(root)
    model, fmt = "resnet18", "int4"
    prepared = preparation_records(root)[f"{model}/{fmt}"]
    approval = read(root / "artifacts/phase3/acceptance" / prepared["configuration_sha256"] / "acceptance.json")
    verify_complete(checked(approval["pilot"], root), root, current_execution=True)
    work = root / "artifacts/phase3/controller/benchmarks" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    reports = []
    with exclusive(root / "artifacts/phase3/controller/controller.lock"), pool_locks(root):
        for workers in levels:
            resources = snapshot(root)
            if resources.get("gpu_free") is None:
                raise RuntimeError(f"GPU unavailable: {resources.get('gpu_error')}")
            required_ram = max((r["peak_worker_rss_bytes"] for r in reports), default=3 * GIB) * 1.25
            if resources["ram_available"] < 6 * GIB + workers * required_ram or resources["gpu_free"] < 2 * GIB + workers * GIB:
                print(f"Skipping {workers} workers: memory admission limit", flush=True)
                continue
            trial_dir = work / f"workers-{workers}"
            job = {"schema_version": "phase3-thread-benchmark-1.0.0", "scope": "diagnostic_concurrency_benchmark",
                   "model": model, "format": fmt, "images": images,
                   "settings": [{"backend": "cuda", "threads": 4} for _ in range(workers)],
                   "source_sha256": source_identity(), "pipeline_sha256": pipeline_identity(root),
                   "campaign_sha256": digest(campaign(root)),
                   "prepared": reference(checked(prepared["configuration"], root).parent / "prepared.json", root),
                   "accepted_pilot": approval["pilot"],
                   "implementation": source_reference(root / "tools/run/phase3_thread_benchmark.py", root),
                   "libraries": {b: reference(root / f"build/phase2/{b}/libprecision_{b}.so", root) for b in ("cpp", "cuda")}}
            write(trial_dir / "job.json", job)
            processes, logs, peak_rss, min_ram, min_vram = [], [], 0, resources["ram_available"], resources["gpu_free"]
            start = time.monotonic()
            print(f"Benchmarking {workers} concurrent CUDA workers, {images} images each: {trial_dir}", flush=True)
            try:
                for ordinal in range(workers):
                    log = (trial_dir / f"{ordinal:02d}.log").open("w")
                    logs.append(log)
                    processes.append(subprocess.Popen([sys.executable, "-m", "tools.phase3.controller_validation",
                        "--trial", str(trial_dir / "job.json"), "--ordinal", str(ordinal)],
                        cwd=root, env=worker_environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                last_print = 0
                while any(p.poll() is None for p in processes):
                    peak_rss = max(peak_rss, *(resident_bytes(p.pid) for p in processes))
                    resource_state = snapshot(root)
                    min_ram = min(min_ram, resource_state["ram_available"])
                    if resource_state["gpu_free"] is not None:
                        min_vram = min(min_vram, resource_state["gpu_free"])
                    if min_ram < 6 * GIB or min_vram < 2 * GIB:
                        raise RuntimeError("benchmark crossed memory reserve; reduce concurrency")
                    if any(p.poll() not in (None, 0) for p in processes):
                        raise RuntimeError(f"benchmark worker failed; inspect {trial_dir}")
                    if time.monotonic() - last_print >= 30:
                        print(f"  elapsed {time.monotonic()-start:.0f}s; peak worker RSS {peak_rss/GIB:.2f} GiB", flush=True)
                        last_print = time.monotonic()
                    time.sleep(2)
                if any(p.returncode != 0 for p in processes):
                    raise RuntimeError("benchmark worker failed")
            finally:
                if any(p.poll() is None for p in processes):
                    terminate_groups(processes)
                for log in logs:
                    log.close()
            elapsed = time.monotonic() - start
            expected = {}
            evidence = []
            for ordinal in range(workers):
                path = trial_dir / f"{ordinal:02d}-completed.json"
                records = read(path)["records"]
                compatibility = read(trial_dir / f"{ordinal:02d}-compatibility.json")
                peak_rss = max(peak_rss, compatibility["peak_rss_bytes"])
                if len(records) != images or len({r["sample_sha256"] for r in records}) != images:
                    raise ValueError("benchmark image population is incomplete")
                for row in records:
                    if not row["matches_accepted_pilot"]:
                        raise ValueError("benchmark output did not match accepted pilot")
                    signature = (row["input_sha256"], row["output_sha256"], row["layers_sha256"])
                    if row["sample_sha256"] in expected and expected[row["sample_sha256"]] != signature:
                        raise ValueError("concurrent workers disagree")
                    expected[row["sample_sha256"]] = signature
                evidence.append(reference(path, root))
                evidence.append(reference(trial_dir / f"{ordinal:02d}-compatibility.json", root))
            report = {"workers": workers, "images_per_worker": images, "verified_image_executions": workers * images,
                      "wall_seconds_including_startup": elapsed, "images_per_hour": workers * images * 3600 / elapsed,
                      "peak_worker_rss_bytes": peak_rss, "minimum_available_ram": min_ram, "minimum_free_vram": min_vram,
                      "job": reference(trial_dir / "job.json", root), "evidence": evidence}
            write(trial_dir / "measurement.json", report)
            reports.append(report)
            print(f"  {report['images_per_hour']:.1f} verified image executions/hour", flush=True)
        if not reports:
            raise RuntimeError("no benchmark setting fit available resources")
        if verify_guard(root) != guard or controller_identity(root) != implementation:
            raise ValueError("source changed during benchmark")
        best = max(reports, key=lambda row: row["images_per_hour"])
        # Prefer fewer processes when the apparent gain is smaller than 5%.
        chosen = min((r for r in reports if r["images_per_hour"] >= best["images_per_hour"] * .95), key=lambda row: row["workers"])
        profile = {"schema_version": "phase3-resource-profile-1.0.0", "guard_sha256": guard,
                   "controller_sha256": implementation, "machine": machine_identity(), "gpu_uuid": resources["gpu_uuid"],
                   "workers": chosen["workers"], "ram_per_worker": max(3 * GIB, int(max(r["peak_worker_rss_bytes"] for r in reports) * 1.25)),
                   "measurements": reports, "scope": "ResNet18 INT4; diagnostic repeated images, not additional screening evidence",
                   "extension": "exact_integer_store_v1",
                   "limits": "small sample including startup; family-specific scaling and steady-state throughput may differ; production requires per-graph eight-image certificates"}
        write(work / "profile.json", profile)
        write(root / "artifacts/phase3/controller/resource-profile.json", {"profile": reference(work / "profile.json", root)})
        return profile


def load_profile(root=ROOT):
    path = root / "artifacts/phase3/controller/resource-profile.json"
    if not path.exists():
        return None
    profile = read(checked(read(path)["profile"], root))
    if (profile["guard_sha256"] != verify_guard(root) or profile["controller_sha256"] != controller_identity(root) or
            profile["machine"] != machine_identity()):
        raise ValueError("resource profile is stale; recalibrate or explicitly select --workers")
    resources = snapshot(root)
    if resources.get("gpu_uuid") != profile["gpu_uuid"]:
        raise ValueError("resource profile GPU differs or is inaccessible")
    for measurement in profile["measurements"]:
        checked(measurement["job"], root)
        for ref in measurement["evidence"]:
            checked(ref, root)
    return profile
