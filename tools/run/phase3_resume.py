"""Resume all currently provable Phase 3 work with bounded CPU/CUDA concurrency.

--dry-run validates inventory without launching experiments. --calibrate measures
resource settings against accepted native images without adding screening rows.
Blocked scientific gates are reported explicitly; exit 2 never means completion.
"""
import argparse
import json
import os
import sys

from tools.phase3.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 7), help="explicit capacity override; otherwise use measured profile or one worker")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4], help="concurrency levels for calibration")
    parser.add_argument("--images", type=int, default=2, help="accepted images per calibration worker")
    args = parser.parse_args()
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["OMP_DYNAMIC"] = "FALSE"
    if args.status:
        from tools.phase3.common import read
        print(json.dumps(read(ROOT / "artifacts/phase3/controller/status.json"), indent=2))
        return 0
    from tools.phase3.execution_guard import verify_guard
    verify_guard()
    if args.calibrate:
        from tools.phase3.controller_benchmark import benchmark
        profile = benchmark(levels=args.levels, images=args.images)
        print(json.dumps({"selected_workers": profile["workers"], "measurements": profile["measurements"]}, indent=2))
        return 0
    from tools.phase3.campaign_inventory import discover
    print("Validating frozen definitions and saved image checkpoints...", flush=True)
    inventory = discover()
    for task in inventory["tasks"]:
        if "progress" in task:
            p = task["progress"]
            print(f"{task['model']}/{task['format']}: {p['saved_images']}/{p['requested_images']} verified images", flush=True)
    print(f"{len(inventory['tasks'])} execution/analysis tasks; {len(inventory['blocked'])} configurations require further proofs", flush=True)
    if args.dry_run:
        print(json.dumps(inventory, indent=2))
        return 0
    from tools.phase3.controller import schedule
    from tools.phase3.controller_resources import GIB
    workers, ram = args.workers, 3 * GIB
    if workers is None:
        from tools.phase3.controller_benchmark import load_profile
        profile = load_profile()
        if profile:
            workers, ram = profile["workers"], profile["ram_per_worker"]
        else:
            workers = 1
            print("No measured resource profile: using one CUDA worker. Use --calibrate to measure safe concurrency.", flush=True)
    return schedule(inventory, gpu_workers=workers, ram_per_worker=ram)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Phase 3 stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
