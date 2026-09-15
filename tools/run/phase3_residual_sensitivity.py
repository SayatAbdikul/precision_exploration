"""Run selected one-residual classifier sensitivity studies, separate from screens."""
import argparse
import fcntl
from datetime import datetime, timezone
from pathlib import Path

from tools.phase3.common import ROOT, digest, write
from tools.phase3.residual_sensitivity import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--backend", choices=("cpp", "reference"), default="cpp")
    args = parser.parse_args()
    # CPU-only diagnostics may coexist with the CUDA screen. Serialize these
    # studies separately so retries cannot write the same image concurrently.
    lock_path = ROOT / "artifacts/phase3/locks/cpu-sensitivity.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for target in args.layers:
            try:
                report = run(args.prepared, target, backend=args.backend, images=args.images)
            except Exception as error:
                failure = {"scope": "diagnostic_one_layer", "status": "failed", "prepared": str(args.prepared),
                           "target": target, "backend": args.backend, "images": args.images,
                           "time": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"}
                write(ROOT / "artifacts/phase3/sensitivity/failures" / f"{digest(failure)}.json", failure)
                raise
            print({key: report[key] for key in ("status", "job_sha256", "scope", "target", "images")}, flush=True)


if __name__ == "__main__":
    main()
