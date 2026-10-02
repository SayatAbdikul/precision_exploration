"""GPU budget ledger of lane Q4 (CPU): job wall seconds recorded by both job ledgers.

Prints the recorded total and the recorded total plus a fixed allowance for jobs that left no ledger line
(old-path verify jobs and CUDA out-of-memory failures, estimated at 1200 s by agent r4 from the queue logs).
"""
from __future__ import annotations

import json
import sys

from tools.experiment_b.common import ROOT

LEDGERS = ("artifacts/experiment_b2_attrib/logs/jobs.jsonl", "artifacts/experiment_b2_attrib/lowmem/logs/jobs.jsonl")
UNRECORDED_ALLOWANCE = 1200.0
BUDGET = 10800.0


def recorded_seconds():
    total = 0.0
    for name in LEDGERS:
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                total += float(record.get("job_wall_seconds", record.get("wall_seconds", 0.0)) or 0.0)
    return total


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    need = float(argv[0]) if argv else 0.0
    used = recorded_seconds() + UNRECORDED_ALLOWANCE
    print(f"recorded {used - UNRECORDED_ALLOWANCE:.0f} s + allowance {UNRECORDED_ALLOWANCE:.0f} s = {used:.0f} s of {BUDGET:.0f} s")
    return 0 if used + need <= BUDGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
