#!/usr/bin/env bash
# Profile only this validation process; do not change the GPU counter policy.
set -euo pipefail
phase2_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$phase2_root"
mkdir -p artifacts/benchmarks/phase2
sudo env OMP_NUM_THREADS=4 /usr/local/cuda/bin/ncu \
  --target-processes all \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active \
  --csv \
  .venv/bin/python -m tools.run.phase2_profile_inputs \
  > artifacts/benchmarks/phase2/counters.csv \
  2> artifacts/benchmarks/phase2/counters-stderr.log
