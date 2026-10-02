#!/usr/bin/env bash
# Repaired-recipe (B2) cases: protocol MB2. Resumable (every step skips sealed evidence).
# Usage: scaled_bridge_v2_b2.sh "CASE ..."      e.g. "resnet18-int8-default-b2 resnet18-fp7_e3m3-default-b2"
# The adapted exports must exist (python -m tools.run.scaled_bridge_v2 b2-export <b2 export folder>).
# One GPU job per case through artifacts/agent_orchestration/gpu_run.sh (shared mode, up to 4 at once):
# primitive conformance of every codebook of the export, both CUDA arms on 32 images, the B2 simulator replay.
# CPU arms (8 images) run two at a time; then every case is gated.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
CASES=$1
RUN=".venv-b/bin/python -m tools.run.scaled_bridge_v2"
GPU=artifacts/agent_orchestration/gpu_run.sh
LOGS=artifacts/scaled_bridge_v2/logs; mkdir -p "$LOGS"
# Build both libraries once, before any parallel job: concurrent first builds of a new source digest race.
$RUN build > "$LOGS/build.log" 2>&1 || { echo "BUILD-FAILED"; exit 1; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do sleep 2; done; }
for CASE in $CASES; do
  EXPORT=$(ls artifacts/scaled_bridge_v2/exports/*/"$CASE"/export.json)
  BOOKS=$(jq -r '.payload.codebooks | keys | join(" ")' "$EXPORT")
  throttle 4
  $GPU bash -c "$RUN conformance $BOOKS --case $CASE && $RUN panel $CASE wide cuda 32 && $RUN panel $CASE control cuda 32 && $RUN b2-replay $CASE 32" \
    > "$LOGS/$CASE.gpu.log" 2>&1 || echo "GPU-FAILED $CASE" &
done
for CASE in $CASES; do
  throttle 6
  ($RUN panel "$CASE" wide cpp 8 && $RUN panel "$CASE" control cpp 8) > "$LOGS/$CASE.cpu.log" 2>&1 || echo "CPU-FAILED $CASE" &
done
wait
for CASE in $CASES; do
  $RUN b2-gate "$CASE" 2>&1 | tail -1
done
echo "ALL-B2-CASES-DONE"
