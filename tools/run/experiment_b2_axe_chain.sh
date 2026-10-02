#!/usr/bin/env bash
# Lane Q8 chain: run every arm of one case through gpu_run.sh, one process at a time, resuming after the budget
# exit (3). A CUDA out-of-memory failure is retried once with --heavy 8000; any other failure stops the chain.
#   tools/run/experiment_b2_axe_chain.sh CASE ARMS [extra runner args]
# Completion marker: artifacts/experiment_b2_axe/logs/chain-CASE.done (or .failed).
set -u
cd /home/maveric/precision_exploration
export GPU_LANE=Q8
CASE=$1; ARMS=$2; shift 2
LOG=artifacts/experiment_b2_axe/logs
mkdir -p "$LOG"
rm -f "$LOG/chain-$CASE.done" "$LOG/chain-$CASE.failed"
oom_retry=0
heavy_next=${FIRST_HEAVY:-0}   # FIRST_HEAVY=1: the first attempt is itself the out-of-memory retry
for attempt in $(seq 1 12); do
  heavy=()
  [ "$heavy_next" = 1 ] && heavy=(--heavy 8000)
  heavy_next=0
  echo "[$(date '+%F %T')] attempt $attempt ${heavy[*]}" >> "$LOG/chain-$CASE.log"
  artifacts/agent_orchestration/gpu_run.sh --wait 7200 --min-free-mib 5000 "${heavy[@]}" \
    .venv-b/bin/python -m tools.run.experiment_b2_axe run --case "$CASE" --arms "$ARMS" --budget "${BUDGET:-720}" "$@" \
    >> "$LOG/chain-$CASE.log" 2>&1
  code=$?
  echo "[$(date '+%F %T')] exit $code" >> "$LOG/chain-$CASE.log"
  if [ $code = 0 ]; then touch "$LOG/chain-$CASE.done"; exit 0; fi
  if [ $code = 3 ]; then continue; fi
  if tail -n 30 "$LOG/chain-$CASE.log" | grep -q "out of memory" && [ "$oom_retry" = 0 ]; then oom_retry=1; heavy_next=1; continue; fi
  touch "$LOG/chain-$CASE.failed"; exit $code
done
touch "$LOG/chain-$CASE.failed"; exit 1
