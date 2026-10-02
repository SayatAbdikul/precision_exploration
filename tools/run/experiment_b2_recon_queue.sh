#!/usr/bin/env bash
# experiment_b2_recon_queue.sh NAME JOBFILE
#
# Runs every line of JOBFILE as  gpu_run.sh .venv-b/bin/python -m tools.run.experiment_b2_recon <line>.
# A job that returns 3 (time budget reached, more layers or arms to do) is run again; any other
# non-zero code is logged and the queue moves on.  Each GPU job stays under about 15 minutes.
# Log: artifacts/experiment_b2_recon/logs/NAME.log ; completion marker: NAME.done (holds the failure count).
set -u
ROOT=/home/maveric/precision_exploration
cd "$ROOT" || exit 1
NAME=$1
JOBS=$2
LOGS=$ROOT/artifacts/experiment_b2_recon/logs
mkdir -p "$LOGS"
LOG=$LOGS/$NAME.log
failures=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  attempts=0
  while :; do
    attempts=$((attempts + 1))
    echo "[$(date -Is)] start ($attempts): $line" >> "$LOG"
    # shellcheck disable=SC2086
    artifacts/agent_orchestration/gpu_run.sh --wait 14400 .venv-b/bin/python -m tools.run.experiment_b2_recon $line >> "$LOG" 2>&1
    code=$?
    echo "[$(date -Is)] exit $code: $line" >> "$LOG"
    if [ "$code" -eq 3 ] && [ "$attempts" -lt 40 ]; then continue; fi
    if [ "$code" -ne 0 ]; then failures=$((failures + 1)); fi
    break
  done
done < "$JOBS"
echo "$failures" > "$LOGS/$NAME.done"
