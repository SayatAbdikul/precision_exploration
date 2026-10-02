#!/usr/bin/env bash
# Run the lines of a job list one after another through the shared GPU launcher (lane Q4).
#   tools/run/experiment_b2_attrib_queue.sh LIST LOGDIR
# A line starting with 'HEAVY ' runs with --heavy 8000 (a retry after a CUDA out-of-memory failure).
# Each line holds the arguments of tools.run.experiment_b2_attrib (e.g. "verify --model resnet18 --format int8").
# Per job: LOGDIR/<n>.log, and a line in LOGDIR/queue.jsonl with exit code and wall seconds (slot wait included).
# A CUDA out-of-memory failure is retried once with --heavy 8000 (LOGDIR/<n>.oom.log keeps the first log).
# Writes LOGDIR/DONE when the list is finished.
set -u
ROOT=/home/maveric/precision_exploration
LIST=$1
LOGDIR=$2
export GPU_LANE=Q4
mkdir -p "$LOGDIR"
cd "$ROOT"
n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n+1))
  if [ -f "$LOGDIR/$n.ok" ]; then continue; fi
  start=$(date +%s)
  launch=""
  case "$line" in "HEAVY "*) line=${line#HEAVY }; launch="--heavy 8000";; esac
  artifacts/agent_orchestration/gpu_run.sh $launch .venv-b/bin/python -m tools.run.experiment_b2_attrib $line > "$LOGDIR/$n.log" 2>&1
  code=$?
  if [ -z "$launch" ] && [ $code -ne 0 ] && grep -q "OutOfMemoryError" "$LOGDIR/$n.log"; then
    # Rule: a CUDA out-of-memory job is retried once as a declared heavy job.
    mv "$LOGDIR/$n.log" "$LOGDIR/$n.oom.log"
    artifacts/agent_orchestration/gpu_run.sh --heavy 8000 .venv-b/bin/python -m tools.run.experiment_b2_attrib $line > "$LOGDIR/$n.log" 2>&1
    code=$?
  fi
  end=$(date +%s)
  printf '{"n": %d, "args": "%s", "exit": %d, "wall_seconds_including_slot_wait": %d, "end_unix": %d}\n' "$n" "$line" "$code" "$((end-start))" "$end" >> "$LOGDIR/queue.jsonl"
  if [ $code -eq 0 ]; then touch "$LOGDIR/$n.ok"; fi
done < "$LIST"
touch "$LOGDIR/DONE"
