#!/usr/bin/env bash
# Lane Q4 (agent r3): run the lines of a job list through the shared GPU launcher, low-memory path.
#   tools/run/experiment_b2_attrib_lm_queue.sh LIST LOGDIR [MIN_FREE_MIB]
# Each line holds the arguments of tools.run.experiment_b2_attrib_lm ("gate --model M --format F" or "arms ...").
# Stop switch: the queue exits before its next job when LOGDIR/STOP or artifacts/experiment_b2_attrib/lowmem/STOP
# exists (the second also makes a running arms job stop before its next arm).
# Per job: LOGDIR/<n>.log, LOGDIR/<n>.ok on exit 0, a line in LOGDIR/queue.jsonl; LOGDIR/DONE at the end.
# A line starting with 'HEAVY ' runs with --heavy 8000; a CUDA out-of-memory failure is retried once with --heavy 8000
# (rule 7; LOGDIR/<n>.oom.log keeps the first log).
# Exit 3 of an arms job = some arms deferred (time cap or an old-path job computing them): rerun the list later.
set -u
ROOT=/home/maveric/precision_exploration
LIST=$1
LOGDIR=$2
MINFREE=${3:-2000}
export GPU_LANE=Q4
mkdir -p "$LOGDIR"
cd "$ROOT"
n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n+1))
  if [ -f "$LOGDIR/$n.ok" ]; then continue; fi
  if [ -f "$LOGDIR/STOP" ] || [ -f artifacts/experiment_b2_attrib/lowmem/STOP ]; then
    echo "{\"stopped_before\": $n}" >> "$LOGDIR/queue.jsonl"; exit 0
  fi
  start=$(date +%s)
  launch="--min-free-mib $MINFREE"
  case "$line" in "HEAVY "*) line=${line#HEAVY }; launch="--heavy 8000";; esac
  artifacts/agent_orchestration/gpu_run.sh $launch --wait 7200 \
    .venv-b/bin/python -m tools.run.experiment_b2_attrib_lm $line > "$LOGDIR/$n.log" 2>&1
  code=$?
  if [ "$launch" != "--heavy 8000" ] && [ $code -ne 0 ] && grep -q "OutOfMemoryError" "$LOGDIR/$n.log"; then
    mv "$LOGDIR/$n.log" "$LOGDIR/$n.oom.log"
    artifacts/agent_orchestration/gpu_run.sh --heavy 8000 --wait 7200 \
      .venv-b/bin/python -m tools.run.experiment_b2_attrib_lm $line > "$LOGDIR/$n.log" 2>&1
    code=$?
  fi
  end=$(date +%s)
  printf '{"n": %d, "args": "%s", "exit": %d, "wall_seconds_including_slot_wait": %d, "end_unix": %d}\n' \
    "$n" "$line" "$code" "$((end-start))" "$end" >> "$LOGDIR/queue.jsonl"
  if [ $code -eq 0 ]; then touch "$LOGDIR/$n.ok"; fi
done < "$LIST"
touch "$LOGDIR/DONE"
