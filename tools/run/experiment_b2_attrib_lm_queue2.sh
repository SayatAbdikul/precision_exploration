#!/usr/bin/env bash
# Lane Q4 (agent r4): like experiment_b2_attrib_lm_queue.sh (low-memory path), plus a lane-wide concurrency gate:
# before each job it waits until at most one other Q4 GPU job holds a ticket (gpu.tickets/*.Q4.*), so that this
# queue together with the lane's other launchers keeps at most 2 Q4 jobs on the GPU (rule 7).
#   tools/run/experiment_b2_attrib_lm_queue2.sh LIST LOGDIR [MIN_FREE_MIB]
# Stop switch: exits before its next job when LOGDIR/STOP, artifacts/experiment_b2_attrib/lowmem/queues/z/STOP or
# artifacts/experiment_b2_attrib/lowmem/STOP exists (the last also stops a running arms job between arms).
# Per job: LOGDIR/<n>.log, LOGDIR/<n>.ok on exit 0, a line in LOGDIR/queue.jsonl; LOGDIR/DONE at the end.
# 'HEAVY ' lines run with --heavy 8000; a CUDA out-of-memory failure is retried once with --heavy 8000 (rule 7).
# Exit 3 of an arms job = some arms deferred (time cap): the line is not marked ok; rerunning the list resumes it.
set -u
ROOT=/home/maveric/precision_exploration
LIST=$1
LOGDIR=$2
MINFREE=${3:-3000}
export GPU_LANE=Q4
TICKETS=$ROOT/artifacts/agent_orchestration/gpu.tickets
mkdir -p "$LOGDIR"
cd "$ROOT"
stopped() { [ -f "$LOGDIR/STOP" ] || [ -f artifacts/experiment_b2_attrib/lowmem/queues/z/STOP ] || [ -f artifacts/experiment_b2_attrib/lowmem/STOP ]; }
q4_jobs() { local c=0 f p; for f in "$TICKETS"/*.Q4.*; do [ -e "$f" ] || continue; p=${f##*/}; p=${p%%.*}; [ -d "/proc/$p" ] && c=$((c+1)); done; echo $c; }
n=0
exec 3< "$LIST"
while IFS= read -r line <&3 || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n+1))
  if [ -f "$LOGDIR/$n.ok" ]; then continue; fi
  while :; do
    if stopped; then echo "{\"stopped_before\": $n}" >> "$LOGDIR/queue.jsonl"; exit 0; fi
    [ "$(q4_jobs)" -le 1 ] && break
    sleep 20
  done
  start=$(date +%s)
  launch="--min-free-mib $MINFREE"
  case "$line" in "HEAVY "*) line=${line#HEAVY }; launch="--heavy 8000";; esac
  artifacts/agent_orchestration/gpu_run.sh $launch --wait 7200 \
    .venv-b/bin/python -m tools.run.experiment_b2_attrib_lm $line > "$LOGDIR/$n.log" 2>&1
  code=$?
  if [ "$launch" != "--heavy 8000" ] && [ $code -ne 0 ] && grep -qE "OutOfMemoryError|CUDA error: out of memory" "$LOGDIR/$n.log"; then
    mv "$LOGDIR/$n.log" "$LOGDIR/$n.oom.log"
    artifacts/agent_orchestration/gpu_run.sh --heavy 8000 --wait 7200 \
      .venv-b/bin/python -m tools.run.experiment_b2_attrib_lm $line > "$LOGDIR/$n.log" 2>&1
    code=$?
  fi
  end=$(date +%s)
  printf '{"n": %d, "args": "%s", "exit": %d, "wall_seconds_including_slot_wait": %d, "end_unix": %d}\n' \
    "$n" "$line" "$code" "$((end-start))" "$end" >> "$LOGDIR/queue.jsonl"
  if [ $code -eq 0 ]; then touch "$LOGDIR/$n.ok"; fi
done
touch "$LOGDIR/DONE"
