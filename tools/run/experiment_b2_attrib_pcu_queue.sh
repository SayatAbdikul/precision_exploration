#!/usr/bin/env bash
# Lane Q4 (agent r6, protocol addendum 8): queue of pcu arms jobs through the low-memory path.
#   tools/run/experiment_b2_attrib_pcu_queue.sh LIST LOGDIR [MIN_FREE_MIB]
# Each LIST line: ESTIMATED_SECONDS then the args of `python -m tools.run.experiment_b2_attrib_pcu`.
# Before each job: stop switch (LOGDIR/STOP or artifacts/experiment_b2_attrib/pcu/STOP), the addendum-8 GPU cap
# (pcurunner budget ESTIMATED_SECONDS; a job that would exceed the cap is skipped and logged), and a lane-wide gate
# (at most one other Q4 GPU ticket alive, so the lane keeps at most 2 GPU jobs; rule 7).
# Per job: LOGDIR/<n>.log, LOGDIR/<n>.ok on exit 0, a line in LOGDIR/queue.jsonl; LOGDIR/DONE at the end.
# A CUDA out-of-memory failure is retried once with --heavy 8000 (rule 7).
set -u
ROOT=/home/maveric/precision_exploration
LIST=$1
LOGDIR=$2
MINFREE=${3:-3000}
export GPU_LANE=Q4
export PYTHONPATH=$ROOT
TICKETS=$ROOT/artifacts/agent_orchestration/gpu.tickets
mkdir -p "$LOGDIR"
cd "$ROOT"
stopped() { [ -f "$LOGDIR/STOP" ] || [ -f artifacts/experiment_b2_attrib/pcu/STOP ]; }
q4_jobs() { local c=0 f p; for f in "$TICKETS"/*.Q4.*; do [ -e "$f" ] || continue; p=${f##*/}; p=${p%%.*}; [ -d "/proc/$p" ] && c=$((c+1)); done; echo $c; }
n=0
exec 3< "$LIST"
while IFS= read -r line <&3 || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n+1))
  if [ -f "$LOGDIR/$n.ok" ]; then continue; fi
  est=${line%% *}
  args=${line#* }
  while :; do
    if stopped; then echo "{\"stopped_before\": $n}" >> "$LOGDIR/queue.jsonl"; exit 0; fi
    [ "$(q4_jobs)" -le 1 ] && break
    sleep 20
  done
  if ! .venv/bin/python -m tools.experiment_b2_attrib.pcurunner budget "$est" >> "$LOGDIR/budget.log" 2>&1; then
    printf '{"n": %d, "args": "%s", "skipped": "addendum-8 GPU cap"}\n' "$n" "$args" >> "$LOGDIR/queue.jsonl"
    continue
  fi
  start=$(date +%s)
  artifacts/agent_orchestration/gpu_run.sh --min-free-mib "$MINFREE" --wait 7200 \
    .venv-b/bin/python -m tools.run.experiment_b2_attrib_pcu $args > "$LOGDIR/$n.log" 2>&1
  code=$?
  if [ $code -ne 0 ] && grep -qE "OutOfMemoryError|CUDA error: out of memory" "$LOGDIR/$n.log"; then
    mv "$LOGDIR/$n.log" "$LOGDIR/$n.oom.log"
    artifacts/agent_orchestration/gpu_run.sh --heavy 8000 --wait 7200 \
      .venv-b/bin/python -m tools.run.experiment_b2_attrib_pcu $args > "$LOGDIR/$n.log" 2>&1
    code=$?
  fi
  end=$(date +%s)
  printf '{"n": %d, "args": "%s", "exit": %d, "wall_seconds_including_slot_wait": %d, "end_unix": %d}\n' \
    "$n" "$args" "$code" "$((end-start))" "$end" >> "$LOGDIR/queue.jsonl"
  if [ $code -eq 0 ]; then touch "$LOGDIR/$n.ok"; fi
done
touch "$LOGDIR/DONE"
