#!/usr/bin/env bash
# Lane Q5 queue: run every protocol arm through gpu_run.sh, at most two jobs of this lane at a time
# (worker 1: scalar cells, worker 2: shared-exponent cells; MobileNet block cells declared --heavy 8000).
# Each job stops itself before ~13 minutes (exit 3 = cells pending, rerun); a job that fails otherwise is
# retried once with --heavy 8000 (CUDA OOM rule), then left for the report.
# Usage: nohup tools/run/experiment_b2_seeds_queue.sh > artifacts/experiment_b2_seeds/logs/queue.out 2>&1 &
set -u
ROOT=/home/maveric/precision_exploration
cd "$ROOT"
export GPU_LANE=Q5
LOGS=$ROOT/artifacts/experiment_b2_seeds/logs
mkdir -p "$LOGS"
ARMS=${ARMS:-"seed_default seed_minimal ladder bias_only"}
MODELS="resnet18 mobilenet_v2 mobilenet_v3_large"

job() {  # model arm kind
  local model=$1 arm=$2 kind=$3 opts=() code tries=0 failures=0
  if [ "$kind" = block ]; then
    if [ "$model" = resnet18 ]; then opts=(--min-free-mib 4500); else opts=(--heavy 8000); fi
  fi
  while :; do
    tries=$((tries + 1))
    [ $tries -gt 40 ] && { echo "$(date -u +%FT%TZ) GIVEUP $model $arm $kind" >> "$LOGS/ledger.txt"; return 1; }
    artifacts/agent_orchestration/gpu_run.sh --wait 7200 "${opts[@]}" .venv-b/bin/python -m tools.run.experiment_b2_seeds \
      cells --model "$model" --arm "$arm" --kind "$kind" --max-seconds 780 >> "$LOGS/$arm-$model-$kind.log" 2>> "$LOGS/$arm-$model-$kind.err"
    code=$?
    echo "$(date -u +%FT%TZ) $model $arm $kind exit=$code" >> "$LOGS/ledger.txt"
    [ $code -eq 0 ] && return 0
    [ $code -eq 3 ] && continue
    failures=$((failures + 1))
    [ $failures -ge 2 ] && return 1
    opts=(--heavy 8000)
  done
}

worker() {  # kind
  local kind=$1
  for arm in $ARMS; do
    for model in $MODELS; do
      if [ "$kind" = block ] && [ "$arm" != seed_default ]; then continue; fi
      job "$model" "$arm" "$kind"
    done
  done
}

worker scalar &
worker block &
wait
.venv/bin/python -m tools.run.experiment_b2_seeds pending --arm seed_default > "$LOGS/pending-final.txt" 2>&1
for arm in seed_minimal ladder bias_only; do
  .venv/bin/python -m tools.run.experiment_b2_seeds pending --arm $arm >> "$LOGS/pending-final.txt" 2>&1
done
date -u +%FT%TZ > "$LOGS/QUEUE_DONE"
