#!/usr/bin/env bash
# Accumulator-policy panels and gates (protocol MS). Resumable.
# Usage: scaled_bridge_v2_policies.sh "CASE ..." "POLICY ..." [OLD_RUN_DIGEST_PREFIX]
# GPU panels go through artifacts/agent_orchestration/gpu_run.sh (shared mode, up to 4 jobs at once);
# CPU panels run two at a time. With OLD_RUN the wide and control arms are first regressed against that run.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
CASES=$1; POLICIES=$2; OLD=${3:-}
RUN=".venv-b/bin/python -m tools.run.scaled_bridge_v2"
GPU=artifacts/agent_orchestration/gpu_run.sh
ROOT_DIR=$($RUN root); LOGS=artifacts/scaled_bridge_v2/logs; mkdir -p "$LOGS"
# Build both libraries once, before any parallel job: concurrent first builds of a new source digest race.
$RUN build > "$LOGS/build.log" 2>&1 || { echo "BUILD-FAILED"; exit 1; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do sleep 2; done; }
for CASE in $CASES; do
  if [ -n "$OLD" ]; then
    throttle 4
    $GPU bash -c "$RUN regress $CASE wide cuda 8 $OLD && $RUN regress $CASE control cuda 8 $OLD" > "$LOGS/$CASE.regress.log" 2>&1 || echo "REGRESS-FAILED $CASE" &
  fi
  for POLICY in wide $POLICIES; do
    throttle 4
    $GPU $RUN panel "$CASE" "$POLICY" cuda 32 > "$LOGS/$CASE.$POLICY.cuda.log" 2>&1 || echo "GPU-FAILED $CASE $POLICY" &
  done
done
wait
for CASE in $CASES; do
  for POLICY in $POLICIES; do
    throttle 2
    $RUN panel "$CASE" "$POLICY" cpp 8 > "$LOGS/$CASE.$POLICY.cpp.log" 2>&1 || echo "CPU-FAILED $CASE $POLICY" &
  done
done
wait
for CASE in $CASES; do
  for POLICY in $POLICIES; do
    $RUN policy-gate "$CASE" "$POLICY" 2>&1 | tail -1
  done
done
echo "ALL-POLICIES-DONE"
