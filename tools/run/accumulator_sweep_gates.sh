#!/usr/bin/env bash
# Lane L8 (accumulator sweep), step 2: new B2 cases and policy gates.
# Runs, one case at a time, exactly the commands of tools/run/scaled_bridge_v2_b2.sh (MB2) and
# tools/run/scaled_bridge_v2_policies.sh (MS), with the logs in artifacts/accumulator_sweep_v1/logs/gates/
# (the original scripts write and overwrite logs under artifacts/scaled_bridge_v2/logs/, which this lane does
# not own). Every engine step skips sealed evidence, so the script is resumable.
#   accumulator_sweep_gates.sh case MODEL FORMAT RECIPE     export + adapt + MB2 gate of one new case
#   accumulator_sweep_gates.sh policies "CASE ..." "POLICY ..."   MS policy panels and gates
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
DIGEST=7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92
RUN=".venv-b/bin/python -m tools.run.scaled_bridge_v2"
GPU=artifacts/agent_orchestration/gpu_run.sh
LOGS=artifacts/accumulator_sweep_v1/logs/gates; mkdir -p "$LOGS"
check_root() {
  local now; now=$($RUN root) || { echo "ROOT-FAILED"; exit 2; }
  if [ "$(basename -- "$now")" != "$DIGEST" ]; then echo "ROOT-MOVED $now"; exit 3; fi
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do sleep 2; done; }
MODE=$1; shift
check_root
if [ "$MODE" = case ]; then
  MODEL=$1; FORMAT=$2; RECIPE=$3; TAG=$MODEL-$FORMAT-$RECIPE
  OUT=$($GPU --min-free-mib 4500 --wait 7200 .venv-b/bin/python -m tools.run.experiment_b2 export --model "$MODEL" --format "$FORMAT" --recipe "$RECIPE" 2> "$LOGS/$TAG.export.err" | tail -1)
  echo "$OUT" > "$LOGS/$TAG.export.json"
  FOLDER=$(echo "$OUT" | jq -r .export) || { echo "EXPORT-FAILED $TAG"; exit 1; }
  [ -d "$FOLDER" ] || { echo "EXPORT-FAILED $TAG"; exit 1; }
  check_root
  $RUN b2-export "$FOLDER" > "$LOGS/$TAG.b2-export.log" 2>&1 || { echo "ADAPT-FAILED $TAG"; tail -3 "$LOGS/$TAG.b2-export.log"; exit 1; }
  CASE=$(ls -d artifacts/scaled_bridge_v2/exports/*/"$MODEL-$FORMAT-"*-b2 | xargs -n1 basename | grep -F -- "-$RECIPE-b2" | head -1)
  # recipe names containing '_' map to the case name the adapter chose; list it for the log
  echo "CASE $CASE"
  [ -n "$CASE" ] || { echo "CASE-NOT-FOUND $TAG"; exit 1; }
  check_root
  $RUN build > "$LOGS/build.log" 2>&1 || { echo "BUILD-FAILED"; exit 1; }
  EXPORT=$(ls artifacts/scaled_bridge_v2/exports/*/"$CASE"/export.json)
  BOOKS=$(jq -r '.payload.codebooks | keys | join(" ")' "$EXPORT")
  $GPU --min-free-mib 4500 --wait 7200 bash -c "$RUN conformance $BOOKS --case $CASE && $RUN panel $CASE wide cuda 32 && $RUN panel $CASE control cuda 32 && $RUN b2-replay $CASE 32" \
    > "$LOGS/$CASE.gpu.log" 2>&1 || echo "GPU-FAILED $CASE"
  ($RUN panel "$CASE" wide cpp 8 && $RUN panel "$CASE" control cpp 8) > "$LOGS/$CASE.cpu.log" 2>&1 || echo "CPU-FAILED $CASE"
  check_root
  $RUN b2-gate "$CASE" 2>&1 | tail -1
  echo "CASE-DONE $CASE"
elif [ "$MODE" = policies ]; then
  CASES=$1; POLICIES=$2
  $RUN build > "$LOGS/build.log" 2>&1 || { echo "BUILD-FAILED"; exit 1; }
  for CASE in $CASES; do
    for POLICY in $POLICIES; do
      throttle 2
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
  check_root
  for CASE in $CASES; do
    for POLICY in $POLICIES; do
      $RUN policy-gate "$CASE" "$POLICY" 2>&1 | tail -1
    done
  done
  echo "ALL-POLICIES-DONE"
else
  echo "usage: $0 case MODEL FORMAT RECIPE | policies \"CASES\" \"POLICIES\""; exit 64
fi
