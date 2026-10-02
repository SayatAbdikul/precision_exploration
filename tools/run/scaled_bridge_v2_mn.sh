#!/usr/bin/env bash
# PROTOCOL_MN (contract 2.2) runner for one MobileNet repaired-recipe case at a time. Resumable: every step skips
# sealed evidence. One B2 job (replay) at a time: run this script for one case, wait, then the next.
#
#   scaled_bridge_v2_mn.sh case CASE              conformance (codebooks + 2.2 operators), wide/control CUDA 32,
#                                                 B2 replay 32 (one GPU job); wide/control CPU 8 (in parallel);
#                                                 whole-graph control witness (CPU); MB2/MN case gate
#   scaled_bridge_v2_mn.sh policies CASE "P ..."  per policy: CUDA 8 (one GPU job for all), CPU 8, whole-graph
#                                                 policy witness, policy gate
# The adapted export must exist (python -m tools.run.scaled_bridge_v2 b2-export <L1 export folder>).
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
MODE=$1; CASE=$2
RUN=".venv-b/bin/python -m tools.run.scaled_bridge_v2"
GPU=artifacts/agent_orchestration/gpu_run.sh
LOGS=artifacts/scaled_bridge_v2/logs/mn; mkdir -p "$LOGS"
$RUN build > "$LOGS/build.log" 2>&1 || { echo "BUILD-FAILED"; exit 1; }
if [ "$MODE" = case ]; then
  EXPORT=$(ls artifacts/scaled_bridge_v2/exports/*/"$CASE"/export.json)
  BOOKS=$(jq -r '.payload.codebooks | keys | join(" ")' "$EXPORT")
  $GPU bash -c "$RUN conformance $BOOKS --case $CASE && $RUN panel $CASE wide cuda 32 && $RUN panel $CASE control cuda 32 && $RUN b2-replay $CASE 32" \
    > "$LOGS/$CASE.gpu.log" 2>&1 || echo "GPU-FAILED $CASE" &
  ($RUN panel "$CASE" wide cpp 8 && $RUN panel "$CASE" control cpp 8 && $RUN graph-witness "$CASE" control) \
    > "$LOGS/$CASE.cpu.log" 2>&1 || echo "CPU-FAILED $CASE" &
  wait
  $RUN b2-gate "$CASE" > "$LOGS/$CASE.gate.log" 2>&1; tail -1 "$LOGS/$CASE.gate.log"
  echo "MN-CASE-DONE $CASE"
elif [ "$MODE" = policies ]; then
  POLICIES=$3
  CMD="true"; for P in $POLICIES; do CMD="$CMD && $RUN panel $CASE $P cuda 8"; done
  $GPU bash -c "$CMD" > "$LOGS/$CASE.policies.gpu.log" 2>&1 || echo "GPU-FAILED $CASE policies" &
  for P in $POLICIES; do
    ($RUN panel "$CASE" "$P" cpp 8 && $RUN graph-witness "$CASE" "$P") > "$LOGS/$CASE.$P.cpu.log" 2>&1 || echo "CPU-FAILED $CASE $P" &
  done
  wait
  for P in $POLICIES; do
    $RUN policy-gate "$CASE" "$P" > "$LOGS/$CASE.$P.gate.log" 2>&1; tail -1 "$LOGS/$CASE.$P.gate.log"
  done
  echo "MN-POLICIES-DONE $CASE"
else
  echo "usage: $0 case CASE | policies CASE \"POLICIES\""; exit 2
fi
