#!/usr/bin/env bash
# Lane Q1 (MobileNet accumulator sweep), step 3: accumulator-policy gates of the MobileNet B2 cases.
#
# Runs, one case at a time, the commands of `tools/run/scaled_bridge_v2_mn.sh policies CASE "P ..."` (the PROTOCOL_MN
# policy gate: CUDA 8-image panel, CPU 8-image panel, whole-graph policy witness on the first CPU image, policy-gate)
# through the ARCHIVED engine `$M/run.sh` (digest 1f75c923..., prebuilt libraries; no build step), with the logs in
# artifacts/accumulator_sweep_mn_v1/logs/gates/. Why not tools/run/scaled_bridge_v2_policies.sh as written: it runs the
# live sources, its CUDA panels are 32 images, it does not run the whole-graph witness that the MobileNet (B2) policy
# gate requires, and it rewrites the existing artifacts/scaled_bridge_v2/logs/build.log. Every engine step skips sealed
# evidence, so the script is resumable.
#
#   accumulator_sweep_mn_gates.sh CASE "POLICY ..."
# GPU jobs go through gpu_run.sh (export GPU_LANE=Q1). CPU panels of the policies of one case run in parallel
# (one process per policy).
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
DIGEST=1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863
M=artifacts/scaled_bridge_v2/implementations/$DIGEST
RUN="$M/run.sh"
GPU=artifacts/agent_orchestration/gpu_run.sh
export GPU_LANE=${GPU_LANE:-Q1}
LOGS=artifacts/accumulator_sweep_mn_v1/logs/gates; mkdir -p "$LOGS"
CASE=$1; POLICIES=$2
NOW=$($RUN root) || { echo "ROOT-FAILED"; exit 2; }
[ "$(basename -- "$NOW")" = "$DIGEST" ] || { echo "ROOT-MOVED $NOW"; exit 3; }
CMD="true"; for P in $POLICIES; do CMD="$CMD && $RUN panel $CASE $P cuda 8"; done
$GPU --min-free-mib 3000 --wait 7200 bash -c "$CMD" > "$LOGS/$CASE.policies.gpu.log" 2>&1 || echo "GPU-FAILED $CASE policies" &
for P in $POLICIES; do
  (nice -n 10 $RUN panel "$CASE" "$P" cpp 8 && nice -n 10 $RUN graph-witness "$CASE" "$P") > "$LOGS/$CASE.$P.cpu.log" 2>&1 || echo "CPU-FAILED $CASE $P" &
done
wait
for P in $POLICIES; do
  $RUN policy-gate "$CASE" "$P" > "$LOGS/$CASE.$P.gate.log" 2>&1; echo "$(tail -1 "$LOGS/$CASE.$P.gate.log")"
done
echo "MN-POLICIES-DONE $CASE"
