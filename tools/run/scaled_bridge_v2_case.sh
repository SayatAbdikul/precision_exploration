#!/usr/bin/env bash
# One case end to end: export, conformance, CPU and CUDA gate panels, B replay,
# gate. GPU work is one locked job per case (rule: one configuration per lock).
# Usage: scaled_bridge_v2_case.sh MODEL FORMAT RECIPE [PANEL_IMAGES]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
MODEL=$1; FORMAT=$2; RECIPE=$3; PANEL=${4:-32}
CASE="${MODEL}-${FORMAT}-${RECIPE}-b1"
PY=.venv-b/bin/python
LOCK=artifacts/agent_orchestration/gpu.lock
RUN="$PY -m tools.run.scaled_bridge_v2"
# CPU arms first (no GPU needed once the export exists).
SCALED_BRIDGE_V2_GPU_LOCKED=1 flock -w 3600 "$LOCK" bash -c "$RUN export $MODEL $FORMAT $RECIPE && $RUN conformance $FORMAT"
$RUN panel "$CASE" wide cpp 8
$RUN panel "$CASE" control cpp 8
SCALED_BRIDGE_V2_GPU_LOCKED=1 flock -w 3600 "$LOCK" bash -c "$RUN replay $CASE 32 && $RUN panel $CASE wide cuda $PANEL && $RUN panel $CASE control cuda $PANEL"
$RUN gate "$CASE"
echo "CASE-DONE $CASE"
