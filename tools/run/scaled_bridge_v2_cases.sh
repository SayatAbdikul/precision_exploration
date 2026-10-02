#!/usr/bin/env bash
# Resumable case runner. Each line of the case file: MODEL FORMAT RECIPE PANEL REPLAY
# GPU work for one case is a single job under the shared GPU lock (export,
# primitive conformance, B replay, both CUDA arms). CPU arms run in a second
# loop without the lock, then every case is gated.
# Usage: scaled_bridge_v2_cases.sh CASE_FILE
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
CASES=$1
PY=.venv-b/bin/python
LOCK=artifacts/agent_orchestration/gpu.lock
RUN="$PY -m tools.run.scaled_bridge_v2"
ROOT_DIR=$($RUN root)
LOGS=artifacts/scaled_bridge_v2/logs; mkdir -p "$LOGS"
gpu_loop() {
  while read -r MODEL FORMAT RECIPE PANEL REPLAY; do
    [ -z "${MODEL:-}" ] && continue
    CASE="${MODEL}-${FORMAT}-${RECIPE}-b1"
    [ -f "$ROOT_DIR/$CASE/gpu.done.$PANEL" ] && continue
    CONF=""; [ -f "$ROOT_DIR/conformance/$FORMAT.json" ] || CONF="$RUN conformance $FORMAT &&"
    SCALED_BRIDGE_V2_GPU_LOCKED=1 flock -w 3600 "$LOCK" bash -c "$RUN export $MODEL $FORMAT $RECIPE && $CONF $RUN replay $CASE $REPLAY && $RUN panel $CASE wide cuda $PANEL && $RUN panel $CASE control cuda $PANEL" \
      > "$LOGS/$CASE.gpu.log" 2>&1 && touch "$ROOT_DIR/$CASE/gpu.done.$PANEL" || echo "GPU-FAILED $CASE"
  done < "$CASES"
}
cpu_loop() {
  while read -r MODEL FORMAT RECIPE PANEL REPLAY; do
    [ -z "${MODEL:-}" ] && continue
    CASE="${MODEL}-${FORMAT}-${RECIPE}-b1"
    [ -f "$ROOT_DIR/$CASE/cpu.done" ] && continue
    EXPORT=$(ls artifacts/scaled_bridge_v2/exports/*/"$CASE"/export.json 2>/dev/null | head -1)
    for _ in $(seq 1 1440); do [ -n "$EXPORT" ] && break; sleep 5; EXPORT=$(ls artifacts/scaled_bridge_v2/exports/*/"$CASE"/export.json 2>/dev/null | head -1); done
    ($RUN panel "$CASE" wide cpp 8 && $RUN panel "$CASE" control cpp 8) > "$LOGS/$CASE.cpu.log" 2>&1 \
      && touch "$ROOT_DIR/$CASE/cpu.done" || echo "CPU-FAILED $CASE"
  done < "$CASES"
}
gpu_loop & cpu_loop & wait
while read -r MODEL FORMAT RECIPE PANEL REPLAY; do
  [ -z "${MODEL:-}" ] && continue
  CASE="${MODEL}-${FORMAT}-${RECIPE}-b1"
  $RUN gate "$CASE" 2>&1 | tail -1
done < "$CASES"
echo "ALL-CASES-DONE"
