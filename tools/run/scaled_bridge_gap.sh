#!/usr/bin/env bash
# One GPU job of the 1k gap study.  Usage: scaled_bridge_gap.sh <action> [args...]
# Runs through the orchestrator's shared-slot launcher artifacts/agent_orchestration/gpu_run.sh
# (this lane reports no throughput claims, so shared mode is right; it yields to exclusive jobs).
# The 1k arm runs (2026-10-01, before this change) used the exclusive flock form; their jobs.jsonl rows have no "mode".
# Logs requested/acquired/finished epochs and exit code to artifacts/scaled_bridge_gap_v1/jobs.jsonl
# ("locked_seconds" = seconds holding the GPU slot).
set -u
ROOT=/home/maveric/precision_exploration
GPU_RUN=$ROOT/artifacts/agent_orchestration/gpu_run.sh
LOGDIR=$ROOT/artifacts/scaled_bridge_gap_v1/logs
mkdir -p "$LOGDIR"
TAG=$(echo "$*" | tr ' ' '_')
LOG=$LOGDIR/$TAG.$(date +%s).log
REQ=$(date +%s.%N)
cd "$ROOT"
"$GPU_RUN" bash -c '
  ACQ=$(date +%s.%N)
  "$0"/.venv-b/bin/python -m tools.run.scaled_bridge_gap "${@:3}" >"$1" 2>&1
  CODE=$?
  END=$(date +%s.%N)
  printf "{\"args\":\"%s\",\"mode\":\"shared\",\"requested\":%s,\"acquired\":%s,\"finished\":%s,\"locked_seconds\":%s,\"exit\":%s,\"log\":\"%s\"}\n" \
    "${*:3}" "$2" "$ACQ" "$END" "$(echo "$END - $ACQ" | bc)" "$CODE" "$1" >> "$0"/artifacts/scaled_bridge_gap_v1/jobs.jsonl
  exit $CODE
' "$ROOT" "$LOG" "$REQ" "$@"
RC=$?
if [ $RC -ne 0 ]; then echo "job failed or no GPU slot ($RC): $*; log $LOG"; fi
tail -n 3 "$LOG" 2>/dev/null
exit $RC
