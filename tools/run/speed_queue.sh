#!/usr/bin/env bash
# Lane S1: run the GPU commands of a queue file one after another, each through gpu_run.sh (GPU_LANE=S1).
# Usage: tools/run/speed_queue.sh QUEUE_FILE   (one line = gpu_run.sh options and command; '#' lines skipped)
# Log per line: artifacts/speed_v1/logs/queue/<queue name>-<line no>.log; marker <queue name>.done with exit codes.
set -u
cd "$(dirname "$0")/../.." || exit 1
export GPU_LANE=S1
Q="$1"; NAME="$(basename "$Q" .txt)"; LOGS=artifacts/speed_v1/logs/queue; mkdir -p "$LOGS"
n=0; codes=""
while IFS= read -r line || [ -n "$line" ]; do
  n=$((n+1)); case "$line" in ''|'#'*) continue;; esac
  echo "$(date +%T) start $n: $line" >> "$LOGS/$NAME.progress"
  eval "artifacts/agent_orchestration/gpu_run.sh $line" > "$LOGS/$NAME-$n.log" 2>&1
  code=$?; codes="$codes $n:$code"
  echo "$(date +%T) end $n exit $code" >> "$LOGS/$NAME.progress"
done < "$Q"
echo "EXIT$codes" > "$LOGS/$NAME.done"
