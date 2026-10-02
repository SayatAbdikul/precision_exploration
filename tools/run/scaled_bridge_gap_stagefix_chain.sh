#!/usr/bin/env bash
# Corrected stage hybrids (scaled-bridge-gap-diag-1a), one shared GPU job per case; resumable.
# Marker: artifacts/scaled_bridge_gap_v1/stagefix.done (one EXIT line per case, then STAGEFIX_CHAIN_COMPLETE).
set -u
ROOT=/home/maveric/precision_exploration
cd "$ROOT"
M=$ROOT/artifacts/scaled_bridge_gap_v1/stagefix.done
for f in fp6_e3m2 fp7_e3m3 fp6_e2m3; do
  tools/run/scaled_bridge_gap.sh stagefix run "$f" 0 1000
  echo "$f EXIT=$?" >> "$M"
done
echo STAGEFIX_CHAIN_COMPLETE >> "$M"
