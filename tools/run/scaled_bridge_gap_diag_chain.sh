#!/usr/bin/env bash
# Step-4 diagnostic chain: one case per locked GPU job (protocol scaled-bridge-gap-diag-1).
set -e
ROOT=/home/maveric/precision_exploration
for F in fp6_e3m2 fp7_e3m3 fp6_e2m3; do
  "$ROOT"/tools/run/scaled_bridge_gap.sh diagnose run $F 0 1000
done
echo DIAG_CHAIN_COMPLETE
