#!/usr/bin/env bash
# Sequential 1k chain: one case/arm per locked GPU job; stops at the first failure.
set -e
ROOT=/home/maveric/precision_exploration
for F in fp6_e2m3 fp6_e3m2 fp7_e3m3; do
  "$ROOT"/tools/run/scaled_bridge_gap.sh replay $F 0 1000
  "$ROOT"/tools/run/scaled_bridge_gap.sh arm $F wide 0 1000
  "$ROOT"/tools/run/scaled_bridge_gap.sh arm $F control 0 1000
done
echo CHAIN_COMPLETE
