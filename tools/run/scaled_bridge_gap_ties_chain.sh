#!/usr/bin/env bash
# Post-hoc tie audit (protocol scaled-bridge-gap-ties-1): one shared GPU job per case.
# A case's tie job runs after its diagnostic job has finished, so every configuration's Top-5 is cross-checked
# against the diagnostic record.
set -u
ROOT=/home/maveric/precision_exploration
cd "$ROOT"
RC=0
for F in fp6_e3m2 fp7_e3m3 fp6_e2m3; do
  while [ "$(ls artifacts/scaled_bridge_gap_v1/diagnostics/$F 2>/dev/null | wc -l)" -lt 125 ]; do sleep 30; done
  "$ROOT"/tools/run/scaled_bridge_gap.sh ties run $F 0 1000 || RC=1
done
echo TIES_CHAIN_COMPLETE rc=$RC
