#!/usr/bin/env bash
# Profile only this validation process; do not change the GPU counter policy.
set -euo pipefail
phase2_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$phase2_root"
exec .venv/bin/python -m tools.run.phase2_profile_counters "$@"
