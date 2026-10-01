#!/usr/bin/env bash
# Resume immutable experiment checkpoints, then verify and report the evidence.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
.venv-b/bin/python -m tools.run.scaled_bridge run
exec .venv-b/bin/python -m tools.analysis.scaled_bridge_audit
