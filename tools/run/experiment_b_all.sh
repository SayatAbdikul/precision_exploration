#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
.venv-b/bin/python -m tools.run.experiment_b --run
.venv-b/bin/python -m tools.run.experiment_b_ext --run
