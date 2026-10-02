"""Lane Q3 r4: redirected B2 matrix cells with the low-GPU-memory block bias correction.

    GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 4000 .venv-b/bin/python -m \
        tools.run.experiment_b2_collapse_lowmem --model resnet18 --format mxfp6_e3m2 --recipes default,default_weight_maxabs
    (MobileNetV2: --validate default --recipes default_weight_maxabs; identity-only check, no evaluation)

Cells go to artifacts/experiment_b2_collapse/matrix/ (never artifacts/experiment_b2/matrix/); when the original
matrix holds the same identity, the reproduction check is written to matrix/checks/<cell>--lowmem.json.
Bit-identity of the correction: tools/experiment_b2_collapse/lowmem.py and tests/unit/test_experiment_b2_collapse_lowmem.py.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="experiment_b2_collapse_lowmem")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--recipes", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate", default=None, help="recipe whose identity is rebuilt and checked first")
    args = parser.parse_args(argv)
    from tools.experiment_b2_collapse.lowmem import run_cells_lowmem
    return run_cells_lowmem(args.model, args.format, args.recipes.split(","), args.device, validate=args.validate)


if __name__ == "__main__":
    sys.exit(main())
