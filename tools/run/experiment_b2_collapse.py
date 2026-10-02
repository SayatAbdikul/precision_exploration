"""CLI of lane Q3 (B2 collapse diagnosis).

    GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.run.experiment_b2_collapse \
        eval --model resnet18 --group mode1 [--only CELL_OR_LABEL ...] [--budget 600]   # exit 3: run again
    .venv/bin/python -m tools.run.experiment_b2_collapse weights            # CPU, mode-2 weight analysis
    GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.run.experiment_b2_collapse \
        cell --model M --format F --recipes R1,R2                            # matrix cell redirected to this lane
    .venv/bin/python -m tools.run.experiment_b2_collapse report --tag r1     # CPU, summaries and figures (once)
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="experiment_b2_collapse")
    sub = parser.add_subparsers(dest="job", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--model", required=True)
    p.add_argument("--group", required=True, choices=("mode1", "mode2", "mode3"))
    p.add_argument("--only", nargs="*")
    p.add_argument("--budget", type=float, default=600.0)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("cell")
    p.add_argument("--model", required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--recipes", required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("weights")
    p.add_argument("--tag", default="r1")
    p = sub.add_parser("report")
    p.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    if args.job == "eval":
        from tools.experiment_b2_collapse.evaluate import run_group
        return run_group(args.model, args.group, args.device, budget_seconds=args.budget, only=args.only)
    if args.job == "cell":
        from tools.experiment_b2_collapse.cells import run_cells
        return run_cells(args.model, args.format, args.recipes.split(","), args.device)
    if args.job == "weights":
        from tools.experiment_b2_collapse.weights import main as weights_main
        return weights_main(args.tag)
    if args.job == "report":
        from tools.experiment_b2_collapse.report import main as report_main
        return report_main(args.tag)
    return 2


if __name__ == "__main__":
    sys.exit(main())
