"""Input-mean drift diagnostic of lane Q3 (protocol addendum 3; correction images only, no evaluation).

    GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 .venv-b/bin/python -m \
        tools.run.experiment_b2_collapse_drift --model resnet18 [--tag r1]
Writes artifacts/experiment_b2_collapse/drift/input-drift--<model>--<tag>.json once.  Exit 5: a rebuilt
arm's deployed state differs from its sealed record (the file is still written, with the flag).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="experiment_b2_collapse_drift")
    parser.add_argument("--model", required=True, choices=("resnet18", "mobilenet_v2"))
    parser.add_argument("--tag", default="r1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    from tools.experiment_b2_collapse.drift import run_model
    return run_model(args.model, args.device, args.tag, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
