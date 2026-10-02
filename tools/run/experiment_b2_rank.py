"""Lane Q2 cells (tools/experiment_b2_rank).  GPU jobs: run under artifacts/agent_orchestration/gpu_run.sh.

  cells --model M --part P --jobs fmt:recipe[:variant[:part]],...   run cells into artifacts/experiment_b2_rank/P/
      (variant plain|unsigned; an item's own part overrides --part)
  reproduce --model M --format F --recipe R [--engine matrix|blocks_unsigned_off] --out FILE
      redirect one L1 cell into artifacts/experiment_b2_rank/reproduction/ and compare it bit for bit
"""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations (as the matrix CLI).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    import argparse
    import json
    from tools.experiment_b.common import ROOT
    from tools.experiment_b2.common import MODELS
    from tools.experiment_b2_rank import cells
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--part", choices=("factorial", "unsigned", "searched"), default="factorial")
    p.add_argument("--jobs", required=True)
    p = sub.add_parser("reproduce")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--engine", choices=("matrix", "blocks_unsigned_off"), default="matrix")
    p.add_argument("--out", required=True)
    for name in ("cells", "reproduce"):
        sub.choices[name].add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    protocol = ROOT / "public/experiments/configs/breadth-study/b2-rank-protocol-v1.json"
    if not protocol.exists():
        raise SystemExit("the Q2 protocol must be on disk before anything is measured")
    if args.command == "cells":
        jobs = []
        for item in args.jobs.split(","):
            parts = item.split(":")
            jobs.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "plain",
                         parts[3] if len(parts) > 3 else args.part))
        return cells.run_jobs(args.model, jobs, args.device, args.part)
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    result = cells.reproduce(args.model, args.format, args.recipe, cells.RANK / "reproduction", args.device, args.engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
