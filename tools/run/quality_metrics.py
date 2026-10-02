"""Lane Q7 entry point (CPU only): quality metrics, resolution and ties from stored records.

  .venv/bin/python -m tools.run.quality_metrics det-jobs [--workers 4]     # parts B/C/D detector workers
  .venv/bin/python -m tools.run.quality_metrics analyse --out DIR         # every summary into DIR (written once)
  .venv/bin/python -m tools.run.quality_metrics addendum1 --out DIR       # addendum-1 summaries from the v1 summaries
  .venv/bin/python -m tools.run.quality_metrics addendum2 --out DIR       # addendum-2 summaries from the v1 summaries

Protocol: public/experiments/configs/breadth-study/quality-metrics-protocol-v1.json.
"""
from __future__ import annotations

import argparse
import os
import sys


def det_jobs(args):
    from concurrent.futures import ProcessPoolExecutor
    from tools.analysis.quality_metrics_v1 import detector
    specs = detector.configurations()
    if args.only:
        specs = [s for s in specs if f"{s[0]}--{s[1]}" in args.only.split(",")]
    # longest jobs first
    specs.sort(key=lambda s: (s not in detector.SUBSAMPLE, s not in detector.AREA_BOOT))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done in pool.map(detector.job, specs):
            print(done, flush=True)
    return 0


def analyse(args):
    from tools.analysis.quality_metrics_v1 import report
    return report.main(args.out, parts=args.parts.split(",") if args.parts else None)


def addendum1(args):
    from tools.analysis.quality_metrics_v1 import addendum1 as A
    return A.main(args.out, v1=args.v1, knife_edge=not args.no_knife_edge)


def addendum2(args):
    from tools.analysis.quality_metrics_v1 import addendum2 as A
    return A.main(args.out, v1=args.v1, exact=not args.no_exact)


def main(argv=None):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    parser = argparse.ArgumentParser(prog="quality_metrics")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("det-jobs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", default="")
    p.set_defaults(func=det_jobs)
    p = sub.add_parser("analyse")
    p.add_argument("--out", required=True)
    p.add_argument("--parts", default="")
    p.set_defaults(func=analyse)
    p = sub.add_parser("addendum1")
    p.add_argument("--out", required=True)
    p.add_argument("--v1", default="results/summaries/quality-metrics-v1")
    p.add_argument("--no-knife-edge", action="store_true")
    p.set_defaults(func=addendum1)
    p = sub.add_parser("addendum2")
    p.add_argument("--out", required=True)
    p.add_argument("--v1", default="results/summaries/quality-metrics-v1")
    p.add_argument("--no-exact", action="store_true")
    p.set_defaults(func=addendum2)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
