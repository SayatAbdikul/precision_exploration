"""Lane Q3 driver around ``tools.experiment_b2_collapse.evaluate.run_group`` (unchanged, imported).

Why: the arms of this lane peak at about 1.6 GB of GPU memory (recorded ``peak_gpu_mib``), but
on the shared GPU a job can still die of CUDA out-of-memory when other lanes grow after it was
admitted.  This driver keeps the job alive instead: on an out-of-memory error it frees the cache,
waits, and calls ``run_group`` again (records are written once, finished arms are skipped).

Options (all patch ``evaluate.GROUPS`` inside this process only; records are unchanged):

* ``--reverse``     evaluate the missing arms from the end of the group's list, so that a
                    concurrent run of the same group from the front (an earlier queue of this
                    lane) meets this one in the middle instead of repeating its arms;
* ``--skip-heavy``  leave out the ``b2`` reference arms whose recipe runs
                    ``tools.experiment_b2.engine.bias_correct`` (it holds the activations of all
                    correction images on the GPU, 5-8 GB), so that the job is not memory-heavy;
* ``--lean-maxabs`` (mode3) add, per wide-exponent format, the arm
                    ``<fmt>-default_weight_maxabs-N--global``: the recipe ``default_weight_maxabs``
                    with the lane's sequential correction ``global`` (bit-identical to
                    ``bias_correct``, checked by the unit tests and by the L6 reproductions) in
                    place of the heavy reference arm ``--b2``;
* ``--core-first``  (default for mode3; stable sort after ``--reverse``) the reference and single-switch arms first, then
                    ``global``/``none``, then the correction variants.

Exit codes: 0 all arms exist, 3 time budget reached (run again), 4 out of memory after the
in-process retries (the queue then retries once with ``--heavy 8000``).

    GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 4000 .venv-b/bin/python -m \
        tools.run.experiment_b2_collapse_drive --model mobilenet_v3_large --group mode1 --reverse --budget 420
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def patch_groups(evaluate, *, reverse, skip_heavy, lean_maxabs, core_first=False):
    from tools.experiment_b2.recipe import named
    original = dict(evaluate.GROUPS)

    def heavy(s):
        return s["correction"] == "b2" and s["recipe"] and named(s["recipe"]).bias_correction != "none"

    def wrap(group):
        def arms(model, graph=None):
            specs = original[group](model, graph) if group == "mode2" else original[group](model)
            if skip_heavy:
                specs = [s for s in specs if not heavy(s)]
            if group == "mode3" and lean_maxabs:
                specs = specs + [evaluate.spec(f"{fmt}-default_weight_maxabs-N", fmt, fmt, "default_weight_maxabs",
                                               "global") for fmt in evaluate.WIDE]
            specs = list(reversed(specs)) if reverse else specs
            if core_first:  # stable: reference / single-switch arms, then 'global' (identity), then the variants
                def rank(s):
                    if s["correction"] == "b2" or s["recipe"] == "default_weight_maxabs":
                        return 0
                    return 1 if s["correction"] in ("global", "none") else 2
                specs = sorted(specs, key=rank)
            return specs
        return arms

    evaluate.GROUPS = {group: wrap(group) for group in original}
    return original


def is_oom(error):
    return "out of memory" in str(error).lower()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="experiment_b2_collapse_drive")
    parser.add_argument("--model", required=True)
    parser.add_argument("--group", required=True, choices=("mode1", "mode2", "mode3"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--budget", type=float, default=420.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--lean-maxabs", action="store_true")
    parser.add_argument("--core-first", action="store_true", help="default for mode3 (see --no-core-first)")
    parser.add_argument("--no-core-first", action="store_true")
    parser.add_argument("--oom-retries", type=int, default=3)
    parser.add_argument("--oom-wait", type=float, default=45.0)
    args = parser.parse_args(argv)
    import torch
    from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipe names)
    from tools.experiment_b2_collapse import evaluate
    patch_groups(evaluate, reverse=args.reverse, skip_heavy=args.skip_heavy, lean_maxabs=args.lean_maxabs,
                 core_first=(args.core_first or args.group == "mode3") and not args.no_core_first)
    started = time.monotonic()
    for attempt in range(args.oom_retries + 1):
        remaining = args.budget - (time.monotonic() - started)
        if remaining <= 30:
            return 3
        try:
            return evaluate.run_group(args.model, args.group, args.device, budget_seconds=remaining, only=args.only)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not is_oom(error):
                raise
            print(json.dumps({"model": args.model, "group": args.group, "oom": attempt + 1,
                              "error": str(error)[:160]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(args.oom_wait)
    return 4


if __name__ == "__main__":
    sys.exit(main())
