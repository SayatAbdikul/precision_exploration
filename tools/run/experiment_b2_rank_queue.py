"""Queue of lane Q2 cells: groups the pending cells of the plan into GPU jobs and runs at most 2 at a time.

  .venv/bin/python -m tools.run.experiment_b2_rank_queue [--max-priority 5] [--parallel 2] [--dry-run] [--tag T]

Each job is ``GPU_LANE=Q2 gpu_run.sh <memory options> .venv-b/bin/python -m tools.run.experiment_b2_rank cells ...``.
A failed group is retried once in a second pass with ``--heavy 8000`` (cells already written are skipped).
Ledger: artifacts/experiment_b2_rank/logs/queue-<tag>.jsonl; stderr of jobs: queue-<tag>.err.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time

from tools.experiment_b.common import ROOT
from tools.experiment_b2_rank.cells import RANK, own_cell
from tools.experiment_b2_rank.plan import BLOCK, cells

SCALAR_GROUP, BLOCK_GROUP, HEAVY_SCALAR_GROUP = 14, 4, 36
BC_RECIPES = ("default", "cum5_act_maxabs", "rank_intrinsic_weight_maxabs", "default_weight_maxabs")


def pending(max_priority):
    out = []
    for cell in cells():
        if cell["source"] != "new" or cell["priority"] > max_priority:
            continue
        if own_cell(cell["part"], cell["model"], cell["format"], cell["recipe"], cell["variant"]) is None:
            out.append(cell)
    return out


def has_bias_correction(cell):
    from tools.experiment_b2.recipe import NAMED
    return NAMED[cell["recipe"]].bias_correction != "none"


def memory_class(cell):
    if cell["format"] not in BLOCK:
        # measured 2026-10-02 (queue r1, MobileNetV2): bias-correction cells peak at 4.8-4.9 GB allocated, the others
        # at 0.5 GB; a ResNet18 bias-correction cell died of CUDA OOM at the default admission -> declared heavy
        return "scalar_bc" if has_bias_correction(cell) else "scalar"
    if cell["model"] == "resnet18" and has_bias_correction(cell) and cell["part"] != "searched":
        return "block_bc"  # 3.1-4.3 GB peak; two such groups died of CUDA OOM at --min-free-mib 4000
    if cell["model"] != "resnet18" and cell["recipe"] in BC_RECIPES:
        return "heavy"
    if cell["part"] == "searched" or (cell["variant"] == "unsigned" and cell["recipe"] == "default"):
        return "heavy" if cell["model"] != "resnet18" else "block"
    return "block"


HEAVY_CLASSES = ("heavy", "scalar_bc", "block_bc")
# seconds per heavy cell (measured on this queue where available, 2026-10-02) for packing jobs of at most PACK_SECONDS
SECONDS = {("scalar_bc", "resnet18"): 45, ("scalar_bc", "mobilenet_v2"): 25, ("scalar_bc", "mobilenet_v3_large"): 30,
           ("block_bc", "resnet18"): 120, ("heavy", "mobilenet_v2"): 200, ("heavy", "mobilenet_v3_large"): 250,
           ("heavy", "resnet18"): 120}
PACK_SECONDS = 780


def groups(cells_):
    keyed, heavy = {}, {}
    for cell in cells_:
        memory = memory_class(cell)
        if memory in HEAVY_CLASSES:
            # the heavy lock is contended by every lane: every heavy cell of a model (all parts and priorities) is
            # packed into as few jobs of at most ~13 minutes as possible, to take the lock as rarely as possible
            heavy.setdefault(cell["model"], []).append(cell)
            continue
        key = (cell["priority"], cell["model"], cell["part"], memory)
        keyed.setdefault(key, []).append(cell)
    out = []
    for key in sorted(keyed, key=lambda k: (k[0], k[1], k[2], k[3])):
        size = SCALAR_GROUP if key[3] == "scalar" else BLOCK_GROUP
        if key[2] == "searched" and key[1] != "resnet18":
            size = 1  # a searched MobileNet block cell takes 5-12 minutes on its own
        items = keyed[key]
        for i in range(0, len(items), size):
            out.append((key, items[i:i + size]))
    packs = []
    for model in sorted(heavy):
        pack, total = [], 0
        for cell in sorted(heavy[model], key=lambda c: (memory_class(c), c["priority"], c["part"])):
            cost = SECONDS[(memory_class(cell), model)]
            if pack and total + cost > PACK_SECONDS:
                packs.append(((0, model, "mixed", _pack_class(pack)), pack))
                pack, total = [], 0
            pack.append(cell)
            total += cost
        if pack:
            packs.append(((0, model, "mixed", _pack_class(pack)), pack))
    # scalar packs first (factorial corners and unsigned scalar variants), then those with block cells
    packs.sort(key=lambda job: (any(c["format"] in BLOCK for c in job[1]), min(c["priority"] for c in job[1])))
    return out + packs


def _pack_class(pack):
    return "heavy" if any(memory_class(c) == "heavy" for c in pack) else "heavy6000"


def interleave(work):
    """Heavy jobs run one at a time across all lanes: alternate heavy and light groups so that the two parallel
    workers of this queue rarely wait on each other's heavy reservation (priority order kept within each kind)."""
    heavy = [w for w in work if w[0][3] in ("heavy", "scalar_bc")]
    light = [w for w in work if w[0][3] not in ("heavy", "scalar_bc")]
    out = []
    while heavy or light:
        for side in (heavy, light):
            if side:
                out.append(side.pop(0))
    return out


def command(key, items, heavy=False):
    _, model, part, memory = key
    options = ["--wait", "21600" if memory in ("heavy", "heavy6000") or heavy else "7200"]
    if heavy or memory == "heavy":
        options += ["--heavy", "8000"]
    elif memory == "heavy6000":
        options += ["--heavy", "6000"]
    elif memory == "block":
        options += ["--min-free-mib", "4000"]
    else:
        options += ["--min-free-mib", "4500"]  # as L1's matrix campaign (a scalar default cell OOMed at the default 2500)
    jobs = ",".join(f"{c['format']}:{c['recipe']}:{c['variant']}:{c['part']}" for c in items)
    return [str(ROOT / "artifacts/agent_orchestration/gpu_run.sh"), *options, str(ROOT / ".venv-b/bin/python"), "-m",
            "tools.run.experiment_b2_rank", "cells", "--model", model, "--jobs", jobs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-priority", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", default="r1")
    args = parser.parse_args()
    os.environ["GPU_LANE"] = "Q2"
    logs = RANK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        work = groups(pending(args.max_priority))
        if args.dry_run:
            for key, items in work:
                print(key, len(items), " ".join(command(key, items))[-160:])
            return 0
        if not work:
            break

        def launch(job):
            key, items = job
            full = command(key, items, heavy=attempt == 2)
            tick = time.time()
            with (logs / f"queue-{args.tag}.err").open("a") as err:
                result = subprocess.run(full, cwd=ROOT, stdout=subprocess.PIPE, stderr=err, text=True)
            entry = {"attempt": attempt, "key": list(key), "cells": len(items), "returncode": result.returncode,
                     "wall_seconds_including_wait": round(time.time() - tick, 1), "command": full[1:],
                     "output": result.stdout.strip().splitlines()}
            with (logs / f"queue-{args.tag}.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
            return result.returncode

        # one worker walks the heavy groups (they run one at a time across all lanes anyway), the other the light
        # groups, so a quick light queue never leaves both workers waiting on the heavy reservation
        heavy = [w for w in work if w[0][3] in ("heavy", "heavy6000")]
        light = [w for w in work if w[0][3] not in ("heavy", "heavy6000")]
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            lanes = [pool.submit(lambda items: [launch(j) for j in items], items) for items in (heavy, light) if items]
            codes = [c for lane in lanes for c in lane.result()]
        print(f"attempt {attempt}: jobs={len(work)} failures={sum(c != 0 for c in codes)}", flush=True)
    left = pending(args.max_priority)
    print(f"QUEUE DONE pending={len(left)}", flush=True)
    (logs / f"queue-{args.tag}.done").write_text(json.dumps({"pending": len(left), "time": time.time()}) + "\n")
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
