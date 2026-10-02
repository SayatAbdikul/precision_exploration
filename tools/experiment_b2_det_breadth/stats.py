"""CPU statistics of the detector breadth study: bootstrap vectors, tie statistics and the `imageid` order.

Same evaluator, draw sequence and image orders as L7 (``tools.experiment_b2_det.runner._bootstrap_job`` and
``tieorders.imageid_job``), reading this study's detection store.  Writes only missing files under
``artifacts/experiment_b2_det_breadth/{bootstrap,imageorder}/``.  Run under ``nice -n 10`` with at most 6 workers:

    nice -n 10 .venv-b/bin/python -m tools.experiment_b2_det_breadth.stats --workers 6
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tools.experiment_b.common import dataset

from . import core

# Part B: paired intervals are needed only for the close-gap pairs (posit8 - int8, log8 - bfp6).
SEED_BOOTSTRAP = ("int8", "posit8_es1", "log8")


def bootstrap_path(identity, tag, images=core.IMAGES):
    return core.BASE / "bootstrap" / f"{identity}-{images}-{tag}.npz"


def imageid_path(identity, images=core.IMAGES):
    return core.BASE / "imageorder" / f"{identity}-{images}-index-imageid.npz"


def _evaluate(arrays, rows, order_ids):
    from tools.experiment_b2_det.post import records
    from tools.experiment_b2_det.stats import Evaluated, annotations
    evaluated = Evaluated(annotations(), records(arrays, rows), order_ids)
    point = evaluated.check()
    ties = evaluated.tie_statistics()
    if max(abs(ties["stable"][0] - point[0]), abs(ties["stable"][1] - point[1])) > 1e-12:
        raise ValueError("tie-aware accumulate does not reproduce COCOeval in stable order")
    scores = arrays["score_e5"]
    ties.update(detections=int(len(scores)), distinct_scores=int(len(np.unique(scores))))
    return evaluated, point, ties


def job(spec):
    identity, tag, kind, draws, target = spec
    _, rows, _ = dataset("coco_screen_1k")
    rows = rows[:core.IMAGES]
    arrays = core.load_detections(identity, core.IMAGES, tag)
    ids = [int(r["image_id"]) for r in rows]
    if kind == "imageid":
        order = sorted(ids)
    else:
        positions = list(range(len(rows)))
        if tag == "reverse":
            positions = positions[::-1]
        elif tag.startswith("random"):
            positions = [int(i) for i in np.random.default_rng(7919 + int(tag[6:])).permutation(len(rows))]
        order = [ids[i] for i in positions]
    evaluated, point, ties = _evaluate(arrays, rows, order)
    if kind == "imageid":
        ties.update(image_order="ascending COCO image_id", within_image_order="fixed rule (as stored)")
    sampled = evaluated.bootstrap() if draws else np.zeros((0, 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial.npz")
    np.savez_compressed(temporary, point=point, draws=sampled, ties=json.dumps(ties))
    temporary.replace(target)
    return target.name


def jobs():
    out = []
    for record in core.runs():
        identity = record["configuration_sha256"]
        draws = record["study_part"] in ("A", "C", "C-fine") or (record["study_part"] == "B" and record["format"] in SEED_BOOTSTRAP)
        for tag in record["detections"]:
            target = bootstrap_path(identity, tag)
            if not target.exists():
                out.append((identity, tag, "order", draws and tag == "index", target))
        if record["study_part"] == "A" and not imageid_path(identity).exists():
            out.append((identity, "index", "imageid", False, imageid_path(identity)))
    # Bootstraps last within the list so that point values arrive first.
    return sorted(out, key=lambda s: (s[3], s[4].name))


def load(identity, tag="index"):
    path = bootstrap_path(identity, tag)
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {"point": saved["point"], "draws": saved["draws"], "ties": json.loads(str(saved["ties"]))}


def load_imageid(identity):
    path = imageid_path(identity)
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {"point": saved["point"], "ties": json.loads(str(saved["ties"]))}


def main(argv=None):
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    pending = jobs()
    if args.list:
        print(json.dumps({"pending": len(pending), "bootstraps": sum(1 for p in pending if p[3])}))
        return 0
    with ProcessPoolExecutor(max_workers=min(args.workers, 6)) as pool:
        for done in pool.map(job, pending):
            print(done, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
