"""Protocol addendum 4: tie orders for already measured configurations, and the `imageid` evaluation order.

``run`` (GPU, through gpu_run.sh) rebuilds one measured configuration, requires its identity to equal the sealed
configuration and a run record to exist, recomputes the fixed-rule detections, requires them to be bit-identical
to the stored file, and then stores the five other tie orders of the default study (``runner.TIE_ORDERS``) beside
them.  Nothing existing is written again.  A sealed record goes to ``artifacts/experiment_b2_det/tieruns/``.
Afterwards ``runner stats --images 1000`` adds their tie statistics.

``imageid`` (CPU) evaluates stored fixed-rule detections with the evaluation images in ascending COCO image_id,
the order that pycocotools uses when the original image ids are kept (the fixed rule uses ascending image
sha256).  Output: ``artifacts/experiment_b2_det/imageorder/<stem>.npz`` (point, tie statistics).

    gpu_run.sh .venv-b/bin/python -m tools.experiment_b2_det.tieorders run --format int8 --recipe conformant
    .venv-b/bin/python -m tools.experiment_b2_det.tieorders imageid --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tools.experiment_b.common import dataset, digest, file_hash, seal, unseal  # noqa: E402
from . import runner  # noqa: E402  (registers every arm)
from .arms4 import conformant_arm  # noqa: E402
from .recipe import named  # noqa: E402

PROTOCOL_ADDENDUM = "public/experiments/configs/breadth-study/b2-detector-protocol-v1-addendum-4.json"
TIE_FORMATS = ("int8", "int6", "fp6_e2m3", "bfp6")
SENTINELS = ("int8", "int6", "fp8_e4m3fn", "fp7_e3m3", "fp6_e2m3", "log8", "posit8_es1", "bfp6", "mxfp8_e4m3")
ORDER_DIR = runner.BASE / "imageorder"
RUN_DIR = runner.BASE / "tieruns"


def same_arrays(left, right):
    return set(left) == set(right) and all(np.array_equal(left[k], right[k]) for k in left)


def run(args):
    session = runner.Session(args.device)
    recipe = named(args.recipe)
    engine, configuration = session.build(args.format, recipe)
    identity = digest(configuration)
    record_path = runner.BASE / "runs" / str(args.images) / f"{args.format}--{args.recipe}--{identity[:12]}.json"
    configuration_path = runner.BASE / "configurations" / f"{identity}.json"
    index_path = runner.detection_file(identity, args.images, "index")
    if not (record_path.exists() and configuration_path.exists() and index_path.exists()):
        raise SystemExit(f"not a measured configuration (no record, sealed configuration or fixed-rule file): {identity[:12]}")
    if digest(unseal(configuration_path)) != identity or unseal(record_path)["configuration_sha256"] != identity:
        raise SystemExit("sealed configuration or run record does not match the rebuilt configuration")
    target = RUN_DIR / f"{args.format}--{args.recipe}--{identity[:12]}-{args.images}.json"
    if target.exists():
        print(json.dumps({"skipped": target.name}), flush=True)
        return 0
    tick = time.monotonic()
    orders = (("index", 0),) + runner.TIE_ORDERS
    results, _, seconds = session.evaluate(engine, args.images, orders)
    if not same_arrays(runner.load_detections(index_path), results["index"]):
        raise SystemExit(f"fixed-rule detections not reproduced bit for bit: {index_path.name}")
    files = {"index": {"sha256": file_hash(index_path), "reproduced_bit_identical": True}}
    for order, seed in runner.TIE_ORDERS:
        tag = runner.order_tag(order, seed)
        path = runner.detection_file(identity, args.images, tag)
        runner.save_detections(path, results[tag])  # refuses to overwrite different detections
        files[tag] = {"sha256": file_hash(path), "detections": int(results[tag]["counts"].astype(np.int64).sum())}
    record = {"configuration_sha256": identity, "format": args.format, "recipe_name": args.recipe,
              "images": args.images, "panel": f"screen{args.images}", "protocol_addendum": PROTOCOL_ADDENDUM,
              "detections": files, "source_sha256": session.source, "runtime": session.runtime,
              "evaluation_seconds": seconds, "total_seconds": time.monotonic() - tick, "finished_at": runner.now(),
              "interpretation": "development_evidence"}
    seal(target, record)
    print(json.dumps({"format": args.format, "recipe": args.recipe, "id": identity[:12], "orders": list(files),
                      "eval_s": round(seconds, 1)}), flush=True)
    return 0


def imageid_targets(images=1000):
    """``[(stem, source)]``: stored fixed-rule files (``source`` = npz path) and sealed v1 predictions."""
    from .summary import records, sealed_stem, stem
    found = records(images)
    targets = [(stem(found[("fp32", "fp32")], images), runner.detection_file(found[("fp32", "fp32")]["configuration_sha256"], images))]
    for name in SENTINELS:
        for recipe in dict.fromkeys(("default", conformant_arm(name))):
            if (name, recipe) in found:
                record = found[(name, recipe)]
                targets.append((stem(record, images), runner.detection_file(record["configuration_sha256"], images)))
    for v1 in ("maxabs", "percentile_99_9"):
        targets.append((sealed_stem("int8", v1, images), ("int8", v1)))
    return targets


def imageid_job(job):
    from .post import records
    from .stats import Evaluated, annotations
    stem, source, images, target = job
    _, rows, _ = dataset("coco_screen_1k")
    rows = rows[:images]
    arrays = runner.load_detections(source) if not isinstance(source, tuple) else runner.sealed_v1(*source, rows)[1]
    ids = [int(r["image_id"]) for r in rows]
    evaluated = Evaluated(annotations(), records(arrays, rows), sorted(ids))
    point = evaluated.check()
    ties = evaluated.tie_statistics()
    if max(abs(ties["stable"][0] - point[0]), abs(ties["stable"][1] - point[1])) > 1e-12:
        raise ValueError("tie-aware accumulate does not reproduce COCOeval in stable order")
    scores = arrays["score_e5"]
    ties.update(detections=int(len(scores)), distinct_scores=int(len(np.unique(scores))),
                image_order="ascending COCO image_id", within_image_order="fixed rule (as stored)")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.npz")
    np.savez_compressed(temporary, point=point, ties=json.dumps(ties))
    temporary.replace(target)
    return target.name


def imageid(args):
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for stem, source in imageid_targets(args.images):
        target = ORDER_DIR / f"{stem}-imageid.npz"
        if not target.exists():
            jobs.append((stem, source, args.images, target))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done in pool.map(imageid_job, jobs):
            print(done, flush=True)
    return 0


def load_imageid(stem):
    path = ORDER_DIR / f"{stem}-imageid.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {"point": saved["point"], "ties": json.loads(str(saved["ties"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    job = commands.add_parser("run")
    job.add_argument("--format", required=True)
    job.add_argument("--recipe", required=True)
    job.add_argument("--images", type=int, choices=(1000,), default=1000)
    job.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    job = commands.add_parser("imageid")
    job.add_argument("--images", type=int, choices=(1000,), default=1000)
    job.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    return {"run": run, "imageid": imageid}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
