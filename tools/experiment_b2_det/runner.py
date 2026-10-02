"""Experiment B2 detector runner: run, regress, stats.  One GPU process evaluates several recipes of one format."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, formats, frozen_inputs, seal, unseal
from . import arms2, frozen, arms3, arms4  # noqa: F401  (register the addendum arms, the frozen recipes, the checks)
from .recipe import DetRecipe, hardware_semantics, named

BASE = ROOT / "artifacts/experiment_b2_det"
PROTOCOL_FILE = ROOT / "public/experiments/configs/breadth-study/b2-detector-protocol-v1.json"
NUMERIC = ("tools/experiment_b2_det/recipe.py", "tools/experiment_b2_det/engine.py", "tools/experiment_b2_det/post.py",
           "tools/experiment_b2/codebook.py", "tools/experiment_b2/scales.py")
TIE_ORDERS = (("reverse", 0), ("random", 0), ("random", 1), ("random", 2), ("random", 3))


def now():
    return datetime.now(timezone.utc).isoformat()


def source_identity():
    from tools.experiment_b_ext import runner as ext
    return digest({"v1_detector_source": ext.source_identity(), "files": {p: file_hash(ROOT / p) for p in NUMERIC}})


def runtime(device):
    from tools.experiment_b import runner as prior
    env = prior.runtime(device)
    env.update({"ultralytics": version("ultralytics"), "pycocotools": version("pycocotools")})
    return env


def order_tag(order, seed):
    return order if order != "random" else f"random{seed}"


def detection_file(identity, images, tag="index"):
    return BASE / "detections" / f"{identity}-{images}-{tag}.npz"


def save_detections(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            if not all(np.array_equal(saved[k], arrays[k]) for k in arrays):
                raise ValueError(f"refusing to overwrite different detections: {path}")
        return
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_detections(path):
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}


class Session:
    """Model, calibration observations and inputs shared by the recipes of one process."""

    def __init__(self, device):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from tools.experiment_b.classifier import configure
        from tools.experiment_b_ext.detector import load_detector
        from .engine import v1_calibration
        configure(device)
        self.device = device
        self.context = frozen_inputs("yolov8n")
        self.graph, self.constants, _ = load_detector(device)
        self.samples, self.maxima, self.calibration = v1_calibration(self.graph, self.context)
        self.runtime, self.source = runtime(device), source_identity()
        self.protocol = json.loads(PROTOCOL_FILE.read_text())["version"]
        self._bias_inputs = None

    def bias_inputs(self):
        from tools.experiment_b_ext.detector import image_batch
        from .engine import BIAS_CORRECTION_IMAGES
        if self._bias_inputs is None:
            _, rows, payload = dataset("coco_calibration_2k")
            self._bias_inputs = image_batch(rows[:BIAS_CORRECTION_IMAGES], payload, self.device)[0]
        return self._bias_inputs

    def build(self, name, recipe):
        import torch
        from .engine import B2DetectorEngine, prepare
        from .post import RULE
        base = {"model_context": self.context, "protocol": self.protocol, "runtime": self.runtime,
                "source_sha256": self.source, "postprocess": RULE}
        if name == "fp32":
            return B2DetectorEngine(self.graph, self.constants, self.device), {**base, "format": "fp32"}
        entry = {e["name"]: e for e in formats()}[name]
        with torch.no_grad():
            engine, metadata = prepare(self.graph, self.constants, name, recipe, self.samples, self.maxima, self.device,
                                       bias_inputs=self.bias_inputs() if recipe.bias_correction == "empirical" else None)
        return engine, {**base, "format": name, "format_sha256": entry["sha256"], "recipe": recipe.as_dict(),
                        "hardware_semantics": hardware_semantics(recipe), "calibration": self.calibration,
                        "engine": metadata}

    def evaluate(self, engine, images, orders=(("index", 0),), legacy=False):
        """Returns ``{tag: compact arrays}`` over the first ``images`` screen images (batches of 4, as v1)."""
        import torch
        from tools.experiment_b_ext.detector import image_batch
        from .post import legacy_nms, merge, ordered_nms, pack
        _, rows, payload = dataset("coco_screen_1k")
        packed = {order_tag(*o): [] for o in orders}
        if legacy:
            packed["legacy"] = []
        seconds = 0.0
        for start in range(0, images, 4):
            inputs, shapes = image_batch(rows[start:start + 4], payload, self.device)
            tick = time.monotonic()
            with torch.inference_mode():
                output = engine.run(inputs)
                for order, seed in orders:
                    packed[order_tag(order, seed)] += pack(ordered_nms(output, order=order, seed=seed, first_ordinal=start), shapes)
                if legacy:
                    packed["legacy"] += pack(legacy_nms(output), shapes)
            seconds += time.monotonic() - tick
        return {tag: merge(p) for tag, p in packed.items()}, rows[:images], seconds


def point_metrics(arrays, rows, truth):
    from .post import records
    from .stats import Evaluated
    return [float(x) for x in Evaluated(truth, records(arrays, rows), [int(r["image_id"]) for r in rows]).metrics()]


def run(args):
    from .stats import annotations
    session, truth = Session(args.device), annotations()
    for recipe_name in args.recipes:
        recipe = None if args.format == "fp32" else named(recipe_name)
        label = "fp32" if recipe is None else recipe_name
        tick = time.monotonic()
        engine, configuration = session.build(args.format, recipe)
        identity = digest(configuration)
        target = BASE / "runs" / str(args.images) / f"{args.format}--{label}--{identity[:12]}.json"
        orders = (("index", 0),) + (TIE_ORDERS if args.ties else ())
        if target.exists() and all(detection_file(identity, args.images, order_tag(*o)).exists() for o in orders):
            print(json.dumps({"skipped": target.name}), flush=True)
            continue
        seal(BASE / "configurations" / f"{identity}.json", configuration)
        prepared = time.monotonic() - tick
        results, rows, seconds = session.evaluate(engine, args.images, orders)
        files = {}
        for tag, arrays in results.items():
            save_detections(detection_file(identity, args.images, tag), arrays)
            files[tag] = {"sha256": file_hash(detection_file(identity, args.images, tag)),
                          "detections": int(arrays["counts"].astype(np.int64).sum())}
        metrics = point_metrics(results["index"], rows, truth)
        record = {"configuration_sha256": identity, "format": args.format, "recipe_name": label, "images": args.images,
                  "panel": "dev128" if args.images == 128 else f"screen{args.images}", "detections": files,
                  "map50_95": metrics[0], "map50": metrics[1], "preparation_seconds": prepared,
                  "evaluation_seconds": seconds, "finished_at": now(), "interpretation": "development_evidence"}
        if not target.exists():
            seal(target, record)
        print(json.dumps({"format": args.format, "recipe": label, "images": args.images, "id": identity[:12],
                          "map50_95": round(100 * metrics[0], 3), "map50": round(100 * metrics[1], 3),
                          "prep_s": round(prepared, 1), "eval_s": round(seconds, 1)}), flush=True)
    return 0


def sealed_v1(name, v1_recipe, rows):
    """Sealed v1 detections of one format and recipe (or the FP32 baseline) on ``rows``, as compact arrays."""
    from tools.experiment_b_ext import runner as ext
    from .post import from_sealed
    folder, matches = ROOT / "artifacts/experiment_b_ext", []
    for path in sorted((folder / "summaries").glob("*-1000.json")):
        identity = path.name.split("-")[0]
        configuration = unseal(folder / "configurations" / f"{identity}.json")
        if (configuration["model_context"].get("model") == "yolov8n" and configuration["source_sha256"] == ext.source_identity()
                and (name == "fp32" or (configuration["format"] == name and configuration["recipe"] == v1_recipe))):
            matches.append(configuration["baseline_sha256"] if name == "fp32" else identity)
    if len(set(matches)) != 1:
        raise ValueError(f"expected one sealed v1 detector configuration for {name} {v1_recipe}, found {len(set(matches))}")
    identity, per_image = matches[0], []
    for row in rows:
        record = unseal(folder / "predictions" / identity / f"{row['sha256']}.json")
        if record["configuration_sha256"] != identity or record["sample"] != row:
            raise ValueError("sealed v1 prediction provenance mismatch")
        per_image.append(record["detections"])
    return identity, from_sealed(per_image)


def regress(args):
    """Every switch off: bit-identical head tensors against v1's engine and identical sealed detections."""
    import torch
    from tools.experiment_b_ext.detector import DetectorEngine, image_batch, prepare_detector
    from .post import same
    session = Session(args.device)
    _, rows, payload = dataset("coco_screen_1k")
    rows = rows[:args.images]
    if args.format == "fp32":
        mine, _ = session.build("fp32", None)
        theirs = DetectorEngine(session.graph, session.constants, args.device)
    else:
        recipe = named("v1_maxabs" if args.v1 == "maxabs" else "v1_percentile_99_9")
        if recipe.v1_equivalent() != args.v1:
            raise ValueError("not a v1-equivalent recipe")
        mine, _ = session.build(args.format, recipe)
        theirs, _ = prepare_detector(session.graph, session.constants, args.format, args.v1, session.samples,
                                     session.maxima, args.device)
    identical = 0
    for start in range(0, len(rows), 4):
        inputs, _ = image_batch(rows[start:start + 4], payload, args.device)
        with torch.inference_mode():
            identical += int(torch.equal(mine.run(inputs), theirs.run(inputs))) * len(inputs)
    results, _, _ = session.evaluate(mine, len(rows), legacy=True)
    identity, sealed = sealed_v1(args.format, args.v1, rows)
    counts = {tag: np.concatenate(([0], np.cumsum(a["counts"].astype(np.int64)))) for tag, a in {**results, "sealed": sealed}.items()}

    def per_image(left, right):
        hits = 0
        for i in range(len(rows)):
            a, b = slice(counts[left][i], counts[left][i + 1]), slice(counts[right][i], counts[right][i + 1])
            source_l, source_r = {**results, "sealed": sealed}[left], {**results, "sealed": sealed}[right]
            hits += int(all(np.array_equal(source_l[k][a], source_r[k][b]) for k in ("category", "box_milli", "score_e5")))
        return hits
    record = {"format": args.format, "v1_recipe": args.v1 if args.format != "fp32" else None, "images": len(rows),
              "sealed_v1_configuration": identity, "images_with_bit_identical_head_output": identical,
              "legacy_postprocess_equals_sealed_images": per_image("legacy", "sealed"),
              "fixed_rule_equals_sealed_images": per_image("index", "sealed"),
              "all_detections_equal_sealed": same(results["legacy"], sealed),
              "source_sha256": session.source, "runtime": session.runtime, "finished_at": now()}
    record["passed"] = identical == len(rows) and record["all_detections_equal_sealed"]
    target = BASE / "regression" / f"{args.format}--{args.v1 if args.format != 'fp32' else 'baseline'}--{len(rows)}.json"
    if not target.exists():
        seal(target, record)
    print(json.dumps({k: v for k, v in record.items() if k != "runtime"}), flush=True)
    return 0 if record["passed"] else 1


def _bootstrap_job(job):
    from .post import records
    from .stats import Evaluated, annotations
    path, images, tag, target, mode = job
    _, rows, _ = dataset("coco_screen_1k")
    rows = rows[:images]
    arrays = load_detections(path) if mode == "npz" else sealed_v1(*path, rows)[1]
    ids = [int(r["image_id"]) for r in rows]
    order = list(range(len(rows)))
    if tag == "reverse":
        order = order[::-1]
    elif tag.startswith("random"):
        order = [int(i) for i in np.random.default_rng(7919 + int(tag[6:])).permutation(len(rows))]
    detections = records(arrays, rows)
    evaluated = Evaluated(annotations(), detections, [ids[i] for i in order])
    point = evaluated.check()
    ties = evaluated.tie_statistics()
    if max(abs(ties["stable"][0] - point[0]), abs(ties["stable"][1] - point[1])) > 1e-12:
        raise ValueError("tie-aware accumulate does not reproduce COCOeval in stable order")
    scores = arrays["score_e5"]
    ties.update(detections=int(len(scores)), distinct_scores=int(len(np.unique(scores))))
    draws = evaluated.bootstrap() if tag == "index" or mode == "sealed" else np.zeros((0, 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.npz")
    np.savez_compressed(temporary, point=point, draws=draws, ties=json.dumps(ties))
    temporary.replace(target)
    return target.name


def stats(args):
    """CPU: bootstrap vectors and tie statistics for every stored detection file that lacks them."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for path in sorted((BASE / "detections").glob(f"*-{args.images}-*.npz")):
        identity, images, tag = path.stem.split("-")
        target = BASE / "bootstrap" / f"{path.stem}.npz"
        if not target.exists() and ".partial" not in path.name:
            jobs.append((path, int(images), tag, target, "npz"))
    for name, v1_recipe in args.sealed or ():
        target = BASE / "bootstrap" / f"v1--{name}--{v1_recipe}-{args.images}-sealed.npz"
        if not target.exists():
            jobs.append(((name, v1_recipe), args.images, "index", target, "sealed"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done in pool.map(_bootstrap_job, jobs):
            print(done, flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    job = commands.add_parser("run")
    job.add_argument("--format", required=True)
    job.add_argument("--recipes", nargs="+", default=["fp32"])
    job.add_argument("--images", type=int, choices=(128, 1000), default=128)
    job.add_argument("--ties", action="store_true")
    job.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    job = commands.add_parser("regress")
    job.add_argument("--format", required=True)
    job.add_argument("--v1", choices=("maxabs", "percentile_99_9"), default="maxabs")
    job.add_argument("--images", type=int, default=128)
    job.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    job = commands.add_parser("stats")
    job.add_argument("--images", type=int, choices=(128, 1000), default=128)
    job.add_argument("--workers", type=int, default=8)
    job.add_argument("--sealed", nargs=2, action="append", metavar=("FORMAT", "V1_RECIPE"))
    commands.add_parser("recipes")
    args = parser.parse_args()
    if args.command == "recipes":
        from .recipe import NAMED
        print(json.dumps({k: {"switches": v.as_dict(), "hardware": hardware_semantics(v)} for k, v in NAMED.items()}, indent=1))
        return 0
    return {"run": run, "regress": regress, "stats": stats}[args.command](args)
