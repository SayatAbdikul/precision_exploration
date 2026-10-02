"""Detector breadth study (lane Q6): a wrapper around the L7 detector runner (``tools.experiment_b2_det``).

Nothing of L7 is edited.  This module adds three things around ``runner.Session``:

* calibration subsets built from the sealed v1 per-batch observations (``batches``, ``subset``),
* "wide group" attribution arms: a patched boundary plan and FP32 weight constants for one group of nodes,
  applied only inside this process while one engine is built (``attribution_plan``, ``BreadthSession.build``),
* a compact store for tie-order detections relative to the fixed-rule file (``encode_relative``).

Every output goes to ``artifacts/experiment_b2_det_breadth/``.  Protocol:
``public/experiments/configs/breadth-study/b2-detector-breadth-protocol-v1.json``.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import re
import time

import numpy as np

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, seal, unseal
from tools.experiment_b2_det import engine as det_engine
from tools.experiment_b2_det import runner
from tools.experiment_b2_det.arms4 import conformant_arm
from tools.experiment_b2_det.recipe import named

BASE = ROOT / "artifacts/experiment_b2_det_breadth"
L7 = runner.BASE
PROTOCOL = "public/experiments/configs/breadth-study/b2-detector-breadth-protocol-v1.json"
IMAGES = 1000

PART_A_FORMATS = ("int5", "int4", "q1_6", "fp8_e5m2", "fp6_e3m2", "fp5_e2m2", "fp4_e2m1", "mxfp6_e3m2", "mxfp4_e2m1",
                  "posit6_es1", "posit4_es0", "log6", "log4", "nf4", "ternary", "binary_pm1")
L7_SENTINELS = ("int8", "int6", "fp8_e4m3fn", "fp7_e3m3", "fp6_e2m3", "log8", "posit8_es1", "bfp6", "mxfp8_e4m3")
CHANCE_MAP = 1.0  # stop rule, mAP50-95 points under `default`

SUBSET_SEED, SUBSET_COUNT, BATCH = 20261002, 5, 8
SEED_FORMATS = ("int8", "int6", "fp6_e2m3", "posit8_es1", "log8", "fp8_e4m3fn")

ATTRIBUTION_FORMATS = ("int6", "fp6_e2m3")


# ----------------------------------------------------------------------------------------------- paths and records
def detection_file(identity, images, tag="index"):
    suffix = "" if tag == "index" else ".ref"
    return BASE / "detections" / f"{identity}-{images}-{tag}{suffix}.npz"


def run_record(fmt, label, identity, images=IMAGES):
    safe = label.replace("/", "_").replace(":", "_")
    return BASE / "runs" / str(images) / f"{fmt}--{safe}--{identity[:12]}.json"


def runs(images=IMAGES):
    folder = BASE / "runs" / str(images)
    return [unseal(p) for p in sorted(folder.glob("*.json"))] if folder.exists() else []


def l7_runs(images=IMAGES):
    return {(r["format"], r["recipe_name"]): r for r in (unseal(p) for p in sorted((L7 / "runs" / str(images)).glob("*.json")))}


# ------------------------------------------------------------------------------ compact tie-order detections
def encode_relative(reference, arrays):
    """Detections of another tie order as references into the fixed-rule detections of the same images.

    ``ref[j]`` is the global row of the fixed-rule file holding the identical detection (same image, category, box
    and score; each reference row used at most once), or -1 if there is none; those rows are stored in ``extra_*``.
    The reference is stored as ``ref - j`` (mostly small and piecewise constant, so it compresses well).
    """
    if len(reference["counts"]) != len(arrays["counts"]):
        raise ValueError("relative store needs the same image list")
    rc = np.concatenate(([0], np.cumsum(reference["counts"].astype(np.int64))))
    ac = np.concatenate(([0], np.cumsum(arrays["counts"].astype(np.int64))))
    ref = np.full(int(ac[-1]), -1, dtype=np.int64)
    for i in range(len(arrays["counts"])):
        lookup = {}
        for r in range(rc[i], rc[i + 1]):
            key = (int(reference["category"][r]), *map(int, reference["box_milli"][r]), int(reference["score_e5"][r]))
            lookup.setdefault(key, []).append(r)
        for j in range(ac[i], ac[i + 1]):
            key = (int(arrays["category"][j]), *map(int, arrays["box_milli"][j]), int(arrays["score_e5"][j]))
            if lookup.get(key):
                ref[j] = lookup[key].pop(0)
    extra = ref < 0
    shift = np.where(extra, np.iinfo(np.int32).min, ref - np.arange(len(ref))).astype(np.int32)
    encoded = {"counts": arrays["counts"], "shift": shift, "extra_category": arrays["category"][extra],
               "extra_box_milli": arrays["box_milli"][extra], "extra_score_e5": arrays["score_e5"][extra]}
    if not all(np.array_equal(decode_relative(reference, encoded)[k], arrays[k]) for k in arrays):
        raise ValueError("relative tie-order store does not round-trip")
    return encoded


def decode_relative(reference, encoded):
    shift = encoded["shift"].astype(np.int64)
    extra = shift == np.iinfo(np.int32).min
    ref = np.where(extra, 0, shift + np.arange(len(shift)))
    out = {"counts": encoded["counts"]}
    for key in ("category", "box_milli", "score_e5"):
        values = reference[key][ref].copy()
        values[extra] = encoded[f"extra_{key}"]
        out[key] = values
    return out


def save_npz(path, arrays):
    """Write once; an existing file must hold the same arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            if set(saved.files) != set(arrays) or not all(np.array_equal(saved[k], arrays[k]) for k in arrays):
                raise ValueError(f"refusing to overwrite different detections: {path}")
        return
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_detections(identity, images=IMAGES, tag="index", base=None):
    """Fixed-rule or tie-order detections (decoded) of one configuration, from this study or (``base=L7``) L7's."""
    if base is not None:
        return runner.load_detections(runner.detection_file(identity, images, tag)) if base == L7 else None
    index = runner.load_detections(detection_file(identity, images))
    if tag == "index":
        return index
    with np.load(detection_file(identity, images, tag)) as saved:
        return decode_relative(index, {k: saved[k] for k in saved.files})


# --------------------------------------------------------------------------------- calibration subsets (part B)
def batches(calibration_identity):
    """Per-batch v1 observations ``[(start, samples{key: array}, maxima{key: float})]`` with the v1 checks."""
    record, rows, _ = dataset("coco_calibration_2k")
    folder = det_engine.V1_CALIBRATION / calibration_identity
    summary = unseal(folder / "summary.json")
    if summary["identity"] != folder.name or summary["images"] != len(rows):
        raise ValueError("calibration summary does not match the frozen calibration list")
    out = []
    for start in range(0, len(rows), BATCH):
        meta = unseal(folder / f"{start:05d}.json")
        npz = folder / f"{start:05d}.npz"
        if meta["identity"] != folder.name or meta["samples"] != rows[start:start + BATCH] or file_hash(npz) != meta["npz_sha256"]:
            raise ValueError("v1 detector calibration checkpoint mismatch")
        with np.load(npz, allow_pickle=False) as saved:
            out.append((start, {k: saved[k] for k in saved.files}, dict(meta["maxima"])))
    return out


def subset_batches(count=SUBSET_COUNT, seed=SUBSET_SEED, total=250):
    """``[sorted batch positions of subset k]``: a seeded partition of the 250 calibration batches."""
    order = np.random.default_rng(seed).permutation(total)
    size = total // count
    return [sorted(int(x) for x in order[k * size:(k + 1) * size]) for k in range(count)]


def subset_observations(per_batch, positions):
    """Samples and maxima of the batches at ``positions`` (ascending), combined exactly as ``v1_calibration``."""
    combined, maxima = {}, {}
    for position in sorted(positions):
        _, samples, batch_maxima = per_batch[position]
        for key, value in samples.items():
            combined.setdefault(key, []).append(value)
        for key, value in batch_maxima.items():
            maxima[key] = max(maxima.get(key, 0), value)
    return {k: np.concatenate(v) for k, v in combined.items()}, maxima


def subset_calibration(base_calibration, per_batch, name, positions):
    """Calibration identity entry of a subset; the full set keeps the original entry (identity reproduces L7)."""
    if len(positions) == len(per_batch):
        return dict(base_calibration)
    _, rows, _ = dataset("coco_calibration_2k")
    images = [r["sha256"] for p in sorted(positions) for r in rows[p * BATCH:(p + 1) * BATCH]]
    return {**base_calibration, "subset": {"name": name, "seed": SUBSET_SEED, "count": SUBSET_COUNT,
                                           "batch_positions": sorted(positions), "images": len(images),
                                           "image_sha256_digest": digest(images), "protocol": PROTOCOL}}


# ------------------------------------------------------------------------------------- attribution (part C)
GROUP_ORDER = ("stem", "stage_p2", "stage_p3", "stage_p4", "stage_p5", "sppf", "neck", "head_p3", "head_p4", "head_p5",
               "dfl", "scores")
_MODULE = re.compile(r"^model_(\d+)(?:_|$)")


def module_index(name):
    match = _MODULE.match(name)
    return int(match.group(1)) if match else None


def group_nodes(graph, group):
    """Node names (and ``images``) of one wide group of the protocol."""
    names = [n["name"] for n in graph["nodes"]]
    by_module = lambda modules: {n for n in names if module_index(n) in modules}  # noqa: E731
    level = {"head_p3": 0, "head_p4": 1, "head_p5": 2}
    if group == "stem":
        return {"images"} | by_module({0, 1})
    if group in ("stage_p2", "stage_p3", "stage_p4", "stage_p5", "sppf"):
        return by_module({"stage_p2": {2}, "stage_p3": {3, 4}, "stage_p4": {5, 6}, "stage_p5": {7, 8}, "sppf": {9}}[group])
    if group == "neck":
        return by_module(set(range(10, 22)))
    if group in level:
        prefix = f"model_22_level_{level[group]}_"
        return {n for n in names if n.startswith(prefix + "box_logits") or n.startswith(prefix + "class_logits")}
    if group == "dfl":
        return {n for n in names if re.match(r"^model_22_level_\d_dfl$", n)}
    if group == "scores":
        return {n for n in names if re.match(r"^model_22_level_\d_scores$", n)}
    raise ValueError(f"unknown attribution group: {group}")


def fine_units(graph, group):
    """``{unit: node set}`` of the finer stage inside one group (fixed in the protocol)."""
    nodes = group_nodes(graph, group)
    pick = lambda test: {n for n in nodes if test(n)}  # noqa: E731
    if group == "stem":
        return {"input": {"images"}, "model_0": pick(lambda n: module_index(n) == 0), "model_1": pick(lambda n: module_index(n) == 1)}
    if group == "stage_p2":
        return {"model_2_cv1": pick(lambda n: n.startswith("model_2_cv1")), "model_2_m_0": pick(lambda n: n.startswith("model_2_m_0")),
                "model_2_split_cat_cv2": pick(lambda n: not n.startswith(("model_2_cv1", "model_2_m_0")))}
    if group in ("stage_p3", "stage_p4", "stage_p5"):
        return {f"model_{m}": pick(lambda n, m=m: module_index(n) == m) for m in sorted({module_index(n) for n in nodes})}
    if group == "sppf":
        return {"sppf_cv1": pick(lambda n: n.startswith("model_9_cv1")),
                "sppf_pools_cat": pick(lambda n: n.startswith(("model_9_pool", "model_9_cat"))),
                "sppf_cv2": pick(lambda n: n.startswith("model_9_cv2"))}
    if group == "neck":
        units = ((10, 11), (12,), (13, 14), (15,), (16, 17), (18,), (19, 20), (21,))
        return {"model_" + "-".join(map(str, u)): pick(lambda n, u=u: module_index(n) in u) for u in units}
    if group.startswith("head_p"):
        branch = lambda b, k: pick(lambda n: any(f"_{b}_{i}" in n for i in k))  # noqa: E731
        return {"box_convs_0_1": branch("box_logits", (0, 1)), "box_logits": branch("box_logits", (2,)),
                "class_convs_0_1": branch("class_logits", (0, 1)), "class_logits": branch("class_logits", (2,))}
    if group in ("dfl", "scores"):
        return {f"level_{i}": pick(lambda n, i=i: n.startswith(f"model_22_level_{i}_")) for i in range(3)}
    raise ValueError(f"unknown attribution group: {group}")


def wide_constants(graph, wide):
    """Weight constants kept FP32 for a wide node set: its convolutions' weights and its DFL projection constants."""
    return sorted(n["inputs"][1] for n in graph["nodes"] if n["name"] in wide and n["op"] in ("conv2d", "dfl"))


def attribution_plan(graph, plan, recipe, format_name, wide):
    """The engine's plan with the ``wide`` boundaries not quantized; pass-through nodes whose source left the grid
    quantize with their own range (as ``engine.analyze`` does for a source that is not on the grid)."""
    from tools.experiment_b2.codebook import supports_unsigned
    if not wide:
        return plan
    plan = {k: dict(v) for k, v in plan.items()}
    unsigned_ok = recipe.unsigned and format_name not in det_engine.BLOCK_FORMATS and supports_unsigned(format_name)
    if "images" in wide:
        plan["images"].update(quantizes=False, reason="attribution_wide", on_grid=False, signedness=None, groups=None)
    for node in graph["nodes"]:
        name, row = node["name"], plan[node["name"]]
        sources = [plan[x] for x in node["inputs"] if x in plan]
        if name in wide:
            row.update(quantizes=False, reason="attribution_wide", on_grid=False, signedness=None, groups=None)
        elif row["reason"] == "code_passthrough" and not sources[0]["on_grid"]:
            row.update(quantizes=True, reason=None, on_grid=True, groups=None,
                       signedness="unsigned" if unsigned_ok and row["nonnegative"] else "signed")
        elif row["reason"] == "concat_passthrough" and not all(s["on_grid"] for s in sources):
            row.update(quantizes=True, reason=None, on_grid=True, groups=None,
                       signedness="unsigned" if unsigned_ok and row["nonnegative"] else "signed")
        elif row["reason"] == "passthrough":
            row["on_grid"] = sources[0]["on_grid"]
    return plan


@contextmanager
def patched_analyze(graph, wide):
    """Inside this process only: ``engine.analyze`` returns the attribution plan while one engine is built."""
    original = det_engine.analyze

    def analyze(g, recipe, format_name):
        return attribution_plan(g, original(g, recipe, format_name), recipe, format_name, wide)
    det_engine.analyze = analyze
    try:
        yield
    finally:
        det_engine.analyze = original


# ------------------------------------------------------------------------------------------------- session
class BreadthSession(runner.Session):
    """L7's session (model, sealed calibration, inputs) with calibration subsets and attribution groups."""

    def __init__(self, device):
        super().__init__(device)
        self.full_samples, self.full_maxima, self.full_calibration = self.samples, self.maxima, self.calibration
        self._batches = None

    def use_subset(self, name=None, positions=None):
        """Select a calibration subset (``None``: the full sealed set, exactly as L7)."""
        if name is None:
            self.samples, self.maxima, self.calibration = self.full_samples, self.full_maxima, self.full_calibration
            return
        if self._batches is None:
            self._batches = batches(self.full_calibration["identity"])
        self.samples, self.maxima = subset_observations(self._batches, positions)
        self.calibration = subset_calibration(self.full_calibration, self._batches, name, positions)

    def build_arm(self, fmt, recipe_name, wide=None, group=None):
        """``(engine, configuration, identity)``; ``wide`` (a node set) builds an attribution arm."""
        recipe = None if fmt == "fp32" else named(recipe_name)
        if not wide:
            engine, configuration = self.build(fmt, recipe)
            return engine, configuration, digest(configuration)
        with patched_analyze(self.graph, frozenset(wide)):
            engine, configuration = self.build(fmt, recipe)
        constants = wide_constants(self.graph, wide)
        for key in constants:
            engine.constants[key] = self.constants[key]
        configuration = {**configuration, "attribution": {
            "group": group, "wide_nodes": sorted(wide), "fp32_constants": constants, "protocol": PROTOCOL,
            "rule": "wide boundaries not quantized, their weight constants FP32; pass-through outside the group whose "
                    "source left the grid quantizes with its own observed range; diagnostic only"}}
        return engine, configuration, digest(configuration)


def map_points(arrays, rows, truth):
    return runner.point_metrics(arrays, rows, truth)


def measure(session, truth, fmt, recipe_name, label, *, ties="never", wide=None, group=None, part, extra=None,
            images=IMAGES):
    """Build, evaluate, store (write-once) and seal one run record; returns it.

    ``ties``: "never" (fixed rule only), "always" (the six GPU orders in one pass) or "unless_chance" (fixed rule
    first; the six orders in a second pass over the same engine only if mAP50-95 >= ``CHANCE_MAP`` points, with the
    fixed-rule detections of both passes required to be bit-identical).
    """
    tick = time.monotonic()
    engine, configuration, identity = session.build_arm(fmt, recipe_name, wide=wide, group=group)
    target = run_record(fmt, label, identity, images)
    if target.exists():
        print(json.dumps({"skipped": target.name}), flush=True)
        return unseal(target)
    seal(BASE / "configurations" / f"{identity}.json", configuration)
    prepared = time.monotonic() - tick
    all_orders = (("index", 0),) + runner.TIE_ORDERS
    results, rows, seconds = session.evaluate(engine, images, all_orders if ties == "always" else (("index", 0),))
    metrics = map_points(results["index"], rows, truth)
    passes = 1
    if ties == "unless_chance" and 100 * metrics[0] >= CHANCE_MAP:
        second, _, more = session.evaluate(engine, images, all_orders)
        if not all(np.array_equal(second["index"][k], results["index"][k]) for k in results["index"]):
            raise SystemExit(f"fixed-rule detections differ between two passes: {fmt} {label}")
        results, seconds, passes = second, seconds + more, 2
    index_path = detection_file(identity, images)
    save_npz(index_path, results["index"])
    files = {"index": {"sha256": file_hash(index_path), "detections": int(results["index"]["counts"].astype(np.int64).sum())}}
    for tag in [t for t in results if t != "index"]:
        path = detection_file(identity, images, tag)
        save_npz(path, encode_relative(results["index"], results[tag]))
        files[tag] = {"sha256": file_hash(path), "detections": int(results[tag]["counts"].astype(np.int64).sum()),
                      "storage": "relative to the fixed-rule file (core.encode_relative)"}
    record = {"configuration_sha256": identity, "format": fmt, "recipe_name": recipe_name, "label": label,
              "study_part": part, "images": images, "panel": f"screen{images}" if images == 1000 else "dev128",
              "detections": files, "tie_policy": ties, "passes": passes,
              "at_chance": bool(100 * metrics[0] < CHANCE_MAP),
              "map50_95": metrics[0], "map50": metrics[1], "distinct_scores": int(len(np.unique(results["index"]["score_e5"]))),
              "preparation_seconds": prepared, "evaluation_seconds": seconds, "finished_at": runner.now(),
              "protocol": PROTOCOL, "interpretation": "development_evidence", **(extra or {})}
    seal(target, record)
    print(json.dumps({"format": fmt, "label": label, "id": identity[:12], "map50_95": round(100 * metrics[0], 3),
                      "orders": list(files), "prep_s": round(prepared, 1), "eval_s": round(seconds, 1)}), flush=True)
    return record
