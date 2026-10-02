"""Matrix cells under a calibration subset, written into artifacts/experiment_b2_seeds/ (never into the matrix).

The cell is built by the unchanged B2 code (``runner.build`` for scalar formats, ``matrix.build_shared`` for
shared-exponent formats) and read out by ``matrix.evaluate`` (batch 8, tie-aware readout).  Inside this
process only, three module attributes are replaced:

* ``tools.experiment_b2.data.v1_calibration`` returns the range statistics of the current subset
  (``subsets.Observations.calibration``; the full set reproduces the v1 arrays bit for bit);
* ``tools.experiment_b2.data.cached_inputs`` returns, for the 256-image calibration request, the preprocessed
  bias-correction images of the current subset (taken from the existing 256-image cache entry where present,
  otherwise preprocessed from the JPEG payload by the v1 ``image_batch``; nothing is persisted), and passes
  every other request to the original with ``build=False``;
* ``tools.experiment_b2.runner.model_setup`` is memoised per model (it only loads the frozen model and the
  screen inputs).

The full-set subset therefore yields exactly the configuration identity of the existing matrix cell
(``reproduce`` job), and every other subset changes the identity only through its calibration record and its
bias-correction rows.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import gzip
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, seal, unseal
from tools.experiment_b2 import data, matrix, readout, runner
from tools.experiment_b2.blocks import SHARED
from tools.experiment_b2.common import MODELS, PROTOCOL, source_identity
from . import subsets

BASE = ROOT / "artifacts/experiment_b2_seeds"
VERSION = "b2-seeds-cells-1"
OWN = ("__init__.py", "subsets.py", "cells.py")
SEEDS = tuple(f"S{k}" for k in range(subsets.SEEDS))
DEFAULT_FORMATS = ("int8", "posit8_es1", "fp8_e4m3fn", "log8", "fp7_e3m3", "int6", "fp6_e2m3", "fp6_e3m2", "log6",
                   "posit6_es1", "mxfp8_e4m3", "bfp6")
MINIMAL_FORMATS = ("int6", "fp6_e2m3", "fp6_e3m2", "log6", "posit6_es1", "nf4")
LADDER_FORMATS = ("int8", "posit8_es1", "fp8_e4m3fn", "int6")
LADDER = ("L32_0", "L32_1", "L32_2", "L128_0", "L128_1", "L128_2", "H0", "H1")
BIAS_ONLY = tuple(f"B{k}" for k in range(subsets.SEEDS))
INTRINSIC = {"default": "cum5_act_maxabs", "minimal": "cum1_fused"}
REPRODUCTION = (("resnet18", "int8", "default"), ("mobilenet_v3_large", "posit8_es1", "default"),
                ("resnet18", "mxfp8_e4m3", "cum5_act_maxabs"))


def now():
    return datetime.now(timezone.utc).isoformat()


def own_sources():
    folder = ROOT / "tools/experiment_b2_seeds"
    return {f"tools/experiment_b2_seeds/{name}": file_hash(folder / name) for name in OWN}


def recipe_for(format_name, family_recipe):
    """The proposal's recipe for a format: shared-exponent formats use their intrinsic (max-abs block) arm."""
    return INTRINSIC[family_recipe] if format_name in SHARED else family_recipe


def arm_jobs(arm):
    """``(format, recipe, subset)`` triples of one protocol arm, in execution order."""
    if arm == "seed_default":
        return [(f, recipe_for(f, "default"), s) for s in SEEDS for f in DEFAULT_FORMATS]
    if arm == "seed_minimal":
        return [(f, "minimal", s) for s in SEEDS for f in MINIMAL_FORMATS]
    if arm == "ladder":
        return [(f, "default", s) for s in LADDER for f in LADDER_FORMATS]
    if arm == "bias_only":
        return [(f, "default", s) for s in BIAS_ONLY for f in LADDER_FORMATS]
    if arm in ADDENDUM_ARMS:
        prefix = "R" if arm == "addendum_range_only" else "B"
        return [(f, "default", f"{prefix}{k}") for k in range(3) for f in ADDENDUM_FORMATS]
    raise SystemExit(f"unknown arm {arm}")


ARMS = ("seed_default", "seed_minimal", "ladder", "bias_only")
# Protocol addendum 1 (b2-seeds-protocol-v1-addendum-1.json): ResNet18 and MobileNetV2 only.
ADDENDUM_ARMS = ("addendum_range_only", "addendum_bias_only")
ADDENDUM_FORMATS = ("fp6_e3m2", "log6", "posit6_es1")
ADDENDUM_MODELS = ("resnet18", "mobilenet_v2")


def cell_glob(model, format_name, recipe_name, subset):
    return sorted((BASE / "cells").glob(f"{model}--{format_name}--{recipe_name}--{subset}--*.json"))


def matrix_cell(model, format_name, recipe_name):
    """The existing full-set matrix cell (read-only) of the same model, format and recipe."""
    found = sorted((matrix.MATRIX / "cells").glob(f"{model}--{format_name}--{recipe_name}--1000--*.json"))
    if len(found) != 1:
        raise ValueError(f"expected one matrix cell for {model}/{format_name}/{recipe_name}, found {len(found)}")
    return found[0], unseal(found[0])


class Context:
    """Per-process state: verified observations, bias inputs per subset, and the in-process patches."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.observations = subsets.Observations(model)
        self.current = None
        self._bias = {}
        self._setup = None

    # -- inputs ---------------------------------------------------------------------------------------------
    def bias_inputs(self, name, transform, original_cached_inputs):
        """Preprocessed bias-correction images of a subset: cache entry where present, JPEG otherwise."""
        if name in self._bias:
            out, wanted, self.bias_info = self._bias[name]
            return out, wanted
        from tools.experiment_b.classifier import image_batch
        record, rows, payload = dataset(subsets.LIST)
        spec = subsets.definitions()[name]
        wanted = subsets.bias_rows(rows, spec["bias_batches"])
        cached, cached_rows = original_cached_inputs(subsets.LIST, subsets.BIAS_IMAGES, transform, build=False)
        where = {row["sha256"]: i for i, row in enumerate(cached_rows)}
        out = np.empty((len(wanted),) + tuple(cached.shape[1:]), dtype=np.float32)
        missing = [i for i, row in enumerate(wanted) if row["sha256"] not in where]
        for i, row in enumerate(wanted):
            if row["sha256"] in where:
                out[i] = cached[where[row["sha256"]]]
        for start in range(0, len(missing), 32):
            part = missing[start:start + 32]
            out[part] = image_batch([wanted[i] for i in part], payload, transform, "cpu").numpy()
        # Guard: the on-the-fly path reproduces the cache bit for bit (checked on two cached images).
        probe = image_batch(cached_rows[:2], payload, transform, "cpu").numpy()
        if not np.array_equal(probe, np.asarray(cached[:2])):
            raise ValueError("on-the-fly preprocessing differs from the B2 input cache")
        if not np.isfinite(out).all():
            raise ValueError("invalid bias-correction inputs")
        self.bias_info = {"images": len(wanted), "from_cache": len(wanted) - len(missing), "decoded": len(missing)}
        self._bias = {name: (out, wanted, self.bias_info)}  # keep one subset in memory at a time
        return out, wanted

    @contextmanager
    def patched(self):
        original_v1, original_cached, original_setup = data.v1_calibration, data.cached_inputs, runner.model_setup
        context = self

        def v1_calibration(model):
            if model != context.model or context.current is None:
                raise ValueError("calibration requested outside a subset cell")
            return context.observations.calibration(context.current)

        def cached_inputs(list_name, limit, transform, *, build=True):
            if list_name == subsets.LIST:
                if limit != subsets.BIAS_IMAGES or context.current is None:
                    raise ValueError("only the bias-correction request is served for the calibration list")
                return context.bias_inputs(context.current, transform, original_cached)
            return original_cached(list_name, limit, transform, build=False)

        def model_setup(model, device):
            if context._setup is None:
                context._setup = original_setup(model, device)
            return context._setup

        data.v1_calibration, data.cached_inputs, runner.model_setup = v1_calibration, cached_inputs, model_setup
        try:
            yield
        finally:
            data.v1_calibration, data.cached_inputs, runner.model_setup = original_v1, original_cached, original_setup

    # -- one cell -------------------------------------------------------------------------------------------
    def build(self, format_name, recipe_name, subset):
        self.current = subset
        self.bias_info = None
        args = SimpleNamespace(model=self.model, format=format_name, recipe=recipe_name, device=self.device, images=1000)
        if format_name in SHARED:
            return matrix.build_shared(args)
        return runner.build(args, seal_configuration=False)

    def run(self, format_name, recipe_name, subset):
        start = time.monotonic()
        existing = cell_glob(self.model, format_name, recipe_name, subset)
        if existing:
            return {**unseal(existing[0]), "status": "exists"}
        setup, recipe, engine, meta, configuration, identity = self.build(format_name, recipe_name, subset)
        arrays, logits_sha, seconds = matrix.evaluate(engine, setup["inputs"], setup["rows"], self.device)
        readout_path = save_readout(identity, arrays)
        reference_path, reference = matrix_cell(self.model, format_name, recipe_name)
        description = subsets.describe(subset, self.observations.rows)
        record = {
            "version": VERSION, "protocol": {"path": str(subsets.PROTOCOL_FILE.relative_to(ROOT)),
                                             "sha256": file_hash(subsets.PROTOCOL_FILE)},
            "model": self.model, "format": format_name, "recipe_name": recipe_name, "recipe": recipe.as_dict(),
            "subset": description, "bias_inputs": self.bias_info, "images": 1000,
            "inference_batch_size": PROTOCOL["inference_batch_size"], "configuration_sha256": identity,
            "configuration_file": save_configuration(identity, configuration),
            "baseline_sha256": setup["baseline_sha256"], "source_sha256": source_identity(),
            "matrix_own_sources": matrix.own_sources(), "own_sources": own_sources(),
            "readout": readout.summary(arrays), "readout_file": str(readout_path.relative_to(ROOT)),
            "readout_file_sha256": file_hash(readout_path), "logits_sha256": logits_sha,
            "rows_sha256": digest(setup["rows"]),
            "full_set_reference": {"cell": str(reference_path.relative_to(ROOT)),
                                   "configuration_sha256": reference["configuration_sha256"],
                                   "top1_expected_percent": reference["readout"]["top1_expected_percent"]},
            "recorded_at": now(),
            "cost": {"cell_wall_seconds": time.monotonic() - start,
                     "preparation_seconds": setup["preparation_seconds"], "evaluation_seconds": seconds,
                     "device": self.device, "note": "shared GPU slot; not a throughput measurement"}}
        path = BASE / "cells" / f"{self.model}--{format_name}--{recipe_name}--{subset}--{identity[:12]}.json"
        seal(path, record)
        return {**record, "status": "computed"}

    def reproduce(self, format_name, recipe_name):
        """Full-set cell through this wrapper; must equal the existing matrix cell (identity and readout)."""
        target = BASE / "reproduction" / f"{self.model}--{format_name}--{recipe_name}.json"
        if target.exists():
            return {**unseal(target), "status": "exists"}
        start = time.monotonic()
        setup, recipe, engine, meta, configuration, identity = self.build(format_name, recipe_name, "FULL")
        arrays, logits_sha, seconds = matrix.evaluate(engine, setup["inputs"], setup["rows"], self.device)
        reference_path, reference = matrix_cell(self.model, format_name, recipe_name)
        saved = readout.load(ROOT / reference["readout_file"])
        equal = {key: bool(np.array_equal(saved[key], arrays[key])) for key in readout.DTYPES}
        record = {"model": self.model, "format": format_name, "recipe_name": recipe_name,
                  "matrix_cell": str(reference_path.relative_to(ROOT)),
                  "configuration_sha256": identity, "matrix_configuration_sha256": reference["configuration_sha256"],
                  "identity_reproduced": identity == reference["configuration_sha256"],
                  "readout_arrays_equal": equal, "all_readout_arrays_equal": all(equal.values()),
                  "logits_sha256": logits_sha, "matrix_logits_sha256": reference["logits_sha256"],
                  "logits_reproduced": logits_sha == reference["logits_sha256"],
                  "top1_expected_percent": readout.summary(arrays)["top1_expected_percent"],
                  "bias_inputs": self.bias_info, "own_sources": own_sources(), "source_sha256": source_identity(),
                  "wall_seconds": time.monotonic() - start, "recorded_at": now()}
        record["passed"] = bool(record["identity_reproduced"] and record["all_readout_arrays_equal"])
        seal(target, record)
        return {**record, "status": "computed"}


def save_readout(identity, arrays):
    path = BASE / "readout" / f"{identity}.npz"
    if path.exists():
        saved = readout.load(path)
        if any(not np.array_equal(saved[key], arrays[key]) for key in readout.DTYPES):
            raise ValueError(f"existing readout record differs from the recomputed one: {path}")
    else:
        readout.save(path, arrays)
    return path


def compact_configuration(configuration):
    """The identity pre-image without the bulky per-channel scale lists (kept by digest)."""
    scales = configuration["scales"]
    kept = {"sha256": digest(scales)}
    for key in ("activation_scales", "bias_correction", "activation_rule", "weight_rule", "block"):
        if key in scales:
            kept[key] = scales[key]
    if "weight_scales" in scales:
        kept["weight_scales_sha256"] = digest(scales["weight_scales"])
    if "weights" in scales:
        kept["weights_sha256"] = digest(scales["weights"])
    return {**{k: v for k, v in configuration.items() if k != "scales"}, "scales": kept}


def save_configuration(identity, configuration):
    path = BASE / "configurations" / f"{identity}.json.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.partial")
        document = {"configuration_sha256": identity, "compact": compact_configuration(configuration),
                    "note": "compact pre-image: the per-channel weight scales and searches are kept by digest only"}
        with gzip.open(temporary, "wt") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        temporary.replace(path)
    return str(path.relative_to(ROOT))


def brief(record):
    r = record["readout"] if "readout" in record else {}
    return {"model": record["model"], "format": record["format"], "recipe": record["recipe_name"],
            "subset": record.get("subset", {}).get("name"), "status": record["status"],
            "top1_expected": round(r.get("top1_expected_percent", float("nan")), 2) if r else None,
            "seconds": round(record.get("cost", {}).get("cell_wall_seconds", 0.0), 1)}


def pending(model, arm, kind):
    jobs = []
    for format_name, recipe_name, subset in arm_jobs(arm):
        block = format_name in SHARED
        if (kind == "block" and not block) or (kind == "scalar" and block):
            continue
        if not cell_glob(model, format_name, recipe_name, subset):
            jobs.append((format_name, recipe_name, subset))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--arm", choices=ARMS + ADDENDUM_ARMS, required=True)
    p.add_argument("--kind", choices=("scalar", "block", "all"), default="all")
    p.add_argument("--max-seconds", type=float, default=780.0)
    p.add_argument("--limit", type=int, default=0, help="at most this many cells (0: no limit)")
    p = sub.add_parser("reproduce")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--recipe", required=True)
    p = sub.add_parser("pending")
    p.add_argument("--arm", choices=ARMS + ADDENDUM_ARMS, required=True)
    for name in ("cells", "reproduce"):
        sub.choices[name].add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.command == "pending":
        for model in MODELS:
            for kind in ("scalar", "block"):
                print(model, kind, len(pending(model, args.arm, kind)))
        return 0
    if not subsets.PROTOCOL_FILE.exists():
        raise SystemExit("the seeds protocol must be on disk before anything is measured")
    from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
    from tools.experiment_b2.recipe import NAMED  # noqa: F401
    context = Context(args.model, args.device)
    tick = time.monotonic()
    with context.patched():
        if args.command == "reproduce":
            record = context.reproduce(args.format, args.recipe)
            print(json.dumps({k: record[k] for k in ("model", "format", "recipe_name", "identity_reproduced",
                                                      "all_readout_arrays_equal", "logits_reproduced", "passed",
                                                      "status")}), flush=True)
            return 0 if record["passed"] else 1
        jobs = pending(args.model, args.arm, args.kind)
        done, longest = 0, 0.0
        for format_name, recipe_name, subset in jobs:
            elapsed = time.monotonic() - tick
            if done and elapsed + 1.5 * longest > args.max_seconds:
                break
            if args.limit and done >= args.limit:
                break
            cell_tick = time.monotonic()
            record = context.run(format_name, recipe_name, subset)
            longest = max(longest, time.monotonic() - cell_tick)
            done += 1
            print(json.dumps(brief(record)), flush=True)
        left = len(pending(args.model, args.arm, args.kind))
        print(json.dumps({"model": args.model, "arm": args.arm, "kind": args.kind, "cells_this_job": done,
                          "pending": left, "seconds": round(time.monotonic() - tick, 1)}), flush=True)
        return 3 if left else 0
