"""B2 format matrix: every accepted format under the frozen recipes, with the tie-aware readout.

One cell = (model, format, recipe) on the full 1k screen at batch size 8.  A cell writes
one compressed readout record (``matrix/readout/<configuration>.npz``, see ``readout.py``)
and one sealed cell record (``matrix/cells/``).  Nothing that stage 1 sealed is rewritten:
a cell whose configuration already exists in ``runs/{exit,sentinel,lowbit}`` must
reproduce the sealed configuration id and the sealed top-5 lists on 1000 of 1000 images
before its readout record is written.  Scalar cells are built by ``runner.build`` itself,
so their identities are exactly those stage 1 would have produced.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import functools
import gzip
import hashlib
import json
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, formats, seal, unseal
from tools.experiment_b.validation import verify_prediction
from . import data, readout, runner
from .blocks import SHARED
from .common import BASE, BIAS_CORRECTION_IMAGES, MODELS, PROTOCOL, numeric_sources, source_identity
from .recipe import named

VERSION = "b2-matrix-1"
MATRIX = BASE / "matrix"
PROTOCOL_FILE = ROOT / "public/experiments/configs/breadth-study/b2-matrix-protocol-v1.json"
IMAGES = 1000
AUDIT_IMAGES = 128
STAGE1 = ("exit", "sentinel", "lowbit")
MATRIX_RECIPES = ("default", "minimal")
# Shared-exponent formats only: the same two recipes with activation blocks scaled by the
# intrinsic max-abs rule instead of the run-time exponent search.
INTRINSIC_ARM = {"default": "cum5_act_maxabs", "minimal": "cum1_fused"}
OWN = ("matrix.py", "readout.py", "blocks.py")
START = time.monotonic()



def now():
    return datetime.now(timezone.utc).isoformat()


def own_sources():
    return {f"tools/experiment_b2/{name}": file_hash(ROOT / "tools/experiment_b2" / name) for name in OWN}


def protocol_identity():
    return {"path": str(PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(PROTOCOL_FILE)}


def format_entry(name):
    return {row["name"]: row for row in formats()}[name]


def stage1_record(model, format_name, recipe_name):
    """The sealed stage-1 run of this cell on the 1k screen, if there is one for the current source."""
    source, found = source_identity(), []
    for stage in STAGE1:
        for path in sorted((BASE / "runs" / stage).glob(f"{model}--{format_name}--{recipe_name}--{IMAGES}--*.json")):
            record = unseal(path)
            if record.get("source_sha256") == source:
                found.append({"path": str(path.relative_to(ROOT)), "stage": stage,
                              "configuration_sha256": record["configuration_sha256"],
                              "sealed_top1_percent": record["metrics"]["top1_percent"]})
    if len({row["configuration_sha256"] for row in found}) > 1:
        raise ValueError(f"ambiguous stage-1 records for {model}/{format_name}/{recipe_name}")
    return found[0] if found else None


def evaluate(engine, inputs, rows, device, limit=IMAGES):
    """Batch-8 evaluation; returns the readout arrays, a digest of all logits and the seconds spent."""
    import torch
    batch_size = PROTOCOL["inference_batch_size"]
    labels = torch.tensor([int(row["label"]) for row in rows[:limit]], device=device)
    parts, sha, tick = [], hashlib.sha256(), time.monotonic()
    for start in range(0, limit, batch_size):
        tensor = torch.from_numpy(np.array(inputs[start:min(limit, start + batch_size)], dtype=np.float32)).to(device)
        with torch.inference_mode():
            output = engine.run(tensor) if hasattr(engine, "run") else engine(tensor)
            if tuple(output.shape) != (len(tensor), 1000) or output.dtype != torch.float32:
                raise ValueError("invalid classifier output")
            parts.append(readout.batch_readout(output, labels[start:start + len(tensor)]))
            sha.update(output.cpu().numpy().tobytes())
    return readout.concatenate(parts), sha.hexdigest(), time.monotonic() - tick


def occupancy_summary(rows):
    def values(key):
        return [row[key] for row in rows.values() if row.get(key) is not None]

    def median(items):
        return float(np.median(items)) if items else None

    sqnr = values("sqnr_db")
    result = {"tensors": len(rows), "median_sqnr_db": median(sqnr), "min_sqnr_db": float(min(sqnr)) if sqnr else None,
              "median_entropy_bits": median(values("entropy_bits")),
              "median_levels_used_fraction": median([row["levels_used"] / row["levels_total"] for row in rows.values()]),
              "median_fraction_at_extremes": median(values("fraction_at_extremes")),
              "max_fraction_at_extremes": float(max(values("fraction_at_extremes"))) if rows else None,
              "median_fraction_zero": median(values("fraction_zero"))}
    if any("fraction_blocks_clipped" in row for row in rows.values()):
        result["median_fraction_blocks_clipped"] = median(values("fraction_blocks_clipped"))
        result["median_mean_clip_octaves"] = median(values("mean_clip_octaves"))
    return result


def scalar_occupancy(engine, meta, inputs, device):
    import torch
    from .engine import B2Interpreter, audit_summary
    probe = B2Interpreter(engine.module, engine.quantizers, engine.plan, meta["activation_scales"], audit=True)
    with torch.inference_mode():
        for start in range(0, AUDIT_IMAGES, 8):
            probe.run(torch.from_numpy(np.array(inputs[start:start + 8], dtype=np.float32)).to(device))
    rows = audit_summary(probe)
    return {"images": AUDIT_IMAGES, "summary": occupancy_summary(rows), "nodes": rows}


def shared_occupancy(engine, inputs, device):
    import torch
    from .blocks import audit_rows
    if engine.quantizer.rule == "percentile_99_9":
        return None
    engine.quantizer.audit = {}
    try:
        with torch.inference_mode():
            for start in range(0, AUDIT_IMAGES, 8):
                engine.run(torch.from_numpy(np.array(inputs[start:start + 8], dtype=np.float32)).to(device))
        rows = audit_rows(engine.quantizer)
    finally:
        engine.quantizer.audit = None
    stored = {k[7:]: v for k, v in rows.items() if k.startswith("stored:")}
    patches = {k[6:]: v for k, v in rows.items() if k.startswith("patch:")}
    return {"images": AUDIT_IMAGES, "summary": occupancy_summary(stored), "patch_summary": occupancy_summary(patches),
            "nodes": stored, "patches": patches}


def build_shared(args):
    """Configuration of one shared-exponent cell (the counterpart of ``runner.build``)."""
    import torch
    from .blocks import block_sources, prepare_blocks
    recipe = named(args.recipe)
    setup = runner.model_setup(args.model, args.device)
    bias_inputs = bias_identity = None
    if recipe.bias_correction != "none":
        bias_inputs, bias_rows = data.cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"])
        bias_identity = {"images": len(bias_rows), "rows_sha256": digest(bias_rows)}
    tick = time.monotonic()
    with torch.inference_mode():
        engine, scales = prepare_blocks(setup["graph"], args.format, recipe, args.device, bias_inputs=bias_inputs)
    setup["preparation_seconds"] = time.monotonic() - tick
    configuration = {"model_context": setup["context"], "format": args.format,
                     "format_sha256": format_entry(args.format)["sha256"], "recipe": recipe.as_dict(),
                     "protocol": PROTOCOL, "runtime": setup["env"], "source_sha256": source_identity(),
                     "numeric_sources": numeric_sources(), "block_sources": block_sources(),
                     "calibration": "not_used_for_intrinsic_block_scales", "bias_correction_inputs": bias_identity,
                     "scales": scales, "baseline_sha256": setup["baseline_sha256"]}
    return setup, recipe, engine, scales, configuration, digest(configuration)


def _save_readout(identity, arrays):
    path = MATRIX / "readout" / f"{identity}.npz"
    if path.exists():
        saved = readout.load(path)
        if any(not np.array_equal(saved[key], arrays[key]) for key in readout.DTYPES):
            raise ValueError(f"existing readout record differs from the recomputed one: {path}")
    else:
        readout.save(path, arrays)
    return path


def _save_configuration(identity, configuration):
    """Keep the pre-image of a new identity (gzip; stage-1 identities are already sealed uncompressed)."""
    if (BASE / "configurations" / f"{identity}.json").exists():
        return str((BASE / "configurations" / f"{identity}.json").relative_to(ROOT))
    path = MATRIX / "configurations" / f"{identity}.json.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        with gzip.open(temporary, "wt") as stream:
            json.dump({"payload": configuration, "sha256": digest(configuration)}, stream, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
        temporary.replace(path)
    return str(path.relative_to(ROOT))


def load_configuration(identity):
    plain = BASE / "configurations" / f"{identity}.json"
    if plain.exists():
        return unseal(plain)
    with gzip.open(MATRIX / "configurations" / f"{identity}.json.gz", "rt") as stream:
        document = json.load(stream)
    if digest(document["payload"]) != document["sha256"] or document["sha256"] != identity:
        raise ValueError("configuration integrity failure")
    return document["payload"]


def cell_path(model, format_name, recipe_name, identity):
    return MATRIX / "cells" / f"{model}--{format_name}--{recipe_name}--{IMAGES}--{identity[:12]}.json"


def run_cell(model, format_name, recipe_name, device):
    import torch
    start = time.monotonic()
    args = SimpleNamespace(model=model, format=format_name, recipe=recipe_name, device=device, images=IMAGES)
    shared = format_name in SHARED
    build = build_shared(args) if shared else runner.build(args, seal_configuration=False)
    setup, recipe, engine, meta, configuration, identity = build
    path = cell_path(model, format_name, recipe_name, identity)
    if path.exists():
        return {**unseal(path), "status": "exists"}
    stage1 = stage1_record(model, format_name, recipe_name)
    if stage1 is not None and stage1["configuration_sha256"] != identity:
        raise ValueError(f"stage-1 identity not reproduced for {model}/{format_name}/{recipe_name}")
    block_metadata = None
    if shared:
        from .blocks import metadata_bits
        sample = torch.from_numpy(np.array(setup["inputs"][:1], dtype=np.float32)).to(device)
        block_metadata = metadata_bits(engine, sample)
    arrays, logits_sha, seconds = evaluate(engine, setup["inputs"], setup["rows"], device)
    if stage1 is not None:
        sealed = runner.load_predictions(identity, setup["rows"])
        equal = sum(p["top5"] == [int(x) for x in row] for p, row in zip(sealed, arrays["top5_topk"]))
        stage1 = {**stage1, "identity_reproduced": True, "sealed_top5_lists_reproduced": equal}
        if equal != IMAGES:
            raise ValueError(f"sealed stage-1 predictions not reproduced: {equal} of {IMAGES}")
    tick = time.monotonic()
    occupancy = (shared_occupancy(engine, setup["inputs"], device) if shared
                 else scalar_occupancy(engine, meta, setup["inputs"], device))
    audit_seconds = time.monotonic() - tick
    readout_file = _save_readout(identity, arrays)
    entry = format_entry(format_name)
    record = {
        "version": VERSION, "protocol": protocol_identity(), "model": model, "format": format_name,
        "family": entry["family"], "bits": int(entry["bits"]), "recipe_name": recipe_name, "recipe": recipe.as_dict(),
        "images": IMAGES, "inference_batch_size": PROTOCOL["inference_batch_size"],
        "configuration_sha256": identity, "configuration_file": _save_configuration(identity, configuration),
        "baseline_sha256": setup["baseline_sha256"], "source_sha256": source_identity(), "own_sources": own_sources(),
        "readout": readout.summary(arrays), "readout_file": str(readout_file.relative_to(ROOT)),
        "readout_file_sha256": file_hash(readout_file), "logits_sha256": logits_sha,
        "rows_sha256": digest(setup["rows"]), "stage1": stage1, "occupancy": occupancy,
        "block_metadata": block_metadata, "recorded_at": now(),
        "cost": {"cell_wall_seconds": time.monotonic() - start, "preparation_seconds": setup["preparation_seconds"],
                 "evaluation_seconds": seconds, "audit_seconds": audit_seconds, "device": device,
                 "note": "shared GPU slot (up to 4 jobs at once); seconds are not a throughput measurement"}}
    seal(path, record)
    return {**record, "status": "computed"}


def run_baseline(model, device):
    """FP32 readout of one model; must reproduce the sealed stage-1 baseline top-5 lists."""
    start = time.monotonic()
    setup = runner.model_setup(model, device)
    identity = setup["baseline_sha256"]
    path = cell_path(model, "fp32", "baseline", identity)
    if path.exists():
        return {**unseal(path), "status": "exists"}
    arrays, logits_sha, seconds = evaluate(setup["graph"], setup["inputs"], setup["rows"], device)
    sealed = runner.load_predictions(identity, setup["rows"])
    equal = sum(p["top5"] == [int(x) for x in row] for p, row in zip(sealed, arrays["top5_topk"]))
    if equal != IMAGES:
        raise ValueError(f"sealed FP32 baseline not reproduced: {equal} of {IMAGES}")
    readout_file = _save_readout(identity, arrays)
    record = {"version": VERSION, "protocol": protocol_identity(), "model": model, "format": "fp32", "family": "fp32",
              "bits": 32, "recipe_name": "baseline", "recipe": None, "images": IMAGES,
              "inference_batch_size": PROTOCOL["inference_batch_size"], "configuration_sha256": identity,
              "baseline_sha256": identity, "source_sha256": source_identity(), "own_sources": own_sources(),
              "readout": readout.summary(arrays), "readout_file": str(readout_file.relative_to(ROOT)),
              "readout_file_sha256": file_hash(readout_file), "logits_sha256": logits_sha,
              "rows_sha256": digest(setup["rows"]),
              "stage1": {"configuration_sha256": identity, "identity_reproduced": True,
                         "sealed_top5_lists_reproduced": equal},
              "recorded_at": now(), "cost": {"cell_wall_seconds": time.monotonic() - start,
                                             "evaluation_seconds": seconds, "device": device}}
    seal(path, record)
    return {**record, "status": "computed"}


def regress_shared(model, format_name, recipe_name, device):
    """Proof on the real model that the block engine with the B2 switches off is the old extension."""
    import csv
    import torch
    from tools.experiment_b_ext.shared import prepare_shared
    from .blocks import prepare_blocks
    recipe = named(recipe_name)
    old_recipe = recipe.v1_equivalent()
    if old_recipe is None:
        raise SystemExit("the shared regression needs a v1-equivalent recipe")
    target = MATRIX / "regression" / f"{model}--{format_name}--{recipe_name}--{IMAGES}--{REGRESSION_TAG}.json"
    if target.exists():
        return {**unseal(target), "status": "exists"}
    setup = runner.model_setup(model, device)
    with torch.inference_mode():
        old, _ = prepare_shared(setup["graph"], format_name, old_recipe, device)
        new, _ = prepare_blocks(setup["graph"], format_name, recipe, device)
    with (ROOT / "results/summaries/b-stage-paired-1k-v2/configurations.csv").open() as stream:
        matches = [row for row in csv.DictReader(stream) if row["model"] == model and row["format"] == format_name
                   and row["recipe"] == old_recipe and row["metric"] == "top1" and row["images"] == "1000"]
    if len(matches) != 1:
        raise ValueError("extension 1k configuration not found in the paired summary")
    old_identity = matches[0]["configuration_sha256"]
    folder = ROOT / "artifacts/experiment_b_ext/predictions" / old_identity
    weights_equal = all(torch.equal(old.weights[k], new.weights[k]) for k in old.weights) and set(old.weights) == set(new.weights)
    identical = top5_equal = audited_identical = 0
    for start in range(0, IMAGES, 8):
        batch = setup["rows"][start:start + 8]
        tensor = torch.from_numpy(np.array(setup["inputs"][start:start + len(batch)], dtype=np.float32)).to(device)
        with torch.inference_mode():
            a, b = old.run(tensor), new.run(tensor)
            if not torch.equal(a, b):
                raise ValueError(f"block engine differs from the extension in batch {start}")
            if old_recipe == "maxabs" and start < 64:
                # The vectorised audit path must give the same numbers as the extension's own code.
                new.quantizer.audit = {}
                try:
                    audited_identical += len(batch) if torch.equal(new.run(tensor), a) else 0
                finally:
                    new.quantizer.audit = None
        identical += len(batch)
        for row, classes in zip(batch, b.topk(5, dim=1).indices.cpu().tolist()):
            top5_equal += verify_prediction(folder / (row["sha256"] + ".json"), old_identity, row)["top5"] == classes
    record = {"model": model, "format": format_name, "recipe_name": recipe_name, "extension_recipe": old_recipe,
              "images": IMAGES, "logit_bit_identical_images": identical, "weights_bit_identical": weights_equal,
              "sealed_extension_top5_reproduced_images": top5_equal, "extension_configuration_sha256": old_identity,
              "extension_sealed_top1_percent": float(matches[0]["candidate_percent"]),
              "audit_path_bit_identical_images": audited_identical if old_recipe == "maxabs" else None,
              "runtime": setup["env"], "source_sha256": source_identity(), "own_sources": own_sources(),
              "wall_seconds": time.monotonic() - START, "recorded_at": now()}
    ok = identical == IMAGES and top5_equal == IMAGES and weights_equal and (old_recipe != "maxabs" or audited_identical == 64)
    record["passed"] = bool(ok)
    seal(target, record)
    return {**record, "status": "computed"}


# Tag r2: records made after blocks.py was changed to bound the memory of the vectorised path (the
# untagged records were made with the first version; one of the six ran out of GPU memory).
REGRESSION_TAG = "r2"
REGRESSION_SAMPLE = (("resnet18", "bfp6", "v1_maxabs"), ("resnet18", "mxfp4_e2m1", "v1_percentile_99_9"),
                     ("mobilenet_v2", "mxfp6_e3m2", "v1_maxabs"), ("mobilenet_v2", "bfp6", "v1_percentile_99_9"),
                     ("mobilenet_v3_large", "mxfp8_e4m3", "v1_maxabs"),
                     ("mobilenet_v3_large", "mxfp6_e3m2", "v1_percentile_99_9"))


def campaign_jobs(stage):
    """``(label, done, command arguments)`` for every job of a stage."""
    names = [row["name"] for row in formats()]

    def cells_done(model, name, recipes):
        return all(any((MATRIX / "cells").glob(f"{model}--{name}--{recipe}--{IMAGES}--*.json")) for recipe in recipes)

    if stage == "baseline":
        for model in MODELS:
            yield f"{model}/fp32", cells_done(model, "fp32", ("baseline",)), ["baseline", "--model", model]
    elif stage == "regress":
        for model, name, recipe in REGRESSION_SAMPLE:
            done = (MATRIX / "regression" / f"{model}--{name}--{recipe}--{IMAGES}--{REGRESSION_TAG}.json").exists()
            yield f"{model}/{name}/{recipe}", done, ["regress-shared", "--model", model, "--format", name, "--recipe", recipe]
    elif stage in ("matrix", "intrinsic"):
        for model in MODELS:
            for name in names:
                if stage == "intrinsic" and name not in SHARED:
                    continue
                recipes = MATRIX_RECIPES if stage == "matrix" else tuple(INTRINSIC_ARM.values())
                for recipe in recipes if name in SHARED else (",".join(recipes),):
                    yield (f"{model}/{name}/{recipe}", cells_done(model, name, recipe.split(",")),
                           ["cells", "--model", model, "--format", name, "--recipes", recipe])
    else:
        raise SystemExit(f"unknown stage {stage}")


def campaign(args):
    logs = MATRIX / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = [(label, command) for label, done, command in campaign_jobs(args.stage) if not done]
    if args.dry_run:
        print("\n".join(" ".join(command) for _, command in pending))
        return 0
    ledger = logs / f"campaign-{args.stage}.jsonl"

    def launch(job):
        label, command = job
        tick = time.time()
        full = [str(ROOT / "artifacts/agent_orchestration/gpu_run.sh"), str(ROOT / ".venv-b/bin/python"), "-m",
                "tools.run.experiment_b2_matrix", *command]
        with (logs / f"campaign-{args.stage}.err").open("a") as err:
            result = subprocess.run(full, cwd=ROOT, stdout=subprocess.PIPE, stderr=err, text=True)
        lines = result.stdout.strip().splitlines()
        entry = {"label": label, "command": command, "returncode": result.returncode, "launched_unix": tick,
                 "wall_seconds_including_slot_wait": time.time() - tick, "output": lines[-3:]}
        with ledger.open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        print(json.dumps(entry, sort_keys=True), flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        codes = list(pool.map(launch, pending))
    failures = sum(code != 0 for code in codes)
    print(f"CAMPAIGN {args.stage} COMPLETE jobs={len(pending)} failures={failures}", flush=True)
    return 1 if failures else 0


def brief(record):
    r = record["readout"]
    return {"model": record["model"], "format": record["format"], "recipe": record["recipe_name"],
            "status": record["status"], "top1_expected": round(r["top1_expected_percent"], 2),
            "top1_lowest_index": r["top1_lowest_index_percent"], "top1_topk": r["top1_topk_percent"],
            "tied": r["images_with_tied_top1"], "configuration": record["configuration_sha256"][:12],
            "stage1": bool(record.get("stage1")), "seconds": round(record["cost"]["cell_wall_seconds"], 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--recipes", default=",".join(MATRIX_RECIPES))
    p = sub.add_parser("baseline")
    p.add_argument("--model", choices=MODELS, required=True)
    p = sub.add_parser("regress-shared")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--format", choices=SHARED, required=True)
    p.add_argument("--recipe", required=True)
    p = sub.add_parser("campaign")
    p.add_argument("stage", choices=("baseline", "regress", "matrix", "intrinsic"))
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    for name in ("cells", "baseline", "regress-shared"):
        sub.choices[name].add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    from . import frozen  # noqa: F401  (registers the frozen recipes)
    # The matrix stage must not enlarge the stage-1 input cache: a missing cache is an error.
    data.cached_inputs = functools.partial(data.cached_inputs, build=False)
    if args.command == "campaign":
        return campaign(args)
    if not PROTOCOL_FILE.exists():
        raise SystemExit("the matrix protocol must be on disk before anything is measured")
    if args.command == "cells":
        for recipe_name in args.recipes.split(","):
            print(json.dumps(brief(run_cell(args.model, args.format, recipe_name, args.device))), flush=True)
        return 0
    if args.command == "baseline":
        print(json.dumps(brief(run_baseline(args.model, args.device))), flush=True)
        return 0
    record = regress_shared(args.model, args.format, args.recipe, args.device)
    print(json.dumps({k: v for k, v in record.items() if k not in ("runtime", "own_sources")}), flush=True)
    return 0 if record["passed"] else 1
