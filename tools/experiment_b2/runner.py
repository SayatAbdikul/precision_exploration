"""B2 runner: one configuration per invocation, sealed per-image predictions as in v1."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, formats, frozen_inputs, seal, unseal
from tools.experiment_b.runner import metrics, runtime
from tools.experiment_b.validation import verify_prediction
from .common import BASE, BIAS_CORRECTION_IMAGES, MODELS, PANELS, PROTOCOL, numeric_sources, source_identity
from .recipe import NAMED, hardware_semantics, named

START = time.monotonic()


def now():
    return datetime.now(timezone.utc).isoformat()


def own_file(name):
    return file_hash(ROOT / "tools/experiment_b2" / name)


def evaluate(engine, inputs, rows, identity, device, limit):
    """Batch-8 evaluation with sealed per-image top-5 records (v1 schema and resume rule)."""
    import torch
    rows = rows[:limit]
    directory = BASE / "predictions" / identity
    directory.mkdir(parents=True, exist_ok=True)
    predictions, elapsed, computed = [], 0.0, 0
    batch_size = PROTOCOL["inference_batch_size"]
    for start in range(0, limit, batch_size):
        batch = rows[start:start + batch_size]
        existing = {row["sha256"]: directory / (row["sha256"] + ".json") for row in batch}
        valid = {row["sha256"]: verify_prediction(existing[row["sha256"]], identity, row)
                 for row in batch if existing[row["sha256"]].exists()}
        if len(valid) != len(batch):
            tick = time.monotonic()
            tensor = torch.from_numpy(np.array(inputs[start:start + len(batch)], dtype=np.float32)).to(device)
            with torch.inference_mode():
                output = engine.run(tensor) if hasattr(engine, "run") else engine(tensor)
            if not torch.isfinite(output).all() or tuple(output.shape) != (len(batch), 1000):
                raise ValueError("invalid classifier output")
            top5 = output.topk(5, dim=1).indices.cpu().tolist()
            duration = time.monotonic() - tick
            elapsed += duration
            computed += len(batch)
            for row, classes in zip(batch, top5):
                saved = {"configuration_sha256": identity, "sample": row, "top5": classes,
                         "batch_start": start, "batch_images": len(batch), "batch_seconds": duration}
                if row["sha256"] in valid:
                    if valid[row["sha256"]]["top5"] != classes:
                        raise ValueError("resumed batch changed saved predictions")
                else:
                    seal(existing[row["sha256"]], saved)
                    valid[row["sha256"]] = saved
        predictions.extend(valid[row["sha256"]] for row in batch)
    return predictions, {"computed_images_this_invocation": computed, "seconds_this_invocation": elapsed}


def load_predictions(identity, rows):
    return [verify_prediction(BASE / "predictions" / identity / (row["sha256"] + ".json"), identity, row) for row in rows]


def model_setup(model, device):
    """Frozen model, cached inputs and the FP32 baseline identity for one model."""
    import torch
    from tools.experiment_b.classifier import configure, load_model
    from tools.experiment_b.validation import validate_fold
    from .data import cached_inputs
    configure(device)
    env = runtime(device)
    context = frozen_inputs(model)
    graph, transform, original = load_model(model, device)
    inputs, rows = cached_inputs("imagenet_screen_1k", 1000, transform)
    probe = torch.from_numpy(np.array(inputs[:8], dtype=np.float32)).to(device)
    check = validate_fold(graph, original, probe)
    original = original.cpu()
    baseline = {"context": context, "source_sha256": source_identity(), "runtime": env,
                "protocol": PROTOCOL, "graph": str(graph.graph), "mode": "folded_fp32_baseline"}
    return {"env": env, "context": context, "graph": graph, "transform": transform, "original": original,
            "inputs": inputs, "rows": rows, "fold_validation": check, "baseline_sha256": digest(baseline),
            "baseline_configuration": baseline}


def record_run(stage, model, format_name, recipe_name, images, identity, body):
    name = f"{model}--{format_name}--{recipe_name}--{images}--{identity[:12]}.json"
    path = BASE / "runs" / stage / name
    seal(path, {"stage": stage, "model": model, "format": format_name, "recipe_name": recipe_name,
                "images": images, "configuration_sha256": identity, "recorded_at": now(), **body})
    return path


def build(args, *, audit=False, seal_configuration=True):
    """Prepare one configuration: frozen model, v1 observations, B2 engine and its sealed identity.

    ``seal_configuration=False`` computes the same identity without writing anything (used by
    the verification jobs, which must not touch sealed evidence).
    """
    import torch
    from .data import cached_inputs, v1_calibration
    from .engine import prepare_b2
    from .guards import check_graph
    entries = {row["name"]: row for row in formats()}
    if args.format not in entries or entries[args.format]["family"] in {"bfp", "mx_float"}:
        raise SystemExit("B2 covers scalar accepted formats only")
    recipe = named(args.recipe)
    setup = model_setup(args.model, args.device)
    graph, env = setup["graph"], setup["env"]
    check_graph(graph)  # the structural non-negativity proof must hold before any unsigned code is used
    arrays, maxima, calibration = v1_calibration(args.model)
    if calibration["graph"] != str(graph.graph):
        raise ValueError("v1 calibration graph differs from the loaded graph")
    bias_inputs, bias_identity = None, None
    if recipe.bias_correction != "none":
        bias_inputs, bias_rows = cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"])
        bias_identity = {"images": len(bias_rows), "rows_sha256": digest(bias_rows)}
    tick = time.monotonic()
    engine_recipe, equalization = recipe, None
    if recipe.equalization == "cle":
        # Equalize first (function-preserving), observe the equalized graph again, then run the
        # ordinary engine on it: bias correction then compares against the equalized FP32 graph.
        import dataclasses
        from .equalization import equalize_graph, observe
        graph, equalization = equalize_graph(graph)
        probe = torch.from_numpy(np.array(setup["inputs"][:8], dtype=np.float32)).to(args.device)
        with torch.inference_mode():
            drift = float((graph(probe) - setup["graph"](probe)).abs().max())
        if drift > 2e-3:
            raise ValueError(f"equalized graph is not function-preserving: {drift}")
        calibration_inputs, calibration_rows = cached_inputs("imagenet_calibration_2k", 2000, setup["transform"])
        key = {"model_context": setup["context"], "equalization": equalization, "runtime": env,
               "rows_sha256": digest(calibration_rows), "source_sha256": source_identity(),
               "equalization_source_sha256": own_file("equalization.py")}
        arrays, maxima, observed = observe(graph, calibration_inputs, args.device, key)
        equalization = {**equalization, "max_abs_logit_drift_8_images": drift, "observations": observed}
        calibration = {**calibration, "equalized_observations": observed}
        engine_recipe = dataclasses.replace(recipe, equalization="none")
        setup["engine_graph"] = graph
    with torch.inference_mode():
        engine, scales = prepare_b2(graph, args.format, engine_recipe, arrays, maxima, args.device,
                                    bias_inputs=bias_inputs, audit=audit)
    if equalization is not None:
        scales = {**scales, "equalization": equalization}
    setup["preparation_seconds"] = time.monotonic() - tick
    configuration = {"model_context": setup["context"], "format": args.format,
                     "format_sha256": entries[args.format]["sha256"], "recipe": recipe.as_dict(),
                     "protocol": PROTOCOL, "runtime": env, "source_sha256": source_identity(),
                     "numeric_sources": numeric_sources(), "calibration": calibration,
                     "bias_correction_inputs": bias_identity, "scales": scales,
                     "baseline_sha256": setup["baseline_sha256"]}
    if recipe.equalization != "none":
        configuration["equalization_source_sha256"] = own_file("equalization.py")
    identity = digest(configuration)
    if seal_configuration:
        seal(BASE / "configurations" / f"{identity}.json", configuration)
    return setup, recipe, engine, scales, configuration, identity


def run_configuration(args):
    import torch
    from .engine import audit_summary
    setup, recipe, engine, scales, configuration, identity = build(args, audit=args.audit)
    graph = setup["graph"]
    preparation = setup["preparation_seconds"]
    if args.audit:
        # The audit engine is a measurement pass of its own; it never writes predictions.
        with torch.inference_mode():
            for start in range(0, args.images, PROTOCOL["inference_batch_size"]):
                engine.run(torch.from_numpy(np.array(setup["inputs"][start:start + 8], dtype=np.float32)).to(args.device))
        occupancy = audit_summary(engine)
        path = BASE / "occupancy" / f"{args.model}--{args.format}--{args.recipe}--{args.images}--{identity[:12]}.json"
        seal(path, {"configuration_sha256": identity, "model": args.model, "format": args.format,
                    "recipe_name": args.recipe, "images": args.images, "nodes": occupancy,
                    "wall_seconds": time.monotonic() - START})
        print(json.dumps({"occupancy": str(path.relative_to(ROOT))}))
        return 0
    seal(BASE / "configurations" / f"{setup['baseline_sha256']}.json", setup["baseline_configuration"])
    baseline, baseline_timing = evaluate(graph, setup["inputs"], setup["rows"], setup["baseline_sha256"],
                                         args.device, args.images)
    predictions, timing = evaluate(engine, setup["inputs"], setup["rows"], identity, args.device, args.images)
    result = metrics(predictions, baseline)
    summary = {"configuration_sha256": identity, "panel_images": args.images, "metrics": result, "timing": timing,
               "fold_validation": setup["fold_validation"], "native_acceptance": False,
               "prediction_digest": digest(predictions), "baseline_prediction_digest": digest(baseline),
               "baseline_sha256": setup["baseline_sha256"]}
    seal(BASE / "summaries" / f"{identity}-{args.images}.json", summary)
    wall = time.monotonic() - START
    record_run(args.stage, args.model, args.format, args.recipe, args.images, identity, {
        "recipe": recipe.as_dict(), "metrics": result, "baseline_sha256": setup["baseline_sha256"],
        "source_sha256": source_identity(), "device": args.device,
        "cost": {"job_wall_seconds": wall, "gpu_seconds": wall if args.device == "cuda" else 0.0,
                 "preparation_seconds": preparation, "evaluation_seconds": timing["seconds_this_invocation"],
                 "baseline_seconds": baseline_timing["seconds_this_invocation"],
                 "note": "gpu_seconds is the whole job wall-clock while the GPU lock was held"}})
    print(json.dumps({"model": args.model, "format": args.format, "recipe": args.recipe, "images": args.images,
                      "top1": result["top1_percent"], "top5": result["top5_percent"],
                      "fp32_top1": result["fp32_top1_percent"], "delta": result["delta_top1_pp"],
                      "configuration": identity[:12], "wall_seconds": round(wall, 1)}), flush=True)
    return 0


def regress(args):
    """Proof on the real model that B2 with all switches off is v1: bit-identical logits and saved top-5."""
    import torch
    from tools.experiment_b.classifier import prepare_qdq
    from .data import v1_calibration
    from .engine import prepare_b2
    setup = model_setup(args.model, args.device)
    graph = setup["graph"]
    arrays, maxima, calibration = v1_calibration(args.model)
    recipe = named(args.recipe)
    v1_recipe = recipe.v1_equivalent()
    if v1_recipe is None:
        raise SystemExit("regression needs a v1-equivalent recipe")
    with torch.inference_mode():
        old, old_scales = prepare_qdq(graph, args.format, v1_recipe, arrays, maxima, args.device)
        new, new_scales = prepare_b2(graph, args.format, recipe, arrays, maxima, args.device)
    if (old_scales["activation_scales"] != new_scales["activation_scales"]
            or old_scales["weight_scales"] != new_scales["weight_scales"]):
        raise ValueError("B2-off scales differ from v1")
    with (ROOT / "results/summaries/b-stage-paired-1k-v2/configurations.csv").open() as stream:
        matches = [row for row in csv.DictReader(stream) if row["model"] == args.model and row["format"] == args.format
                   and row["recipe"] == v1_recipe and row["metric"] == "top1" and row["images"] == "1000"]
    if len(matches) != 1:
        raise ValueError("v1 1k configuration not found in the paired summary")
    v1_identity = matches[0]["configuration_sha256"]
    v1_configuration = unseal(ROOT / "artifacts/experiment_b/configurations" / f"{v1_identity}.json")
    scales_equal = v1_configuration["scales"]["activation_scales"] == new_scales["activation_scales"] and \
        v1_configuration["scales"]["weight_scales"] == new_scales["weight_scales"]
    identical, top5_equal, images = 0, 0, args.images
    for start in range(0, images, 8):
        batch = setup["rows"][start:start + 8]
        tensor = torch.from_numpy(np.array(setup["inputs"][start:start + len(batch)], dtype=np.float32)).to(args.device)
        with torch.inference_mode():
            a, b = old.run(tensor), new.run(tensor)
        if not torch.equal(a, b):
            raise ValueError(f"B2-off logits differ from v1 logits in batch {start}")
        identical += len(batch)
        for row, classes in zip(batch, b.topk(5, dim=1).indices.cpu().tolist()):
            saved = verify_prediction(ROOT / "artifacts/experiment_b/predictions" / v1_identity / (row["sha256"] + ".json"),
                                      v1_identity, row)
            top5_equal += saved["top5"] == classes
    record = {"model": args.model, "format": args.format, "recipe_name": args.recipe, "v1_recipe": v1_recipe,
              "images": images, "logit_bit_identical_images": identical,
              "sealed_v1_top5_reproduced_images": top5_equal, "v1_configuration_sha256": v1_identity,
              "scales_equal_to_sealed_v1_configuration": scales_equal, "runtime": setup["env"],
              "v1_runtime_equal": v1_configuration["runtime"] == setup["env"],
              "source_sha256": source_identity(), "wall_seconds": time.monotonic() - START}
    seal(BASE / "regression" / f"{args.model}--{args.format}--{args.recipe}--{images}.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "runtime"}), flush=True)
    return 0 if identical == images and top5_equal == images and scales_equal else 1


def build_inputs(args):
    from tools.experiment_b.classifier import configure, load_model
    from .data import cached_inputs
    configure("cpu")
    _, transform, _ = load_model(args.model, "cpu")
    for name, limit in (("imagenet_screen_1k", 1000), ("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES),
                        ("imagenet_calibration_2k", 2000)):
        array, rows = cached_inputs(name, limit, transform)
        print(json.dumps({"list": name, "images": len(rows), "shape": list(array.shape)}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for label in ("run", "regress"):
        p = sub.add_parser(label)
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--format", required=True)
        p.add_argument("--recipe", required=True)
        p.add_argument("--images", type=int, choices=PANELS, default=128)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        p.add_argument("--stage", default="diagnosis")
        p.add_argument("--audit", action="store_true")
    p = sub.add_parser("inputs")
    p.add_argument("--model", choices=MODELS, required=True)
    p = sub.add_parser("vendor")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--qconfig", default="x86_default")
    p.add_argument("--images", type=int, choices=PANELS, default=1000)
    p.add_argument("--calibration-images", type=int, default=2000)
    p.add_argument("--stage", default="vendor")
    p = sub.add_parser("export")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("recipes")
    args = parser.parse_args()
    from . import frozen  # noqa: F401  (registers frozen recipes when present)
    if args.command == "run":
        return run_configuration(args)
    if args.command == "regress":
        return regress(args)
    if args.command == "inputs":
        return build_inputs(args)
    if args.command == "vendor":
        from .vendor import run_vendor
        return run_vendor(args)
    if args.command == "export":
        from .export import run_export
        return run_export(args)
    print(json.dumps({name: {"switches": recipe.as_dict(), "hardware": hardware_semantics(recipe)}
                      for name, recipe in NAMED.items()}, indent=2))
    return 0
