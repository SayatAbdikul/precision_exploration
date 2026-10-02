"""Evaluation of nearest and learned-rounding arms on the frozen 1k screen (development evidence).

Every arm is one sealed record plus one small ``npz`` with the per-image
readout (top-5 by ``topk`` as the B2 runner takes it, tie size, whether the
label is among the maxima, whether it is the lowest-index maximum).  Records
are written once.  Nearest arms with one format and a frozen B2 recipe must
reproduce the sealed B2 top-5 lists, which proves that this builder is the B2
engine for them.
"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np

from tools.experiment_b.common import ROOT, digest, formats, seal, unseal
from tools.experiment_b2.common import BASE as B2_BASE, BIAS_CORRECTION_IMAGES, PROTOCOL as B2_PROTOCOL, source_identity

from .fit import BASE, PROTOCOL, find_fit, load_fit, recon_sources

MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
MATRIX_FORMATS = ("int8", "int6", "int4", "fp6_e2m3", "posit8_es1")
NEAREST_RECIPES = {"N-default": "default", "N-nobc": "default_no_bias_correction", "N-minimal": "minimal"}
LEARNED_RECIPES = {"L-nobc": "default_no_bias_correction", "L-bc": "default"}


def arm(name, wformat, aformat, recipe, *, rule=None, fit=None, bias="recipe"):
    """One evaluation arm.  ``fit``: ``(scale rule, mode, seed)`` of the learned rounding, or ``None`` for nearest."""
    return {"arm": name, "weight_format": wformat, "activation_format": aformat, "recipe": recipe,
            "weight_rule": rule, "fit": fit, "bias_correction": bias}


def label(spec):
    parts = [spec["arm"], f"w-{spec['weight_format']}", f"a-{spec['activation_format'] or 'fp32'}"]
    if spec["recipe"]:
        parts.append(spec["recipe"])
    if spec["weight_rule"]:
        parts.append(spec["weight_rule"])
    if spec["bias_correction"] not in ("recipe", "none"):
        parts.append("bias-" + spec["bias_correction"])
    if spec["fit"]:
        parts.append(f"fit-{spec['fit'][0]}-{spec['fit'][1]}-s{spec['fit'][2]}")
    return "--".join(parts)


def matrix_arms(wformats=MATRIX_FORMATS):
    result = []
    for wformat in wformats:
        for aformat in ([wformat, "int8"] if wformat == "int4" else [wformat]):
            for name, recipe in NEAREST_RECIPES.items():
                result.append(arm(name, wformat, aformat, recipe))
            for name, recipe in LEARNED_RECIPES.items():
                result.append(arm(name, wformat, aformat, recipe, fit=("mse_per_channel", "fp32in", 0)))
        result.append(arm("N-A32", wformat, None, None, rule="mse_per_channel", bias="none"))
        result.append(arm("L-A32", wformat, None, None, rule="mse_per_channel", bias="none",
                          fit=("mse_per_channel", "fp32in", 0)))
    return result


def faithful_arms(seeds=(0, 1, 2)):
    """The paper's setting on ResNet18: INT4 weights, one MSE scale per layer, activations FP32 (and INT8)."""
    rule = "mse_per_layer"
    result = [arm("N-A32", "int4", None, None, rule=rule, bias="none"),
              arm("N-A32-bc", "int4", None, None, rule=rule, bias="empirical"),
              arm("N-nobc", "int4", "int8", "default_no_bias_correction", rule=rule)]
    for seed in seeds:
        result.append(arm("L-A32", "int4", None, None, rule=rule, bias="none", fit=(rule, "fp32in", seed)))
        result.append(arm("L-nobc", "int4", "int8", "default_no_bias_correction", rule=rule, fit=(rule, "fp32in", seed)))
    return result


def secondary_arms(kind):
    if kind == "maxabs":  # INT4 learned on the minimal recipe's max-abs weight scales
        fit = ("maxabs_per_channel", "fp32in", 0)
        return [arm("L-minimal", "int4", aformat, "minimal", fit=fit) for aformat in ("int4", "int8")] + \
               [arm("N-A32", "int4", None, None, rule="maxabs_per_channel", bias="none"),
                arm("L-A32", "int4", None, None, rule="maxabs_per_channel", bias="none", fit=fit)]
    if kind == "b2in":  # reconstruction input taken from the B2 engine with quantized activations
        result = []
        for wformat in ("int4", "int6"):
            fit = ("mse_per_channel", f"b2in-{wformat}", 0)
            result += [arm("LQ-nobc", wformat, wformat, "default_no_bias_correction", fit=fit),
                       arm("LQ-bc", wformat, wformat, "default", fit=fit)]
        return result
    raise ValueError(kind)


GROUPS = {"matrix": matrix_arms, "faithful": faithful_arms, "maxabs": lambda: secondary_arms("maxabs"),
          "b2in": lambda: secondary_arms("b2in")}


def run_screen(run, inputs, rows, device, batch):
    """Logits of all screen images (CPU, FP32) and the top-5 lists ``topk`` returns on the device."""
    import torch
    parts, top5 = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            tensor = torch.from_numpy(np.array(inputs[start:start + batch], dtype=np.float32)).to(device)
            output = run(tensor)
            if not torch.isfinite(output).all() or tuple(output.shape) != (len(tensor), 1000):
                raise ValueError("invalid classifier output")
            top5.extend(output.topk(5, dim=1).indices.cpu().tolist())
            parts.append(output.cpu())
    return torch.cat(parts), top5


def sealed_b2_predictions(model, format_name, recipe_name, rows):
    """Sealed B2 top-5 lists of a frozen recipe at the current B2 source identity, or ``None``."""
    from tools.experiment_b2.runner import load_predictions
    found = []
    for path in sorted((B2_BASE / "runs").glob(f"*/{model}--{format_name}--{recipe_name}--1000--*.json")):
        record = unseal(path)
        if record["source_sha256"] == source_identity():
            found.append(record["configuration_sha256"])
    found = sorted(set(found))
    if len(found) != 1:
        return None, None
    return found[0], [row["top5"] for row in load_predictions(found[0], rows)]


def write_arrays(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    temporary.replace(path)


def readout(logits, top5, rows):
    from tools.experiment_b2.verify import tie_statistics
    ties = tie_statistics(logits, rows, per_image=True)
    per_image = ties.pop("per_image")
    labels = np.array([int(row["label"]) for row in rows])
    top = np.asarray(top5, dtype=np.int16)
    arrays = {"top5_topk": top, "tie_size": np.asarray(per_image["tie_size"], dtype=np.uint16),
              "label_among_maxima": np.asarray(per_image["label_among_maxima"], dtype=np.uint8),
              "label_is_lowest_index_maximum": np.asarray(per_image["label_is_lowest_index_maximum"], dtype=np.uint8)}
    ties["top1_percent_topk"] = 100 * float((top[:, 0] == labels).mean())
    ties["top5_percent_topk"] = 100 * float((top == labels[:, None]).any(axis=1).mean())
    return ties, arrays


def run_group(model, group, device, *, budget_seconds=600.0, only=None, log=print):
    """Evaluate every missing arm of ``group`` for ``model``.  Returns 0 when all exist, 3 when stopped by the budget."""
    import torch
    from tools.experiment_b2 import frozen  # noqa: F401
    from tools.experiment_b2.data import cached_inputs, v1_calibration
    from tools.experiment_b2.recipe import named
    from tools.experiment_b2.runner import load_predictions, model_setup
    from .engine import build_engine, state_digest
    started = time.monotonic()
    specs = [spec for spec in GROUPS[group]() if only is None or label(spec) in only or spec["arm"] in only]
    folder = BASE / "evals" / model
    folder.mkdir(parents=True, exist_ok=True)
    missing = [spec for spec in specs if not (folder / f"{label(spec)}.json").exists()]
    if not missing:
        log(json.dumps({"model": model, "group": group, "status": "all arms recorded", "arms": len(specs)}))
        return 0
    setup = model_setup(model, device)
    graph, rows, inputs = setup["graph"], setup["rows"], setup["inputs"]
    batch = B2_PROTOCOL["inference_batch_size"]
    arrays, maxima, calibration = v1_calibration(model)
    bias_inputs, bias_rows = cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"],
                                           build=False)
    entries = {row["name"]: row for row in formats()}
    baseline_path = folder / "fp32.json"
    if not baseline_path.exists():
        logits, top5 = run_screen(graph, inputs, rows, device, batch)
        sealed = [row["top5"] for row in load_predictions(setup["baseline_sha256"], rows)]
        ties, per_image = readout(logits, top5, rows)
        write_arrays(folder / "fp32.npz", per_image)
        seal(baseline_path, {"protocol": PROTOCOL, "model": model, "arm": "fp32", "label": "fp32",
                             "baseline_sha256": setup["baseline_sha256"], "images": len(rows),
                             "rows_sha256": digest(rows), "readout": ties,
                             "sealed_b2_top5_lists_reproduced": sum(a == b for a, b in zip(sealed, top5)),
                             "evidence": "development_evidence_screen1k"})
    fits = {}
    for spec in missing:
        if time.monotonic() - started > budget_seconds:
            log(json.dumps({"model": model, "group": group, "status": "budget reached"}))
            return 3
        tick = time.monotonic()
        name = label(spec)
        recipe = named(spec["recipe"]) if spec["recipe"] else None
        learned, fit_record = None, None
        if spec["fit"]:
            rule, mode, seed = spec["fit"]
            try:
                fit_folder = find_fit(model, spec["weight_format"], rule, seed, mode)
            except ValueError as error:
                log(json.dumps({"model": model, "arm": name, "status": "skipped", "reason": str(error)}))
                continue
            if fit_folder not in fits:
                fits[fit_folder] = load_fit(fit_folder, graph, device)
            learned, summary = fits[fit_folder]
            fit_record = {"folder": str(fit_folder.relative_to(ROOT)), "fit_sha256": summary["fit_sha256"],
                          "learned_state_sha256": summary["learned_state_sha256"]}
        correction = None if spec["bias_correction"] == "recipe" else spec["bias_correction"]
        run, metadata, deployed = build_engine(
            graph, spec["weight_format"], spec["activation_format"], recipe, arrays, maxima, device,
            weight_rule=spec["weight_rule"], learned=learned, bias_inputs=bias_inputs, bias_correction=correction)
        logits, top5 = run_screen(run, inputs, rows, device, batch)
        ties, per_image = readout(logits, top5, rows)
        sealed_check = None
        if (learned is None and recipe is not None and spec["weight_rule"] is None
                and spec["weight_format"] == spec["activation_format"]):
            identity, sealed = sealed_b2_predictions(model, spec["weight_format"], spec["recipe"], rows)
            sealed_check = {"b2_configuration_sha256": identity,
                            "top5_lists_reproduced": None if sealed is None else sum(a == b for a, b in zip(sealed, top5))}
        configuration = {
            "protocol": PROTOCOL, "model": model, "model_context": setup["context"], "arm": spec["arm"], "label": name,
            "weight_format": spec["weight_format"], "weight_format_sha256": entries[spec["weight_format"]]["sha256"],
            "activation_format": spec["activation_format"], "recipe_name": spec["recipe"],
            "recipe": recipe.as_dict() if recipe else None, "weight_scale_rule": metadata["weight_scale_rule"],
            "bias_correction": correction if correction is not None else (recipe.bias_correction if recipe else "none"),
            "bias_correction_images": {"images": len(bias_rows), "rows_sha256": digest(bias_rows)},
            "rounding": "learned" if learned is not None else "nearest", "fit": fit_record,
            "b2_source_sha256": source_identity(), "recon_sources": recon_sources(),
            "calibration_observations": calibration["identity"], "runtime": setup["env"],
            "deployed_state_sha256": state_digest(deployed),
            "activation_scales_sha256": digest(metadata["activation_scales"]),
            "weight_scales_sha256": digest(metadata["weight_scales"]), "inference_batch_size": batch,
            "evaluation_list": "imagenet_screen_1k", "rows_sha256": digest(rows), "baseline_sha256": setup["baseline_sha256"]}
        changed = metadata["rounding"]
        record = {**configuration, "configuration_sha256": digest(configuration), "images": len(rows), "readout": ties,
                  "weights_differing_from_nearest": sum(row["differ_from_nearest"] for row in changed.values()) if changed else 0,
                  "weights": sum(row["weights"] for row in changed.values()) if changed else None,
                  "sealed_b2_check": sealed_check, "evidence": "development_evidence_screen1k",
                  "cost": {"seconds": time.monotonic() - tick, "device": device}}
        write_arrays(folder / f"{name}.npz", per_image)
        seal(folder / f"{name}.json", record)
        log(json.dumps({"model": model, "arm": name, "topk": round(ties["top1_percent_topk"], 2),
                        "expected": round(ties["top1_percent_expected"], 2),
                        "lowest": round(ties["top1_percent_lowest_index"], 2), "tied": ties["images_with_tied_top1"],
                        "sealed": sealed_check and sealed_check["top5_lists_reproduced"],
                        "seconds": round(time.monotonic() - tick, 1)}), flush=True)
        del run, deployed
        torch.cuda.empty_cache() if device == "cuda" else None
    return 0
