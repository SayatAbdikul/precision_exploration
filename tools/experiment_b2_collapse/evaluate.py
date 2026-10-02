"""Arms of lane Q3 on the frozen 1k screen (development evidence) and their sealed records.

One arm = one engine (cell x correction policy, or cell x weight-scale rules) evaluated at batch 8
on the 1000 screen images.  Each arm writes, once, ``evals/<model>/<label>.json`` (sealed: identity,
tie-aware readout, one-class share, per-layer correction report, reproduction check) and
``evals/<model>/<label>.npz`` (per image: top-5 as ``topk`` returns it, tie size, label among the
maxima, label is the lowest-index maximum, lowest-index argmax).
"""
from __future__ import annotations

import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, seal, unseal
from tools.experiment_b2.common import BASE as B2_BASE, PROTOCOL as B2_PROTOCOL, source_identity

from .common import BASE, PROTOCOL_FILE, own_sources

MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
FIT = ("mse_per_channel", "fp32in", 0)
VARIANTS = ("none", "global", "executed", "dfq_weights_only", "cap_rms", "linear_only", "local_empirical",
            "analytic_fp32")
IMAGE_COUNTS = (32, 64, 128)


def spec(cell, wformat, aformat, recipe, correction, *, rule=None, fit=None, images=None, weight_rules=None,
         tag=None):
    return {"cell": cell, "weight_format": wformat, "activation_format": aformat, "recipe": recipe,
            "weight_rule": rule, "fit": fit, "correction": correction, "correction_images": images,
            "weight_rules": weight_rules, "tag": tag}


def label(s):
    parts = [s["cell"], s["correction"]]
    if s["correction_images"]:
        parts.append(f"n{s['correction_images']}")
    if s["tag"]:
        parts.append(s["tag"])
    return "--".join(parts)


EXECUTED_CELLS = ("faith-w4-a32-perlayer-N", "w4-a4-default-N", "w8-a8-default-N")


def _variants(cell, wformat, aformat, recipe, *, rule=None, fit=None, images=False, weights_only_differs=True):
    out = []
    for variant in VARIANTS:
        if variant == "dfq_weights_only" and not weights_only_differs:
            continue
        if variant == "executed" and cell not in EXECUTED_CELLS:
            continue
        out.append(spec(cell, wformat, aformat, recipe, variant, rule=rule, fit=fit))
    if images:
        out += [spec(cell, wformat, aformat, recipe, "global", rule=rule, fit=fit, images=n) for n in IMAGE_COUNTS]
    return out


def mode1_arms(model):
    """Mode 1: the 5 collapse cells, the faithfulness cell, INT4/5/6 equal formats, MobileNetV3 INT8."""
    arms = []
    if model == "resnet18":
        arms += _variants("faith-w4-a32-perlayer-N", "int4", None, None, rule="mse_per_layer", images=True,
                          weights_only_differs=False)
        arms += _variants("w4-a8-default-N", "int4", "int8", "default")
    if model in ("resnet18", "mobilenet_v2"):
        arms += _variants("w4-a4-default-L", "int4", "int4", "default", fit=FIT)
    arms += _variants("w4-a4-default-N", "int4", "int4", "default", images=model != "mobilenet_v3_large")
    arms += _variants("w5-a5-default-N", "int5", "int5", "default")
    arms += _variants("w6-a6-default-N", "int6", "int6", "default")
    if model == "mobilenet_v3_large":
        arms += _variants("w8-a8-default-N", "int8", "int8", "default", images=True)
        arms += _variants("w8-a8-default-L", "int8", "int8", "default", fit=FIT)
    return arms


WIDE = ("fp8_e5m2", "fp6_e3m2", "log6", "posit6_es1")


def mode3_arms(model):
    """Mode 3: the two single-switch recipes for the wide-exponent scalar formats, and correction variants."""
    arms = []
    for fmt in WIDE:
        for recipe in ("default_no_bias_correction", "default_weight_maxabs"):
            arms.append(spec(f"{fmt}-{recipe}-N", fmt, fmt, recipe, "b2"))
        if model != "mobilenet_v3_large" and fmt != "posit6_es1":
            for variant in ("global", "local_empirical", "analytic_fp32", "linear_only", "cap_rms",
                            "dfq_weights_only"):
                arms.append(spec(f"{fmt}-default-N", fmt, fmt, "default", variant))
    return arms


POSITS = ("posit8_es1", "posit6_es1")


def mode2_arms(model, graph=None):
    """Mode 2: weight anchors under the minimal recipe; MobileNetV2 posit8 layer-group swaps."""
    from tools.experiment_b2_recon.engine import weight_layers
    arms = []
    names = [node.name for node, _ in weight_layers(graph)] if graph is not None else None
    for fmt in POSITS:
        for rule in ("maxabs_per_channel", "anchor_one", "mse_per_channel"):
            rules = {n: rule for n in names} if names else None
            arms.append(spec(f"{fmt}-minimal-anchor", fmt, fmt, "minimal", "none", weight_rules=rules, tag=rule))
    if model == "mobilenet_v2" and graph is not None:
        from .build import layer_groups
        for group, members in layer_groups(graph).items():
            only = {n: ("maxabs_per_channel" if n in members else "mse_per_channel") for n in names}
            but = {n: ("mse_per_channel" if n in members else "maxabs_per_channel") for n in names}
            arms.append(spec("posit8_es1-minimal-swap", "posit8_es1", "posit8_es1", "minimal", "none",
                             weight_rules=only, tag=f"maxabs-only-{group}"))
            arms.append(spec("posit8_es1-minimal-swap", "posit8_es1", "posit8_es1", "minimal", "none",
                             weight_rules=but, tag=f"maxabs-except-{group}"))
    return arms


GROUPS = {"mode1": mode1_arms, "mode2": mode2_arms, "mode3": mode3_arms}


def one_class(logits):
    """Largest share of images on which one class is among the maxima, plus the expected and lowest-index shares."""
    import torch
    maxima = logits == logits.amax(dim=1, keepdim=True)
    tied = maxima.sum(dim=1, keepdim=True).double()
    among = maxima.double().sum(dim=0)
    expected = (maxima.double() / tied).sum(dim=0)
    lowest = torch.bincount(maxima.int().argmax(dim=1), minlength=logits.shape[1]).double()
    n = logits.shape[0]
    top = int(among.argmax())
    return {"class": top, "share_among_maxima": float(among[top] / n),
            "share_expected": float(expected.max() / n), "class_expected": int(expected.argmax()),
            "share_lowest_index": float(lowest.max() / n), "class_lowest_index": int(lowest.argmax()),
            "distinct_lowest_index_predictions": int((lowest > 0).sum()), "collapse": bool(among[top] / n >= 0.90)}


def reference_top5(model, s, rows):
    """Top-5 lists of an existing record for this arm (L6 eval or B2 matrix cell), or ``None``."""
    recon = ROOT / "artifacts/experiment_b2_recon/evals" / model
    mapping = None
    cell, correction = s["cell"], s["correction"]
    if s["correction_images"] or s["weight_rules"] and s["tag"] != "maxabs_per_channel":
        return None
    if correction in ("b2", "global"):
        if cell == "faith-w4-a32-perlayer-N":
            mapping = "N-A32-bc--w-int4--a-fp32--mse_per_layer--bias-empirical"
        elif s["recipe"] == "default" and s["fit"] is None:
            mapping = f"N-default--w-{s['weight_format']}--a-{s['activation_format']}--default"
        elif s["recipe"] == "default" and s["fit"] == FIT:
            mapping = (f"L-bc--w-{s['weight_format']}--a-{s['activation_format']}--default--"
                       "fit-mse_per_channel-fp32in-s0")
    elif correction == "none" and (s["recipe"] == "default" or cell == "faith-w4-a32-perlayer-N"):
        if cell == "faith-w4-a32-perlayer-N":
            mapping = "N-A32--w-int4--a-fp32--mse_per_layer"
        elif s["fit"] is None:
            mapping = f"N-nobc--w-{s['weight_format']}--a-{s['activation_format']}--default_no_bias_correction"
        else:
            mapping = (f"L-nobc--w-{s['weight_format']}--a-{s['activation_format']}--default_no_bias_correction--"
                       "fit-mse_per_channel-fp32in-s0")
    if mapping is not None and (recon / f"{mapping}.json").exists():
        record = unseal(recon / f"{mapping}.json")
        arrays = np.load(recon / f"{mapping}.npz")
        return {"source": str((recon / f"{mapping}.json").relative_to(ROOT)),
                "deployed_state_sha256": record.get("deployed_state_sha256"),
                "top5": arrays["top5_topk"].tolist()}
    recipe = s["recipe"] if correction == "b2" or (correction == "global" and s["recipe"] == "default") else None
    if s["weight_rules"] and s["tag"] == "maxabs_per_channel":
        recipe = "minimal"
    if recipe and s["fit"] is None and s["weight_rule"] is None and s["weight_format"] == s["activation_format"]:
        cells = sorted((B2_BASE / "matrix/cells").glob(f"{model}--{s['weight_format']}--{recipe}--1000--*.json"))
        if len(cells) == 1:
            record = unseal(cells[0])
            arrays = np.load(ROOT / record["readout_file"])
            return {"source": str(cells[0].relative_to(ROOT)), "deployed_state_sha256": None,
                    "top5": arrays["top5_topk"].astype(int).tolist()}
    return None


def run_group(model, group, device, *, budget_seconds=600.0, only=None, log=print):
    """Evaluate every missing arm of ``group``; 0 when all exist, 3 when stopped by the time budget."""
    import torch
    from tools.experiment_b2 import frozen  # noqa: F401
    from tools.experiment_b2.common import BIAS_CORRECTION_IMAGES
    from tools.experiment_b2.data import cached_inputs, v1_calibration
    from tools.experiment_b2.recipe import named
    from tools.experiment_b2.runner import model_setup
    from tools.experiment_b2_recon.engine import state_digest
    from tools.experiment_b2_recon.evaluate import readout, run_screen, write_arrays
    from tools.experiment_b2_recon.fit import find_fit, load_fit, recon_sources
    from .build import build
    started = time.monotonic()
    folder = BASE / "evals" / model
    folder.mkdir(parents=True, exist_ok=True)
    setup = model_setup(model, device)
    graph, rows, inputs = setup["graph"], setup["rows"], setup["inputs"]
    specs = [s for s in GROUPS[group](model, graph) if only is None or label(s) in only or s["cell"] in only] \
        if group == "mode2" else [s for s in GROUPS[group](model) if only is None or label(s) in only or s["cell"] in only]
    missing = [s for s in specs if not (folder / f"{label(s)}.json").exists()]
    log(json.dumps({"model": model, "group": group, "arms": len(specs), "missing": len(missing)}), flush=True)
    if not missing:
        return 0
    batch = B2_PROTOCOL["inference_batch_size"]
    arrays, maxima, calibration = v1_calibration(model)
    bias_inputs, bias_rows = cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"],
                                           build=False)
    labels = np.array([int(row["label"]) for row in rows])
    fits = {}
    for s in missing:
        if time.monotonic() - started > budget_seconds:
            log(json.dumps({"model": model, "group": group, "status": "budget reached"}), flush=True)
            return 3
        tick = time.monotonic()
        name = label(s)
        recipe = named(s["recipe"]) if s["recipe"] else None
        learned, fit_record = None, None
        if s["fit"]:
            rule, mode, seed = s["fit"]
            fit_folder = find_fit(model, s["weight_format"], rule, seed, mode)
            if fit_folder not in fits:
                fits[fit_folder] = load_fit(fit_folder, graph, device)
            learned, summary = fits[fit_folder]
            fit_record = {"folder": str(fit_folder.relative_to(ROOT)), "fit_sha256": summary["fit_sha256"]}
        engine_spec = {**s, "recipe": recipe}
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        run, metadata, deployed = build(graph, engine_spec, arrays, maxima, device, bias_inputs, learned=learned)
        prepare_seconds = time.monotonic() - tick
        logits, top5 = run_screen(run, inputs, rows, device, batch)
        ties, per_image = readout(logits, top5, rows)
        per_image["argmax_lowest"] = (logits == logits.amax(dim=1, keepdim=True)).int().argmax(dim=1).numpy().astype(np.uint16)
        reference = reference_top5(model, s, rows)
        check = None
        deployed_sha = state_digest(deployed)
        if reference is not None:
            check = {"source": reference["source"],
                     "top5_lists_equal": int(sum(a == b for a, b in zip(reference["top5"], top5))),
                     "deployed_state_equal": (None if reference["deployed_state_sha256"] is None
                                              else reference["deployed_state_sha256"] == deployed_sha)}
        images_used = s["correction_images"] or len(bias_rows)
        configuration = {
            "protocol": {"path": str(PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(PROTOCOL_FILE)},
            "model": model, "model_context": setup["context"], "label": name,
            **{k: v for k, v in s.items() if k != "weight_rules"},
            "weight_rules_sha256": digest(s["weight_rules"]) if s["weight_rules"] else None,
            "recipe_definition": recipe.as_dict() if recipe else None,
            "correction_images": {"images": images_used, "rows_sha256": digest(bias_rows[:images_used])},
            "fit_record": fit_record, "b2_source_sha256": source_identity(), "recon_sources": recon_sources(),
            "own_sources": own_sources(), "calibration_observations": calibration["identity"],
            "runtime": setup["env"], "deployed_state_sha256": deployed_sha,
            "inference_batch_size": batch, "evaluation_list": "imagenet_screen_1k", "rows_sha256": digest(rows)}
        record = {**configuration, "configuration_sha256": digest(configuration), "images": len(rows),
                  "readout": ties, "one_class": one_class(logits),
                  "top1_expected_percent": ties["top1_percent_expected"],
                  "reproduction_check": check,
                  "correction_report": metadata.get("correction_report") or metadata.get("bias_correction"),
                  "weight_sqnr_db": metadata.get("weight_sqnr_db"),
                  "evidence": "development_evidence_screen1k",
                  "cost": {"seconds": time.monotonic() - tick, "prepare_seconds": prepare_seconds, "device": device,
                           "peak_gpu_mib": (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None}}
        write_arrays(folder / f"{name}.npz", per_image)
        seal(folder / f"{name}.json", record)
        log(json.dumps({"model": model, "arm": name, "expected": round(ties["top1_percent_expected"], 2),
                        "one_class": round(record["one_class"]["share_among_maxima"], 3),
                        "check": check and {k: v for k, v in check.items() if k != "source"},
                        "seconds": round(time.monotonic() - tick, 1),
                        "peak_mib": record["cost"]["peak_gpu_mib"] and round(record["cost"]["peak_gpu_mib"])}),
            flush=True)
        del run, deployed, logits
        torch.cuda.empty_cache() if device == "cuda" else None
    return 0
