"""Summaries of the detector study from the sealed run records and the stored bootstrap vectors (CPU, no model).

``.venv-b/bin/python -m tools.experiment_b2_det.summary [--tag TAG]`` writes, under
``results/summaries/b2-detector-v1/``: ``effects-128``, ``effects-1000``, ``exit-test-1000``, ``sentinels-1000``,
``ties-1000``, ``freeze-tests`` (JSON and CSV) and ``frozen-recipe`` (JSON).  A file that exists with different content is never
overwritten: pass ``--tag`` for a new versioned name.  Everything here is development evidence (screen lists).
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from tools.experiment_b.common import ROOT, atomic_json, digest, unseal
from . import arms2, arms3, frozen  # noqa: F401  (register every named arm)
from .recipe import hardware_semantics, named
from .report import rows as effect_rows
from .runner import BASE
from .stats import CONFIDENCE, RESAMPLES, SEED, paired

OUT = ROOT / "results/summaries/b2-detector-v1"
SENTINELS = ("int8", "int6", "fp8_e4m3fn", "fp7_e3m3", "fp6_e2m3", "log8", "posit8_es1", "bfp6", "mxfp8_e4m3")
TIE_TAGS = ("index", "reverse", "random0", "random1", "random2", "random3")
NOTE = "development evidence (COCO screen list); pointwise paired image-bootstrap intervals, no multiplicity adjustment"
STATISTICS = {"resamples": RESAMPLES, "seed": SEED, "confidence": CONFIDENCE, "units": "mAP points"}
RESIDUAL = (
    ("default_weights_only", "activations and input left in FP32: what weight quantization alone costs"),
    ("default_activations_only", "weights left in FP32: what activation quantization alone costs"),
    ("default_no_input", "input image left in FP32"),
    ("default_fp32_head", "both network outputs (box and class logits) and everything after them in FP32: "
                          "what is left is the backbone and neck"),
    ("default_fp32_box_logits", "box logits leave the accelerator as wide values"),
    ("default_fp32_class_logits", "class logits leave the accelerator as wide values"),
    ("default_head_logits", "projection constant, box expectation and sigmoid scores not stored as k-bit codes"),
    ("default_q_joins", "the 84-channel joins store the scores again as k-bit codes"),
    ("default_q_boxes", "decoded pixel boxes stored as k-bit codes"),
    ("default_unfused_silu", "v1 boundaries: convolution output and SiLU output both requantized"),
    ("default_concat_passthrough", "concatenations forward the codes of their inputs"),
    ("default_signed", "signed codes at the non-negative boundaries"),
    ("default_act_maxabs", "max-abs activation ranges"),
    ("default_weight_maxabs", "max-abs weight ranges"),
    ("default_bias_correction", "empirical bias correction put back (the classifier default)"),
)
DIFFERENCES = [
    {"switch": "bias correction", "classifier_default": "empirical", "detector": "none",
     "why": "leave-one-out arm better on dev128 with an interval excluding zero (rule of addendum 1)",
     "hardware": "none: the folded FP32 bias is used unchanged"},
    {"switch": "activation boundaries", "classifier_default": "fused_relu (clamps only; Hardswish and Hardsigmoid unfused)",
     "detector": "fused_silu",
     "why": "passed the 0.5-point test of the protocol on dev128",
     "hardware": "a convolution that feeds only SiLU keeps the wide accumulator; SiLU is evaluated on the wide value "
                 "and requantized once: a wide-precision nonlinearity or a large table, instead of a k-bit "
                 "code-to-code table"},
    {"switch": "code pass-through", "classifier_default": "max-pool", "detector": "max-pool, channel slice, nearest upsample",
     "why": "the two extra operators do not exist in the classifiers; no arithmetic happens in them",
     "hardware": "fewer requantizers, nothing added; a concatenation still requantizes its inputs to one scale"},
    {"switch": "unsigned codes", "classifier_default": "every ReLU/ReLU6/Hardsigmoid output", "detector":
     "input image, sigmoid scores and box expectation only",
     "why": "same rule (provably non-negative boundaries); SiLU outputs are not non-negative",
     "hardware": "the first convolution has an unsigned k-bit activation operand; all other operands are signed"},
    {"switch": "head: decoded boxes and 84-channel joins", "classifier_default": "does not exist", "detector":
     "not k-bit stores (q_boxes=false, q_joins=false)",
     "why": "putting the box store back costs more than 0.5 point with an interval excluding zero on dev128; the join "
            "exemption was adopted with it without a separate arm (checked afterwards: default_q_joins)",
     "hardware": "anchor decode and stride scaling run in the wide (FP32) postprocessor on the dequantized box "
                 "expectation; no k-bit store of pixel coordinates and no 84-channel k-bit tensor"},
    {"switch": "head: projection constant, box expectation, sigmoid scores, box and class logits",
     "classifier_default": "does not exist (classifier logits are k-bit codes)", "detector": "all stored as k-bit codes",
     "why": "putting them back cost nothing measurable at INT8 on dev128",
     "hardware": "the accelerator emits k-bit box-expectation codes and k-bit score codes; NMS compares score codes, "
                 "so equal scores are frequent and the tie rule is part of the contract"},
    {"switch": "shared-exponent formats", "classifier_default": "not covered by the classifier default", "detector":
     "block semantics of tools/experiment_b_ext/shared.py unchanged; boundary and head switches apply; range rules, "
     "pass-through and unsigned do not",
     "why": "the block scale is computed at run time (block max-abs covering power of two)",
     "hardware": "as v1 block formats, with the fused SiLU boundary and without box/join stores"},
]


def vector(stem):
    with np.load(BASE / "bootstrap" / f"{stem}.npz") as saved:
        return {"point": saved["point"], "draws": saved["draws"], "ties": json.loads(str(saved["ties"]))}


def records(images):
    found = {}
    for path in sorted((BASE / "runs" / str(images)).glob("*.json")):
        record = unseal(path)
        found[(record["format"], record["recipe_name"])] = record
    return found


def stem(record, images, tag="index"):
    return f"{record['configuration_sha256']}-{images}-{tag}"


def sealed_stem(name, recipe, images):
    return f"v1--{name}--{recipe}-{images}-sealed"


def cell(value):
    return {"map50_95": 100 * float(value["point"][0]), "map50": 100 * float(value["point"][1])}


def difference(left, right):
    """right minus left in points: ``{"delta", "interval", "delta_map50", "interval_map50"}``."""
    result = paired(left, right)
    return {"delta": result["map50_95"]["delta"], "interval": result["map50_95"]["interval"],
            "delta_map50": result["map50"]["delta"], "interval_map50": result["map50"]["interval"]}


def excludes_zero(interval):
    return interval[0] > 0 or interval[1] < 0


def exit_test(images=1000):
    found = records(images)
    fp32 = vector(stem(found[("fp32", "fp32")], images))
    default = vector(stem(found[("int8", "default")], images))
    against = difference(fp32, default)
    document = {"panel": f"screen{images}", "interpretation": NOTE, "statistics": STATISTICS,
                "fp32": cell(fp32), "int8_frozen_recipe": cell(default), "int8_minus_fp32": against,
                "target": "within about 1 mAP50-95 point, or a clear account of the residual",
                "point_estimate_within_one_point": bool(abs(against["delta"]) <= 1.0),
                "interval_within_one_point": bool(against["interval"][0] >= -1.0 and against["interval"][1] <= 1.0),
                "v1": {}, "residual_account": [], "not_available": []}
    for recipe in ("maxabs", "percentile_99_9"):
        old = vector(sealed_stem("int8", recipe, images))
        document["v1"][recipe] = {**cell(old), "minus_fp32": difference(fp32, old),
                                  "frozen_recipe_minus_v1": difference(old, default)}
    for name, reading in RESIDUAL:
        if ("int8", name) not in found:
            document["not_available"].append(name)
            continue
        arm = vector(stem(found[("int8", name)], images))
        document["residual_account"].append({"arm": name, "reading": reading, **cell(arm),
                                             "minus_fp32": difference(fp32, arm),
                                             "minus_frozen_recipe": difference(default, arm)})
    return document


def sentinels(images=1000):
    found = records(images)
    fp32 = vector(stem(found[("fp32", "fp32")], images))
    int8 = vector(stem(found[("int8", "default")], images))
    table = []
    for name in SENTINELS:
        if (name, "default") not in found:
            table.append({"format": name, "missing": True})
            continue
        default = vector(stem(found[(name, "default")], images))
        row = {"format": name, "configuration": found[(name, "default")]["configuration_sha256"][:12],
               "frozen": cell(default), "frozen_minus_fp32": difference(fp32, default),
               "frozen_minus_int8_frozen": difference(int8, default) if name != "int8" else None,
               "detections": default["ties"]["detections"], "distinct_scores": default["ties"]["distinct_scores"]}
        if (name, "default_head_logits") in found:
            logits = vector(stem(found[(name, "default_head_logits")], images))
            row.update(head_logits=cell(logits), head_logits_minus_frozen=difference(default, logits),
                       head_logits_minus_fp32=difference(fp32, logits))
        for recipe in ("maxabs", "percentile_99_9"):
            old = vector(sealed_stem(name, recipe, images))
            row[f"v1_{recipe}"] = cell(old)
            row[f"frozen_minus_v1_{recipe}"] = difference(old, default)
        table.append(row)
    return {"panel": f"screen{images}", "interpretation": NOTE, "statistics": STATISTICS, "fp32": cell(fp32),
            "recipe": "frozen detector recipe `default`; `head_logits` is the registered ablation default_head_logits; "
                      "v1 columns are the sealed Experiment B-extension predictions on the same images",
            "rows": table}


def envelope(value):
    ties = value["ties"]
    return {"fixed_rule": [100 * x for x in ties["stable"]],
            "true_positives_first": [100 * x for x in ties["true_positives_first"]],
            "false_positives_first": [100 * x for x in ties["false_positives_first"]],
            "envelope_width_map50_95": 100 * (ties["true_positives_first"][0] - ties["false_positives_first"][0]),
            "envelope_width_map50": 100 * (ties["true_positives_first"][1] - ties["false_positives_first"][1]),
            "detections": ties["detections"], "distinct_scores": ties["distinct_scores"],
            "detections_scored": ties["detections_scored"],
            "share_of_scored_detections_with_a_tied_score": ties["detections_with_a_tied_score"] / max(1, ties["detections_scored"])}


def ties(images=1000):
    found = records(images)
    table = []
    for (name, recipe), record in sorted(found.items()):
        if recipe != "default" and name != "fp32":
            continue
        files = {tag: BASE / "bootstrap" / f"{stem(record, images, tag)}.npz" for tag in TIE_TAGS}
        index = vector(stem(record, images))
        row = {"source": "fixed rule, new engine", "format": name, "recipe": recipe, **envelope(index)}
        if all(path.exists() for path in files.values()):
            orders = {tag: [100 * float(x) for x in vector(stem(record, images, tag))["point"]] for tag in TIE_TAGS}
            values = np.array(list(orders.values()))
            row.update(orders={tag: {"map50_95": v[0], "map50": v[1]} for tag, v in orders.items()},
                       order_range_map50_95=float(values[:, 0].max() - values[:, 0].min()),
                       order_range_map50=float(values[:, 1].max() - values[:, 1].min()),
                       largest_move_from_fixed_map50_95=float(np.abs(values[:, 0] - values[0, 0]).max()),
                       largest_move_from_fixed_map50=float(np.abs(values[:, 1] - values[0, 1]).max()))
        table.append(row)
    for name in ("fp32",) + SENTINELS:
        for recipe in (("baseline",) if name == "fp32" else ("maxabs", "percentile_99_9")):
            path = BASE / "bootstrap" / f"{sealed_stem(name, recipe, images)}.npz"
            if path.exists():
                table.append({"source": "sealed v1 predictions", "format": name, "recipe": f"v1_{recipe}",
                              **envelope(vector(path.stem))})
    return {"panel": f"screen{images}", "interpretation": "development evidence (COCO screen list)",
            "rule": {"fixed": "score descending, then anchor index ascending, then class index ascending, for NMS, for "
                              "the 300-detection cut and for emission; evaluation image order = ascending image sha256; "
                              "COCOeval then orders equal scores by a stable sort (emitted order, image order)",
                     "reverse": "tie key reversed and evaluation image order reversed",
                     "randomN": "random tie key per image (seed N) and random evaluation image order (seed 7919+N)",
                     "envelope": "all orders of equal scores across detections at fixed image-local matching: true "
                                 "positives first (upper) and false positives first (lower), fixed-rule detections"},
            "rows": table}


def recipe_statement():
    recipe = named("default")
    return {"name": "default", "switches": recipe.as_dict(), "hardware_semantics": hardware_semantics(recipe),
            "differences_from_classifier_default": DIFFERENCES,
            "same_as_classifier_default": ["unsigned codes at non-negative boundaries (integer formats only)",
                                           "activation range: MSE search (tools/experiment_b2/scales.py)",
                                           "weight range: MSE search per output channel",
                                           "max-pool code pass-through"],
            "postprocess": json.loads(json.dumps(__import__("tools.experiment_b2_det.post", fromlist=["RULE"]).RULE))}


# Every freeze decision with the arm pair that decides it (dev128, protocol + addendum 1) and its screen1k
# counterpart (descriptive only; no decision depends on it).  ``kind``: "default" = classifier-default switch
# (dropped only if its leave-one-out arm is better on dev128 with an interval excluding zero); "contract" =
# contract-changing option or head exemption (adopted only if the recipe without it is worse on dev128 by more than
# 0.5 point with an interval excluding zero).  Pairs are (base, arm); the delta is arm minus base.
FREEZE = (
    ("bias correction", "default", "dropped", ("cum6_bias_correction", "cum5_weight_mse"),
     ("default_bias_correction", "default"), "before"),
    ("unsigned codes", "default", "kept", ("cum5_weight_mse", "cum5_signed"), ("default", "default_signed"), "before"),
    ("activation MSE ranges", "default", "kept", ("cum5_weight_mse", "cum5_act_maxabs"),
     ("default", "default_act_maxabs"), "before"),
    ("weight MSE ranges", "default", "kept", ("cum5_weight_mse", "cum4_act_mse"), ("default", "default_weight_maxabs"),
     "before"),
    ("code pass-through (max-pool, slice, upsample)", "default", "kept", ("cum5_weight_mse", "cum5_no_passthrough"),
     None, "before"),
    ("fused SiLU boundary", "contract", "adopted", ("cum5_weight_mse", "cum5_fused_silu"),
     ("default_unfused_silu", "default"), "before"),
    ("concatenation pass-through", "contract", "not adopted", ("cum5_weight_mse", "cum5_concat_passthrough"),
     ("default", "default_concat_passthrough"), "before"),
    ("head: projection constant not quantized", "contract", "not adopted (quantized)",
     ("cum5_q_projection", "cum5_weight_mse"), None, "before"),
    ("head: box logits not quantized", "contract", "not adopted (quantized; classed as a diagnostic before running)",
     ("cum5_weight_mse", "cum5_fp32_box_logits"), ("default", "default_fp32_box_logits"), "before"),
    ("head: box expectation not quantized", "contract", "not adopted (quantized)", ("cum5_q_dfl", "cum5_weight_mse"),
     None, "before"),
    ("head: decoded boxes not quantized", "contract", "adopted", ("cum5_q_boxes", "cum5_weight_mse"),
     ("default_q_boxes", "default"), "before"),
    ("head: class logits not quantized", "contract", "not adopted (quantized; classed as a diagnostic before running)",
     ("cum5_weight_mse", "cum5_fp32_class_logits"), ("default", "default_fp32_class_logits"), "before"),
    ("head: sigmoid scores not quantized", "contract", "not adopted (quantized)", ("cum5_q_scores_also", "cum5_weight_mse"),
     ("default", "default_head_logits"), "before"),
    ("head: 84-channel joins not quantized", "contract", "adopted together with the box exemption, no own arm",
     ("default_q_joins", "default"), ("default_q_joins", "default"), "after (addendum 2)"),
)


def freeze_tests():
    tables = {images: records(images) for images in (128, 1000)}

    def pair(images, names):
        found = tables[images]
        if names is None or any(("int8", n) not in found for n in names):
            return None
        base, arm = (vector(stem(found[("int8", n)], images)) for n in names)
        return {"base": names[0], "arm": names[1], **difference(base, arm)}

    rows = []
    for decision, kind, frozen_choice, dev, screen, measured in FREEZE:
        row = {"decision": decision, "kind": kind, "frozen_recipe": frozen_choice, "decision_pair_measured": measured,
               "dev128": pair(128, dev), "screen1k": pair(1000, screen) if screen else None}
        d = row["dev128"]
        if d is None:
            row["test_met_on_dev128"] = None
        elif kind == "default":
            row["test_met_on_dev128"] = bool(d["delta"] > 0 and excludes_zero(d["interval"]))
        else:
            row["test_met_on_dev128"] = bool(d["delta"] > 0.5 and excludes_zero(d["interval"]))
        rows.append(row)
    return {"interpretation": NOTE, "statistics": STATISTICS,
            "rule": {"default": "classifier-default switch. Pair (with the switch, without it); delta = without minus "
                                "with. test_met = dropped: the leave-one-out arm is better on dev128 with a paired 95 "
                                "percent interval excluding zero (addendum 1)",
                     "contract": "contract-changing option or head exemption. Pair (without the option, with it); "
                                 "delta = with minus without. test_met = adopted: the recipe without it is worse on "
                                 "dev128 by more than 0.5 point with a paired 95 percent interval excluding zero",
                     "screen1k": "the same question asked around the frozen recipe (default_* ablations) where such an "
                                 "arm exists; descriptive, no decision depends on it"},
            "rows": rows}


def flat(prefix, value, out):
    if isinstance(value, dict):
        for key, inner in value.items():
            flat(f"{prefix}.{key}" if prefix else key, inner, out)
    elif isinstance(value, list) and len(value) == 2 and all(isinstance(x, float) for x in value):
        out[prefix + ".low"], out[prefix + ".high"] = round(value[0], 3), round(value[1], 3)
    else:
        out[prefix] = round(value, 3) if isinstance(value, float) else value
    return out


def write(name, document, table=None):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}.json"
    if target.exists():
        if json.loads(target.read_text())["sha256"] == digest(document):
            print(f"unchanged: {target.name}")
            return False
        raise SystemExit(f"{target} exists with different content; use --tag for a new versioned file")
    atomic_json(target, {"payload": document, "sha256": digest(document)})
    if table is not None:
        flattened = [flat("", row, {}) for row in table]
        keys = []
        for row in flattened:
            keys += [k for k in row if k not in keys]
        with (OUT / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flattened)
    print(f"wrote {target}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--only", nargs="*", default=None,
                        choices=("effects-128", "effects-1000", "exit-test-1000", "sentinels-1000", "ties-1000", "frozen-recipe",
                                 "freeze-tests"))
    args = parser.parse_args()
    suffix = f"-{args.tag}" if args.tag else ""
    wanted = lambda key: args.only is None or key in args.only  # noqa: E731
    for images in (128, 1000):
        if wanted(f"effects-{images}"):
            table = effect_rows(images)
            write(f"effects-{images}{suffix}", {"panel": "dev128" if images == 128 else "screen1k", "interpretation": NOTE,
                                                "statistics": STATISTICS, "rows": table}, table)
    if wanted("exit-test-1000"):
        document = exit_test()
        write(f"exit-test-1000{suffix}", document, document["residual_account"])
    if wanted("sentinels-1000"):
        document = sentinels()
        write(f"sentinels-1000{suffix}", document, document["rows"])
    if wanted("ties-1000"):
        document = ties()
        write(f"ties-1000{suffix}", document, document["rows"])
    if wanted("frozen-recipe"):
        write(f"frozen-recipe{suffix}", recipe_statement())
    if wanted("freeze-tests"):
        document = freeze_tests()
        write(f"freeze-tests{suffix}", document, document["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
