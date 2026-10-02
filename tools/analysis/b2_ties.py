"""Experiment B2, revision r3: logit ties and batch-size sensitivity tables.

Reads the sealed verification records with one tag (default ``r3c``) under
``artifacts/experiment_b2/verification/{batch-sensitivity,vendor-probe}/`` and
writes new summary files next to the existing B2 summaries.  It reads no image
and runs no model.  Existing summary files are never replaced: the script
refuses to write over a file whose content would change.

Tie rules (see ``tools.experiment_b2.verify.tie_statistics``): ``sealed`` is
what ``topk`` returned in the sealed run, ``strict`` / ``expected`` /
``lowest_index`` / ``optimistic`` are computed from the logits.
"""
from __future__ import annotations

import argparse
import csv
import io
import json

import numpy as np

from tools.experiment_b.common import ROOT, unseal

BASE = ROOT / "artifacts/experiment_b2/verification"
OUT = ROOT / "results/summaries/b2-baseline-repair-v1"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
FORMATS = ("int8", "int6", "int4", "fp8_e4m3fn", "fp7_e3m3", "fp6_e2m3", "log8", "posit8_es1", "nf4")
RESAMPLES, SEED = 10000, 20260927
RULES = ("strict", "expected", "lowest_index", "optimistic")


def credit(per_image, rule):
    """Per-image top-1 credit under one tie rule, from the three per-image lists of a record."""
    size = np.asarray(per_image["tie_size"], dtype=np.float64)
    among = np.asarray(per_image["label_among_maxima"], dtype=np.float64)
    if rule == "strict":
        return among * (size == 1)
    if rule == "expected":
        return among / size
    if rule == "lowest_index":
        return np.asarray(per_image["label_is_lowest_index_maximum"], dtype=np.float64)
    if rule == "optimistic":
        return among
    raise ValueError(rule)


def paired(left, right):
    """Mean credit difference (left minus right) in points with a pointwise 95 percent paired image bootstrap."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("credit vectors are not paired")
    difference = left - right
    index = np.random.default_rng(SEED).integers(0, len(difference), size=(RESAMPLES, len(difference)))
    low, high = np.percentile(difference[index].mean(axis=1), [2.5, 97.5])
    return {"left_percent": 100 * float(left.mean()), "right_percent": 100 * float(right.mean()),
            "difference_pp": 100 * float(difference.mean()),
            "pointwise_95_interval_pp": [100 * float(low), 100 * float(high)]}


def batch_row(value):
    """One batch size of a batch-sensitivity record, with the paired difference under its true direction.

    The records name the field ``paired_top1_sealed_batch8_minus_this``, but ``compare_records(sealed, fresh)``
    returns right minus left, so the stored value is this run minus the sealed batch-8 run.  The summary uses
    the correct name; the sealed records are left as they are.
    """
    row = {k: v for k, v in value.items() if k not in ("ties", "paired_top1_sealed_batch8_minus_this")}
    row["paired_top1_this_minus_sealed_batch8"] = value["paired_top1_sealed_batch8_minus_this"]
    row["tied_top1_images"] = value["ties"]["images_with_tied_top1"]
    row["top1_percent_expected"] = value["ties"]["top1_percent_expected"]
    return row


def one(folder, pattern):
    matches = sorted((BASE / folder).glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one record for {folder}/{pattern}, found {len(matches)}")
    return unseal(matches[0]), str(matches[0].relative_to(ROOT))


def write(path, text):
    if path.exists():
        if path.read_text() == text:
            return "unchanged"
        raise SystemExit(f"{path.relative_to(ROOT)} exists with other content; summaries are not overwritten")
    path.write_text(text)
    return "written"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="r3c")
    args = parser.parse_args()
    tag = args.tag
    rows, credits, batch, sources, hashes = [], {}, {}, [], set()
    configurations = [(model, "int8", recipe) for model in MODELS for recipe in ("v1_maxabs", "v1_percentile_99_9")]
    configurations += [(model, name, "default") for model in MODELS for name in FORMATS]
    for model, name, recipe in configurations:
        record, source = one("batch-sensitivity", f"{model}--{name}--{recipe}--*--{tag}.json")
        if not record["sealed_batch8_predictions_reproduced"]:
            raise SystemExit(f"{source}: sealed batch-8 predictions were not reproduced")
        sources.append(source)
        hashes.add(record["verify_source_sha256"])
        b2, fp32 = record["results"]["b2"]["batch8"], record["results"]["fp32"]["batch8"]
        if fp32["ties"]["images_with_tied_top1"]:
            raise SystemExit("FP32 logits tie; the FP32 reference would need a tie rule too")
        ties = b2["ties"]
        credits[(model, name, recipe)] = (ties["per_image"], record["rows_sha256"])
        credits[(model, "fp32", "fp32")] = (fp32["ties"]["per_image"], record["rows_sha256"])
        rows.append({"model": model, "format": name, "recipe": recipe, "fp32_top1": fp32["top1_percent"],
                     "sealed_top1": b2["top1_percent"], "tied_top1_images": ties["images_with_tied_top1"],
                     "tied_images_involving_label": ties["images_with_tied_top1_involving_the_label"],
                     "largest_tie": ties["largest_tie"],
                     **{f"top1_{rule}": ties[f"top1_percent_{rule}"] for rule in RULES},
                     "median_distinct_logit_values": ties["distinct_logit_values_median_per_image"],
                     "gpu_seconds": record["cost"]["gpu_seconds"], "record": source})
        if name == "int8" and recipe == "default":
            batch[model] = {"record": source, "configuration_sha256": record["configuration_sha256"],
                            **{runner: {key: batch_row(value) for key, value in record["results"][runner].items()}
                               for runner in ("b2", "fp32")}}
    vendor = {}
    for model in MODELS:
        arms = ["x86_default", "x86_fullrange", "qnnpack_fullrange"]
        if model == "mobilenet_v2":
            arms += ["x86_minmax_fullrange", "qnnpack_minmax_fullrange", "qnnpack_fullrange_relu6fix"]
        for arm in arms:
            record, source = one("vendor-probe", f"{model}--{arm}--2000--{tag}.json")
            sources.append(source)
            hashes.add(record["verify_source_sha256"])
            ties = record["ties"]
            credits[(model, "vendor", arm)] = (ties["per_image"], record["screen_rows_sha256"])
            audit = record["kernel_audit_first_8_screen_images"]
            vendor[f"{model}/{arm}"] = {"record": source, "top1_as_run": record["metrics"]["top1_percent"],
                                        "prediction_digest": record["prediction_digest"],
                                        "kernel_audit": {"audited": audit["audited_modules"],
                                                         "wrong": audit["modules_with_code_error_above_1"],
                                                         "by_kind": audit["by_kind"],
                                                         "largest_code_error": audit["largest_code_error"]},
                                        "cpu_seconds": record["cost"]["job_wall_seconds"]}
            rows.append({"model": model, "format": "int8", "recipe": f"vendor_{arm}",
                         "fp32_top1": record["metrics"]["fp32_top1_percent"],
                         "sealed_top1": record["metrics"]["top1_percent"],
                         "tied_top1_images": ties["images_with_tied_top1"],
                         "tied_images_involving_label": ties["images_with_tied_top1_involving_the_label"],
                         "largest_tie": ties["largest_tie"],
                         **{f"top1_{rule}": ties[f"top1_percent_{rule}"] for rule in RULES},
                         "median_distinct_logit_values": ties["distinct_logit_values_median_per_image"],
                         "gpu_seconds": 0.0, "record": source})

    def pair(left, right, rule):
        (a, rows_a), (b, rows_b) = credits[left], credits[right]
        if rows_a != rows_b:
            raise SystemExit("records are not on the same ordered screen rows")
        return paired(credit(a, rule), credit(b, rule))

    exit_pairs = {}
    for model in MODELS:
        best = {"resnet18": "qnnpack_fullrange", "mobilenet_v2": "qnnpack_fullrange_relu6fix",
                "mobilenet_v3_large": "qnnpack_fullrange"}[model]
        exit_pairs[model] = {
            rule: {"b2_default_minus_fp32": pair((model, "int8", "default"), (model, "fp32", "fp32"), rule),
                   "b2_default_minus_vendor_x86_default": pair((model, "int8", "default"),
                                                               (model, "vendor", "x86_default"), rule),
                   f"b2_default_minus_vendor_{best}": pair((model, "int8", "default"), (model, "vendor", best), rule),
                   "b2_default_minus_v1_maxabs": pair((model, "int8", "default"), (model, "int8", "v1_maxabs"), rule)}
            for rule in ("expected", "lowest_index")}
    if len(hashes) != 1:
        raise SystemExit(f"records were made with {len(hashes)} versions of verify.py")
    header = {"tag": tag, "verify_source_sha256": sorted(hashes)[0], "evidence": "development_evidence",
              "interval": f"pointwise 95 percent paired image bootstrap of the mean credit difference, {RESAMPLES} "
                          f"resamples, seed {SEED}; no multiplicity correction",
              "rules": {"sealed": "class returned first by topk in the sealed run (implementation-defined on ties)",
                        "strict": "label must be the unique maximum", "expected": "credit 1/k for a k-way tie",
                        "lowest_index": "the smallest class index among the maxima wins",
                        "optimistic": "label among the maxima"}}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    gpu = sum(row["gpu_seconds"] for row in rows)
    status = {
        "logit-ties-1k.csv": write(OUT / "logit-ties-1k.csv", buffer.getvalue()),
        "logit-ties-1k.json": write(OUT / "logit-ties-1k.json", json.dumps(
            {**header, "configurations": rows, "exit_test_under_tie_rules": exit_pairs, "vendor_probes": vendor,
             "cost": {"gpu_seconds_batch_sensitivity_and_tie_jobs": gpu,
                      "cpu_seconds_vendor_probes": sum(v["cpu_seconds"] for v in vendor.values())},
             "records": sources}, indent=2, sort_keys=True) + "\n"),
        "batch-sensitivity-1k.json": write(OUT / "batch-sensitivity-1k.json", json.dumps(
            {**header, "format": "int8", "recipe": "default",
             "note": "paired_top1_this_minus_sealed_batch8 is the record field paired_top1_sealed_batch8_minus_this, "
                     "whose name states the wrong direction; batch-sensitivity.json (first export, same numbers, "
                     "wrong field name) is superseded by this file",
             "models": batch}, indent=2, sort_keys=True) + "\n")}
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
