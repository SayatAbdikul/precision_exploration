"""Paired analysis of the B2 baseline-repair study (reads sealed runs, writes new summaries only).

Usage: .venv-b/bin/python -m tools.analysis.b2_report {diagnosis,exit,sentinel,cost,all}
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from tools.analysis.b_stage_balanced_comparisons import compare_records
from tools.experiment_b.common import ROOT, dataset, seal, unseal
from tools.experiment_b.validation import verify_prediction
from tools.experiment_b2.common import BASE, MODELS, source_identity
from tools.experiment_b2.recipe import CUMULATIVE, ONE_AT_A_TIME

OUT = ROOT / "results/summaries/b2-baseline-repair-v1"
V1 = ROOT / "artifacts/experiment_b"
LADDER = ("v1_maxabs", "cum1_fused", "cum2_unsigned", "cum3_act_mse", "cum4_weight_mse", "cum5_bias_correction")
SENTINELS = ("int8", "int6", "int4", "fp8_e4m3fn", "fp7_e3m3", "fp6_e2m3", "log8", "posit8_es1", "nf4")


def rows_1k():
    return dataset("imagenet_screen_1k")[1]


def run_record(stage, model, format_name, recipe, images, *, any_source=False):
    source = source_identity()
    found = []
    for path in sorted((BASE / "runs" / stage).glob(f"{model}--{format_name}--{recipe}--{images}--*.json")):
        record = unseal(path)
        if any_source or record.get("source_sha256") == source:
            found.append(record)
    if len(found) > 1:
        raise ValueError(f"ambiguous run records for {stage}/{model}/{format_name}/{recipe}/{images}")
    return found[0] if found else None


def predictions(identity, rows, root=BASE):
    return [verify_prediction(root / "predictions" / identity / (row["sha256"] + ".json"), identity, row) for row in rows]


def brief(comparison):
    top1 = comparison["top1"]
    return {"left_percent": top1["left_percent"], "right_percent": top1["right_percent"],
            "difference_pp": top1["difference_pp"], "interval_pp": top1["pointwise_95_interval_pp"],
            "mcnemar_exact_p": top1["mcnemar_exact_p"], "top5_difference_pp": comparison["top5"]["difference_pp"],
            "top5_right_percent": comparison["top5"]["right_percent"],
            "changed_top1_images": comparison["changed_top1_images"]}


def compare(left, right, select=slice(None)):
    return brief(compare_records(left[select], right[select]))


def diagnosis(images=512):
    rows = rows_1k()[:images]
    panels = {"dev128": slice(0, 128), f"dev{images}": slice(0, images)} if images > 128 else {"dev128": slice(0, 128)}
    document = {"panel_images": images, "evidence": "development (recipe selection panel)", "models": {}}
    table = []
    for model in MODELS:
        reference = run_record("diagnosis", model, "int8", "v1_maxabs", images)
        if reference is None:
            continue
        fp32 = predictions(reference["baseline_sha256"], rows)
        base = predictions(reference["configuration_sha256"], rows)
        arms, loaded = {}, {"v1_maxabs": base}
        for name in ("v1_maxabs", "v1_percentile_99_9", *ONE_AT_A_TIME, *CUMULATIVE):
            record = run_record("diagnosis", model, "int8", name, images)
            if record is None:
                continue
            loaded[name] = predictions(record["configuration_sha256"], rows)
            entry = {"recipe": record["recipe"], "configuration_sha256": record["configuration_sha256"],
                     "cost": record["cost"], "panels": {}}
            for label, select in panels.items():
                entry["panels"][label] = {"versus_fp32": compare(fp32, loaded[name], select),
                                          "versus_v1_maxabs": compare(base, loaded[name], select)}
            arms[name] = entry
        steps = {}
        for before, after in zip(LADDER, LADDER[1:]):
            if before in loaded and after in loaded:
                steps[f"{before}->{after}"] = {label: compare(loaded[before], loaded[after], select)
                                               for label, select in panels.items()}
        leave_one_out = {}
        if "cum5_bias_correction" in loaded:
            for name in ("cum5_signed", "cum5_weight_maxabs", "cum5_act_maxabs", "cum5_v1_boundaries", "cum4_weight_mse",
                         "cum5_fused_all", "cum5_no_logits", "cum5_no_input_no_logits", "cum5_cle"):
                if name in loaded:
                    leave_one_out[f"cum5_bias_correction->{name}"] = {
                        label: compare(loaded["cum5_bias_correction"], loaded[name], select) for label, select in panels.items()}
        document["models"][model] = {"arms": arms, "cumulative_steps": steps, "around_full_ladder": leave_one_out}
        for name, entry in arms.items():
            for label in panels:
                a, b = entry["panels"][label]["versus_fp32"], entry["panels"][label]["versus_v1_maxabs"]
                table.append({"model": model, "arm": name, "panel": label, "top1": a["right_percent"],
                              "top5": a["top5_right_percent"], "fp32_top1": a["left_percent"],
                              "delta_vs_fp32_pp": a["difference_pp"], "fp32_lo": a["interval_pp"][0],
                              "fp32_hi": a["interval_pp"][1], "delta_vs_v1_maxabs_pp": b["difference_pp"],
                              "v1_lo": b["interval_pp"][0], "v1_hi": b["interval_pp"][1],
                              "mcnemar_p_vs_v1": b["mcnemar_exact_p"]})
    OUT.mkdir(parents=True, exist_ok=True)
    seal(OUT / f"diagnosis-{images}.json", document)
    with (OUT / f"diagnosis-{images}.csv").open("w", newline="") as stream:
        if table:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return document


def exit_test():
    from tools.experiment_b2 import frozen
    rows = rows_1k()
    panels = {"screen1k": slice(0, 1000), "dev512": slice(0, 512), "heldout488": slice(512, 1000)}
    document = {"evidence": "development: screen1k contains the dev512 selection panel; heldout488 was not used for selection",
                "target": "B2 default within about 1.0 pp of FP32, or on par with the vendor anchor", "models": {}}
    for model in MODELS:
        default = run_record("exit", model, "int8", "default", 1000)
        if default is None:
            continue
        fp32 = predictions(default["baseline_sha256"], rows)
        loaded = {"default": predictions(default["configuration_sha256"], rows)}
        for name in [n for n in frozen.FROZEN if n != "default"] + ["v1_maxabs", "v1_percentile_99_9"]:
            record = run_record("exit", model, "int8", name, 1000)
            if record is not None:
                loaded[name] = predictions(record["configuration_sha256"], rows)
        vendor = {}
        for qconfig in ("x86_default", "x86_fullrange", "qnnpack_fullrange"):
            record = run_record("vendor", model, "int8", f"vendor_{qconfig}", 1000, any_source=True)
            if record is not None:
                vendor[qconfig] = {"record": record, "predictions": predictions(record["configuration_sha256"], rows),
                                   "cpu_fp32": predictions(record["baseline_sha256"], rows)}
        entry = {"configuration_sha256": default["configuration_sha256"], "fp32_baseline_sha256": default["baseline_sha256"],
                 "panels": {}}
        for label, select in panels.items():
            panel = {"fp32_vs_" + name: compare(fp32, values, select) for name, values in loaded.items()}
            for qconfig, item in vendor.items():
                panel[f"fp32_vs_vendor_{qconfig}"] = compare(fp32, item["predictions"], select)
                panel[f"vendor_{qconfig}_vs_default"] = compare(item["predictions"], loaded["default"], select)
                panel[f"gpu_fp32_vs_cpu_unfolded_fp32_{qconfig}"] = compare(fp32, item["cpu_fp32"], select)
            for name in loaded:
                if name != "default":
                    panel[f"{name}_vs_default"] = compare(loaded[name], loaded["default"], select)
            entry["panels"][label] = panel
        entry["vendor_configurations"] = {q: {"configuration_sha256": item["record"]["configuration_sha256"],
                                              "calibration_images": item["record"]["calibration_images"]}
                                          for q, item in vendor.items()}
        document["models"][model] = entry
    seal(OUT / "exit-test.json", document)
    return document


def v1_summary(model, format_name, recipe):
    with (ROOT / "results/summaries/b-stage-paired-1k-v2/configurations.csv").open() as stream:
        for row in csv.DictReader(stream):
            if (row["model"], row["format"], row["recipe"], row["metric"], row["images"]) == (model, format_name, recipe, "top1", "1000"):
                return row
    return None


def sentinel():
    rows = rows_1k()
    document = {"evidence": "development (screen1k contains dev512)", "recipe": "default", "models": {}}
    table = []
    for model in MODELS:
        entries = {}
        for name in SENTINELS:
            record = run_record("sentinel", model, name, "default", 1000)
            if record is None:
                continue
            fp32 = predictions(record["baseline_sha256"], rows)
            new = predictions(record["configuration_sha256"], rows)
            entry = {"configuration_sha256": record["configuration_sha256"], "versus_fp32": compare(fp32, new),
                     "versus_fp32_heldout488": compare(fp32, new, slice(512, 1000)), "cost": record["cost"], "v1": {}}
            for recipe in ("maxabs", "percentile_99_9"):
                old = v1_summary(model, name, recipe)
                if old is None:
                    continue
                old_predictions = predictions(old["configuration_sha256"], rows, root=V1)
                entry["v1"][recipe] = {"configuration_sha256": old["configuration_sha256"],
                                       "top1_percent": float(old["candidate_percent"]),
                                       "v1_vs_b2_default": compare(old_predictions, new)}
            entries[name] = entry
            table.append({"model": model, "format": name, "b2_default_top1": entry["versus_fp32"]["right_percent"],
                          "b2_default_top5": entry["versus_fp32"]["top5_right_percent"],
                          "fp32_top1": entry["versus_fp32"]["left_percent"],
                          "delta_vs_fp32_pp": entry["versus_fp32"]["difference_pp"],
                          "lo": entry["versus_fp32"]["interval_pp"][0], "hi": entry["versus_fp32"]["interval_pp"][1],
                          "v1_maxabs_top1": entry["v1"].get("maxabs", {}).get("top1_percent"),
                          "v1_percentile_top1": entry["v1"].get("percentile_99_9", {}).get("top1_percent")})
        if entries:
            ordered = sorted(entries, key=lambda k: -entries[k]["versus_fp32"]["right_percent"])
            v1_best = {k: max((v["top1_percent"] for v in entries[k]["v1"].values()), default=None) for k in entries}
            document["models"][model] = {"formats": entries, "b2_default_order": ordered,
                                         "v1_best_recipe_order": sorted((k for k in v1_best if v1_best[k] is not None),
                                                                        key=lambda k: -v1_best[k])}
            pairs = {}
            for name in entries:
                if name != "int8" and "int8" in entries:
                    a = predictions(entries["int8"]["configuration_sha256"], rows)
                    b = predictions(entries[name]["configuration_sha256"], rows)
                    pairs[f"int8_vs_{name}"] = compare(a, b)
            document["models"][model]["versus_int8_default"] = pairs
    seal(OUT / "sentinel-1k.json", document)
    with (OUT / "sentinel-1k.csv").open("w", newline="") as stream:
        if table:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return document


def lowbit():
    """Ablation arms at low bit widths and for non-integer formats (screen1k), paired against the default."""
    rows = rows_1k()
    document = {"evidence": "development (screen1k contains dev512)", "models": {}}
    table = []
    for model in MODELS:
        entries = {}
        for name in ("int6", "int4", "fp6_e2m3", "posit8_es1"):
            default = run_record("sentinel", model, name, "default", 1000)
            if default is None:
                continue
            fp32 = predictions(default["baseline_sha256"], rows)
            base = predictions(default["configuration_sha256"], rows)
            entry = {"default": compare(fp32, base)}
            for arm in ("default_signed", "default_no_bias_correction", "default_weight_maxabs", "minimal"):
                record = run_record("lowbit", model, name, arm, 1000)
                if record is None:
                    continue
                values = predictions(record["configuration_sha256"], rows)
                entry[arm] = {"versus_fp32": compare(fp32, values), "arm_vs_default": compare(values, base)}
                table.append({"model": model, "format": name, "arm": arm,
                              "arm_top1": entry[arm]["versus_fp32"]["right_percent"],
                              "default_top1": entry["default"]["right_percent"],
                              "default_minus_arm_pp": entry[arm]["arm_vs_default"]["difference_pp"],
                              "lo": entry[arm]["arm_vs_default"]["interval_pp"][0],
                              "hi": entry[arm]["arm_vs_default"]["interval_pp"][1]})
            entries[name] = entry
        if entries:
            document["models"][model] = entries
    seal(OUT / "lowbit-ablation-1k.json", document)
    with (OUT / "lowbit-ablation-1k.csv").open("w", newline="") as stream:
        if table:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return table


def occupancy():
    document = {}
    for path in sorted((BASE / "occupancy").glob("*.json")):
        record = unseal(path)
        nodes = record["nodes"]
        def stat(key):
            values = [row[key] for row in nodes.values() if row[key] is not None]
            return {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
        document[f"{record['model']}/{record['format']}/{record['recipe_name']}"] = {
            "configuration_sha256": record["configuration_sha256"], "images": record["images"],
            "quantizing_nodes": len(nodes), "entropy_bits": stat("entropy_bits"), "sqnr_db": stat("sqnr_db"),
            "levels_for_99_percent": stat("levels_for_99_percent"), "fraction_zero": stat("fraction_zero"),
            "fraction_abs_level_ge_1": stat("fraction_abs_level_ge_1"),
            "fraction_at_extremes": stat("fraction_at_extremes"),
            "worst_sqnr_nodes": sorted(((row["sqnr_db"], name) for name, row in nodes.items() if row["sqnr_db"] is not None))[:3]}
    seal(OUT / "occupancy.json", document)
    return document


def cost():
    total = {}
    for stage_dir in sorted((BASE / "runs").iterdir()):
        records = [unseal(p) for p in stage_dir.glob("*.json")]
        total[stage_dir.name] = {"runs": len(records),
                                 "job_wall_seconds": sum(r["cost"]["job_wall_seconds"] for r in records),
                                 "gpu_seconds": sum(r["cost"]["gpu_seconds"] for r in records)}
    regress = [unseal(p) for p in (BASE / "regression").glob("*.json")]
    total["regression"] = {"runs": len(regress), "job_wall_seconds": sum(r["wall_seconds"] for r in regress),
                           "gpu_seconds": sum(r["wall_seconds"] for r in regress)}
    audits = [unseal(p) for p in (BASE / "occupancy").glob("*.json")] if (BASE / "occupancy").exists() else []
    total["audit"] = {"runs": len(audits), "job_wall_seconds": sum(r["wall_seconds"] for r in audits),
                      "gpu_seconds": sum(r["wall_seconds"] for r in audits)}
    total["all"] = {k: sum(v[k] for v in total.values()) for k in ("runs", "job_wall_seconds", "gpu_seconds")}
    seal(OUT / "cost.json", total)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=("diagnosis", "exit", "sentinel", "lowbit", "occupancy", "cost", "all"))
    parser.add_argument("--images", type=int, default=512)
    args = parser.parse_args()
    from tools.experiment_b2 import frozen  # noqa: F401
    if args.what in ("diagnosis", "all"):
        document = diagnosis(args.images)
        for model, body in document["models"].items():
            for name, entry in body["arms"].items():
                panel = entry["panels"][f"dev{args.images}"]
                a, b = panel["versus_fp32"], panel["versus_v1_maxabs"]
                print(f"{model:20s} {name:28s} top1 {a['right_percent']:6.2f}  vsFP32 {a['difference_pp']:+6.2f} "
                      f"[{a['interval_pp'][0]:+.2f},{a['interval_pp'][1]:+.2f}]  vs v1 {b['difference_pp']:+6.2f} "
                      f"[{b['interval_pp'][0]:+.2f},{b['interval_pp'][1]:+.2f}]")
    if args.what in ("exit", "all"):
        document = exit_test()
        for model, body in document["models"].items():
            for label, panel in body["panels"].items():
                for key, value in panel.items():
                    print(f"{model:20s} {label:11s} {key:42s} {value['left_percent']:6.2f} -> {value['right_percent']:6.2f} "
                          f"diff {value['difference_pp']:+6.2f} [{value['interval_pp'][0]:+.2f},{value['interval_pp'][1]:+.2f}]")
    if args.what in ("sentinel", "all"):
        document = sentinel()
        for model, body in document["models"].items():
            for name, entry in body["formats"].items():
                a = entry["versus_fp32"]
                old = {k: v["top1_percent"] for k, v in entry["v1"].items()}
                print(f"{model:20s} {name:12s} top1 {a['right_percent']:6.2f} vsFP32 {a['difference_pp']:+6.2f} "
                      f"[{a['interval_pp'][0]:+.2f},{a['interval_pp'][1]:+.2f}] v1 {old}")
    if args.what in ("lowbit", "all"):
        for row in lowbit():
            print(f"{row['model']:20s} {row['format']:11s} {row['arm']:27s} arm {row['arm_top1']:6.2f} default "
                  f"{row['default_top1']:6.2f}  default-arm {row['default_minus_arm_pp']:+6.2f} [{row['lo']:+.2f},{row['hi']:+.2f}]")
    if args.what in ("occupancy", "all") and (BASE / "occupancy").exists():
        for key, row in occupancy().items():
            print(f"{key:48s} nodes {row['quantizing_nodes']:3d} entropy {row['entropy_bits']['median']:5.2f} "
                  f"sqnr median {row['sqnr_db']['median']:6.2f} min {row['sqnr_db']['min']:6.2f} "
                  f"levels99 {row['levels_for_99_percent']['median']:5.0f} |level|>=1 {row['fraction_abs_level_ge_1']['median']:.3f}")
    if args.what in ("cost", "all"):
        print(json.dumps(cost(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
