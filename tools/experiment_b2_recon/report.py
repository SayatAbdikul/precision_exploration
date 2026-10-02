"""Summaries of the B2 reconstruction baseline (reads evaluation and fit records only; no model, no GPU).

Writes ``results/summaries/b2-recon-v1/<name>--<tag>.{json,csv}`` and, with
``--figure``, ``results/figures/b2-recon-matrix--<tag>.{pdf,png}``.  A file is
never replaced: if it exists with other content the script stops (use a new
``--tag``).  Everything is development evidence on the 1k screen.

Credit per image: ``expected`` = 1/k when the label is among the k classes
tied for the maximum, ``lowest_index`` = the label is the lowest-index
maximum, ``topk`` = what ``topk`` returned on the device.  Differences are
left minus right with the pointwise 95 percent paired image bootstrap of
``tools.analysis.b2_ties.paired`` (10000 resamples, seed 20260927); no
multiplicity correction.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re

import numpy as np

from tools.analysis.b2_ties import paired
from tools.experiment_b.common import ROOT, dataset, digest, unseal

from .fit import BASE, read_layer

OUT = ROOT / "results/summaries/b2-recon-v1"
FIGURES = ROOT / "results/figures"
PROTOCOL_PATH = "public/experiments/configs/breadth-study/b2-recon-protocol-v1.json"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
RULES = ("expected", "lowest_index", "topk")
# learned arm -> the nearest arm that differs from it in the rounding only
NEAREST_OF = {"L-nobc": "N-nobc", "L-bc": "N-default", "L-A32": "N-A32", "LQ-nobc": "N-nobc", "LQ-bc": "N-default",
              "L-minimal": "N-minimal"}
PUBLISHED = {"fp32": 69.68, "w4_a32_nearest": 23.99, "w4_a32_bias_correction": 38.87, "w4_a32_adaround": 68.60,
             "w4_a32_adaround_sd": 0.09, "w4_a8_adaround": 68.55, "w4_a8_adaround_sd": 0.01}


def credit(arrays, labels, rule):
    if rule == "expected":
        return arrays["label_among_maxima"].astype(np.float64) / arrays["tie_size"].astype(np.float64)
    if rule == "lowest_index":
        return arrays["label_is_lowest_index_maximum"].astype(np.float64)
    if rule == "topk":
        return (arrays["top5_topk"][:, 0].astype(np.int64) == labels).astype(np.float64)
    raise ValueError(rule)


def screen():
    _, rows, _ = dataset("imagenet_screen_1k")
    return rows, np.array([int(row["label"]) for row in rows])


def load_model(model, rows):
    arms = {}
    for path in sorted((BASE / "evals" / model).glob("*.json")):
        record = unseal(path)
        if record["rows_sha256"] != digest(rows):
            raise ValueError(f"{path} was evaluated on another list")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            arms[record["label"]] = (record, {key: saved[key] for key in saved.files})
    return arms


def brief(result):
    return {"difference_pp": round(result["difference_pp"], 3),
            "interval_pp": [round(x, 3) for x in result["pointwise_95_interval_pp"]]}


def nearest_counterpart(name, record):
    """Label of the nearest arm that differs from a learned arm in the rounding only (``None`` for nearest arms)."""
    if record["rounding"] != "learned":
        return None
    base = re.sub(r"--fit-.*$", "", name)
    head, rest = base.split("--", 1)
    return f"{NEAREST_OF[head]}--{rest}"


def default_reference(record):
    if record["activation_format"] is None or record["weight_scale_rule"] == "mse_per_layer":  # faithfulness arms
        return None
    return "--".join(["N-default", f"w-{record['weight_format']}", f"a-{record['activation_format']}", "default"])


def compare(arms, labels, own_name, other_name):
    if other_name is None or other_name not in arms or other_name == own_name:
        return None
    return {rule: brief(paired(credit(arms[own_name][1], labels, rule), credit(arms[other_name][1], labels, rule)))
            for rule in RULES}


def fit_seed(record):
    return None if not record["fit"] else int(record["fit"]["folder"].split("--s")[1].split("--")[0])


def matrix(all_arms, labels):
    table = []
    for model, arms in all_arms.items():
        for name, (record, arrays) in arms.items():
            row = {"model": model, "label": name, "arm": "fp32" if name == "fp32" else record["arm"],
                   **{f"top1_{rule}": round(100 * float(credit(arrays, labels, rule).mean()), 3) for rule in RULES}}
            if name != "fp32":
                row.update({
                    "weight_format": record["weight_format"], "activation_format": record["activation_format"] or "fp32",
                    "recipe": record["recipe_name"], "weight_scale_rule": record["weight_scale_rule"],
                    "bias_correction": record["bias_correction"], "rounding": record["rounding"],
                    "fit_mode": None if not record["fit"] else record["fit"]["folder"].split("--")[3],
                    "fit_seed": fit_seed(record), "tied_images": record["readout"]["images_with_tied_top1"],
                    "weights_differing_from_nearest": record["weights_differing_from_nearest"],
                    "sealed_b2_top5_reproduced": (record["sealed_b2_check"] or {}).get("top5_lists_reproduced"),
                    "configuration_sha256": record["configuration_sha256"],
                    "evaluation_seconds": round(record["cost"]["seconds"], 1),
                    "minus_fp32": compare(arms, labels, name, "fp32"),
                    "default_reference": default_reference(record),
                    "minus_default": compare(arms, labels, name, default_reference(record)),
                    "nearest_counterpart": nearest_counterpart(name, record),
                    "minus_nearest_counterpart": compare(arms, labels, name, nearest_counterpart(name, record))})
            table.append(row)
    return table


def six_bit(all_arms, labels):
    """INT6 minus fp6_e2m3, per model and arm (same recipe and rounding on both sides)."""
    result = []
    for model, arms in all_arms.items():
        for name, (record, _) in arms.items():
            if name == "fp32" or record["weight_format"] != "int6" or record["activation_format"] not in ("int6", None):
                continue
            other = name.replace("w-int6", "w-fp6_e2m3").replace("a-int6", "a-fp6_e2m3")
            if other in arms:
                result.append({"model": model, "arm": record["arm"], "int6": name, "fp6_e2m3": other,
                               "int6_top1_expected": round(100 * float(credit(arms[name][1], labels, "expected").mean()), 3),
                               "fp6_top1_expected": round(100 * float(credit(arms[other][1], labels, "expected").mean()), 3),
                               "int6_minus_fp6": compare(arms, labels, name, other)})
    return result


def faithfulness(all_arms, labels):
    arms = all_arms["resnet18"]
    fp32 = arms["fp32"]
    def top1(name):
        return round(100 * float(credit(arms[name][1], labels, "expected").mean()), 3)

    def drop(name):  # FP32 minus arm (positive = loss), expected credit (no ties with FP32 activations)
        return brief(paired(credit(fp32[1], labels, "expected"), credit(arms[name][1], labels, "expected")))

    learned_a32 = {s: f"L-A32--w-int4--a-fp32--mse_per_layer--fit-mse_per_layer-fp32in-s{s}" for s in (0, 1, 2)}
    learned_a8 = {s: f"L-nobc--w-int4--a-int8--default_no_bias_correction--mse_per_layer--fit-mse_per_layer-fp32in-s{s}"
                  for s in (0, 1, 2)}
    nearest_a32 = "N-A32--w-int4--a-fp32--mse_per_layer"
    bc_a32 = "N-A32-bc--w-int4--a-fp32--mse_per_layer--bias-empirical"
    nearest_a8 = "N-nobc--w-int4--a-int8--default_no_bias_correction--mse_per_layer"
    published_drop = round(PUBLISHED["fp32"] - PUBLISHED["w4_a32_adaround"], 2)
    seeds = {s: {"top1": top1(n), "drop_vs_fp32": drop(n),
                 "weights_differing_from_nearest": arms[n][0]["weights_differing_from_nearest"]}
             for s, n in learned_a32.items()}
    first = seeds[0]["drop_vs_fp32"]
    spread = max(v["top1"] for v in seeds.values()) - min(v["top1"] for v in seeds.values())
    inside = first["interval_pp"][0] <= published_drop <= first["interval_pp"][1]
    w4a8 = {s: {"top1_expected": top1(n), "top1_lowest_index": round(100 * float(credit(arms[n][1], labels, "lowest_index").mean()), 3),
                "tied_images": arms[n][0]["readout"]["images_with_tied_top1"], "drop_vs_fp32": drop(n)}
            for s, n in learned_a8.items()}
    return {
        "setting": "ResNet18, INT4 weights (-8..7), one MSE scale per layer, all layers, 1024 calibration images, "
                   "10000 iterations, batch 32; FP32 activations (and INT8 with B2 activation handling)",
        "fp32_screen": top1("fp32"), "published": PUBLISHED,
        "published_drop_pp": published_drop,
        "adaround_w4_a32_seeds": seeds,
        "adaround_w4_a32_mean_top1": round(float(np.mean([v["top1"] for v in seeds.values()])), 3),
        "adaround_w4_a32_seed_spread_pp": round(spread, 3),
        "criterion": "published drop inside the seed-0 paired interval of the measured drop, and the three seeds within 1 point",
        "criterion_published_drop_inside_seed0_interval": inside,
        "criterion_seeds_within_1pp": spread <= 1.0,
        "verdict": "consistent" if inside and spread <= 1.0 else "not consistent",
        "nearest_w4_a32": {"top1": top1(nearest_a32), "drop_vs_fp32": drop(nearest_a32),
                           "published_top1": PUBLISHED["w4_a32_nearest"],
                           "published_drop_pp": round(PUBLISHED["fp32"] - PUBLISHED["w4_a32_nearest"], 2)},
        "bias_correction_w4_a32": {"top1": top1(bc_a32), "drop_vs_fp32": drop(bc_a32),
                                   "published_top1": PUBLISHED["w4_a32_bias_correction"],
                                   "distinct_top1_classes": int(len(set(arms[bc_a32][1]["top5_topk"][:, 0].tolist())))},
        "nearest_w4_a8": {"top1_expected": top1(nearest_a8), "drop_vs_fp32": drop(nearest_a8)},
        "adaround_w4_a8_seeds": w4a8,
        "adaround_w4_a8_published": {"top1": PUBLISHED["w4_a8_adaround"],
                                     "drop_pp": round(PUBLISHED["fp32"] - PUBLISHED["w4_a8_adaround"], 2)},
        "known_differences": unseal_protocol()["faithfulness_check"]["known_differences"]}


def unseal_protocol():
    return json.loads((ROOT / PROTOCOL_PATH).read_text())


def questions(all_arms, labels):
    """The three questions, decided by the criteria written in the protocol before measuring."""
    def row(model, name):
        arms = all_arms[model]
        if name not in arms:
            return None
        return {"label": name, "top1_expected": round(100 * float(credit(arms[name][1], labels, "expected").mean()), 3),
                "top1_lowest_index": round(100 * float(credit(arms[name][1], labels, "lowest_index").mean()), 3),
                "minus_fp32": compare(arms, labels, name, "fp32"),
                "minus_default": compare(arms, labels, name, default_reference(arms[name][0]))}

    fit = "fit-mse_per_channel-fp32in-s0"
    def learned(wf, af, which):
        recipe = "default" if which == "L-bc" else "default_no_bias_correction"
        return f"{which}--w-{wf}--a-{af}--{recipe}--{fit}"

    # 1. MobileNetV3-Large INT8 gap
    gap = {name: row("mobilenet_v3_large", name) for name in
           ("N-default--w-int8--a-int8--default", learned("int8", "int8", "L-nobc"), learned("int8", "int8", "L-bc"),
            "N-A32--w-int8--a-fp32--mse_per_channel", f"L-A32--w-int8--a-fp32--mse_per_channel--{fit}")}
    closed = []
    for which in ("L-nobc", "L-bc"):
        r = gap[learned("int8", "int8", which)]
        ok = (-r["minus_fp32"]["expected"]["difference_pp"] <= 1.0 and
              (r["minus_default"]["expected"]["interval_pp"][0] > 0 or r["minus_default"]["expected"]["interval_pp"][1] < 0))
        closed.append({"arm": which, "closed": bool(ok)})
    bc_vs_nobc = compare(all_arms["mobilenet_v3_large"], labels, learned("int8", "int8", "L-bc"), learned("int8", "int8", "L-nobc"))
    # 2. 4-bit rescue
    rescue = {}
    for model in MODELS:
        cells = {}
        for name in ("N-default", "N-nobc", "N-minimal"):
            for af in ("int4", "int8"):
                recipe = {"N-default": "default", "N-nobc": "default_no_bias_correction", "N-minimal": "minimal"}[name]
                cells[f"{name} W4/A{af[3:]}"] = row(model, f"{name}--w-int4--a-{af}--{recipe}")
        for which in ("L-nobc", "L-bc"):
            for af in ("int4", "int8"):
                cells[f"{which} W4/A{af[3:]}"] = row(model, learned("int4", af, which))
        for which, recipe in (("LQ-nobc", "default_no_bias_correction"), ("LQ-bc", "default")):
            cells[f"{which} W4/A4 (adaptation: activation-aware input)"] = row(
                model, f"{which}--w-int4--a-int4--{recipe}--fit-mse_per_channel-b2in-int4-s0")
        cells["N-A32 W4/A32"] = row(model, "N-A32--w-int4--a-fp32--mse_per_channel")
        cells["L-A32 W4/A32"] = row(model, f"L-A32--w-int4--a-fp32--mse_per_channel--{fit}")
        best_w4a4 = max((v["top1_expected"], k) for k, v in cells.items()
                        if v and k.startswith(("L-", "LQ-")) and "A4" in k and "A32" not in k)
        cells = {k: v for k, v in cells.items() if v}
        fp = 100 * float(credit(all_arms[model]["fp32"][1], labels, "expected").mean())
        rescue[model] = {"cells": cells, "best_learned_w4a4": {"arm": best_w4a4[1], "top1_expected": best_w4a4[0]},
                         "fp32": round(fp, 3), "w4a4_rescued": bool(best_w4a4[0] >= fp - 5)}
    # 3. six-bit order: sign of INT6 minus fp6_e2m3 under N-default against each other arm
    order = {}
    for item in six_bit(all_arms, labels):
        order.setdefault(item["model"], {})[item["arm"]] = item["int6_minus_fp6"]["expected"]
    six = {}
    for model, by_arm in order.items():
        base = by_arm.get("N-default")
        six[model] = {"int6_minus_fp6_expected": by_arm, "order_changes_against_N-default": {
            arm: bool(base and np.sign(v["difference_pp"]) != np.sign(base["difference_pp"])
                      and (v["interval_pp"][0] > 0 or v["interval_pp"][1] < 0))
            for arm, v in by_arm.items() if arm != "N-default"}}
    q = unseal_protocol()["readout_and_statistics"]["questions"]
    return {"mobilenet_v3_int8_gap": {"criterion": q["mobilenet_v3_int8_gap"],
                                      "arms": gap, "decision": closed, "L-bc_minus_L-nobc": bc_vs_nobc},
            "four_bit_rescue": {"criterion": q["four_bit_rescue"], "models": rescue},
            "six_bit_order": {"criterion": q["six_bit_order"], "models": six}}


def fits():
    result = []
    for folder in sorted((BASE / "fits").iterdir()):
        spec = unseal(folder / "spec.json")
        layers = [read_layer(path)[2] for path in sorted((folder / "layers").glob("*.npz"))]
        done = (folder / "complete.json").exists()
        weights = sum(s["weights"] for s in layers)
        result.append({
            "fit": folder.name, "complete": done, "layers_done": len(layers), "model": spec["model_context"]["model"],
            "weight_format": spec["weight_format"], "weight_scale_rule": spec["weight_scale_rule"], "seed": spec["seed"],
            "reconstruction_input": spec["reconstruction_input"], "calibration_images": spec["calibration"]["images"],
            "iterations": spec["settings"]["iterations"], "weights": weights,
            "fraction_differing_from_nearest": sum(s["differ_from_nearest"] for s in layers) / max(1, weights),
            "non_binary_soft_variables": sum(s["far_from_binary_soft_variables"] for s in layers),
            "clipped_weights": sum(s["clipped_weights"] for s in layers),
            "layers_worse_than_nearest": [s["node"] for s in layers
                                          if s["calibration_error_learned"] > s["calibration_error_nearest"]],
            "median_error_ratio_learned_to_nearest": float(np.median(
                [s["calibration_error_learned"] / s["calibration_error_nearest"] for s in layers
                 if s["calibration_error_nearest"] > 0])) if layers else None,
            "gpu_seconds_sum_over_layers": sum(s["seconds"] for s in layers),
            "layers_held_in_host_memory": sum(s["data_placement"] == "cpu" for s in layers),
            "disk_bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
            "layers": [{k: s[k] for k in ("node", "activation", "weights", "differ_from_nearest",
                                          "calibration_error_nearest", "calibration_error_learned",
                                          "calibration_error_fp32_weight", "far_from_binary_soft_variables",
                                          "data_placement", "seconds")} for s in layers]})
    return result


def figure(all_arms, labels, stem):
    """Top-1 (expected credit) per model and setting; nearest arms hollow, learned arms filled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = [("INT8", "int8", "int8"), ("Posit8*", "posit8_es1", "posit8_es1"), ("INT6", "int6", "int6"),
                ("FP6 e2m3*", "fp6_e2m3", "fp6_e2m3"), ("W4/A8", "int4", "int8"), ("W4/A4", "int4", "int4")]
    series = [("N-default", "default", "#2a78d6", "o", False), ("N-nobc", "default_no_bias_correction", "#eb6834", "s", False),
              ("N-minimal", "minimal", "#1baf7a", "D", False), ("L-nobc", "default_no_bias_correction", "#eda100", "^", True),
              ("L-bc", "default", "#e87ba4", "v", True)]
    titles = {"resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2", "mobilenet_v3_large": "MobileNetV3-Large"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        arms = all_arms[model]
        fp = 100 * float(credit(arms["fp32"][1], labels, "expected").mean())
        ax.axhline(fp, color="#6b6a64", lw=1, ls="--", zorder=1)
        ax.text(len(settings) - 0.5, fp + 1.2, f"FP32 {fp:.1f}", ha="right", va="bottom", fontsize=7, color="#45443f")
        for j, (name, recipe, color, marker, filled) in enumerate(series):
            xs, ys = [], []
            for i, (_, wf, af) in enumerate(settings):
                label = f"{name}--w-{wf}--a-{af}--{recipe}" + ("--fit-mse_per_channel-fp32in-s0" if filled else "")
                if label in arms:
                    xs.append(i + (j - 2) * 0.13)
                    ys.append(100 * float(credit(arms[label][1], labels, "expected").mean()))
            ax.scatter(xs, ys, s=30, marker=marker, linewidths=1.4, zorder=3, label=name,
                       facecolors=color if filled else "none", edgecolors=color)
        ax.set_xticks(range(len(settings)), [s[0] for s in settings], fontsize=8)
        ax.set_title(titles[model], fontsize=10)
        ax.set_ylim(-3, 82)
        ax.grid(axis="y", color="#e6e5df", lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("top-1 %, expected credit (1k screen)", fontsize=8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["nearest, default", "nearest, no bias corr.", "nearest, minimal", "AdaRound, no bias corr.",
                         "AdaRound + B2 bias corr."], loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.text(0.5, 0.955, "Development evidence (imagenet_screen_1k). * non-uniform codebook: learned rounding is an "
             "adaptation, not the published method.", ha="center", fontsize=7, color="#45443f")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    paths = []
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{stem}.{suffix}"
        if path.exists():
            raise SystemExit(f"{path} exists; choose another --tag")
        fig.savefig(path, dpi=200, metadata={"CreationDate": None} if suffix == "pdf" else None)
        paths.append(path)
        print(f"wrote {_show(path)}")
    plt.close(fig)
    return paths


def _show(path):
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def write(path, text):
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with different content; choose another --tag")
        print(f"unchanged {_show(path)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {_show(path)}")


def flat_interval(value, rule="expected"):
    if not value:
        return ""
    v = value[rule]
    return f"{v['difference_pp']:+.2f} [{v['interval_pp'][0]:+.2f}, {v['interval_pp'][1]:+.2f}]"


def main(argv=None):
    global OUT, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="report")
    parser.add_argument("--tag", required=True, help="suffix of the output files, e.g. r2")
    parser.add_argument("--out", help="other output folder (for a dry run)")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args(argv)
    if args.out:
        from pathlib import Path
        OUT = FIGURES = Path(args.out).resolve()
    rows, labels = screen()
    all_arms = {model: load_model(model, rows) for model in MODELS}
    table = matrix(all_arms, labels)
    header = {"evidence": "development_evidence_screen1k", "statistics": "left minus right; pointwise 95 percent paired "
              "image bootstrap, 10000 resamples, seed 20260927 (tools.analysis.b2_ties.paired); no multiplicity correction",
              "protocol": PROTOCOL_PATH}
    dump = lambda body: json.dumps({**header, **body}, indent=1, sort_keys=True) + "\n"  # noqa: E731
    write(OUT / f"matrix-1k--{args.tag}.json", dump({"rows": table}))
    write(OUT / f"six-bit-order--{args.tag}.json", dump({"rows": six_bit(all_arms, labels)}))
    write(OUT / f"faithfulness--{args.tag}.json", dump(faithfulness(all_arms, labels)))
    write(OUT / f"questions--{args.tag}.json", dump(questions(all_arms, labels)))
    write(OUT / f"fits--{args.tag}.json", dump({"fits": fits()}))
    stream = io.StringIO()
    columns = ["model", "arm", "weight_format", "activation_format", "recipe", "weight_scale_rule", "bias_correction",
               "rounding", "fit_mode", "fit_seed", "tied_images", "top1_expected", "top1_lowest_index", "top1_topk",
               "minus_fp32_expected", "minus_fp32_lowest_index", "minus_default_expected", "minus_default_lowest_index",
               "minus_nearest_expected", "minus_nearest_lowest_index", "sealed_b2_top5_reproduced", "label"]
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in table:
        flat = dict(row)
        for rule in ("expected", "lowest_index"):
            flat[f"minus_fp32_{rule}"] = flat_interval(row.get("minus_fp32"), rule)
            flat[f"minus_default_{rule}"] = flat_interval(row.get("minus_default"), rule)
            flat[f"minus_nearest_{rule}"] = flat_interval(row.get("minus_nearest_counterpart"), rule)
        writer.writerow(flat)
    write(OUT / f"matrix-1k--{args.tag}.csv", stream.getvalue())
    if args.figure:
        figure(all_arms, labels, f"b2-recon-matrix--{args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
