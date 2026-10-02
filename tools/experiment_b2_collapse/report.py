"""Lane Q3 report: summaries (written once per tag) and figures from the sealed arm records (CPU).

    .venv/bin/python -m tools.run.experiment_b2_collapse report --tag r1

Development evidence (1k screen).  Statistics: paired image bootstrap of the expected-credit difference
(``tools.analysis.b2_ties.paired``: 10000 resamples, seed 20260927, pointwise 95 percent intervals).
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from tools.analysis.b2_ties import paired
from tools.experiment_b.common import ROOT, file_hash, unseal

from .common import BASE, PROTOCOL_FILE, own_sources

OUT = ROOT / "results/summaries/b2-collapse-v1"
FIGURES = ROOT / "results/figures"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
SHORT = {"resnet18": "R18", "mobilenet_v2": "MBv2", "mobilenet_v3_large": "MBv3-L"}
POLICY_ORDER = ("none", "global", "executed", "dfq_weights_only", "cap_rms", "linear_only", "local_empirical",
                "analytic_fp32")
COLLAPSE5 = (("resnet18", "w4-a8-default-N"), ("resnet18", "w4-a4-default-N"), ("resnet18", "w4-a4-default-L"),
             ("mobilenet_v2", "w4-a4-default-N"), ("mobilenet_v2", "w4-a4-default-L"))
WIDE = ("fp8_e5m2", "fp6_e3m2", "log6", "posit6_es1", "mxfp6_e3m2")


# ----------------------------------------------------------------------------- loading
def own_record(model, name):
    path = BASE / "evals" / model / f"{name}.json"
    if not path.exists():
        return None
    record = unseal(path)
    arrays = np.load(path.with_suffix(".npz"))
    size = arrays["tie_size"].astype(np.float64)
    record["_credit"] = arrays["label_among_maxima"].astype(np.float64) / size
    record["_lowest"] = arrays["label_is_lowest_index_maximum"].astype(np.float64)
    record["_path"] = str(path.relative_to(ROOT))
    return record


def matrix_cell(model, fmt, recipe, base=None):
    base = base or ROOT / "artifacts/experiment_b2/matrix"
    found = sorted((base / "cells").glob(f"{model}--{fmt}--{recipe}--1000--*.json"))
    if len(found) != 1:
        return None
    record = unseal(found[0])
    arrays = np.load(ROOT / record["readout_file"])
    credit = (arrays["greater"] == 0).astype(np.float64) / arrays["equal"].astype(np.float64)
    predicted = np.bincount(arrays["argmax_lowest"].astype(np.int64))
    return {"_credit": credit, "_lowest": (arrays["argmax_lowest"] == arrays["label"]).astype(np.float64),
            "_lowest_index_share": float(predicted.max() / len(credit)),
            "_path": str(found[0].relative_to(ROOT)), "top1_expected_percent": 100 * credit.mean(),
            "configuration_sha256": record["configuration_sha256"]}


def pct(record):
    return None if record is None else round(100 * float(record["_credit"].mean()), 2)


def diff(left, right):
    if left is None or right is None:
        return None
    out = paired(left["_credit"], right["_credit"])
    return {"difference_pp": round(out["difference_pp"], 2),
            "interval_pp": [round(x, 2) for x in out["pointwise_95_interval_pp"]]}


def records_of(model, cell):
    folder = BASE / "evals" / model
    out = {}
    for path in sorted(folder.glob(f"{cell}--*.json")):
        name = path.stem
        rest = name[len(cell) + 2:]
        out[rest] = own_record(model, name)
    return out


# ----------------------------------------------------------------------------- mode 1
_ORDER = {}


def layer_order(model):
    """Topological conv/linear order of ``model`` (sealed records store the per-layer reports with sorted keys;
    the CPU weight analysis keeps the order of ``recon.engine.weight_layers``)."""
    if model not in _ORDER:
        weights = sorted((BASE / "weights").glob("weight-placement-*.json"))
        data = json.loads(weights[-1].read_text())["models"][model]
        _ORDER[model] = list(next(iter(data.values())))
    return _ORDER[model]


def ordered(report, model):
    order = layer_order(model)
    if set(order) != set(report):
        raise ValueError(f"layer names of the correction report differ from the weight analysis ({model})")
    return order


def mechanism(report, none_report=None, model=None):
    """H1.1 readout from the per-layer instrumentation of the global policy (layers in topological order)."""
    if not report:
        return None
    names = ordered(report, model)
    over = [n for n in names if report[n]["rms_global"] > report[n]["rms_fp32_channel_mean"]]
    rectified = [n for n in names if report[n].get("off_fraction_q") is not None]
    worst = max(rectified, key=lambda n: report[n]["off_fraction_q"] - report[n]["off_fraction_fp32"]) if rectified else None
    last = names[-1]
    row = {"layers": len(names), "layers_correction_rms_above_fp32_mean_rms": over,
           "inherited_exceeds_local_in_those": all(report[n]["rms_inherited"] > report[n]["rms_local"] for n in over)
           if over else None,
           "median_inherited_over_local_in_those": float(np.median([report[n]["rms_inherited"] / max(report[n]["rms_local"], 1e-12)
                                                                    for n in over])) if over else None,
           "worst_off_layer": worst,
           "worst_off_fraction": {"fp32": report[worst]["off_fraction_fp32"], "corrected": report[worst]["off_fraction_q"],
                                  "uncorrected": none_report[worst]["off_fraction_q"] if none_report else None} if worst else None,
           "mean_off_fraction_rectified": {"fp32": float(np.mean([report[n]["off_fraction_fp32"] for n in rectified])),
                                           "corrected": float(np.mean([report[n]["off_fraction_q"] for n in rectified])),
                                           "uncorrected": float(np.mean([none_report[n]["off_fraction_q"] for n in rectified]))
                                           if none_report else None} if rectified else None,
           "last_layer": last,
           "last_layer_spread": {"fp32": report[last]["spread_fp32"], "corrected": report[last]["spread_q"],
                                 "uncorrected": none_report[last]["spread_q"] if none_report else None},
           "last_layer_spread_ratio_corrected": report[last]["spread_q"] / report[last]["spread_fp32"]}
    return row


def mode1(model_cells):
    rows, cells = [], {}
    for model, cell in model_cells:
        recs = records_of(model, cell)
        if not recs:
            continue
        none, glob = recs.get("none"), recs.get("global")
        entry = {"model": model, "cell": cell, "policies": {}}
        for key, rec in recs.items():
            if rec is None:
                continue
            row = {"model": model, "cell": cell, "arm": key, "top1_expected": pct(rec),
                   "top1_lowest_index": round(100 * float(rec["_lowest"].mean()), 2),
                   "one_class_share": round(rec["one_class"]["share_among_maxima"], 4),
                   "one_class": rec["one_class"]["class"], "collapse": rec["one_class"]["collapse"],
                   "minus_none": diff(rec, none) if key != "none" else None,
                   "minus_global": diff(rec, glob) if key != "global" else None,
                   "reproduction_check": rec.get("reproduction_check"), "record": rec["_path"]}
            rows.append(row)
            entry["policies"][key] = row
        entry["mechanism_global"] = mechanism(glob["correction_report"] if glob else None,
                                              none["correction_report"] if none else None, model)
        variants = {k: v for k, v in entry["policies"].items() if k in ("local_empirical", "analytic_fp32", "linear_only",
                                                                        "cap_rms", "executed", "dfq_weights_only")}
        entry["h13"] = {k: {"avoids_collapse": not v["collapse"],
                            "not_worse_than_none_by_2": (v["minus_none"]["difference_pp"] >= -2.0) if v["minus_none"] else None}
                        for k, v in variants.items()}
        entry["h12_images"] = {k: v["collapse"] for k, v in entry["policies"].items() if k.startswith("global")}
        entry["guard"] = guard(entry["mechanism_global"], entry["policies"])
        cells[f"{model}/{cell}"] = entry
    return rows, cells


GUARD_RATIO = 0.2  # the H1.1 indicator of the protocol: logit spread corrected / FP32 below 0.2


def guard(mechanism_row, rows):
    """Exploratory (post hoc, development evidence): keep the B2 global correction unless its logit spread on the
    correction images (label-free, no screen image) falls below GUARD_RATIO of the FP32 spread; then use the local
    own-error correction instead.  Reported as a virtual arm next to the measured ones."""
    if not mechanism_row or "global" not in rows or "local_empirical" not in rows:
        return None
    ratio = mechanism_row["last_layer_spread_ratio_corrected"]
    choice = "local_empirical" if ratio < GUARD_RATIO else "global"
    best = max(("global", "local_empirical", "none"), key=lambda k: rows[k]["top1_expected"] if k in rows else -1)
    return {"logit_spread_ratio": round(ratio, 4), "choice": choice, "top1_expected": rows[choice]["top1_expected"],
            "minus_none": rows[choice]["minus_none"], "best_of_global_local_none": best}


def mode1_cells():
    out = []
    for model in MODELS:
        for path in sorted((BASE / "evals" / model).glob("*--none.json")):
            cell = path.stem[:-len("--none")]
            if cell.startswith(("faith", "w")):
                out.append((model, cell))
    return out


# ----------------------------------------------------------------------------- mode 2
def mode2():
    rows = []
    for model in MODELS:
        for fmt in ("posit8_es1", "posit6_es1"):
            base = {r: own_record(model, f"{fmt}-minimal-anchor--none--{r}")
                    for r in ("maxabs_per_channel", "anchor_one", "mse_per_channel")}
            ref = base["mse_per_channel"]
            for rule, rec in base.items():
                if rec is None:
                    continue
                rows.append({"model": model, "format": fmt, "arm": f"anchor:{rule}", "top1_expected": pct(rec),
                             "one_class_share": round(rec["one_class"]["share_among_maxima"], 4),
                             "one_class": rec["one_class"]["class"], "collapse": rec["one_class"]["collapse"],
                             "minus_mse_weights": diff(rec, ref) if rule != "mse_per_channel" else None,
                             "median_weight_sqnr_db": float(np.median([v for v in rec["weight_sqnr_db"].values() if v is not None])),
                             "reproduction_check": rec.get("reproduction_check"), "record": rec["_path"]})
    for path in sorted((BASE / "evals" / "mobilenet_v2").glob("posit8_es1-minimal-swap--none--*.json")):
        rec = own_record("mobilenet_v2", path.stem)
        rows.append({"model": "mobilenet_v2", "format": "posit8_es1", "arm": path.stem.split("--")[-1],
                     "top1_expected": pct(rec), "one_class_share": round(rec["one_class"]["share_among_maxima"], 4),
                     "one_class": rec["one_class"]["class"], "collapse": rec["one_class"]["collapse"],
                     "record": rec["_path"]})
    weights = sorted((BASE / "weights").glob("weight-placement-*.json"))
    summary = json.loads(weights[-1].read_text())["summary"] if weights else None
    return rows, summary


# ----------------------------------------------------------------------------- mode 3
def cell_provenance(path, model, fmt):
    """Redirected matrix cell: which bias-correction path produced it, and the lane's reproduction checks
    (r4: tools/experiment_b2_collapse/lowmem.py, protocol addenda 1-2)."""
    record = unseal(ROOT / path)
    checks = {}
    for check in sorted((BASE / "matrix" / "checks").glob(f"{model}--{fmt}--*.json")):
        checks[check.name] = unseal(check)
    return {"cell": path, "configuration_sha256": record["configuration_sha256"],
            "low_memory_bias_correction": "tools/experiment_b2_collapse/lowmem.py" in record.get("own_sources", {}),
            "reproduction_checks": checks}


def mode3():
    rows, decisions = [], {}
    for model in MODELS:
        for fmt in WIDE:
            default, minimal = matrix_cell(model, fmt, "default"), matrix_cell(model, fmt, "minimal")
            if default is None or minimal is None:
                continue
            arms = {"default": default, "minimal": minimal}
            identity, provenance = {}, {}
            for recipe in ("default_no_bias_correction", "default_weight_maxabs"):
                if fmt == "mxfp6_e3m2":
                    arms[recipe] = matrix_cell(model, fmt, recipe, base=BASE / "matrix")
                    if arms[recipe] is not None:
                        provenance[recipe] = cell_provenance(arms[recipe]["_path"], model, fmt)
                    continue
                # reference arm (B2 engine.bias_correct) or, for default_weight_maxabs, the lane's lean
                # bit-identical sequential correction 'global' (tools/run/experiment_b2_collapse_drive.py)
                b2 = own_record(model, f"{fmt}-{recipe}-N--b2")
                lean = own_record(model, f"{fmt}-{recipe}-N--global") if recipe == "default_weight_maxabs" else None
                arms[recipe] = b2 if b2 is not None else lean
                if b2 is not None and lean is not None:
                    identity[recipe] = {"b2": b2["_path"], "global": lean["_path"],
                                        "deployed_state_equal": b2["deployed_state_sha256"] == lean["deployed_state_sha256"],
                                        "credit_arrays_equal": bool(np.array_equal(b2["_credit"], lean["_credit"]))}
            for policy in ("global", "local_empirical", "analytic_fp32", "linear_only", "cap_rms", "dfq_weights_only"):
                arms[f"bc:{policy}"] = own_record(model, f"{fmt}-default-N--{policy}")
            loss = diff(minimal, default)
            entry = {"model": model, "format": fmt, "minimal_minus_default": loss, "arms": {},
                     "b2_versus_lean_identity": identity or None}
            if fmt == "mxfp6_e3m2":
                entry["redirected_cell_provenance"] = provenance or None
            for key, rec in arms.items():
                if rec is None:
                    continue
                d = diff(rec, default) if key != "default" else None
                frac = (d["difference_pp"] / loss["difference_pp"]) if d and loss["difference_pp"] else None
                row = {"model": model, "format": fmt, "arm": key, "top1_expected": pct(rec),
                       "minus_default": d, "recovered_fraction_of_loss": None if frac is None else round(frac, 3),
                       "one_class_share": round(rec["one_class"]["share_among_maxima"], 4) if "one_class" in rec else None,
                       "collapse": rec["one_class"]["collapse"] if "one_class" in rec else None,
                       "lowest_index_prediction_share": round(rec["one_class"]["share_lowest_index"], 4)
                       if "one_class" in rec else round(rec["_lowest_index_share"], 4),
                       "record": rec["_path"]}
                if "reproduction_check" in rec:
                    row["reproduction_check"] = rec["reproduction_check"]
                rows.append(row)
                entry["arms"][key] = row
            glob = arms.get("bc:global")
            entry["mechanism_global"] = mechanism(glob["correction_report"], None, model) if glob else None
            if glob is not None and "bc:local_empirical" in entry["arms"]:
                ratio = entry["mechanism_global"]["last_layer_spread_ratio_corrected"]
                pick = "bc:local_empirical" if ratio < GUARD_RATIO else "default"
                entry["guard"] = {"logit_spread_ratio": round(ratio, 4), "choice": pick,
                                  "top1_expected": entry["arms"][pick]["top1_expected"],
                                  "minus_default": entry["arms"][pick]["minus_default"]}
            if loss["difference_pp"] > 0 and loss["interval_pp"][0] > 0:
                cause = []
                for key in ("default_no_bias_correction", "default_weight_maxabs"):
                    row = entry["arms"].get(key)
                    if row and row["recovered_fraction_of_loss"] is not None and row["recovered_fraction_of_loss"] >= 0.5 \
                            and row["minus_default"]["interval_pp"][0] > 0:
                        cause.append(key)
                entry["cause_by_rule"] = cause
            else:
                entry["cause_by_rule"] = "no significant default-minus-minimal loss"
            decisions[f"{model}/{fmt}"] = entry
    return rows, decisions


# ----------------------------------------------------------------------------- output
def write_once(path, text):
    if path.exists():
        raise SystemExit(f"{path} exists (summaries are written once; use a new tag)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def to_csv(rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})
    return buffer.getvalue()


def figures(tag, m1cells, m3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    ink, muted = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    written = []
    # Figure 1: top-1 per correction policy, one panel per model
    policies = ("none", "global", "cap_rms", "linear_only", "local_empirical", "analytic_fp32", "dfq_weights_only")
    names = {"none": "no correction", "global": "B2 (global, = DFQ App. D at A32)", "cap_rms": "capped at FP32 mean RMS",
             "linear_only": "linear-output layers only", "local_empirical": "own error, quantized input",
             "analytic_fp32": "own error, FP32 input", "dfq_weights_only": "DFQ App. D (weights only)"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True)
    seen = set()
    for ax, model in zip(axes, MODELS):
        cells = [c for c in m1cells.values() if c["model"] == model]
        for i, cell in enumerate(cells):
            for j, policy in enumerate(policies):
                row = cell["policies"].get(policy)
                if row is None:
                    continue
                ax.scatter(row["top1_expected"], i + (j - 3) * 0.09, s=14, color=colors[j],
                           marker="x" if row["collapse"] else "o", linewidths=1.2,
                           label=names[policy] if policy not in seen else None, zorder=3)
                seen.add(policy)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([c["cell"] for c in cells])
        ax.set_title(SHORT[model], color=ink)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        ax.set_xlabel("top-1, expected credit (%), 1k screen")
    handles, labels = [sum(x, []) for x in zip(*(ax.get_legend_handles_labels() for ax in axes))]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.text(0.99, 0.01, "x = one class among the maxima on >= 90% of images", ha="right", color=muted, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext in ("pdf", "png"):
        path = FIGURES / f"b2-collapse-mode1--{tag}.{ext}"
        if path.exists():
            raise SystemExit(f"{path} exists")
        fig.savefig(path, dpi=200)
        written.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    # Figure 2: per-layer correction of the faithfulness cell (global): inherited vs local vs FP32 channel-mean RMS
    rec = own_record("resnet18", "faith-w4-a32-perlayer-N--global")
    none = own_record("resnet18", "faith-w4-a32-perlayer-N--none")
    if rec is not None and none is not None:
        report, base = rec["correction_report"], none["correction_report"]
        layers = ordered(report, "resnet18")
        x = np.arange(len(layers))
        fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        a.plot(x, [report[n]["rms_inherited"] for n in layers], color=colors[0], lw=2, marker="o", ms=4,
               label="inherited part (upstream drift)")
        a.plot(x, [report[n]["rms_local"] for n in layers], color=colors[1], lw=2, marker="o", ms=4,
               label="own weight-error part")
        a.plot(x, [report[n]["rms_fp32_channel_mean"] for n in layers], color=muted, lw=1, ls="--",
               label="RMS of FP32 channel means")
        a.set_ylabel("RMS over channels")
        a.legend(frameon=False, loc="upper left")
        a.set_title("ResNet18, INT4 weights (one scale per layer), FP32 activations: B2 correction per layer", color=ink)
        b.plot(x, [report[n]["spread_q"] / report[n]["spread_fp32"] for n in layers], color=colors[0], lw=2, marker="o",
               ms=4, label="with B2 correction")
        b.plot(x, [base[n]["spread_q"] / base[n]["spread_fp32"] for n in layers], color=colors[1], lw=2, marker="o",
               ms=4, label="without correction")
        rect = [i for i, n in enumerate(layers) if report[n]["off_fraction_q"] is not None]
        b.bar([i for i in rect], [report[layers[i]]["off_fraction_q"] for i in rect], color=colors[0], alpha=0.25,
              width=0.6, label="always-off channels, corrected (share)")
        b.set_ylabel("image-to-image spread\nquantized / FP32")
        b.set_xticks(x)
        b.set_xticklabels(layers, rotation=60, ha="right", fontsize=6.5)
        b.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = FIGURES / f"b2-collapse-layers--{tag}.{ext}"
            if path.exists():
                raise SystemExit(f"{path} exists")
            fig.savefig(path, dpi=200)
            written.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    # Figure 3: mode-3 decomposition
    entries = [e for e in m3.values() if e["model"] in ("resnet18", "mobilenet_v2")]
    if entries:
        arms = ("default", "default_no_bias_correction", "default_weight_maxabs", "minimal")
        labels = ("default", "default without bias correction", "default with max-abs weights", "minimal")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        width = 0.2
        for j, (arm, lab) in enumerate(zip(arms, labels)):
            vals = [e["arms"].get(arm, {}).get("top1_expected") for e in entries]
            xs = [i + (j - 1.5) * width for i, v in enumerate(vals) if v is not None]
            ax.bar(xs, [v for v in vals if v is not None], width=width - 0.02, color=colors[j], label=lab)
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels([f"{SHORT[e['model']]}\n{e['format']}" for e in entries], fontsize=7)
        ax.set_ylabel("top-1, expected credit (%)")
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = FIGURES / f"b2-collapse-mode3--{tag}.{ext}"
            if path.exists():
                raise SystemExit(f"{path} exists")
            fig.savefig(path, dpi=200)
            written.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    # Figure 4: mode 2, where the per-channel weight scale puts the weights on the posit8 grid
    weights = sorted((BASE / "weights").glob("weight-placement-*.json"))
    if weights:
        import torch
        from tools.experiment_b2.codebook import TableQuantizer
        data = json.loads(weights[-1].read_text())
        layers = data["models"]["mobilenet_v2"]["posit8_es1"]
        levels = TableQuantizer("posit8_es1", "signed", torch.device("cpu")).levels.double().numpy()
        levels = levels[levels > 0]
        ratio = float(np.median([v["mse_median_ratio_to_maxabs"] for v in layers.values()]))
        anchors = (("max-abs at maxpos 4096 (minimal)", float(levels.max())), ("max-abs at 1.0", 1.0),
                   (f"MSE search (default; median: max at {levels.max() / ratio:.1f})", float(levels.max()) / ratio))
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": (1, 1.3)})
        for j, (name, top) in enumerate(anchors):
            rel = levels / top
            rel = rel[(rel >= 1e-3) & (rel <= 1.0)]
            gaps = np.diff(rel) / rel[1:]
            a.plot(rel[1:], 100 * gaps, color=colors[j], lw=1.6, marker="o", ms=2.5, label=name)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("|weight| / channel max-abs")
        a.set_ylabel("relative level spacing (%)")
        a.set_title("posit8_es1 grid seen by the weights", color=ink)
        a.grid(color="#e6e5e0", linewidth=0.6)
        a.legend(frameon=False, fontsize=6.5, loc="upper left")
        names = list(layers)
        x = np.arange(len(names))
        for j, (key, name) in enumerate((("maxabs", "max-abs at maxpos"), ("anchor_one", "max-abs at 1.0"),
                                         ("mse", "MSE search"))):
            b.plot(x, [layers[n][key]["sqnr_db"] for n in names], color=colors[j], lw=1.6, label=name)
        b.set_xlabel("conv/linear layer of MobileNetV2 (input to output)")
        b.set_ylabel("weight SQNR (dB)")
        b.set_title("MobileNetV2, posit8_es1, per layer", color=ink)
        b.grid(color="#e6e5e0", linewidth=0.6)
        b.legend(frameon=False, fontsize=6.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = FIGURES / f"b2-collapse-mode2--{tag}.{ext}"
            if path.exists():
                raise SystemExit(f"{path} exists")
            fig.savefig(path, dpi=200)
            written.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    return written


def main(tag, dry=False):
    m1rows, m1cells = mode1(mode1_cells())
    m2rows, weights = mode2()
    m3rows, m3 = mode3()
    if dry:
        return {"mode1": (m1rows, m1cells), "mode2": (m2rows, weights), "mode3": (m3rows, m3)}
    addenda = sorted(PROTOCOL_FILE.parent.glob(PROTOCOL_FILE.stem + "-addendum-*.json"))
    header = {"tag": tag, "protocol": {"path": str(PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(PROTOCOL_FILE)},
              "protocol_addenda": [{"path": str(a.relative_to(ROOT)), "sha256": file_hash(a)} for a in addenda],
              "own_sources": {**own_sources(), "tools/experiment_b2_collapse/report.py":
                              file_hash(ROOT / "tools/experiment_b2_collapse/report.py")},
              "evidence": "development_evidence_screen1k",
              "statistics": "paired image bootstrap, expected credit, 10000 resamples, seed 20260927, pointwise 95%"}
    outputs = {
        f"mode1-correction-variants--{tag}.json": {**header, "rows": m1rows, "cells": m1cells},
        f"mode2-posit-anchors--{tag}.json": {**header, "rows": m2rows, "weight_placement_summary": weights},
        f"mode3-wide-exponent--{tag}.json": {**header, "rows": m3rows, "decisions": m3}}
    for name, document in outputs.items():
        write_once(OUT / name, json.dumps(document, indent=1, allow_nan=False, default=float) + "\n")
    for name, rows in ((f"mode1-correction-variants--{tag}.csv", m1rows), (f"mode2-posit-anchors--{tag}.csv", m2rows),
                       (f"mode3-wide-exponent--{tag}.csv", m3rows)):
        write_once(OUT / name, to_csv(rows))
    written = figures(tag, m1cells, m3)
    print(json.dumps({"summaries": sorted(outputs), "figures": written}))
    return 0
