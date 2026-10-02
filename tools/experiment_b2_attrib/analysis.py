"""Lane Q4 analysis (CPU): stored SQNR (stage A0) and the arm records (stages A1, A2, B, C).

Reads sealed B2 matrix cells (read-only) and this lane's arm records; writes nothing unless ``write`` is
called with a new versioned folder (results/summaries/b2-attrib-v1/, written once).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tools.analysis.b2_matrix import interval
from tools.experiment_b.common import ROOT, unseal
from tools.experiment_b2 import readout

MATRIX = ROOT / "artifacts/experiment_b2/matrix"
ARMS = ROOT / "artifacts/experiment_b2_attrib/arms"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
PRIMARY = "top1_expected"


def nsr(values):
    """Total noise-to-signal ratio of a set of boundaries: sum of 10^(-SQNR/10) (linear)."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(sum(10 ** (-v / 10) for v in finite)) if finite else None


def db(value):
    return None if value is None or value <= 0 else -10 * math.log10(value)


def cells(recipe="default"):
    """Sealed matrix cells ``{(model, format): payload}`` of one recipe, plus the FP32 baselines."""
    out = {}
    for path in sorted((MATRIX / "cells").glob(f"*--*--{recipe}--1000--*.json")):
        payload = unseal(path)
        out[(payload["model"], payload["format"])] = {**payload, "_path": str(path.relative_to(ROOT))}
    return out


def baselines():
    return {model: cells("baseline")[(model, "fp32")] for model in MODELS}


def credits_of(path):
    return readout.credits(readout.load(ROOT / path) if str(path).endswith(".npz") and "matrix" in str(path)
                           else load_npz(path))


def load_npz(path):
    with np.load(ROOT / path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def cell_sqnr(cell):
    nodes = (cell.get("occupancy") or {}).get("nodes") or {}
    return {name: row.get("sqnr_db") for name, row in nodes.items()}


def cross_cell(recipe="default"):
    """Stage A0 (ii): does network-level stored SQNR predict the top-1 drop to FP32 across formats?"""
    base = baselines()
    rows, excluded = [], []
    for (model, fmt), cell in cells(recipe).items():
        sqnr = cell_sqnr(cell)
        if not sqnr or cell.get("family") in ("bfp", "mx_float"):
            continue
        nodes = cell["occupancy"]["nodes"]
        if any("sqnr_db_nonfinite" in row for row in nodes.values()):
            excluded.append(f"{model}/{fmt}")
            continue
        # sqnr_db None means zero noise (a dead, all-zero boundary): it adds nothing to the NSR.
        values = [v for v in sqnr.values() if v is not None]
        drop = base[model]["readout"]["top1_expected_percent"] - cell["readout"]["top1_expected_percent"]
        rows.append({"model": model, "format": fmt, "bits": cell["bits"], "family": cell["family"],
                     "top1_drop_pp": drop, "median_sqnr_db": float(np.median(values)), "min_sqnr_db": float(min(values)),
                     "network_nsr_db": db(nsr(values)), "tensors": len(values)})
    stats = {}
    for scope in (*MODELS, "pooled"):
        subset = [r for r in rows if scope == "pooled" or r["model"] == scope]
        entry = {"cells": len(subset)}
        for key in ("median_sqnr_db", "min_sqnr_db", "network_nsr_db"):
            rho, p = spearmanr([r[key] for r in subset], [r["top1_drop_pp"] for r in subset])
            entry[key] = {"spearman_rho_vs_drop": float(rho), "p_value": float(p)}
        # Within the usable range only (drop below 20 points): does SQNR still rank the formats?
        usable = [r for r in subset if r["top1_drop_pp"] < 20]
        if len(usable) >= 5:
            rho, p = spearmanr([r["median_sqnr_db"] for r in usable], [r["top1_drop_pp"] for r in usable])
            entry["median_sqnr_db_drop_below_20pp"] = {"cells": len(usable), "spearman_rho_vs_drop": float(rho),
                                                       "p_value": float(p)}
        stats[scope] = entry
    return {"recipe": recipe, "rows": rows, "excluded_nonfinite": excluded, "spearman": stats}


def group_sqnr(model, fmt, groups):
    """Stage A0 (i): stored per-node SQNR of one default cell summarised per group."""
    cell = cells("default")[(model, fmt)]
    sqnr = cell_sqnr(cell)
    out = {}
    for group, members in groups.items():
        values = [sqnr[n] for n in members if n in sqnr]
        finite = [v for v in values if v is not None and math.isfinite(v)]
        worst = sorted(((sqnr[n], n) for n in members if sqnr.get(n) is not None), key=lambda t: t[0])[:3]
        out[group] = {"nodes": len(members), "nsr_db": db(nsr(values)), "min_sqnr_db": min(finite) if finite else None,
                      "median_sqnr_db": float(np.median(finite)) if finite else None,
                      "share_of_network_nsr": (nsr(values) or 0) / nsr(list(sqnr.values())),
                      "lowest": [[n, round(v, 2)] for v, n in worst]}
    return {"cell": cell["_path"], "network_nsr_db": db(nsr(list(sqnr.values()))),
            "median_sqnr_db": float(np.median([v for v in sqnr.values() if v is not None])), "groups": out}


# ---------------------------------------------------------------- arm records (stages A1, A2, B, C)

def arm_records(model, fmt):
    """``{arm name: record payload}`` of one (model, format)."""
    folder = ARMS / f"{model}--{fmt}"
    out = {}
    for path in sorted(folder.glob("*.json")):
        payload = unseal(path)
        out[payload["arm_name"]] = payload
    return out


def arm_credits(record, rule=PRIMARY):
    return readout.credits(load_npz(record["readout_file"]))[rule]


def arm_kl(record):
    return load_npz(record["readout_file"])["kl_nats"].astype(np.float64)


def fp32_credits(model, rule=PRIMARY):
    base = baselines()[model]
    return readout.credits(readout.load(ROOT / base["readout_file"]))[rule]


def paired(left, right):
    d, low, high = interval(left, right)
    return {"diff_pp": round(d, 2), "ci95_pp": [round(low, 2), round(high, 2)]}


def arm_table(model, fmt, records=None):
    """Per arm: top-1 (expected), paired differences to the default, to weights-only and to FP32, mean KL."""
    records = records or arm_records(model, fmt)
    default = arm_credits(records["ref_default"])
    fp32 = fp32_credits(model)
    weights_only = arm_credits(records["ref_weights_only"]) if "ref_weights_only" in records else None
    rows = {}
    for name, record in records.items():
        c = arm_credits(record)
        rows[name] = {"top1_expected": round(100 * float(c.mean()), 2), "kl_mean": record["kl_nats_mean"],
                      "kind": record["arm_kind"], "minus_default": paired(c, default), "minus_fp32": paired(c, fp32),
                      "minus_weights_only": paired(c, weights_only) if weights_only is not None else None,
                      "top1_lowest_index": record["readout"]["top1_lowest_index_percent"],
                      "tied_images": record["readout"]["images_with_tied_top1"],
                      "wide_boundaries": record["meta"]["wide_boundaries"]}
    return rows


def spearman(x, y):
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rho, p = spearmanr(x, y)
    return {"rho": round(float(rho), 3), "p": float(p), "n": len(x)}


def attribution(model, fmt, groups, scheme="role"):
    """Damage (only:G) and recovery (wide:G) per group, additivity, and the SQNR predictors."""
    records = arm_records(model, fmt)
    table = arm_table(model, fmt, records)
    sqnr = cell_sqnr(cells("default")[(model, fmt)])
    default, weights_only = table["ref_default"], table["ref_weights_only"]
    loss = weights_only["top1_expected"] - default["top1_expected"]
    rows = {}
    for group, members in groups.items():
        only, wide = table.get(f"only:{scheme}.{group}"), table.get(f"wide:{scheme}.{group}")
        if only is None or wide is None:
            continue
        values = [sqnr.get(n) for n in members]
        rows[group] = {"nodes": len(members),
                       "damage_pp": round(weights_only["top1_expected"] - only["top1_expected"], 2),
                       "only_minus_weights_only": only["minus_weights_only"],
                       "recovery_pp": round(wide["top1_expected"] - default["top1_expected"], 2),
                       "wide_minus_default": wide["minus_default"],
                       "kl_only": round(only["kl_mean"], 4), "kl_reduction": round(default["kl_mean"] - wide["kl_mean"], 4),
                       "nsr_db": db(nsr(values)), "min_sqnr_db": min([v for v in values if v is not None], default=None)}
    predictors = {}
    if len(rows) >= 4:
        names = list(rows)
        for key, sign in (("nsr_db", -1), ("min_sqnr_db", -1), ("nodes", 1)):
            x = [rows[g][key] for g in names]
            if any(v is None for v in x):
                continue
            predictors[key] = {target: spearman(x, [rows[g][target] for g in names])
                               for target in ("damage_pp", "recovery_pp", "kl_only", "kl_reduction")}
    return {"model": model, "format": fmt, "scheme": scheme, "default_top1": default["top1_expected"],
            "weights_only_top1": weights_only["top1_expected"], "fp32_top1": round(100 * float(fp32_credits(model).mean()), 2),
            "activation_loss_pp": round(loss, 2), "weights_only_minus_default": weights_only["minus_default"],
            "sum_damage_pp": round(sum(r["damage_pp"] for r in rows.values()), 2),
            "sum_recovery_pp": round(sum(r["recovery_pp"] for r in rows.values()), 2),
            "groups": rows, "sqnr_predictors_spearman": predictors}
