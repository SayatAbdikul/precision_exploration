"""Addendum 1 to the Q7 quality-metrics protocol (2026-10-02, after independent review 1).

Gives code and a versioned summary to numbers the v1 document derived by hand (near-equal subsets, their power table,
the agreement / top-1 correlation) and to the corrections of three document statements (coverage extremes, slope
quantiles, tie-scheme gap). Reads the v1 summaries only (read-only) and re-runs one v1 coverage scenario.

Protocol: public/experiments/configs/breadth-study/quality-metrics-protocol-v1-addendum-1.json.
Development evidence (ImageNet and COCO screen-1k lists).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from tools.experiment_b.common import ROOT
from . import PROTOCOL, SEED, resolution
from .report import Writer

ADDENDUM = "public/experiments/configs/breadth-study/quality-metrics-protocol-v1-addendum-1.json"
V1 = "results/summaries/quality-metrics-v1"
KINDS = ("format_pair", "recipe_pair", "cell_vs_int8", "cell_vs_fp32", "accumulator_vs_wide")
THRESHOLDS = (1.0, 2.0, 3.0)
SCENARIO_KNIFE = ("pd_q5_asym_q95", 4)  # name and v1 scenario index (5 x 5 grid, discordance-major)
NULL_TAG = 100


def _rows(v1, name):
    with (Path(v1) / name).open() as f:
        return list(csv.DictReader(f))


def _f(value):
    return float(value) if value not in ("", None) else float("nan")


def mdd_pp(n, pd):
    return 100 * resolution.mdd(n, pd)


# ----------------------------------------------------------------------------------------------- near-equal subsets
def near_equal(sub_rows, thresholds=THRESHOLDS):
    out = []
    for t in thresholds:
        for kind in KINDS:
            rs = [r for r in sub_rows if r["kind"] == kind and abs(_f(r["diff_expected_pp"])) < t]
            if not rs:
                continue
            pd = float(np.median([_f(r["discordance"]) for r in rs]))
            half = float(np.median([_f(r["width_expected_median_n1000"]) / 2 for r in rs]))
            row = {"threshold_pp": t, "kind": kind, "contrasts": len(rs),
                   "contrasts_all": sum(r["kind"] == kind for r in sub_rows),
                   "discordance_median": pd, "halfwidth_n1000_median_pp": half}
            for N in (4000, 9000):
                row[f"projected_halfwidth_n{N}_pp"] = half * math.sqrt(1000 / N)
            for n in (1000, 4000, 9000):
                row[f"mdd_pp_n{n}"] = mdd_pp(n, pd)
            row["post_hoc"] = t == 2.0
            out.append(row)
    return out


def all_median_discordance(sub_rows, kind):
    return float(np.median([_f(r["discordance"]) for r in sub_rows if r["kind"] == kind]))


POWER_ROWS = (  # (hypothesis type, contrast kind whose near-equal median discordance is used) - the v1 document table
    ("difference", "format_pair"),
    ("equivalence", "format_pair"),
    ("noninferiority", "accumulator_vs_wide"),
    ("equivalence", "accumulator_vs_wide"),
    ("beyond_margin", "recipe_pair"),
)


def power_near_equal(sub_rows, near_rows, threshold=2.0):
    med = {r["kind"]: r["discordance_median"] for r in near_rows if r["threshold_pp"] == threshold}
    out = []
    for hyp, kind in POWER_ROWS:
        for basis, pd in (("near_equal", med[kind]), ("all_contrasts", all_median_discordance(sub_rows, kind))):
            row = {"hypothesis": hyp, "kind": kind, "basis": basis, "threshold_pp": threshold if basis == "near_equal" else None,
                   "discordance": pd}
            for m in (1.0, 0.5, 0.25):
                row[f"images_margin{m}"] = resolution.images_needed(pd, m / 100, hyp)
            out.append(row)
    # ratio all-contrast / near-equal
    for i in range(0, len(out), 2):
        out[i + 1]["ratio_to_near_equal"] = out[i + 1]["images_margin1.0"] / out[i]["images_margin1.0"]
    return out


# ----------------------------------------------------------------------------------------------- slopes, correlation
def slope_quantiles(sub_rows):
    out = []
    for kind in KINDS:
        row = {"kind": kind}
        for col in ("slope_expected", "slope_binary"):
            s = np.array([_f(r[col]) for r in sub_rows if r["kind"] == kind])
            s = s[np.isfinite(s)]
            row[f"{col}_count"] = len(s)
            row.update({f"{col}_q10": float(np.percentile(s, 10)), f"{col}_median": float(np.median(s)),
                        f"{col}_q90": float(np.percentile(s, 90)), f"{col}_min": float(s.min()), f"{col}_max": float(s.max())})
        out.append(row)
    return out


def own_cell(u):
    return (u["source"] == "matrix" and u["group"] in ("default", "minimal") and "@" not in u["unit"]
            and u["format"] != "fp32" and _f(u["top1_expected"]) >= 10.0)


def agreement_correlation(unit_rows):
    us = [u for u in unit_rows if own_cell(u)]
    agree = np.array([_f(u["agree_lowest"]) for u in us])
    drop = np.array([_f(u["top1_expected"]) - _f(u["fp32_top1"]) for u in us])
    return {"cells": len(us), "pearson_r": float(pearsonr(agree, drop)[0]), "spearman_rho": float(spearmanr(agree, drop)[0]),
            "selection": "matrix own cells (default/minimal, no '@' arm, not FP32), top1_expected >= 10 percent",
            "x": "agree_lowest (percent)", "y": "top1_expected - fp32_top1 (pp)"}


# ----------------------------------------------------------------------------------------------- coverage
def coverage_extremes(cov_rows):
    extremes, gaps = [], []
    for method in ("percentile", "bca", "wald"):
        for n in resolution.NS:
            rs = [r for r in cov_rows if int(r["n"]) == n]
            worst = min(rs, key=lambda r: _f(r[method]))
            extremes.append({"method": method, "n": n, "min_coverage": _f(worst[method]), "scenario": worst["scenario"],
                             "discordance": _f(worst["discordance"]), "right_only_share": _f(worst["right_only_share"]),
                             "true_diff_pp": _f(worst["true_diff_pp"]),
                             "median_coverage": float(np.median([_f(r[method]) for r in rs]))})
    for r in cov_rows:
        g = _f(r["bca"]) - _f(r["percentile"])
        gaps.append({"scenario": r["scenario"], "n": int(r["n"]), "discordance": _f(r["discordance"]),
                     "right_only_share": _f(r["right_only_share"]), "true_diff_pp": _f(r["true_diff_pp"]),
                     "percentile": _f(r["percentile"]), "bca": _f(r["bca"]), "wald": _f(r["wald"]),
                     "bca_minus_percentile": g, "flag": "bca_better" if g >= 0.02 else ("bca_worse" if g <= -0.02 else "")})
    return extremes, gaps


def _coverage_counts(p_left, p_right, n, scenario, truths, datasets=resolution.SIM_DATASETS):
    """Same draws as resolution.coverage (same seed tags); coverage of several candidate truths."""
    rng = np.random.default_rng([SEED, 21, scenario, n])
    counts = rng.multinomial(n, [p_left, 1 - p_left - p_right, p_right], size=datasets)
    hits = np.zeros((len(truths), 3))
    zero_endpoint = np.zeros(3)
    for lo, _, ro in counts:
        for j, (low, high) in enumerate(resolution.intervals(n, int(lo), int(ro))):
            for i, t in enumerate(truths):
                hits[i, j] += low <= t + 1e-12 and t - 1e-12 <= high
            zero_endpoint[j] += abs(low) <= 1e-12 or abs(high) <= 1e-12
    return hits / datasets, zero_endpoint / datasets


def coverage_knife_edge(cov_rows):
    name, index = SCENARIO_KNIFE
    base = [r for r in cov_rows if r["scenario"] == name]
    p, a = _f(base[0]["discordance"]), _f(base[0]["right_only_share"])
    truth = p * a - p * (1 - a)
    out = []
    for r in sorted(base, key=lambda r: int(r["n"])):
        n = int(r["n"])
        share, zero = _coverage_counts(p * (1 - a), p * a, n, index, (truth, 0.0))
        for j, method in enumerate(("percentile", "bca", "wald")):
            out.append({"scenario": name, "n": n, "method": method, "true_diff_pp": 100 * truth,
                        "coverage_v1": _f(r[method]), "coverage_recomputed": float(share[0, j]),
                        "coverage_of_zero": float(share[1, j]), "share_endpoint_at_zero": float(zero[j])})
    for n in resolution.NS:
        c = resolution.coverage(p / 2, p / 2, n, scenario=NULL_TAG)
        for method in ("percentile", "bca", "wald"):
            out.append({"scenario": f"null_pd{p:.4f}", "n": n, "method": method, "true_diff_pp": 0.0,
                        "coverage_v1": None, "coverage_recomputed": float(c[method]), "coverage_of_zero": float(c[method]),
                        "share_endpoint_at_zero": None})
    return out


# ----------------------------------------------------------------------------------------------- detector
def tie_scheme_gap(tie_rows):
    out = []
    for r in tie_rows:
        out.append({"format": r["format"], "recipe": r["recipe"], "fixed_rule": _f(r["fixed_rule"]),
                    "image_order_mean": _f(r["image_order_mean"]), "uniform_mean": _f(r["uniform_mean"]),
                    "image_order_minus_uniform": _f(r["image_order_mean"]) - _f(r["uniform_mean"]),
                    "fixed_rank_image_order": _f(r["image_order_fixed_rank"]), "fixed_rank_uniform": _f(r["uniform_fixed_rank"]),
                    "image_order_sd": _f(r["image_order_sd"]), "uniform_sd": _f(r["uniform_sd"])})
    return out


def new_false_positives(det_rows):
    return [{"format": r["format"], "recipe": r["recipe"], "fp_new_s025": _f(r["fp_new_s025"]),
             "tp_kept_s025_iou50": _f(r["tp_kept_s025_iou50"]), "map50_95_minus_fp32": _f(r["map50_95_minus_fp32"])}
            for r in det_rows if r["format"] != "fp32"]


# ----------------------------------------------------------------------------------------------- main
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out, v1=V1, knife_edge=True):
    v1 = Path(ROOT) / v1 if not Path(v1).is_absolute() else Path(v1)
    w = Writer(out)
    sub = _rows(v1, "partB-subsample.csv")
    near = near_equal(sub)
    w.csv("partB-near-equal.csv", near)
    w.csv("partB-power-near-equal.csv", power_near_equal(sub, near))
    w.csv("partB-slope-quantiles.csv", slope_quantiles(sub))
    w.json("partA-agreement-correlation.json", agreement_correlation(_rows(v1, "partA-units.csv")))
    cov = _rows(v1, "partB-coverage.csv")
    extremes, gaps = coverage_extremes(cov)
    w.csv("partB-coverage-extremes.csv", extremes)
    w.csv("partB-coverage-bca-vs-percentile.csv", gaps)
    if knife_edge:
        w.csv("partB-coverage-knife-edge.csv", coverage_knife_edge(cov))
    w.csv("partC-detector-tie-scheme-gap.csv", tie_scheme_gap(_rows(v1, "partC-detector-tie-orders.csv")))
    w.csv("partD-new-false-positives.csv", new_false_positives(_rows(v1, "partD-detector-agreement.csv")))
    code = Path(__file__).parent
    w.json("manifest-addendum-1.json", {
        "written": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": PROTOCOL, "protocol_sha256": _sha(Path(ROOT) / PROTOCOL),
        "addendum": ADDENDUM, "addendum_sha256": _sha(Path(ROOT) / ADDENDUM),
        "seed": SEED, "evidence": "development evidence: ImageNet and COCO screen-1k lists only",
        "inputs_sha256": {p.name: _sha(p) for p in sorted(v1.glob("*.csv"))},
        "analysis_code_sha256": {p.name: _sha(p) for p in sorted(code.glob("*.py"))},
        "note": "v1 manifest-A-B-C-D-E.json lists figures.py aee9194b...; that file was edited afterwards (tick labels only) "
                "and its current hash is listed here",
        "files": [p.name for p in w.files] + ["manifest-addendum-1.json"],
    })
    return 0
