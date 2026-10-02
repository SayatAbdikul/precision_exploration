"""Addendum 2 to the Q7 quality-metrics protocol (2026-10-02, after independent review 2).

Separates contrasts identical to their reference (binary discordance 0 at n = 1,000) from the planning subsets, gives a
full power grid and per-contrast sufficiency shares, splits the knife-edge misses by end point, and enumerates the
coverage of the low-discordance scenarios exactly over the discordant counts (no data-set Monte Carlo error).
Reads the v1 summaries only (read-only).

Protocol: public/experiments/configs/breadth-study/quality-metrics-protocol-v1-addendum-2.json.
Development evidence (ImageNet and COCO screen-1k lists).
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import binom

from tools.experiment_b.common import ROOT
from . import PROTOCOL, SEED, resolution
from .addendum1 import ADDENDUM as ADDENDUM1, KINDS, SCENARIO_KNIFE, V1, _f, _rows, mdd_pp
from .report import Writer

ADDENDUM2 = "public/experiments/configs/breadth-study/quality-metrics-protocol-v1-addendum-2.json"
THRESHOLDS = (1.0, 2.0, 3.0)
HYPOTHESES = ("difference", "equivalence", "noninferiority", "beyond_margin")
MARGINS = (1.0, 0.5, 0.25)
QUANTILES = (("q25", 25), ("median", 50), ("q75", 75))
EXACT_MAX_DISCORDANCE = 0.05
EXACT_SEEDS = (0, 1, 2)
MASS = 1e-6
NULL_EXTRA = ("null_pd0.0255_addendum1", 0.0255)


# ----------------------------------------------------------------------------------------------- subsets
def identical(r):
    return _f(r["discordance"]) == 0.0


def label(r):
    if r["kind"] != "accumulator_vs_wide":
        return "other"
    return "control" if r["contrast"].endswith("/control") else "event_free_width"


def subset(rows, kind, name):
    rs = [r for r in rows if r["kind"] == kind]
    if name.endswith("_differing"):
        rs = [r for r in rs if not identical(r)]
        name = name[: -len("_differing")]
    if name.startswith("near_equal_"):
        t = float(name.split("_")[-1])
        rs = [r for r in rs if abs(_f(r["diff_expected_pp"])) < t]
    return rs


SUBSETS = ("all", "all_differing") + tuple(f"near_equal_{t:g}{s}" for t in THRESHOLDS for s in ("", "_differing"))


def identical_contrasts(rows):
    return [{"kind": r["kind"], "contrast": r["contrast"], "label": label(r), "diff_expected_pp": _f(r["diff_expected_pp"]),
             "width_expected_median_n1000": _f(r["width_expected_median_n1000"])} for r in rows if identical(r)]


def subsets(rows):
    out = []
    for kind in KINDS:
        for name in SUBSETS:
            rs = subset(rows, kind, name)
            if not rs:
                continue
            pd = np.array([_f(r["discordance"]) for r in rs])
            row = {"kind": kind, "subset": name, "contrasts": len(rs), "identical": sum(identical(r) for r in rs)}
            for q, p in QUANTILES:
                row[f"discordance_{q}"] = float(np.percentile(pd, p))
            half = float(np.median([_f(r["width_expected_median_n1000"]) / 2 for r in rs]))
            row["halfwidth_n1000_median_pp"] = half
            for n in resolution.NS:
                row[f"width_median_n{n}_pp"] = float(np.median([_f(r[f"width_expected_median_n{n}"]) for r in rs]))
            for N in (4000, 9000):
                row[f"projected_halfwidth_n{N}_pp"] = half * math.sqrt(1000 / N)
            for n in (1000, 4000, 9000):
                row[f"mdd_pp_n{n}"] = mdd_pp(n, row["discordance_median"])
            out.append(row)
    return out


GRID_SUBSETS = ("all", "all_differing", "near_equal_2", "near_equal_2_differing")


def power_grid(subset_rows):
    out = []
    for s in subset_rows:
        if s["subset"] not in GRID_SUBSETS:
            continue
        for hyp in HYPOTHESES:
            for q, _ in QUANTILES:
                pd = s[f"discordance_{q}"]
                row = {"kind": s["kind"], "subset": s["subset"], "contrasts": s["contrasts"], "hypothesis": hyp,
                       "discordance_quantile": q, "discordance": pd}
                for m in MARGINS:
                    row[f"images_margin{m}"] = resolution.images_needed(pd, m / 100, hyp) if pd > 0 else 0
                out.append(row)
    return out


SUFF_SUBSETS = ("near_equal_2", "near_equal_2_differing", "all_differing")


def sufficiency(rows):
    out = []
    for kind in KINDS:
        for name in SUFF_SUBSETS:
            rs = subset(rows, kind, name)
            if not rs:
                continue
            pds = [_f(r["discordance"]) for r in rs]
            for hyp in ("noninferiority", "equivalence", "difference"):
                for m in (1.0, 0.5):
                    need = [resolution.images_needed(p, m / 100, hyp) if p > 0 else 0 for p in pds]
                    row = {"kind": kind, "subset": name, "contrasts": len(rs), "hypothesis": hyp, "margin_pp": m}
                    for N in (1000, 4000, 9000):
                        row[f"share_sufficient_n{N}"] = float(np.mean([x <= N for x in need]))
                    row["images_median"] = float(np.median(need))
                    out.append(row)
    return out


# ----------------------------------------------------------------------------------------------- knife edge
def knife_edge_endpoints(cov_rows):
    name, index = SCENARIO_KNIFE
    base = sorted([r for r in cov_rows if r["scenario"] == name], key=lambda r: int(r["n"]))
    p, a = _f(base[0]["discordance"]), _f(base[0]["right_only_share"])
    pl, pr = p * (1 - a), p * a
    truth = pr - pl
    out = []
    for r in base:
        n = int(r["n"])
        rng = np.random.default_rng([SEED, 21, index, n])
        counts = rng.multinomial(n, [pl, 1 - pl - pr, pr], size=resolution.SIM_DATASETS)
        tally = np.zeros((3, 6))  # covered, lower0 only, upper0 only, both0, miss with lower0, miss other
        for lo, _, ro in counts:
            for j, (low, high) in enumerate(resolution.intervals(n, int(lo), int(ro))):
                l0, h0 = abs(low) <= 1e-12, abs(high) <= 1e-12
                cov = low <= truth + 1e-12 and truth - 1e-12 <= high
                tally[j] += [cov, l0 and not h0, h0 and not l0, l0 and h0, (not cov) and l0, (not cov) and not l0]
        tally /= resolution.SIM_DATASETS
        for j, method in enumerate(("percentile", "bca", "wald")):
            out.append({"scenario": name, "n": n, "method": method, "true_diff_pp": 100 * truth, "coverage_v1": _f(r[method]),
                        "coverage_recomputed": tally[j, 0], "share_lower_end_at_zero_only": tally[j, 1],
                        "share_upper_end_at_zero_only": tally[j, 2], "share_both_ends_at_zero": tally[j, 3],
                        "miss_lower_end_at_zero": tally[j, 4], "miss_other": tally[j, 5]})
    return out


# ----------------------------------------------------------------------------------------------- exact-cell coverage
def intervals_seeded(n, left, right, seed):
    if seed == 0:
        return resolution.intervals(n, left, right)
    same = n - left - right
    rng = np.random.default_rng([SEED, 20, n, left, right, seed])
    draws = rng.multinomial(n, np.array([left, same, right]) / n, size=resolution.SIM_RESAMPLES)
    theta = (draws[:, 2] - draws[:, 0]) / n
    theta_hat = (right - left) / n
    percentile = tuple(np.quantile(theta, [0.025, 0.975]))
    bca = resolution.bca_bounds(theta_hat, theta, n, left, right)
    pd = (left + right) / n
    half = resolution.Z975 * math.sqrt(max(pd - theta_hat ** 2, 0.0) / n)
    return percentile, bca, (theta_hat - half, theta_hat + half)


def count_cells(n, pl, pr, mass=MASS):
    """(left, right, probability) for the multinomial discordant counts, enumerated to captured mass >= 1 - mass."""
    lmax = int(binom.ppf(1 - mass / 4, n, pl)) + 1 if pl > 0 else 0
    cells = []
    for lo in range(lmax + 1):
        p_lo = binom.pmf(lo, n, pl)
        if p_lo == 0:
            continue
        q = pr / (1 - pl) if pl < 1 else 0.0
        rmax = int(binom.ppf(1 - mass / 4, n - lo, q)) + 1 if q > 0 else 0
        ros = np.arange(rmax + 1)
        p_ro = binom.pmf(ros, n - lo, q)
        for ro, pp in zip(ros, p_ro):
            if pp * p_lo > 0:
                cells.append((lo, int(ro), float(p_lo * pp)))
    return cells


def exact_coverage(n, pl, pr, seeds=EXACT_SEEDS):
    truth = pr - pl
    cells = count_cells(n, pl, pr)
    captured = sum(c[2] for c in cells)
    cov = np.zeros((len(seeds), 3))
    for lo, ro, prob in cells:
        for i, s in enumerate(seeds):
            for j, (low, high) in enumerate(intervals_seeded(n, lo, ro, s)):
                cov[i, j] += prob * (low <= truth + 1e-12 and truth - 1e-12 <= high)
    return cov / captured, captured, len(cells)


def exact_cell_coverage(cov_rows):
    scen = {}
    for r in cov_rows:
        if _f(r["discordance"]) <= EXACT_MAX_DISCORDANCE + 1e-12:
            scen.setdefault(r["scenario"], (_f(r["discordance"]), _f(r["right_only_share"]), {}))[2][int(r["n"])] = r
    name, p = NULL_EXTRA
    scen[name] = (p, 0.5, {})
    out = []
    for name, (p, a, v1rows) in sorted(scen.items()):
        for n in resolution.NS:
            cov, captured, ncells = exact_coverage(n, p * (1 - a), p * a)
            for j, method in enumerate(("percentile", "bca", "wald")):
                row = {"scenario": name, "discordance": p, "right_only_share": a, "true_diff_pp": 100 * p * (2 * a - 1),
                       "n": n, "method": method, "cells": ncells, "mass_captured": captured,
                       "coverage_v1_montecarlo": _f(v1rows[n][method]) if n in v1rows else None}
                for i, s in enumerate(EXACT_SEEDS):
                    row[f"exact_seed{s}"] = float(cov[i, j])
                row["exact_mean_seeds"] = float(cov[:, j].mean())
                row["exact_range_seeds"] = float(cov[:, j].max() - cov[:, j].min())
                out.append(row)
    return out


# ----------------------------------------------------------------------------------------------- main
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out, v1=V1, exact=True):
    v1 = Path(ROOT) / v1 if not Path(v1).is_absolute() else Path(v1)
    w = Writer(out)
    sub = _rows(v1, "partB-subsample.csv")
    w.csv("partB-identical-contrasts.csv", identical_contrasts(sub))
    subs = subsets(sub)
    w.csv("partB-subsets.csv", subs)
    w.csv("partB-power-grid.csv", power_grid(subs))
    w.csv("partB-sufficiency.csv", sufficiency(sub))
    cov = _rows(v1, "partB-coverage.csv")
    w.csv("partB-knife-edge-endpoints.csv", knife_edge_endpoints(cov))
    if exact:
        w.csv("partB-coverage-exact-cells.csv", exact_cell_coverage(cov))
    code = Path(__file__).parent
    w.json("manifest-addendum-2.json", {
        "written": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": PROTOCOL, "protocol_sha256": _sha(Path(ROOT) / PROTOCOL),
        "addendum_1": ADDENDUM1, "addendum_1_sha256": _sha(Path(ROOT) / ADDENDUM1),
        "addendum_2": ADDENDUM2, "addendum_2_sha256": _sha(Path(ROOT) / ADDENDUM2),
        "seed": SEED, "evidence": "development evidence: ImageNet and COCO screen-1k lists only",
        "inputs_sha256": {p.name: _sha(p) for p in sorted(v1.glob("*.csv"))},
        "analysis_code_sha256": {p.name: _sha(p) for p in sorted(code.glob("*.py"))},
        "files": [p.name for p in w.files] + ["manifest-addendum-2.json"],
    })
    print(f"addendum 2: {len(w.files)} files in {w.out}", flush=True)
    return 0
