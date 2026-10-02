"""Lane Q2 analysis (CPU, read-only over sealed records): factorial decomposition, signedness, rank robustness.

Usage: .venv/bin/python -m tools.experiment_b2_rank.analysis [--out results/summaries/b2-rank-v1] [--partial DIR]
Statistics as fixed in public/experiments/configs/breadth-study/b2-rank-protocol-v1.json.  Development evidence
(ImageNet 1k screen).  Files are written once: an existing file with other content is never replaced.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import math
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from tools.analysis import b2_matrix as bm
from tools.experiment_b.common import ROOT, dataset, file_hash, formats, unseal
from tools.experiment_b2 import readout
from .plan import BLOCK, INTEGERS, INTRINSIC_CORNERS, MODELS, SCALAR_CORNERS, SEARCHED_CORNERS, recipe_for

RANK = ROOT / "artifacts/experiment_b2_rank"
RECON = ROOT / "artifacts/experiment_b2_recon/evals"
PROTOCOL = ROOT / "public/experiments/configs/breadth-study/b2-rank-protocol-v1.json"
DELTA = 1.0
CHANCE = 0.01
CORNER_VIEWS = {name: corner for corner, name in SCALAR_CORNERS.items()}
STRONG = ("default", "default_signed", "default_no_bias_correction", "default_weight_maxabs", "default_unsigned",
          "adaround_bc", "adaround_nobc")
WEAK = ("v1_maxabs", "v1_percentile_99_9")
ADAROUND = {"adaround_bc": "L-bc--w-{f}--a-{f}--default--fit-mse_per_channel-fp32in-s0",
            "adaround_nobc": "L-nobc--w-{f}--a-{f}--default_no_bias_correction--fit-mse_per_channel-fp32in-s0"}
ADAROUND_FORMATS = ("int8", "int6", "int4", "fp6_e2m3", "posit8_es1")


def width_class(bits):
    return "8" if bits == 8 else "7" if bits == 7 else "6" if bits == 6 else "le5"


def tier(view):
    if view.endswith("_searched"):
        return "searched"  # literal-recipe arm of the block formats: a sensitivity analysis (protocol)
    if view in WEAK:
        return "weak"
    return "strong" if view in STRONG else "other"


def norm(credits):
    return {"expected": credits["top1_expected"], "lowest_index": credits["top1_lowest_index"]}


def load_own(rows_sha):
    """Cells of this lane: ``(model, format, recipe, variant, part) -> credit arrays``."""
    out = {}
    for part in ("factorial", "unsigned", "searched"):
        for path in sorted((RANK / part / "cells").glob("*.json")):
            record = unseal(path)
            variant = "unsigned" if "--unsigned--" in path.name else "plain"
            file = ROOT / record["readout_file"]
            if file_hash(file) != record["readout_file_sha256"] or record["rows_sha256"] != rows_sha[record["model"]]:
                raise ValueError(f"readout drift or row mismatch: {path}")
            key = (record["model"], record["format"], record["recipe_name"], variant, part)
            if key in out:
                raise ValueError(f"duplicate cell {key}")
            arrays = readout.load(file)
            out[key] = {"arrays": arrays, "credit": norm(readout.credits(arrays)), "record": record,
                        "path": str(path.relative_to(ROOT))}
    return out


def load_adaround(rows_sha):
    out = {}
    for model in MODELS:
        for view, pattern in ADAROUND.items():
            for name in ADAROUND_FORMATS:
                stem = RECON / model / pattern.format(f=name)
                if not stem.with_suffix(".npz").exists():
                    continue
                record = unseal(stem.with_suffix(".json"))
                if record["rows_sha256"] != rows_sha[model] or record["images"] != 1000:
                    raise ValueError(f"AdaRound rows differ: {stem}")
                z = np.load(stem.with_suffix(".npz"))
                credit = z["label_among_maxima"].astype(np.float64) / z["tie_size"].astype(np.float64)
                if abs(100 * credit.mean() - record["readout"]["top1_percent_expected"]) > 1e-6:
                    raise ValueError(f"AdaRound credit does not reproduce its record: {stem}")
                lowest = z["label_is_lowest_index_maximum"].astype(np.float64)
                out[(model, name, view)] = {"credit": {"expected": credit, "lowest_index": lowest},
                                            "path": str(stem.relative_to(ROOT))}
    return out


def views(model, names, l1, own, ada, v1):
    """``view -> format -> (credit dict, source)`` for one model (see protocol, recipes_compared)."""
    out = {}

    def cell(name, recipe, variant="plain", part="factorial"):
        if variant == "plain" and (model, name, recipe) in l1:
            c = l1[(model, name, recipe)]
            return norm(c["credit"]), f"l1:{recipe}"
        c = own.get((model, name, recipe, variant, part))
        if c is None:
            return None
        return c["credit"], f"q2:{part}:{recipe}" + (":unsigned" if variant == "unsigned" else "")

    for view, (u, w, b) in CORNER_VIEWS.items():
        out[view] = {}
        for name in names:
            found = cell(name, recipe_for(name, u, w, b))
            if found:
                out[view][name] = found
    out["default_unsigned"] = {}
    for name in names:
        recipe = "cum5_act_maxabs" if name in BLOCK else "default"
        found = cell(name, recipe, "unsigned", "unsigned") or out["default"].get(name)
        if found:
            out["default_unsigned"][name] = found
    for (w, b), recipe in SEARCHED_CORNERS.items():
        view = f"{recipe}_searched"
        out[view] = {}
        for name in names:
            base = out[{(1, 1): "default", (1, 0): "default_no_bias_correction", (0, 1): "default_weight_maxabs",
                        (0, 0): "minimal"}[(w, b)]].get(name)
            found = (cell(name, recipe) or cell(name, recipe, part="searched")) if name in BLOCK else base
            if found:
                out[view][name] = found
    for view in ADAROUND:
        out[view] = {name: (ada[(model, name, view)]["credit"], "adaround:" + ada[(model, name, view)]["path"])
                     for name in names if (model, name, view) in ada}
    for view in WEAK:
        out[view] = {name: ({"expected": v1[(model, name, view)], "lowest_index": v1[(model, name, view)]}, "v1_sealed_topk")
                     for name in names if (model, name, view) in v1}
    return out


class Boot:
    """One paired image resampling per network (``b2_matrix.interval`` weights, seed 20260927)."""

    def __init__(self, n=1000):
        bm.interval(np.zeros(n), np.zeros(n))
        self.w = bm._W[n] / n

    def means(self, vectors):
        return self.w @ np.column_stack(vectors)


def tau_b(a, b):
    """Kendall tau-b along the last axis between two arrays of shape (..., F)."""
    i, j = np.triu_indices(a.shape[-1], 1)
    sa, sb = np.sign(a[..., i] - a[..., j]), np.sign(b[..., i] - b[..., j])
    den = np.sqrt((sa != 0).sum(-1) * (sb != 0).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (sa * sb).sum(-1) / np.where(den > 0, den, 1), np.nan)


def spearman(a, b):
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra, rb = ra - ra.mean(-1, keepdims=True), rb - rb.mean(-1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(-1) / np.where(den > 0, den, 1), np.nan)


def ci(samples):
    samples = samples[np.isfinite(samples)]
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])] if len(samples) else [None, None]


def diff(a, b):
    d, lo, hi = bm.interval(a, b)
    return d, lo, hi


def shapley(credit, factors):
    """Shapley contribution of each switch to f(all on) - f(all off); ``credit``: corner tuple -> vector."""
    k = len(factors)
    out = {}
    for index, name in enumerate(factors):
        plus, minus = np.zeros_like(next(iter(credit.values()))), np.zeros_like(next(iter(credit.values())))
        for order in itertools.permutations(range(k)):
            before = set(order[:order.index(index)])
            on = tuple(1 if (f in before) else 0 for f in range(k))
            on_with = tuple(1 if (f in before or f == index) else 0 for f in range(k))
            plus += credit[on_with] / math.factorial(k)
            minus += credit[on] / math.factorial(k)
        out[name] = (plus, minus)
    return out


def decomposition(model, names, entries, vw):
    rows = []
    for name in names:
        integer = name in INTEGERS
        corners = {}
        for view, (u, w, b) in CORNER_VIEWS.items():
            if name in vw[view]:
                corners[(u, w, b) if integer else (w, b)] = vw[view][name][0]["expected"]
        need = 8 if integer else 4
        if len(set(corners)) < need:
            rows.append({"model": model, "format": name, "complete": False})
            continue
        top, bottom = (corners[(1, 1, 1)], corners[(0, 0, 0)]) if integer else (corners[(1, 1)], corners[(0, 0)])
        total = diff(top, bottom)
        row = {"model": model, "format": name, "bits": int(entries[name]["bits"]), "complete": True,
               "default": 100 * top.mean(), "minimal": 100 * bottom.mean(),
               "default_minus_minimal": total[0], "dm_low": total[1], "dm_high": total[2]}
        factors = ("unsigned", "weight_mse", "bias_correction") if integer else ("weight_mse", "bias_correction")
        for factor, (plus, minus) in shapley(corners, factors).items():
            d, lo, hi = diff(plus, minus)
            row.update({f"shapley_{factor}": d, f"shapley_{factor}_low": lo, f"shapley_{factor}_high": hi})
        loo = {"unsigned": "default_signed", "weight_mse": "default_weight_maxabs",
               "bias_correction": "default_no_bias_correction"}
        for factor in factors:
            d, lo, hi = diff(top, vw[loo[factor]][name][0]["expected"])
            row.update({f"loo_{factor}": d, f"loo_{factor}_low": lo, f"loo_{factor}_high": hi})
        rows.append(row)
    return rows


def correlations(model, names, entries, vw, boot):
    rows, view_names = [], [v for v in vw if vw[v]]
    means = {v: (np.array([vw[v][n][0]["expected"].mean() for n in names if n in vw[v]]),
                 boot.means([vw[v][n][0]["expected"] for n in names if n in vw[v]]),
                 [n for n in names if n in vw[v]]) for v in view_names}
    for cls in ("8", "6", "le5", "all"):
        members = [n for n in names if cls == "all" or width_class(int(entries[n]["bits"])) == cls]
        for v1, v2 in itertools.combinations(view_names, 2):
            common = [n for n in members if n in vw[v1] and n in vw[v2]]
            for subset in ("all", "above_chance"):
                use = common if subset == "all" else [n for n in common if max(
                    vw[v1][n][0]["expected"].mean(), vw[v2][n][0]["expected"].mean()) >= CHANCE]
                if len(use) < 3:
                    continue
                i1 = [means[v1][2].index(n) for n in use]
                i2 = [means[v2][2].index(n) for n in use]
                p1, p2 = means[v1][0][i1], means[v2][0][i2]
                b1, b2 = means[v1][1][:, i1], means[v2][1][:, i2]
                rows.append({"model": model, "class": cls, "view_a": v1, "view_b": v2, "tier_a": tier(v1),
                             "tier_b": tier(v2), "subset": subset, "formats": len(use),
                             "kendall_tau_b": float(tau_b(p1, p2)), "kendall_low": ci(tau_b(b1, b2))[0],
                             "kendall_high": ci(tau_b(b1, b2))[1], "spearman": float(spearman(p1, p2)),
                             "spearman_low": ci(spearman(b1, b2))[0], "spearman_high": ci(spearman(b1, b2))[1]})
    return rows


def pairwise(model, names, entries, vw):
    """Every same-width-class pair under every view: difference X - Y with its interval."""
    diffs = {}
    for cls in ("8", "6", "le5"):
        members = [n for n in names if width_class(int(entries[n]["bits"])) == cls]
        for x, y in itertools.combinations(members, 2):
            seen = set()
            for view, content in vw.items():
                if x in content and y in content:
                    # a view whose cells for X and Y are the same records as an earlier view's (e.g. default_signed
                    # for two non-integers, or a *_searched view for two scalar formats) is the same comparison
                    sources = (content[x][1], content[y][1])
                    if sources in seen:
                        continue
                    seen.add(sources)
                    a, b = content[x][0]["expected"], content[y][0]["expected"]
                    d, lo, hi = diff(a, b)
                    diffs[(cls, x, y, view)] = (d, lo, hi, a.mean() < CHANCE or b.mean() < CHANCE)
    return diffs


def reversals(model, diffs):
    rows = []
    keys = sorted({(c, x, y) for c, x, y, _ in diffs})
    for cls, x, y in keys:
        present = {v: diffs[(cls, x, y, v)] for (c, a, b, v) in diffs if (c, a, b) == (cls, x, y)}
        for v1, v2 in itertools.combinations(sorted(present), 2):
            (d1, l1, h1, c1), (d2, l2, h2, c2) = present[v1], present[v2]
            point = (d1 > DELTA and d2 < -DELTA) or (d1 < -DELTA and d2 > DELTA)
            interval = (l1 > DELTA and h2 < -DELTA) or (h1 < -DELTA and l2 > DELTA)
            if not point:
                continue
            t = {tier(v1), tier(v2)}
            category = ("strong_vs_strong" if t == {"strong"} else "involves_weak" if "weak" in t
                        else "involves_searched_sensitivity" if "searched" in t
                        else "strong_vs_other" if "strong" in t else "other_vs_other")
            rows.append({"model": model, "class": cls, "x": x, "y": y, "view_a": v1, "view_b": v2,
                         "d_a": d1, "d_a_low": l1, "d_a_high": h1, "d_b": d2, "d_b_low": l2, "d_b_high": h2,
                         "point_reversal": point, "interval_reversal": interval, "chance_pair": bool(c1 or c2),
                         "category": category})
    return rows


PAIR_CLASSES = ("strong_vs_strong", "strong_vs_other", "other_vs_other", "involves_weak",
                "involves_searched_sensitivity")


def pair_summary(revs):
    """One row per (model, class, X, Y) that reverses anywhere: where the reversal lives (point / interval).

    ``verdict`` (interval level, non-chance views only): ``survives_among_strong`` when two strong recipes reverse
    the pair; ``minimal_side`` when only a minimal-side corner against a strong or another corner does;
    ``maxabs_percentile_only`` when only a weak recipe (sealed v1 max-abs / percentile) does; ``searched_only`` when
    only the block-format searched sensitivity arm does; ``point_only`` when no interval reversal exists.
    """
    rows = {}
    for r in revs:
        key = (r["model"], r["class"], r["x"], r["y"])
        row = rows.setdefault(key, {"model": key[0], "class": key[1], "x": key[2], "y": key[3],
                                    **{f"point_{c}": 0 for c in PAIR_CLASSES},
                                    **{f"interval_{c}": 0 for c in PAIR_CLASSES},
                                    **{f"interval_nonchance_{c}": 0 for c in PAIR_CLASSES}})
        row[f"point_{r['category']}"] += 1
        if r["interval_reversal"]:
            row[f"interval_{r['category']}"] += 1
            if not r["chance_pair"]:
                row[f"interval_nonchance_{r['category']}"] += 1
    for row in rows.values():
        n = {c: row[f"interval_nonchance_{c}"] for c in PAIR_CLASSES}
        row["verdict"] = ("survives_among_strong" if n["strong_vs_strong"] else
                          "minimal_side" if n["strong_vs_other"] or n["other_vs_other"] else
                          "maxabs_percentile_only" if n["involves_weak"] else
                          "searched_only" if n["involves_searched_sensitivity"] else
                          "interval_only_with_chance_format" if any(row[f"interval_{c}"] for c in PAIR_CLASSES)
                          else "point_only")
    return list(rows.values())


RUNS = ROOT / "artifacts/experiment_b2/runs"


def runs_crosscheck(own):
    """Earlier B2 runner records (artifacts/experiment_b2/runs/*, no per-image readout) with the same configuration
    identity as one of our cells: their top-1 (sealed top-k rule, 0.1 pp) must equal our cell's top-1 top-k."""
    rows = []
    for (model, name, recipe, variant, part), c in sorted(own.items()):
        if variant != "plain":
            continue
        identity = c["record"]["configuration_sha256"]
        for path in sorted(RUNS.glob(f"*/{model}--{name}--{recipe}--1000--{identity[:12]}.json")):
            run = unseal(path)
            mine = c["record"]["readout"]["top1_topk_percent"]
            rows.append({"model": model, "format": name, "recipe": recipe, "run": str(path.relative_to(ROOT)),
                         "identity_equal": run["configuration_sha256"] == identity,
                         "run_top1_percent": run["metrics"]["top1_percent"], "cell_top1_topk_percent": round(mine, 4),
                         "equal": run["configuration_sha256"] == identity
                         and abs(run["metrics"]["top1_percent"] - mine) < 0.051})
    return rows


def write_once(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"refusing to replace {path} (write a new versioned set)")
        return
    path.write_text(text)


def to_csv(rows, digits=4):
    if not rows:
        return ""
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (round(v, digits) if isinstance(v, float) else v) for k, v in row.items()})
    return buffer.getvalue()


def signedness(model, names, entries, vw, l1, own):
    """Unsigned variant minus default per format; default minus default_signed for integers; INT vs non-INT gaps."""
    rows = []
    for name in names:
        if name in vw["default_unsigned"] and name in vw["default"]:
            u, s = vw["default_unsigned"][name], vw["default"][name]
            if u[1] == s[1] and name not in INTEGERS:
                continue
            if name in INTEGERS:
                u, s = vw["default"][name], vw["default_signed"].get(name)
                if s is None:
                    continue
            d, lo, hi = diff(u[0]["expected"], s[0]["expected"])
            rows.append({"model": model, "format": name, "family": entries[name]["family"],
                         "bits": int(entries[name]["bits"]), "comparison": "unsigned_minus_signed_under_default",
                         "unsigned": 100 * u[0]["expected"].mean(), "signed": 100 * s[0]["expected"].mean(),
                         "d": d, "low": lo, "high": hi, "unsigned_source": u[1], "signed_source": s[1]})
    gaps = []
    for bits, integer in ((8, "int8"), (6, "int6"), (5, "int5"), (4, "int4")):
        for name in names:
            if int(entries[name]["bits"]) != bits or name == integer or name not in vw["default_unsigned"]:
                continue
            for label, iv, fv in (("signed_vs_signed", "default_signed", "default"),
                                  ("unsigned_int_vs_signed_other (default as frozen)", "default", "default"),
                                  ("unsigned_vs_unsigned", "default", "default_unsigned")):
                if integer in vw[iv] and name in vw[fv]:
                    d, lo, hi = diff(vw[fv][name][0]["expected"], vw[iv][integer][0]["expected"])
                    gaps.append({"model": model, "bits": bits, "format": name, "integer": integer,
                                 "comparison": label, "format_minus_integer": d, "low": lo, "high": hi})
    return rows, gaps


def q16_alias(l1, own):
    out = {}
    for model in MODELS:
        mine = own.get((model, "int8", "default_signed", "plain", "factorial"))
        theirs = l1.get((model, "q1_6", "default"))
        if mine is None or theirs is None:
            out[model] = None
            continue
        out[model] = all(np.array_equal(mine["arrays"][k], theirs["arrays"][k]) for k in readout.DTYPES)
    return out


def proposal_report(all_views, entries, names, fp32):
    """Rule (d) re-applied with every format at one view (report only)."""
    saved = dict(bm.INTRINSIC)
    reference = list(csv.DictReader((ROOT / "results/summaries/b2-matrix-v1/proposal-d4.csv").open()))
    ref = [(r["role"], r["format"]) for r in reference]
    rows = []
    try:
        for view in [v for v in all_views[MODELS[0]] if tier(v) != "weak" and not v.startswith("adaround")]:
            bm.INTRINSIC.clear()
            bm.INTRINSIC.update({view: view})
            credit = {(m, "fp32", "baseline"): fp32[m] for m in MODELS}
            for m in MODELS:
                for name, (c, _) in all_views[m][view].items():
                    credit[(m, name, view)] = c["expected"]
            picks = {name: view for name in names}
            result = bm.propose(credit, entries, MODELS, names, picks, ("q1_6",))
            mine = [("anchor", "fp32"), ("anchor", "int8")] + [("selected", c["format"]) for c in result["selected"]] \
                + [("stress", c["format"]) for c in result["stress"]]
            rows.append({"view": view, "entries": len(mine), "passing": result["passing_candidates"],
                         "selected": " ".join(c["format"] for c in result["selected"]),
                         "stress": " ".join(c["format"] for c in result["stress"]),
                         "added_vs_proposal": " ".join(f"{r}:{f}" for r, f in mine if (r, f) not in ref),
                         "dropped_vs_proposal": " ".join(f"{r}:{f}" for r, f in ref if (r, f) not in mine),
                         "identical_to_proposal": sorted(mine) == sorted(ref)})
    finally:
        bm.INTRINSIC.clear()
        bm.INTRINSIC.update(saved)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/summaries/b2-rank-v1")
    parser.add_argument("--partial", action="store_true", help="allow missing cells (development aid)")
    args = parser.parse_args()
    out = ROOT / args.out
    entries = {row["name"]: row for row in formats()}
    names = [row["name"] for row in formats()]
    l1 = bm.load_cells()
    rows_sha = {m: l1[(m, "fp32", "baseline")]["record"]["rows_sha256"] for m in MODELS}
    own = load_own(rows_sha)
    ada = load_adaround(rows_sha)
    rows1k = dataset("imagenet_screen_1k")[1]
    v1 = bm.v1_cells(rows1k, MODELS)
    boot = Boot()
    fp32 = {m: l1[(m, "fp32", "baseline")]["credit"]["top1_expected"] for m in MODELS}
    all_views, decomp, corr, revs, sign_rows, gap_rows, cell_rows = {}, [], [], [], [], [], []
    comparisons = {"format_pairs": 0, "recipe_pair_tests": 0}
    for model in MODELS:
        vw = views(model, names, l1, own, ada, v1)
        all_views[model] = vw
        for view, content in vw.items():
            for name, (c, source) in content.items():
                cell_rows.append({"model": model, "view": view, "tier": tier(view), "format": name,
                                  "bits": int(entries[name]["bits"]), "class": width_class(int(entries[name]["bits"])),
                                  "top1_expected": 100 * c["expected"].mean(),
                                  "top1_lowest_index": 100 * c["lowest_index"].mean(), "source": source})
        decomp += decomposition(model, names, entries, vw)
        corr += correlations(model, names, entries, vw, boot)
        pw = pairwise(model, names, entries, vw)
        revs += reversals(model, pw)
        for key in {(c, x, y) for c, x, y, _ in pw}:
            k = sum(1 for (c, x, y, _) in pw if (c, x, y) == key)
            comparisons["format_pairs"] += 1
            comparisons["recipe_pair_tests"] += k * (k - 1) // 2
        s, g = signedness(model, names, entries, vw, l1, own)
        sign_rows += s
        gap_rows += g
    incomplete = [r for r in decomp if not r["complete"]]
    if incomplete and not args.partial:
        raise SystemExit(f"{len(incomplete)} incomplete factorials, e.g. {incomplete[:3]} (use --partial)")
    proposals = proposal_report(all_views, entries, names, fp32)
    robust = [r for r in revs if r["interval_reversal"] and not r["chance_pair"]]
    pairs = pair_summary(revs)
    crosscheck = runs_crosscheck(own)
    summary = {"protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": file_hash(PROTOCOL)},
               "evidence": "development evidence, ImageNet 1k screen", "partial": bool(args.partial),
               "own_cells": len(own), "adaround_arms": len(ada), "q1_6_alias_int8_default_signed": q16_alias(l1, own),
               "reversal_counts": {cat: {"point": sum(r["category"] == cat for r in revs),
                                         "interval": sum(r["category"] == cat and r["interval_reversal"] for r in revs),
                                         "interval_non_chance": sum(r["category"] == cat for r in robust)}
                                   for cat in PAIR_CLASSES},
               "pair_verdicts": {v: sum(p["verdict"] == v for p in pairs) for v in sorted({p["verdict"] for p in pairs})},
               "comparisons": comparisons,
               "runs_crosscheck": {"cells": len(crosscheck), "equal": sum(r["equal"] for r in crosscheck)},
               "robust_strong_reversals": sorted({(r["model"], r["class"], r["x"], r["y"]) for r in robust
                                                  if r["category"] == "strong_vs_strong"}),
               "proposal_identical_under": [r["view"] for r in proposals if r["identical_to_proposal"]],
               "sources": {"tools/experiment_b2_rank/analysis.py": file_hash(Path(__file__))}}
    files = {"cells.csv": to_csv(cell_rows), "decomposition.csv": to_csv(decomp), "rank-correlation.csv": to_csv(corr),
             "reversals.csv": to_csv(revs), "signedness.csv": to_csv(sign_rows), "int-vs-format-gaps.csv": to_csv(gap_rows),
             "proposal-by-view.csv": to_csv(proposals),
             "pair-reversals.csv": to_csv(pairs), "runs-crosscheck.csv": to_csv(crosscheck), "summary.json": json.dumps(summary, indent=1, default=str) + "\n"}
    for name, text in files.items():
        write_once(out / name, text)
    print(json.dumps({k: v for k, v in summary.items() if k != "robust_strong_reversals"}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
