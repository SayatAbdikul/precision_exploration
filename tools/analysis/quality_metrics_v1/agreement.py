"""Part A: classifier agreement with FP32, flips, top-5 overlap, error overlap, McNemar, and resolution of pairs."""
from __future__ import annotations

import itertools

import numpy as np

from tools.analysis.b2_matrix import interval
from tools.analysis.b_stage_balanced_comparisons import paired_outcomes

FLOOR = 0.10  # pairs and resolution comparisons only between units with top-1 (expected) >= 10 percent


def per_image(unit, fp32):
    """Per-image agreement arrays of one unit against the FP32 unit of its network."""
    c0 = fp32["argmax_lowest"]
    k = unit["tie_size"].astype(np.float64)
    in_tie = unit["in_tie"]
    lowest = np.where(unit["argmax_lowest"] < 0, np.nan, (unit["argmax_lowest"] == c0).astype(np.float64))
    expected_lo = np.where(in_tie == 1, 1.0 / k, 0.0)
    expected_hi = np.where(in_tie == 0, 0.0, 1.0 / k)
    if "failed" in unit:
        lowest = np.where(unit["failed"], 0.0, lowest)
        expected_lo = np.where(unit["failed"], 0.0, expected_lo)
        expected_hi = np.where(unit["failed"], 0.0, expected_hi)
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(unit["top5"].tolist(), fp32["top5"].tolist())]) / 5.0
    in_top5 = (unit["top5"] == c0[:, None]).any(axis=1).astype(np.float64)
    return {"agree_lowest": lowest, "agree_expected_lo": expected_lo, "agree_expected_hi": expected_hi,
            "agree_topk": (unit["top5_topk"][:, 0] == c0).astype(np.float64), "top5_overlap": overlap,
            "fp32_top1_in_top5": in_top5, "undetermined": in_tie < 0}


def unit_row(unit, fp32, agree):
    q, f = unit["credit_lowest"].astype(np.int8), fp32["credit_lowest"].astype(np.int8)
    m = paired_outcomes(f, q)  # right minus left = quantized minus FP32
    ce, fe = unit["credit_expected"], fp32["credit_expected"]
    undetermined = int(agree["undetermined"].sum())
    lowest_known = ~np.isnan(agree["agree_lowest"])
    return {
        "source": unit["source"], "model": unit["model"], "group": unit["group"], "unit": unit["name"],
        "format": unit["format"], "recipe": unit["recipe"], "family": unit["family"], "bits": unit["bits"],
        "top1_expected": 100 * ce.mean(), "top1_lowest": 100 * q.mean(), "top1_topk": 100 * unit["credit_topk"].mean(),
        "fp32_top1": 100 * fe.mean(),
        "agree_expected_lo": 100 * agree["agree_expected_lo"].mean(), "agree_expected_hi": 100 * agree["agree_expected_hi"].mean(),
        "agree_undetermined_images": undetermined,
        "agree_lowest": 100 * np.nanmean(agree["agree_lowest"]) if lowest_known.all() else None,
        "agree_lowest_undetermined_images": int((~lowest_known).sum()),
        "agree_topk": 100 * agree["agree_topk"].mean(),
        "flip_correct_to_wrong": m["left_only_correct"], "flip_wrong_to_correct": m["right_only_correct"],
        "flip_expected_loss": float(np.maximum(0, fe - ce).sum()), "flip_expected_gain": float(np.maximum(0, ce - fe).sum()),
        "top5_overlap": 100 * agree["top5_overlap"].mean(), "fp32_top1_in_top5": 100 * agree["fp32_top1_in_top5"].mean(),
        "tied_top1_images": int((unit["tie_size"] > 1).sum()),
        "diff_lowest_vs_fp32_pp": m["difference_pp"], "diff_lowest_vs_fp32_ci95": m["pointwise_95_interval_pp"],
        "mcnemar_p_vs_fp32": m["mcnemar_exact_p"], "record": unit["record"],
    }


def pair_row(a, b, aa, ab):
    """b minus a for every statistic (a, b: units of one model and group; aa, ab: their agreement arrays)."""
    ea, eb = 1 - a["credit_lowest"], 1 - b["credit_lowest"]
    both, either = (ea * eb).sum(), np.maximum(ea, eb).sum()
    phi = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else None
    top1 = interval(b["credit_expected"], a["credit_expected"])
    row = {"model": a["model"], "source": a["source"], "group": a["group"], "a": a["name"], "b": b["name"],
           "top1_a": 100 * a["credit_expected"].mean(), "top1_b": 100 * b["credit_expected"].mean(),
           "error_jaccard": float(both / either) if either else None, "error_phi": phi,
           "top1_diff": top1[0], "top1_lo": top1[1], "top1_hi": top1[2]}
    mc = paired_outcomes(a["credit_lowest"].astype(np.int8), b["credit_lowest"].astype(np.int8))
    row.update(top1_lowest_diff=mc["difference_pp"], top1_lowest_mcnemar_p=mc["mcnemar_exact_p"],
               top1_discordant=mc["left_only_correct"] + mc["right_only_correct"])
    if not (aa["undetermined"].any() or ab["undetermined"].any()):
        agree = interval(ab["agree_expected_lo"], aa["agree_expected_lo"])
        row.update(agree_diff=agree[0], agree_lo=agree[1], agree_hi=agree[2])
    else:
        row.update(agree_diff=None, agree_lo=None, agree_hi=None)
    la, lb = aa["agree_lowest"], ab["agree_lowest"]
    if not (np.isnan(la).any() or np.isnan(lb).any()):
        ma = paired_outcomes(la.astype(np.int8), lb.astype(np.int8))
        row.update(agree_lowest_diff=ma["difference_pp"], agree_lowest_mcnemar_p=ma["mcnemar_exact_p"],
                   agree_discordant=ma["left_only_correct"] + ma["right_only_correct"])
    else:
        row.update(agree_lowest_diff=None, agree_lowest_mcnemar_p=None, agree_discordant=None)
    return row


def analyse(units):
    fp32 = {m: units[("matrix", m, "baseline", "fp32")] for m in {u["model"] for u in units.values()}}
    agree = {key: per_image(u, fp32[u["model"]]) for key, u in units.items()}
    unit_rows = [unit_row(u, fp32[u["model"]], agree[key]) for key, u in units.items()]
    groups = {}
    for key, u in units.items():
        if u["group"] == "baseline" or u["name"] == "fp32" or u["credit_expected"].mean() < FLOOR:
            continue
        groups.setdefault((u["source"], u["model"], u["group"]), []).append(key)
    pair_rows = []
    for _, keys in sorted(groups.items()):
        for ka, kb in itertools.combinations(sorted(keys), 2):
            row = pair_row(units[ka], units[kb], agree[ka], agree[kb])
            row["own_cells"] = "@" not in ka[3] and "@" not in kb[3]
            pair_rows.append(row)
    return unit_rows, pair_rows, agree


def resolution_summary(pair_rows):
    """Counts of pairs resolved by top-1 only / agreement only / both / neither, per source and group set."""
    out = []
    selections = {
        "matrix_own_cells": lambda r: r["source"] == "matrix" and r["own_cells"],
        "matrix_all_arms": lambda r: r["source"] == "matrix",
        "adaround": lambda r: r["source"] == "recon",
        "sweep": lambda r: r["source"] == "sweep",
    }
    for name, keep in selections.items():
        rows = [r for r in pair_rows if keep(r)]
        boot = [r for r in rows if r["agree_diff"] is not None]
        mc = [r for r in rows if r["agree_lowest_mcnemar_p"] is not None]
        def sep(lo, hi):
            return lo > 0 or hi < 0
        t = np.array([sep(r["top1_lo"], r["top1_hi"]) for r in boot])
        g = np.array([sep(r["agree_lo"], r["agree_hi"]) for r in boot])
        tm = np.array([r["top1_lowest_mcnemar_p"] < 0.05 for r in mc])
        gm = np.array([r["agree_lowest_mcnemar_p"] < 0.05 for r in mc])
        widths = np.array([(r["agree_hi"] - r["agree_lo"]) / (r["top1_hi"] - r["top1_lo"]) for r in boot
                           if r["top1_hi"] > r["top1_lo"]])
        sign = np.array([np.sign(r["agree_diff"]) == np.sign(r["top1_diff"]) for r in boot if r["agree_diff"] and r["top1_diff"]])
        both_sep = [r for r in boot if sep(r["agree_lo"], r["agree_hi"]) and sep(r["top1_lo"], r["top1_hi"])]
        out.append({
            "selection": name, "pairs_bootstrap": len(boot),
            "top1_only": int((t & ~g).sum()), "agreement_only": int((~t & g).sum()), "both": int((t & g).sum()),
            "neither": int((~t & ~g).sum()),
            "pairs_mcnemar": len(mc), "mcnemar_top1_only": int((tm & ~gm).sum()), "mcnemar_agreement_only": int((~tm & gm).sum()),
            "mcnemar_both": int((tm & gm).sum()), "mcnemar_neither": int((~tm & ~gm).sum()),
            "median_width_ratio_agreement_over_top1": float(np.median(widths)) if len(widths) else None,
            "median_top1_width": float(np.median([r["top1_hi"] - r["top1_lo"] for r in boot])) if boot else None,
            "median_agreement_width": float(np.median([r["agree_hi"] - r["agree_lo"] for r in boot])) if boot else None,
            "sign_agreement_share": float(sign.mean()) if len(sign) else None,
            "both_separated_opposite_sign": int(sum(np.sign(r["agree_diff"]) != np.sign(r["top1_diff"]) for r in both_sep)),
        })
    return out
