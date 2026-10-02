"""Lane Q2, protocol addendum-2 (review 1 findings A and B): tie-tolerant rank statistics and reversal switches.

Usage: OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python -m tools.experiment_b2_rank.addendum2 [--out DIR] [--skip-tau]
Rule (public/experiments/configs/breadth-study/b2-rank-protocol-v1-addendum-2.json): mean credits (point and each
bootstrap resample mean) are rounded to 10 decimals before Kendall tau-b / Spearman, so formats whose means are equal
up to floating-point summation order are exact ties.  Reversal switches are derived from the sealed reversals.csv.
Development evidence (ImageNet 1k screen).  Files are written once (analysis.write_once).
"""
from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, file_hash
from . import analysis as an

DECIMALS = 10
ADDENDUM = ROOT / "public/experiments/configs/breadth-study/b2-rank-protocol-v1-addendum-2.json"

# Recipe switches of each strong view (U: unsigned integer codes, W: MSE weight range, B: bias correction,
# S: unsigned sibling codebook for non-integers, R: AdaRound rounding).
SWITCHES = {"default": dict(U=1, W=1, B=1, S=0, R=0), "default_signed": dict(U=0, W=1, B=1, S=0, R=0),
            "default_no_bias_correction": dict(U=1, W=1, B=0, S=0, R=0),
            "default_weight_maxabs": dict(U=1, W=0, B=1, S=0, R=0),
            "default_unsigned": dict(U=1, W=1, B=1, S=1, R=0),
            "adaround_bc": dict(U=1, W=1, B=1, S=0, R=1), "adaround_nobc": dict(U=1, W=1, B=0, S=0, R=1)}


def switched(view_a, view_b):
    """Switches that differ between two strong views, as a sorted string such as 'B+W'."""
    a, b = SWITCHES[view_a], SWITCHES[view_b]
    return "+".join(k for k in "UWBSR" if a[k] != b[k])


def tol(x):
    return np.round(np.asarray(x, dtype=float), DECIMALS)


@contextmanager
def tie_tolerant():
    """Inside this process only: analysis.tau_b / spearman see means rounded to DECIMALS."""
    tau, rho = an.tau_b, an.spearman
    an.tau_b = lambda a, b: tau(tol(a), tol(b))
    an.spearman = lambda a, b: rho(tol(a), tol(b))
    try:
        yield
    finally:
        an.tau_b, an.spearman = tau, rho


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def switch_rows(reversals):
    rows = []
    for r in reversals:
        if r["category"] != "strong_vs_strong" or r["interval_reversal"] != "True" or r["chance_pair"] != "False":
            continue
        rows.append({"model": r["model"], "class": r["class"], "x": r["x"], "y": r["y"], "view_a": r["view_a"],
                     "view_b": r["view_b"], "switches": switched(r["view_a"], r["view_b"]),
                     "involves_no_bias_correction": "default_no_bias_correction" in (r["view_a"], r["view_b"]),
                     "d_a": float(r["d_a"]), "d_b": float(r["d_b"])})
    return rows


def switch_summary(rows):
    pairs, nobc = {}, set()
    for r in rows:
        key = (r["model"], r["class"], r["x"], r["y"])
        pairs.setdefault(key, set()).add(r["switches"])
        if r["involves_no_bias_correction"]:
            nobc.add(key)
    by_switch = {}
    for r in rows:
        by_switch[r["switches"]] = by_switch.get(r["switches"], 0) + 1
    single = {s for s in SWITCHES["default"]}
    return {"rows": len(rows), "rows_with_no_bias_correction": sum(r["involves_no_bias_correction"] for r in rows),
            "rows_by_switches": dict(sorted(by_switch.items())), "pairs": len(pairs),
            "pairs_with_no_bias_correction_row": len(nobc),
            "pairs_reversing_under_single_switch": {
                s: sorted("/".join(k) for k, v in pairs.items() if s in v) for s in sorted(single)
                if any(s in v for v in pairs.values())},
            "pairs_only_under_combined_switches": sorted("/".join(k) + ":" + ",".join(sorted(v))
                                                         for k, v in pairs.items() if not v & single)}


def tau_rows(out_dir):
    """Recompute the correlation table with the tie rule; the v1 value is put beside each row."""
    entries = {row["name"]: row for row in an.formats()}
    names = [row["name"] for row in an.formats()]
    l1 = an.bm.load_cells()
    rows_sha = {m: l1[(m, "fp32", "baseline")]["record"]["rows_sha256"] for m in an.MODELS}
    own, ada = an.load_own(rows_sha), an.load_adaround(rows_sha)
    v1 = an.bm.v1_cells(an.dataset("imagenet_screen_1k")[1], an.MODELS)
    boot = an.Boot()
    old = {(r["model"], r["class"], r["view_a"], r["view_b"], r["subset"]): r
           for r in read_csv(ROOT / "results/summaries/b2-rank-v1/rank-correlation.csv")}
    rows = []
    with tie_tolerant():
        for model in an.MODELS:
            vw = an.views(model, names, l1, own, ada, v1)
            for r in an.correlations(model, names, entries, vw, boot):
                o = old.get((r["model"], r["class"], r["view_a"], r["view_b"], r["subset"]))
                r["v1_kendall_tau_b"] = float(o["kendall_tau_b"]) if o else None
                r["v1_spearman"] = float(o["spearman"]) if o else None
                rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/summaries/b2-rank-v1")
    parser.add_argument("--skip-tau", action="store_true", help="only the switch classification (development aid)")
    args = parser.parse_args()
    out = ROOT / args.out
    srows = switch_rows(read_csv(ROOT / "results/summaries/b2-rank-v1/reversals.csv"))
    summary = {"addendum": {"path": str(ADDENDUM.relative_to(ROOT)), "sha256": file_hash(ADDENDUM)},
               "evidence": "development evidence, ImageNet 1k screen", "reversal_switches": switch_summary(srows),
               "sources": {"tools/experiment_b2_rank/addendum2.py": file_hash(Path(__file__)),
                           "tools/experiment_b2_rank/analysis.py": file_hash(Path(an.__file__))}}
    files = {"reversal-switches-v2.csv": an.to_csv(srows)}
    if not args.skip_tau:
        trows = tau_rows(out)
        changed = [r for r in trows if r["v1_kendall_tau_b"] is not None
                   and abs(round(r["kendall_tau_b"], 4) - r["v1_kendall_tau_b"]) > 0.005]
        summary["tie_tolerant_tau"] = {"rows": len(trows), "decimals": DECIMALS,
                                       "rows_changed_by_more_than_0_005": len(changed),
                                       "max_abs_change": max((abs(round(r["kendall_tau_b"], 4) - r["v1_kendall_tau_b"])
                                                              for r in trows if r["v1_kendall_tau_b"] is not None),
                                                             default=None)}
        files["rank-correlation-tol-v2.csv"] = an.to_csv(trows)
    files["addendum2-summary.json"] = json.dumps(summary, indent=1, default=str) + "\n"
    for name, text in files.items():
        an.write_once(out / name, text)
    print(json.dumps(summary, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
