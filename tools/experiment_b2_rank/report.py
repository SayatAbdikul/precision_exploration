"""Markdown tables for docs/analysis/b2-ranking-robustness-2026-10-02.md from results/summaries/b2-rank-v1 (stdout).

  .venv/bin/python -m tools.experiment_b2_rank.report [--summary DIR] [--section NAME]
"""
from __future__ import annotations

import argparse
import csv
import json

from tools.experiment_b.common import ROOT
from .plan import MODELS

NICE = {"resnet18": "ResNet18", "mobilenet_v2": "MNV2", "mobilenet_v3_large": "MNV3-L"}
PAIRS = (("default", "minimal"), ("default", "v1_maxabs"), ("default", "v1_percentile_99_9"),
         ("minimal", "v1_maxabs"), ("default", "default_no_bias_correction"), ("default", "default_weight_maxabs"),
         ("default", "default_signed"), ("default", "default_unsigned"), ("default_no_bias_correction", "default_weight_maxabs"),
         ("default", "default_searched"), ("default", "adaround_bc"), ("adaround_bc", "adaround_nobc"))


def rows(summary, name):
    return list(csv.DictReader((summary / name).open()))


def f(x, digits=1):
    return "" if x in (None, "") else f"{float(x):+.{digits}f}" if digits == 1 else f"{float(x):.{digits}f}"


def iv(r, d, lo, hi, digits=1):
    return f"{f(r[d], digits)} [{f(r[lo], digits)}, {f(r[hi], digits)}]" if r.get(lo) not in (None, "") else f(r[d], digits)


def tau(summary, cls="all", subset="all"):
    corr = rows(summary, "rank-correlation.csv")
    out = [f"| recipe A | recipe B | " + " | ".join(f"{NICE[m]} tau-b [95%] (n)" for m in MODELS) + " |",
           "|---|---|" + "---|" * len(MODELS)]
    for a, b in PAIRS:
        cells = []
        for m in MODELS:
            r = next((r for r in corr if r["model"] == m and r["class"] == cls and r["subset"] == subset
                      and {r["view_a"], r["view_b"]} == {a, b}), None)
            cells.append("" if r is None else f"{float(r['kendall_tau_b']):.2f} [{float(r['kendall_low']):.2f}, "
                         f"{float(r['kendall_high']):.2f}] ({r['formats']})")
        out.append(f"| {a} | {b} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def decomposition(summary):
    out = ["| model | format | bits | default | minimal | default - minimal [95%] | U (unsigned) | W (MSE weights) | "
           "B (bias corr.) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows(summary, "decomposition.csv"):
        if r["complete"] != "True":
            out.append(f"| {NICE[r['model']]} | {r['format']} | | incomplete | | | | | |")
            continue
        if max(float(r["default"]), float(r["minimal"])) < 1.0:
            continue
        u = iv(r, "shapley_unsigned", "shapley_unsigned_low", "shapley_unsigned_high") if r.get("shapley_unsigned") else "n/a"
        out.append(f"| {NICE[r['model']]} | {r['format']} | {r['bits']} | {float(r['default']):.1f} | "
                   f"{float(r['minimal']):.1f} | {iv(r, 'default_minus_minimal', 'dm_low', 'dm_high')} | {u} | "
                   f"{iv(r, 'shapley_weight_mse', 'shapley_weight_mse_low', 'shapley_weight_mse_high')} | "
                   f"{iv(r, 'shapley_bias_correction', 'shapley_bias_correction_low', 'shapley_bias_correction_high')} |")
    return "\n".join(out)


def strong_reversals(summary):
    revs = [r for r in rows(summary, "reversals.csv") if r["category"] == "strong_vs_strong"]
    out = ["| model | class | X | Y | recipe A: X - Y [95%] | recipe B: X - Y [95%] | interval | chance |",
           "|---|---|---|---|---|---|---|---|"]
    for r in revs:
        out.append(f"| {NICE[r['model']]} | {r['class']} | {r['x']} | {r['y']} | {r['view_a']}: "
                   f"{iv(r, 'd_a', 'd_a_low', 'd_a_high')} | {r['view_b']}: {iv(r, 'd_b', 'd_b_low', 'd_b_high')} | "
                   f"{r['interval_reversal']} | {r['chance_pair']} |")
    return "\n".join(out)


def verdicts(summary):
    pairs = rows(summary, "pair-reversals.csv")
    kinds = sorted({p["verdict"] for p in pairs})
    out = ["| model | class | " + " | ".join(kinds) + " |", "|---|---|" + "---|" * len(kinds)]
    for m in MODELS:
        for cls in ("8", "6", "le5"):
            sub = [p for p in pairs if p["model"] == m and p["class"] == cls]
            out.append(f"| {NICE[m]} | {cls} | " + " | ".join(str(sum(p["verdict"] == k for p in sub)) for k in kinds) + " |")
    lists = []
    for k in ("survives_among_strong", "minimal_side", "maxabs_percentile_only"):
        names = [f"{NICE[p['model']]} {p['x']}/{p['y']}" for p in pairs if p["verdict"] == k]
        lists.append(f"- **{k}** ({len(names)}): " + ("; ".join(names) if names else "none"))
    return "\n".join(out) + "\n\n" + "\n".join(lists)


def signedness(summary):
    out = ["| model | format | bits | unsigned | signed | unsigned - signed [95%] |", "|---|---|---|---|---|---|"]
    for r in rows(summary, "signedness.csv"):
        out.append(f"| {NICE[r['model']]} | {r['format']} | {r['bits']} | {float(r['unsigned']):.1f} | "
                   f"{float(r['signed']):.1f} | {iv(r, 'd', 'low', 'high')} |")
    return "\n".join(out)


def gaps(summary):
    out = ["| model | bits | format | vs | signed vs signed | unsigned INT vs signed format (frozen default) | "
           "unsigned vs unsigned |", "|---|---|---|---|---|---|---|"]
    table = {}
    for r in rows(summary, "int-vs-format-gaps.csv"):
        table.setdefault((r["model"], r["bits"], r["format"], r["integer"]), {})[r["comparison"]] = r
    for (m, bits, name, integer), c in table.items():
        cells = [iv(c[k], "format_minus_integer", "low", "high") if k in c else ""
                 for k in ("signed_vs_signed", "unsigned_int_vs_signed_other (default as frozen)", "unsigned_vs_unsigned")]
        out.append(f"| {NICE[m]} | {bits} | {name} | {integer} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def proposal(summary):
    out = ["| view | entries | passing | selected | stress | added | dropped | identical |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows(summary, "proposal-by-view.csv"):
        out.append(f"| {r['view']} | {r['entries']} | {r['passing']} | {r['selected']} | {r['stress']} | "
                   f"{r['added_vs_proposal']} | {r['dropped_vs_proposal']} | {r['identical_to_proposal']} |")
    return "\n".join(out)


SECTIONS = {"tau_all": lambda s: tau(s), "tau_8": lambda s: tau(s, "8"), "tau_6": lambda s: tau(s, "6"),
            "tau_le5": lambda s: tau(s, "le5"), "tau_all_above": lambda s: tau(s, "all", "above_chance"),
            "decomposition": decomposition, "verdicts": verdicts, "strong_reversals": strong_reversals,
            "signedness": signedness, "gaps": gaps, "proposal": proposal,
            "summary": lambda s: json.dumps(json.loads((s / "summary.json").read_text()), indent=1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summaries/b2-rank-v1")
    parser.add_argument("--section", choices=sorted(SECTIONS), action="append")
    args = parser.parse_args()
    summary = ROOT / args.summary
    for name in args.section or SECTIONS:
        print(f"\n<!-- {name} -->\n{SECTIONS[name](summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
