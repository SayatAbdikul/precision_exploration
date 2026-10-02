"""Part E: format ranking at a fixed accumulator register width (ResNet18, lane L8 sweep, sat.w<W> only)."""
from __future__ import annotations

import csv
import itertools

import numpy as np
from scipy.stats import kendalltau

from tools.analysis.b2_matrix import interval
from tools.experiment_b.common import ROOT

WIDTHS = tuple(range(16, 33))
LABEL = {"resnet18-int8-default-b2": "INT8", "resnet18-int8-default_signed-b2": "INT8 signed act.",
         "resnet18-int6-default-b2": "INT6", "resnet18-fp6_e2m3-default-b2": "FP6 E2M3",
         "resnet18-fp7_e3m3-default-b2": "FP7 E3M3", "resnet18-fp8_e4m3fn-default-b2": "FP8 E4M3",
         "resnet18-posit8_es1-default-b2": "Posit8 es1", "resnet18-fp8_e5m2-default-b2": "FP8 E5M2"}


def noevent_widths():
    rows = list(csv.DictReader((ROOT / "results/summaries/accumulator-sweep-v1/widths.csv").open()))
    out = {}
    for r in rows:
        v = r["W_noevent"]
        out[r["case"]] = int(v) if v and v.lstrip("-").isdigit() else None
    return out


def grid(units):
    """cell[(case, W)] = dict(status, credit vector or None, top1)."""
    noevent = noevent_widths()
    cells = {}
    for case in LABEL:
        wide = units[("sweep", "resnet18", case, "wide")]
        measured = {int(k[3][5:]): u for k, u in units.items() if k[0] == "sweep" and k[2] == case and k[3].startswith("sat.w")}
        lowest = min(measured) if measured else None
        for w in WIDTHS:
            if w in measured:
                u = measured[w]
                status = "measured"
                credit = u["credit_expected"]
                if noevent.get(case) is not None and w >= noevent[case] and not np.array_equal(credit, wide["credit_expected"]):
                    raise ValueError(f"event-free width differs from exact: {case} W={w}")
            elif noevent.get(case) is not None and w >= noevent[case]:
                status, credit = "exact_by_no_event", wide["credit_expected"]
            else:
                status, credit = "collapsed_not_measured", None
                if lowest is None or w > lowest:
                    raise ValueError(f"unmeasured width inside the bracket: {case} W={w}")
            cell = {"case": case, "format": LABEL[case], "W": w, "status": status, "credit": credit,
                    "wide_top1": 100 * wide["credit_expected"].mean(),
                    "narrowest_measured_W": lowest,
                    "narrowest_measured_top1": 100 * measured[lowest]["credit_expected"].mean() if lowest else None}
            if credit is not None:
                d, lo, hi = interval(credit, wide["credit_expected"])
                cell.update(top1=100 * credit.mean(), diff_vs_wide=d, diff_lo=lo, diff_hi=hi, keeps_1pp=d >= -1.0,
                            keeps_1pp_interval=lo >= -1.0)
            else:
                cell.update(top1=None, diff_vs_wide=None, diff_lo=None, diff_hi=None, keeps_1pp=False, keeps_1pp_interval=False)
            cells[(case, w)] = cell
    return cells


def orders(cells):
    """Per width: ranking (collapsed cells tied at the bottom), resolved pairs, and Kendall tau to the next width."""
    out, previous = [], None
    for w in WIDTHS:
        row = [cells[(c, w)] for c in LABEL]
        score = {c["case"]: (c["top1"] if c["top1"] is not None else -1.0) for c in row}
        ranked = sorted(score, key=lambda c: -score[c])
        resolved = []
        for a, b in itertools.combinations(LABEL, 2):
            ca, cb = cells[(a, w)], cells[(b, w)]
            if ca["credit"] is None or cb["credit"] is None:
                continue
            d, lo, hi = interval(ca["credit"], cb["credit"])
            if lo > 0 or hi < 0:
                resolved.append((LABEL[a], LABEL[b], round(d, 2)))
        tau = None
        vector = [score[c] for c in LABEL]
        if previous is not None:
            tau = float(kendalltau(previous, vector).statistic)
        previous = vector
        out.append({"W": w, "order": " > ".join(f"{LABEL[c]} {score[c]:.1f}" if score[c] >= 0 else f"{LABEL[c]} (collapsed)" for c in ranked),
                    "keeps_1pp": ", ".join(LABEL[c] for c in LABEL if cells[(c, w)]["keeps_1pp"]),
                    "keeps_1pp_interval": ", ".join(LABEL[c] for c in LABEL if cells[(c, w)]["keeps_1pp_interval"]),
                    "resolved_pairs": len(resolved), "measured_pairs": sum(1 for a, b in itertools.combinations(LABEL, 2)
                                                                           if cells[(a, w)]["credit"] is not None and cells[(b, w)]["credit"] is not None),
                    "kendall_tau_vs_previous_width": tau, "resolved": resolved})
    return out


def float_types(units):
    """fp16-rule and f21 accumulators against wide, per case (a separate accumulator type each)."""
    rows = []
    for case in LABEL:
        wide = units[("sweep", "resnet18", case, "wide")]
        for key, u in units.items():
            if key[0] != "sweep" or key[2] != case or not (key[3].startswith("fp16") or key[3] == "f21" or key[3] == "control"):
                continue
            d, lo, hi = interval(u["credit_expected"], wide["credit_expected"])
            rows.append({"case": case, "format": LABEL[case], "policy": key[3],
                         "type": "fp16" if key[3].startswith("fp16") else key[3], "top1": 100 * u["credit_expected"].mean(),
                         "diff_vs_wide": d, "diff_lo": lo, "diff_hi": hi})
    return rows
