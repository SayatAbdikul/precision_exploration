"""Lane Q8, round 3: numbers behind the review-1 wording fixes (no new model run; reads sealed records and exports).

1. margins: for every derived narrowest width of summary.json, the exact (rational) margin to the threshold at that P
   and at P - 1, in images (expected credit = sum over images of [label among maxima] / tie size).
2. bias: |bias| of every MAC output channel in product-grid units, |b_c| / (input scale * weight scale_c), from the
   L8 adapted B2 exports (the B2 default network, bias-corrected), and how many channels exceed 2^(P-1) - 1.
3. axe_vs_naive: post hoc paired AXE - naive differences at the widths where the constraint binds (the report's own
   compare(): paired bootstrap on expected credit, exact McNemar on lowest-index outcomes).

    PYTHONPATH=. CUDA_VISIBLE_DEVICES= .venv/bin/python -m tools.experiment_b2_axe.review_r3 \
        --out artifacts/experiment_b2_axe/review-r3/review-r3.json
    PYTHONPATH=. CUDA_VISIBLE_DEVICES= .venv/bin/python -m tools.experiment_b2_axe.review_r3 --channels-only \
        --out artifacts/experiment_b2_axe/review-r3/bias-channels.json
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, file_hash
from tools.experiment_b2_axe.report import BASE, BASE_A2, CASES, compare, own_arms

SUMMARY = ROOT / "results/summaries/b2-axe-v1/summary.json"
EXPORTS = ROOT / "artifacts/scaled_bridge_v2/exports/9ecfd61292dc72b1c37e8955b29e054ea9b056c52538d562a3785d407d8e135f"


def exact_credit(folder: Path, arm: str) -> tuple[Fraction, int]:
    with np.load(folder / f"{arm}.npz") as z:
        tie, among, lowest = z["tie_size"], z["label_among_maxima"], z["label_is_lowest_index_maximum"]
    credit = sum((Fraction(int(a), max(int(t), 1)) for a, t in zip(among, tie)), Fraction(0))
    return credit, int(lowest.sum())


def margins(summary: dict) -> list[dict]:
    out = []
    for case_result in summary["cases"]:
        case = case_result["case"]
        rows = [r for r in summary["rows"] if r["case"] == case]
        ref = case_result["references"]
        for key, P in sorted(case_result["derived"].items()):
            if P is None or key.endswith("_vs_optq"):
                continue
            method, rest = key.split("_", 1)
            score = "lowest_index" if rest.startswith("lowest_index") else "expected"
            tol = float(key.rsplit("acc", 1)[1])
            base = ref["l8_wide"] if method == "sat" else ref["rtn"]
            pts = {r["P"]: r[f"top1_{score}"] for r in rows if r["method"] == method}
            thr = base[f"top1_{score}"] - tol
            here, below = pts[P] - thr, (pts[P - 1] - thr if P - 1 in pts else None)
            out.append({"case": case, "rule": key, "P": P, "margin_images": round(10 * here, 4),
                        "next_narrower_margin_images": None if below is None else round(10 * below, 4),
                        "at_threshold": abs(here) < 1e-6})
    return out


def exact_edges() -> list[dict]:
    """The three edge cases recomputed in exact rational arithmetic from the per-image records."""
    out = []
    int8 = "resnet18-int8"
    rtn = exact_credit(BASE / int8, "rtn")
    for arm, tol in (("axers-P19", 1), ("rescale-P20", Fraction(1, 2))):
        credit, lowest = exact_credit(BASE_A2 / int8, arm)
        diff = credit - rtn[0]
        out.append({"case": int8, "arm": arm, "expected_credit_images": str(credit), "rtn_credit_images": str(rtn[0]),
                    "difference_images": str(diff), "difference_points": float(diff / 10),
                    "tolerance_points": float(tol), "exactly_at_threshold": diff == -10 * tol})
    return out


def bias_units() -> dict:
    from tools.scaled_bridge_v2.export import load_export
    result = {}
    for case, (l8_case, _) in CASES.items():
        export, arrays, _ = load_export(l8_case, EXPORTS / l8_case / "export.json")
        nodes = {n["name"]: n for n in export["nodes"]}

        def stored(name):
            node = nodes[name]
            while node.get("store") is None:
                node = nodes[node["inputs"][0]]
            return node

        units = []
        for node in export["nodes"]:
            if node["op"] not in ("conv", "linear"):
                continue
            s_in = float(stored(node["inputs"][0])["store"]["scale"])
            s_w = np.asarray(node["mac"]["weight_scales"], dtype=np.float64)
            units.append(np.abs(np.asarray(arrays[node["mac"]["bias"]], dtype=np.float64)) / (s_in * s_w))
        u = np.concatenate(units)
        widths = range(27, 16, -1) if case.endswith("int8") else range(23, 12, -1)
        result[case] = {"channels": int(u.size), "median": float(np.median(u)), "p99": float(np.percentile(u, 99)),
                        "max": float(u.max()),
                        "channels_above_2^(P-1)-1": {str(P): int((u > 2 ** (P - 1) - 1).sum()) for P in widths}}
    return result


def bias_channels(widths=None) -> dict:
    """Which channels exceed the register on the bias alone at the narrowest width within 1 point (INT8 19, INT6 15)."""
    from tools.scaled_bridge_v2.export import load_export
    widths = widths or {"resnet18-int8": 19, "resnet18-int6": 15}
    result = {}
    for case, (l8_case, _) in CASES.items():
        export, arrays, _ = load_export(l8_case, EXPORTS / l8_case / "export.json")
        nodes = {n["name"]: n for n in export["nodes"]}

        def stored(name):
            node = nodes[name]
            while node.get("store") is None:
                node = nodes[node["inputs"][0]]
            return node

        limit, rows = 2 ** (widths[case] - 1) - 1, []
        for node in export["nodes"]:
            if node["op"] not in ("conv", "linear"):
                continue
            s_in = float(stored(node["inputs"][0])["store"]["scale"])
            s_w = np.asarray(node["mac"]["weight_scales"], dtype=np.float64)
            bias = np.asarray(arrays[node["mac"]["bias"]], dtype=np.float64)
            units = np.abs(bias) / (s_in * s_w)
            for c in np.nonzero(units > limit)[0]:
                rows.append({"node": node["name"], "channel": int(c), "bias": float(bias[c]), "weight_scale": float(s_w[c]),
                             "node_median_weight_scale": float(np.median(s_w)), "bias_units": float(units[c])})
        result[case] = {"P": widths[case], "limit": limit, "channels": rows}
    return result


def axe_vs_naive() -> list[dict]:
    out = []
    for case, widths in (("resnet18-int8", (25, 24, 23)), ("resnet18-int6", (21, 20, 19))):
        arms = own_arms(case)
        for P in widths:
            a, n = arms[f"axe-P{P}"], arms[f"naive-P{P}"]
            c = compare(a, n)
            out.append({"case": case, "P": P, "axe_expected": 100 * float(a["expected"].mean()),
                        "naive_expected": 100 * float(n["expected"].mean()), "axe_minus_naive_expected_pp": c["expected_pp"],
                        "ci95": c["expected_ci95"], "axe_minus_naive_lowest_pp": c["lowest_pp"],
                        "mcnemar_exact_p": c["lowest_detail"]["mcnemar_exact_p"]})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--channels-only", action="store_true", help="write only the bias_channels() listing")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.channels_only:
        if out.exists():
            raise SystemExit(f"{out} exists; written once")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"code_sha256": file_hash(Path(__file__)), "bias_channels": bias_channels()}, indent=1))
        return
    if out.exists():
        raise SystemExit(f"{out} exists; written once")
    summary = json.loads(SUMMARY.read_text())
    payload = {"what": "lane Q8 round 3: numbers behind the review-1 wording fixes; post hoc, development evidence "
                       "(imagenet_screen_1k); no new model run",
               "summary_sha256": file_hash(SUMMARY), "code_sha256": file_hash(Path(__file__)),
               "rule": "narrowest P with Top-1 >= reference - tol at P and every wider measured P (inclusive, as L8)",
               "margins": margins(summary), "exact_edges": exact_edges(), "bias_product_grid_units": bias_units(),
               "axe_vs_naive_post_hoc": axe_vs_naive()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"exact_edges": payload["exact_edges"],
                      "at_threshold": [m for m in payload["margins"] if m["at_threshold"]]}, indent=1))


if __name__ == "__main__":
    main()
