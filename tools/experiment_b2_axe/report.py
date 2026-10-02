"""Lane Q8 analysis: AXE / naive / OPTQ / RTN (B2 simulator) against lane L8's saturating register (exact engine).

    .venv/bin/python -m tools.experiment_b2_axe.report --out results/summaries/b2-axe-v1 [--figure results/figures/b2-axe-top1-vs-P-v1]

Reads only sealed records (own evals; L8's predict files through tools.accumulator_sweep_v1.load, read-only).
Writes the summary folder once (refuses to overwrite).  Development evidence on the 1k screen.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.analysis.b2_ties import paired
from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
from tools.experiment_b.common import ROOT, file_hash, unseal

BASE = ROOT / "artifacts/experiment_b2_axe/evals-v2"
BASE_A2 = ROOT / "artifacts/experiment_b2_axe/evals-a2"  # addendum-2 arms (exploratory, post hoc)
BASE_LEAN = (ROOT / "artifacts/experiment_b2_axe/evals-v2-lean",  # v1 arms run with the lean bias correction
             ROOT / "artifacts/experiment_b2_axe/evals-v2-lean-b")
DUPLICATES = {}  # case -> list of arms recorded in both evals-v2 and evals-v2-lean (must be identical)
PRIMARY = ("axe", "naive")
EXPLORATORY = ("rescale", "axers")
CASES = {"resnet18-int8": ("resnet18-int8-default-b2", 21), "resnet18-int6": ("resnet18-int6-default-b2", 17)}


def own_arms(case, base=None, base_a2=None, base_lean=None):
    arms = _arms((base or BASE) / case)
    for folder in [f / case for f in (base_lean or BASE_LEAN)]:
        if not folder.exists():
            continue
        for name, arm in _arms(folder).items():
            if name in arms:
                old = arms[name]
                same = (old["record"]["weight_codes_sha256"] == arm["record"]["weight_codes_sha256"]
                        and np.array_equal(old["expected"], arm["expected"]) and np.array_equal(old["lowest"], arm["lowest"]))
                if not same:
                    raise ValueError(f"{case}/{name}: records of the same v1 arm differ between folders")
                DUPLICATES.setdefault(case, []).append(name)
            else:
                arms[name] = arm
    folder = (base_a2 or BASE_A2) / case
    if folder.exists():
        extra = _arms(folder)
        if set(extra) & set(arms):
            raise ValueError("addendum-2 arm names collide with v1 arms")
        arms.update(extra)
    return arms


def _arms(folder):
    arms = {}
    for path in sorted(folder.glob("*.json")):
        record = unseal(path)
        source = record.get("identical_to") or record["arm"]
        with np.load(folder / f"{source}.npz") as z:
            tie, among, lowest = z["tie_size"], z["label_among_maxima"], z["label_is_lowest_index_maximum"]
        arms[record["arm"]] = {"record": record, "expected": among / np.maximum(tie, 1),
                               "lowest": lowest.astype(np.int8), "file": _relative(path),
                               "sha256": file_hash(path)}
    return arms


def _relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def l8_arms(l8_case, widths, noevent):
    from tools.accumulator_sweep_v1.load import assemble
    from tools.accumulator_sweep_v1.score import image_scores
    out = {}
    wide = assemble(l8_case, "wide", 0, 1000)
    scored = image_scores(wide["images"])
    out["wide"] = {"expected": scored["expected"], "lowest": scored["lowest"],
                   "labels": np.array([r["label"] for r in wide["images"]]), "source": "measured"}
    for P in widths:
        if P >= noevent:
            measured = assemble(l8_case, f"sat.w{P}", 0, 1000)
            if measured is None:  # above L8's bracket: no saturation event at W_noevent, so equal to wide
                out[P] = {**out["wide"], "source": f"derived: P >= W_noevent={noevent}, equals wide"}
                continue
        else:
            measured = assemble(l8_case, f"sat.w{P}", 0, 1000)
        if measured is None:
            out[P] = None
            continue
        s = image_scores(measured["images"])
        out[P] = {"expected": s["expected"], "lowest": s["lowest"], "source": "measured",
                  "event_images": int(s["event"].sum())}
    return out


def compare(left, right):
    e = paired(left["expected"], right["expected"])
    lo = paired_outcomes(right["lowest"], left["lowest"])
    return {"expected_pp": e["difference_pp"], "expected_ci95": e["pointwise_95_interval_pp"],
            "lowest_pp": lo["right_percent"] - lo["left_percent"], "lowest_detail": lo}


def narrowest(points, reference, tol):
    """Narrowest P with score >= reference - tol at P and at every wider measured P (L8 rule)."""
    best = None
    for P in sorted(points, reverse=True):
        if points[P] is None:  # not measured for this method: the rule is over measured widths only
            continue
        if points[P] < reference - tol:
            break
        best = P
    return best


def analyse_case(case, base=None, base_a2=None):
    l8_case, noevent = CASES[case]
    arms = own_arms(case, base, base_a2)
    widths = sorted({a["record"]["P"] for a in arms.values() if a["record"]["P"] is not None}, reverse=True)
    sat = l8_arms(l8_case, widths, noevent)
    labels_ok = True
    rows, table = [], {}
    ref, ref_optq, wide = arms["rtn"], arms.get("optq"), sat["wide"]
    for P in widths:
        entry = {"P": P}
        for method in PRIMARY + EXPLORATORY:
            arm = arms.get(f"{method}-P{P}")
            if arm is None:
                continue
            r = arm["record"]
            row = {"case": case, "method": method, "P": P, "arm": r["arm"],
                   "top1_expected": 100 * float(arm["expected"].mean()),
                   "top1_lowest_index": 100 * float(arm["lowest"].mean()),
                   "identical_to": r.get("identical_to"),
                   "cert_structural": r["certificate"]["signed_bits_structural"],
                   "cert_range": r["certificate"]["signed_bits_range"],
                   "cert_absolute": r["certificate"]["signed_bits_absolute"],
                   "bound_holds": r["bound_holds"], "max_beta": r["max_beta"], "max_neg": r["max_neg"],
                   "integer_limit": r["integer_limit"],
                   "prefix_bits_32img": (r.get("prefix_check") or {}).get("prefix_bits"),
                   "orderfree_bits_32img": (r.get("prefix_check") or {}).get("orderfree_bits"),
                   "holds_at_P_32img": (r.get("prefix_check") or {}).get("holds_at_P"),
                   "status": "exploratory (addendum-2)" if method in EXPLORATORY else "primary (v1)"}
            if method in EXPLORATORY:
                factors = r["factors"].values()
                channels = sum(f["channels"] for f in factors)
                row.update({"max_k": max(f["max_k"] for f in factors),
                            "fraction_channels_rescaled": sum(f["fraction_rescaled"] * f["channels"] for f in factors) / channels,
                            "mean_log2_k": sum(f["mean_log2_k"] * f["channels"] for f in factors) / channels})
            c = compare(arm, ref)
            row.update({"vs_rtn_expected_pp": c["expected_pp"], "vs_rtn_expected_ci95": c["expected_ci95"],
                        "vs_rtn_lowest_pp": c["lowest_pp"], "vs_rtn_lowest_ci95": c["lowest_detail"]["pointwise_95_interval_pp"],
                        "vs_rtn_mcnemar_p": c["lowest_detail"]["mcnemar_exact_p"]})
            if ref_optq is not None:
                c = compare(arm, ref_optq)
                row.update({"vs_optq_expected_pp": c["expected_pp"], "vs_optq_expected_ci95": c["expected_ci95"]})
            if sat.get(P) is not None:
                c = compare(arm, sat[P])
                row.update({"vs_sat_expected_pp": c["expected_pp"], "vs_sat_expected_ci95": c["expected_ci95"],
                            "vs_sat_lowest_pp": c["lowest_pp"]})
            rows.append(row)
            entry[method] = row
        if sat.get(P) is not None:
            s = sat[P]
            c = compare(s, wide)
            row = {"case": case, "method": "sat", "P": P, "arm": f"sat.w{P}", "source": s["source"],
                   "top1_expected": 100 * float(s["expected"].mean()), "top1_lowest_index": 100 * float(s["lowest"].mean()),
                   "vs_wide_expected_pp": c["expected_pp"], "vs_wide_expected_ci95": c["expected_ci95"],
                   "vs_wide_lowest_pp": c["lowest_pp"]}
            rows.append(row)
            entry["sat"] = row
        table[P] = entry
    refs = {"rtn": ref, "optq": ref_optq}
    references = {}
    for name, arm in refs.items():
        if arm is None:
            continue
        r = arm["record"]
        references[name] = {"top1_expected": 100 * float(arm["expected"].mean()),
                            "top1_lowest_index": 100 * float(arm["lowest"].mean()),
                            "certificate": {k: r["certificate"][k] for k in ("signed_bits_absolute", "signed_bits_range",
                                                                              "signed_bits_structural")},
                            "prefix_bits_32img": (r.get("prefix_check") or {}).get("prefix_bits"),
                            "orderfree_bits_32img": (r.get("prefix_check") or {}).get("orderfree_bits"),
                            "sealed_b2_check": r.get("sealed_b2_check"),
                            "rtn_equals_adapted_export_weight_units": r.get("rtn_equals_adapted_export_weight_units")}
    if ref_optq is not None:
        c = compare(ref_optq, ref)
        references["optq"]["vs_rtn"] = {"expected_pp": c["expected_pp"], "expected_ci95": c["expected_ci95"],
                                        "lowest_pp": c["lowest_pp"]}
    from tools.experiment_b.common import dataset
    screen = np.array([int(row["label"]) for row in dataset("imagenet_screen_1k")[1][:1000]])
    labels_ok = bool(np.array_equal(screen, wide["labels"]))
    if not labels_ok:
        raise ValueError("L8 image order differs from the B2 screen order; pairing would be wrong")
    references["l8_wide"] = {"top1_expected": 100 * float(wide["expected"].mean()),
                             "top1_lowest_index": 100 * float(wide["lowest"].mean())}
    c = compare(ref, wide)
    references["rtn_minus_l8_wide"] = {"expected_pp": c["expected_pp"], "expected_ci95": c["expected_ci95"],
                                       "lowest_pp": c["lowest_pp"]}
    derived = {}
    for method, base in (("axe", references["rtn"]), ("naive", references["rtn"]), ("rescale", references["rtn"]),
                         ("axers", references["rtn"]), ("sat", references["l8_wide"])):
        if not any(method in table[P] for P in widths):
            continue
        for score in ("expected", "lowest_index"):
            pts = {P: table[P][method][f"top1_{score}"] if method in table[P] else None for P in widths}
            for tol in (0.5, 1.0):
                derived[f"{method}_{score}_acc{tol}"] = narrowest(pts, base[f"top1_{score}"], tol)
        if method in ("axe", "axers") and "optq" in references:
            pts = {P: table[P][method]["top1_expected"] if method in table[P] else None for P in widths}
            derived[f"{method}_expected_acc1.0_vs_optq"] = narrowest(pts, references["optq"]["top1_expected"], 1.0)
    sources = {name: arm["file"] for name, arm in sorted(arms.items())}
    return {"case": case, "l8_case": l8_case, "widths": widths, "references": references, "derived": derived,
            "label_order_checked": labels_ok, "record_files": sources,
            "identical_in_both_v2_folders": DUPLICATES.get(case, [])}, rows


def figure(results, rows, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.0), squeeze=False)
    # categorical slots 1-5 of the dataviz reference palette (validated: CVD/normal-vision pass; contrast WARN ->
    # every series also has its own marker, and every value is in arms.csv)
    style = {"axe": ("#2a78d6", "o", "AXE-OPTQ, fixed B2 scales (simulator)"),
             "naive": ("#eb6834", "s", "projection + clip, no error comp. (simulator)"),
             "sat": ("#1baf7a", "^", "saturating register sat.wP (exact engine, L8)"),
             "rescale": ("#eda100", "D", "exploratory: rescaled grid + RTN (simulator)"),
             "axers": ("#e87ba4", "v", "exploratory: rescaled grid + AXE-OPTQ (simulator)")}
    for ax, result in zip(axes[0], results):
        case = result["case"]
        for method, (color, marker, label) in style.items():
            pts = sorted((r["P"], r["top1_expected"]) for r in rows if r["case"] == case and r["method"] == method)
            if pts:
                dashed = method in EXPLORATORY
                ax.plot([p for p, _ in pts], [v for _, v in pts], marker=marker, color=color, label=label,
                        lw=1.2 if dashed else 1.6, ls="--" if dashed else "-", ms=4.5,
                        mfc="white" if dashed else color)
        ref = result["references"]
        ax.axhline(ref["rtn"]["top1_expected"], color="#5a5a5a", ls=":", lw=1, label="B2 default, wide (simulator)")
        if "optq" in ref:
            ax.axhline(ref["optq"]["top1_expected"], color="#5a5a5a", ls="-.", lw=0.8, label="OPTQ, wide (simulator)")
        cert = ref["rtn"]["certificate"]
        ax.axvline(cert["signed_bits_structural"], color="#222222", lw=1, ls="--")
        # the x axis is inverted, so a left-aligned label at P - 0.2 sits just right of the line, in the plot
        ax.text(cert["signed_bits_structural"] - 0.2, 35, f"certificate {cert['signed_bits_structural']}\n(structural, B2 default)",
                ha="left", va="center", fontsize=8, color="#222222")
        ax.set_xlabel("accumulator width P (signed bits, every MAC node)")
        ax.set_ylabel("Top-1 % (expected credit), 1k screen")
        ax.set_title(case.replace("resnet18-", "ResNet18 ").upper().replace("RESNET18", "ResNet18"))
        ax.set_ylim(-2, 75)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "pdf"):
        path = Path(f"{stem}.{ext}")
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=150)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure", default="")
    parser.add_argument("--cases", default="resnet18-int8,resnet18-int6")
    parser.add_argument("--figure-only", default="", help="redraw the figure from an existing summary.json")
    parser.add_argument("--evals", default="", help="testing only: another records folder")
    parser.add_argument("--evals-a2", default="", help="testing only: another addendum-2 records folder")
    args = parser.parse_args(argv)
    if args.figure_only:
        summary = json.loads(Path(args.figure_only).read_text())
        figure(summary["cases"], summary["rows"], args.figure)
        return
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists; summaries are written once")
    results, rows = [], []
    for case in args.cases.split(","):
        result, case_rows = analyse_case(case, Path(args.evals) if args.evals else None,
                                         Path(args.evals_a2) if args.evals_a2 else None)
        results.append(result)
        rows.extend(case_rows)
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with (out / "arms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    configs = ROOT / "public/experiments/configs/breadth-study"
    summary = {"protocol": "b2-axe-protocol-v1",
               "protocol_sha256": file_hash(configs / "b2-axe-protocol-v1.json"),
               "addenda_sha256": {name: file_hash(configs / f"{name}.json") for name in
                                  ("b2-axe-protocol-v1-addendum-1", "b2-axe-protocol-v1-addendum-2")
                                  if (configs / f"{name}.json").exists()},
               "methods_status": {"primary (v1)": list(PRIMARY) + ["optq", "rtn", "sat"],
                                  "exploratory, post hoc (addendum-2)": list(EXPLORATORY)},
               "evidence_class": "development evidence, imagenet_screen_1k (1000 images); one network, one calibration draw",
               "report_code_sha256": file_hash(Path(__file__)), "cases": results, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if args.figure:
        figure(results, rows, args.figure)
    print(json.dumps({"cases": [{"case": r["case"], "derived": r["derived"]} for r in results]}, indent=1))


if __name__ == "__main__":
    main()
