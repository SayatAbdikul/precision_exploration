"""Figure v3 of lane Q8 (review-1 fix): Top-1 against P, drawn from the sealed summary.json without recomputing.

Same series, colours and markers as report.figure (v1/v2). The difference: the saturating register's points above L8's
event-free width (INT8 P >= 22, INT6 P >= 18) are not measured runs but derived (no saturation event at W_noevent on any
of the 1,000 images, so every wider register computes the wide arm bit for bit); they are drawn hollow on a thin dotted
line with their own legend entry. Points at exactly the 1-point threshold are listed in the document, not on the plot.

    .venv/bin/python -m tools.experiment_b2_axe.figure_v3 --summary results/summaries/b2-axe-v1/summary.json \
        --figure results/figures/b2-axe-top1-vs-P-v3
"""
import argparse
import json
from pathlib import Path

from tools.experiment_b2_axe.report import EXPLORATORY

STYLE = {"axe": ("#2a78d6", "o", "AXE-OPTQ, fixed B2 scales (simulator)"),
         "naive": ("#eb6834", "s", "projection + clip, no error comp. (simulator)"),
         "sat": ("#1baf7a", "^", "saturating register sat.wP (exact engine, L8; measured)"),
         "rescale": ("#eda100", "D", "exploratory: rescaled grid + RTN (simulator)"),
         "axers": ("#e87ba4", "v", "exploratory: rescaled grid + AXE-OPTQ (simulator)")}


def split_sat(rows, case):
    measured, derived = [], []
    for r in rows:
        if r["case"] == case and r["method"] == "sat":
            (derived if str(r.get("source", "")).startswith("derived") else measured).append((r["P"], r["top1_expected"]))
    return sorted(measured), sorted(derived)


def draw(summary, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results, rows = summary["cases"], summary["rows"]
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.0), squeeze=False)
    for ax, result in zip(axes[0], results):
        case = result["case"]
        for method, (color, marker, label) in STYLE.items():
            dashed = method in EXPLORATORY
            if method == "sat":
                pts, derived = split_sat(rows, case)
            else:
                pts = sorted((r["P"], r["top1_expected"]) for r in rows if r["case"] == case and r["method"] == method)
                derived = []
            if pts:
                ax.plot([p for p, _ in pts], [v for _, v in pts], marker=marker, color=color, label=label,
                        lw=1.2 if dashed else 1.6, ls="--" if dashed else "-", ms=4.5,
                        mfc="white" if dashed else color)
            if derived and pts:  # join the derived run to the widest measured point
                joined = [pts[-1]] + derived
                ax.plot([p for p, _ in joined], [v for _, v in joined], color=color, lw=1.0, ls=":", marker=None)
                ax.plot([p for p, _ in derived], [v for _, v in derived], color=color, lw=0, marker=marker, ms=4.5,
                        mfc="white", label="sat.wP above L8's event-free width: derived (= wide), not run")
        ref = result["references"]
        ax.axhline(ref["rtn"]["top1_expected"], color="#5a5a5a", ls=":", lw=1, label="B2 default, wide (simulator)")
        if "optq" in ref:
            ax.axhline(ref["optq"]["top1_expected"], color="#5a5a5a", ls="-.", lw=0.8, label="OPTQ, wide (simulator)")
        cert = ref["rtn"]["certificate"]
        ax.axvline(cert["signed_bits_structural"], color="#222222", lw=1, ls="--")
        ax.text(cert["signed_bits_structural"] - 0.2, 20, f"certificate {cert['signed_bits_structural']}\n(structural)",
                ha="left", va="center", fontsize=8, color="#222222")
        ax.set_xlabel("accumulator width P (signed bits, every MAC node; bias outside the register)")
        ax.set_ylabel("Top-1 % (expected credit), 1k screen")
        ax.set_title(case.replace("resnet18-", "ResNet18 ").upper().replace("RESNET18", "ResNet18"))
        ax.set_ylim(-2, 75)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    for ext in ("png", "pdf"):
        path = Path(f"{stem}.{ext}")
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=150)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args(argv)
    draw(json.loads(Path(args.summary).read_text()), args.figure)


if __name__ == "__main__":
    main()
