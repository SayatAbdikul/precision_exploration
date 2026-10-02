"""Lane Q2 figures from results/summaries/b2-rank-v1 (CPU; files written once, as the summaries).

  .venv/bin/python -m tools.experiment_b2_rank.figures [--summary results/summaries/b2-rank-v1] [--out results/figures]

* b2-rank-tau-heatmap.png: Kendall tau-b between recipe orderings, all 25 formats, one panel per network.
* b2-rank-decomposition.png: default-minus-minimal top-1 split into Shapley parts (unsigned, weight MSE range, bias
  correction), with the total; per network, formats above chance under either corner.
* b2-rank-unsigned.png: unsigned-sibling minus signed codebook under the default recipe, with 95 percent intervals.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

from tools.experiment_b.common import ROOT  # noqa: E402
from .plan import MODELS  # noqa: E402

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (dataviz reference palette)
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#e34948", "#f0efec", "#2a78d6"])
VIEWS = ("v1_maxabs", "v1_percentile_99_9", "minimal", "cum3_act_mse", "rank_signed_no_bias_correction",
         "rank_signed_weight_maxabs", "default_no_bias_correction", "default_weight_maxabs", "default_signed",
         "default", "default_unsigned")
SHORT = {"v1_maxabs": "v1 max-abs", "v1_percentile_99_9": "v1 pct 99.9", "minimal": "minimal",
         "cum3_act_mse": "U  .  .", "rank_signed_no_bias_correction": ".  W  .",
         "rank_signed_weight_maxabs": ".  .  B", "default_no_bias_correction": "U  W  .",
         "default_weight_maxabs": "U  .  B", "default_signed": ".  W  B", "default": "default (U W B)",
         "default_unsigned": "default + unsigned"}
NICE = {"resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2", "mobilenet_v3_large": "MobileNetV3-Large"}


def rows(path):
    return list(csv.DictReader(path.open()))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=7)


def save_once(fig, path):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    data = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"refusing to replace {path}")
        return
    path.write_bytes(data)


def tau_heatmap(summary, out):
    corr = [r for r in rows(summary / "rank-correlation.csv") if r["class"] == "all" and r["subset"] == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), facecolor=SURFACE, constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        present = [v for v in VIEWS if any(r["model"] == model and v in (r["view_a"], r["view_b"]) for r in corr)]
        n = len(present)
        grid = np.full((n, n), np.nan)
        np.fill_diagonal(grid, 1.0)
        for r in corr:
            if r["model"] == model and r["view_a"] in present and r["view_b"] in present:
                i, j = present.index(r["view_a"]), present.index(r["view_b"])
                grid[i, j] = grid[j, i] = float(r["kendall_tau_b"])
        grid[np.triu_indices(n, 1)] = np.nan  # lower triangle only (symmetric)
        cmap = DIVERGING.copy()
        cmap.set_bad(SURFACE)
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                if i > j and np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color=TEXT)
        ax.set_xticks(range(n), [SHORT[v] for v in present], rotation=60, ha="right", fontsize=6.5, color=TEXT)
        ax.set_yticks(range(n), [SHORT[v] for v in present], fontsize=6.5, color=TEXT)
        ax.set_title(NICE[model], fontsize=9, color=TEXT)
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(image, ax=axes, shrink=0.7)
    bar.set_label("Kendall tau-b, 25 formats (point estimate)", fontsize=7, color=TEXT)
    bar.ax.tick_params(labelsize=6, colors=MUTED)
    fig.suptitle("Format orderings under each recipe (U unsigned, W MSE weight range, B bias correction); "
                 "1k screen, development evidence", fontsize=8.5, color=MUTED)
    save_once(fig, out / "b2-rank-tau-heatmap.png")


def decomposition(summary, out):
    decomp = [r for r in rows(summary / "decomposition.csv") if r["complete"] == "True"]
    parts = (("shapley_unsigned", "unsigned (integers)"), ("shapley_weight_mse", "MSE weight range"),
             ("shapley_bias_correction", "bias correction"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2), facecolor=SURFACE, sharex=True, constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        use = [r for r in decomp if r["model"] == model and max(float(r["default"]), float(r["minimal"])) >= 1.0]
        use.sort(key=lambda r: (-int(r["bits"]), r["format"]))
        y = np.arange(len(use))
        style(ax)
        ax.axvline(0, color=MUTED, lw=0.8)
        for k, (key, label) in enumerate(parts):
            values = np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in use])
            ax.barh(y + (k - 1) * 0.26, np.nan_to_num(values), height=0.24, color=SERIES[k], label=label,
                    edgecolor=SURFACE, linewidth=0.5)
        total = np.array([float(r["default_minus_minimal"]) for r in use])
        ax.scatter(total, y, marker="D", s=14, color=TEXT, zorder=3, label="total: default - minimal")
        ax.set_yticks(y, [f"{r['format']}" for r in use], fontsize=6.5)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(NICE[model], fontsize=9, color=TEXT)
        ax.set_xlabel("top-1 points (expected credit)", fontsize=7, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="outside lower center", ncol=4)
    fig.suptitle("Default minus minimal, split into its recipe switches (Shapley over the factorial corners; block formats: "
                 "intrinsic arm); 1k screen, development evidence", fontsize=8.5, color=MUTED)
    save_once(fig, out / "b2-rank-decomposition-v2.png")  # v1 had the legend over the ResNet18 bars


def unsigned(summary, out):
    sign = rows(summary / "signedness.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), facecolor=SURFACE, sharey=True, constrained_layout=True)
    names = sorted({r["format"] for r in sign}, key=lambda n: (-max(int(r["bits"]) for r in sign if r["format"] == n), n))
    for ax, model in zip(axes, MODELS):
        style(ax)
        ax.axvline(0, color=MUTED, lw=0.8)
        for i, name in enumerate(names):
            r = next((r for r in sign if r["model"] == model and r["format"] == name), None)
            if r is None:
                continue
            d, lo, hi = float(r["d"]), float(r["low"]), float(r["high"])
            color = SERIES[1] if name.startswith("int") else SERIES[0]
            ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round")
            ax.scatter([d], [i], color=color, s=16, zorder=3, edgecolor=SURFACE, linewidth=0.8)
        ax.set_yticks(range(len(names)), names, fontsize=6.5)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(NICE[model], fontsize=9, color=TEXT)
        ax.set_xlabel("unsigned minus signed codebook at non-negative nodes, top-1 points", fontsize=7, color=TEXT)
    handles = [plt.Line2D([], [], color=SERIES[1], lw=2, label="integer (B2 default vs default_signed)"),
               plt.Line2D([], [], color=SERIES[0], lw=2, label="unsigned sibling (rule b2_rank_unsigned_sibling_v1)")]
    axes[0].legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("Spending the sign bit on magnitude at non-negative nodes, default recipe, 95% paired intervals; "
                 "1k screen, development evidence", fontsize=8.5, color=MUTED)
    save_once(fig, out / "b2-rank-unsigned.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summaries/b2-rank-v1")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    summary, out = ROOT / args.summary, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tau_heatmap(summary, out)
    decomposition(summary, out)  # writes b2-rank-decomposition-v2.png (v1 kept as written)
    unsigned(summary, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
