"""Figures of lane Q7 from the written summaries (results/figures/quality-metrics-*-v1.{png,pdf}); never overwrites."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tools.experiment_b.common import ROOT  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NS = (128, 256, 512, 1000)


def rows(folder, name):
    return list(csv.DictReader((Path(folder) / name).open()))


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    from matplotlib.ticker import NullFormatter, FormatStrFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))


def save(fig, stem):
    for ext in ("png", "pdf"):
        path = ROOT / "results/figures" / f"{stem}.{ext}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def width_vs_n(folder):
    summary = rows(folder, "partB-subsample-summary.csv")
    det = rows(folder, "partB-detector-resolution.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.subplots_adjust(wspace=0.32)
    labels = {"cell_vs_fp32": "cell vs FP32", "cell_vs_int8": "cell vs INT8", "format_pair": "format pair",
              "recipe_pair": "recipe pair", "accumulator_vs_wide": "accumulator vs wide"}
    for i, kind in enumerate(labels):
        r = [x for x in summary if x["kind"] == kind]
        if not r:
            continue
        y = [float(r[0][f"width_expected_median_n{n}_median"]) for n in NS]
        a.plot(NS, y, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=labels[kind])
    ref = np.array(NS, float)
    a.plot(ref, 2.0 * np.sqrt(1000 / ref), color=MUTED, linestyle="--", linewidth=1, label="slope -1/2 (2 pp at 1k)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(NS); a.set_xticklabels([str(n) for n in NS])
    a.set_xlabel("images (disjoint blocks)", color=INK, fontsize=9)
    a.set_ylabel("median 95% interval width, top-1 pp", color=INK, fontsize=9)
    a.set_title("Classifiers, paired top-1 (expected credit)", fontsize=10, color=INK, loc="left")
    style(a); a.legend(fontsize=7, frameon=False, loc="upper right")
    for i, r in enumerate(det):
        y = [float(r[f"width_n{n}_median"]) for n in NS[:3]] + [float(r["width_n1000"])]
        b.plot(NS, y, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=r["contrast"])
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xticks(NS); b.set_xticklabels([str(n) for n in NS])
    b.set_xlabel("images (disjoint blocks)", color=INK, fontsize=9)
    b.set_ylabel("median 95% interval width, mAP50-95 points", color=INK, fontsize=9)
    b.set_title("YOLOv8n, paired AP difference", fontsize=10, color=INK, loc="left")
    style(b); b.legend(fontsize=7, frameon=False, loc="lower left")
    fig.text(0.0, -0.04, "Development evidence: ImageNet and COCO screen-1k lists; widths are medians over disjoint blocks.",
             fontsize=7, color=MUTED)
    save(fig, "quality-metrics-width-vs-n-v1")


def coverage(folder):
    data = rows(folder, "partB-coverage.csv")
    picks = [("pd_q5_asym_q50", "discordance 5th pct"), ("pd_q50_asym_q50", "median discordance"),
             ("pd_q95_asym_q50", "discordance 95th pct"), ("null_pd1", "null, 1% discordant")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, (name, title) in zip(axes, picks):
        r = [x for x in data if x["scenario"] == name]
        if not r:
            continue
        for i, (key, label) in enumerate((("percentile", "percentile"), ("bca", "BCa"), ("wald", "Wald"))):
            ax.plot(NS, [float(x[key]) for x in r], color=SERIES[i], linewidth=2, marker="o", markersize=4, label=label)
        ax.axhline(0.95, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
        ax.set_title(f"{title} ({100 * float(r[0]['discordance']):.1f}%)", fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("coverage of the true difference", color=INK, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    fig.text(0.0, -0.05, "Simulated paired binary outcomes (2,000 data sets x 2,000 resamples per point); dashed line 0.95.",
             fontsize=7, color=MUTED)
    save(fig, "quality-metrics-coverage-v1")


def fixed_width(folder):
    grid = rows(folder, "partE-grid.csv")
    fmts = list(dict.fromkeys(r["format"] for r in grid))
    widths = sorted({int(r["W"]) for r in grid})
    z = np.full((len(fmts), len(widths)), np.nan)
    keep = np.zeros_like(z, dtype=bool)
    for r in grid:
        i, j = fmts.index(r["format"]), widths.index(int(r["W"]))
        if r["top1"]:
            z[i, j] = float(r["top1"])
        keep[i, j] = r["keeps_1pp"] == "True"
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#2a78d6", "#0d2f5c"])
    cmap.set_bad("#e4e3df")
    im = ax.imshow(z, aspect="auto", cmap=cmap, vmin=0, vmax=70)
    for i in range(len(fmts)):
        for j in range(len(widths)):
            if keep[i, j]:
                ax.text(j, i, "•", ha="center", va="center", color="#ffffff", fontsize=9)
            elif np.isnan(z[i, j]):
                ax.text(j, i, "x", ha="center", va="center", color=MUTED, fontsize=6)
    ax.set_xticks(range(len(widths))); ax.set_xticklabels(widths, fontsize=8)
    ax.set_yticks(range(len(fmts))); ax.set_yticklabels(fmts, fontsize=8)
    ax.set_xlabel("saturating accumulator register width W (bits, product grid)", fontsize=9, color=INK)
    bar = fig.colorbar(im, ax=ax, fraction=0.03)
    bar.set_label("top-1 % (expected credit)", fontsize=8)
    ax.set_title("ResNet18 B2 recipe, uniform sat.w<W>: dot = within 1 pp of own exact arm; x = collapsed, not measured",
                 fontsize=9, color=INK, loc="left")
    save(fig, "quality-metrics-fixed-width-v1")


if __name__ == "__main__":
    folder = sys.argv[1]
    width_vs_n(folder); coverage(folder); fixed_width(folder)
