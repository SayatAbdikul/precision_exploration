"""Lane Q5 figures from results/summaries/b2-seeds-v1 (written once to results/figures/b2-seeds-*-v1.{png,pdf})."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tools.experiment_b.common import ROOT  # noqa: E402

SUMMARY = ROOT / "results/summaries/b2-seeds-v1"
FIGURES = ROOT / "results/figures"
MODELS = {"resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2", "mobilenet_v3_large": "MobileNetV3-Large"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEEDS = [f"S{k}" for k in range(5)]


def rows(name, folder):
    with (Path(folder) / name).open() as stream:
        return list(csv.DictReader(stream))


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def save(fig, stem, out, partial):
    for ext in ("png", "pdf"):
        path = Path(out) / f"{stem}.{ext}"
        if path.exists() and not partial:
            raise FileExistsError(path)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cells_figure(folder, out, partial):
    data = rows("seed-cells.csv", folder)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharey=False)
    for ax, (model, title) in zip(axes, MODELS.items()):
        mine = [r for r in data if r["model"] == model]
        labels = [f"{r['format']}\n{r['recipe'].replace('cum5_act_maxabs', 'cum5')}" for r in mine]
        x = np.arange(len(mine))
        ax.axhspan(-0.5, 0.5, color="#f1f0ec", zorder=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for i, r in enumerate(mine):
            full = float(r["expected_full_2000"])
            values = [float(r[f"expected_{s}"]) - full for s in SEEDS]
            ax.scatter(np.full(5, i), values, s=14, color=SERIES[0], edgecolor="white", linewidth=0.5, zorder=3)
            ax.scatter([i], [float(r["expected_mean"]) - full], marker="_", s=120, color=INK, zorder=4)
        ax.set_xticks(x, labels, fontsize=6)
        ax.set_ylabel("top-1 minus 2k value (pp)", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        style(ax)
    fig.suptitle("Calibration seeds (5 disjoint 400-image subsets): expected-credit top-1 relative to the 2,000-image "
                 "calibration; band = +-delta/2; bar = seed mean. 1k screen, development evidence.", fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, "b2-seeds-cells-v1", out, partial)


def contrasts_figure(folder, out, partial):
    data = rows("contrasts.csv", folder)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    top = 0.0
    for color, (model, title) in zip(SERIES, MODELS.items()):
        mine = [r for r in data if r["model"] == model]
        xs = [float(r["expected_image_sd_n1000"]) for r in mine]
        ys = [float(r["expected_sd"]) for r in mine]
        top = max(top, *xs, *ys)
        ax.scatter(xs, ys, s=16, color=color, edgecolor="white", linewidth=0.5, label=title, zorder=3)
    ax.plot([0, top * 1.05], [0, top * 1.05], color=MUTED, linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color=SERIES[3], linewidth=1.0)
    ax.text(0.02, 0.52, "delta/2 = 0.5 pp", fontsize=7, color=MUTED)
    ax.set_xlabel("image-sampling SD of the contrast at n = 1,000 (pp, paired bootstrap)", fontsize=7, color=MUTED)
    ax.set_ylabel("calibration-seed SD of the contrast (pp, 5 seeds)", fontsize=7, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Same-width contrasts: seed versus image variability", fontsize=8, color=INK, loc="left")
    style(ax)
    save(fig, "b2-seeds-contrasts-v1", out, partial)


def size_figure(folder, out, partial):
    data = [r for r in rows("size.csv", folder) if r["readout"] == "expected"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
    formats = sorted({r["format"] for r in data}, key=["int8", "posit8_es1", "fp8_e4m3fn", "int6"].index)
    for ax, (model, title) in zip(axes, MODELS.items()):
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for offset, (color, fmt) in enumerate(zip(SERIES, formats)):
            mine = sorted((r for r in data if r["model"] == model and r["format"] == fmt), key=lambda r: int(r["range_images"]))
            x = np.array([int(r["range_images"]) for r in mine], dtype=float) * (1 + 0.04 * (offset - 1.5))
            y = [float(r["mean_minus_full"]) for r in mine]
            e = [float(r["sd"]) if r["sd"] else 0.0 for r in mine]
            ax.errorbar(x, y, yerr=e, color=color, marker="o", markersize=4, linewidth=1.5, capsize=2, label=fmt)
        ax.set_xscale("log")
        ax.set_xticks([32, 128, 400, 1000], ["32", "128", "400", "1000"])
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("calibration images for the ranges (bias correction: min(n, 256))", fontsize=6, color=MUTED)
        style(ax)
    axes[0].set_ylabel("top-1 minus 2k value (pp); mean +- SD", fontsize=7, color=MUTED)
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "b2-seeds-size-v1", out, partial)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(SUMMARY))
    parser.add_argument("--out", default=str(FIGURES))
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    cells_figure(args.summary, args.out, args.partial)
    contrasts_figure(args.summary, args.out, args.partial)
    size_figure(args.summary, args.out, args.partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
