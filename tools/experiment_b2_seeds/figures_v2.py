"""Lane Q5 cells figure, version 2 (review 2: the x tick labels of b2-seeds-cells-v1 overlapped).

Same data as v1 (results/summaries/b2-seeds-v1/seed-cells.csv, unchanged); only the layout differs: short display
names, cells ordered by width and recipe, a wider canvas, and a check after drawing that no two x tick labels
overlap (the script fails otherwise). Writes once to results/figures/b2-seeds-cells-v2.{png,pdf}.

    .venv/bin/python -m tools.experiment_b2_seeds.figures_v2 [--out <scratch> --partial]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tools.experiment_b2_seeds.figures import (  # noqa: E402
    FIGURES, INK, MODELS, MUTED, SEEDS, SERIES, SUMMARY, rows, save, style,
)

# (format, recipe) in display order: 8-bit, 7-bit, 6-bit default, 6-bit minimal, 4-bit minimal.
ORDER = [
    ("int8", "default"), ("posit8_es1", "default"), ("fp8_e4m3fn", "default"), ("log8", "default"),
    ("mxfp8_e4m3", "cum5_act_maxabs"), ("fp7_e3m3", "default"),
    ("int6", "default"), ("fp6_e2m3", "default"), ("fp6_e3m2", "default"), ("bfp6", "cum5_act_maxabs"),
    ("log6", "default"), ("posit6_es1", "default"),
    ("int6", "minimal"), ("fp6_e2m3", "minimal"), ("fp6_e3m2", "minimal"), ("log6", "minimal"),
    ("posit6_es1", "minimal"), ("nf4", "minimal"),
]
NAMES = {
    "int8": "INT8", "posit8_es1": "Posit8", "fp8_e4m3fn": "FP8\nE4M3", "log8": "LOG8", "mxfp8_e4m3": "MXFP8",
    "fp7_e3m3": "FP7\nE3M3", "int6": "INT6", "fp6_e2m3": "FP6\nE2M3", "fp6_e3m2": "FP6\nE3M2", "bfp6": "BFP6",
    "log6": "LOG6", "posit6_es1": "Posit6", "nf4": "NF4",
}
RECIPES = {"default": "def.", "minimal": "min.", "cum5_act_maxabs": "cum5"}
GROUPS = [(0, 5, "8-bit"), (5, 6, "7-bit"), (6, 12, "6-bit, default / cum5"), (12, 17, "6-bit, minimal"),
          (17, 18, "4-bit")]


def overlapping_ticks(fig, ax):
    """Pairs of x tick labels whose rendered boxes intersect (empty when the layout is clean)."""
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()]
    return [(i, i + 1) for i in range(len(boxes) - 1) if boxes[i].x1 > boxes[i + 1].x0]


def cells_figure_v2(folder, out, partial):
    data = rows("seed-cells.csv", folder)
    index = {(r["model"], r["format"], r["recipe"]): r for r in data}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5))
    for ax, (model, title) in zip(axes, MODELS.items()):
        mine = [index[(model, f, rec)] for f, rec in ORDER]
        assert len(mine) == len({(r["format"], r["recipe"]) for r in data if r["model"] == model}) == len(ORDER)
        ax.axhspan(-0.5, 0.5, color="#f1f0ec", zorder=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for i, r in enumerate(mine):
            full = float(r["expected_full_2000"])
            values = [float(r[f"expected_{s}"]) - full for s in SEEDS]
            ax.scatter(np.full(5, i), values, s=16, color=SERIES[0], edgecolor="white", linewidth=0.5, zorder=3)
            ax.scatter([i], [float(r["expected_mean"]) - full], marker="_", s=160, color=INK, zorder=4)
        for start, stop, _ in GROUPS[1:]:
            ax.axvline(start - 0.5, color=MUTED, linewidth=0.6, linestyle=":")
        low, high = ax.get_ylim()
        for start, stop, name in GROUPS:
            ax.text((start + stop - 1) / 2, high, name, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        labels = [f"{NAMES[r['format']]}\n{RECIPES[r['recipe']]}" for r in mine]
        ax.set_xticks(np.arange(len(mine)), labels, fontsize=6.5)
        ax.set_xlim(-0.6, len(mine) - 0.4)
        ax.set_ylabel("top-1 minus 2k value (pp)", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=8, color=INK, loc="left", pad=12)
        style(ax)
    fig.suptitle("Calibration seeds (5 disjoint 400-image subsets): expected-credit top-1 minus the 2,000-image "
                 "calibration.\nDots = seeds; bar = seed mean; band = +-delta/2 (0.5 pp). "
                 "1k screen, development evidence.", fontsize=8, color=INK)
    fig.tight_layout()
    fig.canvas.draw()
    for ax in axes:
        clash = overlapping_ticks(fig, ax)
        if clash:
            raise RuntimeError(f"x tick labels overlap: {clash}")
    save(fig, "b2-seeds-cells-v2", out, partial)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(SUMMARY))
    parser.add_argument("--out", default=str(FIGURES))
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    cells_figure_v2(args.summary, args.out, args.partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
