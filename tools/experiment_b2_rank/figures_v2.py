"""Lane Q2 figure after review 1 (addendum-2): Kendall tau-b heat map from rank-correlation-tol-v2.csv.

  .venv/bin/python -m tools.experiment_b2_rank.figures_v2 [--summary results/summaries/b2-rank-v1] [--out results/figures]

* b2-rank-tau-heatmap-v2.png: tau-b between recipe orderings over the formats at or above 1 percent under either
  recipe of each pair (ties declared at 10 decimals of mean credit), one panel per network.  Replaces
  b2-rank-tau-heatmap.png (all 25 formats, chance ties broken by summation order), which stays as written.
"""
from __future__ import annotations

import argparse

import numpy as np

from tools.experiment_b.common import ROOT
from . import figures as fg


def tau_heatmap_v2(summary, out):
    corr = [r for r in fg.rows(summary / "rank-correlation-tol-v2.csv")
            if r["class"] == "all" and r["subset"] == "above_chance"]
    fig, axes = fg.plt.subplots(1, 3, figsize=(13.5, 4.9), facecolor=fg.SURFACE, constrained_layout=True)
    for ax, model in zip(axes, fg.MODELS):
        present = [v for v in fg.VIEWS if any(r["model"] == model and v in (r["view_a"], r["view_b"]) for r in corr)]
        n = len(present)
        grid = np.full((n, n), np.nan)
        np.fill_diagonal(grid, 1.0)
        for r in corr:
            if r["model"] == model and r["view_a"] in present and r["view_b"] in present:
                i, j = present.index(r["view_a"]), present.index(r["view_b"])
                grid[i, j] = grid[j, i] = float(r["kendall_tau_b"])
        grid[np.triu_indices(n, 1)] = np.nan
        cmap = fg.DIVERGING.copy()
        cmap.set_bad(fg.SURFACE)
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                if i > j and np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color=fg.TEXT)
        ax.set_xticks(range(n), [fg.SHORT[v] for v in present], rotation=60, ha="right", fontsize=6.5, color=fg.TEXT)
        ax.set_yticks(range(n), [fg.SHORT[v] for v in present], fontsize=6.5, color=fg.TEXT)
        ax.set_title(fg.NICE[model], fontsize=9, color=fg.TEXT)
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(image, ax=axes, shrink=0.7)
    bar.set_label("Kendall tau-b, formats >= 1% under either recipe (point estimate)", fontsize=7, color=fg.TEXT)
    bar.ax.tick_params(labelsize=6, colors=fg.MUTED)
    fig.suptitle("Format orderings under each recipe, above-chance formats (U unsigned, W MSE weight range, "
                 "B bias correction); 1k screen, development evidence", fontsize=8.5, color=fg.MUTED)
    fg.save_once(fig, out / "b2-rank-tau-heatmap-v2.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/summaries/b2-rank-v1")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    tau_heatmap_v2(ROOT / args.summary, ROOT / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
