"""Figures of the B2 format matrix, drawn from the summary tables (no model is run).

Usage: .venv/bin/python -m tools.analysis.b2_matrix_figures [--source DIR] [--out DIR] [--version v1]

* ``b2-matrix-quality-<version>``: heat map of expected-credit top-1 for every format, model and
  frozen recipe; colour is the drop against FP32 (one hue, clipped at 20 points), numbers are top-1.
* ``b2-matrix-vs-int8-<version>``: paired difference of every format against INT8 of the same recipe
  with its pointwise 95 percent interval and the one-point band of the protocol.  From ``v2`` on, the
  shared-exponent formats also get a row for their intrinsic arm (max-abs activation blocks, the arm used
  by the exact-engine proposal), the axis starts at -18, and a point estimate less than one point inside
  the left edge (or beyond it) is drawn as an off-scale arrow with its value instead of a dot at the edge,
  even when part of its interval is on scale (review 1, findings 2 and 8).
  ``--only vs-int8`` draws just this figure.

Existing figure files are never replaced; pass a new ``--version``.  Development evidence (1k screen).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from tools.experiment_b.common import ROOT  # noqa: E402

SOURCE = ROOT / "results/summaries/b2-matrix-v1"
FIGURES = ROOT / "results/figures"
MODELS = {"resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2", "mobilenet_v3_large": "MobileNetV3-Large"}
RECIPES = ("default", "minimal")
INTRINSIC = {"cum5_act_maxabs": "default", "cum1_fused": "minimal"}
# One-hue sequential ramp (light = near zero) and the two categorical slots, validated with the
# dataviz palette validator (light surface): blue #2a78d6, orange #eb6834.
RAMP = ("#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SERIES = {"default": ("#2a78d6", "o"), "minimal": ("#eb6834", "D")}
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CLIP_DROP, LEFT_LIMIT, RIGHT_LIMIT = 20.0, -16.0, 5.0


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(value):
    return None if value in ("", None) else float(value)


def format_order(cells):
    seen = {}
    for row in cells:
        if row["arm"] == "matrix":
            seen[row["format"]] = int(row["bits"])
    families = ["integer", "fixed_point", "float", "posit", "logarithmic", "bfp", "mx_float", "codebook", "ternary", "binary"]
    family = {row["format"]: row["family"] for row in cells}
    return sorted(seen, key=lambda f: (-seen[f], families.index(family[f]), f)), seen


def style(axis):
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=SECONDARY, length=0, labelsize=7.5)


def save(figure, out, name):
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    if any(path.exists() for path in paths):
        raise SystemExit(f"{paths[0].name} or {paths[1].name} exists; figures are not overwritten, pass a new --version")
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(paths[0], facecolor=SURFACE, bbox_inches="tight")
    figure.savefig(paths[1], facecolor=SURFACE, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return [str(path) for path in paths]


def quality(cells, out, version):
    names, bits = format_order(cells)
    models = [m for m in MODELS if any(row["model"] == m for row in cells)]
    fp32 = {row["model"]: float(row["top1_expected"]) for row in cells if row["arm"] == "baseline"}
    value = {}
    for row in cells:
        if row["arm"] == "matrix":
            value[(row["format"], row["model"], row["recipe"])] = float(row["top1_expected"])
        elif row["arm"] == "intrinsic":
            value[(row["format"] + " (max-abs act. blocks)", row["model"], INTRINSIC[row["recipe"]])] = float(row["top1_expected"])
    rows = []
    for name in names:
        rows.append(name)
        if (name + " (max-abs act. blocks)", models[0], "default") in value:
            rows.append(name + " (max-abs act. blocks)")
    columns = [(m, rc) for m in models for rc in RECIPES]
    cmap = LinearSegmentedColormap.from_list("b2_drop", RAMP)
    figure, axis = plt.subplots(figsize=(1.6 + 1.05 * len(columns), 0.4 + 0.25 * len(rows)), facecolor=SURFACE)
    style(axis)
    for y, name in enumerate(rows):
        for x, (model, recipe) in enumerate(columns):
            top1 = value.get((name, model, recipe))
            if top1 is None:
                continue
            drop = min(CLIP_DROP, max(0.0, fp32[model] - top1))
            gap = 0.04 + (0.12 if x % 2 == 0 and x else 0.0)
            axis.add_patch(plt.Rectangle((x + gap, y + 0.04), 0.96 - gap, 0.92, color=cmap(drop / CLIP_DROP), linewidth=0))
            axis.text(x + 0.5 + gap / 2, y + 0.5, f"{top1:.1f}", ha="center", va="center", fontsize=7.5,
                      color="#ffffff" if drop / CLIP_DROP > 0.5 else INK)
    axis.set_xlim(0, len(columns))
    axis.set_ylim(len(rows), 0)
    axis.set_xticks([x + 0.5 for x in range(len(columns))])
    axis.set_xticklabels([rc for _, rc in columns])
    axis.xaxis.tick_top()
    axis.set_yticks([y + 0.5 for y in range(len(rows))])
    axis.set_yticklabels([f"{name}" for name in rows])
    for index, model in enumerate(models):
        axis.text(2 * index + 1.03, -1.45, f"{MODELS[model]}\nFP32 {fp32[model]:.1f}", ha="center", va="bottom",
                  fontsize=8, color=INK, linespacing=1.3)
    # thin separators between bit-width groups
    previous = None
    for y, name in enumerate(rows):
        width = bits[name.split(" ")[0]]
        if previous is not None and width != previous:
            axis.plot([0, len(columns)], [y, y], color=AXIS, linewidth=0.8)
        if width != previous:
            axis.text(len(columns) + 0.08, y + 0.5, f"{width} bit", ha="left", va="center", fontsize=7.5, color=MUTED)
        previous = width
    colorbar = figure.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, CLIP_DROP)), ax=axis,
                               orientation="horizontal", fraction=0.025, pad=0.02, aspect=40)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7.5, colors=SECONDARY, length=2)
    colorbar.set_label("top-1 drop against FP32 in points (clipped at 20); cell text: expected-credit top-1, 1k screen",
                       fontsize=7.5, color=SECONDARY)
    return save(figure, out, f"b2-matrix-quality-{version}")


def against_int8(cells, out, version, intrinsic_rows=False):
    names, bits = format_order(cells)
    names = [name for name in names if name != "int8"]
    fix_edge = intrinsic_rows
    left = -18.0 if intrinsic_rows else LEFT_LIMIT  # v2: room for the R18 E3M2/Log6 dots near -15
    models = [m for m in MODELS if any(row["model"] == m for row in cells)]
    data = {(row["format"], row["model"], row["recipe"]): row for row in cells if row["arm"] == "matrix"}
    if intrinsic_rows:
        for row in cells:
            if row["arm"] == "intrinsic":
                data[(row["format"] + " (max-abs act. blocks)", row["model"], INTRINSIC[row["recipe"]])] = row
        expanded = []
        for name in names:
            expanded.append(name)
            if (name + " (max-abs act. blocks)", models[0], "default") in data:
                expanded.append(name + " (max-abs act. blocks)")
                bits[name + " (max-abs act. blocks)"] = bits[name]
        names = expanded
    figure, axes = plt.subplots(1, len(models), figsize=(2.2 + 2.35 * len(models), 0.9 + 0.24 * len(names)), sharey=True,
                                facecolor=SURFACE, squeeze=False)
    for axis, model in zip(axes[0], models):
        style(axis)
        axis.axvspan(-1, 1, color="#f0efec", linewidth=0)
        axis.axvline(0, color=AXIS, linewidth=1)
        for x in range(-15, int(RIGHT_LIMIT), 5):
            if x:
                axis.axvline(x, color=GRID, linewidth=0.6, zorder=0)
        previous = None
        for y, name in enumerate(names):
            if previous is not None and bits[name] != previous:
                axis.axhline(y - 0.5, color=GRID, linewidth=0.8)
            previous = bits[name]
            for offset, recipe in ((-0.17, "default"), (0.17, "minimal")):
                row = data.get((name, model, recipe))
                if row is None:
                    continue
                color, marker = SERIES[recipe]
                d, low, high = (number(row[k]) for k in ("d_int8", "d_int8_low", "d_int8_high"))
                if high < left + 1.2 or (fix_edge and d < left + 1.0):  # off scale: an arrow at the edge with the value
                    axis.plot([left + 0.7], [y + offset], marker="<", color=color, markersize=4.5, linestyle="none")
                    axis.text(left + 1.5, y + offset, f"{d:.0f}", va="center", ha="left", fontsize=6, color=SECONDARY)
                    continue
                axis.plot([max(low, left), high], [y + offset] * 2, color=color, linewidth=1.2, solid_capstyle="round")
                axis.plot([max(d, left)], [y + offset], marker=marker, color=color, markersize=4.2, linestyle="none",
                          markeredgecolor=SURFACE, markeredgewidth=0.6)
        axis.set_xlim(left, RIGHT_LIMIT)
        axis.set_xticks(range(-15, int(RIGHT_LIMIT) + 1, 5))
        axis.set_ylim(len(names) - 0.5, -0.5)
        axis.set_title(MODELS[model], fontsize=9, color=INK, pad=6)
        axis.set_xlabel("top-1 minus INT8 (points)", fontsize=7.5, color=SECONDARY)
    axes[0][0].set_yticks(range(len(names)))
    axes[0][0].set_yticklabels(names)
    handles = [Line2D([0], [0], color=SERIES[rc][0], marker=SERIES[rc][1], markersize=4.5, linewidth=1.2, label=f"{rc} recipe")
               for rc in RECIPES]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#f0efec", label="within one point of INT8"))
    handles.append(Line2D([0], [0], color=MUTED, marker="<", markersize=4.5, linestyle="none", label="off scale (value printed)"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.005),
                  labelcolor=SECONDARY)
    figure.subplots_adjust(wspace=0.06, bottom=0.085)
    return save(figure, out, f"b2-matrix-vs-int8-{version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=str(FIGURES))
    parser.add_argument("--version", default="v1")
    parser.add_argument("--only", choices=("all", "quality", "vs-int8"), default="all")
    parser.add_argument("--tag", default="", help="suffix of the summary tables to read (as in b2_matrix --tag)")
    args = parser.parse_args()
    suffix = f"-{args.tag}" if args.tag else ""
    cells = read(Path(args.source) / f"cells{suffix}.csv")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42})
    revised = args.version != "v1"
    paths = [] if args.only == "vs-int8" else quality(cells, Path(args.out), args.version)
    if args.only != "quality":
        paths += against_int8(cells, Path(args.out), args.version, intrinsic_rows=revised)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
