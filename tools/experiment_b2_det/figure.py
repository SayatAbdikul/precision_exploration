"""Figure of the detector study from the written summaries (CPU, no model, no new measurement).

``.venv/bin/python -m tools.experiment_b2_det.figure`` writes ``results/figures/b2-detector-v1.{png,pdf}``
(refuses to overwrite; ``--tag`` for a new versioned name).  Left: the INT8 path from the v1 recipe to the frozen
detector recipe on screen1k.  Right: the nine sentinel formats, sealed v1 max-abs against the frozen recipe.
Development evidence (COCO screen list).
"""
from __future__ import annotations

import argparse
import json

from tools.experiment_b.common import ROOT

SUMMARIES = ROOT / "results/summaries/b2-detector-v1"
FIGURES = ROOT / "results/figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1 and 2 (light surface)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e0"
LADDER = (("v1_maxabs", "v1 max-abs (sealed recipe)"), ("cum1_head_logits", "+ head wide after the outputs"),
          ("cum2_passthrough", "+ slice/pool/upsample pass-through"), ("cum3_unsigned", "+ unsigned codes"),
          ("cum4_act_mse", "+ activation MSE ranges"), ("cum5_weight_mse", "+ weight MSE ranges"),
          ("cum5_fused_silu", "+ fused SiLU"), ("default", "frozen recipe (scores etc. quantized)"),
          ("cum6_bias_correction", "cum5 + bias correction"), ("default_bias_correction", "frozen + bias correction"))


def load(name):
    return json.loads((SUMMARIES / f"{name}.json").read_text())["payload"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = [FIGURES / f"b2-detector-{args.tag}.{ext}" for ext in ("png", "pdf")]
    if any(t.exists() for t in targets):
        raise SystemExit(f"{targets[0]} exists; use --tag for a new versioned name")
    effects = {(r["format"], r["recipe"]): r for r in load("effects-1000")["rows"]}
    sentinels = load("sentinels-1000")
    fp32 = sentinels["fp32"]["map50_95"]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK, "font.family": "DejaVu Sans"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")
    for ax in (left, right):
        ax.set_facecolor("white")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.axvline(fp32, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
    # Left: INT8 path.
    rows = [(label, effects[("int8", name)]) for name, label in LADDER]
    for y, (label, row) in enumerate(rows):
        value = row["map50_95"]
        left.plot([value], [y], "o", color=BLUE, markersize=5.5, markeredgecolor="white", markeredgewidth=1)
        left.text(value + 0.5 if value < 37.5 else value - 0.5, y, f"{value:.1f}", va="center",
                  ha="left" if value < 37.5 else "right", color=INK, fontsize=7)
    left.set_yticks(range(len(rows)), [label for label, _ in rows])
    left.invert_yaxis()
    left.set_xlim(20, 41)
    left.set_xlabel("COCO mAP50-95, screen1k (INT8)")
    left.axhline(len(rows) - 2.5, color=GRID, linewidth=0.8)
    left.set_title("INT8: v1 recipe to frozen detector recipe", loc="left", fontsize=8.5, color=INK)
    # Right: sentinels, v1 max-abs against frozen recipe.
    table = [r for r in sentinels["rows"] if not r.get("missing")]
    table.sort(key=lambda r: -r["frozen"]["map50_95"])
    for y, row in enumerate(table):
        old, new = row["v1_maxabs"]["map50_95"], row["frozen"]["map50_95"]
        right.plot([old, new], [y, y], color=GRID, linewidth=2, solid_capstyle="round", zorder=1)
        right.plot([old], [y], "o", color=ORANGE, markersize=5.5, markeredgecolor="white", markeredgewidth=1, zorder=2,
                   label="v1 max-abs" if y == 0 else None)
        right.plot([new], [y], "o", color=BLUE, markersize=5.5, markeredgecolor="white", markeredgewidth=1, zorder=3,
                   label="frozen recipe" if y == 0 else None)
        right.text(41, y, f"{new:.1f}", va="center", ha="left", color=INK, fontsize=7)
    right.set_yticks(range(len(table)), [r["format"] for r in table])
    right.invert_yaxis()
    right.set_xlim(-1, 45)
    right.set_xlabel("COCO mAP50-95, screen1k")
    right.set_title("Sentinel formats", loc="left", fontsize=8.5, color=INK)
    right.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncol=2, frameon=False, fontsize=7,
                 handletextpad=0.2, columnspacing=1.0)
    fig.text(0.01, 0.01, f"YOLOv8n, COCO screen 1k (development evidence); dashed line: FP32 {fp32:.2f}",
             color=MUTED, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for target in targets:
        fig.savefig(target, dpi=200)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
