"""Figure of the detector breadth study: results/figures/b2-detector-breadth-v1.{png,pdf} (write-once).

Reads results/summaries/b2-detector-breadth-v1/{formats-1000,seeds-1000,attribution-1000}.json.
(a) 25 formats under `default` and `conformant` (dagger: nearest available recipe); (b) the five calibration
subsets per format and recipe beside the full 2000-image value; (c) gain of keeping one group wide at 6 bits.

    .venv/bin/python -m tools.experiment_b2_det_breadth.figure [--tag r2]
"""
from __future__ import annotations

import argparse
import json

from tools.experiment_b.common import ROOT

from .summary import OUT

FIG = ROOT / "results/figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def load(name, tag):
    suffix = f"-{tag}" if tag else ""
    return json.loads((OUT / f"{name}{suffix}.json").read_text())


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="summary tag to read")
    parser.add_argument("--out-tag", default="v1")
    parser.add_argument("--set-apart", default="", help="comma-separated formats drawn below a rule, marked with a "
                        "double dagger and left out of the ordering (v2: q1_6, whose default record is bit-identical "
                        "to L7's INT8 default_signed)")
    args = parser.parse_args(argv)
    targets = [FIG / f"b2-detector-breadth-{args.out_tag}.{e}" for e in ("png", "pdf")]
    if any(t.exists() for t in targets):
        raise SystemExit("refusing to overwrite the figure (use --out-tag)")
    formats = load("formats-1000", args.tag)
    seeds = load("seeds-1000", args.tag)
    attribution = load("attribution-1000", args.tag)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (a) the 25 formats
    ax = axes[0]
    rows = [r for r in formats["rows"] if r.get("status") == "measured"]
    value = {(r["format"], r["recipe"]): r for r in rows}
    apart = [f for f in args.set_apart.split(",") if f]
    names = sorted({r["format"] for r in rows} - set(apart), key=lambda f: value[(f, "default")]["map50_95"])
    ypos = list(range(len(names)))
    if apart:
        ypos = [y + len(apart) + 0.6 for y in ypos]
        names = apart + names
        ypos = list(range(len(apart))) + ypos
        ax.axhline(len(apart) - 0.2, color=MUTED, lw=0.8, ls=":")
    for y, f in zip(ypos, names):
        d = value[(f, "default")]
        c = value.get((f, "conformant"))
        if c is not None:
            ax.plot([d["map50_95"], c["map50_95"]], [y, y], color=GRID, lw=2, zorder=1)
            ax.scatter(c["map50_95"], y, s=30, marker="s", color=ORANGE, edgecolor="white", lw=1, zorder=3)
        ax.scatter(d["map50_95"], y, s=30, marker="o", color=BLUE, edgecolor="white", lw=1, zorder=3)
    labels = [f + (" †" if value.get((f, "conformant"), {}).get("rule_conformance", "").startswith("nearest") else "")
              + ("  (chance)" if value[(f, "default")]["at_chance"] else "")
              + (" ‡ (= INT8 signed-code recipe)" if f in apart else "") for f in names]
    ax.set_yticks(ypos, labels)
    ax.axvline(formats["fp32_map50_95"], color=MUTED, lw=1, ls="--")
    ax.text(formats["fp32_map50_95"], max(ypos) + 0.6, " FP32", color=MUTED, va="bottom")
    ax.scatter([], [], marker="o", color=BLUE, label="default (frozen)")
    ax.scatter([], [], marker="s", color=ORANGE, label="conformant († nearest available)")
    ax.legend(loc="upper left" if apart else "lower right", frameon=False)
    ax.set_xlabel("mAP50-95 (COCO 1k screen, development evidence)")
    if apart:
        ax.set_title(f"(a) YOLOv8n, {len(names) - len(apart)} formats (‡ set apart)", loc="left", color=INK)
        ax.text(0.0, -0.14, "‡ q1_6 default is bit-identical to INT8 default_signed (signed-code recipe on the INT8\n"
                "grid); its position is not a format ranking.", transform=ax.transAxes, color=MUTED, fontsize=7,
                va="top")
    else:
        ax.set_title("(a) YOLOv8n, 25 formats", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)

    # (b) calibration subsets
    ax = axes[1]
    srows = sorted(seeds["rows"], key=lambda r: (r["format"], r["recipe"]))
    for y, r in enumerate(srows):
        color = BLUE if r["recipe"] == "default" else ORANGE
        values = [v for k, v in r.items() if k.startswith("subset_s")]
        ax.scatter(values, [y] * len(values), s=14, color=color, alpha=0.8, zorder=3)
        if r.get("full_2000") is not None:
            ax.scatter(r["full_2000"], y, s=40, marker="D", facecolor="none", edgecolor=INK, lw=1, zorder=4)
        ax.text(1.0, y, f"SD {r['sd']:.2f}", transform=ax.get_yaxis_transform(), ha="left", va="center", color=MUTED, fontsize=7)
    ax.set_yticks(range(len(srows)), [f"{r['format']} {r['recipe']}" for r in srows])
    ax.scatter([], [], s=14, color=BLUE, label="400-image subset, default")
    ax.scatter([], [], s=14, color=ORANGE, label="400-image subset, conformant")
    ax.scatter([], [], s=40, marker="D", facecolor="none", edgecolor=INK, label="full 2000-image set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=7)
    ax.set_xlabel("mAP50-95")
    ax.set_title("(b) Five disjoint calibration subsets", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)

    # (c) attribution (groups, then the finer units inside the worst group)
    from matplotlib.patches import Patch
    ax = axes[2]
    arows = [r for r in attribution["rows"] if r["stage"] in ("group", "fine")]
    names = {"weights_only": "all activations wide", "activations_only": "all weights wide"}

    def key(r):
        return (r["group"], r.get("unit"))

    keys = list(dict.fromkeys(key(r) for r in arows))
    keys = ([k for k in keys if k[1] is None and k[0] not in names] + [k for k in keys if k[1] is not None]
            + [k for k in keys if k[0] in names])
    for fmt, color, dy in (("int6", BLUE, -0.18), ("fp6_e2m3", ORANGE, 0.18)):
        for y, k in enumerate(keys):
            r = next((x for x in arows if x["format"] == fmt and key(x) == k), None)
            if r is None or r["gain_over_default"] is None:
                continue
            gain = r["gain_over_default"]
            ax.barh(y + dy, gain["delta"], height=0.34, color=color, zorder=2)
            if gain["interval"]:
                ax.plot(gain["interval"], [y + dy] * 2, color=INK, lw=0.8, zorder=3)
    labels = [names.get(g, g.replace("_", " ")) if u is None else f"{g} > {u.replace('_', ' ')}" for g, u in keys]
    ax.set_yticks(range(len(keys)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("gain over default when kept wide (mAP points, 95% CI)")
    ax.set_title("(c) Where the 6-bit loss lives (diagnostic)", loc="left", color=INK)
    ax.legend(handles=[Patch(color=BLUE, label="INT6"), Patch(color=ORANGE, label="FP6 E2M3")],
              loc="center right", frameon=False)
    ax.grid(axis="x", color=GRID, lw=0.6)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for target in targets:
        fig.savefig(target, dpi=200, bbox_inches="tight")
    print("\n".join(str(t.relative_to(ROOT)) for t in targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
