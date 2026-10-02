"""Lane Q4 figures (CPU) from results/summaries/b2-attrib-v1/ (written once by summarize.py --write).

    PYTHONPATH=. .venv/bin/python -m tools.experiment_b2_attrib.figures [--out results/figures] [--draft DIR]

Writes results/figures/b2-attrib-<name>-v1.{png,pdf}; refuses to overwrite.  ``--draft DIR`` builds the summary
in memory and writes the figures into DIR instead (for checking before the summaries are written).
Development evidence: ImageNet 1k screen, top-1 with expected credit for tied logits, paired-bootstrap 95 % CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.experiment_b.common import ROOT  # noqa: E402

SUMMARY = ROOT / "results/summaries/b2-attrib-v1"
# Reference categorical palette (dataviz skill, light mode), slots 1-3, plus neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR = {"mobilenet_v3_large": BLUE, "mobilenet_v2": ORANGE, "resnet18": AQUA}
MODEL_LABEL = {"mobilenet_v3_large": "MobileNetV3-L", "mobilenet_v2": "MobileNetV2", "resnet18": "ResNet18"}
FMT_LABEL = {"int8": "INT8", "int6": "INT6", "fp6_e2m3": "FP6 E2M3", "fp8_e4m3fn": "FP8 E4M3", "fp6_e3m2": "FP6 E3M2",
             "posit6_es1": "posit6", "log6": "LOG6"}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load(draft):
    if draft:
        from . import summarize
        return json.loads(json.dumps(summarize.build(), default=float))
    return {p.stem: json.loads(p.read_text()) for p in SUMMARY.glob("*.json")}


def save(fig, out, name):
    for ext in ("png", "pdf"):
        path = out / f"b2-attrib-{name}-v1.{ext}"
        if path.exists():
            raise SystemExit(f"{path} exists: figures are written once (use a new version)")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def groups_figure(s, out):
    a1 = s["a1"]
    fmts = [f for f in ("int8", "int6", "fp6_e2m3") if f in a1]
    order = sorted(a1["int6"]["role"]["groups"], key=lambda g: a1["int6"]["role"]["groups"][g]["damage_pp"]) \
        if "int6" in a1 else sorted(a1[fmts[0]]["role"]["groups"])
    fig, axes = plt.subplots(1, len(fmts), figsize=(3.3 * len(fmts), 4.2), sharey=True)
    for ax, fmt in zip(axes, fmts):
        res = a1[fmt]["role"]
        for i, g in enumerate(order):
            row = res["groups"].get(g)
            if row is None:
                continue
            ci_d = row["only_minus_weights_only"]["ci95_pp"]
            ci_r = row["wide_minus_default"]["ci95_pp"]
            ax.errorbar([row["damage_pp"]], [i + 0.17], xerr=[[row["damage_pp"] + ci_d[1]], [-ci_d[0] - row["damage_pp"]]],
                        fmt="o", ms=4, color=BLUE, elinewidth=1.2, capsize=0, label="damage (only G quantised)" if i == 0 else None)
            ax.errorbar([row["recovery_pp"]], [i - 0.17], xerr=[[row["recovery_pp"] - ci_r[0]], [ci_r[1] - row["recovery_pp"]]],
                        fmt="s", ms=4, color=ORANGE, elinewidth=1.2, capsize=0, label="recovery (G kept FP32)" if i == 0 else None)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{FMT_LABEL[fmt]}: default {res['default_top1']:.1f}, act. loss {res['activation_loss_pp']:.1f} pp",
                     fontsize=8.5, color=INK)
        ax.set_xlabel("top-1 points (screen, 95 % CI)", fontsize=8, color=INK)
        style(ax)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([g.replace("_", " ") for g in order], fontsize=8)
    axes[min(1, len(axes) - 1)].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("MobileNetV3-L: where the activation loss lives (role groups, B2 default recipe)", fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, out, "mbv3-groups")


def repairs_figure(s, out):
    b = s["b"]
    fmts = [f for f in ("int8", "int6", "fp6_e2m3", "fp8_e4m3fn") if b.get(f)]
    arms = []
    for fmt in fmts:
        for name in b[fmt]:
            if name != "ref_default" and name not in arms:
                arms.append(name)
    arms.sort(key=lambda a: (a.count("+"), a.split(":")[0], a))
    fig, axes = plt.subplots(1, len(fmts), figsize=(2.9 * len(fmts), 0.28 * len(arms) + 1.6), sharey=True)
    for ax, fmt in zip(axes, fmts):
        table = b[fmt]
        default = table["ref_default"]["top1_expected"]
        fp32 = default - table["ref_default"]["minus_fp32"]["diff_pp"]
        for i, name in enumerate(arms):
            row = table.get(name)
            if row is None:
                continue
            d, (lo, hi) = row["minus_default"]["diff_pp"], row["minus_default"]["ci95_pp"]
            color = BLUE if d > 0 and lo > 0 else (ORANGE if hi < 0 else MUTED)
            ax.errorbar([default + d], [i], xerr=[[d - lo], [hi - d]], fmt="o", ms=4, color=color, elinewidth=1.2)
        ax.axvline(default, color=MUTED, linestyle="--", linewidth=0.9)
        ax.axvline(fp32, color=INK, linestyle=":", linewidth=0.9)
        ax.set_title(f"{FMT_LABEL[fmt]} (default {default:.1f}; FP32 {fp32:.1f})", fontsize=8.5, color=INK)
        ax.set_xlabel("top-1 expected, % (screen)", fontsize=8, color=INK)
        style(ax)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels(arms, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("MobileNetV3-L uniform-precision repairs (dashed: frozen default, dotted: FP32; blue/orange: 95 % CI above/below default)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out, "mbv3-repairs")


def predictor_figure(s, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    entries = [(f"mobilenet_v3_large/{fmt}", v["role"]) for fmt, v in s["a1"].items()] + list((s.get("c") or {}).items())
    for key, res in entries:
        model = key.split("/")[0]
        xs = [g["nsr_db"] for g in res["groups"].values() if g["nsr_db"] is not None]
        ys = [g["damage_pp"] for g in res["groups"].values() if g["nsr_db"] is not None]
        left.scatter(xs, ys, s=14, color=MODEL_COLOR[model], alpha=0.8, linewidths=0)
    for model, color in MODEL_COLOR.items():
        left.scatter([], [], s=14, color=color, label=MODEL_LABEL[model])
    pool = s.get("group_sqnr_predicts_effect", {}).get("nsr_db_vs_damage_pp") or {}
    left.set_title(f"group NSR vs damage (pooled Spearman {pool.get('rho', float('nan')):.2f}, n={pool.get('n', 0)})",
                   fontsize=8.5, color=INK)
    left.set_xlabel("group SQNR from summed stored noise-to-signal (dB; higher = less noise)", fontsize=8, color=INK)
    left.set_ylabel("damage: weights-only minus only:G (top-1 pp)", fontsize=8, color=INK)
    left.legend(fontsize=7, frameon=False)
    style(left)
    left.grid(axis="y", color=GRID, linewidth=0.6)
    rows = s["a0_cross_cell"]["rows"]
    for model, color in MODEL_COLOR.items():
        sub = [r for r in rows if r["model"] == model]
        right.scatter([r["median_sqnr_db"] for r in sub], [r["top1_drop_pp"] for r in sub], s=14, color=color,
                      label=MODEL_LABEL[model], linewidths=0)
    rho = s["a0_cross_cell"]["spearman"]["pooled"]["median_sqnr_db"]["spearman_rho_vs_drop"]
    right.set_title(f"sealed default cells: median SQNR vs top-1 drop (pooled Spearman {rho:.2f})", fontsize=8.5, color=INK)
    right.set_xlabel("median per-boundary SQNR (dB)", fontsize=8, color=INK)
    right.set_ylabel("top-1 drop to FP32 (pp)", fontsize=8, color=INK)
    right.set_yscale("symlog", linthresh=1)
    style(right)
    right.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    save(fig, out, "sqnr-predictor")


def loo_figure(s, out):
    a2 = {f: v for f, v in (s.get("a2") or {}).items() if v}
    if not a2:
        return
    fmts = list(a2)
    fig, axes = plt.subplots(1, len(fmts), figsize=(3.3 * len(fmts), 5.2), sharey=False)
    axes = axes if len(fmts) > 1 else [axes]
    for ax, fmt in zip(axes, fmts):
        nodes = sorted(a2[fmt]["nodes"].items(), key=lambda kv: kv[1]["recovery_pp"])
        for i, (node, row) in enumerate(nodes):
            d, (lo, hi) = row["wide_minus_default"]["diff_pp"], row["wide_minus_default"]["ci95_pp"]
            color = BLUE if row["role"] == "stem" else ORANGE if row["role"] == "hswish" else MUTED
            ax.errorbar([d], [i], xerr=[[d - lo], [hi - d]], fmt="o", ms=3.5, color=color, elinewidth=1)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels([n for n, _ in nodes], fontsize=6)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{FMT_LABEL[fmt]}: one node kept FP32", fontsize=8.5, color=INK)
        ax.set_xlabel("recovery vs default (top-1 pp)", fontsize=8, color=INK)
        style(ax)
    fig.suptitle("MobileNetV3-L leave-one-out (blue: stem nodes, orange: Hardswish outputs)", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out, "mbv3-leave-one-out")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results/figures"))
    parser.add_argument("--draft", default=None, help="build in memory and write into this folder instead")
    args = parser.parse_args(argv)
    s = load(args.draft)
    out = Path(args.draft or args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups_figure(s, out)
    repairs_figure(s, out)
    predictor_figure(s, out)
    loo_figure(s, out)
    print("written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
