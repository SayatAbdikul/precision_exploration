"""Figures of the accumulator-width sweep (lane L8). CPU only; reads the sweep summary JSON.

    .venv/bin/python -m tools.analysis.accumulator_sweep_figures SUMMARY_JSON [--out results/figures] [--tag v1]

accumulator-sweep-top1-vs-width-<tag>.{png,pdf}: one panel per case, Top-1 (tie-aware expected credit, and lowest
    index) against the width W of the uniform saturating register sat.w<W>; exact arm as a horizontal line;
    certified absolute width W_cert and W_noevent marked.  Development evidence (ImageNet screen-1k).
accumulator-sweep-widths-<tag>.{png,pdf}: per case, the derived widths and the published closed-form width on one
    width axis.
Refuses to overwrite existing figures.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# reference palette of the dataviz method (categorical slots 1-5, light mode), text and surface tokens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

LABELS = {
    'resnet18-int8-default-b2': 'INT8 (B2)', 'resnet18-int8-default_signed-b2': 'INT8 signed act. (B2)',
    'resnet18-int6-default-b2': 'INT6 (B2)', 'resnet18-fp6_e2m3-default-b2': 'FP6 E2M3 (B2)',
    'resnet18-fp7_e3m3-default-b2': 'FP7 E3M3 (B2)', 'resnet18-fp8_e4m3fn-default-b2': 'FP8 E4M3 (B2)',
    'resnet18-fp8_e5m2-default-b2': 'FP8 E5M2 (B2)', 'resnet18-posit8_es1-default-b2': 'posit8 es1 (B2)',
    'resnet18-int8-maxabs-b1': 'INT8 (B1 max-abs)', 'resnet18-fp7_e3m3-maxabs-b1': 'FP7 E3M3 (B1 max-abs)'}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2); ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=3)
    ax.grid(axis='y', color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def uniform_points(summary, case):
    pts = sorted((r['parameter'], r['top1_expected'], r['top1_lowest_index'])
                 for r in summary['policies'] if r['case'] == case and r['family'] == 'uniform')
    return pts


def top1_vs_width(summary, path_stem):
    cases = [c['case'] for c in summary['cases'] if uniform_points(summary, c['case'])]
    widths = {w['case']: w for w in summary['widths']}
    n = len(cases); cols = 5 if n > 6 else 3; rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.95 * cols, 1.85 * rows + 0.55), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, case in zip(axes.flat, cases):
        style(ax)
        c = next(x for x in summary['cases'] if x['case'] == case)
        pts = uniform_points(summary, case)
        w = [p[0] for p in pts]
        ax.axhline(c['wide_top1_expected'], color=INK2, linewidth=0.8, linestyle=(0, (1, 1.5)))
        ax.plot(w, [p[2] for p in pts], color=ORANGE, linewidth=1.2, marker='s', markersize=2.6, label='lowest index')
        ax.plot(w, [p[1] for p in pts], color=BLUE, linewidth=2, marker='o', markersize=3.2, label='expected credit')
        cert = widths[case]['W_cert_abs']
        ax.axvline(cert, color=INK, linewidth=0.9, linestyle='--')
        ax.text(cert, 4, f' W_cert {cert}', fontsize=6, color=INK, ha='right' if cert > max(w) + 3 else 'left',
                va='bottom', rotation=90)
        ne = widths[case].get('W_noevent')
        if ne is not None:
            ax.plot([ne], [c['wide_top1_expected']], marker='v', color=INK2, markersize=4, linestyle='none')
        if cert > 63:
            ax.axvline(63, color=INK2, linewidth=0.8, linestyle=(0, (4, 2)))
            ax.text(63, 45, ' contract max 63', fontsize=6, color=INK2, rotation=90, va='bottom', ha='right')
        ax.set_xlim(min(w) - 1, max(cert, max(w)) + 1); ax.set_ylim(-2, 85)
        ax.set_title(LABELS.get(case, case), fontsize=7.5, color=INK, pad=3)
    for ax in axes[:, 0]:
        ax.set_ylabel('Top-1 (%)', fontsize=7, color=INK2)
    for ax in axes[-1, :]:
        ax.set_xlabel('register width W (bits)', fontsize=7, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK2, linewidth=0.8, linestyle=(0, (1, 1.5))),
                plt.Line2D([], [], color=INK, linewidth=0.9, linestyle='--'),
                plt.Line2D([], [], color=INK2, marker='v', linestyle='none', markersize=4)]
    labels += ['exact arm (wide)', 'certified width', 'W_noevent (1k)']
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, 'ResNet18, uniform saturating accumulator sat.w<W>; ImageNet screen-1k (development evidence)',
             ha='center', fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    write(fig, path_stem)


def widths_figure(summary, path_stem):
    rows = [w for w in summary['widths'] if w.get('W_noevent') is not None or w.get('W_half') is not None]
    marks = [('published_formula_bits', 'published formula [Agg24]', 'D', YELLOW),
             ('W_cert_abs', 'certified (absolute)', 's', INK),
             ('W_noevent', 'W_noevent', 'v', AQUA),
             ('W_same', 'W_same', 'o', BLUE),
             ('W_acc_1.0', 'W_acc(1.0)', '^', MAGENTA),
             ('W_half', 'W_half', 'x', ORANGE)]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(rows) + 1.1))
    fig.patch.set_facecolor(SURFACE); style(ax); ax.grid(axis='x', color=GRID, linewidth=0.6); ax.grid(axis='y', visible=False)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        vals = [r.get(k) for k, *_ in marks if r.get(k) is not None]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=2, zorder=1)
        for j, (k, label, m, col) in enumerate(marks):
            if r.get(k) is not None:  # small vertical offsets keep coinciding widths visible
                ax.plot([r[k]], [y + {'W_noevent': -0.14, 'W_same': 0.14}.get(k, 0.0)], marker=m, color=col, markersize=5, linestyle='none', zorder=3,
                        label=label if i == 0 else None, markeredgewidth=1.1)
        if r['W_cert_abs'] > 63:
            ax.text(63, y + 0.3, 'register cap 63 bits', fontsize=6, color=INK2, ha='center', va='bottom')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS.get(r['case'], r['case']) for r in reversed(rows)], fontsize=7, color=INK)
    ax.set_xlabel('accumulator width (bits, sign included)', fontsize=7, color=INK2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6.5, frameon=False, ncol=6, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
    fig.text(0.5, 0.995, 'ResNet18, uniform register sat.w<W>; derived widths on ImageNet screen-1k (development evidence)',
             ha='center', va='top', fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    write(fig, path_stem)


def write(fig, stem):
    for ext in ('png', 'pdf'):
        path = Path(f'{stem}.{ext}')
        if path.exists():
            raise SystemExit(f'{path} exists; figures are written once (use a new tag)')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=200 if ext == 'png' else None, facecolor=SURFACE)
    plt.close(fig)
    print(stem)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('summary')
    ap.add_argument('--out', default='results/figures')
    ap.add_argument('--tag', default='v1')
    args = ap.parse_args(argv)
    summary = json.load(open(args.summary))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    top1_vs_width(summary, f'{args.out}/accumulator-sweep-top1-vs-width-{args.tag}')
    widths_figure(summary, f'{args.out}/accumulator-sweep-widths-{args.tag}')


if __name__ == '__main__':
    main()
