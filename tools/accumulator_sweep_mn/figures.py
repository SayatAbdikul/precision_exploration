"""Figures of the MobileNet accumulator sweep (write-once).

    .venv/bin/python -m tools.accumulator_sweep_mn.figures [SUMMARY_DIR] [--out-prefix results/figures/accumulator-sweep-mn] [--tag v1]

Row 1: expected-credit Top-1 (1k screen, development) against the uniform register width W, one panel per format;
MobileNetV2 and MobileNetV3-Large as two series, ResNet18 (lane L8, results/summaries/accumulator-sweep-v1/policies.csv)
as a thin gray reference; the certified width of each network as a vertical tick line in its colour; the exact arm as a
dotted line.  Row 2: Top-1 against the per-node deficit d of sat.struct-<d>.
Colours: categorical slots 1-2 of the dataviz reference palette (blue, orange); text in neutral ink.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'mobilenet_v2': '#2a78d6', 'mobilenet_v3_large': '#eb6834', 'resnet18': '#8a8985'}
LABELS = {'mobilenet_v2': 'MobileNetV2', 'mobilenet_v3_large': 'MobileNetV3-L', 'resnet18': 'ResNet18 (L8)'}
FORMATS = ('int8', 'int6', 'fp6_e2m3', 'fp7_e3m3', 'fp8_e4m3fn')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load(summary_dir):
    s = json.load(open(Path(summary_dir) / 'summary.json'))
    l8 = []
    path = ROOT / 'results/summaries/accumulator-sweep-v1/policies.csv'
    if path.exists():
        l8 = [r for r in csv.DictReader(open(path)) if r['family'] == 'uniform']
    l8w = {r['case']: r for r in s.get('resnet18_l8_widths', [])}
    return s, l8, l8w


def series(rows, case, family):
    pts = sorted((int(r['parameter']), float(r['top1_expected'])) for r in rows
                 if r['case'] == case and r['family'] == family)
    return [p[0] for p in pts], [p[1] for p in pts]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('summary_dir', nargs='?', default=str(ROOT / 'results/summaries/accumulator-sweep-mn-v1'))
    ap.add_argument('--out-prefix', default=str(ROOT / 'results/figures/accumulator-sweep-mn'))
    ap.add_argument('--tag', default='v1')
    args = ap.parse_args(argv)
    out = Path(f'{args.out_prefix}-top1-vs-width-{args.tag}')
    if any(out.with_suffix(x).exists() for x in ('.png', '.pdf')):
        sys.exit(f'{out}.png/.pdf exist; figures are written once')
    s, l8, l8w = load(args.summary_dir)
    widths = {w['case']: w for w in s['widths']}
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, len(FORMATS), figsize=(13, 5.6), constrained_layout=True)
    for j, fmt in enumerate(FORMATS):
        ax, bx = axes[0, j], axes[1, j]
        for net in ('mobilenet_v2', 'mobilenet_v3_large'):
            case = f'{net}-{fmt}-default-b2'
            w = widths.get(case)
            if w is None:
                continue
            stress = w.get('stress_case')
            style = dict(color=COLORS[net], lw=2, marker='o', ms=4, ls='--' if stress else '-',
                         label=LABELS[net] + (' (stress)' if stress else ''))
            x, y = series(s['policies'], case, 'uniform')
            if x:
                ax.plot(x, y, **style)
            ax.axvline(w['W_cert_abs'], color=COLORS[net], lw=1, ls=(0, (1, 2)))
            ax.axhline(w['wide_top1_expected'], color=COLORS[net], lw=0.8, ls=':', alpha=0.7)
            d, y2 = series(s['policies'], case, 'per_node')
            if d:
                bx.plot(d, y2, **style)
            bx.axhline(w['wide_top1_expected'], color=COLORS[net], lw=0.8, ls=':', alpha=0.7)
        rcase = f'resnet18-{fmt}-default-b2'
        x, y = series(l8, rcase, 'uniform')
        if x:
            ax.plot(x, y, color=COLORS['resnet18'], lw=1, marker='.', ms=3, label=LABELS['resnet18'])
            if rcase in l8w:
                ax.axvline(int(l8w[rcase]['W_cert_abs']), color=COLORS['resnet18'], lw=1, ls=(0, (1, 2)))
        ax.set_title(fmt.upper().replace('_', ' '), color=INK)
        ax.set_xlabel('uniform register width W (bits, sign incl.)')
        bx.set_xlabel('per-node deficit d (sat.struct-d)')
        for a in (ax, bx):
            a.set_ylim(-2, 80); a.grid(axis='y', color=GRID, lw=0.6)
        if j == 0:
            ax.set_ylabel('Top-1 % (expected credit, 1k dev)')
            bx.set_ylabel('Top-1 % (expected credit, 1k dev)')
            ax.legend(frameon=False, loc='lower right', fontsize=7)
    fig.suptitle('Top-1 against accumulator width; dotted vertical = certified lossless width, dotted horizontal = exact arm',
                 color=MUTED, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(out.with_suffix(ext), dpi=200)
    print(out)


if __name__ == '__main__':
    main()
