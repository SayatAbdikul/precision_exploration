"""Evidence-linked report and figure for the twelve balanced B extensions."""
from __future__ import annotations

import json
import os
from pathlib import Path

from tools.analysis.b_stage_shortlist import classifier_cross
from tools.experiment_b.common import ROOT, seal, unseal
from tools.phase3.common import checked, reference

OLD = ROOT/'results/summaries/b-stage-paired-1k-v1'
NEW = ROOT/'results/summaries/b-stage-paired-1k-v2'
PLAN = ROOT/'artifacts/breadth_study/balanced_b_v1/plan.json'
DOC = ROOT/'docs/analysis/b-stage-balanced-results-2026-09-27.md'
FIG = ROOT/'results/figures/b-stage-balanced-recipe-effects-v2'


def build():
    previous, current, plan = unseal(OLD/'analysis.json'), unseal(NEW/'analysis.json'), unseal(PLAN)
    for ref in (current['analysis_source'], current['selected_plan'], current['balance_plan']):
        checked(ref)
    if current['coverage'] != {'configurations': 200, 'at_1000': 183, 'at_128': 17,
                                'datatype_model_pairs': 100, 'paired_at_1000': 83,
                                'classifier_pairs_at_1000': 58}:
        raise ValueError('balanced coverage was not achieved')
    selected = {(t['model'], t['format'], t['recipe']): t for g in plan['groups'].values() for t in g['tasks']}
    if len(selected) != 12:
        raise ValueError('balanced selection must have exactly twelve identities')
    old_rows = {(r['model'], r['format'], r['recipe']): r for r in previous['configurations']}
    rows = {(r['model'], r['format'], r['recipe']): r for r in current['configurations']}
    pairs = {(r['model'], r['format']): r for r in current['paired_recipe_comparisons']}
    changes = []
    for key, task in selected.items():
        before, after = old_rows[key], rows[key]
        if (before['images'] != 128 or after['images'] != 1000
                or before['configuration_sha256'] != after['configuration_sha256']
                or before['prediction_digest'] != task['prediction_digest_128']):
            raise ValueError(f'original 128-image evidence or identity changed: {key}')
        pair = pairs[key[:2]]
        if pair['images'] != 1000 or pair['available_images'] != [1000, 1000]:
            raise ValueError(f'balanced recipe counterpart is missing: {key}')
        changes.append({'model': key[0], 'format': key[1], 'extended_recipe': key[2],
                        'configuration_sha256': task['configuration_sha256'],
                        'original_128_top1_percent': before['candidate_percent'],
                        'full_1000_top1_percent': after['candidate_percent'],
                        'new_872_top1_percent': after['extension_872_vs_fp32']['top1']['right_percent'],
                        'recipe_pair_1000': pair})
    # Recompute the earlier, preselected classifier comparisons on their new common panels.
    comparisons = []
    models = ('resnet18', 'mobilenet_v2', 'mobilenet_v3_large')
    for model in models:
        comparisons.append(classifier_cross((model, 'fp8_e4m3fn', 'maxabs'), (model, 'fp7_e3m3', 'maxabs'), rows))
    for model in models[:2]:
        for fmt in ('bfp6', 'fp7_e3m3', 'mxfp8_e4m3'):
            comparisons.append(classifier_cross((model, 'int8', 'maxabs'), (model, fmt, 'maxabs'), rows))
    for model in models:
        for recipe in ('maxabs', 'percentile_99_9'):
            comparisons.append(classifier_cross((model, 'int6', recipe), (model, 'bfp6', recipe), rows))
        comparisons.append(classifier_cross((model, 'fp6_e3m2', 'maxabs'), (model, 'fp6_e2m3', 'maxabs'), rows))
    for recipe in ('maxabs', 'percentile_99_9'):
        for fmt in ('nf4', 'mxfp4_e2m1'):
            comparisons.append(classifier_cross(('resnet18', 'int4', recipe), ('resnet18', fmt, recipe), rows))
    if any(r['images'] != 1000 for r in comparisons):
        raise ValueError('selected deeper classifier comparisons remain unpaired')
    detector = [reference(p) for p in sorted((OLD/'detector-bootstrap').glob('*.json'))]
    if len(detector) != 3:
        raise ValueError('three original detector bootstrap results are missing')
    for ref in detector:
        checked(ref)
    evidence = {'version': 'b-stage-balanced-v2', 'analysis_source': reference(Path(__file__)),
                'previous_analysis': reference(OLD/'analysis.json'),
                'balanced_analysis': reference(NEW/'analysis.json'), 'balance_plan': reference(PLAN),
                'coverage': current['coverage'], 'extended_configurations': changes,
                'paired_cross_format': comparisons, 'retained_detector_bootstrap': detector,
                'scope': 'development FP32-QDQ; selected after seeing 128-image data; no independent confirmation'}
    seal(NEW/'balanced-evidence-ledger.json', evidence)
    report(changes, comparisons, current)
    plot(changes)
    print(json.dumps({'coverage': current['coverage'], 'extensions': len(changes),
                      'cross_format_pairs': len(comparisons), 'report': str(DOC)}, indent=2))


def report(changes, comparisons, current):
    lines = ['# B-stage balanced comparison results', '',
             '2026-09-27. The twelve frozen existing configurations were extended from 128 to 1,000 evaluation images with the original numerical runners, unchanged identities and per-image checkpoints. The complete audited B ledger now has 183 of 200 configurations at 1,000 images; 17 remain at 128. Both recipes have 1,000 paired images for 83 datatype/model pairs, including 58 classifier pairs.', '',
             'These are development FP32-QDQ results. The twelve arms were selected after inspecting their first 128 outcomes. Pointwise paired 95% bootstrap intervals describe this panel; they are not final confirmation. Accuracy differences below are percentage points (pp), with percentile 99.9 minus maxabs.', '',
             '| Model | Format | Extended recipe | 128 top-1 % | New 872 top-1 % | Full 1,000 top-1 % | Recipe difference pp [95% interval] |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in changes:
        pair = row['recipe_pair_1000']; lo, hi = pair['pointwise_95_interval_pp']
        lines.append(f"| {row['model']} | {row['format']} | {row['extended_recipe']} | {row['original_128_top1_percent']:.2f} | {row['new_872_top1_percent']:.2f} | {row['full_1000_top1_percent']:.2f} | {pair['difference_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', 'The 128 and new-872 columns expose selection-panel versus extension-panel behavior; their different sample sizes and image identities should not be directly subtracted as a treatment effect.', '',
              '## Selected cross-format comparisons', '',
              'Difference is right minus left on the same 1,000 images and the same saved FP32 baseline. A change in format may also change scaling or block support policy.', '',
              '| Model | Left | Right | Top-1 difference pp [95% interval] |',
              '| --- | --- | --- | ---: |']
    for row in comparisons:
        top = row['statistics']['top1']; lo, hi = top['pointwise_95_interval_pp']
        left, right = row['left'], row['right']
        lines.append(f"| {left[0]} | {left[1]} ({left[2]}) | {right[1]} ({right[2]}) | {top['difference_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', 'The three previously sealed detector paired-bootstrap results are retained by hash in the new evidence ledger. No detector predictions or detector bootstrap were rerun for this balance step.', '',
              '## Reproduce', '',
              '- Resume the exact twelve configurations: `.venv-b/bin/python -m tools.run.balanced_b run`',
              '- Audit all saved predictions: `.venv/bin/python -m tools.analysis.b_stage_balanced_comparisons analyze`',
              '- Rebuild this report and figures: `.venv/bin/python -m tools.analysis.b_stage_balanced_report`', '',
              'Evidence: `results/summaries/b-stage-paired-1k-v2/analysis.json`, `balanced-evidence-ledger.json` in the same directory, and `results/figures/b-stage-balanced-recipe-effects-v2.png`.', '']
    DOC.write_text('\n'.join(lines))


def plot(changes):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'resnet18': '#196785', 'mobilenet_v2': '#b35e20', 'mobilenet_v3_large': '#6d54a2'}
    for i, row in enumerate(changes):
        pair = row['recipe_pair_1000']; value = pair['difference_pp']
        lo, hi = pair['pointwise_95_interval_pp']
        ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', capsize=4,
                    color=colors[row['model']], markersize=6)
    ax.axvline(0, ls='--', lw=1, color='#555')
    ax.set_yticks(range(len(changes)), [f"{r['model']} / {r['format']}" for r in changes])
    ax.invert_yaxis()
    ax.set_xlabel('Top-1 percentile 99.9 − maxabs (percentage points)')
    ax.set_title('Recipe effects after balancing twelve B comparisons', loc='left')
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.06, .02, '1,000 paired development images • Pointwise 95% paired-bootstrap intervals • FP32-QDQ', fontsize=9)
    fig.subplots_adjust(left=.34, right=.96, top=.9, bottom=.13)
    for suffix in ('png', 'pdf'):
        fig.savefig(FIG.with_suffix('.'+suffix), dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    build()
