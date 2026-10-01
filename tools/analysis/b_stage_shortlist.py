"""Build the evidence-linked B shortlist and figures from audited predictions.

Selection is explicitly post-development. This companion document never changes
the running comparison matrix or enrolls new inference jobs.
"""
from __future__ import annotations

import csv
from importlib.metadata import version
import os
from pathlib import Path

from tools.analysis.b_stage_deeper_comparisons import OUT, MODELS, RECIPES, compare_records, prediction_records
from tools.experiment_b.common import ROOT, dataset, digest, formats, seal, unseal
from tools.phase3.common import checked, reference

DEST = ROOT/'public/experiments/configs/breadth-study/b-deeper-comparisons-v1.json'
CLASSIFIERS = MODELS[:3]


def immutable(path, payload):
    if path.exists():
        if unseal(path) != payload:
            raise ValueError(f'published analysis/selection changed; use a new version: {path}')
    else:
        seal(path, payload)


def cohort(identifier, title, members, reason):
    return {'id': identifier, 'title': title, 'members': [list(x) for x in sorted(set(members))], 'reason': reason}


def cohorts():
    return [
        cohort('B2_recipe_anchors', 'Model-dependent recipe effects',
               [(m, 'int8', r) for m in MODELS for r in RECIPES],
               'Common conventional anchor across all models; inspect top1 and top5 separately and isolate detector head calibration.'),
        cohort('B2_six_bit', 'Six-bit representation and recipe interactions',
               [(m, f, r) for m in CLASSIFIERS for f in ('int6', 'bfp6', 'mxfp6_e3m2', 'fp6_e3m2', 'fp6_e2m3') for r in RECIPES],
               'Compare same nominal payload width under both recipes. Shared scale metadata, block grouping and support cost are additional factors.'),
        cohort('B2_seven_vs_eight', 'Seven-bit near-baseline candidates',
               [(m, f, 'maxabs') for m in MODELS for f in ('fp7_e3m3', 'fp8_e4m3fn')],
               'Small observed FP7/FP8 gaps deserve paired uncertainty and cost analysis; similar scores do not prove equivalence.'),
        cohort('B2_four_bit', 'Four-bit model/representation failure contrast',
               [('resnet18', f, r) for f in ('int4', 'nf4', 'mxfp4_e2m1') for r in RECIPES],
               'INT4, NF4 and MXFP4 differ substantially on ResNet18. Compare recipes and diagnose before attributing the gap to datatype alone.'),
        cohort('B2_detector_shared', 'Detector shared-scale alternatives',
               [('yolov8n', 'bfp6', r) for r in RECIPES] + [('yolov8n', 'mxfp8_e4m3', 'maxabs')],
               'Retain shared-format detector evidence despite lower quality than INT8; evaluate head-domain sensitivity and support costs.'),
    ]


def classifier_cross(left_key, right_key, index):
    left, right = index[left_key], index[right_key]
    if left['model'] != right['model']:
        raise ValueError('cross comparison changes model')
    n = min(left['images'], right['images'])
    rows = dataset('imagenet_screen_1k')[1][:n]
    candidates, baselines, contexts = [], [], []
    for row in (left, right):
        root = ROOT/'artifacts'/row['component']
        candidates.append(prediction_records(root, row['configuration_sha256'], rows))
        baselines.append(prediction_records(root, row['baseline_sha256'], rows))
        contexts.append(unseal(ROOT/row['configuration']['path'])['model_context'])
    if contexts[0] != contexts[1] or any(a['sample'] != b['sample'] or a['top5'] != b['top5'] for a,b in zip(*baselines)):
        raise ValueError('cross-component FP32 context or paired baseline outputs disagree')
    return {'left': list(left_key), 'right': list(right_key), 'images': n,
            'same_recipe': left['recipe'] == right['recipe'],
            'baseline_context_and_top5_verified_equal': True,
            'left_configuration': left['configuration_sha256'], 'right_configuration': right['configuration_sha256'],
            'statistics': compare_records(*candidates),
            'interpretation': 'selected descriptive configuration comparison; representation and support/scaling policy can both differ'}


def build():
    analysis = unseal(OUT/'analysis.json')
    checked(analysis['analysis_source'])
    checked(analysis['selected_plan'])
    index = {(r['model'], r['format'], r['recipe']): r for r in analysis['configurations']}
    groups = cohorts()
    keys = sorted({tuple(k) for c in groups for k in c['members']})
    tasks = []
    for key in keys:
        row = index[key]
        if row['images'] < 1000:
            tasks.append({'model': key[0], 'format': key[1], 'recipe': key[2],
                          'configuration': row['configuration'], 'configuration_sha256': row['configuration_sha256'],
                          'component': row['component'], 'retained_images': row['images'], 'target_images': 1000,
                          'additional_images': 1000-row['images'], 'state': 'selected_not_launched',
                          'reason': [c['id'] for c in groups if list(key) in c['members']],
                          'reuse': 'only original numerical identity and validated matching per-image checkpoints'})
    if len(keys) != 55 or len(tasks) != 12 or sum(t['additional_images'] for t in tasks) != 10464:
        raise ValueError('unexpected bounded shortlist size')
    diagnostic = [
        {'family': 'fixed_point', 'cases': [[m, 'q1_6'] for m in MODELS],
         'action': 'Analyze normalized level/scale correspondence and retain alias controls; do not count identical saved outputs as independent support.'},
        {'family': 'posit', 'cases': [['resnet18', f] for f in ('posit4_es0', 'posit6_es1', 'posit8_es1')],
         'action': 'Inspect scale/range, code occupancy and layerwise reconstruction; higher-bit collapse is a recipe/implementation diagnosis, not grounds to reject the family.'},
        {'family': 'logarithmic', 'cases': [['resnet18', 'log8'], ['mobilenet_v3_large', 'log8'], ['mobilenet_v2', 'log6']],
         'action': 'Keep viable LOG8 support-cost candidates and diagnose lower-bit losses; use retained predictions before allocating more images.'},
        {'family': 'binary', 'cases': [['mobilenet_v3_large', 'binary_pm1']],
         'action': 'Preserve collapse and counterexample evidence; code occupancy/calibration and arithmetic reachability come before more quality images.'},
        {'family': 'ternary', 'cases': [['mobilenet_v3_large', 'ternary']],
         'action': 'Reuse current E1 correctness evidence when complete; B collapse remains a separate recipe-quality issue.'},
    ]
    cross = []
    for m in CLASSIFIERS:
        cross.append(classifier_cross((m, 'fp8_e4m3fn', 'maxabs'), (m, 'fp7_e3m3', 'maxabs'), index))
    for m in CLASSIFIERS[:2]:
        for f in ('bfp6', 'fp7_e3m3', 'mxfp8_e4m3'):
            cross.append(classifier_cross((m, 'int8', 'maxabs'), (m, f, 'maxabs'), index))
    for m in CLASSIFIERS:
        cross.append(classifier_cross((m, 'int6', 'percentile_99_9'), (m, 'bfp6', 'percentile_99_9'), index))
    for f in ('nf4', 'mxfp4_e2m1'):
        cross.append(classifier_cross(('resnet18', 'int4', 'percentile_99_9'), ('resnet18', f, 'percentile_99_9'), index))
    for m in CLASSIFIERS:
        cross.append(classifier_cross((m, 'fp6_e3m2', 'maxabs'), (m, 'fp6_e2m3', 'maxabs'), index))
    detector = [unseal(p) for p in sorted((OUT/'detector-bootstrap').glob('*.json'))]
    if len(detector) != 3 or any(d['statistics']['resamples'] != 500 for d in detector):
        raise ValueError('all three bounded detector bootstrap comparisons must complete')
    for evidence in detector:
        checked(evidence['analysis_source'])
    # This is an offline level-table inspection, not a new calibration or run.
    import numpy as np
    from tools.experiment_b.quantizer import table
    grids = {}
    for name in ('int8', 'q1_6', 'posit8_es1', 'posit6_es1', 'posit4_es0'):
        levels, _, _ = table(name)
        grids[name] = {'maximum_finite_level': float(levels[-1]),
                       'positive_levels': int((levels > 0).sum()),
                       'levels_above_10_percent_of_positive_range': int((levels/levels[-1] > .1).sum())}
    a, ab, at = table('int8'); b, bb, bt = table('q1_6')
    grid_audit = {'tables': grids, 'int8_q1_6': {
        'q1_6_levels_times_64_equal_int8': bool(np.array_equal(b*64, a)),
        'q1_6_boundaries_times_64_equal_int8': bool(np.array_equal(bb*64, ab)),
        'identical_tie_preferences': bool(np.array_equal(bt, at))},
        'source': reference(ROOT/'tools/experiment_b/quantizer.py'),
        'interpretation': 'Power-of-two grid/scale correspondence explains observed INT8/Q1.6 aliasing under external scaling; no universal execution proof. Posit tail sparsity motivates a scale-policy diagnosis, not a demonstrated cause of task loss.'}
    supplemental = {'classifier_comparisons': cross, 'detector_comparisons': detector,
                    'offline_grid_audit': grid_audit,
                    'runtime_packages': {p: version(p) for p in ('numpy', 'scipy', 'pycocotools', 'matplotlib')},
                    'statistical_sources': [reference(ROOT/'public/analysis/phase3'/p) for p in ('statistics.py', 'coco_cache.py')],
                    'analysis_source': reference(Path(__file__)), 'upstream_analysis': reference(OUT/'analysis.json')}
    immutable(OUT/'deeper-paired-statistics.json', supplemental)
    proposal = {
        'version': 'b-deeper-comparisons-v1', 'status': 'selected_after_development_analysis; no_new_inference_launched',
        'purpose': 'bounded follow-up to completed selected B extensions; companion to frozen comparison-matrix-v1',
        'prior_matrix': reference(ROOT/'public/experiments/configs/breadth-study/comparison-matrix-v1.json'),
        'analysis': reference(OUT/'analysis.json'), 'deeper_statistics': reference(OUT/'deeper-paired-statistics.json'),
        'selection_source': reference(Path(__file__)), 'cohorts': groups,
        'core_existing_configurations': len(keys), 'balance_extensions': tasks,
        'additional_unchanged_configuration_images': sum(t['additional_images'] for t in tasks),
        'projected_coverage_after_extensions': {'existing_configurations': 200, 'at_1000': 183,
            'at_128': 17, 'recipe_pairs_at_1000': 83, 'scope': 'projection only; extensions not launched'},
        'diagnostic_family_retention': diagnostic,
        'new_recipe_pilot_proposal': {
            'cases': [['resnet18', 'int4'], ['mobilenet_v2', 'int4'], ['mobilenet_v2', 'int6']],
            'arms': ['calibration_MSE_scale_only', 'BRECQ_block_reconstruction_baseline'],
            'published_method_reference': 'https://openreview.net/forum?id=POWv6hDd9XH',
            'initial_images_per_new_arm': 128, 'nominal_new_arms': 6, 'nominal_candidate_images': 768,
            'status': 'requires_new_versioned_recipe_implementations_and_frozen_protocol',
            'conditions': ['Use independent training calibration with fixed budget and seeds.',
                           'Do not label the current exact-A rounding adaptation a BRECQ/AdaRound reproduction.',
                           'Match W/A precision and operator exceptions; if these change, add explicit matched controls outside the nominal six-arm budget.',
                           'Declare scalar and shared scale policies; do not apply an integer method to every format without validation.',
                           'Expand to 1000 only after a predeclared useful-effect and absolute-quality diagnostic gate.']},
        'nonuniform_scale_pilot_proposal': {
            'cases': [['resnet18', 'posit8_es1'], ['resnet18', 'nf4'], ['mobilenet_v2', 'fp6_e2m3']],
            'arm': 'calibration_reconstruction_MSE_scale_search_for_declared_codebook',
            'initial_images_per_new_arm': 128, 'nominal_new_arms': 3, 'nominal_candidate_images': 384,
            'reason': 'Test endpoint-scaling failure in Posit, codebook sensitivity in NF4, and exponent/mantissa recipe interaction in FP6.',
            'status': 'requires_frozen_scale_search_budget_and_new_recipe_identity',
            'conditions': ['Inspect cached calibration reconstruction and code occupancy first.',
                           'Keep the declared codebook, W/A precision, graph and operator exceptions fixed.',
                           'Fit on training calibration only; preserve FP32-QDQ semantics and report the method as a new B recipe.',
                           'No blanket claim that integer reconstruction methods apply unchanged to these grids.']},
        'nominal_new_recipe_pilot_total': {'arms': 9, 'images_per_arm': 128, 'candidate_images': 1152,
            'excludes': 'additional matched controls if published-method operator policies differ, detector diagnostics and calibration fitting'},
        'detector_followup': {'case': ['yolov8n', 'int8'], 'initial_images': 128,
            'action': 'Inspect clipping/calibration before and after DFL and decoded boxes, with separate score/box domains already present in B. Change one factor per new arm.',
            'state': 'diagnostic protocol needed; any head/operator exemption receives a new numerical identity'},
        'larger_panels': {'state': 'not_enrolled', 'requirements': ['Close key recipe pairing gaps first.',
            'Freeze claim-specific candidates, calibration seeds, data exposure and compute budget.',
            'Use larger development panels only where uncertainty changes selection; full benchmark confirmation remains E4.',
            'Keep all 100 datatype/model pairs in the ledger; no family is eliminated by this shortlist.']},
        'selection_rules': ['Model and family coverage, matched conventional anchors, same-bit alternatives, recipe sensitivity and uncertain useful-quality cases.',
                           'No automatic winner-only or p-value-only ranking; collapsed families receive diagnosis rather than large duplicate sweeps.',
                           'Shared-scale nominal bits exclude scale metadata and do not represent total hardware cost.',
                           'Paired statistical agreement is not proof of arithmetic or hardware equivalence.'],
        'execution': {'current_study_next_queue_unchanged': True, 'current_selected_b_enrollment_unchanged': True,
                      'runner_for_new_enrollment': 'not implemented in this analysis task',
                      'inference_launched': False},
    }
    covered = {index[k]['family'] for k in keys} | {d['family'] for d in diagnostic}
    if covered != {f['family'] for f in formats()}:
        raise ValueError('shortlist loses a datatype family')
    immutable(DEST, proposal)
    with (OUT/'selected-extensions.csv').open('w', newline='') as stream:
        fields = ['model', 'format', 'recipe', 'retained_images', 'target_images', 'additional_images', 'configuration_sha256']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(tasks)
    report(analysis, proposal, supplemental)
    plot(analysis, supplemental)
    print(f'Selected {len(keys)} existing configurations in {len(groups)} cohorts; {len(tasks)} extensions / 10464 new images')
    print(DEST)


def report(analysis, plan, supplemental):
    lines = ['# B-stage paired prediction analysis and deeper comparisons', '',
        '2026-09-27. Read-only analysis of saved predictions; no new inference launched.', '',
        'Audited all 200 configurations: 171 at 1,000 images and 29 at 128. All original 128-image prefixes and candidate/baseline seals, hashes, configuration identities and ordered sample contexts were verified. Classification metrics were recomputed. The three selected detector comparisons were independently recomputed with dataset-level COCO AP and 500 paired image-bootstrap draws.', '',
        '**Pairing:** 71 datatype/model pairs have both recipes at 1,000 images (46 classifier, 25 detector). The other 29 recipe comparisons use only 128 shared images. Full-panel classifier comparisons also retain separate 128-image selection and 872-image extension statistics.', '',
        '**Interpretation:** every result is B FP32-QDQ development evidence. Classifier intervals use 10,000 paired bootstrap draws; the equivalent three-outcome multinomial avoids repeated image allocations. Exact McNemar tests and Holm adjustment across the 75 classifier recipe comparisons are supplied as exploratory diagnostics. Neither adjustment nor the 872-image split turns adaptive development into independent final confirmation. Detector intervals are exploratory pointwise intervals from 500 draws; close claims need more resamples and confirmation.', '',
        '## Main findings', '',
        '- A single clipping recipe is not consistently best across architectures or formats. Inspect both top-1 and top-5: MobileNetV2 INT8 changes by -0.1 top-1 points but +4.3 top-5 points, with 283 changed top-1 predictions.',
        '- BFP6 warrants deeper comparison with INT6/INT8. Its nominal six-bit payload excludes shared scale metadata; the cost comparison must include support arithmetic and memory.',
        '- FP7 and FP8 E4M3 have small quality gaps under maxabs in all four models. This motivates uncertainty and hardware analysis, not a claim of equivalent quality.',
        '- ResNet18 has useful four-bit contrasts (INT4, NF4, MXFP4), while several MobileNet low-bit variants collapse. These results motivate architecture/recipe diagnosis rather than selecting one universally best format.',
        '- INT8 and Q1.6 have identical complete saved top-five lists/detections wherever both reach 1,000 images: seven model/recipe comparisons. The offline quantizer audit finds identical tie preferences and Q1.6 levels/boundaries exactly equal to INT8 divided by 64. External scaling can compensate for that factor. Preserve separate identities but avoid counting identical outputs as independent support; this is not a universal execution proof.',
        '- Posit6/8, binary and ternary frequently have near-zero quality under these recipes. The offline Posit8 table has only three positive levels above 10% of its positive maximum, versus 115 for INT8. This motivates testing endpoint-based scaling versus reconstruction-driven scaling; it does not establish the cause of accuracy collapse. Scale/range, code-occupancy and layerwise diagnostics come before larger evaluation panels.', '',
        '## Selected recipe comparisons', '',
        'Difference = percentile 99.9 minus maxabs, in percentage points. Scores are top-1 %, except YOLO which uses mAP50–95 points.', '',
        '| Model / format | Paired images | Maxabs | Percentile | Difference | Pointwise 95% interval |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    desired = {('resnet18', 'int8'), ('mobilenet_v2', 'int8'), ('mobilenet_v3_large', 'int8'),
               ('resnet18', 'bfp6'), ('mobilenet_v2', 'bfp6'), ('resnet18', 'mxfp4_e2m1'),
               ('mobilenet_v2', 'fp6_e2m3'), ('mobilenet_v3_large', 'fp6_e2m3')}
    for p in analysis['paired_recipe_comparisons']:
        if (p['model'], p['format']) in desired:
            lo, hi = p['pointwise_95_interval_pp']
            lines.append(f"| {p['model']} / {p['format']} | {p['images']} | {p['left_percent']:.2f} | {p['right_percent']:.2f} | {p['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    for d in supplemental['detector_comparisons']:
        s = d['statistics']['metrics']['map50_95']
        if d['left'][1] == d['right'][1]:
            lo, hi = [100*x for x in s['delta_interval']]
            lines.append(f"| yolov8n / {d['left'][1]} | 1000 | {100*s['fp32']:.3f} | {100*s['candidate']:.3f} | {100*s['delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', 'The 128-image rows above are deliberately not mixed with their one-sided 1,000-image scores.', '']
    for d in supplemental['detector_comparisons']:
        if d['left'][1] != d['right'][1]:
            s = d['statistics']['metrics']['map50_95']; lo, hi = [100*x for x in s['delta_interval']]
            lines.append(f"YOLO {d['left'][1]} → {d['right'][1]} under maxabs: {100*s['fp32']:.3f} → {100*s['candidate']:.3f} mAP points; difference {100*s['delta']:+.3f}, paired interval [{lo:+.3f}, {hi:+.3f}] on 1,000 images.")
    lines += ['',
        '## Selected same-recipe format comparisons', '',
        'Difference = right minus left. The baseline model context and every paired FP32 top-five list were checked across original/extension runners. These comparisons change representation and sometimes its required scaling/block policy; they do not isolate intrinsic number representation alone.', '',
        '| Model | Left | Right | Images | Left top-1 | Right top-1 | Difference | Pointwise 95% interval |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for p in supplemental['classifier_comparisons']:
        s = p['statistics']['top1']; lo, hi = s['pointwise_95_interval_pp']
        lines.append(f"| {p['left'][0]} | {p['left'][1]} ({p['left'][2]}) | {p['right'][1]} ({p['right'][2]}) | {p['images']} | {s['left_percent']:.2f} | {s['right_percent']:.2f} | {s['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '## Next enrollment: balance twelve recipe counterparts', '',
        'Five core comparison groups retain 55 existing configuration identities. Twelve need their missing 872-image extension; the other 43 already have 1,000 images. This costs **10,464 new candidate-image evaluations**, with no changed calibration or numerical recipe. Once completed, the overall existing ledger would contain 183 configurations at 1,000 images and 83 full recipe pairs; 17 other recipes would remain exploratory at 128.', '',
        '| Model | Format | Recipe to extend | Existing → target | New images |', '| --- | --- | --- | --- | ---: |']
    for t in plan['balance_extensions']:
        lines.append(f"| {t['model']} | {t['format']} | {t['recipe']} | 128 → 1000 | 872 |")
    lines += ['', '## Deeper comparisons selected', '', '| Group | Scope | Purpose |', '| --- | --- | --- |']
    for c in plan['cohorts']:
        lines.append(f"| {c['title']} | {len(c['members'])} configurations | {c['reason']} |")
    lines += ['', 'After balancing, the proposed integer new-recipe pilots are ResNet18 INT4 and MobileNetV2 INT4/INT6, with MSE scale-only and a [BRECQ block-reconstruction baseline](https://openreview.net/forum?id=POWv6hDd9XH): six new arms × 128 images = 768 evaluations.', '',
        'Also select three nonuniform scale-search pilots: ResNet18 Posit8, ResNet18 NF4 and MobileNetV2 FP6 E2M3, each at 128 images after an offline calibration reconstruction/code-occupancy audit. These add 384 evaluations and directly test scaling choices beyond integer formats. Keep each codebook, precision and graph fixed; optimize reconstruction using training calibration data only.', '',
        'Together these are **nine proposed new-recipe arms × 128 images = 1,152 candidate-image evaluations**. Implementations, calibration data, search budgets, operator policy and promotion thresholds must be frozen first. Changed operator exceptions require additional matched controls outside that nominal count. The exact-A E2 adaptation is not a reproduction of a published B baseline, and integer methods cannot be assumed to apply unchanged to nonuniform grids.', '',
        'Detector follow-up starts with YOLO INT8 head-domain/calibration diagnostics on 128 images. Existing B already separates score/box scale domains. Any new exemption or DFL/box policy gets a distinct configuration and matched control.', '',
        'All ten datatype families remain represented through either core comparisons or diagnosis. LOG8 remains a hardware-support candidate; Posit, binary and ternary need failure diagnosis; Q1.6 remains an alias/control case. The full 200-configuration ledger is preserved. No 5k/full-validation enrollment was selected before calibration/confirmation roles and budget are frozen.', '',
        '## Reproduction and artifacts', '',
        '- Analysis: `results/summaries/b-stage-paired-1k-v1/analysis.json`',
        '- Paired cross-format and detector statistics: `results/summaries/b-stage-paired-1k-v1/deeper-paired-statistics.json`',
        '- Complete CSV tables and selected extensions: `results/summaries/b-stage-paired-1k-v1/`',
        '- Selected companion protocol: `public/experiments/configs/breadth-study/b-deeper-comparisons-v1.json`',
        '- Figure: `results/figures/b-stage-recipe-effects-1k-v1.png`', '',
        'Run `.venv/bin/python -m tools.analysis.b_stage_deeper_comparisons analyze` to rebuild the audit. The detector-bootstrap subcommand records the three selected comparisons with seed 20260927 and 500 draws. Then run `.venv/bin/python -m tools.analysis.b_stage_shortlist`. Analysis uses saved data and CPU only. The existing inference matrix, runtime sources and controller enrollments remain unchanged; their resume command does not launch these twelve newly selected extensions.', '']
    path = ROOT/'docs/analysis/b-stage-deeper-comparisons-2026-09-27.md'
    path.write_text('\n'.join(lines))


def plot(analysis, supplemental=None):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fmts = formats()
    pairs = {(p['model'], p['format']): p for p in analysis['paired_recipe_comparisons']}
    values = np.array([[pairs[m, f['name']]['difference_pp'] for m in MODELS] for f in fmts])
    fig, ax = plt.subplots(figsize=(10, 12))
    limit = float(abs(values).max())
    im = ax.imshow(values, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    ax.set_xticks(range(4), ['ResNet18\nTop-1', 'MobileNetV2\nTop-1', 'MobileNetV3\nTop-1', 'YOLOv8n\nmAP50–95'])
    ax.xaxis.tick_top(); ax.set_yticks(range(len(fmts)), [f['name'] for f in fmts]); ax.tick_params(length=0, pad=8)
    for y, f in enumerate(fmts):
        for x, m in enumerate(MODELS):
            p = pairs[m, f['name']]
            label = f"{p['difference_pp']:+.1f}" + (' *' if p['images'] == 128 else '')
            ax.text(x, y, label, ha='center', va='center', fontsize=10, color='white' if abs(values[y, x]) > limit*.55 else '#18202b')
        if y and fmts[y-1]['family'] != f['family']:
            ax.axhline(y-.5, color='white', lw=2)
    ax.set_title('Recipe effects on paired saved predictions', loc='left', pad=52, fontsize=16)
    fig.colorbar(im, ax=ax, fraction=.045, pad=.04, label='Percentile 99.9 − maxabs (percentage points)')
    fig.text(.04, .045, '71 pairs: 1,000 shared images  •  29 pairs marked *: 128 shared images\n'
             'Positive favors percentile; negative favors maxabs. Unequal panels are never subtracted.\n'
             'Development FP32-QDQ results; descriptive differences, not exact-inference or final rankings.', fontsize=10)
    fig.subplots_adjust(left=.24, right=.9, top=.88, bottom=.13)
    base = ROOT/'results/figures/b-stage-recipe-effects-1k-v1'
    for extension in ['png', 'pdf']:
        fig.savefig(base.with_suffix('.'+extension), dpi=160, facecolor='white')
    plt.close(fig)
    if supplemental is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    lookup = {(tuple(p['left']), tuple(p['right'])): p for p in supplemental['classifier_comparisons']}
    labels = ['ResNet18', 'MobileNetV2', 'MobileNetV3']
    for ax, left, right, recipe, title in [
        (axes[0], 'int6', 'bfp6', 'percentile_99_9', 'BFP6 minus INT6\nBoth use percentile 99.9'),
        (axes[1], 'fp8_e4m3fn', 'fp7_e3m3', 'maxabs', 'FP7 E3M3 minus FP8 E4M3\nBoth use maxabs')]:
        for i, m in enumerate(CLASSIFIERS):
            s = lookup[((m, left, recipe), (m, right, recipe))]['statistics']['top1']
            delta = s['difference_pp']; lo, hi = s['pointwise_95_interval_pp']
            ax.errorbar(delta, i, xerr=[[delta-lo], [hi-delta]], fmt='o', color='#196785', capsize=5, markersize=7)
            ax.annotate(f'{delta:+.1f}', (delta, i), xytext=(0, 13), textcoords='offset points', ha='center', fontsize=11)
        ax.axvline(0, color='#9b3333', lw=1, ls='--')
        ax.set_yticks(range(3), labels); ax.set_ylim(2.55, -.65)
        ax.set_title(title, fontsize=12, pad=13)
        ax.set_xlabel('Top-1 difference (percentage points)')
        ax.grid(axis='x', alpha=.18); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_xlim(-2, 38); axes[1].set_xlim(-3, 3)
    fig.suptitle('Two priorities for deeper B comparisons', fontsize=16, y=.97)
    fig.text(.04, .035, '1,000 paired development images • Pointwise 95% paired-bootstrap intervals • FP32-QDQ simulation\n'
             'BFP6 requires shared-scale metadata/support costs. An interval crossing zero does not establish equivalence.', fontsize=10)
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.74, wspace=.42)
    base = ROOT/'results/figures/b-stage-priority-comparisons-v1'
    for extension in ['png', 'pdf']:
        fig.savefig(base.with_suffix('.'+extension), dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    build()
