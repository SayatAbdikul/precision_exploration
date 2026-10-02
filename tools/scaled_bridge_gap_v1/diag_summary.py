"""Summarise the step-4 diagnostic records (DIAGNOSTIC; harness not bit-certified between endpoints)."""
from __future__ import annotations
from collections import defaultdict
import json
from scipy.stats import binomtest
from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
from .common import *
from .analysis import correctness, directions
from .diagnostics import DIAG_PROTOCOL, diag_protocol_path, configurations, SINGLE, STAGES


def case(name):
    folder = BASE / 'diagnostics' / name
    recs = [unseal(p) for p in sorted(folder.glob('*.json'))]
    if [r['start'] for r in recs] != list(range(0, 8 * len(recs), 8)):
        raise ValueError('noncontiguous diagnostic batches')
    n = 8 * len(recs)
    labels = [int(unseal(arm_folder(name, 'B') / f'{i:04d}.json')['sample']['label']) for i in range(n)]
    preds = defaultdict(list); local = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))
    prop = defaultdict(lambda: defaultdict(lambda: [0, 0])); prop_images = defaultdict(lambda: defaultdict(int))
    for r in recs:
        for label, e in r['configs'].items():
            preds[label] += e['top5']
            for k, v in e.get('local_flips', {}).items():
                for j in range(5):
                    local[label][k][j] += v[j] if j < 4 else 0
            for k, v in e.get('propagated_flips', {}).items():
                prop[label][k][0] += v[0]; prop[label][k][1] += v[1]
    from .analysis import execution_order
    smap = execution_order(name)[1]; stage = smap.__getitem__  # corrected (execution-graph) stages for per-stage tallies
    out = {'format': name, 'images': n, 'configs': {},
           'stage_tallies': 'per-stage flip tallies use the corrected execution-graph stages (residual adds in their block); the '
                            'Xstage:<s> configurations are the diag-1 runs AS RUN, whose stem included all eight residual adds '
                            '(see stage_hybrids_corrected.json for the rerun)'}
    for label in configurations():
        p = preds[label]
        row = {}
        for ref in ('B', 'X'):
            t1 = paired_outcomes(correctness(preds[ref], labels, 'top1'), correctness(p, labels, 'top1'))
            t5 = paired_outcomes(correctness(preds[ref], labels, 'top5'), correctness(p, labels, 'top5'))
            row[f'minus_{ref}'] = {'top1_pp': t1['difference_pp'], 'top1_95': t1['pointwise_95_interval_pp'],
                                   'top5_pp': t5['difference_pp'], 'mcnemar_p': t1['mcnemar_exact_p'],
                                   'changes': directions(preds[ref], p, labels)}
        row['top1_percent'] = 100 * sum(correctness(p, labels, 'top1')) / n
        row['top5_percent'] = 100 * sum(correctness(p, labels, 'top5')) / n
        if label in local or label in SINGLE:
            L = local[label]
            up = sum(v[0] for v in L.values()); down = sum(v[1] for v in L.values())
            mu = sum(v[2] for v in L.values()); md = sum(v[3] for v in L.values())
            row['local_flips'] = {'up': up, 'down': down, 'magnitude_up': mu, 'magnitude_down': md,
                                  'sign_test_p': float(binomtest(up, up + down).pvalue) if up + down else 1.0,
                                  'magnitude_test_p': float(binomtest(mu, mu + md).pvalue) if mu + md else 1.0,
                                  'per_node': {k: v[:4] for k, v in L.items()},
                                  'per_stage': {s: [sum(v[j] for k, v in L.items() if stage(k) == s) for j in range(4)]
                                                for s in STAGES}}
        if label in prop:
            P = prop[label]
            row['propagated_flips_per_stage'] = {s: [sum(v[j] for k, v in P.items() if stage(k) == s) for j in range(2)]
                                                 for s in STAGES}
            row['propagated_flips_fc'] = P.get('fc', [0, 0])
        out['configs'][label] = row
    out['seconds'] = sum(r['seconds'] for r in recs)
    return out


def write():
    if unseal(diag_protocol_path()) != DIAG_PROTOCOL:
        raise ValueError('diagnostic protocol drift')
    cases = [case(n) for n in FORMATS if (BASE / 'diagnostics' / n).exists()]
    data = {'revision': 'r2 (2026-10-01): per-stage tallies use corrected stages (independent review 1, nonblocking 2); r1 file kept at '
                        'artifacts/scaled_bridge_gap_v1/superseded/diagnostics-pre-stage-fix.json',
            'protocol': reference(diag_protocol_path()), 'certification': 'diagnostic harness; endpoints checked against B and wide records '
            'on every image (run_case raises otherwise); mixed configurations not certified', 'sources': sources(), 'cases': cases}
    RESULTS.mkdir(parents=True, exist_ok=True)
    seal(RESULTS / 'diagnostics.json', data)
    for c in cases:
        print('##', c['format'], c['images'], f"{c['seconds']:.0f}s")
        for label, r in c['configs'].items():
            lf = r.get('local_flips')
            ch = r['minus_B']['changes']
            print(f"{label:14s} top1 {r['top1_percent']:.1f} vsB {r['minus_B']['top1_pp']:+.1f} {r['minus_B']['top1_95']} "
                  f"c2w {ch['correct_to_wrong']} w2c {ch['wrong_to_correct']} vsX {r['minus_X']['top1_pp']:+.1f}"
                  + (f" | local up/down {lf['up']}/{lf['down']} p{lf['sign_test_p']:.2g} mag+/- {lf['magnitude_up']}/{lf['magnitude_down']}" if lf else ''))
    return data
