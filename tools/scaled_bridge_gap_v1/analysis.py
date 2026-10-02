"""Paired 1k analysis of the gap-study arms (read-only over sealed records)."""
from __future__ import annotations
from collections import Counter
import csv
import io
import json
import statistics
import numpy as np
from scipy.stats import beta, binomtest
from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
from .common import *

ARMS = ('B', 'wide', 'control')


def load(name):
    out = {}
    for arm in ARMS:
        paths = sorted(arm_folder(name, arm).glob('*.json'))
        recs = [unseal(p) for p in paths]
        if [r['index'] for r in recs] != list(range(len(recs))):
            raise ValueError(f'noncontiguous records {name} {arm}')
        out[arm] = recs
    return out


def upper95(k, n):
    """Clopper-Pearson one-sided 95% upper bound on a binomial rate."""
    return float(1 - .05 ** (1 / n)) if k == 0 else float(beta.ppf(.95, k + 1, n - k))


def first_divergence(left, right, order=None):
    """First node, in `order` (execution order), whose stored-code hash differs; None if none differs.

    The sealed records store `layers` with sorted keys (alphabetical), so callers must pass the
    execution order (execution_order()).  Review 1 found that the original version iterated the
    sealed key order; summary.json written before the fix is kept under artifacts/.../superseded/."""
    nodes = list(left['layers']) if order is None else order
    if set(nodes) != set(left['layers']) or set(nodes) != set(right['layers']):
        raise ValueError('first_divergence: order does not cover the recorded layers')
    for node in nodes:
        if left['layers'][node]['codes'] != right['layers'][node]['codes']:
            return node
    return None


def execution_order(name):
    """Recorded layer names in execution order, and the corrected stage of each (diagnostics.exec_stages)."""
    from tools.scaled_bridge_v1.export import load_export
    from .diagnostics import exec_stages
    ex = load_export(name)[0]
    return [n['name'] for n in ex['nodes']], exec_stages(ex['nodes'])


def directions(base, other, labels):
    """Changed top-1 predictions of `other` relative to `base`, by correctness transition."""
    c = Counter()
    for p, q, y in zip(base, other, labels):
        if p[0] == q[0]:
            continue
        c['correct_to_wrong' if p[0] == y else 'wrong_to_correct' if q[0] == y else 'wrong_to_other_wrong'] += 1
    k, m = c['wrong_to_correct'], c['correct_to_wrong']
    return {'changed_top1': sum(c.values()), **{x: c[x] for x in ('correct_to_wrong', 'wrong_to_correct', 'wrong_to_other_wrong')},
            'sign_test_two_sided_p': float(binomtest(k, k + m).pvalue) if k + m else 1.0}


def correctness(preds, labels, metric):
    return [int(p[0] == y) if metric == 'top1' else int(y in p) for p, y in zip(preds, labels)]


def compare(preds, labels, left, right, sl):
    out = {}
    for metric in ('top1', 'top5'):
        a = correctness(preds[left][sl], labels[sl], metric); b = correctness(preds[right][sl], labels[sl], metric)
        out[metric] = paired_outcomes(a, b)
    out['top1_prediction_changes'] = directions(preds[left][sl], preds[right][sl], labels[sl])
    out['changed_ordered_top5'] = sum(p != q for p, q in zip(preds[left][sl], preds[right][sl]))
    return out


PAIRS = (('B', 'wide'), ('wide', 'control'), ('FP32', 'wide'), ('FP32', 'control'), ('FP32', 'B'), ('B', 'control'))


def case(name):
    d = load(name); n = min(len(v) for v in d.values())
    labels = [int(r['sample']['label']) for r in d['B'][:n]]
    for arm in ARMS:
        if [r['sample'] for r in d[arm][:n]] != [r['sample'] for r in d['B'][:n]]:
            raise ValueError('arms are not paired')
    preds = {arm: [r['top5'] for r in d[arm][:n]] for arm in ARMS}
    preds['FP32'] = [r['FP32_top5'] for r in d['B'][:n]]
    slices = {'all': slice(0, n), 'v1_prefix_0_127': slice(0, min(n, 128)), 'new_128_999': slice(128, n)}
    result = {'format': name, 'images': n, 'comparisons': {}}
    for label, sl in slices.items():
        if sl.stop - sl.start <= 0:
            continue
        result['comparisons'][label] = {f'{r}_minus_{l}': compare(preds, labels, l, r, sl) for l, r in PAIRS}
    cw = result['comparisons']['all']['control_minus_wide']
    k_pred = cw['top1_prediction_changes']['changed_top1']
    k_corr = cw['top1']['left_only_correct'] + cw['top1']['right_only_correct']
    k_top5 = cw['changed_ordered_top5']
    numer = sum(a['layers'] != b['layers'] or a['output'] != b['output'] for a, b in zip(d['wide'][:n], d['control'][:n]))
    result['control_vs_wide_discordance'] = {
        'n': n, 'changed_top1_predictions': k_pred, 'changed_top1_correctness': k_corr, 'changed_ordered_top5': k_top5,
        'images_with_any_numerical_difference': numer,
        'one_sided_95_upper_percent': {'changed_top1_prediction': 100 * upper95(k_pred, n),
                                       'changed_top1_correctness': 100 * upper95(k_corr, n),
                                       'changed_ordered_top5': 100 * upper95(k_top5, n),
                                       'any_numerical_difference': 100 * upper95(numer, n)},
        'method': 'Clopper-Pearson one-sided 95% (zero events: 1-0.05^(1/n))'}
    per_image = []; first_wb = Counter(); first_cw = Counter(); first_wb_by_dir = {}; stage_wb_by_dir = {}
    node_disagree = Counter()
    order, stages = execution_order(name)
    order = [x for x in order if x in d['wide'][0]['layers']]
    for i in range(n):
        w, c, b = d['wide'][i], d['control'][i], d['B'][i]
        fwb = first_divergence(w, b, order); fcw = first_divergence(w, c, order)
        for node in w['layers']:
            if w['layers'][node]['codes'] != b['layers'][node]['codes']:
                node_disagree[node] += 1
        first_wb[fwb] += 1; first_cw[fcw] += 1
        y = labels[i]
        trans = ('same' if preds['wide'][i][0] == preds['B'][i][0] else
                 'B_correct_to_wide_wrong' if preds['B'][i][0] == y else
                 'B_wrong_to_wide_correct' if preds['wide'][i][0] == y else 'wrong_to_other_wrong')
        first_wb_by_dir.setdefault(trans, Counter())[fwb] += 1
        stage_wb_by_dir.setdefault(trans, Counter())[stages.get(fwb, 'none')] += 1
        per_image.append({'index': i, 'sha256': d['B'][i]['sample']['sha256'], 'label': y,
                          'FP32_top1': preds['FP32'][i][0], 'B_top1': preds['B'][i][0], 'wide_top1': preds['wide'][i][0],
                          'control_top1': preds['control'][i][0], 'wide_vs_B_transition': trans,
                          'first_code_divergence_wide_vs_B': fwb, 'first_code_divergence_control_vs_wide': fcw})
    result['layers'] = {'node_order': order, 'node_order_source': 'execution order of the sealed v1 export graph',
                        'first_code_divergence_wide_vs_B_stage_by_transition': {t: dict(c) for t, c in stage_wb_by_dir.items()},
                        'first_code_divergence_wide_vs_B': {k or 'none': v for k, v in first_wb.items()},
                        'first_code_divergence_wide_vs_B_by_transition': {t: {k or 'none': v for k, v in c.items()} for t, c in first_wb_by_dir.items()},
                        'first_code_divergence_control_vs_wide': {k or 'none': v for k, v in first_cw.items()},
                        'images_with_code_disagreement_wide_vs_B_per_node': dict(node_disagree)}
    timing = {}
    for arm in ('wide', 'control'):
        timing[arm] = {'median_execution_seconds': statistics.median(r['timing']['execution'] for r in d[arm][:n]),
                       'total_execution_seconds': sum(r['timing']['execution'] for r in d[arm][:n])}
    timing['B'] = {'total_execution_seconds': sum(d['B'][i]['execution_batch_seconds'] for i in range(0, n, 8))}
    result['timing'] = timing
    result['v1_reproduced_images'] = {arm: sum(bool(r.get('v1_reproduced')) for r in d[arm][:n]) for arm in ARMS}
    return result, per_image


def jobs():
    path = BASE / 'jobs.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    rows = [r for r in rows if r['args'].split()[0] in ('arm', 'replay')]  # the 1k arms only (diagnostic jobs are reported separately)
    return {'jobs': len(rows), 'failed': sum(r['exit'] != 0 for r in rows),
            'locked_gpu_seconds': sum(float(r['locked_seconds']) for r in rows),
            'lock_wait_seconds': sum(float(r['acquired']) - float(r['requested']) for r in rows)}


def write_results():
    require_protocol()
    cases = []; rows = []
    for name in FORMATS:
        r, p = case(name); cases.append(r)
        rows += [{'format': name, **x} for x in p]
    data = {'revision': 'r2 (2026-10-01): first code divergence computed in execution order (independent review 1, finding B1); '
                        'cost restricted to the 1k arm jobs; the r1 file is kept at '
                        'artifacts/scaled_bridge_gap_v1/superseded/summary-pre-first-divergence-fix.json',
            'protocol': reference(protocol_path()), 'protocol_version': PROTOCOL['version'], 'sources': sources(),
            'v1_source_digest': V1_SOURCE_DIGEST, 'cases': cases, 'cost': jobs()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    seal(RESULTS / 'summary.json', data)
    text = io.StringIO(); w = csv.DictWriter(text, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (RESULTS / 'per_image.csv').write_text(text.getvalue())
    for c in cases:
        a = c['comparisons']['all']; t = a['wide_minus_B']
        print(c['format'], c['images'], 'B', round(t['top1']['left_percent'], 2), 'wide', round(t['top1']['right_percent'], 2),
              'control', round(a['control_minus_wide']['top1']['right_percent'], 2), 'FP32', round(a['wide_minus_FP32']['top1']['left_percent'], 2),
              'w-B', round(t['top1']['difference_pp'], 2), t['top1']['pointwise_95_interval_pp'], t['top1_prediction_changes'])
    return data
