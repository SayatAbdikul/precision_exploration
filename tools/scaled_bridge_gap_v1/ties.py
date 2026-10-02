"""Top-k tie audit: how much of the wide-versus-B difference is tie-breaking between equal quantized logits?

POST-HOC, EXPLORATORY (requested after the 1k analysis).  The logits of every
arm are reconstructions of the stored codes of the final fc node, so two classes
whose fc codes are equal have exactly equal logits, and the arms' `topk` calls
(B: batched CUDA FP32; engine: per-image CPU binary64) order them differently.

For every image and every diagnostic configuration this re-runs the diagnostic
harness (no teacher forcing), checks both endpoints exactly as run_case does
(all-B == B replay codes at every node + Top-5; all-X == sealed wide codes at
every node + Top-5), checks that each configuration's own Top-5 equals the one
already recorded by the diagnostic run (when that record exists), and records:
  tie1        classes whose fc code equals the maximum (sorted); len > 1 => top-1 tie
  common_top5 Top-5 under one deterministic rule for every arm:
              larger fc code first, then smaller class index
  tie5        True when the 5th and 6th fc codes are equal (Top-5 membership depends on the rule)
"""
from __future__ import annotations
import json
import time
import numpy as np
from .common import *
from .diagnostics import Harness, configurations, topk, ALL_B, ALL_X, diag_protocol_path, DIAG_PROTOCOL

TIES_PROTOCOL = {
 'version': 'scaled-bridge-gap-ties-1',
 'date': '2026-10-01',
 'parent': 'scaled-bridge-gap-diag-1 (post-hoc addendum; exploratory)',
 'question': 'How many images have a top-1 (or top-5 boundary) tie between equal quantized logits, how many wide-versus-B top-1 '
             'changes occur with identical fc codes, and what is each reported difference under one deterministic tie rule?',
 'tie_rule': 'common rule for every arm and configuration: order classes by fc stored code (descending), ties by class index (ascending)',
 'also_reported': 'expected top-1 under a uniformly random tie break (1/|tie set| credit when the label is in the top-1 tie set)',
 'panel': 'frozen imagenet_screen_1k, all 1000 images, original batch-8 membership; same three cases',
 'gates': 'all-B and all-X endpoints as in scaled-bridge-gap-diag-1; every configuration Top-5 under its own topk must equal the '
          'diagnostic record',
 'statistics': 'tools/analysis/b_stage_balanced_comparisons.paired_outcomes, as the parent studies',
 'evidence_level': 'development, post-hoc, exploratory',
}


def ties_protocol_path():
    return BASE / 'protocol-ties-v1.json'


def common_order(units):
    """units: (n, classes) float64 numpy of fc stored units.  Returns per row (tie1, common_top5, tie5)."""
    out = []
    for row in units:
        order = np.lexsort((np.arange(row.size), -row))
        top = row[order[0]]
        tie1 = sorted(int(i) for i in np.flatnonzero(row == top))
        out.append((tie1, [int(i) for i in order[:5]], bool(row[order[4]] == row[order[5]])))
    return out


def run_case(name, start=0, stop=PANEL):
    import torch
    if unseal(diag_protocol_path()) != DIAG_PROTOCOL:
        raise ValueError('diagnostic protocol drift')
    immutable(ties_protocol_path(), TIES_PROTOCOL)
    from tools.experiment_b.classifier import configure, load_model, image_batch
    from tools.experiment_b.common import dataset
    from tools.scaled_bridge_v1.export import load_export
    started = time.perf_counter()
    configure('cuda')
    ex, arrays, ref = load_export(name)
    graph, transform, original = load_model('resnet18', 'cpu'); del original, graph
    _, rows, payload = dataset('imagenet_screen_1k')
    h = Harness(ex, arrays, 'cuda'); cfgs = configurations()
    logit_node = [n for n in h.nodes if n['op'] == 'output'][0]['inputs'][0]
    out = BASE / 'ties' / name; out.mkdir(parents=True, exist_ok=True)
    diag = BASE / 'diagnostics' / name
    folder_B = arm_folder(name, 'B'); folder_W = arm_folder(name, 'wide')
    for s in range(start, stop, 8):
        path = out / f'{s:04d}.json'
        if path.exists():
            continue
        tick = time.perf_counter()
        x = image_batch(rows[s:s + 8], payload, transform, 'cuda')
        recB = [unseal(folder_B / f'{i:04d}.json') for i in range(s, s + 8)]
        recW = [unseal(folder_W / f'{i:04d}.json') for i in range(s, s + 8)]
        drec = unseal(diag / f'{s:04d}.json') if (diag / f'{s:04d}.json').exists() else None
        result = {'start': s, 'samples': [r['sample']['sha256'] for r in recB], 'logit_node': logit_node, 'configs': {}}
        with torch.inference_mode():
            for label, (cfg, only) in cfgs.items():
                st, lg, _ = h.forward(x, cfg, only)
                top = topk(lg)
                ref_rec = recB if label == 'B' else recW if label == 'X' else None
                if ref_rec is not None:
                    for key, u in st.items():
                        hs = h.codes_hash(u)
                        if any(hs[j] != ref_rec[j]['layers'][key]['codes'] for j in range(8)):
                            raise ValueError(f'{label} endpoint does not reproduce recorded codes at {key}, batch {s}')
                    if top != [r['top5'] for r in ref_rec]:
                        raise ValueError(f'{label} endpoint Top-5 mismatch batch {s}')
                if drec is not None and drec['configs'][label]['top5'] != top:
                    raise ValueError(f'{label} Top-5 differs from the diagnostic record, batch {s}')
                units = st[logit_node].reshape(8, -1).cpu().numpy()
                co = common_order(units)
                result['configs'][label] = {'top5': top, 'tie1': [c[0] for c in co], 'common_top5': [c[1] for c in co],
                                            'tie5': [c[2] for c in co],
                                            'fc_hash': h.codes_hash(st[logit_node].reshape(8, -1))}
        result['seconds'] = time.perf_counter() - tick
        immutable(path, result)
        if (s // 8) % 25 == 0:
            print(f'{name} ties {s + 8}/{stop} {result["seconds"]:.2f}s/batch', flush=True)
    print(json.dumps({'job': 'ties', 'format': name, 'start': start, 'stop': stop,
                      'wall_seconds': time.perf_counter() - started}), flush=True)


# ---- summary ---------------------------------------------------------------

def load_case(name):
    folder = BASE / 'ties' / name
    recs = [unseal(p) for p in sorted(folder.glob('*.json'))]
    if [r['start'] for r in recs] != list(range(0, 8 * len(recs), 8)):
        raise ValueError('noncontiguous tie batches')
    n = 8 * len(recs)
    labels = [int(unseal(arm_folder(name, 'B') / f'{i:04d}.json')['sample']['label']) for i in range(n)]
    d = {}
    for label in configurations():
        d[label] = {k: sum((r['configs'][label][k] for r in recs), []) for k in ('top5', 'tie1', 'common_top5', 'tie5', 'fc_hash')}
    return d, labels, n


def expected_top1(tie1, labels):
    return [1 / len(t) if y in t else 0. for t, y in zip(tie1, labels)]


def tie_luck(preds, tie1, labels):
    """On top-1 tie images: label picked by the rule versus a uniformly random pick (count, expectation, sd, z)."""
    idx = [i for i, t in enumerate(tie1) if len(t) > 1]
    obs = sum(preds[i][0] == labels[i] for i in idx)
    p = [1 / len(tie1[i]) if labels[i] in tie1[i] else 0. for i in idx]
    exp = sum(p); sd = sum(q * (1 - q) for q in p) ** .5
    return {'tie_images': len(idx), 'label_picked': obs, 'expected_random': exp, 'sd_random': sd,
            'z': (obs - exp) / sd if sd else 0., 'excess_pp_of_panel': 100 * (obs - exp) / len(labels)}


def summarize_case(name):
    from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
    from .analysis import correctness, directions, load as load_arms
    d, labels, n = load_case(name)
    arms = load_arms(name)
    fp32 = [r['FP32_top5'] for r in arms['B'][:n]]
    # the tie audit's B / X Top-5 are exactly the recorded B / wide Top-5 (gated); control == wide in every record
    if d['B']['top5'] != [r['top5'] for r in arms['B'][:n]] or d['X']['top5'] != [r['top5'] for r in arms['wide'][:n]]:
        raise ValueError('tie audit endpoints differ from arm records')
    control_same = [r['top5'] for r in arms['control'][:n]] == d['X']['top5']
    diag_checked = 0
    for s in range(0, n, 8):  # batches written before their diagnostic record existed are cross-checked here
        p = BASE / 'diagnostics' / name / f'{s:04d}.json'
        if p.exists():
            rec = unseal(p)
            for label in configurations():
                if rec['configs'][label]['top5'] != d[label]['top5'][s:s + 8]:
                    raise ValueError(f'tie audit {label} Top-5 differs from diagnostic record, batch {s}')
            diag_checked += 8
    B, X = d['B'], d['X']
    same_fc = [a == b for a, b in zip(B['fc_hash'], X['fc_hash'])]
    changed = [p[0] != q[0] for p, q in zip(B['top5'], X['top5'])]
    tie_changed = [c and s for c, s in zip(changed, same_fc)]
    trans = lambda idx: directions([B['top5'][i] for i in idx], [X['top5'][i] for i in idx], [labels[i] for i in idx])

    def diff(left, right, sl):
        res = {}
        for rule, key in (('measured', 'top5'), ('common_rule', 'common_top5')):
            a = left[key][sl] if isinstance(left, dict) else left[sl]
            b = right[key][sl] if isinstance(right, dict) else right[sl]
            t1 = paired_outcomes(correctness(a, labels[sl], 'top1'), correctness(b, labels[sl], 'top1'))
            t5 = paired_outcomes(correctness(a, labels[sl], 'top5'), correctness(b, labels[sl], 'top5'))
            res[rule] = {'top1_pp': t1['difference_pp'], 'top1_95': t1['pointwise_95_interval_pp'], 'mcnemar_p': t1['mcnemar_exact_p'],
                         'top5_pp': t5['difference_pp'], 'changes': directions(a, b, labels[sl])}
        return res

    fp = {'top5': fp32, 'common_top5': fp32}
    slices = {'all': slice(0, n), 'prefix_0_127': slice(0, 128), 'rest_128_999': slice(128, n)}
    out = {'format': name, 'images': n, 'control_top5_equals_wide': control_same, 'images_cross_checked_with_diagnostics': diag_checked,
           'top1_tie_images': {k: sum(len(t) > 1 for t in d[k]['tie1']) for k in ('B', 'X')},
           'top1_tie_images_label_in_tie': {k: sum(len(t) > 1 and y in t for t, y in zip(d[k]['tie1'], labels)) for k in ('B', 'X')},
           'top5_boundary_tie_images': {k: sum(d[k]['tie5']) for k in ('B', 'X')},
           'max_top1_tie_size': {k: max(len(t) for t in d[k]['tie1']) for k in ('B', 'X')},
           'identical_fc_codes_images': sum(same_fc),
           'top1_changes_with_identical_fc_codes': {'count': sum(tie_changed),
                                                    **trans([i for i in range(n) if tie_changed[i]])},
           'top1_changes_with_different_fc_codes': trans([i for i in range(n) if changed[i] and not same_fc[i]]),
           'measured_rule_agrees_with_common_rule': {k: sum(p[0] == q[0] for p, q in zip(d[k]['top5'], d[k]['common_top5'])) for k in ('B', 'X')},
           'accuracy': {}, 'differences': {}, 'configs': {}}
    for k in ('B', 'X'):
        out['accuracy'][k] = {'top1_measured': 100 * sum(correctness(d[k]['top5'], labels, 'top1')) / n,
                              'top1_common_rule': 100 * sum(correctness(d[k]['common_top5'], labels, 'top1')) / n,
                              'top1_expected_random_tie': 100 * sum(expected_top1(d[k]['tie1'], labels)) / n,
                              'top5_measured': 100 * sum(correctness(d[k]['top5'], labels, 'top5')) / n,
                              'top5_common_rule': 100 * sum(correctness(d[k]['common_top5'], labels, 'top5')) / n}
    out['tie_luck'] = {k: {rule: tie_luck(d[k][key], d[k]['tie1'], labels) for rule, key in (('measured', 'top5'), ('common_rule', 'common_top5'))}
                       for k in ('B', 'X')}
    ex = lambda k: expected_top1(d[k]['tie1'], labels)
    for sname, sl in slices.items():
        out['differences'][sname] = {'wide_minus_B': diff(B, X, sl), 'B_minus_FP32': diff(fp, B, sl), 'wide_minus_FP32': diff(fp, X, sl),
                                     'wide_minus_B_expected_random_tie_pp': 100 * (sum(ex('X')[sl]) - sum(ex('B')[sl])) / len(labels[sl])}
    for label in configurations():
        c = d[label]
        out['configs'][label] = {
            'top1_measured': 100 * sum(correctness(c['top5'], labels, 'top1')) / n,
            'top1_common_rule': 100 * sum(correctness(c['common_top5'], labels, 'top1')) / n,
            'top1_tie_images': sum(len(t) > 1 for t in c['tie1']),
            'minus_B_common_rule': diff(B, c, slice(0, n))['common_rule'],
            'minus_X_common_rule': diff(X, c, slice(0, n))['common_rule']}
    return out


def write():
    if unseal(ties_protocol_path()) != TIES_PROTOCOL:
        raise ValueError('ties protocol drift')
    cases = [summarize_case(n) for n in FORMATS if (BASE / 'ties' / n).exists()]
    data = {'protocol': reference(ties_protocol_path()), 'evidence_level': TIES_PROTOCOL['evidence_level'],
            'certification': 'diagnostic harness; endpoints and every configuration Top-5 checked against sealed records',
            'sources': sources(), 'cases': cases}
    RESULTS.mkdir(parents=True, exist_ok=True)
    seal(RESULTS / 'ties.json', data)
    for c in cases:
        a = c['differences']['all']['wide_minus_B']
        print(c['format'], c['images'], 'ties B/X', c['top1_tie_images'], 'tie-changed', c['top1_changes_with_identical_fc_codes']['count'],
              'wide-B measured', a['measured']['top1_pp'], 'common', a['common_rule']['top1_pp'], a['common_rule']['top1_95'],
              'expected', round(c['differences']['all']['wide_minus_B_expected_random_tie_pp'], 2))
    return data


def main(args):
    if args[0] == 'run':
        run_case(args[1], int(args[2]), int(args[3]))
    elif args[0] == 'summarize':
        write()
    else:
        raise ValueError('unknown ties action')
