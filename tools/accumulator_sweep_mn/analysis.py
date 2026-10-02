"""Analysis of the MobileNet accumulator sweep (lane Q1, protocol accumulator-sweep-mn-protocol-v1). CPU only.

    .venv/bin/python -m tools.accumulator_sweep_mn.analysis                 # write-once into results/summaries/accumulator-sweep-mn-v1/
    .venv/bin/python -m tools.accumulator_sweep_mn.analysis --out DIR       # any other (scratch) folder; may be partial

Reads only sealed prediction files of the archived 1f75c923 run root, the certificates and adapted exports, lane L1's
sealed B2 matrix readouts (read-only) and lane L8's ResNet18 summaries.  Statistics: tools.analysis.b2_ties.paired
(expected credit) and tools.analysis.b_stage_balanced_comparisons.paired_outcomes (lowest index), as lane L8.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from tools.accumulator_sweep_v1.score import image_scores, node_event_rates, derived
from tools.analysis.b2_ties import paired as paired_expected
from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
from tools.experiment_b2.readout import credits

from . import certs, plan
from .load import assemble, prediction_files, covering

ROOT = certs.ROOT
OUT = ROOT / 'results/summaries/accumulator-sweep-mn-v1'
PROTOCOL = ROOT / 'public/experiments/configs/breadth-study/accumulator-sweep-mn-protocol-v1.json'
L8_SUMMARY = ROOT / 'results/summaries/accumulator-sweep-v1'
MN_SUMMARY = ROOT / 'results/summaries/scaled-bridge-v2/MN-1f75c9232c8a0202/summary.json'
SCREEN = plan.SCREEN
LOCATION = plan.LOCATION
CODE_FILES = ('tools/accumulator_sweep_mn/analysis.py', 'tools/accumulator_sweep_mn/certs.py',
              'tools/accumulator_sweep_mn/load.py', 'tools/accumulator_sweep_mn/plan.py',
              'tools/accumulator_sweep_v1/score.py', 'tools/accumulator_sweep_v1/certs.py',
              'tools/accumulator_sweep_v1/load.py')


def policy_family(policy):
    if policy in ('wide', 'control'):
        return policy, None
    m = re.fullmatch(r'sat\.w(\d+)', policy)
    if m:
        return 'uniform', int(m[1])
    m = re.fullmatch(r'sat\.struct-(\d+)', policy)
    if m:
        return 'per_node', int(m[1])
    m = re.fullmatch(r'(fp16|f21)(?:\.x(-?\d+))?', policy)
    if m:
        return m[1], int(m[2] or 0)
    raise ValueError(policy)


def policies_with(case, start, stop):
    folder = certs.RUN_ROOT / case / 'predictions'
    names = {re.sub(r'-cuda-\d{5}-\d{5}\.json$', '', p.name) for p in folder.glob('*-cuda-*.json')}
    return sorted(p for p in names if covering(prediction_files(case, p), start, stop) is not None)


def compare(case, policy, data, s, wide, ws):
    """One policy against the exact arm on the same images (both tie rules, paired intervals, counts)."""
    if [r['sha256'] for r in data['images']] != [r['sha256'] for r in wide['images']]:
        raise ValueError(f'{case} {policy}: images are not paired with the exact arm')
    family, param = policy_family(policy)
    pe = paired_expected(s['expected'], ws['expected'])
    pl = paired_outcomes(ws['lowest'], s['lowest'])
    return {'case': case, 'policy': policy, 'family': family, 'parameter': param, 'images': len(s['expected']),
            'top1_expected': 100 * float(s['expected'].mean()), 'top1_lowest_index': 100 * float(s['lowest'].mean()),
            'diff_expected_pp': pe['difference_pp'], 'diff_expected_ci95': pe['pointwise_95_interval_pp'],
            'diff_lowest_pp': pl['difference_pp'], 'diff_lowest_ci95': pl['pointwise_95_interval_pp'],
            'mcnemar_p_lowest': pl['mcnemar_exact_p'],
            'policy_only_correct_lowest': pl['right_only_correct'], 'wide_only_correct_lowest': pl['left_only_correct'],
            'changed_top1_images': int(((s['top1'] != ws['top1']) | s['failed']).sum()),
            'output_changed_images': int(sum(a.get('output') != b.get('output') for a, b in zip(data['images'], wide['images']))),
            'event_images': int(s['event'].sum()), 'failed_images': int(s['failed'].sum()),
            'tied_top1_images': int((s['tied'] > 1).sum()), 'execution_seconds': data['execution_seconds'],
            'files': data['files']}


def node_rows(case, policy, data, kinds, cert_rows):
    out = []
    family, param = policy_family(policy)
    for node, r in node_event_rates(data['images'], data['nodes']).items():
        c = cert_rows[node]
        out.append({'case': case, 'policy': policy, 'family': family, 'parameter': param, 'node': node,
                    'kind': kinds[node], 'K': c['K'], 'W_node_abs': c['abs'], 'W_node_struct': c['struct'], **r})
    return out


def kind_rows(nrows):
    """Aggregate node rows by (case, policy, kind)."""
    groups = {}
    for r in nrows:
        groups.setdefault((r['case'], r['policy'], r['kind']), []).append(r)
    out = []
    for (case, policy, kind), rows in groups.items():
        hit = [r for r in rows if r['images_with_event']]
        rates = [r['output_rate'] or 0.0 for r in rows]
        out.append({'case': case, 'policy': policy, 'family': rows[0]['family'], 'parameter': rows[0]['parameter'],
                    'kind': kind, 'nodes': len(rows), 'nodes_with_event': len(hit),
                    'max_image_rate': max(r['image_rate'] for r in rows), 'max_output_rate': max(rates),
                    'mean_output_rate': float(np.mean(rates)),
                    'leading_node': max(rows, key=lambda r: (r['image_rate'], r['output_rate'] or 0))['node'] if hit else ''})
    return out


def node_needed_width(width_rows, bracket_bottom):
    """Per node: the narrowest width at which it has no event on the 1k images (one more than the widest width with
    an event; events are monotone in W).  Nodes without any event in the measured bracket get None (censored:
    their needed width is <= the bracket bottom)."""
    widest = {}
    for r in width_rows:
        if r['images_with_event']:
            widest[r['node']] = max(widest.get(r['node'], 0), r['parameter'])
    return {n: (w + 1) for n, w in widest.items()}


def spearman(x, y):
    """Spearman rank correlation (average ranks for ties); None for fewer than 3 points or a constant input."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None

    def ranks(v):
        order = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); r[order] = np.arange(len(v))
        for val in np.unique(v):
            idx = v == val; r[idx] = r[idx].mean()
        return r
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def governing(case, nrows_uniform, kinds, cert_rows, bottom):
    """Node-level slack: per node needed width a_n (1k) against its own certificate; which node sets W_noevent;
    slack (cert_n - a_n) by kind and its rank correlation with log2 K."""
    need = node_needed_width(nrows_uniform, bottom)
    if not need:
        return None
    rows = [{'node': n, 'kind': kinds[n], 'K': cert_rows[n]['K'], 'W_node_abs': cert_rows[n]['abs'], 'needed': a,
             'node_slack': cert_rows[n]['abs'] - a} for n, a in need.items()]
    top = max(r['needed'] for r in rows)
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r['kind'], []).append(r['node_slack'])
    return {'case': case, 'nodes_with_event_in_bracket': len(rows), 'nodes_total': len(cert_rows),
            'W_noevent_nodes': sorted((r['node'], r['kind'], r['K']) for r in rows if r['needed'] == top),
            'node_slack_by_kind': {k: {'n': len(v), 'min': min(v), 'median': float(np.median(v)), 'max': max(v)}
                                   for k, v in sorted(by_kind.items())},
            'censored_nodes_by_kind': {k: sum(1 for n in cert_rows if kinds[n] == k and n not in need)
                                       for k in sorted(set(kinds.values()))},
            'spearman_node_slack_vs_log2K': spearman([math.log2(r['K']) for r in rows], [r['node_slack'] for r in rows]),
            'spearman_needed_vs_W_node_abs': spearman([r['W_node_abs'] for r in rows], [r['needed'] for r in rows]),
            'nodes': sorted(rows, key=lambda r: (-r['needed'], r['node']))}


def find_readout(identity):
    for cell in (ROOT / 'artifacts/experiment_b2/matrix/cells').glob('mobilenet*.json'):
        p = json.load(open(cell))['payload']
        if p.get('configuration_sha256') == identity:
            return ROOT / p['readout_file'], cell
    return None, None


def simulator_comparison(case, wide, ws):
    """Exact wide arm (1k) against L1's sealed B2 readout of the same configuration identity, three tie rules."""
    identity = certs.load_export(case)['retained']['B2_configuration_sha256']
    readout, cell = find_readout(identity)
    out = {'case': case, 'B2_configuration_sha256': identity}
    if readout is None:
        out['source'] = None
        return out
    z = np.load(readout)
    labels = np.array([r['label'] for r in wide['images']])
    if not np.array_equal(z['label'].astype(int), labels):
        raise ValueError(f'{case}: readout rows are not the engine rows')
    arrays = {k: z[k] for k in z.files}
    c = credits(arrays)
    b2_topk = z['top5_topk'][:, 0].astype(int); b2_low = z['argmax_lowest'].astype(int)
    exact_low = ws['lowest'].astype(np.int8)
    topk_correct = (b2_topk == labels).astype(np.int8); low_correct = (b2_low == labels).astype(np.int8)
    out.update({
        'source': str(readout.relative_to(ROOT)), 'cell': str(cell.relative_to(ROOT)),
        'readout_sha256': hashlib.sha256(readout.read_bytes()).hexdigest(),
        'b2_top1_retained_order': 100 * float(topk_correct.mean()), 'b2_top1_lowest_index': 100 * float(low_correct.mean()),
        'b2_top1_expected': 100 * float(c['top1_expected'].mean()),
        'exact_top1_lowest_index': 100 * float(exact_low.mean()), 'exact_top1_expected': 100 * float(ws['expected'].mean()),
        'same_class_retained_order': int((b2_topk == ws['top1']).sum()),
        'same_class_lowest_index': int((b2_low == ws['top1']).sum()),
        'b2_tied_top1_images': int((z['tie_size'] > 1).sum()), 'exact_tied_top1_images': int((ws['tied'] > 1).sum()),
        'exact_minus_b2_retained_order': paired_outcomes(topk_correct, exact_low),
        'exact_minus_b2_lowest_index': paired_outcomes(low_correct, exact_low),
        'exact_minus_b2_expected': paired_expected(ws['expected'], c['top1_expected'])})
    return out


def mn_code_divergence():
    """Stored-code divergence of the 32-image MN panels (lane L2c's sealed summary), per case."""
    try:
        cases = json.load(open(MN_SUMMARY))['payload']['cases']
    except OSError:
        return {}
    return {c['case']: {'changed_fraction': c['wide_vs_B2'].get('changed_fraction'),
                        'images_compared': c['wide_vs_B2'].get('B2_retained_top5_reproduced_images')} for c in cases}


def location_rows(spec):
    a, b = LOCATION
    case = spec['case']
    wide = assemble(case, 'wide', a, b)
    if wide is None:
        return []
    ws = image_scores(wide['images'])
    out = []
    for policy in policies_with(case, a, b):
        if policy == 'wide':
            continue
        data = assemble(case, policy, a, b); s = image_scores(data['images'])
        family, param = policy_family(policy)
        out.append({'case': case, 'policy': policy, 'family': family, 'parameter': param, 'images': b - a,
                    'top1_expected': 100 * float(s['expected'].mean()), 'wide_top1_expected': 100 * float(ws['expected'].mean()),
                    'top1_lowest_index': 100 * float(s['lowest'].mean()),
                    'changed_top1_images': int(((s['top1'] != ws['top1']) | s['failed']).sum()),
                    'event_images': int(s['event'].sum()), 'failed_images': int(s['failed'].sum())})
    return out


def l8_rows():
    path = L8_SUMMARY / 'widths.csv'
    if not path.exists():
        return []
    with open(path) as handle:
        return [dict(r, network='resnet18') for r in csv.DictReader(handle)]


def analyse(protocol):
    policy_out, node_out, width_out, sim_out, gov_out = [], [], [], [], []
    loc_out = []
    divergence = mn_code_divergence()
    for spec in protocol['cases']:
        case = spec['case']
        loc_out += location_rows(spec)
        wide = assemble(case, 'wide', *SCREEN)
        if wide is None:
            continue
        ws = image_scores(wide['images'])
        wide_exp = 100 * float(ws['expected'].mean()); wide_low = 100 * float(ws['lowest'].mean())
        cert_rows = certs.v1.node_table(certs.v1.load_certificate(case, certs.RUN_ROOT))
        kinds = certs.node_kinds(case)
        uniform, per_node, data_by = {}, {}, {}
        for policy in policies_with(case, *SCREEN):
            if policy == 'wide':
                continue
            data = assemble(case, policy, *SCREEN); s = image_scores(data['images'])
            row = compare(case, policy, data, s, wide, ws)
            policy_out.append(row)
            fam, par = row['family'], row['parameter']
            g = {'events': row['event_images'], 'changed': row['changed_top1_images'], 'expected_percent': row['top1_expected']}
            if fam == 'uniform':
                uniform[par] = g; data_by[policy] = data
            elif fam == 'per_node':
                per_node[par] = g; data_by[policy] = data
        du = derived(uniform, wide_exp, True) if uniform else {}
        dn = derived(per_node, wide_exp, False) if per_node else {}
        # node rows: uniform widths from W_noevent - 1 (first width with events) down to W_half; every measured d
        widths = sorted(uniform, reverse=True)
        nrows_u = []
        for w in widths:
            nrows = node_rows(case, f'sat.w{w}', data_by[f'sat.w{w}'], kinds, cert_rows)
            nrows_u += nrows
            if du.get('noevent') is None or w < du['noevent']:
                if du.get('half') is None or w >= du['half']:
                    node_out += nrows
        for d in sorted(per_node):
            node_out += node_rows(case, f'sat.struct-{d}', data_by[f'sat.struct-{d}'], kinds, cert_rows)
        gov = governing(case, nrows_u, kinds, cert_rows, min(widths) if widths else None) if widths else None
        if gov:
            gov_out.append(gov)
        sim = simulator_comparison(case, wide, ws)
        sim['stored_code_divergence_32img'] = divergence.get(case)
        sim_out.append(sim)
        wcert = spec['W_cert_abs']
        width_out.append({
            'network': spec['network'], 'case': case, 'format': spec['format'], 'stress_case': spec.get('stress_case', False),
            'wide_top1_expected': wide_exp, 'wide_top1_lowest_index': wide_low,
            'published_formula_bits': spec['published_width_Agg24'], 'W_cert_abs': wcert, 'W_cert_struct': spec['W_cert_struct'],
            'W_noevent': du.get('noevent'), 'W_same': du.get('same'), 'W_acc_0.5': du.get('acc_0.5'),
            'W_acc_1.0': du.get('acc_1.0'), 'W_half': du.get('half'),
            'slack_cert_minus_acc1': (wcert - du['acc_1.0']) if du.get('acc_1.0') is not None else None,
            'slack_cert_minus_noevent': (wcert - du['noevent']) if du.get('noevent') is not None else None,
            'd_noevent': dn.get('noevent'), 'd_same': dn.get('same'), 'd_acc_0.5': dn.get('acc_0.5'),
            'd_acc_1.0': dn.get('acc_1.0'), 'd_half': dn.get('half'),
            'uniform_widths_measured': sorted(uniform), 'per_node_d_measured': sorted(per_node),
            'W_noevent_nodes': gov['W_noevent_nodes'] if gov else None})
    return {'policies': policy_out, 'node_events': node_out, 'kind_events': kind_rows(node_out), 'widths': width_out,
            'simulator': sim_out, 'governing': gov_out, 'location_128': loc_out, 'resnet18_l8_widths': l8_rows()}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'x', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'{out} exists and is not empty; summaries are written once (use a new version)')
    out.mkdir(parents=True, exist_ok=True)
    with open(PROTOCOL) as handle:
        protocol = json.load(handle)
    addenda = sorted(PROTOCOL.parent.glob('accumulator-sweep-mn-protocol-v1-addendum-*.json'))
    for a in addenda:
        protocol['cases'] += json.load(open(a)).get('cases_added', [])
    result = analyse(protocol)
    for name in ('policies', 'node_events', 'kind_events', 'widths', 'location_128'):
        write_csv(out / f'{name}.csv', [{k: v for k, v in r.items() if k != 'files'} for r in result[name]])
    sims = [{k: v for k, v in r.items()} for r in result['simulator']]
    write_csv(out / 'simulator.csv', sims)
    summary = {
        'id': 'accumulator-sweep-mn-v1', 'written': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'evidence': 'development evidence, ImageNet screen-1k list (images 0-999; location 0-127); no held-out image',
        'engine_digest': certs.DIGEST,
        'protocol': {'path': str(PROTOCOL.relative_to(ROOT)), 'sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                     'addenda': [{'path': str(a.relative_to(ROOT)), 'sha256': hashlib.sha256(a.read_bytes()).hexdigest()} for a in addenda]},
        'analysis_code_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES},
        **{k: result[k] for k in ('widths', 'governing', 'simulator', 'resnet18_l8_widths')},
        'policies': result['policies']}
    with open(out / 'summary.json', 'x') as handle:
        json.dump(summary, handle, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        handle.write('\n')
    print(out, len(result['policies']), 'policy rows')


if __name__ == '__main__':
    main()
