"""Analysis of the accumulator-width sweep (lane L8, protocol accumulator-sweep-protocol-v1). CPU only, reads only.

    .venv/bin/python -m tools.analysis.accumulator_sweep [--out results/summaries/accumulator-sweep-v1] [--partial DIR]

Writes once (refuses to overwrite): summary.json, policies.csv, widths.csv, node_events.csv, simulator.csv, join.csv,\nlocation_128.csv, location_128_derived.csv.
Development evidence (ImageNet screen-1k list).  `--partial DIR` analyses whatever exists into a scratch directory.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from tools.accumulator_sweep_v1 import certs
from tools.accumulator_sweep_v1.load import assemble, prediction_files, covering
from tools.accumulator_sweep_v1.score import image_scores, node_event_rates, derived
from tools.analysis.b2_ties import paired as paired_expected
from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
from tools.experiment_b.common import unseal
from tools.experiment_b2.readout import credits

ROOT = certs.ROOT
PROTOCOL = ROOT / 'public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json'
HARDWARE = ROOT / 'results/tables/ics55-integer-accwidth-v1.csv'
JOIN_TARGET_NS = 8
N_TAPS = 4608
SCREEN = (0, 1000)

# Operand formats for the published closed-form width ([Agg24], verified in docs/analysis/related-work-audit-2026-10-01.md)
FORMULA_OPERANDS = {
    'int8': ('int', 8), 'int6': ('int', 6), 'fp6_e2m3': ('fp', 2, 3), 'fp7_e3m3': ('fp', 3, 3),
    'fp8_e4m3fn': ('fp', 4, 3), 'fp8_e5m2': ('fp', 5, 2), 'posit8_es1': None}


def published_width(fmt, n=N_TAPS):
    """Closed-form accumulator width of [Agg24] for equal operand formats; None when no formula applies.

    Integer: ra + rb + ceil(log2 n) + 1.  Minifloat ExMy: 2^ea + ma + 2^eb + mb + ceil(log2 n) - 1.
    Convention assumed here: the formula's count includes the sign bit (as the certificates do); whether the
    paper counts it the same way is unchecked (open item, review 3 item 9 of lane L2).
    """
    spec = FORMULA_OPERANDS.get(fmt)
    if spec is None:
        return None
    log = math.ceil(math.log2(n))
    if spec[0] == 'int':
        return 2 * spec[1] + log + 1
    return 2 * (2 ** spec[1] + spec[2]) + log - 1


def case_format(case):
    return case.split('-')[1]


def policy_family(policy):
    if policy in ('wide', 'control'):
        return policy, None
    m = re.fullmatch(r'sat\.w(\d+)', policy)
    if m:
        return 'uniform', int(m[1])
    m = re.fullmatch(r'sat\.struct-(\d+)', policy)
    if m:
        return 'per_node', int(m[1])
    m = re.fullmatch(r'sat\.abs-(\d+)', policy)
    if m:
        return 'per_node_abs', int(m[1])
    m = re.fullmatch(r'(fp16|f21)(?:\.x(-?\d+))?', policy)
    if m:
        return m[1], int(m[2] or 0)
    raise ValueError(policy)


def screen_policies(case):
    folder = certs.RUN_ROOT / case / 'predictions'
    names = {re.sub(r'-cuda-\d{5}-\d{5}\.json$', '', p.name) for p in folder.glob('*-cuda-*.json')}
    return sorted(p for p in names if covering(prediction_files(case, p), *SCREEN) is not None)


def policy_row(case, policy, wide, wide_scores):
    data = assemble(case, policy, *SCREEN)
    s = image_scores(data['images'])
    if [r['sha256'] for r in data['images']] != [r['sha256'] for r in wide['images']]:
        raise ValueError('images are not paired with the exact arm')
    family, param = policy_family(policy)
    changed = int(((s['top1'] != wide_scores['top1']) | s['failed']).sum())
    pe = paired_expected(s['expected'], wide_scores['expected'])
    pl = paired_outcomes(wide_scores['lowest'], s['lowest'])
    row = {'case': case, 'policy': policy, 'family': family, 'parameter': param, 'images': len(s['expected']),
           'top1_expected': 100 * float(s['expected'].mean()), 'top1_lowest_index': 100 * float(s['lowest'].mean()),
           'diff_expected_pp': pe['difference_pp'], 'diff_expected_ci95': pe['pointwise_95_interval_pp'],
           'diff_lowest_pp': pl['difference_pp'], 'diff_lowest_ci95': pl['pointwise_95_interval_pp'],
           'mcnemar_p_lowest': pl['mcnemar_exact_p'], 'changed_top1_images': changed,
           'output_changed_images': int(sum(a.get('output') != b.get('output') for a, b in zip(data['images'], wide['images']))),
           'event_images': int(s['event'].sum()), 'failed_images': int(s['failed'].sum()),
           'tied_top1_images': int((s['tied'] > 1).sum()), 'execution_seconds': data['execution_seconds'],
           'files': data['files']}
    return row, data, s


def node_rates_table(case, data_by_width, widths):
    out = []
    for w in widths:
        d = data_by_width.get(w)
        if d is None:
            continue
        for node, r in node_event_rates(d['images'], d['nodes']).items():
            out.append({'case': case, 'policy': f'sat.w{w}', 'width': w, 'node': node, **r})
    return out


def first_nodes(rates, width):
    hit = [r for r in rates if r['width'] == width and r['images_with_event']]
    return [r['node'] for r in sorted(hit, key=lambda r: -r['image_rate'])]


def simulator_comparison(case, wide, wide_scores, identity_hint=None):
    """Exact wide arm against the B2 simulator's sealed 1k predictions (both tie rules where available)."""
    ex = json.load(open(next((ROOT / 'artifacts/scaled_bridge_v2/exports').glob(f'*/{case}/export.json'))))['payload']
    identity = ex['retained']['B2_configuration_sha256']
    labels = np.array([r['label'] for r in wide['images']])
    out = {'case': case, 'B2_configuration_sha256': identity}
    readout = None
    for cell in (ROOT / 'artifacts/experiment_b2/matrix/cells').glob('resnet18--*.json'):
        p = json.load(open(cell))['payload']
        if p.get('configuration_sha256') == identity:
            readout = ROOT / p['readout_file']; break
    if readout is not None:
        z = np.load(readout)
        if not np.array_equal(z['label'].astype(int), labels):
            raise ValueError('readout rows are not the engine rows')
        b2_topk = z['top5_topk'][:, 0].astype(int); b2_low = z['argmax_lowest'].astype(int)
        b2_expected = credits({k: z[k] for k in z.files})['top1_expected']  # L1's readout rule
        out['source'] = str(readout.relative_to(ROOT))
    else:
        folder = ROOT / 'artifacts/experiment_b2/predictions' / identity
        if not folder.exists():
            out['source'] = None; return out
        top = []
        for r in wide['images']:
            rec = unseal(folder / f"{r['sha256']}.json")
            top.append(rec['top5'][0])
        b2_topk = np.array(top); b2_low = None; b2_expected = None
        out['source'] = str(folder.relative_to(ROOT)) + ' (retained torch.topk order only; no tie readout)'
    exact_low = wide_scores['lowest'].astype(int)
    retained_correct = (b2_topk == labels).astype(np.int8)
    out['b2_top1_retained_order'] = 100 * float(retained_correct.mean())
    out['exact_top1_lowest_index'] = 100 * float(exact_low.mean())
    out['exact_top1_expected'] = 100 * float(wide_scores['expected'].mean())
    out['same_class_retained_order'] = int((b2_topk == wide_scores['top1']).sum())
    out['exact_minus_b2_retained_order'] = paired_outcomes(retained_correct, exact_low)
    if b2_low is not None:
        low_correct = (b2_low == labels).astype(np.int8)
        out['b2_top1_lowest_index'] = 100 * float(low_correct.mean())
        out['b2_top1_expected'] = 100 * float(b2_expected.mean())
        out['same_class_lowest_index'] = int((b2_low == wide_scores['top1']).sum())
        out['exact_minus_b2_lowest_index'] = paired_outcomes(low_correct, exact_low)
        out['exact_minus_b2_expected'] = paired_expected(wide_scores['expected'], b2_expected)
    return out


def hardware_areas(target=JOIN_TARGET_NS):
    rows = list(csv.DictReader(open(HARDWARE)))
    return {(int(r['width_bits']), int(r['accumulator_bits'])): float(r['core_area'])
            for r in rows if float(r['target_ns']) == target}


def join_rows_for(case, w_cert_abs, rows, wide_expected, wide_lowest, noevent, areas, widths=(16, 20, 24, 28, 32)):
    """Preliminary accuracy-area join of one integer case at the hardware widths (pure; unit tested).

    rows: {policy: policy row} of the 1k runs; a hardware width with a 1k run is 'measured'; a width above W_noevent
    (no saturation event on any image at a narrower width, hence none at this one) equals the exact arm and is
    'derived'; otherwise 'not run'.  Widths above the certified width are omitted (the register is lossless there).
    The multiplier width is the format's code width; the area is the core area of `areas[(bits, W)]`.
    """
    bits = int(re.sub(r'\D', '', case_format(case)))
    out = []
    for w in widths:
        if w > w_cert_abs:
            continue
        r = rows.get(f'sat.w{w}')
        if r is not None:
            top, low, how = r['top1_expected'], r['top1_lowest_index'], 'measured'
        elif noevent is not None and w > noevent:
            top, low, how = wide_expected, wide_lowest, 'derived: above W_noevent, equals exact arm'
        else:
            top = low = None; how = 'not run'
        out.append({'case': case, 'multiplier_bits': bits, 'accumulator_bits': w, 'top1_expected': top,
                    'top1_lowest_index': low, 'source': how, 'core_area_8ns': areas.get((bits, w)),
                    'W_cert_abs': w_cert_abs})
    return out


def analyse(protocol):
    cases_out, policy_rows, width_rows, node_rows, sim_rows, join_rows = [], [], [], [], [], []
    areas = hardware_areas()
    for spec in protocol['cases']:
        case = spec['case']
        if not spec.get('enters_sweep', True):
            continue
        wide = assemble(case, 'wide', *SCREEN)
        if wide is None:
            continue
        ws = image_scores(wide['images'])
        wide_exp = 100 * float(ws['expected'].mean())
        rows, data_by_width = {}, {}
        for policy in screen_policies(case):
            row, data, s = policy_row(case, policy, wide, ws)
            rows[policy] = row
            if row['family'] == 'uniform':
                data_by_width[row['parameter']] = data
            policy_rows.append({k: v for k, v in row.items() if k != 'files'})
        uni = {r['parameter']: {'events': r['event_images'], 'changed': r['changed_top1_images'],
                                'expected_percent': r['top1_expected']} for r in rows.values() if r['family'] == 'uniform'}
        per = {r['parameter']: {'events': r['event_images'], 'changed': r['changed_top1_images'],
                                'expected_percent': r['top1_expected']} for r in rows.values() if r['family'] == 'per_node'}
        if per:
            per.setdefault(0, {'events': 0, 'changed': 0, 'expected_percent': wide_exp})  # lossless by the gate
        du = derived(uni, wide_exp) if uni else {}
        dp = derived(per, wide_exp, safer_is_larger=False) if per else {}
        # node rates around the transition: from the widest width with an event down to W_half
        evw = sorted((w for w, g in uni.items() if g['events']), reverse=True)
        around = [w for w in evw if du.get('half') is None or w >= du['half']][:6]
        rates = node_rates_table(case, data_by_width, around)
        node_rows.extend(rates)
        fmt = case_format(case)
        widths = {'case': case, 'format': fmt, 'recipe': spec['recipe'],
                  'published_formula_bits': published_width(fmt) if spec['recipe'] in ('b2', 'b1') else None,
                  'W_cert_abs': spec['W_cert_abs'], 'W_cert_struct': spec['W_cert_struct'],
                  'W_noevent': du.get('noevent'), 'W_same': du.get('same'), 'W_acc_0.5': du.get('acc_0.5'),
                  'W_acc_1.0': du.get('acc_1.0'), 'W_half': du.get('half'),
                  'd_noevent': dp.get('noevent'), 'd_same': dp.get('same'), 'd_acc_0.5': dp.get('acc_0.5'),
                  'd_acc_1.0': dp.get('acc_1.0'), 'd_half': dp.get('half'),
                  'uniform_widths_measured': sorted(uni), 'per_node_d_measured': sorted(per),
                  'first_saturating_nodes': first_nodes(rates, around[0]) if around else []}
        width_rows.append(widths)
        if spec['recipe'] == 'b2':
            try:
                sim_rows.append(simulator_comparison(case, wide, ws))
            except (StopIteration, FileNotFoundError, ValueError) as error:
                sim_rows.append({'case': case, 'error': str(error)})
        if spec.get('integer'):
            join_rows.extend(join_rows_for(case, spec['W_cert_abs'], rows, wide_exp, 100 * float(ws['lowest'].mean()),
                                           du.get('noevent'), areas))
        cases_out.append({'case': case, 'wide_top1_expected': wide_exp,
                          'wide_top1_lowest_index': 100 * float(ws['lowest'].mean()),
                          'wide_tied_images': int((ws['tied'] > 1).sum()), 'derived_uniform': du,
                          'derived_per_node': dp, 'policies': sorted(rows)})
    return {'cases': cases_out, 'policies': policy_rows, 'widths': width_rows, 'node_events': node_rows,
            'simulator': sim_rows, 'join': join_rows}


LOCATION = (0, 128)


def location_rows(protocol):
    """Location phase (images 0-127 of the screen list; development, orientation only): per case and policy the
    Top-1 and event counts, paired to the same case's wide arm on the same 128 images.  The only evidence of the
    original-recipe (b1) cases and of sat.struct-<d>, whose 1k runs were deferred by the budget rule (addendum 2)."""
    out = []
    for spec in protocol['cases']:
        case = spec['case']
        wide = assemble(case, 'wide', *LOCATION)
        if wide is None:
            continue
        ws = image_scores(wide['images'])
        folder = certs.RUN_ROOT / case / 'predictions'
        names = sorted({re.sub(r'-cuda-\d{5}-\d{5}\.json$', '', q.name) for q in folder.glob('*-cuda-*.json')})
        for policy in names:
            if covering(prediction_files(case, policy), *LOCATION) is None:
                continue
            d = assemble(case, policy, *LOCATION)
            sc = image_scores(d['images'])
            family, param = policy_family(policy)
            out.append({'case': case, 'recipe': spec['recipe'], 'policy': policy, 'family': family, 'parameter': param,
                        'images': len(sc['expected']), 'top1_expected': 100 * float(sc['expected'].mean()),
                        'top1_lowest_index': 100 * float(sc['lowest'].mean()),
                        'wide_top1_expected': 100 * float(ws['expected'].mean()),
                        'changed_top1_images': int(((sc['top1'] != ws['top1']) | sc['failed']).sum()),
                        'event_images': int(sc['event'].sum()), 'failed_images': int(sc['failed'].sum())})
    return out


def location_derived(rows):
    """Derived widths of the uniform family on the 128 location images (orientation only)."""
    out = []
    for case in dict.fromkeys(r['case'] for r in rows):
        rs = [r for r in rows if r['case'] == case]
        uni = {r['parameter']: {'events': r['event_images'], 'changed': r['changed_top1_images'],
                                'expected_percent': r['top1_expected']} for r in rs if r['family'] == 'uniform'}
        if uni:
            out.append({'case': case, 'images': 128, **{f'W_{k}': v for k, v in derived(uni, rs[0]['wide_top1_expected']).items()},
                        'wide_top1_expected': rs[0]['wide_top1_expected'], 'widths_measured': sorted(uni)})
    return out


def write_csv(path, rows):
    if not rows:
        path.write_text(''); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/summaries/accumulator-sweep-v1')
    ap.add_argument('--partial', default=None, help='scratch directory for a development look (never results/)')
    args = ap.parse_args(argv)
    out = Path(args.partial) if args.partial else ROOT / args.out
    if not args.partial and out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists; summaries are written once (use a new version)')
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.load(open(PROTOCOL))
    addenda = sorted(PROTOCOL.parent.glob('accumulator-sweep-protocol-v1-addendum-*.json'))
    for path in addenda:  # cases admitted by an addendum (fp8_e5m2, addendum 1)
        protocol['cases'] += json.load(open(path)).get('cases_added', [])
    result = analyse(protocol)
    result['location_128'] = location_rows(protocol)
    result['location_128_derived'] = location_derived(result['location_128'])
    result['protocol'] = {'path': str(PROTOCOL.relative_to(ROOT)), 'sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                          'addenda': [{'path': str(a.relative_to(ROOT)), 'sha256': hashlib.sha256(a.read_bytes()).hexdigest()}
                                      for a in addenda]}
    result['evidence_class'] = 'development evidence, ImageNet screen-1k list (images 0-999)'
    result['engine_digest'] = certs.DIGEST
    # sha256 of the analysis code (added after review 2; summaries v1 predate it, see review2_check.json)
    code = sorted([*(ROOT / 'tools/accumulator_sweep_v1').glob('*.py'), *(ROOT / 'tools/analysis').glob('accumulator_sweep*.py')])
    result['analysis_code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    result['hardware_join'] = {'table': str(HARDWARE.relative_to(ROOT)), 'target_ns': JOIN_TARGET_NS,
                               'status': 'preliminary: hardware sweep unreviewed, RTL work on hold'}
    (out / 'summary.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    for name in ('policies', 'widths', 'node_events', 'simulator', 'join', 'location_128', 'location_128_derived'):
        write_csv(out / f'{name}.csv', result[name])
    print(out)


if __name__ == '__main__':
    main()
