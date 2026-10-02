"""Lane S1: collect the speed-v1 evidence into results/summaries/speed-v1/ (protocol speed-protocol-v1; written once).

  .venv/bin/python -m tools.run.speed_summary [--out results/summaries/speed-v1] [--check]

Reads only lane S1's own outputs under artifacts/speed_v1/ (validation, timing, profile, cocoeval, fastblocks,
runs, implementations) and the archives' sealed predict files (read-only, for the collapse check of the sat.w
widths). --check prints the summary without writing it. An existing summary file is never overwritten.
"""
from __future__ import annotations
import argparse
import glob
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from tools.experiment_b.common import ROOT, unseal, file_hash

SPEED = ROOT / 'artifacts/speed_v1'
V2 = ROOT / 'artifacts/scaled_bridge_v2'
R7 = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'
R1 = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'
PROTOCOL = ROOT / 'public/experiments/configs/breadth-study/speed-protocol-v1.json'


def fast_tag():
    from tools.scaled_bridge_fast.common import digest, identity
    return digest(identity())


def _sealed(paths):
    return [(Path(p), unseal(p)) for p in sorted(paths)]


def full_trace(tag):
    out = {}
    for kind in ('regress', 'trace'):
        rows = _sealed(glob.glob(str(SPEED / 'validation' / kind / tag[:16] / '*.json')))
        ok = [r for _, r in rows if r['identical'] == r['images'] and not r['differences']]
        cases = defaultdict(lambda: [0, 0])
        for _, r in rows:
            cases[r['case']][0] += 1
            cases[r['case']][1] += r['images']
        out[kind] = {'sets': len(rows), 'sets_identical': len(ok), 'image_records': sum(r['images'] for _, r in rows),
                     'image_records_identical': sum(r['identical'] for _, r in rows),
                     'batches': sorted({r['batch'] for _, r in rows}),
                     'per_case_sets_images': {k: v for k, v in sorted(cases.items())}}
    return out


def compare(tag):
    rows = _sealed(glob.glob(str(SPEED / 'validation' / 'compare' / tag[:16] / '*.json')))
    per = defaultdict(dict)
    bad = []
    for _, r in rows:
        key = f"{r['policy']}@b{r['batch']}"
        per[r['case']][key] = [r['compared'], r['different']]
        if r['different'] or r['compared'] < r['stop'] - r['start'] or r['node_constants_equal'] is False:
            bad.append({k: r[k] for k in ('case', 'policy', 'batch', 'compared', 'different', 'node_constants_equal', 'first_difference')})
    return {'sets': len(rows), 'item_comparisons': sum(r['compared'] for _, r in rows),
            'items_identical': sum(r['identical'] for _, r in rows), 'items_different': sum(r['different'] for _, r in rows),
            'sets_failing': bad, 'cases': len(per),
            'per_case': {c: v for c, v in sorted(per.items())}}


def archive_items(case, policy):
    root = V2 / 'runs' / (R7 if case.startswith('resnet18-') else R1)
    found = {}
    for base in (root, SPEED / 'archive-runs' / R7, SPEED / 'archive-runs' / R1):
        for path in sorted((base / case / 'predictions').glob(f'{policy}-cuda-*.json')):
            for item in unseal(path)['images']:
                if item['index'] < 64:
                    found[item['index']] = item
    return [found[i] for i in sorted(found)]


def collapse():
    """Top-1 and event incidence of the archive's own items for the planned sat.w widths (first 64 images)."""
    plan = json.loads((SPEED / 'validation' / 'plan.json').read_text())
    out = {}
    for case, entry in plan.items():
        row = {}
        for role in ('wide', 'sat_w_events', 'sat_w_collapse'):
            policy = entry['policies'][role]
            items = archive_items(case, policy)
            if not items:
                row[role] = {'policy': policy, 'images': 0}
                continue
            top1 = sum(int(it['top5'][0] == it['label']) for it in items) / len(items)
            events = sum(1 for it in items if it.get('events'))
            failures = sum(1 for it in items if it.get('failure'))
            row[role] = {'policy': policy, 'images': len(items), 'top1': top1, 'images_with_events': events,
                         'images_with_failure_record': failures}
        out[case] = row
    return out


def timing():
    runs = {}
    for path in sorted((SPEED / 'timing').glob('*.json')):
        doc = json.loads(path.read_text())
        runs[path.stem] = {'path': doc['path'], 'case': doc['case'], 'policy': doc['policy'], 'images': doc['stop'] - doc['start'],
                           'setup_seconds': doc['setup_seconds'], 'items_sha': doc['items_sha'],
                           'seconds_per_image': {str(r['batch']): r['seconds_per_image'] for r in doc['runs']},
                           'peak_gpu_memory_bytes': doc.get('peak_gpu_memory_bytes'), 'written': doc['written'],
                           'file': str(path.relative_to(ROOT))}
    pairs = {}
    for name, a in runs.items():
        if not name.startswith('archive-'):
            continue
        f = runs.get('fast-' + name[len('archive-'):])
        if not f:
            continue
        b8 = a['seconds_per_image'].get('8'), f['seconds_per_image'].get('8')
        best_a = min(a['seconds_per_image'].items(), key=lambda kv: kv[1])
        best_f = min(f['seconds_per_image'].items(), key=lambda kv: kv[1])
        pairs[name[len('archive-'):]] = {
            'case': a['case'], 'policy': a['policy'], 'images': a['images'], 'items_sha_equal': a['items_sha'] == f['items_sha'],
            'archive_seconds_per_image_b8': b8[0], 'fast_seconds_per_image_b8': b8[1],
            'speedup_b8': b8[0] / b8[1] if all(b8) else None,
            'archive_best': {'batch': int(best_a[0]), 'seconds_per_image': best_a[1]},
            'fast_best': {'batch': int(best_f[0]), 'seconds_per_image': best_f[1]},
            'speedup_best_vs_best': best_a[1] / best_f[1],
            'images_per_second_fast_best': 1 / best_f[1], 'images_per_second_archive_best': 1 / best_a[1]}
    return {'runs': runs, 'pairs': pairs}


def concurrency():
    out = {}
    for path in sorted((SPEED / 'concurrency').glob('*.json')):
        out[path.stem] = json.loads(path.read_text())
    return out


def profile():
    out = {}
    for path in sorted((SPEED / 'profile').glob('*.json')):
        doc = json.loads(path.read_text())
        for r in doc['results']:
            key = f"{r['path']}|{r['case']}|{r['policy']}"
            out[key] = {'seconds_per_image': r['seconds_per_image'], 'images': r['images'], 'batch': r['batch'],
                        'items_sha': r['items_sha'], 'loop_wall_seconds': r['loop_wall_seconds'],
                        'stages_seconds': r.get('stages_seconds') or r.get('stages_seconds_instrumented'),
                        'file': str(path.relative_to(ROOT))}
    return {'note': 'shared GPU (other lanes running), batch 8, 64 images, trace none: indicative stage shares', 'entries': out}


def gates_and_archive(tag):
    root = SPEED / 'runs' / tag
    case_gates = sorted(p.parent.name for p in root.glob('*/gate.json') if unseal(p)['status'] == 'pass')
    policy_gates = defaultdict(list)
    for p in sorted(root.glob('*/gate-*.json')):
        if unseal(p)['status'] == 'pass':
            policy_gates[p.parent.name].append(unseal(p)['policy'])
    folder = SPEED / 'implementations' / tag
    manifest = unseal(folder / 'manifest.json') if (folder / 'manifest.json').exists() else None
    return {'fast_sources_digest': tag, 'run_root': str(root.relative_to(ROOT)), 'case_gates': case_gates,
            'policy_gates': dict(policy_gates),
            'archive': None if manifest is None else {'folder': str(folder.relative_to(ROOT)), 'files': len(manifest['files']),
                                                     'manifest_sha256': file_hash(folder / 'manifest.json')}}


def cocoeval():
    folders = sorted((SPEED / 'cocoeval').glob('[0-9a-f]' * 16))
    if len(folders) != 1:
        raise ValueError(f'expected one fast_cocoeval source folder, found {folders}')
    folder = folders[0]
    rows = {p.name: json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))}
    points = [r for n, r in rows.items() if n.endswith('--d0-r0-b0.json')]
    draws = [r for r in rows.values() if r.get('draws_compared')]
    timed = [r for r in rows.values() if r.get('reference_cpu_seconds_per_draw')]
    big = [r for r in rows.values() if any(k.startswith('fast_cpu_seconds_1') for k in r)]
    per_draw = [r['fast_cpu_seconds_per_draw'] for r in draws if r['draws_compared'] == 2000]
    evaluate = [r['evaluate_cpu_seconds'] for r in rows.values()]
    return {
        'fast_source': 'tools/analysis/fast_cocoeval/__init__.py', 'fast_source_sha16': folder.name,
        'points': {'configurations': len(points),
                   'identical_to_pycocotools': sum(r['point_identical'] for r in points),
                   'identical_to_stored_bootstrap_point': sum(bool(r.get('stored_point_identical')) for r in points),
                   'max_abs_difference': max(r['point_max_abs_difference'] for r in points) if points else None},
        'bootstrap_2000': [{k: r.get(k) for k in ('stem', 'draws_compared', 'draws_identical', 'draws_max_abs_difference',
                                                  'intervals_identical', 'fast_cpu_seconds_per_draw')} for r in draws],
        'fast_cpu_seconds_per_draw_2000': {'n': len(per_draw), 'min': min(per_draw, default=None),
                                           'median': statistics.median(per_draw) if per_draw else None,
                                           'max': max(per_draw, default=None)},
        'evaluate_cpu_seconds': {'n': len(evaluate), 'min': min(evaluate), 'median': statistics.median(evaluate), 'max': max(evaluate)},
        'reference_timing': [{k: v for k, v in r.items() if not k.startswith('point_')} for r in timed],
        'fast_big': [{k: v for k, v in r.items() if not k.startswith('point_')} for r in big]}


def fastblocks():
    out = {}
    for path in sorted((SPEED / 'fastblocks').glob('*/*.execution.json')):
        doc = json.loads(path.read_text())
        doc.pop('cell_cost', None)
        out[f'{path.parent.name}/{path.name[:-len(".execution.json")]}'] = doc
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results/summaries/speed-v1'); p.add_argument('--check', action='store_true')
    args = p.parse_args()
    tag = fast_tag()
    summary = {
        'protocol': {'file': str(PROTOCOL.relative_to(ROOT)), 'sha256': file_hash(PROTOCOL)},
        'evidence_label': 'development evidence: ImageNet screen-1k (first 64 images for identity), COCO screen-1k, '
                          'first 256 calibration images of the B2 input cache; no held-out image read',
        'exact_engine': {'fast_sources_digest': tag, 'base_archive': R1, 'resnet18_archive': R7,
                         'full_trace': full_trace(tag), 'predict_compare': compare(tag), 'sat_w_widths': collapse(),
                         'gates': gates_and_archive(tag), 'timing_exclusive': timing(),
                         'concurrency_shared': concurrency(), 'profile_shared': profile()},
        'fast_cocoeval': cocoeval(), 'fastblocks': fastblocks(),
        'lane_disk_bytes': int(subprocess.run(['du', '-sb', str(SPEED)], capture_output=True, text=True).stdout.split()[0]),
        'written': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if args.check:
        json.dump(summary, sys.stdout, indent=1)
        return
    out = ROOT / args.out
    target = out / 'summary.json'
    if target.exists():
        raise SystemExit(f'{target} exists (written once; write a new versioned file instead)')
    out.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(target, file_hash(target))


if __name__ == '__main__':
    main()
