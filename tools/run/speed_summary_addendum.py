"""Lane S1: summary of the review-1 follow-up (protocol speed-protocol-v1-addendum-1), written once.

  .venv/bin/python -m tools.run.speed_summary_addendum [--check]
-> results/summaries/speed-v1/summary-addendum-1.json. Recomputes the corrected document numbers from the raw files
(finding B1 and N2), and collects N1 (device memory of a fast block cell), N3 (process setup) and N5 (mixed
failed/valid float batches: scan, archive runs, comparisons). --check prints the result without writing.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

from tools.experiment_b.common import ROOT

SPEED = ROOT / 'artifacts/speed_v1'
OUT = ROOT / 'results/summaries/speed-v1/summary-addendum-1.json'
PROTOCOL = ROOT / 'public/experiments/configs/breadth-study/speed-protocol-v1-addendum-1.json'
TAG = 'edfecd5cdb0dc5ad'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mtime(path):
    return time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(os.path.getmtime(path)))


def concurrency():
    out = {}
    for path in sorted((SPEED / 'concurrency').glob('*.json')):
        d = json.loads(path.read_text())
        spi = [r['seconds_per_image'] for r in d['runs']]
        out[path.stem] = {'case': d['case'], 'batch': d['runs'][0]['batch'], 'images': d['stop'] - d['start'],
                          'passes_seconds_per_image': spi, 'median': statistics.median(spi), 'min': min(spi), 'max': max(spi),
                          'loops': [min(r['loop_start_epoch'] for r in d['runs']), max(r['loop_end_epoch'] for r in d['runs'])],
                          'images_per_second_at_median': 1 / statistics.median(spi)}
    return out


def profile():
    out = {}
    for path in sorted((SPEED / 'profile').glob('archive-*.json')):
        for r in json.loads(path.read_text())['results']:
            s = r['stages_seconds']; engine = s['engine_run']
            kernel = next(v for k, v in s.items() if k.startswith('kernel_only'))
            store = next(v for k, v in s.items() if k.startswith('store_quantize'))
            out[f'{path.stem}/{r["policy"]}'] = {
                'engine_seconds': engine, 'elementwise_operators_seconds': sum(v for k, v in s.items() if k.startswith('op_')),
                'hash_and_top5_seconds': sum(v for k, v in s.items() if k.startswith(('hash', 'top5'))),
                'kernel_share_of_engine': kernel / engine, 'store_quantize_share_of_engine': store / engine,
                'seconds_per_image': r['seconds_per_image']}
    return out


def fastblocks_ratio():
    f = {p.name.split('--')[0] + '--' + p.name.split('--')[1]: json.loads(p.read_text())
         for p in (SPEED / 'fastblocks' / 'fast').glob('*.execution.json')}
    o = {p.name.split('--')[0] + '--' + p.name.split('--')[1]: json.loads(p.read_text())
         for p in (SPEED / 'fastblocks' / 'original').glob('*.execution.json')}
    return {k: {'fast_seconds': f[k]['bias_correction_seconds'], 'original_seconds': o[k]['bias_correction_seconds'],
                'fast_over_original': f[k]['bias_correction_seconds'] / o[k]['bias_correction_seconds']} for k in o if k in f}


def cocoeval_evaluate():
    all_, eight = [], []
    for path in glob.glob(str(SPEED / 'cocoeval' / '03fa4eaee3d836a7' / '*.json')):
        d = json.loads(open(path).read())
        if 'evaluate_cpu_seconds' in d:
            all_.append(d['evaluate_cpu_seconds'])
            if '--d2000-' in path:
                eight.append(d['evaluate_cpu_seconds'])
    return {'all_runs': {'n': len(all_), 'min': min(all_), 'max': max(all_)},
            'bootstrap_2000_configurations': {'n': len(eight), 'min': min(eight), 'max': max(eight)}}


def failmix():
    picks_path = SPEED / 'validation' / 'failmix-picks.json'
    if not picks_path.exists():
        return None
    picks = json.loads(picks_path.read_text())
    out = {'picks_file': str(picks_path.relative_to(ROOT)), 'picks_sha256': sha(picks_path), 'cases': {}}
    for case, entry in picks['cases'].items():
        row = {'rule_policy': entry['rule_policy'],
               'scan_0_64': [{k: r[k] for k in ('policy', 'failed_of_64')} for r in entry['scan_0_64']],
               'scan_0_1000': [{k: r[k] for k in ('policy', 'failed_of_1000', 'mixed_batches_of_8')} for r in entry['scan_0_1000']],
               'pick': entry['pick'], 'compare': []}
        p = entry['pick']
        if p:
            for path in sorted((SPEED / 'validation' / 'compare' / TAG).glob(f'{case}--{p["policy"]}--{p["start"]}-{p["stop"]}--b*.json')):
                d = json.loads(path.read_text())
                d = d.get('payload', d) if isinstance(d, dict) else d
                row['compare'].append({'file': str(path.relative_to(ROOT)), **{k: d.get(k) for k in (
                    'batch', 'compared', 'identical', 'different', 'node_constants_equal', 'first_difference', 'archive_files')}})
        out['cases'][case] = row
    return out


def du(path):
    r = subprocess.run(['du', '-sb', str(path)], capture_output=True, text=True).stdout.split()
    return int(r[0]) if r else None


def main():
    mem = sorted((SPEED / 'fastblocks-mem-v1' / 'fast').glob('*.memory.json'))
    setup = SPEED / 'setup-v1' / 'setup.json'
    result = {
        'protocol': {'file': str(PROTOCOL.relative_to(ROOT)), 'sha256': sha(PROTOCOL), 'mtime': mtime(PROTOCOL)},
        'evidence_label': 'development evidence (ImageNet screen-1k; first 256 calibration images of the B2 cache, build=False)',
        'review': 'artifacts/agent_orchestration/handoffs/S1-speed-review.md, review 1 (approve_with_fixes)',
        'b1_n2_recomputed': {'concurrency_shared': concurrency(), 'archive_profile': profile(),
                             'fastblocks_fast_over_original': fastblocks_ratio(), 'cocoeval_evaluate_cpu_seconds': cocoeval_evaluate(),
                             'file_times': {k: mtime(ROOT / k) for k in (
                                 'public/experiments/configs/breadth-study/speed-protocol-v1.json',
                                 'results/summaries/speed-v1/summary.json')}},
        'n1_device_memory': [json.loads(p.read_text()) for p in mem],
        'n3_process_setup': json.loads(setup.read_text()) if setup.exists() else None,
        'n5_mixed_failure_batches': failmix(),
        'disk_bytes': {k: du(SPEED / k) for k in ('.', 'runs', 'review-scratch', 'archive-runs', 'fastblocks-mem-v1', 'validation')},
        'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    text = json.dumps(result, indent=1, sort_keys=True)
    if '--check' in sys.argv:
        print(text[:4000]); return
    if OUT.exists():
        raise SystemExit(f'{OUT} exists (written once)')
    OUT.write_text(text)
    print(OUT, sha(OUT))


if __name__ == '__main__':
    main()
