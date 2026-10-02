"""Analysis of the MobileNet accumulator sweep over both engine roots (lane Q1, protocol v1 with addendum 3). CPU only.

    .venv/bin/python -m tools.accumulator_sweep_mn.analysis2                 # write-once into results/summaries/accumulator-sweep-mn-v1/
    .venv/bin/python -m tools.accumulator_sweep_mn.analysis2 --out DIR       # any other (scratch) folder; may be partial

The statistics are those of tools/accumulator_sweep_mn/analysis.py, unchanged: this module imports it and, inside this
process only, points its loader at the two-root loader of fast.py (1f75c923 archive root + lane S1's fast root
edfecd5c; addendum 3 assembly rule).  Added: per row the engine identity of every file (root, engine_sources,
base_engine_sources, execution path); timing columns only from archive files (the fast path's seconds are not used
for timing statements); the outcome of checks K1-K4; summary.json names both engines instead of one digest.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from . import analysis as a1
from . import fast

ROOT = a1.ROOT
OUT = a1.OUT
CODE_FILES = a1.CODE_FILES + ('tools/accumulator_sweep_mn/analysis2.py', 'tools/accumulator_sweep_mn/fast.py')


def policies_with(case, start, stop):
    names = set()
    for root in (fast.ARCHIVE_ROOT, fast.FAST_ROOT):
        folder = root / case / 'predictions'
        names |= {re.sub(r'-cuda-\d{5}-\d{5}\.json$', '', p.name) for p in folder.glob('*-cuda-*.json')} if folder.exists() else set()
    return sorted(p for p in names if fast.choose(case, p, start, stop) is not None)


def install():
    a1.assemble = fast.assemble
    a1.policies_with = policies_with


def engine_columns(row):
    files = row.pop('files', None) or []
    roots = sorted({f['root'] for f in files})
    row['engine_roots'] = '+'.join(roots)
    row['engine_files'] = [{k: f[k] for k in ('path', 'root', 'sha256', 'start', 'stop', 'batch', 'engine_sources',
                                               'base_engine_sources', 'execution_path')} for f in files]
    arch = [f for f in files if f['root'] == 'archive']
    row['archive_execution_seconds'] = sum(f['execution_seconds'] for f in arch) if arch else None
    row['archive_images_timed'] = sum(f['stop'] - f['start'] for f in arch) if arch else 0
    row.pop('execution_seconds', None)
    return row


def checks():
    out = {}
    d = fast.CHECKS
    if not d.exists():
        return out
    k1 = sorted(d.glob('k1-*.json'))
    out['K1'] = {'records': len(k1), 'all_pass': all(json.load(open(p))['status'] == 'pass' for p in k1)}
    out['K2'] = {p.stem[3:]: {'status': (j := json.load(open(p)))['status'],
                              **{pol: [v['same'], v['differ']] for pol, v in j['policies'].items()}} for p in sorted(d.glob('k2-*.json'))}
    out['K3'] = [{k: (j := json.load(open(p)))[k] for k in ('case', 'policy', 'start', 'stop', 'same', 'differ', 'status')}
                 for p in sorted(d.glob('k3-*.json'))]
    xr = sorted(d.glob('xroot-*.json'))
    if xr:
        j = json.load(open(xr[-1]))
        out['cross_root_last'] = {'file': str(xr[-1].relative_to(ROOT)), 'pairs': len(j['pairs']),
                                  'same': sum(p['same'] for p in j['pairs']), 'differ': sum(p['differ'] for p in j['pairs'])}
    k4 = sorted((fast.LANE / 'k4').glob('k4-round*-result-*.json'))
    out['K4'] = [{'file': str(p.relative_to(ROOT)),
                  'files': len(j['results']), 'pass': sum(r['status'] == 'pass' for r in j['results']),
                  'same': sum(r.get('same') or 0 for r in j['results']), 'differ': sum(r.get('differ') or 0 for r in j['results'])}
                 for p in k4 for j in [json.load(open(p))]]
    out['FAST_HALT'] = fast.HALT.exists()
    return out


def main(argv=None):
    install()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'{out} exists and is not empty; summaries are written once (use a new version)')
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.load(open(a1.PROTOCOL))
    addenda = sorted(a1.PROTOCOL.parent.glob('accumulator-sweep-mn-protocol-v1-addendum-*.json'))
    for a in addenda:
        protocol['cases'] += json.load(open(a)).get('cases_added', [])
    result = a1.analyse(protocol)
    result['policies'] = [engine_columns(r) for r in result['policies']]
    for name in ('policies', 'node_events', 'kind_events', 'widths', 'location_128'):
        a1.write_csv(out / f'{name}.csv', [{k: v for k, v in r.items() if k not in ('files', 'engine_files')} for r in result[name]])
    a1.write_csv(out / 'simulator.csv', result['simulator'])
    summary = {
        'id': 'accumulator-sweep-mn-v1', 'written': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'evidence': 'development evidence, ImageNet screen-1k list (images 0-999; location 0-127); no held-out image',
        'engines': {
            'archive': {'engine_sources': fast.BASE_DIGEST, 'run_root': str(fast.ARCHIVE_ROOT.relative_to(ROOT)),
                        'execution_path': 'archive 1f75c923 (scaled bridge v2): host NumPy post-operations, kernels on copied buffers'},
            'fast': {'engine_sources': fast.FAST_DIGEST, 'base_engine_sources': fast.BASE_DIGEST,
                     'run_root': str(fast.FAST_ROOT.relative_to(ROOT)), 'execution_path': fast.FAST_PATH_TEXT,
                     'review': 'artifacts/agent_orchestration/handoffs/S1-engine-review.md (COMBINED VERDICT)'},
            'per_file': 'policies[*].engine_files',
            'timing_rule': 'archive_execution_seconds only; the fast path is not used for timing statements'},
        'checks': checks(),
        'protocol': {'path': str(a1.PROTOCOL.relative_to(ROOT)), 'sha256': hashlib.sha256(a1.PROTOCOL.read_bytes()).hexdigest(),
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
