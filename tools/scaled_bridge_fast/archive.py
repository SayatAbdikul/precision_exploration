"""Gates and archive of the fast exact-engine path (lane S1; not part of the fast-path digest).

gates(): for every case of artifacts/speed_v1/validation/plan.json, writes under the fast run root
  <case>/gate.json            case gate: (1) a passing archive case gate exists for the case (any scaled bridge v2
                              run root), (2) every full-trace comparison of the case (MN-R regression sets and the
                              archive's sealed gate panels) is identical, including wide and control, (3) wide and
                              control predict items identical to the archive's on >= 64 images at batch 1, 8 and 32
  <case>/gate-<policy>.json   policy gate: the case gate passes; the policy's predict items identical to the
                              archive's on >= 64 images at batch 1, 8 and 32 (node constants equal); every
                              full-trace set of the policy on the case identical
  `predict` of the fast path then follows the archives' rule (worker.require_gates): the case gate, and for a
  parameterised policy a passing policy gate of the same family on a case of the same model.
archive(): artifacts/speed_v1/implementations/<digest>/ with py/scaled_bridge_fast/, lib/cuda/fastdot.{so,json},
  run.sh and manifest.json; its run root is artifacts/speed_v1/runs/<digest>/ (the live package's run root).
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from .common import ROOT, SPEED, V2, PACKAGE, digest, file_hash, identity, run_root, unseal, immutable, BASE_DIGEST

BATCHES = (1, 8, 32)
IMAGES = 64
RUN = '''#!/usr/bin/env bash
# Runs the archived fast exact-engine path (lane S1) {name} from the repository root.
# Usage: run.sh <command> [args]   (predict CASE POLICY cuda START STOP [--batch B], as the scaled bridge v2 archives;
# default batch 8 here: fastest measured, about 1-2 GB of GPU memory; batch 32 on MobileNets can exceed 4 GB -> --heavy 8000)
# GPU commands go through artifacts/agent_orchestration/gpu_run.sh (shared mode, --min-free-mib 3000).
HERE="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
cd -- "$HERE/../../../.." || exit 1
PY="${{SCALED_BRIDGE_FAST_PYTHON:-.venv-b/bin/python}}"
export CUDA_CACHE_PATH="${{CUDA_CACHE_PATH:-$PWD/artifacts/speed_v1/cuda-cache}}"
# The run root also depends on the base archive 1f75c923 and its tracked dependencies: refuse if it moved.
NOW="$(PYTHONPATH="$HERE/py" "$PY" -m scaled_bridge_fast root)" || exit 1
if [ "$(basename -- "$NOW")" != "{name}" ]; then
  echo "fast archive {name}: the run root resolves to $NOW (a source or dependency changed); refusing to run" >&2
  exit 1
fi
PYTHONPATH="$HERE/py" exec "$PY" -m scaled_bridge_fast "$@"
'''


def _results(kind):
    folder = SPEED / 'validation' / kind / digest(identity())[:16]
    out = []
    for path in sorted(folder.glob('*.json')):
        r = unseal(path)
        if r.get('fast_identity') == identity():
            out.append((path, r))
    return out


def _archive_gate(case):
    for path in sorted((V2 / 'runs').glob(f'*/{case}/gate.json')):
        if unseal(path)['status'] == 'pass':
            return str(path.relative_to(ROOT))
    return None


def gates():
    plan = json.loads((SPEED / 'validation' / 'plan.json').read_text())
    full = _results('regress') + _results('trace')
    pred = _results('compare')
    report = {}
    for case, entry in plan.items():
        sets = [(p, r) for p, r in full if r['case'] == case]
        full_ok = {r['policy']: all(x['identical'] == x['images'] and not x['differences'] for _, x in sets if x['policy'] == r['policy'])
                   for _, r in sets}

        def predicted(policy):
            rows = [(p, r) for p, r in pred if r['case'] == case and r['policy'] == policy and r['start'] == 0]
            ok = {b: any(r['batch'] == b and r['compared'] >= IMAGES and r['different'] == 0
                         and r['node_constants_equal'] is not False for _, r in rows) for b in BATCHES}
            return all(ok.values()) and all(r['different'] == 0 for _, r in rows), rows
        refs = lambda rows: [{'file': str(p.relative_to(ROOT)), 'sha256': file_hash(p)} for p, _ in rows]  # noqa: E731
        archive_gate = _archive_gate(case)
        wc = [predicted(p) for p in ('wide', 'control')]
        case_ok = bool(archive_gate) and full_ok.get('wide') and full_ok.get('control') and all(full_ok.values()) \
            and all(ok for ok, _ in wc)
        report[case] = {'case_gate': bool(case_ok)}
        if not case_ok:
            continue
        immutable(run_root() / case / 'gate.json', {
            'status': 'pass', 'case': case, 'identity': identity(), 'engine_sources': digest(identity()),
            'base_engine_sources': BASE_DIGEST, 'archive_case_gate': archive_gate,
            'full_trace_sets': refs(sets), 'predict_comparisons': refs(wc[0][1] + wc[1][1]),
            'criteria': __doc__.split('gates():')[1].split('archive():')[0].strip()})
        for policy in entry['policies'].values():
            ok, rows = predicted(policy)
            ok = ok and full_ok.get(policy, True)
            report[case][policy] = ok
            if ok:
                immutable(run_root() / case / f'gate-{policy}.json', {
                    'status': 'pass', 'case': case, 'policy': policy, 'identity': identity(),
                    'engine_sources': digest(identity()), 'predict_comparisons': refs(rows),
                    'full_trace_sets': refs([(p, r) for p, r in sets if r['policy'] == policy])})
    return report


def archive():
    from .native import build
    name = digest(identity()); folder = SPEED / 'implementations' / name
    if (folder / 'manifest.json').exists():
        saved = unseal(folder / 'manifest.json')
        if any(file_hash(folder / k) != v for k, v in saved['files'].items()):
            raise ValueError('archived fast implementation drift')
        return folder
    for path in sorted(PACKAGE.glob('*')):
        if path.suffix in {'.py', '.cu', '.h'}:
            target = folder / 'py/scaled_bridge_fast' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    library = build(); target = folder / 'lib' / 'cuda'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library, target / 'fastdot.so')
    shutil.copyfile(library.with_suffix('.json'), target / 'fastdot.json')
    script = folder / 'run.sh'
    script.write_text(RUN.format(name=name)); script.chmod(0o755)
    listed = sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    immutable(folder / 'manifest.json', {
        'fast_sources_digest': name, 'identity': identity(), 'native': unseal(target / 'fastdot.json'),
        'files': {str(p.relative_to(folder)): file_hash(p) for p in listed},
        'run_root': str((SPEED / 'runs' / name).relative_to(ROOT)),
        'base_archive': f'artifacts/scaled_bridge_v2/implementations/{BASE_DIGEST}',
        'usage': 'from the repository root: GPU_LANE=<lane> artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 '
                 f'artifacts/speed_v1/implementations/{name}/run.sh predict CASE POLICY cuda START STOP --batch 8',
        'not_numeric': 'archive.py, profile.py, cli.py, __main__.py, validate.py and timing.py are not part of the digest'})
    return folder
