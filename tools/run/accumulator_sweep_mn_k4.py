"""Check K4 of lane Q1 (protocol accumulator-sweep-mn-protocol-v1, addendum 3): a seeded 10 % sample of this lane's
fast 1k files recomputed in a second process at batch 5 into a scratch root, compared field by field.

    .venv/bin/python -m tools.run.accumulator_sweep_mn_k4 plan                 # CPU: draw the round's sample
    GPU_LANE=Q1 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 --wait 7200 \
        env PYTHONDONTWRITEBYTECODE=1 .venv-b/bin/python -m tools.run.accumulator_sweep_mn_k4 run <plan.json>

`run` imports the ARCHIVED fast package (artifacts/speed_v1/implementations/<edfecd5c...>/py, run.sh's sys.path
order) and, inside this process only, points its speed root at artifacts/accumulator_sweep_mn_v1/k4/speed (as lane
S1's reviewer D did), copies the fast gates of the model there, and calls its own CLI `predict ... --batch 5`.
Nothing is written under artifacts/speed_v1/.  The result is compared with the real fast file (images and nodes);
any difference writes FAST-HALT.  Stop switch: artifacts/accumulator_sweep_mn_v1/STOP (checked before each file).
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.accumulator_sweep_mn import fast  # noqa: E402

K4 = fast.LANE / 'k4'


def lane_fast_1k():
    """(case, policy, 0, 1000) of every fast 1k file this lane's fast jobs made (exit 0, file present)."""
    out = set()
    if fast.FAST_JOBS.exists():
        for line in fast.FAST_JOBS.read_text().splitlines():
            r = json.loads(line)
            if r.get('type') == 'call' and r.get('exit') == 0 and (r['start'], r['stop']) == (0, 1000):
                p = fast.FAST_ROOT / r['case'] / 'predictions' / f"{r['policy']}-cuda-00000-01000.json"
                if p.exists():
                    out.add((r['case'], r['policy'], 0, 1000))
    return sorted(out)


def plan():
    K4.mkdir(parents=True, exist_ok=True)
    rounds = sorted(K4.glob('k4-round*-plan.json'))
    already = [tuple(x) for p in rounds for x in json.loads(p.read_text())['sample']]
    cands = lane_fast_1k()
    sample = fast.k4_sample(cands, already)
    n = len(rounds) + 1
    rec = {'check': 'K4', 'round': n, 'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'seed': fast.SEED_K4,
           'rule': 'ceil(10 %) of the not yet sampled fast 1k files with the smallest sha256(seed:case:policy:start:stop)',
           'candidates': len(cands), 'previously_sampled': len(already), 'sample': [list(s) for s in sample], 'batch': fast.K4_BATCH}
    path = K4 / f'k4-round{n}-plan.json'
    path.write_text(json.dumps(rec, indent=1) + '\n')
    print(path.relative_to(ROOT), len(sample), 'of', len(cands))
    return path


def compare_existing(plan_path):
    """CPU: compare the repeat files already in the scratch root with the fast files (used when `run` computed them but
    could not write its result file, 2026-10-02 round 1)."""
    from tools.experiment_b.common import unseal
    plan_path = Path(plan_path).resolve()
    spec = json.loads(plan_path.read_text())
    results = []
    for case, policy, a, b in spec['sample']:
        name = f'{policy}-cuda-{a:05d}-{b:05d}.json'
        rep = K4 / 'speed' / 'runs' / fast.FAST_DIGEST / case / 'predictions' / name
        ref = fast.FAST_ROOT / case / 'predictions' / name
        rec = {'case': case, 'policy': policy, 'start': a, 'stop': b, 'fast_file': str(ref.relative_to(ROOT)),
               'fast_sha256': hashlib.sha256(ref.read_bytes()).hexdigest()}
        if not rep.exists():
            rec['status'] = 'missing'; results.append(rec); continue
        r, o = unseal(rep), unseal(ref)
        cmp = fast.compare_records(r['images'], r['nodes'], o['images'], o['nodes'])
        rec.update({'repeat_file': str(rep.relative_to(ROOT)), 'repeat_batch': r['batch'], 'fast_batch': o['batch'],
                    'same': cmp['same'], 'differ': cmp['differ'], 'nodes_equal': r['nodes'] == o['nodes'],
                    'images_complete': len(r['images']) == len(o['images']) == b - a, 'first_difference': cmp['first_difference'],
                    'repeat_engine_sources': r['engine_sources'], 'repeat_wall_seconds': r['wall_seconds']})
        rec['status'] = 'pass' if cmp['differ'] == 0 and rec['nodes_equal'] and rec['images_complete'] else 'fail'
        if rec['status'] == 'fail':
            fast.halt(f'K4 repeat differs: {case} {policy}', rec)
        results.append(rec)
    out = plan_path.with_name(plan_path.name.replace('-plan.json', f"-result-{time.strftime('%H%M%S')}.json"))
    out.write_text(json.dumps({'plan': str(plan_path.relative_to(ROOT)), 'time': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                               'mode': 'compare of existing repeat files', 'results': results}, indent=1) + '\n')
    print(out.relative_to(ROOT), sum(r['status'] == 'pass' for r in results), 'pass of', len(results), flush=True)


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    spec = json.loads(plan_path.read_text())
    arch = ROOT / fast.FAST_ARCHIVE
    os.chdir(ROOT)
    os.environ['CUDA_CACHE_PATH'] = str(K4 / 'cuda-cache')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT), str(arch / 'py')]
    import scaled_bridge_fast.common as c
    assert Path(c.__file__).resolve().parent == arch / 'py/scaled_bridge_fast', c.__file__
    assert c.run_root().name == fast.FAST_DIGEST, c.run_root()
    import scaled_bridge_fast.native as n
    import scaled_bridge_fast.cli as cli
    real_root = c.run_root()
    speed = K4 / 'speed'
    c.SPEED = speed; n.SPEED = speed
    root = c.run_root()
    assert root.is_relative_to(K4) and root.name == fast.FAST_DIGEST, root
    from tools.experiment_b.common import unseal
    results = []
    for case, policy, a, b in spec['sample']:
        if fast.STOP.exists() or fast.HALT.exists():
            print('STOP/HALT before', case, policy, flush=True); break
        model = case.split('-')[0]
        for g in sorted(real_root.glob(f'{model}-*/gate*.json')):
            t = root / g.parent.name / g.name
            if not t.exists():
                t.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(g, t)
        tick = time.time(); err = None
        try:
            cli.main(['predict', case, policy, 'cuda', str(a), str(b), '--batch', str(spec['batch'])])
        except Exception as e:  # noqa: BLE001  (recorded)
            err = f'{type(e).__name__}: {e}'
        name = f'{policy}-cuda-{a:05d}-{b:05d}.json'
        rep, ref = root / case / 'predictions' / name, real_root / case / 'predictions' / name
        rec = {'case': case, 'policy': policy, 'start': a, 'stop': b, 'error': err, 'seconds': time.time() - tick,
               'fast_file': str(ref.relative_to(ROOT)), 'fast_sha256': hashlib.sha256(ref.read_bytes()).hexdigest()}
        if err is None:
            r, o = unseal(rep), unseal(ref)
            cmp = fast.compare_records(r['images'], r['nodes'], o['images'], o['nodes'])
            rec.update({'repeat_file': str(rep.relative_to(ROOT)), 'repeat_batch': r['batch'], 'fast_batch': o['batch'],
                        'same': cmp['same'], 'differ': cmp['differ'], 'nodes_equal': r['nodes'] == o['nodes'],
                        'images_complete': len(r['images']) == len(o['images']) == b - a,
                        'first_difference': cmp['first_difference']})
            rec['status'] = 'pass' if cmp['differ'] == 0 and rec['nodes_equal'] and rec['images_complete'] else 'fail'
            if rec['status'] == 'fail':
                fast.halt(f'K4 repeat differs: {case} {policy}', rec)
        else:
            rec['status'] = 'error'
        results.append(rec)
        print(json.dumps({k: rec[k] for k in ('case', 'policy', 'status', 'seconds')} | {'same': rec.get('same'), 'differ': rec.get('differ')}), flush=True)
    out = plan_path.with_name(plan_path.name.replace('-plan.json', f"-result-{time.strftime('%H%M%S')}.json"))
    out.write_text(json.dumps({'plan': str(plan_path.relative_to(ROOT)), 'time': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                               'results': results}, indent=1) + '\n')
    print(out.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'plan':
        plan()
    elif sys.argv[1] == 'compare':
        compare_existing(sys.argv[2])
    elif sys.argv[1] == 'run':
        run(sys.argv[2])
    else:
        sys.exit(__doc__)
