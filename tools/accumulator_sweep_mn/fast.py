"""Lane Q1 on lane S1's fast exact-engine path (protocol addendum 3): engine identities, check K1, the two-root
loader, cross-root identity comparisons (K2, K3, K4) and the GPU-time accounting of fast jobs.

The fast archive edfecd5c... (artifacts/speed_v1/implementations/<digest>/run.sh) computes the same records as the
1f75c923 archive and writes them under its own run root artifacts/speed_v1/runs/<digest>/ (lane S1's tree; this lane
adds files there only through that run.sh's `predict`).  Certificates and every earlier file stay in the 1f75c923
run root.  Nothing here writes into either run root.
"""
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

from tools.experiment_b.common import unseal

from . import certs
from .load import covering, prediction_files

ROOT = certs.ROOT
BASE_DIGEST = certs.DIGEST                     # 1f75c923...: archive engine, certificates, earlier files
ARCHIVE_ROOT = certs.RUN_ROOT
FAST_DIGEST = 'edfecd5cdb0dc5adf06eff2807afa023b12d781cf0d08961a0b023da510a293b'
FAST_ARCHIVE = f'artifacts/speed_v1/implementations/{FAST_DIGEST}'
FAST_ROOT = ROOT / 'artifacts/speed_v1/runs' / FAST_DIGEST
FAST_PATH_TEXT = 'scaled_bridge_fast (lane S1): archived kernels on device pointers, GPU post-operations'

LANE = ROOT / 'artifacts/accumulator_sweep_mn_v1'
CHECKS = LANE / 'checks'
HALT = LANE / 'FAST-HALT'        # written when any fast record differs from an archive record or its repeat
STOP = LANE / 'STOP'             # stop switch of the runner and the fast jobs (exit before the next call)
FAST_JOBS = LANE / 'logs' / 'fast-jobs.jsonl'

# Versions the combined verdict's tested links depend on (recorded in addendum 3; K1 requires equality).
VERSIONS = {'torch': '2.3.0+cu121', 'numpy': '1.26.4', 'cuda_runtime': '12.1', 'driver': '595.91.07'}
SEED_K3 = 3102001
SEED_K4 = 3102002
K3_IMAGES = 32
K4_SHARE = 0.10
K4_BATCH = 5


# --- K1 -----------------------------------------------------------------------------------------------------------
def manifest_check(root=ROOT):
    """sha256 of every file listed in the fast archive's manifest.json (14 files); returns (ok, mismatches)."""
    archive = Path(root) / FAST_ARCHIVE
    doc = unseal(archive / 'manifest.json')
    bad = []
    for rel, want in sorted(doc['files'].items()):
        path = archive / rel
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if got != want:
            bad.append({'file': rel, 'manifest': want, 'found': got})
    if doc['fast_sources_digest'] != FAST_DIGEST or doc['identity']['base_engine_sources_digest'] != BASE_DIGEST:
        bad.append({'file': 'manifest identity', 'found': [doc['fast_sources_digest'], doc['identity']['base_engine_sources_digest']]})
    return not bad, bad, len(doc['files'])


def versions():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    out = subprocess.run([str(ROOT / '.venv-b/bin/python'), '-c',
                          'import torch, numpy, json; print(json.dumps([torch.__version__, numpy.__version__, torch.version.cuda]))'],
                         capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    torch_v, numpy_v, cuda_v = json.loads(out.stdout.strip().splitlines()[-1])
    drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True,
                         text=True, timeout=60).stdout.strip().splitlines()[0].strip()
    return {'torch': torch_v, 'numpy': numpy_v, 'cuda_runtime': cuda_v, 'driver': drv}


def run_root_check():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    out = subprocess.run([f'{FAST_ARCHIVE}/run.sh', 'root'], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    text = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
    return out.returncode == 0 and text.endswith(FAST_DIGEST), text


def k1(write=True, who='runner'):
    """Check K1 (combined verdict (c)): manifest hashes, `run.sh root`, torch/NumPy/driver versions.  Raises on failure;
    the record goes to artifacts/accumulator_sweep_mn_v1/checks/k1-<time>-<who>.json."""
    ok_m, bad, n = manifest_check()
    ok_r, root_text = run_root_check()
    found = versions()
    ok_v = found == VERSIONS
    rec = {'check': 'K1', 'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'who': who, 'manifest_files': n,
           'manifest_ok': ok_m, 'manifest_mismatches': bad, 'run_root': root_text, 'run_root_ok': ok_r,
           'versions': found, 'versions_expected': VERSIONS, 'versions_ok': ok_v,
           'pythondontwritebytecode': os.environ.get('PYTHONDONTWRITEBYTECODE') == '1',
           'status': 'pass' if ok_m and ok_r and ok_v else 'fail'}
    if write:
        CHECKS.mkdir(parents=True, exist_ok=True)
        (CHECKS / f"k1-{time.strftime('%Y%m%d-%H%M%S')}-{who}-{os.getpid()}.json").write_text(json.dumps(rec, indent=1) + '\n')
    if rec['status'] != 'pass':
        raise SystemExit(f'K1 failed: {json.dumps(rec)}')
    return rec


# --- cases and routing --------------------------------------------------------------------------------------------
def gated(case, root=None):
    """True when lane S1's fast case gate exists and passes (the fast predict refuses a case without it)."""
    path = Path(root or FAST_ROOT) / case / 'gate.json'
    if not path.exists():
        return False
    try:
        return unseal(path).get('status') == 'pass'
    except (OSError, ValueError):
        return False


def k2_record(case):
    return CHECKS / f'k2-{case}.json'


def k2_passed(case):
    path = k2_record(case)
    return path.exists() and json.loads(path.read_text()).get('status') == 'pass'


def engine_for(case, gate_root=None):
    """'fast' for a gated case (its runs go through the fast run.sh after K2), 'archive' otherwise."""
    return 'fast' if gated(case, gate_root) else 'archive'


# --- two-root loader ----------------------------------------------------------------------------------------------
def files(case, policy, root):
    return [(a, b, p) for a, b, p in prediction_files(case, policy, root=root)]


def choose(case, policy, start, stop, arch_root=None, fast_root=None):
    """Tiles of [start, stop): from the 1f75c923 root alone if it covers the range, else from the fast root alone,
    else from both (greedy, furthest reach first, archive first on ties).  Returns [(a, b, path, file_start, root_kind)]."""
    arch = [(*f, 'archive') for f in files(case, policy, arch_root or ARCHIVE_ROOT)]
    fast = [(*f, 'fast') for f in files(case, policy, fast_root or FAST_ROOT)]
    for pool in (arch, fast, arch + fast):
        kinds = {f[2]: f[3] for f in pool}
        tiles = covering([f[:3] for f in pool], start, stop)
        if tiles is not None:
            return [(a, b, p, f0, kinds[p]) for a, b, p, f0 in tiles]
    return None


def fast_covers(case, policy, start, stop, fast_root=None):
    return covering(files(case, policy, fast_root or FAST_ROOT), start, stop) is not None


def check_identity(payload, kind, case, policy, f0, path):
    """Engine identity of one file by its root: archive files must carry engine_sources 1f75c923; fast files
    engine_sources edfecd5c with base_engine_sources 1f75c923 and the fast execution path."""
    if payload['case'] != case or payload['policy'] != policy or payload['start'] != f0:
        raise ValueError(f'prediction file does not match its name {path}')
    if kind == 'archive':
        ok = payload['engine_sources'] == BASE_DIGEST and 'base_engine_sources' not in payload
    elif kind == 'fast':
        ok = (payload['engine_sources'] == FAST_DIGEST and payload.get('base_engine_sources') == BASE_DIGEST
              and payload.get('execution_path') == FAST_PATH_TEXT)
    else:
        ok = False
    if not ok:
        raise ValueError(f'{path}: engine identity {payload.get("engine_sources")} / {payload.get("base_engine_sources")} '
                         f'is not admitted for a file of the {kind} root')


def engine_ref(kind, payload):
    if kind == 'archive':
        return {'engine_sources': BASE_DIGEST, 'base_engine_sources': None, 'execution_path': 'archive 1f75c923 (scaled bridge v2)'}
    return {'engine_sources': payload['engine_sources'], 'base_engine_sources': payload['base_engine_sources'],
            'execution_path': payload['execution_path']}


def assemble(case, policy, start, stop, backend='cuda', root=None, check=True, arch_root=None, fast_root=None):
    """Image records [start, stop) of one case and policy from the two run roots (same return shape as load.assemble,
    with each file reference naming its root and engine identity)."""
    tiles = choose(case, policy, start, stop, arch_root, fast_root)
    if tiles is None:
        return None
    images, nodes, refs, seconds = [], None, [], 0.0
    for a, b, path, f0, kind in tiles:
        p = unseal(path) if check else json.load(open(path))['payload']
        check_identity(p, kind, case, policy, f0, path)
        if p['stop'] < b:
            raise ValueError(f'prediction file does not match its name {path}')
        if nodes is None or not nodes:
            nodes = p['nodes'] if nodes is None or p['nodes'] else nodes
        elif p['nodes'] and nodes != p['nodes']:
            raise ValueError('node constants differ between files')
        images.extend(p['images'][a - p['start']:b - p['start']]); seconds += p['execution_seconds']
        try:
            rel = str(Path(path).relative_to(ROOT))
        except ValueError:
            rel = str(path)
        refs.append({'path': rel, 'root': kind, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     'start': a, 'stop': b, 'execution_seconds': p['execution_seconds'], 'batch': p['batch'],
                     **engine_ref(kind, p)})
    return {'images': images, 'nodes': nodes, 'files': refs, 'execution_seconds': seconds,
            'roots': sorted({r['root'] for r in refs})}


def prefix_end(case, policy, stop, root=None):
    """End of the longest contiguous coverage from image 0 (at most stop) over both roots."""
    best = 0
    cands = sorted({b for a, b, _ in files(case, policy, ARCHIVE_ROOT) + files(case, policy, FAST_ROOT)})
    for b in cands:
        if best < b <= stop and choose(case, policy, 0, b) is not None:
            best = b
    return best


# --- identity comparisons ------------------------------------------------------------------------------------------
def compare_records(fast_images, fast_nodes, ref_images, ref_nodes):
    """Field-by-field equality on the overlapping image indices; node tables equal when both are non-empty."""
    ref = {r['index']: r for r in ref_images}
    same = differ = 0
    first = None
    for r in fast_images:
        o = ref.get(r['index'])
        if o is None:
            continue
        if o == r:
            same += 1
        else:
            differ += 1
            if first is None:
                first = {'index': r['index'], 'fields': sorted(k for k in set(o) | set(r) if o.get(k) != r.get(k))}
    nodes_equal = (not fast_nodes or not ref_nodes or fast_nodes == ref_nodes)
    return {'same': same, 'differ': differ, 'nodes_equal': nodes_equal, 'first_difference': first,
            'status': 'pass' if differ == 0 and nodes_equal and same > 0 else ('fail' if differ or not nodes_equal else 'empty')}


def file_ref(path):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def cross_root(case, policy, start, stop):
    """Compare every fast-root record in [start, stop) with every archive-root record of the same images."""
    fast_imgs, fast_nodes, fast_refs = [], None, []
    for a, b, p in files(case, policy, FAST_ROOT):
        if b <= start or a >= stop:
            continue
        d = unseal(p); check_identity(d, 'fast', case, policy, a, p)
        fast_imgs += [r for r in d['images'] if start <= r['index'] < stop]
        fast_nodes = fast_nodes or d['nodes']; fast_refs.append(file_ref(p))
    arch_imgs, arch_nodes, arch_refs = [], None, []
    for a, b, p in files(case, policy, ARCHIVE_ROOT):
        if b <= start or a >= stop:
            continue
        d = unseal(p); check_identity(d, 'archive', case, policy, a, p)
        arch_imgs += [r for r in d['images'] if start <= r['index'] < stop]
        arch_nodes = arch_nodes or d['nodes']; arch_refs.append(file_ref(p))
    res = compare_records(fast_imgs, fast_nodes, arch_imgs, arch_nodes)
    return {**res, 'case': case, 'policy': policy, 'start': start, 'stop': stop, 'fast_files': fast_refs, 'archive_files': arch_refs}


def halt(reason, evidence):
    """Stop every fast run: write FAST-HALT with the evidence (kept; nothing is deleted)."""
    rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'reason': reason, 'evidence': evidence}
    with open(HALT, 'a') as h:
        h.write(json.dumps(rec) + '\n')
    return rec


def k2(case):
    """K2: fast wide and control over 0-1000 against this lane's archive files (all 1,000 images, every field)."""
    out = {'check': 'K2', 'case': case, 'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'policies': {}}
    for pol in ('wide', 'control'):
        r = cross_root(case, pol, 0, 1000)
        out['policies'][pol] = r
    statuses = [r['status'] for r in out['policies'].values()]
    full = all(r['same'] == 1000 for r in out['policies'].values())
    out['status'] = 'pass' if statuses == ['pass', 'pass'] and full else 'fail'
    CHECKS.mkdir(parents=True, exist_ok=True)
    k2_record(case).write_text(json.dumps(out, indent=1) + '\n')
    if out['status'] != 'pass':
        halt(f'K2 failed for {case}', str(k2_record(case).relative_to(ROOT)))
    return out


# --- seeded draws of K3 and K4 (addendum 3) -----------------------------------------------------------------------
def k3_start(case):
    """Start of the 32-image K3 anchor range inside the location tile 0-128 (seeded, per case)."""
    return random.Random(f'{SEED_K3}:{case}:start').randrange(0, 128 - K3_IMAGES + 1)


def k3_policies(case, location_uniform, grid_policies, sensitivity_f21=None):
    """K3 anchor policies: (1) the widest location width with an event (sat.w<W>), (2) one further policy drawn
    with seed SEED_K3 from the case's 1k grid (wide, control and (1) excluded; sorted names), (3) on a sensitivity
    case, its f21.x<e> location policy."""
    out = []
    ev = [w for w, r in location_uniform.items() if r['events'] > 0]
    if ev:
        out.append(f'sat.w{max(ev)}')
    rest = sorted(p for p in set(grid_policies) if p not in ('wide', 'control') and p not in out)
    if rest:
        out.append(random.Random(f'{SEED_K3}:{case}:policy').choice(rest))
    if sensitivity_f21:
        out.append(sensitivity_f21)
    return out


def k4_key(case, policy, start, stop):
    return int(hashlib.sha256(f'{SEED_K4}:{case}:{policy}:{start}:{stop}'.encode()).hexdigest(), 16)


def k4_sample(candidates, already=()):
    """K4 round: of the fast 1k files not sampled before, the ceil(10 %) with the smallest seeded hash."""
    import math
    pool = sorted((k4_key(*c), c) for c in candidates if tuple(c) not in {tuple(a) for a in already})
    n = math.ceil(K4_SHARE * len(pool)) if pool else 0
    return [c for _, c in pool[:n]]


# --- GPU-time accounting of fast jobs --------------------------------------------------------------------------------
def fast_job_seconds(since=0.0, path=None, now=None):
    """GPU job seconds of this lane's fast jobs: for every job, the wall time from its start (inside gpu_run, i.e.
    after admission) to its end record, or to its last call record if it has no end record (a job that died), or
    to now while it is still running (pid alive).  Calls are not summed separately (they lie inside their job)."""
    path = Path(path or FAST_JOBS)
    if not path.exists():
        return 0.0
    jobs = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        j = jobs.setdefault(r['job'], {'begin': None, 'end': None, 'last': None, 'pid': r.get('pid')})
        if r['type'] == 'begin':
            j['begin'] = r['time']; j['pid'] = r.get('pid')
        elif r['type'] == 'end':
            j['end'] = r['time']
        else:
            j['last'] = max(j['last'] or 0, r.get('ended') or r['time'])
    total = 0.0
    now = time.time() if now is None else now
    for j in jobs.values():
        if j['begin'] is None or j['begin'] < since:
            continue
        end = j['end']
        if end is None:
            alive = j['pid'] is not None and Path(f"/proc/{j['pid']}").exists()
            end = now if alive else (j['last'] or j['begin'])
        total += max(0.0, end - j['begin'])
    return total
