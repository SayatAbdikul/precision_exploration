"""Second runner of the MobileNet accumulator sweep (lane Q1): the same protocol rules, routed over two engines
(protocol accumulator-sweep-mn-protocol-v1, addendum 3).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m tools.run.accumulator_sweep_mn_run2 [--jobs 2] [--max-priority 4] [--dry-run] [--tag T]

- Check K1 first (manifest hashes of the fast archive, its `run.sh root`, torch/NumPy/driver versions); refuses to start
  if it fails.
- Gated cases (lane S1's fast case gate exists): fast jobs, i.e. gpu_run.sh --min-free-mib 3000 --wait 7200
  .venv/bin/python -m tools.run.accumulator_sweep_mn_fastjob (a job takes queued runs one call at a time while it
  holds the slot); check K2 (fast wide/control 0-1000 = archive files) before any other run of the case.
- Ungated cases: one archive call per GPU job, exactly as the first runner (1f75c923 run.sh, addendum-2 tiles).
- Check K3 (archive anchors on the location tile, seeded draws of addendum 3) once a gated case's location phase is
  complete: archive 32-image calls, compared with the fast records as soon as both exist.
- Any difference -> FAST-HALT: no new job, fast jobs stop before their next call; the runner then exits.
- At most --jobs GPU jobs of this lane at once, counting the first runner's archive calls still in flight.
- Budget: archive calls by the first runner's rule + fast jobs by their wall time after admission (sched.gpu_seconds).
Stop switch: artifacts/accumulator_sweep_mn_v1/STOP (no new job; waits for its own jobs).  Marker at the end:
logs/run2-p<N>[-<tag>].done.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.accumulator_sweep_mn import fast, plan, sched

r1 = sched.r1
LOGS = sched.LOGS
GPU_RUN = 'artifacts/agent_orchestration/gpu_run.sh'
FASTJOB_MIN_ITEMS_FOR_SECOND = 6


def archive_cmd(case, policy, start, stop):
    return [GPU_RUN, '--min-free-mib', '3000', '--wait', '7200', f'{r1.ARCHIVE}/run.sh', 'predict', case, policy, 'cuda',
            str(start), str(stop), '--batch', '8']


def fastjob_cmd(max_priority, cases='', engine='fast'):
    cmd = [GPU_RUN, '--min-free-mib', '3000', '--wait', '7200', '.venv/bin/python', '-m',
           'tools.run.accumulator_sweep_mn_fastjob', '--max-priority', str(max_priority), '--engine', engine]
    return cmd + (['--cases', cases] if cases else [])


# --- K3 -----------------------------------------------------------------------------------------------------------------
def k3_plan(spec):
    """[(policy, start, stop)] of the case's K3 anchors (empty until its location phase is complete)."""
    case = spec['case']
    if not sched.location_complete(spec):
        return []
    res = r1.uniform_location(spec)
    grid, _, _ = sched.grid_with_extensions(spec, {})
    f21x = next((p for p in spec.get('float_sensitivity', []) if p.startswith('f21.x')), None)
    if f21x and not fast.files(case, f21x, fast.FAST_ROOT):
        f21x = None                       # its location run was made on the archive: nothing fast to anchor
    s = fast.k3_start(case)
    return [(p, s, s + fast.K3_IMAGES) for p in fast.k3_policies(case, res, [g['policy'] for g in grid], f21x)]


def k3_record(case, policy):
    return fast.CHECKS / f'k3-{case}-{policy}.json'


def k3_step(cases, busy, compare=True):
    """Archive anchor calls still needed [(case, policy, a, b)]; compares (and records) anchors whose fast side exists."""
    need = []
    for spec in cases:
        case = spec['case']
        if fast.engine_for(case) != 'fast' or not fast.k2_passed(case):
            continue
        for pol, a, b in k3_plan(spec):
            if k3_record(case, pol).exists():
                continue
            if fast.covering(fast.files(case, pol, fast.ARCHIVE_ROOT), a, b) is None:
                if not any(k[:2] == (case, pol) for k in busy):
                    need.append((case, pol, a, b))
                continue
            if not compare or not fast.fast_covers(case, pol, a, b):
                continue                  # fast record not made yet (its 1k run comes later)
            r = fast.cross_root(case, pol, a, b)
            r.update(check='K3', time=time.strftime('%Y-%m-%d %H:%M:%S %z'), seed=fast.SEED_K3)
            k3_record(case, pol).write_text(json.dumps(r, indent=1) + '\n')
            print(time.strftime('%H:%M:%S'), 'K3', case, pol, a, b, r['status'], r['same'], r['differ'], flush=True)
            if r['status'] != 'pass':
                fast.halt(f'K3 failed for {case} {pol}', str(k3_record(case, pol).relative_to(fast.ROOT)))
    return need


class Jobs:
    def __init__(self, n, dry):
        self.n, self.dry, self.procs = n, dry, {}
        LOGS.mkdir(parents=True, exist_ok=True)
        self.ledger = None if dry else open(LOGS / 'ledger.jsonl', 'a')
        self.fastlog = LOGS / 'run2-fastjobs'
        self.failures = {}

    def own_pids(self):
        return {p.pid for p, *_ in self.procs.values() if p is not None}

    def fastjobs(self, engine='fast'):
        return [k for k in self.procs if k[0] == 'fastjob' and k[2] == engine]

    def external(self):
        return r1.external_running(self.own_pids())

    def full(self):
        return len(self.procs) + len(self.external()) >= self.n

    def launch_archive(self, case, policy, start, stop, why):
        key = (case, policy, start, stop)
        if self.dry:
            print('DRY archive', why, *key, flush=True); self.procs[key] = (None, time.time(), None); return
        folder = LOGS / case; folder.mkdir(exist_ok=True)
        log = open(folder / f'{policy}-{start:05d}-{stop:05d}.log', 'a')
        env = dict(os.environ, GPU_LANE='Q1', PYTHONDONTWRITEBYTECODE='1')
        proc = subprocess.Popen(archive_cmd(*key), cwd=fast.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        self.procs[key] = (proc, time.time(), log)
        print(time.strftime('%H:%M:%S'), 'START archive', why, *key, flush=True)

    def launch_fastjob(self, max_priority, cases, engine='fast'):
        self.count = getattr(self, 'count', 0) + 1
        key = ('fastjob', time.strftime('%H%M%S'), engine, self.count)
        if self.dry:
            print('DRY slot job', engine, flush=True); self.procs[key] = (None, time.time(), None); return
        self.fastlog.mkdir(exist_ok=True)
        log = open(self.fastlog / f"{time.strftime('%Y%m%d-%H%M%S')}-{engine}-{self.count}.log", 'a')
        env = dict(os.environ, GPU_LANE='Q1', PYTHONDONTWRITEBYTECODE='1')
        proc = subprocess.Popen(fastjob_cmd(max_priority, cases, engine), cwd=fast.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        self.procs[key] = (proc, time.time(), log)
        print(time.strftime('%H:%M:%S'), 'START slot job', engine, proc.pid, flush=True)

    def poll(self):
        for key, (proc, t0, log) in list(self.procs.items()):
            if proc is None:
                continue
            code = proc.poll()
            if code is None:
                continue
            log.close(); del self.procs[key]
            if key[0] == 'fastjob':
                print(time.strftime('%H:%M:%S'), 'END slot job', key[2], code, flush=True); continue
            if code:
                self.failures[key[:2]] = self.failures.get(key[:2], 0) + 1
            self.ledger.write(json.dumps({'case': key[0], 'policy': key[1], 'start': key[2], 'stop': key[3], 'exit': code,
                                          'launched': t0, 'ended': time.time(), 'runner': 'run2'}) + '\n')
            self.ledger.flush()
            print(time.strftime('%H:%M:%S'), 'END archive', code, *key, flush=True)

    def wait_all(self):
        while any(p is not None for p, *_ in self.procs.values()):
            self.poll(); time.sleep(5)


def step(cases, jobs, max_priority, cases_arg='', deferred=None):
    """One pass: queue, K3, launches.  Returns (queue, k3 needs, launched)."""
    deferred = set() if deferred is None else deferred
    busy = sched.busy_keys(jobs.own_pids()) + [k for k in jobs.procs if k[0] != 'fastjob']
    queue = sched.build_queue(cases, busy, max_priority)
    k3_need = k3_step(cases, busy) if not jobs.dry else []
    budget_left = r1.BUDGET_HOURS * 3600 - sched.gpu_seconds()
    ffail = sched.fast_failures()

    def live(q):
        if q['engine'] == 'fast':
            if ffail.get((q['case'], q['policy']), 0) > sched.RETRIES:
                return False
            if q['rank'] >= 3 and budget_left < sched.EST_FAST_CALL_S:
                if (q['case'], q['policy']) not in deferred:
                    print(time.strftime('%H:%M:%S'), 'BUDGET-DEFER', q['case'], q['policy'], flush=True)
                    deferred.add((q['case'], q['policy']))
                return False
            return True
        if jobs.failures.get((q['case'], q['policy']), 0) + ffail.get((q['case'], q['policy']), 0) > sched.RETRIES:
            return False
        rng = sched.archive_range(q)
        if rng is None:
            return False
        est = r1.EST_SETUP_S + r1.EST_S_PER_IMAGE * (rng[1] - rng[0])
        if q['rank'] >= 3 and budget_left < est:
            if (q['case'], q['policy']) not in deferred:
                print(time.strftime('%H:%M:%S'), 'BUDGET-DEFER', q['case'], q['policy'], flush=True)
                deferred.add((q['case'], q['policy']))
            return False
        return True
    queue = [q for q in queue if live(q)]
    items = [dict(q) for q in queue] + [{'rank': 1.6, 'i': -1, 'case': c, 'policy': p, 'start': a, 'stop': b,
                                         'engine': 'archive', 'kind': 'k3'} for c, p, a, b in k3_need]
    items.sort(key=lambda q: (q['rank'], q['i']))
    pending = {e: sum(q['engine'] == e for q in items) for e in ('fast', 'archive')}
    launched, used = 0, set()
    for it in items:
        e = it['engine']
        if jobs.full():
            break
        if e in used:
            continue
        n = len(jobs.fastjobs(e))
        if n == 0 or (n < 2 and pending[e] >= FASTJOB_MIN_ITEMS_FOR_SECOND):
            jobs.launch_fastjob(max_priority, cases_arg, e); launched += 1
        used.add(e)                         # at most one new slot job per engine and pass
    return queue, k3_need, launched


def run(args):
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['GPU_LANE'] = 'Q1'
    k1 = fast.k1(write=not args.dry_run, who='run2')
    print(time.strftime('%H:%M:%S'), 'K1', k1['status'], k1['manifest_files'], 'files', k1['versions'], flush=True)
    if fast.HALT.exists():
        sys.exit(f'{fast.HALT} exists: fast runs are halted (see the handoff)')
    p = r1.protocol()
    cases = p['cases']
    if args.cases:
        keep = set(args.cases.split(','))
        cases = [c for c in cases if c['case'] in keep]
    jobs = Jobs(args.jobs, args.dry_run)
    deferred = set()
    last_xroot = 0.0
    while True:
        jobs.poll()
        if fast.HALT.exists() or fast.STOP.exists():
            print(time.strftime('%H:%M:%S'), 'HALT' if fast.HALT.exists() else 'STOP', '- no new jobs', flush=True)
            break
        queue, k3_need, launched = step(cases, jobs, args.max_priority, args.cases, deferred)
        if args.dry_run:
            for spec in cases:
                if fast.engine_for(spec['case']) == 'fast' and fast.k2_passed(spec['case']):
                    print('K3-PLAN', spec['case'], k3_plan(spec))
            for q in queue[:60]:
                rng = (q['start'], q['stop']) if q['engine'] == 'fast' else sched.archive_range(q)
                print('QUEUE', q['rank'], q['engine'], q['kind'], q['case'], q['policy'], rng)
            print('queue', len(queue), 'fast', sum(q['engine'] == 'fast' for q in queue), 'k3', len(k3_need),
                  'gpu_job_hours %.2f' % (sched.gpu_seconds() / 3600))
            break
        if time.time() - last_xroot > 1800:
            xroot(cases); last_xroot = time.time()
        if not queue and not k3_need and not jobs.procs and not jobs.external() and not sched.claimed_keys():
            break
        time.sleep(15)
    jobs.wait_all()
    if not args.dry_run:
        xroot(cases)
        tag = f'-{args.tag}' if args.tag else ''
        (LOGS / f'run2-p{args.max_priority}{tag}.done').write_text(
            time.strftime('%Y-%m-%d %H:%M:%S') + '\n' + json.dumps({
                'budget_deferred': sorted(map(list, deferred)), 'halt': fast.HALT.exists(), 'stop': fast.STOP.exists(),
                'archive_failures': {' '.join(map(str, k)): v for k, v in jobs.failures.items()},
                'fast_failures': {' '.join(k): v for k, v in sched.fast_failures().items()},
                'gpu_job_hours': sched.gpu_seconds() / 3600}) + '\n')


def xroot(cases):
    """K3+: every (gated case, policy) with files in both roots: all overlapping records compared (any difference halts)."""
    out = {'check': 'cross-root overlap', 'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'pairs': []}
    for spec in cases:
        case = spec['case']
        fdir = fast.FAST_ROOT / case / 'predictions'
        if not fdir.exists():
            continue
        pols = sorted({m['policy'] for m in (r1_name(p.name) for p in fdir.glob('*.json')) if m})
        for pol in pols:
            if not fast.files(case, pol, fast.ARCHIVE_ROOT):
                continue
            r = fast.cross_root(case, pol, 0, 1000)
            out['pairs'].append({k: r[k] for k in ('case', 'policy', 'same', 'differ', 'nodes_equal', 'status', 'first_difference')})
            if r['status'] == 'fail':
                fast.halt(f'cross-root difference {case} {pol}', r)
    fast.CHECKS.mkdir(parents=True, exist_ok=True)
    path = fast.CHECKS / f"xroot-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1) + '\n')
    n = sum(p['same'] for p in out['pairs']); d = sum(p['differ'] for p in out['pairs'])
    print(time.strftime('%H:%M:%S'), 'XROOT', len(out['pairs']), 'pairs', n, 'same', d, 'differ', flush=True)
    return out


def r1_name(name):
    from tools.accumulator_sweep_v1.load import NAME
    return NAME.match(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--max-priority', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--cases', default='')
    ap.add_argument('--tag', default='')
    args = ap.parse_args(argv)
    if args.jobs > 2:
        sys.exit('at most 2 GPU jobs of this lane at a time')
    run(args)


if __name__ == '__main__':
    main()
