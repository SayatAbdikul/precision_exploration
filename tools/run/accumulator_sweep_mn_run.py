"""Resumable runner of the MobileNet accumulator sweep (lane Q1, protocol accumulator-sweep-mn-protocol-v1).

    .venv/bin/python -m tools.run.accumulator_sweep_mn_run [--jobs 2] [--max-priority 4] [--dry-run] [--tag T]

One loop over all phases, priority-major (protocol screen_grid): 1 = wide/control 1k; L = location phase (128 images:
uniform ladder + refinement, per-node d ladder, exponent sensitivity); 2 = sat.w 1k bracket (+ extension rules);
3 = rule fp16 and f21 1k; 4 = sat.struct-<d> 1k.  Priorities 3 and 4 are launched only while the lane's GPU job time
plus the running jobs' estimate stays below the protocol budget.

Every measurement is one call of the ARCHIVED engine through the GPU wrapper (GPU_LANE=Q1):
    artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 --wait 7200 $M/run.sh predict CASE POLICY cuda START STOP --batch 8
State is read from the sealed prediction files only (a covered range is never run again; a 1k run continues from the
end of the longest contiguous prefix from image 0).  Ledger and per-run logs: artifacts/accumulator_sweep_mn_v1/logs/;
marker `run-p<N>[-<tag>].done` at the end.  This process runs on the CPU (`.venv`) and only reads prediction files.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.accumulator_sweep_mn import certs, plan
from tools.accumulator_sweep_mn.load import assemble, prediction_files, covering, prefix_end
from tools.accumulator_sweep_v1.score import image_scores

ROOT = certs.ROOT
ARCHIVE = certs.ARCHIVE
LANE = ROOT / 'artifacts/accumulator_sweep_mn_v1'
LOGS = LANE / 'logs'
PROTOCOL = ROOT / 'public/experiments/configs/breadth-study/accumulator-sweep-mn-protocol-v1.json'
ENGINE_LEDGER = certs.RUN_ROOT / 'ledger'
RETRIES = 4
TILE_BOUNDS = (128, 564, 1000)  # addendum 2: a 1k run is made of up to three contiguous calls
BUDGET_HOURS = 14.0
EST_S_PER_IMAGE = 0.30
EST_SETUP_S = 35.0


def protocol():
    """The frozen protocol plus cases added by its addenda (accumulator-sweep-mn-protocol-v1-addendum-<n>.json)."""
    with open(PROTOCOL) as handle:
        p = json.load(handle)
    p['addenda'] = []
    for path in sorted(PROTOCOL.parent.glob('accumulator-sweep-mn-protocol-v1-addendum-*.json')):
        with open(path) as handle:
            a = json.load(handle)
        p['cases'] += a.get('cases_added', [])
        p['addenda'].append(a)
    return p


def protocol_epoch():
    return time.mktime(time.strptime(protocol()['written'][:19], '%Y-%m-%d %H:%M:%S'))


_MEMO = {}


def summary(case, policy, start, stop):
    """Events, expected-credit Top-1 (percent) and failures of a sealed range, or None if not covered."""
    tiles = covering(prediction_files(case, policy), start, stop)
    if tiles is None:
        return None
    key = (case, policy, start, stop, tuple(str(t[2]) for t in tiles))
    if key not in _MEMO:
        s = image_scores(assemble(case, policy, start, stop)['images'])
        _MEMO[key] = {'events': int(s['event'].sum()), 'expected_percent': 100 * float(s['expected'].mean()),
                      'failed': int(s['failed'].sum())}
    return _MEMO[key]


def lane_runs():
    """(case, policy, start, stop) of every call this lane's runners launched (ledger and per-run log names, so that
    calls of a runner that was stopped before writing its ledger line are included)."""
    out = set()
    path = LOGS / 'ledger.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out.add((r['case'], r['policy'], r['start'], r['stop']))
    for case in certs.CASES:
        for log in (LOGS / case).glob('*.log') if (LOGS / case).exists() else []:
            pol, a, b = log.stem.rsplit('-', 2)
            out.add((case, pol, int(a), int(b)))
    return out


def tile_stop(start, stop):
    """End of the next call of a run on [start, stop): the first tile boundary after start (addendum 2)."""
    return min([t for t in TILE_BOUNDS if start < t <= stop] or [stop])


def gpu_job_seconds(since=None):
    """Sum of the engine ledger wall_seconds of this lane's predict calls (matched by arguments to the runner's
    ledger, plus the step-1 smoke run) started after the protocol was written."""
    since = protocol_epoch() if since is None else since
    mine = lane_runs()
    total = 0.0
    for path in ENGINE_LEDGER.glob('*.json'):
        try:
            p = json.load(open(path))['payload']
        except (OSError, ValueError):
            continue
        if p.get('label') != 'predict' or p.get('started_epoch', 0) < since:
            continue
        a = p['arguments']
        if (a[0], a[1], a[3], a[4]) in mine:
            total += p['wall_seconds']
    return total


def wrapped_predict(argv):
    """(case, policy, start, stop) when argv is `bash .../gpu_run.sh [options] <ARCHIVE>/run.sh predict CASE POLICY BACKEND START STOP ...`."""
    if len(argv) < 2 or not argv[1].endswith('gpu_run.sh'):
        return None
    for i, a in enumerate(argv):
        if a == f'{ARCHIVE}/run.sh' and i + 6 < len(argv) and argv[i + 1] == 'predict':
            return argv[i + 2], argv[i + 3], int(argv[i + 5]), int(argv[i + 6])
    return None


def external_running(own_pids=()):
    """Archived-engine predict wrappers of THIS lane (environment GPU_LANE=Q1) started by another process (an earlier
    runner).  They count towards --jobs and their (case, policy) is not launched again while they run."""
    out = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) in own_pids:
            continue
        try:
            argv = (Path('/proc') / entry / 'cmdline').read_bytes().split(b'\0')
            env = (Path('/proc') / entry / 'environ').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'GPU_LANE=Q1' not in env:
            continue
        hit = wrapped_predict([a.decode(errors='replace') for a in argv if a])
        if hit is not None:
            out.append(hit)
    return out


class Pool:
    def __init__(self, jobs, dry):
        self.jobs, self.dry, self.running, self.failures = jobs, dry, {}, {}
        self.dry_launched = []
        LOGS.mkdir(parents=True, exist_ok=True)
        self.ledger = open(LOGS / 'ledger.jsonl', 'a') if not dry else None

    def external(self):
        return external_running({p.pid for p, _, _ in self.running.values()})

    def busy(self, case, policy):
        return any(k[:2] == (case, policy) for k in self.running) or \
            any(k[:2] == (case, policy) for k in self.external()) or \
            any(k[:2] == (case, policy) for k in self.dry_launched)

    def busy_family(self, case, prefix):
        keys = list(self.running) + self.external() + self.dry_launched
        return any(k[0] == case and k[1].startswith(prefix) for k in keys)

    def failed(self, case, policy, stop=None):
        return self.failures.get((case, policy), 0) > RETRIES  # failures of any tile of (case, policy)

    def full(self):
        return len(self.running) + len(self.external()) + len(self.dry_launched) >= self.jobs

    def running_estimate_s(self):
        return sum(EST_SETUP_S + EST_S_PER_IMAGE * (k[3] - k[2]) for k in list(self.running) + self.dry_launched)

    def launch(self, case, policy, start, stop):
        start = max(start, prefix_end(case, policy, stop))
        if start >= stop:
            return False
        stop = tile_stop(start, stop)
        key = (case, policy, start, stop)
        if self.dry:
            print('DRY', *key, flush=True); self.dry_launched.append(key); return True
        folder = LOGS / case; folder.mkdir(exist_ok=True)
        log = open(folder / f'{policy}-{start:05d}-{stop:05d}.log', 'a')
        cmd = ['artifacts/agent_orchestration/gpu_run.sh', '--min-free-mib', '3000', '--wait', '7200',
               f'{ARCHIVE}/run.sh', 'predict', case, policy, 'cuda', str(start), str(stop), '--batch', '8']
        env = dict(os.environ, GPU_LANE='Q1')
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        self.running[key] = (proc, time.time(), log)
        print(time.strftime('%H:%M:%S'), 'START', *key, flush=True)
        return True

    def poll(self):
        for key, (proc, t0, log) in list(self.running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close(); del self.running[key]
            if code:
                fk = (key[0], key[1])
                self.failures[fk] = self.failures.get(fk, 0) + 1
            self.ledger.write(json.dumps({'case': key[0], 'policy': key[1], 'start': key[2], 'stop': key[3],
                                          'exit': code, 'launched': t0, 'ended': time.time()}) + '\n')
            self.ledger.flush()
            print(time.strftime('%H:%M:%S'), 'END', code, *key, flush=True)

    def wait_all(self):
        while self.running:
            self.poll(); time.sleep(5)


def measured(case, policies, a, b):
    return {k: r for k, pol in policies.items() if (r := summary(case, pol, a, b)) is not None}


def uniform_location(spec):
    a, b = plan.LOCATION
    return measured(spec['case'], {w: f'sat.w{w}' for w in range(plan.MIN_WIDTH, spec['W_cert_abs'])}, a, b)


def struct_location(spec):
    a, b = plan.LOCATION
    return measured(spec['case'], {d: f'sat.struct-{d}' for d in plan.STRUCT_LADDER}, a, b)


def uniform_location_complete(spec, wide):
    res = uniform_location(spec)
    if not res:
        return False
    rungs = spec['location_ladder']
    if not (plan.ladder_done(res, wide) or all(w in res for w in rungs)):
        return False
    return not plan.refine(res, wide, spec['W_cert_abs'])


def struct_location_complete(spec, wide):
    return plan.struct_next(struct_location(spec), wide) is None


def location_wants(spec, wide, pool):
    """Location runs (policy, start, stop) of one case that may be launched now, in launch order."""
    a, b = plan.LOCATION
    case = spec['case']
    out = [(p, a, b) for p in spec.get('float_sensitivity', []) if summary(case, p, a, b) is None]
    res = uniform_location(spec)
    rungs = spec['location_ladder']
    if not (plan.ladder_done(res, wide) or all(w in res for w in rungs)):
        nxt = [w for w in rungs if w not in res and not pool.busy(case, f'sat.w{w}')]
        if nxt:
            out.append((f'sat.w{nxt[0]}', a, b))  # rungs in descending order
    elif not pool.busy_family(case, 'sat.w'):
        out += [(f'sat.w{w}', a, b) for w in plan.refine(res, wide, spec['W_cert_abs'])]
    if not spec.get('stress_case') and not pool.busy_family(case, 'sat.struct-'):
        d = plan.struct_next(struct_location(spec), wide)
        if d is not None:
            out.append((f'sat.struct-{d}', a, b))
    return out


def grid_wants(spec, wide1k_known, extra):
    """1k runs (priority, policy) of one case once its location phase is complete, with extension widths."""
    a, b = plan.LOCATION
    case = spec['case']
    wide128 = summary(case, 'wide', a, b)['expected_percent']
    res = uniform_location(spec)
    bottom, top = plan.bracket(res, wide128, spec['W_cert_abs'])
    sres = struct_location(spec) if not spec.get('stress_case') else {}
    ds = plan.struct_set(sres, wide128) if sres else []
    grid = plan.screen_grid(case, bottom, top, spec['fp16_policy'], spec['f21_policy'], ds, spec.get('stress_case', False))
    grid += [{'case': case, 'policy': f'sat.w{w}', 'priority': 2} for w in extra.get(case, [])]
    return grid, bottom, top


def apply_extensions(spec, bottom, top, extra):
    """Protocol extension rules on complete 1k bracket results; returns True if a width was added."""
    a, b = plan.SCREEN
    case = spec['case']
    if spec.get('stress_case'):
        return False
    widths = sorted({*range(bottom, top + 1), *extra.get(case, [])})
    res = {w: summary(case, f'sat.w{w}', a, b) for w in widths}
    wide = summary(case, 'wide', a, b)
    if wide is None or any(r is None for r in res.values()):
        return False
    added = False
    hi = max(res)
    up = plan.extend_top(hi, res[hi]['events'] > 0, spec['W_cert_abs'])
    if up is not None and up not in extra.setdefault(case, []):
        extra[case].append(up); added = True
    down = plan.extend_bottom(res, wide['expected_percent'])
    if down is not None and down not in extra[case]:
        extra[case].append(down); added = True
    return added


def run(args):
    p = protocol(); pool = Pool(args.jobs, args.dry_run)
    cases = p['cases']
    if args.cases:
        keep = set(args.cases.split(','))
        cases = [c for c in cases if c['case'] in keep]
    a128, b128 = plan.LOCATION
    a, b = plan.SCREEN
    extra = {}
    deferred = set()
    while True:
        pool.poll()
        queue = []  # (rank, case index, policy, start, stop)
        for i, spec in enumerate(cases):
            case = spec['case']
            for pol in ('wide', 'control'):
                if prefix_end(case, pol, b) < b:
                    queue.append((1.0, i, pol, 0, b))
            wide = summary(case, 'wide', a128, b128)
            if wide is None:
                continue
            w128 = wide['expected_percent']
            for pol, s, t in location_wants(spec, w128, pool):
                queue.append((1.5, i, pol, s, t))
            uni_done = uniform_location_complete(spec, w128)
            str_done = spec.get('stress_case') or struct_location_complete(spec, w128)
            if not (uni_done and str_done):
                continue
            grid, bottom, top = grid_wants(spec, None, extra)
            if apply_extensions(spec, bottom, top, extra):
                grid, bottom, top = grid_wants(spec, None, extra)
            for g in grid:
                if g['priority'] > args.max_priority or g['policy'] in ('wide', 'control'):
                    continue
                if prefix_end(case, g['policy'], b) < b:
                    queue.append((float(g['priority']), i, g['policy'], 0, b))
        queue.sort()
        budget_left = BUDGET_HOURS * 3600 - (gpu_job_seconds() + pool.running_estimate_s())
        launched = 0; waiting = 0
        for rank, i, pol, s, t in queue:
            case = cases[i]['case']
            if pool.failed(case, pol, t):
                continue
            if pool.busy(case, pol):
                continue
            if rank >= 3 and budget_left < EST_SETUP_S + EST_S_PER_IMAGE * (t - s):
                if (case, pol) not in deferred:
                    print(time.strftime('%H:%M:%S'), 'BUDGET-DEFER', case, pol, flush=True)
                    deferred.add((case, pol))
                continue
            waiting += 1
            if pool.full():
                continue
            if pool.launch(case, pol, s, t):
                launched += 1
                budget_left -= EST_SETUP_S + EST_S_PER_IMAGE * (t - s)
        if args.dry_run:
            print('queue', len(queue), 'deferred', len(deferred), 'gpu_job_hours %.2f' % (gpu_job_seconds() / 3600))
            break
        if not launched and not waiting and not pool.running and not pool.external():
            break
        time.sleep(10)
    pool.wait_all()
    if not args.dry_run:
        tag = f'-{args.tag}' if args.tag else ''
        (LOGS / f'run-p{args.max_priority}{tag}.done').write_text(
            time.strftime('%Y-%m-%d %H:%M:%S') + '\n' + json.dumps({'extensions': extra, 'budget_deferred': sorted(map(list, deferred)),
                                                                    'failures': {' '.join(map(str, k)): v for k, v in pool.failures.items()},
                                                                    'gpu_job_hours': gpu_job_seconds() / 3600}) + '\n')


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
