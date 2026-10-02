"""Resumable runner of the accumulator sweep (lane L8, protocol accumulator-sweep-protocol-v1).

    .venv/bin/python -m tools.run.accumulator_sweep_run location [--jobs 3] [--dry-run]
    .venv/bin/python -m tools.run.accumulator_sweep_run screen [--jobs 3] [--max-priority N] [--dry-run]

Every measurement is one call of the ARCHIVED engine through the GPU wrapper:
    artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 --wait 7200 $A/run.sh predict CASE POLICY cuda START STOP --batch 8
State is read from the sealed prediction files (a range that is already covered is never run again; a 1k run
continues from the end of the longest contiguous prefix that exists).  Logs and a JSON-lines ledger go to
artifacts/accumulator_sweep_v1/logs/; a marker `<phase>.done` is written at the end.
This process itself runs on the CPU (`.venv`) and only reads prediction files.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.accumulator_sweep_v1 import certs, plan
from tools.accumulator_sweep_v1.load import assemble, prediction_files, covering
from tools.accumulator_sweep_v1.score import image_scores

ROOT = certs.ROOT
ARCHIVE = f'artifacts/scaled_bridge_v2/implementations/{certs.DIGEST}'
LOGS = ROOT / 'artifacts/accumulator_sweep_v1/logs'
PROTOCOL = ROOT / 'public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json'
RETRIES = 2


def protocol():
    """The frozen protocol plus the cases added by its addenda (accumulator-sweep-protocol-v1-addendum-<n>.json)."""
    with open(PROTOCOL) as handle:
        p = json.load(handle)
    for path in sorted(PROTOCOL.parent.glob('accumulator-sweep-protocol-v1-addendum-*.json')):
        with open(path) as handle:
            p['cases'] += json.load(handle).get('cases_added', [])
    return p


def prefix_end(case, policy, stop):
    """End of the longest contiguous run of sealed files from image 0 (at most stop)."""
    files = prediction_files(case, policy)
    best = 0
    for _, b, _ in files:
        if b <= stop and b > best and covering(files, 0, b) is not None:
            best = b
    return best


_MEMO = {}


def summary(case, policy, start, stop):
    tiles = covering(prediction_files(case, policy), start, stop)
    if tiles is None:
        return None
    key = (case, policy, start, stop, tuple(str(t[2]) for t in tiles))
    if key in _MEMO:
        return _MEMO[key]
    a = assemble(case, policy, start, stop)
    s = image_scores(a['images'])
    _MEMO[key] = {'events': int(s['event'].sum()), 'expected_percent': 100 * float(s['expected'].mean()),
            'failed': int(s['failed'].sum()), 'top1': s['top1']}
    return _MEMO[key]


def external_running(own_pids=()):
    """(case, policy) of archived-engine predict calls of this lane that some OTHER process started (an earlier
    runner, a location runner): the gpu_run.sh wrappers whose command is `<ARCHIVE>/run.sh predict CASE POLICY ...`.
    Added in round 2 so that a runner can be restarted while an earlier runner's jobs finish; they count towards the
    lane's job limit and their (case, policy) is never launched twice."""
    out = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) in own_pids:
            continue
        try:
            argv = (Path('/proc') / entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        hit = wrapped_predict([a.decode(errors='replace') for a in argv if a])
        if hit is not None:
            out.append(hit)
    return out


def wrapped_predict(argv):
    """(case, policy) when argv is `bash .../gpu_run.sh [options] <ARCHIVE>/run.sh predict CASE POLICY ...`."""
    if len(argv) < 2 or not argv[1].endswith('gpu_run.sh'):
        return None
    for i, a in enumerate(argv):
        if a == f'{ARCHIVE}/run.sh' and i + 3 < len(argv) and argv[i + 1] == 'predict':
            return argv[i + 2], argv[i + 3]
    return None


class Pool:
    def __init__(self, jobs, dry):
        self.jobs, self.dry, self.running, self.failures = jobs, dry, {}, {}
        self.yield_to = None
        LOGS.mkdir(parents=True, exist_ok=True)
        self.ledger = open(LOGS / 'ledger.jsonl', 'a')

    def external(self):
        return external_running({p.pid for p, _, _ in self.running.values()})

    def busy(self, key):
        return any(k[:2] == key[:2] for k in self.running) or tuple(key[:2]) in self.external()

    def failed(self, key):
        return self.failures.get((key[0], key[1], key[-1]), 0) > RETRIES

    def full(self):
        n = len(self.external())
        if self.yield_to and Path(f'/proc/{self.yield_to[0]}').exists():
            n = max(n, self.yield_to[1])  # slots kept for a runner that does not see this one's jobs
        return len(self.running) + n >= self.jobs

    def launch(self, case, policy, start, stop):
        start = max(start, prefix_end(case, policy, stop))  # never re-run images that are already sealed
        key = (case, policy, start, stop)
        if self.dry:
            print('DRY', *key, flush=True); return
        if self.failed(key):
            return
        folder = LOGS / case; folder.mkdir(exist_ok=True)
        log = open(folder / f'{policy}-{start:05d}-{stop:05d}.log', 'a')
        cmd = ['artifacts/agent_orchestration/gpu_run.sh', '--min-free-mib', '3000', '--wait', '7200',
               f'{ARCHIVE}/run.sh', 'predict', case, policy, 'cuda', str(start), str(stop), '--batch', '8']
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.running[key] = (proc, time.time(), log)
        print(time.strftime('%H:%M:%S'), 'START', *key, flush=True)

    def poll(self):
        done = []
        for key, (proc, t0, log) in list(self.running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close(); del self.running[key]
            if code:
                fk = (key[0], key[1], key[3])
                self.failures[fk] = self.failures.get(fk, 0) + 1
            self.ledger.write(json.dumps({'case': key[0], 'policy': key[1], 'start': key[2], 'stop': key[3],
                                          'exit': code, 'launched': t0, 'ended': time.time()}) + '\n')
            self.ledger.flush()
            print(time.strftime('%H:%M:%S'), 'END', code, *key, flush=True)
            done.append(key)
        return done

    def wait_all(self):
        while self.running:
            self.poll(); time.sleep(5)


def location_jobs(case, spec):
    """Static location jobs of one case (protocol item location)."""
    a, b = plan.LOCATION
    jobs = [('wide', a, b)]
    if spec['recipe'] == 'b2':
        jobs += [(spec['fp16_policy'], a, b), (spec['f21_policy'], a, b)]
    if spec.get('float_sensitivity'):
        e = certs.float_exponent('fp16', certs.node_table(certs.load_certificate(case)))
        for k in plan.FP16_SENSITIVITY:
            if e + k != 0:
                jobs.append((f'fp16.x{e + k}', a, b))
        if 15 - spec['ub'] != 0:
            jobs.append((f'f21.x{15 - spec["ub"]}', a, b))
    return jobs


def gated(spec):
    """A B2 case enters only with its MB2 gate record and one policy gate per family in the archived run root."""
    if spec['recipe'] != 'b2':
        return True
    root = certs.RUN_ROOT / spec['case']
    return (root / 'gate.json').exists() and len(list(root.glob('gate-*.json'))) >= 4


def sweep_cases(p):
    return [c for c in p['cases'] if c['enters_sweep'] and gated(c)]


def run_location(args):
    p = protocol(); pool = Pool(args.jobs, args.dry_run)
    cases = sweep_cases(p)
    a, b = plan.LOCATION
    while True:
        pool.poll(); pending = False
        for spec in cases:
            case = spec['case']
            for policy, s, t in location_jobs(case, spec):
                if covering(prediction_files(case, policy), s, t) is None and not pool.failed((case, policy, s, t)):
                    pending = True
                    if not pool.full() and not pool.busy((case, policy, s, t)) and not pool.failed((case, policy, s, t)):
                        pool.launch(case, policy, s, t)
            wide = summary(case, 'wide', a, b)
            results = measured(case, spec['W_cert_abs'], a, b)
            rungs = plan.ladder(spec['W_cert_abs'])
            done = wide is not None and (plan.ladder_done(results, wide['expected_percent'])
                                         or all(w in results for w in rungs))
            if not done:
                for w in rungs:
                    key = (case, f'sat.w{w}', a, b)
                    if w in results or pool.failed(key):
                        continue
                    pending = True
                    if pool.busy(key):
                        continue
                    if not pool.full() and not pool.failed(key):
                        pool.launch(*key)
                    break  # rungs are launched in descending order, one new rung per case per pass
                continue
            if any(k[0] == case and k[1].startswith('sat.w') for k in pool.running):
                pending = True
                continue  # refinement waits for every started rung
            for w in plan.refine(results, wide['expected_percent'], spec['W_cert_abs']):
                key = (case, f'sat.w{w}', a, b)
                if pool.failed(key):
                    continue
                pending = True
                if not pool.full() and not pool.busy(key) and not pool.failed(key):
                    pool.launch(*key)
        if args.dry_run:
            break
        if not pending and not pool.running:
            break
        time.sleep(5)
    pool.wait_all()
    if not args.dry_run:
        (LOGS / 'location.done').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')


def measured(case, w_cert_abs, a, b):
    return {w: r for w in range(plan.MIN_WIDTH, w_cert_abs) if (r := summary(case, f'sat.w{w}', a, b)) is not None}


def location_outcome(spec):
    a, b = plan.LOCATION
    case = spec['case']
    wide = summary(case, 'wide', a, b)
    results = measured(case, spec['W_cert_abs'], a, b)
    bottom, top = plan.bracket(results, wide['expected_percent'], spec['W_cert_abs'])
    return wide, results, bottom, top


def location_complete(spec):
    """True when the location phase of this case is finished by its own rule (ladder stopped, every refinement
    width measured); the bracket is only defined then (added in round 2: a screen runner started while a location
    runner works on other cases must skip them)."""
    a, b = plan.LOCATION
    case = spec['case']
    wide = summary(case, 'wide', a, b)
    if wide is None:
        return False
    results = measured(case, spec['W_cert_abs'], a, b)
    rungs = plan.ladder(spec['W_cert_abs'])
    if not results or not (plan.ladder_done(results, wide['expected_percent']) or all(w in results for w in rungs)):
        return False
    return not plan.refine(results, wide['expected_percent'], spec['W_cert_abs'])


def screen_runs(spec):
    case = spec['case']
    _, _, bottom, top = location_outcome(spec)
    grid = plan.screen_grid(case, spec['W_cert_abs'], bottom, top, spec['integer'], spec['fp16_policy'], spec['f21_policy'],
                            tuple(spec.get('struct_d_primary', plan.STRUCT_D_PRIMARY)),
                            tuple(spec.get('struct_d_secondary', plan.STRUCT_D_SECONDARY)))
    if spec['recipe'] == 'b1':
        grid = [dict(g, priority=5) for g in grid if g['priority'] == 1]
    return grid, bottom, top


def run_screen(args):
    p = protocol(); pool = Pool(args.jobs, args.dry_run)
    if args.yield_to:
        pid, n = args.yield_to.split(':'); pool.yield_to = (int(pid), int(n))
    cases = sweep_cases(p)
    if args.cases:
        keep = set(args.cases.split(','))
        cases = [c for c in cases if c['case'] in keep]
    a, b = plan.SCREEN
    extra = {c['case']: [] for c in cases}
    while True:
        pool.poll(); pending = False; queue = []
        for spec in cases:
            case = spec['case']
            if not location_complete(spec):
                if not (LOGS / 'location.done').exists():
                    pending = True  # a location runner is still working on this case
                continue
            try:
                grid, bottom, top = screen_runs(spec)
            except (TypeError, KeyError, IndexError):
                continue  # location phase of this case not complete
            grid += [{'case': case, 'policy': f'sat.w{w}', 'priority': 1 if spec['recipe'] == 'b2' else 5} for w in extra[case]]
            for g in grid:
                if g['priority'] > args.max_priority:
                    continue
                start = prefix_end(case, g['policy'], b)
                if start < b:
                    queue.append((g['priority'], case, g['policy'], start, b))
            # extension rules (protocol): evaluated only on complete 1k results
            widths = sorted({top, bottom, *range(bottom, top + 1), *extra[case]})
            res = {w: summary(case, f'sat.w{w}', a, b) for w in widths}
            if all(r is not None for r in res.values()) and summary(case, 'wide', a, b) is not None:
                wide = summary(case, 'wide', a, b)
                hi = max(res); up = plan.extend_top(hi, res[hi]['events'] > 0, spec['W_cert_abs'])
                if up is not None and up not in extra[case]:
                    extra[case].append(up); pending = True
                down = plan.extend_bottom(res, wide['expected_percent'])
                if down is not None and down not in extra[case]:
                    extra[case].append(down); pending = True
        queue.sort()
        for pr, case, policy, s, t in queue:
            pending = True
            key = (case, policy, s, t)
            if pool.full():
                break
            if not pool.busy(key) and not pool.failed(key):
                pool.launch(*key)
        if args.dry_run:
            break
        if not pending and not pool.running:
            break
        if all(pool.failed((c, po, s, t)) for _, c, po, s, t in queue) and not pool.running:
            break
        time.sleep(10)
    pool.wait_all()
    if not args.dry_run:
        tag = f'-{args.tag}' if args.tag else ''
        (LOGS / f'screen-p{args.max_priority}{tag}.done').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n'
                                                                 + json.dumps(extra) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['location', 'screen'])
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--max-priority', type=int, default=6)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--cases', default='', help='comma-separated case names (default: every gated case)')
    ap.add_argument('--yield-to', default='', help='PID:N - while process PID lives, keep N of --jobs for it')
    ap.add_argument('--tag', default='', help='suffix of the completion marker screen-p<N>-<tag>.done')
    args = ap.parse_args(argv)
    if args.jobs > 3:
        sys.exit('at most 3 GPU jobs of this lane at a time')
    (run_location if args.phase == 'location' else run_screen)(args)


if __name__ == '__main__':
    main()
