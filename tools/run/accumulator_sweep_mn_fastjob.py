"""One GPU job of lane Q1 on lane S1's fast exact-engine path (protocol accumulator-sweep-mn-protocol-v1, addendum 3).

Started by tools/run/accumulator_sweep_mn_run2.py inside gpu_run.sh (GPU_LANE=Q1, --min-free-mib 3000), i.e. after
the GPU slot is granted.  While it holds the slot it takes the next fast run of the protocol's queue
(tools/accumulator_sweep_mn/sched.py: gated cases only; K2 calls first), claims it, runs

    artifacts/speed_v1/implementations/<edfecd5c...>/run.sh predict CASE POLICY cuda START STOP --batch 8

(PYTHONDONTWRITEBYTECODE=1), and rebuilds the queue, so that a location ladder advances by one rung per call
instead of one rung per GPU-slot wait.  It starts no new call after --limit seconds (default 720), when the queue
has had nothing for it for 60 s, or when artifacts/accumulator_sweep_mn_v1/STOP or FAST-HALT exists.  Each call
is one process (the fast CLI takes one case and policy per call).  Ledger: artifacts/accumulator_sweep_mn_v1/logs/
fast-jobs.jsonl (begin, call, end records; the lane's fast GPU seconds are the job wall times, fast.fast_job_seconds).
"""
import argparse
import json
import os
import subprocess
import sys
import time

from tools.accumulator_sweep_mn import fast, sched

IDLE_S = 60
CALL_TIMEOUT_S = 900


def record(rec):
    fast.FAST_JOBS.parent.mkdir(parents=True, exist_ok=True)
    with open(fast.FAST_JOBS, 'a') as h:
        h.write(json.dumps(rec) + '\n')


def pending_k2(cases):
    """Run check K2 for every gated case whose fast wide and control 0-1000 files both exist and have no K2 record."""
    for spec in cases:
        case = spec['case']
        if fast.gated(case) and not fast.k2_record(case).exists() and \
                all(fast.fast_covers(case, p, 0, 1000) for p in ('wide', 'control')):
            r = fast.k2(case)
            print(time.strftime('%H:%M:%S'), 'K2', case, r['status'],
                  {p: (v['same'], v['differ']) for p, v in r['policies'].items()}, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-priority', type=int, default=4)
    ap.add_argument('--limit', type=float, default=720.0)
    ap.add_argument('--cases', default='')
    ap.add_argument('--budget-hours', type=float, default=sched.r1.BUDGET_HOURS)
    ap.add_argument('--engine', choices=('fast', 'archive'), default='fast')
    args = ap.parse_args(argv)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    job = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    t0 = time.time()
    record({'type': 'begin', 'job': job, 'pid': os.getpid(), 'time': t0, 'argv': sys.argv[1:], 'engine': args.engine})
    ok, bad, n = fast.manifest_check()
    if not ok:
        record({'type': 'end', 'job': job, 'time': time.time(), 'reason': 'manifest mismatch', 'mismatches': bad})
        sys.exit('fast archive manifest mismatch: ' + json.dumps(bad))
    cases = sched.r1.protocol()['cases']
    if args.cases:
        keep = set(args.cases.split(','))
        cases = [c for c in cases if c['case'] in keep]
    tried, idle_since, reason, calls = set(), None, 'limit', 0
    while True:
        if fast.STOP.exists() or fast.HALT.exists():
            reason = 'STOP' if fast.STOP.exists() else 'FAST-HALT'; break
        if time.time() - t0 > args.limit:
            reason = 'limit'; break
        pending_k2(cases)
        if fast.HALT.exists():
            reason = 'FAST-HALT'; break
        failures = sched.fast_failures()
        budget_left = args.budget_hours * 3600 - sched.gpu_seconds()
        chosen = None
        busy = sched.busy_keys()
        queue = sched.build_queue(cases, busy, args.max_priority)
        if args.engine == 'archive':          # addendum 4: K3 anchors and ungated cases' runs inside one archive slot job
            from tools.run.accumulator_sweep_mn_run2 import k3_step
            queue += [{'rank': 1.6, 'i': -1, 'case': c, 'policy': p, 'start': a, 'stop': b, 'engine': 'archive', 'kind': 'k3'}
                      for c, p, a, b in k3_step(cases, busy, compare=False)]
            queue.sort(key=lambda q: (q['rank'], q['i']))
        for it in queue:
            if it['engine'] != args.engine or failures.get((it['case'], it['policy']), 0) > sched.RETRIES:
                continue
            if args.engine == 'archive' and it['kind'] != 'k3':
                rng = sched.archive_range(it)
                if rng is None:
                    continue
                it = dict(it, start=rng[0], stop=rng[1])
            key = (it['case'], it['policy'], it['start'], it['stop'])
            if key in tried:
                continue
            est = sched.EST_FAST_CALL_S if args.engine == 'fast' else \
                sched.r1.EST_SETUP_S + sched.r1.EST_S_PER_IMAGE * (key[3] - key[2])
            if it['rank'] >= 3 and budget_left < est:
                continue
            path = sched.claim(*key)
            if path is not None:
                chosen = (it, key, path); break
        if chosen is None:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > IDLE_S:
                reason = 'idle'; break
            time.sleep(10); continue
        idle_since = None
        it, key, path = chosen
        engine_dir = fast.FAST_ARCHIVE if args.engine == 'fast' else sched.r1.ARCHIVE
        cmd = [f'{engine_dir}/run.sh', 'predict', key[0], key[1], 'cuda', str(key[2]), str(key[3]), '--batch', '8']
        # archive calls inside a slot job log under logs/jobarchive/ (not logs/<case>/), so that the first runner's
        # ledger rule does not count them a second time: the job's wall time already counts them
        logdir = sched.LOGS / ('fast' if args.engine == 'fast' else 'jobarchive') / key[0]; logdir.mkdir(parents=True, exist_ok=True)
        log = logdir / f'{key[1]}-{key[2]:05d}-{key[3]:05d}.log'
        started = time.time()
        print(time.strftime('%H:%M:%S'), 'CALL', *key, it['kind'], 'rank', it['rank'], flush=True)
        try:
            with open(log, 'a') as h:
                code = subprocess.run(cmd, cwd=fast.ROOT, stdout=h, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), timeout=CALL_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            path.unlink(missing_ok=True)
        calls += 1
        if code:
            tried.add(key)       # not again in this job; a later job retries (a failed call leaves no file)
        record({'type': 'call', 'job': job, 'time': started, 'ended': time.time(), 'case': key[0], 'policy': key[1],
                'start': key[2], 'stop': key[3], 'kind': it['kind'], 'rank': it['rank'], 'exit': code, 'engine': args.engine,
                'log': str(log.relative_to(fast.ROOT))})
        print(time.strftime('%H:%M:%S'), 'END', code, *key, '%.1fs' % (time.time() - started), flush=True)
        if code:
            time.sleep(20)       # out-of-memory under contention: give the GPU a moment
    pending_k2(cases)
    record({'type': 'end', 'job': job, 'time': time.time(), 'reason': reason, 'calls': calls})
    print(time.strftime('%H:%M:%S'), 'JOB-END', reason, calls, flush=True)


if __name__ == '__main__':
    main()
