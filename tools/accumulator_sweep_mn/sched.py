"""Work queue of the MobileNet sweep over the two engines (protocol addendum 3; used by tools/run/accumulator_sweep_mn_run2.py
and tools/run/accumulator_sweep_mn_fastjob.py).

The rules are those of the frozen protocol, unchanged: this module imports plan.py and the decision functions of the
first runner (tools/run/accumulator_sweep_mn_run.py) and, inside this process only, points their `summary` at the
two-root loader of fast.py.  What is new is the routing: a case with a passing fast gate goes to the fast run.sh once
its check K2 has passed; a case without one stays on the 1f75c923 archive (with the archive's 128/564/1000 tiles of
addendum 2).  A fast run is always one call over its whole range (0-128 location tile, 0-1000 for a 1k run).
"""
import json
import os
import time
from pathlib import Path

import tools.run.accumulator_sweep_mn_run as r1
from tools.accumulator_sweep_v1.score import image_scores

from . import fast, plan

LOGS = r1.LOGS
CLAIMS = fast.LANE / 'claims'
RETRIES = 4                     # a (case, policy) is skipped after 5 failed calls (addendum 2), counted per engine ledger
EST_FAST_CALL_S = 40.0          # budget estimate of one fast call (run.sh root check + setup + up to 1,000 images)

_MEMO = {}


def summary(case, policy, start, stop):
    """Events, expected-credit Top-1 (percent) and failures of a covered range (both roots), or None."""
    tiles = fast.choose(case, policy, start, stop)
    if tiles is None:
        return None
    key = (case, policy, start, stop, tuple(str(t[2]) for t in tiles))
    if key not in _MEMO:
        s = image_scores(fast.assemble(case, policy, start, stop)['images'])
        _MEMO[key] = {'events': int(s['event'].sum()), 'expected_percent': 100 * float(s['expected'].mean()),
                      'failed': int(s['failed'].sum())}
    return _MEMO[key]


def install():
    """Point the first runner's decision functions at the two-root loader (this process only)."""
    r1.summary = summary
    r1.prefix_end = fast.prefix_end


install()


class BusyView:
    """busy()/busy_family() over a list of (case, policy, start, stop) keys, as the first runner's Pool offers them."""
    def __init__(self, keys):
        self.keys = list(keys)

    def busy(self, case, policy):
        return any(k[0] == case and k[1] == policy for k in self.keys)

    def busy_family(self, case, prefix):
        return any(k[0] == case and k[1].startswith(prefix) for k in self.keys)


# --- claims of in-flight fast calls ---------------------------------------------------------------------------------
def claim_name(case, policy, start, stop):
    return f'{case}__{policy}__{start:05d}-{stop:05d}'


def claim(case, policy, start, stop):
    CLAIMS.mkdir(parents=True, exist_ok=True)
    path = CLAIMS / claim_name(case, policy, start, stop)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, json.dumps({'pid': os.getpid(), 'time': time.time()}).encode()); os.close(fd)
    return path


def claimed_keys(clean=True):
    """Keys of live claims; a claim whose process has ended is removed (it is this lane's own temporary file)."""
    out = []
    for path in sorted(CLAIMS.glob('*__*__*')) if CLAIMS.exists() else []:
        try:
            pid = json.loads(path.read_text())['pid']
        except (OSError, ValueError):
            continue
        if not Path(f'/proc/{pid}').exists():
            if clean:
                path.unlink(missing_ok=True)
            continue
        case, policy, rng = path.name.split('__')
        a, b = rng.split('-')
        out.append((case, policy, int(a), int(b)))
    return out


def busy_keys(own_pids=()):
    """In-flight work of this lane: archive predict wrappers (any runner, GPU_LANE=Q1) and claimed fast calls."""
    return r1.external_running(set(own_pids)) + claimed_keys()


# --- failures ---------------------------------------------------------------------------------------------------------
def fast_failures(path=None):
    """{(case, policy): failed fast calls} from the fast-jobs ledger."""
    path = Path(path or fast.FAST_JOBS)
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('type') == 'call' and r.get('exit'):
                k = (r['case'], r['policy'])
                out[k] = out.get(k, 0) + 1
    return out


# --- the queue ----------------------------------------------------------------------------------------------------------
def grid_with_extensions(spec, extra):
    """The case's 1k grid with the protocol's extension widths, rebuilt from the sealed 1k results (deterministic:
    apply the extension rules until they add nothing)."""
    grid, bottom, top = r1.grid_wants(spec, None, extra)
    for _ in range(64):
        if not r1.apply_extensions(spec, bottom, top, extra):
            break
        grid, bottom, top = r1.grid_wants(spec, None, extra)
    return grid, bottom, top


def build_queue(cases, busy, max_priority=4, extra=None, gate_root=None):
    """Runs that may be launched now: dicts {rank, case, policy, start, stop, engine, kind}, sorted by (rank, case
    order).  kind 'k2' = a check-K2 call (fast wide/control 0-1000 of a gated case); 'run' = a protocol run.
    Busy (case, policy) pairs are left out, as are all runs of a gated case until its K2 has passed."""
    extra = {} if extra is None else extra
    view = BusyView(busy)
    a128, b128 = plan.LOCATION
    a, b = plan.SCREEN
    out = []

    def add(rank, i, case, policy, start, stop, engine, kind='run'):
        if not view.busy(case, policy):
            out.append({'rank': rank, 'i': i, 'case': case, 'policy': policy, 'start': start, 'stop': stop,
                        'engine': engine, 'kind': kind})

    for i, spec in enumerate(cases):
        case = spec['case']
        engine = fast.engine_for(case, gate_root)
        if engine == 'fast' and not fast.k2_passed(case):
            for pol in ('wide', 'control'):
                if not fast.fast_covers(case, pol, a, b):
                    add(0.5, i, case, pol, a, b, 'fast', 'k2')
            continue
        for pol in ('wide', 'control'):
            if fast.prefix_end(case, pol, b) < b:
                add(1.0, i, case, pol, a, b, engine)
        wide = summary(case, 'wide', a128, b128)
        if wide is None:
            continue
        w128 = wide['expected_percent']
        for pol, s, t in r1.location_wants(spec, w128, view):
            add(1.5, i, case, pol, s, t, engine)
        uni_done = r1.uniform_location_complete(spec, w128)
        str_done = spec.get('stress_case') or r1.struct_location_complete(spec, w128)
        if not (uni_done and str_done):
            continue
        grid, bottom, top = grid_with_extensions(spec, extra)
        for g in grid:
            if g['priority'] > max_priority or g['policy'] in ('wide', 'control'):
                continue
            if fast.prefix_end(case, g['policy'], b) < b:
                add(float(g['priority']), i, case, g['policy'], a, b, engine)
    out.sort(key=lambda r: (r['rank'], r['i']))
    return out


def archive_range(item):
    """Range of the next archive call of a queued run (archive root only: continue its prefix, addendum-2 tiles)."""
    from .load import prefix_end as archive_prefix
    case, policy, start, stop = item['case'], item['policy'], item['start'], item['stop']
    start = max(start, archive_prefix(case, policy, stop))
    if start >= stop:
        return None
    return start, r1.tile_stop(start, stop)


def location_complete(spec):
    a128, b128 = plan.LOCATION
    wide = summary(spec['case'], 'wide', a128, b128)
    if wide is None:
        return False
    w = wide['expected_percent']
    return r1.uniform_location_complete(spec, w) and (spec.get('stress_case') or r1.struct_location_complete(spec, w))


def gpu_seconds():
    """Lane GPU job seconds since the protocol: archive calls (first runner's rule: 1f75c923 engine ledger matched
    to this lane's calls) + fast jobs (fast.fast_job_seconds: job wall time after admission)."""
    since = r1.protocol_epoch()
    return r1.gpu_job_seconds(since) + fast.fast_job_seconds(since)
