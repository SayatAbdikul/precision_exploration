"""Lane Q1 on the fast path (addendum 3): routing, the two-root loader, budget accounting, seeded draws, comparisons."""
import json
import os

import pytest

from tools.experiment_b.common import seal
from tools.accumulator_sweep_mn import fast, sched

CASE = 'mobilenet_v2-fp6_e2m3-default-b2'
UNGATED = 'mobilenet_v2-int6-default-b2'


def payload(case, policy, start, stop, kind, label=0):
    p = {'case': case, 'policy': policy, 'start': start, 'stop': stop, 'batch': 8, 'execution_seconds': 1.0,
         'nodes': {'n1': {'width': 20}},
         'images': [{'index': i, 'label': label, 'top5': [1, 2, 3, 4, 5], 'output': f'h{i}'} for i in range(start, stop)]}
    if kind == 'archive':
        p['engine_sources'] = fast.BASE_DIGEST
    else:
        p.update(engine_sources=fast.FAST_DIGEST, base_engine_sources=fast.BASE_DIGEST, execution_path=fast.FAST_PATH_TEXT)
    return p


def put(root, case, policy, start, stop, kind, **kw):
    path = root / case / 'predictions' / f'{policy}-cuda-{start:05d}-{stop:05d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    seal(path, payload(case, policy, start, stop, kind, **kw))
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    arch, fst = tmp_path / 'arch', tmp_path / 'fast'
    monkeypatch.setattr(fast, 'ARCHIVE_ROOT', arch)
    monkeypatch.setattr(fast, 'FAST_ROOT', fst)
    monkeypatch.setattr(fast, 'ROOT', tmp_path)
    return arch, fst


# --- routing --------------------------------------------------------------------------------------------------------
def test_routing_gated_case_to_fast_ungated_to_archive(tmp_path):
    gates = tmp_path / 'gates'
    seal(gates / CASE / 'gate.json', {'status': 'pass', 'case': CASE})
    seal(gates / 'mobilenet_v3_large-int8-default-b2' / 'gate.json', {'status': 'fail'})
    assert fast.engine_for(CASE, gates) == 'fast'
    assert fast.engine_for(UNGATED, gates) == 'archive'
    assert fast.engine_for('mobilenet_v3_large-int8-default-b2', gates) == 'archive'   # a failing gate does not admit


def test_queue_gated_case_waits_for_k2_and_routes(tmp_path, monkeypatch):
    gates = tmp_path / 'gates'
    seal(gates / CASE / 'gate.json', {'status': 'pass'})
    monkeypatch.setattr(fast, 'k2_passed', lambda case: False)
    monkeypatch.setattr(fast, 'fast_covers', lambda *a, **k: False)
    cases = [{'case': CASE}, {'case': UNGATED}]
    monkeypatch.setattr(fast, 'prefix_end', lambda case, policy, stop, root=None: 1000)
    monkeypatch.setattr(sched, 'summary', lambda *a: None)        # no location data: only K2 / P1 items can appear
    q = sched.build_queue(cases, [], gate_root=gates)
    assert [(x['case'], x['policy'], x['engine'], x['kind'], x['start'], x['stop']) for x in q] == [
        (CASE, 'wide', 'fast', 'k2', 0, 1000), (CASE, 'control', 'fast', 'k2', 0, 1000)]
    # after K2 the gated case's P1 runs (if any were missing) are fast, the ungated case's are archive runs
    monkeypatch.setattr(fast, 'k2_passed', lambda case: True)
    monkeypatch.setattr(fast, 'prefix_end', lambda case, policy, stop, root=None: 0)
    q = sched.build_queue(cases, [], gate_root=gates)
    assert {(x['case'], x['engine']) for x in q} == {(CASE, 'fast'), (UNGATED, 'archive')}
    # busy (case, policy) pairs are left out
    q = sched.build_queue(cases, [(UNGATED, 'wide', 0, 128)], gate_root=gates)
    assert (UNGATED, 'wide') not in {(x['case'], x['policy']) for x in q}


def test_claims_exclusive_and_stale_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(sched, 'CLAIMS', tmp_path / 'claims')
    p = sched.claim(CASE, 'sat.w14', 0, 128)
    assert p is not None and sched.claim(CASE, 'sat.w14', 0, 128) is None
    assert sched.claimed_keys() == [(CASE, 'sat.w14', 0, 128)]
    p.write_text(json.dumps({'pid': 2 ** 30, 'time': 0}))            # a process that does not exist
    assert sched.claimed_keys() == [] and not p.exists()


# --- two-root loader -------------------------------------------------------------------------------------------------
def test_loader_prefers_single_root_then_mixes(roots):
    arch, fst = roots
    put(arch, CASE, 'sat.w14', 0, 128, 'archive')
    put(fst, CASE, 'sat.w14', 0, 1000, 'fast')
    t = fast.choose(CASE, 'sat.w14', 0, 128)
    assert [x[4] for x in t] == ['archive']                           # archive alone covers 0-128
    t = fast.choose(CASE, 'sat.w14', 0, 1000)
    assert [x[4] for x in t] == ['fast']                              # fast alone covers 0-1000
    put(arch, CASE, 'sat.w12', 0, 128, 'archive')
    put(fst, CASE, 'sat.w12', 128, 1000, 'fast')
    d = fast.assemble(CASE, 'sat.w12', 0, 1000)
    assert d['roots'] == ['archive', 'fast'] and len(d['images']) == 1000
    assert [(f['root'], f['engine_sources'][:8], (f['base_engine_sources'] or '')[:8]) for f in d['files']] == [
        ('archive', '1f75c923', ''), ('fast', 'edfecd5c', '1f75c923')]
    assert fast.prefix_end(CASE, 'sat.w12', 1000) == 1000
    assert fast.choose(CASE, 'sat.w10', 0, 128) is None


def test_loader_engine_identity_by_root(roots):
    arch, fst = roots
    put(arch, CASE, 'sat.w14', 0, 128, 'fast')                        # a fast file inside the archive root
    with pytest.raises(ValueError, match='engine identity'):
        fast.assemble(CASE, 'sat.w14', 0, 128)
    put(fst, CASE, 'sat.w16', 0, 128, 'archive')                      # an archive-digest file inside the fast root
    with pytest.raises(ValueError, match='engine identity'):
        fast.assemble(CASE, 'sat.w16', 0, 128)
    p = payload(CASE, 'sat.w18', 0, 128, 'fast'); p['base_engine_sources'] = 'x' * 64
    seal(fst / CASE / 'predictions' / 'sat.w18-cuda-00000-00128.json', p)
    with pytest.raises(ValueError, match='engine identity'):
        fast.assemble(CASE, 'sat.w18', 0, 128)


def test_cross_root_comparison(roots):
    arch, fst = roots
    put(arch, CASE, 'wide', 0, 128, 'archive'); put(arch, CASE, 'wide', 128, 1000, 'archive')
    put(fst, CASE, 'wide', 0, 1000, 'fast')
    r = fast.cross_root(CASE, 'wide', 0, 1000)
    assert r['status'] == 'pass' and r['same'] == 1000 and r['differ'] == 0
    put(fst, CASE, 'control', 0, 1000, 'fast', label=1)
    put(arch, CASE, 'control', 0, 1000, 'archive')
    r = fast.cross_root(CASE, 'control', 0, 1000)
    assert r['status'] == 'fail' and r['differ'] == 1000 and r['first_difference']['fields'] == ['label']


# --- budget accounting ------------------------------------------------------------------------------------------------
def test_fast_job_seconds(tmp_path):
    path = tmp_path / 'fast-jobs.jsonl'
    rows = [{'type': 'begin', 'job': 'a', 'pid': 2 ** 30, 'time': 1000.0},
            {'type': 'call', 'job': 'a', 'time': 1001.0, 'ended': 1030.0, 'exit': 0, 'case': CASE, 'policy': 'wide'},
            {'type': 'end', 'job': 'a', 'time': 1100.0},
            {'type': 'begin', 'job': 'b', 'pid': 2 ** 30, 'time': 2000.0},        # died after one call: to its last call
            {'type': 'call', 'job': 'b', 'time': 2005.0, 'ended': 2050.0, 'exit': 1, 'case': CASE, 'policy': 'sat.w9'},
            {'type': 'begin', 'job': 'c', 'pid': os.getpid(), 'time': 3000.0},   # still running: to now
            {'type': 'begin', 'job': 'old', 'pid': 2 ** 30, 'time': 10.0},
            {'type': 'end', 'job': 'old', 'time': 20.0}]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert fast.fast_job_seconds(since=500, path=path, now=3010.0) == pytest.approx(100 + 50 + 10)
    assert fast.fast_job_seconds(since=0, path=path, now=3010.0) == pytest.approx(170)
    assert sched.fast_failures(path) == {(CASE, 'sat.w9'): 1}


def test_old_ledger_rule_would_count_zero_for_fast_ledger():
    """The reason for the new rule (reviewer D, C1): the fast engine ledger has no label and string arguments."""
    fast_ledger = {'arguments': ['predict', CASE, 'wide', '0', '1000', '8'], 'started_epoch': 1e10, 'wall_seconds': 20.0}
    assert fast_ledger.get('label') != 'predict'


# --- seeded draws -----------------------------------------------------------------------------------------------------
def test_k3_draws_deterministic():
    assert fast.k3_start(CASE) == fast.k3_start(CASE) and 0 <= fast.k3_start(CASE) <= 96
    loc = {24: {'events': 0}, 22: {'events': 0}, 20: {'events': 3}, 18: {'events': 9}}
    grid = ['wide', 'control', 'sat.w21', 'sat.w20', 'sat.w19', 'fp16.x-9', 'f21']
    p = fast.k3_policies(CASE, loc, grid, 'f21.x-9')
    assert p[0] == 'sat.w20' and p[1] in {'sat.w21', 'sat.w19', 'fp16.x-9', 'f21'} and p[2] == 'f21.x-9'
    assert fast.k3_policies(CASE, loc, list(reversed(grid))) == p[:2]


def test_k4_sample_ten_percent_and_rounds():
    cands = [(CASE, f'sat.w{w}', 0, 1000) for w in range(2, 25)]           # 23 files -> 3
    s1 = fast.k4_sample(cands)
    assert len(s1) == 3 and s1 == fast.k4_sample(list(reversed(cands)))
    s2 = fast.k4_sample(cands, s1)
    assert len(s2) == 2 and not set(s1) & set(s2)
