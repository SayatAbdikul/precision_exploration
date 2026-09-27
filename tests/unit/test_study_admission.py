from copy import deepcopy

import pytest

from tools.breadth_study.family_admission import verify_shapes


def evidence():
    graph = {'nodes': [{'name': 'conv'}, {'name': 'pool'}]}
    observed = {'input': [1, 3, 4, 4], 'folded_bn': [1, 2, 4, 4],
                'conv': [1, 2, 4, 4], 'pool': [1, 2, 1, 1]}
    layers = {n: {'shape': observed[n]} for n in ('conv', 'pool')}
    records = [{'backends': {b: {'layers': deepcopy(layers)} for b in ('cpp', 'cuda')}}]
    return graph, observed, records


def test_shape_evidence_allows_folded_nodes_but_covers_every_deployed_node():
    graph, observed, records = evidence()
    verify_shapes(graph, observed, records)
    del observed['pool']
    with pytest.raises(ValueError, match='omits'):
        verify_shapes(graph, observed, records)


@pytest.mark.parametrize('backend', ['cpp', 'cuda'])
def test_shape_evidence_rejects_missing_and_mismatched_native_shapes(backend):
    graph, observed, records = evidence()
    records[0]['backends'][backend]['layers']['pool']['shape'] = [1, 2, 2, 2]
    with pytest.raises(ValueError, match='native shapes'):
        verify_shapes(graph, observed, records)
    del records[0]['backends'][backend]['layers']['pool']
    with pytest.raises(ValueError, match='native shapes'):
        verify_shapes(graph, observed, records)


def test_downstream_failure_is_recorded_even_before_native_leases(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from tools.breadth_study import study_next
    monkeypatch.setattr(study_next, 'BASE', tmp_path)
    monkeypatch.setattr(study_next, 'exclusive', lambda _: nullcontext())
    def fail(_):
        raise RuntimeError('test preflight failure')
    monkeypatch.setattr(study_next, 'run_stages', fail)
    with pytest.raises(RuntimeError, match='test preflight failure'):
        study_next.run()
    status = study_next.read(tmp_path / 'status.json')
    assert status['status'] == 'failed'
    assert 'test preflight failure' in status['error']


def test_e2_runs_while_one_family_worker_is_active(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from threading import Event
    from types import SimpleNamespace
    from tools.breadth_study import study_next
    active, release = Event(), Event()
    plan = tmp_path / 'pending' / 'plan.json'
    monkeypatch.setattr(study_next, 'BASE', tmp_path)
    monkeypatch.setattr(study_next, 'read', lambda _: {'status': 'ready_cases_completed'})
    monkeypatch.setattr(study_next, 'exclusive', lambda _: nullcontext(SimpleNamespace(fileno=lambda: 11)))
    monkeypatch.setattr(study_next, 'pool_locks', lambda _: nullcontext({'cuda': 12, 'cpu': 13}))
    monkeypatch.setattr(study_next, 'fit_in_cuda_environment', lambda *a, **k: [])
    monkeypatch.setattr(study_next, 'family_inventory', lambda: {'cases': [{'status': 'ready', 'plan': str(plan)}]})
    monkeypatch.setattr(study_next, 'checked', lambda _: plan)
    def image_job(path, images, fds):
        active.set()
        assert release.wait(2), 'E2 was serialized behind E1'
        return {'completed': 'family'}
    def e2(fds, status, workers):
        assert active.wait(2), 'family did not run alongside E2'
        assert workers == 3
        release.set()
    monkeypatch.setattr(study_next, 'run_job', image_job)
    monkeypatch.setattr(study_next, 'run_e2', e2)
    status = {}
    study_next.run_stages(status)
    assert status['concurrency'] == {'E1_workers': 1, 'E2_workers': 3, 'threads_per_worker': 4}
    assert status['family_results'] == [{'completed': 'family'}]
