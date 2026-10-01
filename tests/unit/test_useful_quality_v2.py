"""Scheduling and accounting gates for the optimized useful-quality controller."""
from pathlib import Path

import pytest

from tools.experiment_b.common import seal
from tools.breadth_study import useful_quality_v2 as study
from tools.breadth_study.useful_quality_v2_worker import validate_record, requires_legacy_whole_graph
from tools.phase3.common import digest
from tools.run.exact_execution import event_signature


def _plan(tmp_path, case, fmt):
    path = tmp_path / case.replace('/', '-') / 'plan.json'
    path.parent.mkdir(parents=True)
    seal(path, {'case': case, 'format': fmt, 'parent_plan': {'path': 'unused', 'sha256': 'unused'}})
    return path


def test_lifetime_budget_charges_old_compute_and_unreconciled_reservations(monkeypatch):
    monkeypatch.setattr(study, 'old_sunk_seconds', lambda: (23 * 3600, {'old': {'charged_seconds': 23 * 3600}}))
    monkeypatch.setattr(study, 'interrupted_charge', lambda: (0.0, False, []))
    monkeypatch.setattr(study, 'pre_enrollment_cost', lambda: 0.0)
    monkeypatch.setattr(study, 'committed_allocations', lambda: (900, 1800, 300, [
        {'stage': 'extension', 'state': 'reserved_unreconciled', 'reserved_seconds': 1800}]))
    result = study.budget()
    assert result['remaining_seconds'] == 900
    assert result['extension_remaining_seconds'] == 12 * 3600 - 300 - 1800
    assert result['v1_sunk_seconds'] == 23 * 3600


def test_probe_precedes_eight_image_gate_and_admitted_panel_can_run_early(tmp_path, monkeypatch):
    quick = _plan(tmp_path, 'resnet18/int8', 'int8')
    slow = _plan(tmp_path, 'resnet18/fp6_e2m3', 'fp6_e2m3')
    (quick.parent / 'native-admission.json').write_text('{}')
    have = {
        quick: {('exact', 'cpp'): 8, ('exact', 'cuda'): 8,
                ('control', 'cpp'): 8, ('control', 'cuda'): 8},
        slow: {('exact', 'cpp'): 1, ('exact', 'cuda'): 0,
               ('control', 'cpp'): 0, ('control', 'cuda'): 0},
    }
    monkeypatch.setattr(study, 'counts', lambda path: have[path])
    tasks = study.ready_tasks([(quick, {'case': 'resnet18/int8'}),
                               (slow, {'case': 'resnet18/fp6_e2m3'})])
    assert tasks[0].case == 'resnet18/int8' and tasks[0].stage == 'panel'
    assert any(t.case == 'resnet18/fp6_e2m3' and t.stage == 'probe' and t.backend == 'cuda' for t in tasks)
    assert not any(t.case == 'resnet18/fp6_e2m3' and t.stage == 'panel' for t in tasks)
    assert not any(t.case == 'resnet18/fp6_e2m3' and t.stage == 'native_gate' for t in tasks)


def test_unaudited_stage_cannot_consume_more_than_remaining_budget(tmp_path, monkeypatch):
    path = _plan(tmp_path, 'resnet18/fp6_e3m2', 'fp6_e3m2')
    task = study.Task(path, 'exact', 'cpp', 8, 'native_gate')
    monkeypatch.setattr(study, 'estimate', lambda task: 3700.0)
    monkeypatch.setattr(study, 'BASE', tmp_path)
    assert study.allocate(task, current_budget={'remaining_seconds': 3600,
                                                'extension_remaining_seconds': 3600}) is None
    assert list((tmp_path / 'allocations').glob('*.json')) == [] if (tmp_path / 'allocations').exists() else True


def test_imported_integer_exact_has_no_per_image_inference_charge(tmp_path, monkeypatch):
    path = _plan(tmp_path, 'resnet18/int8', 'int8')
    monkeypatch.setattr(study, 'counts', lambda _: {('exact', 'cuda'): 8})
    monkeypatch.setattr(study, 'checked', lambda _: Path('/does/not/matter'))
    monkeypatch.setattr(study, 'measured_rates', lambda *args: 1000.0)
    assert study.estimate(study.Task(path, 'exact', 'cuda', 128, 'extension')) == 135.0


def test_floating_control_gate_requires_fresh_execution(tmp_path, monkeypatch):
    path = _plan(tmp_path, 'resnet18/fp6_e2m3', 'fp6_e2m3')
    monkeypatch.setattr(study, 'counts', lambda _: {('control', 'cpp'): 0})
    monkeypatch.setattr(study, 'checked', lambda _: Path('/does/not/matter'))
    monkeypatch.setattr(study, 'measured_rates', lambda *args: 10.0)
    assert study.estimate(study.Task(path, 'control', 'cpp', 8, 'native_gate')) == 1.5 * (8 * 10 + 90)


def test_third_worker_waits_for_all_timing_probes(tmp_path, monkeypatch):
    path = _plan(tmp_path, 'resnet18/int8', 'int8')
    monkeypatch.setattr(study.v1, 'memory_available', lambda: 16 * 2**30)
    monkeypatch.setattr(study, 'old_plans', lambda: (None, []))
    have = {(mode, backend): 0 for mode in study.MODES for backend in study.BACKENDS}
    monkeypatch.setattr(study, 'counts', lambda _: have)
    assert study.memory_workers([(path, {'case': 'resnet18/int8'})]) == 2
    have.update({key: 1 for key in have})
    assert study.memory_workers([(path, {'case': 'resnet18/int8'})]) == 3


def test_interrupted_v1_work_has_no_spendable_budget_without_charge(tmp_path, monkeypatch):
    path = _plan(tmp_path, 'resnet18/fp6_e2m3', 'fp6_e2m3')
    (path.parent / '0000-exact-cpp.json').write_text('{}')
    monkeypatch.setattr(study, 'old_plans', lambda: (None, [(path, {'case': 'resnet18/fp6_e2m3'})]))
    monkeypatch.setattr(study, 'INTERRUPTION_CHARGE', tmp_path / 'missing-charge.json')
    monkeypatch.setattr(study, 'old_sunk_seconds', lambda: (10_000.0, {}))
    monkeypatch.setattr(study, 'pre_enrollment_cost', lambda: 0.0)
    monkeypatch.setattr(study, 'committed_allocations', lambda: (0.0, 0.0, 0.0, []))
    assert study.interrupted_charge() == (0.0, True, ['resnet18/fp6_e2m3'])
    assert study.budget()['remaining_seconds'] == 0


def test_pre_enrollment_full_image_compute_is_charged_once(monkeypatch):
    monkeypatch.setattr(study, 'old_sunk_seconds', lambda: (70_000.0, {}))
    monkeypatch.setattr(study, 'interrupted_charge', lambda: (2_000.0, False, []))
    monkeypatch.setattr(study, 'pre_enrollment_cost', lambda: 900.0)
    monkeypatch.setattr(study, 'committed_allocations', lambda: (100.0, 200.0, 0.0, []))
    result = study.budget()
    assert result['remaining_seconds'] == 86_400 - 70_000 - 2_000 - 900 - 100 - 200
    assert result['v2_pre_enrollment_compute_seconds'] == 900


def test_resume_rejects_checkpoint_gap_before_scheduling(tmp_path, monkeypatch):
    path = _plan(tmp_path, 'resnet18/fp6_e2m3', 'fp6_e2m3')
    protocol = tmp_path / 'protocol.json'
    seal(protocol, {'sample_rows': [{'sha256': str(index)} for index in range(128)]})
    (path.parent / '0001-exact-cpp.json').write_text('{}')
    monkeypatch.setattr(study, 'checked', lambda item: Path(item['path']))
    with pytest.raises(ValueError, match='checkpoint sequence contains a gap'):
        study.verify_resume_state([(path, {'protocol': {'path': str(protocol)}})])


def test_floating_checkpoint_requires_all_certified_mac_strategies():
    plan = {'format': 'fp6_e2m3', 'deployed_nodes': ['fc'],
            'mac_nodes': ['fc'], 'graph_sha256': 'graph'}
    sample = {'sha256': 'sample'}
    record = {'job_sha256': digest(plan), 'sample': sample, 'mode': 'exact',
              'backend': 'cpp', 'index': 0, 'graph_sha256': 'graph',
              'layers': {'fc': {}}, 'prediction': [0, 1, 2, 3, 4],
              'diagnostics': {}, 'diagnostic_signature': event_signature({}),
              'source_kind': 'new_verified_execution',
              'execution_modes': {'fc': 'fallback_rational'}}
    with pytest.raises(ValueError, match='did not use every certified native MAC'):
        validate_record(record, plan, sample, 'exact', 'cpp', 0)


def test_pre_enrollment_charge_includes_superseded_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'BASE', tmp_path)
    certificate = tmp_path / 'optimized-runtime-certificate.json'
    monkeypatch.setattr(study, 'CERTIFICATE_PATH', certificate)
    (tmp_path / 'benchmarks').mkdir()
    seal(tmp_path / 'benchmarks' / 'old-control.json',
         {'seconds': {'total_new_compute': 105.5}})
    seal(tmp_path / 'benchmarks' / 'new-control.json',
         {'seconds': {'total_new_compute': 15.0}})
    seal(certificate, {'pre_enrollment_compute_seconds': 120.0})
    with pytest.raises(ValueError, match='no longer covers sealed benchmark compute'):
        study.pre_enrollment_cost()


def test_uncompiled_first_image_comparison_is_exact_only():
    assert requires_legacy_whole_graph('exact', 'cpp', 0)
    assert not requires_legacy_whole_graph('control', 'cpp', 0)
    assert not requires_legacy_whole_graph('exact', 'cuda', 0)
    assert not requires_legacy_whole_graph('exact', 'cpp', 1)
