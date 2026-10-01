import json

import pytest

from tools.analysis import useful_quality_e1_v2 as analysis
from tools.breadth_study import useful_quality as v1
from tools.experiment_b.common import seal
from tools.phase3 import common


def counts(n=0):
    return {f'{mode}/{backend}': n for mode in ('exact', 'control') for backend in ('cpp', 'cuda')}


@pytest.mark.parametrize('fields,expected', [
    ({'counts': counts(), 'panels': [], 'admitted': False, 'promotion': None,
      'blocked': [], 'controller_status': 'running'}, 'enrolled_pending'),
    ({'counts': counts(1), 'panels': [], 'admitted': False, 'promotion': None,
      'blocked': [], 'controller_status': 'running'}, 'native_gate_in_progress'),
    ({'counts': counts(8), 'panels': [], 'admitted': True, 'promotion': None,
      'blocked': [], 'controller_status': 'running'}, 'native_admitted_panel_pending'),
    ({'counts': counts(8), 'panels': [], 'admitted': True, 'promotion': None,
      'blocked': [{'stage': 'panel'}], 'controller_status': 'budget_limited'}, 'panel_budget_limited'),
    ({'counts': counts(32), 'panels': [32], 'admitted': True,
      'promotion': {'quality_eligible': True, 'promote': False},
      'blocked': [], 'controller_status': 'budget_limited'}, 'completed_32_extension_budget_limited'),
    ({'counts': counts(32), 'panels': [32], 'admitted': True,
      'promotion': {'quality_eligible': False, 'promote': False},
      'blocked': [], 'controller_status': 'stages_complete'}, 'completed_32'),
    ({'counts': counts(128), 'panels': [32, 128], 'admitted': True,
      'promotion': {'quality_eligible': True, 'promote': True},
      'blocked': [], 'controller_status': 'stages_complete'}, 'completed_128'),
])
def test_classify_keeps_native_gate_panel_and_budget_states_distinct(fields, expected):
    assert analysis.classify(**fields) == expected


def test_checkpoint_inventory_rejects_gap_and_keeps_original_provenance(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    sample = [{'sha256': str(i)} for i in range(3)]
    plan = {'case': 'resnet18/int8'}
    monkeypatch.setattr(analysis, 'validate_record', lambda record, *_: None)
    seal(tmp_path / '0000-exact-cuda.json', {'source_kind': 'retained_verified_exact',
                                           'new_compute_seconds': 0})
    seal(tmp_path / '0001-exact-cuda.json', {'source_kind': 'new_verified_execution',
                                           'new_compute_seconds': 2.5, 'peak_rss_kib': 321})
    records, have, provenance, cost, comparisons = analysis._records(path, plan, sample)
    assert have['exact/cuda'] == 2
    assert provenance['exact/cuda'] == {'retained_verified_exact': 1, 'new_verified_execution': 1}
    assert cost['exact/cuda']['new_compute_seconds'] == 2.5
    assert cost['exact/cuda']['median_new_compute_seconds_per_image'] == 2.5
    assert comparisons['exact/cuda'] == 0
    (tmp_path / '0001-exact-cuda.json').unlink()
    seal(tmp_path / '0002-exact-cuda.json', {'source_kind': 'new_verified_execution',
                                           'new_compute_seconds': 2.5})
    with pytest.raises(ValueError, match='gap'):
        analysis._records(path, plan, sample)


def test_report_does_not_invent_quality_for_unenrolled_or_incomplete_cases():
    budget = {'limit_seconds': 86400, 'v1_sunk_seconds': 60000,
              'v2_pre_enrollment_compute_seconds': 120, 'v2_spent_seconds': 500,
              'v2_reserved_seconds': 100, 'remaining_seconds': 25680}
    result = {'controller_status': 'budget_limited', 'enrollment_status': 'sealed',
              'budget': budget, 'candidates': [
                  {'case': 'resnet18/fp6_e2m3', 'status': 'native_gate_budget_limited',
                   'images': 0, 'counts': counts(1)}],
              'limitations': ['Development only.']}
    text = analysis.report(result)
    assert 'native_gate_budget_limited' in text
    assert '0 | — | — | — | —' in text
    assert 'Development only.' in text


def test_completed_panel_recomputes_paired_statistics_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    monkeypatch.setattr(analysis, 'reference', lambda path: common.reference(path, root=tmp_path))
    monkeypatch.setattr(analysis, 'checked', lambda ref: common.checked(ref, root=tmp_path))
    job = tmp_path / 'jobs' / 'case'
    job.mkdir(parents=True)
    path = job / 'plan.json'
    path.write_text('{}')
    (job / 'native-admission.json').write_text('{}')
    samples = [{'sha256': f's{i}', 'label': str(i % 2)} for i in range(32)]
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'records': [
        {'sample_sha256': sample['sha256'], 'ground_truth': i % 2,
         'fp32_prediction': [i % 2, 2, 3, 4, 5]}
        for i, sample in enumerate(samples)]}))
    bdir = tmp_path / 'artifacts' / 'experiment_b' / 'predictions' / 'config'
    bdir.mkdir(parents=True)
    for i, sample in enumerate(samples):
        seal(bdir / f"{sample['sha256']}.json",
             {'configuration_sha256': 'config', 'sample': sample, 'top5': [i % 2, 2, 3, 4, 5]})
    mapping = tmp_path / 'mapping.json'
    seal(mapping, {'ordinary_B': {'component': 'experiment_b',
                                  'configuration_sha256': 'config', 'configuration': {'name': 'different-B'}}})
    plan = {'deployed_nodes': ['fc'], 'baseline': analysis.reference(baseline),
            'mapping': analysis.reference(mapping)}
    records = {}
    for mode in ('exact', 'control'):
        arm = []
        for i, sample in enumerate(samples):
            correct = i % 2
            prediction = ([correct, 2, 3, 4, 5] if mode == 'exact' or i % 4
                          else [1-correct, 2, 3, 4, 5])
            record = {'index': i, 'sample': sample, 'ground_truth': correct,
                      'fp32_prediction': [correct, 2, 3, 4, 5],
                      'prediction': prediction, 'layers': {'fc': {'sha256': str(prediction[0])}},
                      'output_sha256': str(prediction[0])}
            seal(job / f'{i:04d}-{mode}-cuda.json', record)
            arm.append(record)
        records[f'{mode}/cuda'] = arm
    exact, control = records['exact/cuda'], records['control/cuda']
    stats = {}
    for rank in (1, 5):
        truth = [r['ground_truth'] in r['fp32_prediction'][:rank] for r in exact]
        a = [r['ground_truth'] in r['prediction'][:rank] for r in exact]
        b = [r['ground_truth'] in r['prediction'][:rank] for r in control]
        stats[f'top{rank}'] = {'control_minus_exact': v1.paired_stats(a, b),
                               'exact_minus_FP32': v1.paired_stats(truth, a),
                               'control_minus_FP32': v1.paired_stats(truth, b)}
    divergences = [{'index': i, 'sample_sha256': samples[i]['sha256'],
                    'first_layer': 'fc' if i % 4 == 0 else None,
                    'changed_layers': ['fc'] if i % 4 == 0 else [],
                    'output_matches': i % 4 != 0} for i in range(32)]
    summary = {'plan': analysis.reference(path), 'images': 32, 'statistics': stats,
               'records': {mode: [analysis.reference(job / f'{i:04d}-{mode}-cuda.json')
                                  for i in range(32)] for mode in ('exact', 'control')},
               'divergences': divergences, 'all_layer_agreement_images': 24,
               'output_agreement_images': 24}
    seal(job / 'summary-32.json', summary)
    result = analysis._panel(path, plan, 32, records, samples)
    assert result['quality']['exact']['top1']['correct'] == 32
    assert result['quality']['control']['top1']['correct'] == 24
    summary['statistics']['top1']['control_minus_exact']['difference_pp'] = 0
    seal(job / 'summary-32.json', summary)
    with pytest.raises(ValueError, match='paired statistic'):
        analysis._panel(path, plan, 32, records, samples)
