from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.breadth_study.useful_quality import paired_stats, promotion, validate_record, measured_cost
from tools.experiment_b.common import seal
from tools.phase3.common import digest


def summary(exact, control, baseline=70, agreement=32):
    return {'images': 32, 'all_layer_agreement_images': agreement, 'statistics': {'top1': {
        'control_minus_exact': {'left_percent': exact, 'right_percent': control},
        'exact_minus_FP32': {'right_percent': exact, 'difference_pp': exact-baseline},
        'control_minus_FP32': {'right_percent': control, 'difference_pp': control-baseline}}}}


def test_promotion_requires_useful_quality_or_informative_divergence():
    assert promotion(summary(0, 0))[0] is False
    assert promotion(summary(9, 10, agreement=0))[0] is False
    assert promotion(summary(50, 50))[0] is True
    assert promotion(summary(45, 45))[0] is False
    assert promotion(summary(45, 40, agreement=31))[0] is True
    assert promotion(summary(39, 39, agreement=0))[0] is False


def test_zero_observed_gap_keeps_nonzero_uncertainty_bound():
    result = paired_stats([1, 0]*16, [1, 0]*16)
    assert result['difference_pp'] == 0
    assert result['zero_discordance_upper_95_percent'] > 8
    assert paired_stats([0, 0], [1, 0])['difference_pp'] == 50
    with pytest.raises(ValueError):
        paired_stats([1], [1, 0])


def test_checkpoint_rejects_changed_identity_sample_or_missing_layer():
    plan = {'graph_sha256': 'g', 'deployed_nodes': ['a', 'b']}
    sample = {'sha256': 's'}
    record = {'job_sha256': digest(plan), 'sample': sample, 'mode': 'control', 'backend': 'cuda',
              'index': 0, 'graph_sha256': 'g', 'layers': {'a': {}, 'b': {}}, 'prediction': [0, 1, 2, 3, 4]}
    validate_record(record, plan, sample, 'control', 'cuda', 0)
    for field, value in [('job_sha256', 'other'), ('sample', {'sha256': 'different'}), ('layers', {'a': {}})]:
        changed = deepcopy(record); changed[field] = value
        with pytest.raises(ValueError, match='identity'):
            validate_record(changed, plan, sample, 'control', 'cuda', 0)


def test_cost_ignores_conformance_metadata_and_counts_reference_validation(tmp_path):
    seal(tmp_path/'control-conformance-cuda.json', {'status': 'passed'})
    seal(tmp_path/'0000-exact-cuda.json', {'mode': 'exact', 'new_compute_seconds': 0})
    seal(tmp_path/'0000-control-cuda.json', {'mode': 'control', 'new_compute_seconds': 5, 'peak_rss_kib': 100})
    seal(tmp_path/'prepared-reference-compatibility.json', {'seconds': 3})
    cost = measured_cost(tmp_path/'plan.json')
    assert cost == {'new_compute_seconds': 8, 'median_seconds': {'exact': 0, 'control': 5}, 'max_rss_kib': 100}


def test_independent_audit_checks_labels_and_full_precision_reference():
    from tools.analysis.useful_quality_e1 import check_pairing
    sample = {'sha256': 's', 'label': 3}
    baseline = {'records': [{'sample_sha256': 's', 'ground_truth': 3, 'fp32_prediction': [3, 0, 1, 2, 4]}]}
    records = [{'sample': sample, 'ground_truth': 3, 'fp32_prediction': [3, 0, 1, 2, 4], 'prediction': [0, 1, 2, 3, 4]}]
    check_pairing(records, baseline, [sample])
    records[0]['ground_truth'] = 2
    with pytest.raises(ValueError, match='pairing'):
        check_pairing(records, baseline, [sample])


def test_arithmetic_envelope_conditions_on_finite_inputs_and_rejects_nan_weights():
    from public.inference.reference.arithmetic import format_named
    from public.inference.tensor import Encoding, Tensor
    from tools.analysis.useful_quality_e1 import arithmetic_envelope
    encoding = Encoding('fp6_e2m3')
    fmt = format_named(encoding.format)
    nan = next(code for code in range(1 << fmt.bits) if fmt.decode(code).is_nan())
    def graph(weight_code):
        return {'inputs': {'x': encoding.document()},
                'constants': {'w': Tensor((1, 1), (weight_code,), encoding).document()},
                'nodes': [{'name': 'fc', 'op': 'linear', 'inputs': ['x', 'w'],
                           'attrs': {'accumulator': 'fp64_e11m52_accumulator', 'output': encoding.document()}}]}
    finite = arithmetic_envelope(graph(1))
    row = finite['MAC_bounds'][0]
    assert row['status'] == 'conditional_finite_inputs'
    assert nan in row['excluded_nonfinite_input_codes']
    assert row['maximum_absolute_prefix_quantum_units'] > 0
    invalid = arithmetic_envelope(graph(nan))
    assert invalid['MAC_bounds'][0]['status'] == 'nonfinite_stored_weight'
    assert invalid['all_MAC_numeric_prefix_bounds_fit_FP32'] is False
    stored_input = graph(1)
    stored_input['inputs'] = {}
    stored_input['constants']['x'] = Tensor((1, 1), (nan,), encoding).document()
    invalid_input = arithmetic_envelope(stored_input)
    assert invalid_input['MAC_bounds'][0]['status'] == 'nonfinite_stored_input'
    assert invalid_input['all_MAC_numeric_prefix_bounds_fit_FP32'] is False
