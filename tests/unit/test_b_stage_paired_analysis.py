import numpy as np
import pytest

from tools.analysis.b_stage_deeper_comparisons import SEED, compare_records, holm, paired_outcomes


def test_paired_interval_matches_explicit_image_resampling_distribution():
    # Many concordant observations and few disagreements: unpaired resampling
    # would yield a much wider interval despite identical marginal accuracies.
    left = np.array([1]*480 + [0]*480 + [1]*30 + [0]*10)
    right = np.array([1]*480 + [0]*480 + [0]*30 + [1]*10)
    result = paired_outcomes(left, right, resamples=20000)
    rng = np.random.default_rng(SEED)
    delta = right-left
    samples = []
    for _ in range(200):
        samples.extend(100*delta[rng.integers(0, 1000, (100, 1000))].mean(axis=1))
    np.testing.assert_allclose(result['pointwise_95_interval_pp'], np.quantile(samples, [.025, .975]), atol=.11)
    assert result['difference_pp'] == -2
    assert result['left_only_correct'] == 30
    assert result['right_only_correct'] == 10


def test_zero_disagreement_is_not_reported_as_proven_equivalence():
    result = paired_outcomes([1]*1000, [1]*1000)
    assert result['pointwise_95_interval_pp'] == [0, 0]
    assert .29 < result['zero_discordance_upper_95_percent'] < .31
    assert result['mcnemar_exact_p'] == 1


def test_pairing_rejects_equal_length_different_order():
    rows = [{'sample': {'sha256': str(i), 'label': i}, 'top5': [0, 1, 2, 3, 4]} for i in range(2)]
    with pytest.raises(ValueError, match='identical ordered samples'):
        compare_records(rows, rows[::-1])


def test_pairing_rejects_unequal_panels():
    rows = [{'sample': {'sha256': str(i), 'label': i}, 'top5': [0, 1, 2, 3, 4]} for i in range(2)]
    with pytest.raises(ValueError, match='identical ordered samples'):
        compare_records(rows, rows[:1])


def test_holm_monotone_adjustment_with_tied_and_unsorted_pvalues():
    assert holm([.04, .01, .03]) == [.06, .03, .06]
    assert holm([1, 1]) == [1, 1]
