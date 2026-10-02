"""Lane Q7 (quality metrics v1): tie-set rules, interval helpers and the detector accumulate."""
import numpy as np
import pytest

from tools.analysis.quality_metrics_v1 import resolution, units


def test_matrix_membership_rules():
    top5 = np.array([[3, 7, 1, 2, 4], [2, 5, 9, 11, 13], [2, 5, 9, 11, 13], [2, 5, 9, 11, 13]])
    k = np.array([2, 8, 8, 8])
    c0 = np.array([7, 9, 6, 20])
    # k<=5: listed among the first k -> in; k>5: listed -> in, below the fifth listed -> out, above -> undetermined
    assert units.matrix_membership(c0, top5, k).tolist() == [1, 1, 0, -1]
    assert units.matrix_membership(np.array([1]), top5[:1], k[:1]).tolist() == [0]


def test_topk_membership_and_lowest():
    top5 = np.array([[9, 3, 4, 0, 1], [9, 3, 4, 0, 1]])
    k = np.array([3, 7])
    assert units.topk_membership(np.array([4, 2]), top5, k).tolist() == [1, -1]
    assert units.topk_lowest_argmax(top5, k).tolist() == [3, -1]


def test_binary_width_matches_paired_outcomes():
    from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
    a = np.zeros(200, np.int8); b = np.zeros(200, np.int8)
    a[:5] = 1; b[5:15] = 1
    low, high = paired_outcomes(a, b)["pointwise_95_interval_pp"]
    assert resolution.binary_width(200, 5, 10) == pytest.approx(high - low)


def test_intervals_contain_estimate_and_bca_degenerates():
    p, bca, wald = resolution.intervals(1000, 20, 35)
    for low, high in (p, bca, wald):
        assert low <= 0.015 <= high
    p0, bca0, _ = resolution.intervals(500, 0, 0)
    assert p0 == (0.0, 0.0) and bca0 == (0.0, 0.0)


def test_mdd_and_sample_size_are_consistent():
    pd = 0.05
    n = resolution.images_needed(pd, 0.01, "difference")
    assert resolution.mdd(n, pd) == pytest.approx(0.01, rel=0.01)


@pytest.mark.slow
def test_own_accumulate_reproduces_cocoeval():
    from tools.analysis.quality_metrics_v1 import detector
    from tools.experiment_b2_det.stats import annotations
    side = detector.side("int6", "default", detector.screen_rows(), annotations())
    fixed = detector.ap(detector.tables(side["ev"]), detector.stable_key)
    assert np.abs(fixed - side["point"]).max() < 1e-12
    assert abs(detector.full_stats(side["ev"])[0] - side["point"][0]) < 1e-12
