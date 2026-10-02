import numpy as np

from tools.experiment_b2_recon.collapse_check import readout, verdict


def test_readout_detects_one_class_collapse():
    top1 = np.full(1000, 463)
    top1[:50] = np.arange(50)
    stats = readout(top1, np.ones(1000), np.zeros(1000))
    assert stats["most_frequent_class"] == 463
    assert stats["most_frequent_share"] == 0.95
    assert stats["distinct_top1_classes"] == 51
    assert stats["collapsed"]
    assert stats["top1_expected_pct"] == 0.0


def test_readout_varied_predictions_and_tie_credit():
    top1 = np.arange(1000) % 100
    ties = np.ones(1000)
    ties[0] = 4
    hit = np.zeros(1000)
    hit[0] = 1
    hit[1] = 1
    stats = readout(top1, ties, hit)
    assert not stats["collapsed"]
    assert stats["distinct_top1_classes"] == 100
    assert stats["max_tie_size"] == 4
    assert abs(stats["top1_expected_pct"] - 0.125) < 1e-9


def test_verdict_rule():
    c, n = {"collapsed": True}, {"collapsed": False}
    assert verdict(c, n) == "supported"
    assert verdict(n, n) == "not_supported"
    assert verdict(c, c) == "undecided"
    assert verdict(None, n) is None
