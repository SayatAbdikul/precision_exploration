import pytest

pytest.importorskip('scipy')
from tools.scaled_bridge_gap_v1 import analysis
from tools.scaled_bridge_gap_v1.common import PROTOCOL, protocol_path
from tools.experiment_b.common import unseal


def test_upper_bound_zero_and_nonzero():
    assert abs(analysis.upper95(0, 1000) - (1 - .05 ** (1 / 1000))) < 1e-15
    assert abs(100 * analysis.upper95(0, 1000) - 0.29914) < 1e-4
    assert analysis.upper95(3, 1000) > analysis.upper95(0, 1000)
    assert 0.0077 < analysis.upper95(3, 1000) < 0.0078


def test_directions_counts_and_sign_test():
    base = [[1], [2], [3], [4], [5]]
    other = [[1], [9], [3], [5], [7]]
    labels = [1, 2, 8, 5, 6]
    d = analysis.directions(base, other, labels)
    assert d['changed_top1'] == 3
    assert (d['correct_to_wrong'], d['wrong_to_correct'], d['wrong_to_other_wrong']) == (1, 1, 1)
    assert d['sign_test_two_sided_p'] == 1.0


def test_first_divergence_order():
    a = {'layers': {'x': {'codes': 1}, 'y': {'codes': 2}, 'z': {'codes': 3}}}
    b = {'layers': {'x': {'codes': 1}, 'y': {'codes': 5}, 'z': {'codes': 6}}}
    assert analysis.first_divergence(a, b) == 'y'
    assert analysis.first_divergence(a, a) is None


def test_protocol_sealed_matches_code():
    if protocol_path().exists():
        assert unseal(protocol_path()) == PROTOCOL


def test_first_divergence_uses_execution_order_not_key_order():
    # sealed records store layers with sorted keys; 'add' sorts before 'conv1' but runs after it
    a = {'layers': {'add': {'codes': 1}, 'conv1': {'codes': 2}, 'x': {'codes': 0}}}
    b = {'layers': {'add': {'codes': 9}, 'conv1': {'codes': 8}, 'x': {'codes': 0}}}
    assert analysis.first_divergence(a, b, ['x', 'conv1', 'add']) == 'conv1'
    with pytest.raises(ValueError):
        analysis.first_divergence(a, b, ['x', 'conv1'])
