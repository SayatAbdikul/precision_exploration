import numpy as np
import pytest

pytest.importorskip('scipy')
from tools.experiment_b.common import unseal
from tools.scaled_bridge_gap_v1 import ties
from tools.scaled_bridge_gap_v1.diagnostics import DIAG_PROTOCOL, diag_protocol_path, stage, configurations, SINGLE


def test_common_order_breaks_ties_by_smaller_class_index():
    row = np.array([1., 7., 3., 7., 0., 3., 3., 2.])
    (tie1, top5, tie5), = ties.common_order(row[None])
    assert tie1 == [1, 3]
    assert top5 == [1, 3, 2, 5, 6]
    assert tie5 is False
    (tie1, top5, tie5), = ties.common_order(np.array([[5., 4., 4., 4., 4., 4., 0.]]))
    assert tie1 == [0] and top5 == [0, 1, 2, 3, 4] and tie5 is True


def test_common_order_is_permutation_consistent():
    rng = np.random.default_rng(0)
    units = rng.integers(-4, 5, size=(16, 50)).astype(np.float64)
    for row, (tie1, top5, _) in zip(units, ties.common_order(units)):
        assert row[top5[0]] == row.max() and top5[0] == min(tie1)
        assert all(row[a] > row[b] or (row[a] == row[b] and a < b) for a, b in zip(top5, top5[1:]))


def test_expected_top1_credit():
    assert ties.expected_top1([[3], [1, 2], [4, 5, 6], [7]], [3, 2, 9, 8]) == [1., .5, 0., 0.]


def test_stage_assignment_and_configurations():
    assert stage('conv1') == 'stem' and stage('maxpool') == 'stem'
    assert stage('layer3_1_conv2') == 'layer3' and stage('fc') == 'head'
    c = configurations()
    assert len(c) == 17 and set(SINGLE) <= set(c)


def test_sealed_protocols_match_code():
    if diag_protocol_path().exists():
        assert unseal(diag_protocol_path()) == DIAG_PROTOCOL
    if ties.ties_protocol_path().exists():
        assert unseal(ties.ties_protocol_path()) == ties.TIES_PROTOCOL


def test_tie_luck_counts_only_tie_images():
    tie1 = [[3], [1, 2], [4, 5], [6, 7, 8]]
    preds = [[3], [2], [5], [6]]
    r = ties.tie_luck(preds, tie1, [3, 2, 4, 9])
    assert r['tie_images'] == 3 and r['label_picked'] == 1
    assert abs(r['expected_random'] - 1.0) < 1e-12 and abs(r['sd_random'] - .5 ** .5) < 1e-12
    assert r['z'] == 0.0 and r['excess_pp_of_panel'] == 0.0


def test_exec_stages_places_residual_adds_in_their_block():
    from tools.scaled_bridge_gap_v1.diagnostics import exec_stages
    nodes = [{'name': 'x', 'inputs': []}, {'name': 'conv1', 'inputs': ['x']}, {'name': 'maxpool', 'inputs': ['conv1']},
             {'name': 'layer1_0_conv1', 'inputs': ['maxpool']}, {'name': 'add', 'inputs': ['layer1_0_conv1', 'maxpool']},
             {'name': 'layer2_0_downsample_1', 'inputs': ['add']}, {'name': 'layer2_0_bn2', 'inputs': ['add']},
             {'name': 'add_2', 'inputs': ['layer2_0_bn2', 'layer2_0_downsample_1']}, {'name': 'avgpool', 'inputs': ['add_2']},
             {'name': 'fc', 'inputs': ['avgpool']}, {'name': 'output', 'inputs': ['fc']}]
    m = exec_stages(nodes)
    assert m['add'] == 'layer1' and m['add_2'] == 'layer2'
    assert [k for k, v in m.items() if v == 'stem'] == ['x', 'conv1', 'maxpool']
    assert m['fc'] == 'head' and m['output'] == 'head'
    assert stage('add') == 'stem'  # diag-1 as-run assignment is kept unchanged for its sealed records


def test_stagefix_protocol_matches_code():
    from tools.scaled_bridge_gap_v1 import stagefix
    if stagefix.stagefix_protocol_path().exists():
        assert unseal(stagefix.stagefix_protocol_path()) == stagefix.STAGEFIX_PROTOCOL
    assert stagefix.labels_cfg() == ['Xstage2:' + s for s in ('stem', 'layer1', 'layer2', 'layer3', 'layer4', 'head')]
