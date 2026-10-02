import math

import pytest

from tools.accumulator_sweep_mn import analysis, certs
from tools.run.accumulator_sweep_mn_run import wrapped_predict


def test_policy_family():
    assert analysis.policy_family('sat.w17') == ('uniform', 17)
    assert analysis.policy_family('sat.struct-3') == ('per_node', 3)
    assert analysis.policy_family('fp16') == ('fp16', 0)
    assert analysis.policy_family('fp16.x-9') == ('fp16', -9)
    assert analysis.policy_family('f21.x-8') == ('f21', -8)
    assert analysis.policy_family('control') == ('control', None)
    with pytest.raises(ValueError):
        analysis.policy_family('sat.abs-2x')


def test_published_width():
    # [Agg24]: integer 2r + ceil(log2 n) + 1; minifloat 2 (2^e + m) + ceil(log2 n) - 1 (L8's reading)
    assert certs.published_width('int8', 4608) == 30
    assert certs.published_width('fp7_e3m3', 4608) == 34
    assert certs.published_width('int8', 1280) == 28
    assert certs.published_width('fp8_e4m3fn', 1280) == 48
    assert certs.published_width('posit8_es1', 1280) is None


def test_mac_kind_pointwise_split():
    nodes = [
        {'name': 'x', 'op': 'input', 'inputs': []},
        {'name': 'c0', 'op': 'conv', 'inputs': ['x']},
        {'name': 'a0', 'op': 'relu6', 'inputs': ['c0']},
        {'name': 'dw', 'op': 'conv', 'inputs': ['a0']},
        {'name': 'a1', 'op': 'relu6', 'inputs': ['dw']},
        {'name': 'pj', 'op': 'conv', 'inputs': ['a1']},
        {'name': 'i', 'op': 'identity', 'inputs': ['pj']},
        {'name': 'ex', 'op': 'conv', 'inputs': ['i']},
        {'name': 'i2', 'op': 'identity', 'inputs': ['ex']},
        {'name': 'hs', 'op': 'hardswish', 'inputs': ['i2']},
        {'name': 'p', 'op': 'avgpool', 'inputs': ['hs']},
        {'name': 'se1', 'op': 'conv', 'inputs': ['p']},
        {'name': 'se2', 'op': 'conv', 'inputs': ['se1']},
        {'name': 'hsig', 'op': 'hardsigmoid', 'inputs': ['se2']},
        {'name': 'fc', 'op': 'linear', 'inputs': ['p']},
    ]
    one = {'groups': 1, 'shape': [8, 8, 1, 1]}
    assert certs.mac_kind(nodes, 'c0', {'groups': 1, 'shape': [8, 3, 3, 3]}) == 'stem'
    assert certs.mac_kind(nodes, 'dw', {'groups': 8, 'shape': [8, 1, 3, 3]}) == 'depthwise'
    assert certs.mac_kind(nodes, 'pj', one) == 'pointwise_project'
    assert certs.mac_kind(nodes, 'ex', one) == 'pointwise_expand'
    assert certs.mac_kind(nodes, 'se1', one) == 'se_reduce'
    assert certs.mac_kind(nodes, 'se2', one) == 'se_expand'
    assert certs.mac_kind(nodes, 'fc', one) == 'classifier'


def test_node_needed_width_and_spearman():
    rows = [{'node': 'a', 'parameter': 20, 'images_with_event': 3}, {'node': 'a', 'parameter': 19, 'images_with_event': 9},
            {'node': 'b', 'parameter': 20, 'images_with_event': 0}, {'node': 'b', 'parameter': 18, 'images_with_event': 1}]
    assert analysis.node_needed_width(rows, 18) == {'a': 21, 'b': 19}
    assert analysis.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert analysis.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert analysis.spearman([1, 1, 1], [1, 2, 3]) is None
    assert analysis.spearman([1, 2], [1, 2]) is None


def test_kind_rows():
    rows = [{'case': 'c', 'policy': 'sat.w5', 'family': 'uniform', 'parameter': 5, 'kind': 'stem', 'node': 'n1',
             'images_with_event': 2, 'image_rate': 0.2, 'output_rate': 0.01},
            {'case': 'c', 'policy': 'sat.w5', 'family': 'uniform', 'parameter': 5, 'kind': 'depthwise', 'node': 'n2',
             'images_with_event': 0, 'image_rate': 0.0, 'output_rate': 0.0},
            {'case': 'c', 'policy': 'sat.w5', 'family': 'uniform', 'parameter': 5, 'kind': 'depthwise', 'node': 'n3',
             'images_with_event': 1, 'image_rate': 0.1, 'output_rate': 0.004}]
    out = {r['kind']: r for r in analysis.kind_rows(rows)}
    assert out['stem']['nodes_with_event'] == 1 and out['stem']['leading_node'] == 'n1'
    assert out['depthwise']['nodes'] == 2 and out['depthwise']['nodes_with_event'] == 1
    assert out['depthwise']['mean_output_rate'] == pytest.approx(0.002)
    assert out['depthwise']['leading_node'] == 'n3'


def test_wrapped_predict():
    argv = ['bash', 'artifacts/agent_orchestration/gpu_run.sh', '--min-free-mib', '3000', f'{certs.ARCHIVE}/run.sh',
            'predict', 'mobilenet_v2-int8-default-b2', 'sat.w17', 'cuda', '0', '1000', '--batch', '8']
    assert wrapped_predict(argv) == ('mobilenet_v2-int8-default-b2', 'sat.w17', 0, 1000)
    assert wrapped_predict(['bash', 'other.sh']) is None


def test_covering_slices_larger_files():
    from tools.accumulator_sweep_mn.load import covering
    files = [(0, 1000, 'a'), (0, 64, 'b'), (64, 128, 'c')]
    assert covering(files, 0, 128) == [(0, 128, 'a', 0)]
    assert covering([(0, 64, 'b'), (64, 128, 'c'), (128, 564, 'd')], 0, 300) == [(0, 64, 'b', 0), (64, 128, 'c', 64), (128, 300, 'd', 128)]
    assert covering([(0, 64, 'b')], 0, 128) is None
