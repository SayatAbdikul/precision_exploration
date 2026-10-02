"""Scaled bridge v2, repaired-recipe (B2) graphs: adapter codebooks, whole-graph reference, capture hook."""
import numpy as np
import pytest
import torch
from torch import nn
from tools.scaled_bridge_v2.accumulators import resolve
from tools.scaled_bridge_v2.b2_adapter import codebook, fp32
from tools.scaled_bridge_v2.codebooks import Codebook, scalar_codebook
from tools.scaled_bridge_v2.engine import Engine, numerical
from tools.scaled_bridge_v2.reference import FXReference


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=1, padding=1); self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1); self.conv2 = nn.Conv2d(4, 4, 3, padding=1); self.relu2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(4, 5)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        y = self.conv2(x)
        x = self.relu2(y + x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def b2_entry(base, signedness):
    from tools.experiment_b2.codebook import describe
    return describe(base, signedness)


def fused_export(weight_format, signed, activation, rng):
    """The B2 default boundary pattern: conv and add fused into ReLU, a max-pool that forwards codes."""
    wb = Codebook(scalar_codebook(weight_format))
    books = {weight_format: wb.data, signed['id']: signed, activation['id']: activation}
    def scale():
        return float(np.float32(rng.uniform(.01, .2)))
    arrays = {}
    def mac(key, shape):
        arrays[key + '_w'] = rng.choice(wb.units, size=shape).astype(np.int64)
        arrays[key + '_b'] = rng.normal(size=shape[0]).astype(np.float32).astype(np.float64)
        return {'weight_codebook': weight_format, 'weight_units': key + '_w', 'bias': key + '_b',
                'weight_scales': [scale() for _ in range(shape[0])]}
    conv = {'stride': [1, 1], 'padding': [1, 1], 'dilation': [1, 1], 'groups': 1}
    pool = {'kernel_size': [3, 3], 'stride': [2, 2], 'padding': [1, 1], 'dilation': [1, 1]}
    def act():
        return {'codebook': activation['id'], 'scale': scale()}
    nodes = [
        {'name': 'x', 'op': 'input', 'inputs': [], 'attrs': {}, 'store': {'codebook': signed['id'], 'scale': scale()}},
        {'name': 'conv1', 'op': 'conv', 'inputs': ['x'], 'attrs': conv, 'store': None, 'mac': mac('conv1', (4, 3, 3, 3))},
        {'name': 'relu', 'op': 'relu', 'inputs': ['conv1'], 'attrs': {}, 'store': act()},
        {'name': 'maxpool', 'op': 'maxpool', 'inputs': ['relu'], 'attrs': pool, 'store': None},
        {'name': 'conv2', 'op': 'conv', 'inputs': ['maxpool'], 'attrs': conv, 'mac': mac('conv2', (4, 4, 3, 3)),
         'store': {'codebook': signed['id'], 'scale': scale()}},
        {'name': 'add', 'op': 'add', 'inputs': ['conv2', 'maxpool'], 'attrs': {}, 'store': None},
        {'name': 'relu2', 'op': 'relu', 'inputs': ['add'], 'attrs': {}, 'store': act()},
        {'name': 'avgpool', 'op': 'avgpool', 'inputs': ['relu2'], 'attrs': {}, 'store': act()},
        {'name': 'flatten', 'op': 'flatten', 'inputs': ['avgpool'], 'attrs': {}, 'store': None},
        {'name': 'fc', 'op': 'linear', 'inputs': ['flatten'], 'attrs': {}, 'mac': mac('fc', (5, 4)),
         'store': {'codebook': signed['id'], 'scale': scale()}},
        {'name': 'output', 'op': 'output', 'inputs': ['fc'], 'attrs': {}, 'store': None}]
    return {'schema': 'scaled-bridge-export-2', 'case': 'tiny-b2', 'model': 'tiny', 'codebooks': books, 'nodes': nodes}, arrays


@pytest.mark.parametrize('base,unsigned', [('int8', True), ('int6', True), ('fp6_e2m3', False), ('fp8_e4m3fn', False)])
def test_reference_equals_engine_on_fused_graph(base, unsigned):
    rng = np.random.default_rng(7)
    signed = codebook(b2_entry(base, 'signed'))
    activation = codebook(b2_entry(base, 'unsigned')) if unsigned else signed
    export, arrays = fused_export(base, signed, activation, rng)
    graph = torch.fx.symbolic_trace(Tiny())
    assert [n.name for n in graph.graph.nodes] == [n['name'] for n in export['nodes']]
    engine = Engine(export, arrays, 'cpp', 'wide')
    for _ in range(3):
        inputs = rng.normal(size=(1, 3, 9, 10)).astype(np.float32)
        capture = {}
        record = engine.run(inputs, oracle=True, capture=capture)[0]
        witness = FXReference(graph, export, arrays).evaluate(inputs)
        assert witness == numerical(record)
        assert record['layers']['conv1']['codes'] is None and record['layers']['add']['diagnostics'] == {'unstored': True}
        assert record['layers']['maxpool']['diagnostics'] == {'code_passthrough': True}
        assert set(capture) == {'x', 'relu', 'conv2', 'relu2', 'avgpool', 'fc', 'output'}
        assert capture['relu'].min() >= 0 and capture['output'].shape == (1, 5)
    cert = engine.certificates['conv2']
    assert cert['input_codebook'] == activation['id'] and cert['structural_input_range_units'][0] == 0


def test_adapter_codebooks():
    unsigned = codebook(b2_entry('int8', 'unsigned'))
    assert unsigned['id'] == 'int8.unsigned' and unsigned['units'] == list(range(256)) and unsigned['shift'] == 0
    assert unsigned['codes'] == list(range(256)) and unsigned['choose_upper_tie'][:4] == [False, True, False, True]
    assert codebook(b2_entry('fp7_e3m3', 'signed')) == scalar_codebook('fp7_e3m3')
    broken = dict(b2_entry('int6', 'unsigned')); broken['choose_upper_tie'] = [not t for t in broken['choose_upper_tie']]
    with pytest.raises(ValueError):
        codebook(broken)
    broken = dict(b2_entry('int6', 'signed')); broken['choose_upper_tie'] = [not t for t in broken['choose_upper_tie']]
    with pytest.raises(ValueError):
        codebook(broken)
    broken = dict(b2_entry('int6', 'signed')); broken['levels'] = [v * 2 for v in broken['levels']]
    with pytest.raises(ValueError):
        codebook(broken)
    assert fp32('3caa4a58') == float(np.float32(0.020787402987480164))


def test_one_name_per_float_accumulator():
    assert resolve('fp16').scale_exponent == 0 and resolve('f21.x-3').scale_exponent == -3
    for name in ('fp16.x0', 'f21.x0', 'fp16.x-0'):
        with pytest.raises(ValueError):
            resolve(name)
