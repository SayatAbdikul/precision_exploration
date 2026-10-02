"""Scaled bridge v2, contract 2.2 (MobileNet operators): ReLU6, hard-sigmoid, hard-swish, tensor multiply.

A tiny MobileNetV3-like graph (hard-swish stem, depthwise convolution into ReLU6, squeeze-excite with average
pool, ReLU, hard-sigmoid and a broadcast multiply, residual add, classifier with hard-swish) is run by the engine
and checked against four other implementations: a rational contract oracle, the FX/limb reference, the policy
witness (`graph_witness.py`, fed with the same graph in lane L1's raw export format) and, for the operators,
the primitive rational witnesses of `conformance.operator_checks`.
"""
import os
import struct
from fractions import Fraction as Q
import numpy as np
import pytest
import torch
from torch import nn
from test_scaled_bridge_v2_engine import Oracle, r64
from tools.scaled_bridge_v2.b2_adapter import codebook
from tools.scaled_bridge_v2.codebooks import Codebook, scalar_codebook
from tools.scaled_bridge_v2.certificates import certify, certify_products, closure
from tools.scaled_bridge_v2.conformance import operator_checks
from tools.scaled_bridge_v2.engine import Engine, numerical
from tools.scaled_bridge_v2.graph_witness import Witness
from tools.scaled_bridge_v2.reference import FXReference


class TinyV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3, padding=1); self.hs1 = nn.Hardswish()
        self.dw = nn.Conv2d(4, 4, 3, padding=1, groups=4); self.r6 = nn.ReLU6()
        self.pool = nn.AdaptiveAvgPool2d(1); self.fc1 = nn.Conv2d(4, 2, 1); self.r = nn.ReLU()
        self.fc2 = nn.Conv2d(2, 4, 1); self.hsig = nn.Hardsigmoid(); self.proj = nn.Conv2d(4, 4, 1)
        self.gap = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(4, 6); self.hs2 = nn.Hardswish()
        self.ident = nn.Identity(); self.fc3 = nn.Linear(6, 5)

    def forward(self, x):
        a = self.hs1(self.c1(x))
        b = self.r6(self.dw(a))
        s = self.hsig(self.fc2(self.r(self.fc1(self.pool(b)))))
        y = self.proj(s * b) + a
        z = torch.flatten(self.gap(y), 1)
        return self.fc3(self.ident(self.hs2(self.fc(z))))


def b2_entry(base, signedness):
    from tools.experiment_b2.codebook import describe
    return describe(base, signedness)


SHAPES = {'c1': (4, 3, 3, 3), 'dw': (4, 1, 3, 3), 'fc1': (2, 4, 1, 1), 'fc2': (4, 2, 1, 1), 'proj': (4, 4, 1, 1),
          'fc': (6, 4), 'fc3': (5, 6)}
CONV = {'stride': [1, 1], 'padding': [1, 1], 'dilation': [1, 1], 'groups': 1}
POINT = {'stride': [1, 1], 'padding': [0, 0], 'dilation': [1, 1], 'groups': 1}


def tiny_v3(fmt, unsigned, fused, rng):
    """Engine export and the same graph in lane L1's raw format. fused=True leaves the hard-swish and
    hard-sigmoid inputs and one multiply operand unstored (the fused_all pattern)."""
    signed = codebook(b2_entry(fmt, 'signed'))
    positive = codebook(b2_entry(fmt, 'unsigned')) if unsigned else signed
    books = {signed['id']: signed, positive['id']: positive}
    wb = Codebook(signed)
    code_of = {int(u): int(c) for u, c in zip(wb.units, wb.codes)}
    arrays, raw_arrays = {}, {}

    def scale():
        return float(np.float32(rng.uniform(.02, .3)))

    def mac(key):
        w = rng.choice(wb.units, size=SHAPES[key]).astype(np.int64)
        scales = [scale() for _ in range(SHAPES[key][0])]
        bias = rng.normal(scale=.3, size=SHAPES[key][0]).astype(np.float32)
        arrays[key + '_w'] = w; arrays[key + '_b'] = bias.astype(np.float64)
        raw_arrays[key + '.weight_codes'] = np.vectorize(code_of.get)(w).astype(np.int32)
        raw_arrays[key + '.weight_scales'] = np.array(scales, dtype=np.float32)
        raw_arrays[key + '.bias'] = bias
        return {'weight_codebook': signed['id'], 'weight_units': key + '_w', 'bias': key + '_b', 'weight_scales': scales}

    def st(book):
        return {'codebook': book['id'], 'scale': scale()}

    rows = [('x', 'input', [], {}, st(signed)),
            ('c1', 'conv', ['x'], CONV, None if fused else st(signed)),
            ('hs1', 'hardswish', ['c1'], {}, st(signed)),
            ('dw', 'conv', ['hs1'], {**CONV, 'groups': 4}, None),
            ('r6', 'relu6', ['dw'], {}, st(positive)),
            ('pool', 'avgpool', ['r6'], {}, st(positive)),
            ('fc1', 'conv', ['pool'], POINT, None),
            ('r', 'relu', ['fc1'], {}, st(positive)),
            ('fc2', 'conv', ['r'], POINT, None if fused else st(signed)),
            ('hsig', 'hardsigmoid', ['fc2'], {}, None if fused else st(positive)),
            ('mul', 'mul', ['hsig', 'r6'], {}, st(positive)),
            ('proj', 'conv', ['mul'], POINT, st(signed)),
            ('add', 'add', ['proj', 'hs1'], {}, st(signed)),
            ('gap', 'avgpool', ['add'], {}, st(signed)),
            ('flatten', 'flatten', ['gap'], {}, None),
            ('fc', 'linear', ['flatten'], {}, None if fused else st(signed)),
            ('hs2', 'hardswish', ['fc'], {}, st(signed)),
            ('ident', 'identity', ['hs2'], {}, None),
            ('fc3', 'linear', ['ident'], {}, st(signed)),
            ('output', 'output', ['fc3'], {}, None)]
    nodes, raw_nodes = [], []
    for name, op, inputs, attrs, store in rows:
        node = {'name': name, 'op': op, 'inputs': inputs, 'attrs': dict(attrs), 'store': store}
        if op in ('conv', 'linear'):
            node['mac'] = mac(name)
        nodes.append(node)
        boundary = None if op == 'output' else {
            'quantizes': store is not None,
            'codebook': None if store is None else ('U' if store['codebook'] == positive['id'] and unsigned else 'S'),
            'scale_fp32_hex': None if store is None else struct.pack('>f', store['scale']).hex()}
        raw_nodes.append({'name': name, 'op': op, 'inputs': inputs, 'attrs': dict(attrs), 'boundary': boundary})
    export = {'schema': 'scaled-bridge-export-2', 'case': 'tiny-v3', 'model': 'tiny', 'codebooks': books, 'nodes': nodes}
    raw = {'format': fmt, 'codebooks': {'S': {'signedness': 'signed'}, **({'U': {'signedness': 'unsigned'}} if unsigned else {})},
           'nodes': raw_nodes}
    return export, arrays, raw, raw_arrays


class OracleV3(Oracle):
    """The engine-test oracle plus the contract 2.2 operators, one scalar at a time in rationals."""
    def run(self, image):
        values, shapes = {}, {}
        for node in self.ex['nodes']:
            key, op = node['name'], node['op']
            if op in ('input', 'conv', 'linear', 'relu', 'add', 'avgpool', 'flatten'):
                if op == 'input':
                    shapes[key] = image.shape
                    values[key] = {idx: self.store(node, Q(float(v))) for idx, v in np.ndenumerate(image)}
                elif op in ('conv', 'linear'):
                    values[key], shapes[key] = self.mac(node, values[node['inputs'][0]], shapes[node['inputs'][0]])
                else:
                    values[key], shapes[key] = self.generic(node, values, shapes)
                continue
            src = values[node['inputs'][0]]; shapes[key] = shapes[node['inputs'][0]]
            if op in ('identity',):
                values[key] = src
            elif op == 'relu6':
                def relu6(s):
                    r = max(s[1], Q(0)) if s[0] == 'raw' else r64(Q(max(s[1], 0), 1 << s[2].shift) * Q(s[3]))
                    return min(r, Q(6))
                values[key] = {i: self.store(node, relu6(s)) for i, s in src.items()}
            elif op in ('hardsigmoid', 'hardswish'):
                def hard(s):
                    r = self.real(s); t = min(max(r64(r + 3), Q(0)), Q(6))
                    return r64(t / 6) if op == 'hardsigmoid' else r64(r64(r * t) / 6)
                values[key] = {i: self.store(node, hard(s)) for i, s in src.items()}
            elif op == 'mul':
                a, b = (values[n] for n in node['inputs']); shapes[key] = shapes[node['inputs'][1]]
                def product(sa, sb):
                    if sa[0] == 'code' and sb[0] == 'code':
                        p = Q(sa[1] * sb[1], 1 << (sa[2].shift + sb[2].shift))
                        return r64(r64(p * Q(sa[3])) * Q(sb[3]))
                    return r64(self.real(sa) * self.real(sb))
                values[key] = {(c, y, x): self.store(node, product(a[(c, 0, 0)], s)) for (c, y, x), s in b.items()}
            elif op == 'output':
                return [float(self.real(src[(c, 0, 0)])) for c in range(shapes[node['inputs'][0]][0])]
            else:
                raise AssertionError(op)

    def generic(self, node, values, shapes):
        key, op = node['name'], node['op']
        if op == 'relu':
            src = values[node['inputs'][0]]
            def relu(s):
                return max(s[1], Q(0)) if s[0] == 'raw' else r64(Q(max(s[1], 0), 1 << s[2].shift) * Q(s[3]))
            return {i: self.store(node, relu(s)) for i, s in src.items()}, shapes[node['inputs'][0]]
        if op == 'add':
            a, b = (values[n] for n in node['inputs'])
            return {i: self.store(node, r64(self.real(a[i]) + self.real(b[i]))) for i in a}, shapes[node['inputs'][0]]
        if op == 'avgpool':
            src = values[node['inputs'][0]]; c, h, w = shapes[node['inputs'][0]]; out = {}
            for ch in range(c):
                states = [src[(ch, y, x)] for y in range(h) for x in range(w)]
                mean = r64(Q(sum(s[1] for s in states), 1 << states[0][2].shift) / (h * w))
                out[(ch, 0, 0)] = self.store(node, r64(mean * Q(states[0][3])))
            return out, (c, 1, 1)
        if op == 'flatten':
            return values[node['inputs'][0]], shapes[node['inputs'][0]]
        raise AssertionError(op)


# CUDA variants: SCALED_BRIDGE_V2_CUDA=1 (run through artifacts/agent_orchestration/gpu_run.sh).
BACKENDS = ['cpp'] + (['cuda'] if os.environ.get('SCALED_BRIDGE_V2_CUDA') == '1' else [])
CASES = [('int8', True, False), ('int8', True, True), ('fp6_e2m3', False, False), ('fp8_e4m3fn', False, True)]


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fmt,unsigned,fused', CASES)
def test_engine_equals_oracle_reference_and_witness(fmt, unsigned, fused, backend):
    rng = np.random.default_rng(11)
    export, arrays, raw, raw_arrays = tiny_v3(fmt, unsigned, fused, rng)
    graph = torch.fx.symbolic_trace(TinyV3())
    assert [n.name for n in graph.graph.nodes] == [n['name'] for n in export['nodes']]
    witness = Witness(raw, raw_arrays)
    for policy in ('wide', 'control'):
        engine = Engine(export, arrays, backend, policy)
        for trial in range(2):
            image = rng.normal(scale=1.5, size=(1, 3, 6, 6)).astype(np.float32)
            capture = {}
            record = engine.run(image, oracle=True, capture=capture)[0]
            expected = OracleV3(export, arrays, policy).run(image[0].astype(np.float64))
            assert np.array_equal(np.array(expected).view('u8'), capture['output'][0].view('u8'))
            if policy == 'wide':
                assert FXReference(graph, export, arrays).evaluate(image) == numerical(record)
            want = witness.run(image[0], policy)
            assert set(want['codes']) == set(capture) - {'output'}
            for key, codes in want['codes'].items():
                assert np.array_equal(codes, capture[key][0]), key
            assert want['output'].tobytes() == capture['output'][0].tobytes()
    diagnostics = record['layers']
    assert diagnostics['dw']['diagnostics'] == {'unstored': True}
    assert (diagnostics['hsig']['diagnostics'] == {'unstored': True}) == fused


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('policy', ['sat.struct-0', 'sat.struct-2', 'sat.abs-1', 'sat.w9', 'fp16', 'fp16.x-6', 'f21', 'f21.x-3'])
def test_policy_witness_on_mobilenet_operators(policy, backend):
    rng = np.random.default_rng(5)
    export, arrays, raw, raw_arrays = tiny_v3('int8', True, False, rng)
    engine = Engine(export, arrays, backend, policy); witness = Witness(raw, raw_arrays)
    for trial in range(3):
        image = rng.normal(scale=1.5, size=(1, 3, 6, 6)).astype(np.float32)
        capture = {}
        record = engine.run(image, oracle=True, capture=capture)[0]
        want = witness.run(image[0], policy)
        if want['failure'] is not None:
            assert record['failure'] is not None and all(record['failure'][k] == v for k, v in want['failure'].items())
            continue
        assert record['failure'] is None
        for key, codes in want['codes'].items():
            assert np.array_equal(codes, capture[key][0]), key
        assert want['output'].tobytes() == capture['output'][0].tobytes()
        for key, stats in want['acc'].items():
            assert all(record['accumulator'][key][k] == v for k, v in stats.items()), (key, stats)
        if policy == 'sat.struct-0':
            assert not any(v['saturated_elements'] for v in record['accumulator'].values())


def test_closure_and_product_certificates():
    rng = np.random.default_rng(3)
    export, arrays, _, _ = tiny_v3('int8', True, False, rng)
    certs = certify(export, arrays); products = certify_products(export); bounds = closure(export)
    for c in certs.values():
        lo, hi = c['closure_input_range_units']; slo, shi = c['structural_input_range_units']
        assert slo <= lo <= 0 <= hi <= shi and c['signed_bits_closure'] <= c['signed_bits_structural']
    # hard-swish output >= -0.375: the depthwise input range stops near -0.375 / scale, far above -128
    scale = next(n for n in export['nodes'] if n['name'] == 'hs1')['store']['scale']
    assert certs['dw']['closure_input_range_units'][0] == int(np.rint(-0.375 / scale)) > -128
    assert certs['dw']['groups'] == 4 and certs['dw']['K'] == 9
    assert certs['proj']['closure_input_range_units'][0] == 0           # mul of two non-negative tensors
    assert products['mul']['max_abs_product_units'] == 255 * 255 and products['mul']['signed_bits_product'] == 17
    assert bounds['hsig'][0] == 'units' and bounds['hsig'][1] >= 0


def test_operator_primitive_witnesses():
    rng = np.random.default_rng(1)
    signed = Codebook(codebook(b2_entry('int6', 'signed'))); unsigned = Codebook(codebook(b2_entry('int6', 'unsigned')))
    counts = operator_checks(signed, [signed, unsigned], rng)
    assert counts['unary_stored'] == 3 * 5 * len(signed.units) and counts['mul_stored'] > 2 * 64 * 64
    assert counts['refusals'] == 6
    fp = Codebook(scalar_codebook('fp6_e2m3'))
    assert operator_checks(fp, [fp], rng)['mul_unstored'] == 8 * 35


def test_fail_closed_operators():
    rng = np.random.default_rng(2)
    export, arrays, _, _ = tiny_v3('int8', True, False, rng)
    bad = dict(export, nodes=[dict(n, inputs=['hsig']) if n['name'] == 'mul' else n for n in export['nodes']])
    with pytest.raises(ValueError):
        Engine(bad, arrays, 'cpp', 'wide')
    bad = dict(export, nodes=[dict(n, attrs={'inplace': True}) if n['name'] == 'r6' else n for n in export['nodes']])
    with pytest.raises(ValueError):
        Engine(bad, arrays, 'cpp', 'wide')
    engine = Engine(export, arrays, 'cpp', 'wide')
    engine.certificates['proj']['closure_input_range_units'] = [0, 0]       # a wrong closure must fail closed
    with pytest.raises(ValueError):
        engine.run(rng.normal(scale=3, size=(1, 3, 6, 6)).astype(np.float32))
