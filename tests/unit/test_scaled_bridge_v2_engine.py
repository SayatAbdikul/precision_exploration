"""Scaled bridge v2: recipe-as-data engine against a rational contract oracle (CPU backend)."""
from fractions import Fraction as Q
import numpy as np
import pytest
from public.inference.reference.arithmetic import format_named, model_c
from tools.scaled_bridge_v2.codebooks import Codebook, NoDyadicGrid, scalar_codebook, quantize
from tools.scaled_bridge_v2.certificates import certify, signed_bits
from tools.scaled_bridge_v2.engine import Engine, numerical

F64 = format_named('fp64_e11m52_accumulator')
F32 = format_named('fp32_e8m23_accumulator')


def r64(value):
    return Q(float(F64.rounded(value)))


def unsigned_book(source, name):
    data = scalar_codebook(source)
    keep = [i for i, u in enumerate(data['units']) if u >= 0]
    return {'id': name, 'bits': data['bits'], 'shift': data['shift'], 'units': [data['units'][i] for i in keep],
            'codes': [data['codes'][i] for i in keep],
            'choose_upper_tie': [data['choose_upper_tie'][i] for i in keep[:-1]]}


def tiny_export(weight_format, activation_format, rng, fused):
    """input -> conv -> relu -> depthwise conv -> add(relu) -> avgpool -> flatten -> linear."""
    wb = Codebook(scalar_codebook(weight_format)); ab = scalar_codebook(activation_format)
    books = {weight_format: wb.data, activation_format: ab, 'unsigned': unsigned_book(activation_format, 'unsigned')}
    def scale():
        return float(np.float32(rng.uniform(.01, .2)))
    def mac(key, shape):
        return {'weight_codebook': weight_format, 'weight_units': key + '_w', 'bias': key + '_b',
                'weight_scales': [scale() for _ in range(shape[0])]}
    arrays = {}
    for key, shape in (('c1', (4, 3, 3, 3)), ('dw', (4, 1, 3, 3)), ('fc', (5, 4))):
        arrays[key + '_w'] = rng.choice(wb.units, size=shape).astype(np.int64)
        arrays[key + '_b'] = rng.normal(size=shape[0]).astype(np.float32).astype(np.float64)
    act = {'codebook': activation_format, 'scale': scale()}
    nodes = [
        {'name': 'x', 'op': 'input', 'inputs': [], 'attrs': {}, 'store': {'codebook': activation_format, 'scale': scale()}},
        {'name': 'c1', 'op': 'conv', 'inputs': ['x'], 'store': None if fused else dict(act), 'mac': mac('c1', (4,)),
         'attrs': {'stride': [2, 2], 'padding': [1, 1], 'dilation': [1, 1], 'groups': 1}},
        {'name': 'r1', 'op': 'relu', 'inputs': ['c1'], 'attrs': {}, 'store': {'codebook': 'unsigned', 'scale': scale()}},
        {'name': 'dw', 'op': 'conv', 'inputs': ['r1'], 'store': None if fused else {'codebook': activation_format, 'scale': scale()},
         'mac': mac('dw', (4,)), 'attrs': {'stride': [1, 1], 'padding': [1, 1], 'dilation': [1, 1], 'groups': 4}},
        {'name': 'sum', 'op': 'add', 'inputs': ['dw', 'r1'], 'attrs': {}, 'store': {'codebook': activation_format, 'scale': scale()}},
        {'name': 'pool', 'op': 'avgpool', 'inputs': ['sum'], 'attrs': {}, 'store': {'codebook': activation_format, 'scale': scale()}},
        {'name': 'flat', 'op': 'flatten', 'inputs': ['pool'], 'attrs': {}, 'store': None},
        {'name': 'fc', 'op': 'linear', 'inputs': ['flat'], 'attrs': {}, 'mac': mac('fc', (5,)),
         'store': {'codebook': activation_format, 'scale': scale()}},
        {'name': 'out', 'op': 'output', 'inputs': ['fc'], 'attrs': {}, 'store': None}]
    return {'schema': 'scaled-bridge-export-2', 'case': 'tiny', 'model': 'tiny', 'codebooks': books, 'nodes': nodes}, arrays


class Oracle:
    """The contract, one scalar at a time, in exact rationals with explicit RNE64 steps."""
    def __init__(self, export, arrays, policy):
        self.ex, self.arrays, self.policy = export, arrays, policy
        self.books = {k: Codebook(v) for k, v in export['codebooks'].items()}

    def store(self, node, value):
        spec = node['store']
        if spec is None:
            return ('raw', value)
        book = self.books[spec['codebook']]; y = r64(value / Q(spec['scale']))
        levels = [Q(int(u), 1 << book.shift) for u in book.units]
        best = min(range(len(levels)), key=lambda i: (abs(levels[i] - y), int(book.codes[i]) & 1, int(book.codes[i])))
        return ('code', int(book.units[best]), book, spec['scale'])

    def real(self, state):
        return state[1] if state[0] == 'raw' else r64(Q(state[1], 1 << state[2].shift) * Q(state[3]))

    def dot(self, pairs, sa, sw):
        if self.policy == 'wide':
            return r64(Q(sum(a * b for a, b in pairs), 1 << (sa + sw)))
        return Q(float(F32.decode(model_c(((Q(a, 1 << sa), Q(b, 1 << sw)) for a, b in pairs), F32))))

    def mac(self, node, source, shape_in):
        w = self.arrays[node['mac']['weight_units']]; bias = self.arrays[node['mac']['bias']]
        wshift = self.books[node['mac']['weight_codebook']].shift
        if node['op'] == 'linear':
            w = w[:, :, None, None]
        attrs = node['attrs'] or {'stride': [1, 1], 'padding': [0, 0], 'dilation': [1, 1], 'groups': 1}
        ci, h, wi = shape_in; co, cg, kh, kw = w.shape; g = attrs['groups']
        oh = (h + 2 * attrs['padding'][0] - kh) // attrs['stride'][0] + 1
        ow = (wi + 2 * attrs['padding'][1] - kw) // attrs['stride'][1] + 1
        out = {}
        for oc in range(co):
            for oy in range(oh):
                for ox in range(ow):
                    pairs = []; book = scale = None
                    for c in range(cg):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = oy * attrs['stride'][0] - attrs['padding'][0] + ky
                                ix = ox * attrs['stride'][1] - attrs['padding'][1] + kx
                                channel = (oc // (co // g)) * cg + c
                                state = source[(channel, 0, 0)] if book is None else None
                                if book is None:
                                    assert state[0] == 'code'; book, scale = state[2], state[3]
                                a = source[(channel, iy, ix)][1] if 0 <= iy < h and 0 <= ix < wi else 0
                                pairs.append((a, int(w[oc, c, ky, kx])))
                    value = r64(self.dot(pairs, book.shift, wshift) * Q(scale))
                    value = r64(value * Q(node['mac']['weight_scales'][oc]))
                    out[(oc, oy, ox)] = self.store(node, r64(value + Q(float(bias[oc]))))
        return out, (co, oh, ow)

    def run(self, image):
        values, shapes = {}, {}
        for node in self.ex['nodes']:
            key, op = node['name'], node['op']
            if op == 'input':
                shapes[key] = image.shape
                values[key] = {idx: self.store(node, Q(float(v))) for idx, v in np.ndenumerate(image)}
            elif op in ('conv', 'linear'):
                values[key], shapes[key] = self.mac(node, values[node['inputs'][0]], shapes[node['inputs'][0]])
            elif op == 'relu':
                src = values[node['inputs'][0]]; shapes[key] = shapes[node['inputs'][0]]
                def relu(s):
                    if s[0] == 'raw':
                        return max(s[1], Q(0))
                    return r64(Q(max(s[1], 0), 1 << s[2].shift) * Q(s[3]))
                values[key] = {i: self.store(node, relu(s)) for i, s in src.items()}
            elif op == 'add':
                a, b = (values[n] for n in node['inputs']); shapes[key] = shapes[node['inputs'][0]]
                values[key] = {i: self.store(node, r64(self.real(a[i]) + self.real(b[i]))) for i in a}
            elif op == 'avgpool':
                src = values[node['inputs'][0]]; c, h, w = shapes[node['inputs'][0]]; shapes[key] = (c, 1, 1)
                values[key] = {}
                for ch in range(c):
                    states = [src[(ch, y, x)] for y in range(h) for x in range(w)]
                    mean = r64(Q(sum(s[1] for s in states), 1 << states[0][2].shift) / (h * w))
                    values[key][(ch, 0, 0)] = self.store(node, r64(mean * Q(states[0][3])))
            elif op == 'flatten':
                values[key] = values[node['inputs'][0]]; shapes[key] = shapes[node['inputs'][0]]
            elif op == 'output':
                src = values[node['inputs'][0]]
                return [float(self.real(src[(c, 0, 0)])) for c in range(shapes[node['inputs'][0]][0])]


@pytest.mark.parametrize('weights,activations,fused', [
    ('int8', 'int8', False), ('int4', 'int8', True), ('fp6_e3m2', 'fp7_e3m3', True),
    ('fp8_e5m2', 'fp8_e5m2', False), ('posit8_es1', 'posit8_es1', True)])
def test_engine_matches_rational_contract_oracle(weights, activations, fused):
    rng = np.random.default_rng(7)
    export, arrays = tiny_export(weights, activations, rng, fused)
    images = rng.normal(size=(3, 3, 6, 6)).astype(np.float32)
    for policy in ('wide', 'control'):
        engine = Engine(export, arrays, 'cpp', policy)
        batch = engine.run(images, trace='full')
        single = [engine.run(images[i:i + 1])[0] for i in range(3)]
        assert [numerical(r) for r in batch] == [numerical(r) for r in single]
        expected = Oracle(export, arrays, policy).run(images[0])
        from tools.scaled_bridge_v2.common import array_hash
        assert batch[0]['output'] == array_hash(np.array([expected], dtype=np.float64))


def test_wide_operand_cases_use_two_limbs_and_other_cases_do_not():
    rng = np.random.default_rng(3)
    for name, expected in (('int8', 32), ('fp8_e4m3fn', 32), ('posit8_es1', 32), ('fp8_e5m2', 64)):
        export, arrays = tiny_export(name, name, rng, False)
        assert {c['kernel_operand_bits'] for c in certify(export, arrays).values()} == {expected}


def test_certificate_intervals_are_attained_and_tight():
    rng = np.random.default_rng(11)
    export, arrays = tiny_export('int6', 'int8', rng, False)
    certs = certify(export, arrays)
    cert = certs['dw']          # producer is a ReLU storing an unsigned codebook
    assert cert['input_range_units'][0] == 0 and cert['structural_input_range_units'][0] == 0
    cert = certs['c1']; w = arrays['c1_w'].reshape(4, -1); lo_a, hi_a = cert['input_range_units']
    for c in range(4):
        best = int(np.where(w[c] > 0, w[c] * hi_a, w[c] * lo_a).sum())
        worst = int(np.where(w[c] > 0, w[c] * lo_a, w[c] * hi_a).sum())
        assert cert['per_channel_range_units'][0][c] == worst and cert['per_channel_range_units'][1][c] == best
        assert cert['per_channel_abs_prefix_units'][c] == max(abs(lo_a), abs(hi_a)) * int(np.abs(w[c]).sum())
    assert signed_bits(-128, 127) == 8 and signed_bits(-129, 0) == 9 and signed_bits(0, 128) == 9 and signed_bits(0, 0) == 1


def test_formats_without_a_dyadic_grid_are_refused():
    for name in ('log4', 'log6', 'log8', 'nf4'):
        with pytest.raises(NoDyadicGrid):
            scalar_codebook(name)
    for name in ('bfp6', 'mxfp8_e4m3'):
        with pytest.raises(ValueError):
            scalar_codebook(name)


def test_engine_fails_closed_outside_the_contract():
    rng = np.random.default_rng(5)
    export, arrays = tiny_export('int8', 'int8', rng, False)
    export['nodes'][2]['store'] = None            # the depthwise MAC would read an unstored tensor
    with pytest.raises(ValueError):
        Engine(export, arrays, 'cpp', 'wide')
    export, arrays = tiny_export('int8', 'int8', rng, False)
    engine = Engine(export, arrays, 'cpp', 'wide')
    with pytest.raises(ValueError):
        engine.run(np.full((1, 3, 6, 6), np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        quantize([1.0], -1.0, Codebook(scalar_codebook('int8')))
