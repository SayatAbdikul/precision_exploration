"""Accumulator policies of scaled bridge v2 against references written from the definitions.

The float reference below is written from scratch (exact Fractions, explicit
round-to-nearest-even on the format grid); binary16 is additionally compared
with NumPy's IEEE float16. Neither shares code with the kernels or with
tools/scaled_bridge_v2/oracles.py, which is cross-checked here too.
"""
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from tools.scaled_bridge_v2.accumulators import resolve
from tools.scaled_bridge_v2.common import array_hash
from tools.scaled_bridge_v2.engine import Engine, numerical, window_max
from tools.scaled_bridge_v2.native import Native
from tools.scaled_bridge_v2 import oracles

_spec = importlib.util.spec_from_file_location('v2_engine_tests', Path(__file__).with_name('test_scaled_bridge_v2_engine.py'))
base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)
FORMATS = {'fp16': (11, -24, 15), 'f21': (13, -138, 127)}
INF = float('inf')


def rne(value, P, emin, emax):
    """Round an exact rational to the float format; returns a Fraction or +-inf."""
    if value == 0:
        return Q(0)
    sign = -1 if value < 0 else 1; mag = abs(value)
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if Q(2) ** e > mag:
        e -= 1
    assert Q(2) ** e <= mag < Q(2) ** (e + 1)
    grid = Q(2) ** max(e - P + 1, emin)
    n, rem = divmod(mag, grid)
    if rem * 2 > grid or (rem * 2 == grid and n % 2):
        n += 1
    result = n * grid
    return sign * INF if result >= Q(2) ** (emax + 1) else sign * result


def float_reference(a, b, shift, name, x):
    P, emin, emax = FORMATS[name]; state = Q(0); bad = 0
    for p, q in zip(a, b):
        if state in (INF, -INF):
            bad += 1; continue
        state = rne(state + Q(int(p) * int(q)) * Q(2) ** (x - shift), P, emin, emax)
        bad += state in (INF, -INF)
    return (state if state in (INF, -INF) else float(state / Q(2) ** x)), bad


def kernel(a, b, shift, spec, native=[]):
    if not native:
        native.append(Native('cpp'))
    x = np.array(a, dtype=np.int64).reshape(1, -1, 1, 1); w = np.array(b, dtype=np.int64).reshape(1, -1, 1, 1)
    y, e = native[0].conv_stat(x, w, {}, shift, spec)
    return float(y.flat[0]), (int(e.flat[0]) if e is not None else None)


def sequences(rng, count):
    for trial in range(count):
        k = int(rng.integers(1, 70)); mode = trial % 5
        if mode == 0:
            a = rng.integers(-255, 256, size=k); b = rng.integers(-128, 128, size=k)
        elif mode == 1:
            a = rng.integers(-15, 16, size=k) << rng.integers(0, 14, size=k); b = rng.integers(-15, 16, size=k) << rng.integers(0, 14, size=k)
        elif mode == 2:
            a = rng.integers(-2**32 + 1, 2**32, size=k); b = rng.integers(-2**32 + 1, 2**32, size=k)
        elif mode == 3:
            a = rng.integers(-2, 3, size=k); b = rng.integers(-2, 3, size=k)
        else:
            half = rng.integers(-2**24, 2**24, size=(k + 1) // 2); a = np.concatenate([half, -half])[:k]; b = np.full(k, 3)
        yield [int(v) for v in a], [int(v) for v in b], int(rng.choice([0, 0, 5, 12, 24, 30, 61, 100, 126]))


@pytest.mark.parametrize('spec', ['fp16', 'f21', 'fp16.x-12', 'f21.x-20', 'fp16.x7', 'f21.x64'])
def test_float_accumulators_match_a_from_scratch_rational_reference(spec):
    policy = resolve(spec); rng = np.random.default_rng(7); seen_inf = seen_sub = 0
    for a, b, shift in sequences(rng, 400):
        value, bad = float_reference(a, b, shift, policy.float_format, policy.scale_exponent)
        got = kernel(a, b, shift, spec)
        assert (np.float64(value).tobytes(), bad) == (np.float64(got[0]).tobytes(), got[1]), (a, b, shift)
        assert oracles.expected_dot(list(zip(a, b)), shift, 0, spec) == got
        seen_inf += abs(value) == INF
        seen_sub += value != 0 and abs(value) != INF and abs(Q(value)) * Q(2) ** policy.scale_exponent < Q(2) ** (FORMATS[policy.float_format][1] + FORMATS[policy.float_format][0] - 1)
    assert seen_inf or spec in ('fp16.x-12', 'f21', 'f21.x-20'), 'no overflow exercised'
    assert seen_sub or spec not in ('fp16', 'f21.x-20'), 'no subnormal result exercised'


def test_binary16_accumulator_equals_ieee_float16_arithmetic():
    rng = np.random.default_rng(11); overflow = 0
    with np.errstate(over='ignore'):
        for trial in range(500):
            k = int(rng.integers(1, 60)); shift = int(rng.choice([0, 3, 10, 16, 22, 28]))
            a = rng.integers(-255, 256, size=k); b = rng.integers(-128, 128, size=k)
            state = np.float16(0)
            for p, q in zip(a, b):
                state = np.float16(np.float64(state) + np.ldexp(np.float64(int(p) * int(q)), -shift))   # exact sum, one RNE
            got, bad = kernel(a, b, shift, 'fp16')
            assert np.float64(state).tobytes() == np.float64(got + 0.0).tobytes() or (state == 0 and got == 0)
            overflow += bool(np.isinf(state))
    assert overflow > 20


def test_float_special_cases():
    assert kernel([65504, 15], [1, 1], 0, 'fp16') == (65504.0, 0)
    assert kernel([65504, 16, -65504], [1, 1, 1], 0, 'fp16') == (INF, 2)        # tie at the overflow threshold
    assert kernel([-65504, -16, 1], [1, 1, 1], 0, 'fp16') == (-INF, 2)
    assert kernel([32], [1], 30, 'fp16') == (0.0, 0)                            # 2^-25: tie to even -> 0
    assert kernel([33], [1], 30, 'fp16') == (2.0 ** -24, 0)
    assert kernel([96], [1], 30, 'fp16') == (2.0 ** -23, 0)
    assert kernel([2047, 1], [32, 16], 0, 'fp16') == (65520.0 if False else INF, 1)
    assert kernel([4095, 4095], [1, -1], 0, 'f21') == (0.0, 0)
    assert kernel([8191, 1], [1, 1], 0, 'f21') == (8192.0, 0)
    assert kernel([8192, 1], [1, 1], 0, 'f21') == (8192.0, 0)                   # 13-bit significand: tie to even
    assert kernel([8192, 3], [1, 1], 0, 'f21') == (8196.0, 0)
    assert kernel([8192, 1], [1, 1], 0, 'fp16') == (8192.0, 0)
    assert kernel([8192, 4, 4], [1, 1, 1], 0, 'fp16') == (8192.0, 0)            # sequential rounding loses both adds
    assert kernel([8192, 8], [1, 1], 0, 'fp16') == (8200.0, 0)


@pytest.mark.parametrize('width', [2, 5, 8, 13, 24, 32, 47, 63])
def test_saturating_accumulator_matches_integer_reference(width):
    rng = np.random.default_rng(width); hi = (1 << (width - 1)) - 1; lo = -hi - 1; clamps = 0
    for a, b, shift in sequences(rng, 300):
        total = up = down = 0
        for p, q in zip(a, b):
            total += p * q
            if total > hi:
                total, up = hi, up + 1
            elif total < lo:
                total, down = lo, down + 1
        got = kernel(a, b, shift, f'sat.w{width}')
        assert got == (float(Q(total, 1 << shift)), min(up, 65535) | (min(down, 65535) << 16))
        assert lo <= Q(got[0]) * (1 << shift) <= float(hi)
        exact = sum(p * q for p, q in zip(a, b)); clamps += bool(up or down)
        prefixes = np.cumsum([p * q for p, q in zip(a, b)], dtype=object)
        if lo <= min(prefixes) and max(prefixes) <= hi:
            assert (up, down) == (0, 0) and total == exact
            if abs(exact) < 2**62:
                assert got[0] == kernel(a, b, shift, 'wide')[0]
    assert clamps or width == 63


def test_policy_names_are_canonical_and_fail_closed():
    assert resolve('sat.w24').width() == 24 and resolve('fp16.x-12').scale_exponent == -12
    cert = {'signed_bits_absolute': 26, 'signed_bits_structural': 25}
    assert resolve('sat.abs-2').width(cert) == 24 and resolve('sat.struct-0').width(cert) == 25
    assert resolve('sat.struct-40').width(cert) == 2
    for bad in ('sat.w1', 'sat.w64', 'sat.w024', 'sat', 'fp16.x', 'fp16.x+3', 'f21.x65', 'fp32', 'sat.abs--1', 'wide '):
        with pytest.raises(ValueError):
            resolve(bad)
    with pytest.raises(ValueError):
        resolve('sat.abs-1').width()


def test_engine_policies_on_a_small_graph():
    rng = np.random.default_rng(3)
    export, arrays = base.tiny_export('int8', 'int8', rng, True)
    images = rng.normal(size=(3, 3, 9, 9)) * np.array([0.05, 1.0, 30.0]).reshape(3, 1, 1, 1)
    wide = Engine(export, arrays, 'cpp', 'wide').run(images, oracle=True)
    for spec in ('sat.abs-0', 'sat.struct-0'):
        rows = Engine(export, arrays, 'cpp', spec).run(images, oracle=True)
        for r, w in zip(rows, wide):
            assert r['layers'] == w['layers'] and r['output'] == w['output'] and r['failure'] is None
            assert all(v['saturated_elements'] == 0 for v in r['accumulator'].values())
    narrow = Engine(export, arrays, 'cpp', 'sat.w9').run(images, oracle=True)
    assert sum(v['saturated_elements'] for r in narrow for v in r['accumulator'].values()) > 0
    assert all(v['width'] == 9 and v['high_clamps'] + v['low_clamps'] >= v['saturated_elements']
               for r in narrow for v in r['accumulator'].values())
    assert any(r['layers'] != w['layers'] for r, w in zip(narrow, wide))
    engine = Engine(export, arrays, 'cpp', 'fp16')
    rows = engine.run(images, oracle=True)
    failed = [r['failure'] is not None for r in rows]
    assert any(failed) and not all(failed), failed
    for i, r in enumerate(rows):
        single = engine.run(images[i:i + 1])[0]
        assert numerical(single) == numerical(r)                    # batch invariance, including a failed neighbour
        if r['failure'] is not None:
            assert r['top5'] is None and r['output'] is None and r['failure']['nonfinite_elements'] > 0
            assert r['failure']['node'] in r['accumulator'] and list(r['layers'])[-1] == r['failure']['node']
            assert r['failure']['positive_infinite'] + r['failure']['negative_infinite'] == r['failure']['nonfinite_elements']
        else:
            assert len(r['top5']) == 5 and all(v['nonfinite_elements'] == 0 for v in r['accumulator'].values())
    scaled = Engine(export, arrays, 'cpp', 'fp16.x-12').run(images, oracle=True)
    assert all(r['failure'] is None for r in scaled)
    f21 = Engine(export, arrays, 'cpp', 'f21').run(images, oracle=True)
    assert all(r['failure'] is None for r in f21)
    assert 'accumulator' not in wide[0] and 'failure' not in wide[0]


def pooled_graph(rng):
    book = base.scalar_codebook('int6'); unsigned = base.unsigned_book('int6', 'u6')
    arrays = {'c_w': rng.choice(np.array(book['units']), size=(2, 3, 2, 2)).astype(np.int64), 'c_b': rng.normal(size=2)}
    pool = {'kernel_size': [3, 3], 'stride': [2, 2], 'padding': [1, 1], 'dilation': [1, 1]}
    nodes = [{'name': 'x', 'op': 'input', 'inputs': [], 'attrs': {}, 'store': None},
             {'name': 'r', 'op': 'relu', 'inputs': ['x'], 'attrs': {}, 'store': {'codebook': 'u6', 'scale': 0.031}},
             {'name': 'mp', 'op': 'maxpool', 'inputs': ['r'], 'attrs': pool, 'store': None},
             {'name': 'c', 'op': 'conv', 'inputs': ['mp'], 'attrs': {'stride': [1, 1], 'padding': [0, 0], 'dilation': [1, 1], 'groups': 1},
              'store': None, 'mac': {'weight_codebook': 'int6', 'weight_units': 'c_w', 'bias': 'c_b', 'weight_scales': [0.5, 0.25]}},
             {'name': 'f', 'op': 'flatten', 'inputs': ['c'], 'attrs': {}, 'store': None},
             {'name': 'out', 'op': 'output', 'inputs': ['f'], 'attrs': {}, 'store': None}]
    return {'schema': 'scaled-bridge-export-2', 'case': 'pooled', 'model': 'tiny', 'codebooks': {'int6': book, 'u6': unsigned},
            'nodes': nodes}, arrays, pool


def test_unquantised_maxpool_forwards_codes_and_unstored_values_are_canonical():
    rng = np.random.default_rng(5); export, arrays, pool = pooled_graph(rng)
    images = rng.normal(size=(2, 3, 9, 9)); images[0, 0, 0, 0] = -0.0
    engine = Engine(export, arrays, 'cpp', 'wide')
    rows = engine.run(images, oracle=True)
    cert = engine.certificates['c']
    assert cert['input_codebook'] == 'u6' and cert['structural_input_range_units'][0] == 0
    levels = np.array(export['codebooks']['u6']['units'])
    y = np.maximum(images, 0.0) / 0.031
    units = levels[np.abs(y[..., None] - levels).argmin(-1)]            # no exact ties for random inputs
    pooled = window_max(units, pool, np.iinfo(np.int64).min)
    dot = Native('cpp').conv(pooled, arrays['c_w'], {}, 0, 'wide')
    raw = (dot * 0.031) * np.array([0.5, 0.25]).reshape(1, 2, 1, 1) + arrays['c_b'].reshape(1, 2, 1, 1)
    for i, r in enumerate(rows):
        assert r['layers']['mp']['codes'] == array_hash(np.array(export['codebooks']['u6']['codes'], dtype=np.uint8)[
            np.searchsorted(levels, pooled[i:i + 1])])
        assert r['layers']['mp']['diagnostics'] == {'code_passthrough': True}
        assert r['layers']['c']['mac'] == array_hash(dot[i:i + 1]) and r['layers']['c']['raw'] == array_hash(raw[i:i + 1] + 0.0)
        assert r['output'] == array_hash((raw[i:i + 1] + 0.0).reshape(1, -1))
    export['nodes'][0]['store'] = None
    assert Engine(export, arrays, 'cpp', 'wide').run(-np.abs(images) * 0.0)[0]['layers']['x']['raw'] == array_hash(np.zeros((1, 3, 9, 9)))


def test_engine_rejects_undefined_attributes_and_shapes():
    rng = np.random.default_rng(6)
    for change in ({'ceil_mode': True}, {'padding': [2, 2]}, {'dilation': [2, 2]}):
        export, arrays, pool = pooled_graph(rng)
        export['nodes'][2]['attrs'] = {**pool, **change}
        with pytest.raises(ValueError):
            Engine(export, arrays, 'cpp', 'wide')
    export, arrays, _ = pooled_graph(rng); export['nodes'][3]['attrs']['padding_mode'] = 'reflect'
    with pytest.raises(ValueError):
        Engine(export, arrays, 'cpp', 'wide')
    export, arrays, _ = pooled_graph(rng); arrays['c_b'] = arrays['c_b'][:1]
    with pytest.raises(ValueError):
        Engine(export, arrays, 'cpp', 'wide')
    export, arrays, _ = pooled_graph(rng); export['nodes'][1]['op'] = 'sigmoid'     # hardswish is admitted from contract 2.2
    with pytest.raises(ValueError):
        Engine(export, arrays, 'cpp', 'wide')
    export, arrays, _ = pooled_graph(rng); export['nodes'][4]['attrs'] = {'start_dim': 2}
    with pytest.raises(ValueError):
        Engine(export, arrays, 'cpp', 'wide')
