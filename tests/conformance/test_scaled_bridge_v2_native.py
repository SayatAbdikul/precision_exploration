"""Scaled bridge v2 native kernels against Python integer and rational oracles.

The CPU backend always runs. The CUDA backend runs only when
SCALED_BRIDGE_V2_CUDA=1 (set it under the shared GPU lock).
"""
from fractions import Fraction as Q
import os
import numpy as np
import pytest
from public.inference.reference.arithmetic import format_named, model_c
from tools.scaled_bridge_v2.codebooks import Codebook, scalar_codebook
from tools.scaled_bridge_v2.native import Native

BACKENDS = ['cpp'] + (['cuda'] if os.environ.get('SCALED_BRIDGE_V2_CUDA') == '1' else [])
F32 = format_named('fp32_e8m23_accumulator')


def expected(pairs, shift, policy):
    if policy == 'wide':
        return float(Q(sum(a * b for a, b in pairs), 1 << shift))
    return float(F32.decode(model_c(((Q(a), Q(b)) for a, b in pairs), F32))) / (1 << shift)


@pytest.mark.parametrize('backend', BACKENDS)
def test_exact_and_fma_accumulators_match_python_oracles(backend):
    native = Native(backend); rng = np.random.default_rng(20261001)
    cases = [([2**27, 1], [2**26, 1]), ([2**27, 3], [2**26, 1]), ([2**31, 1], [2**31, 1]),
             ([-(2**31), -1], [2**31, 1]), ([2**32 - 256] * 4 + [-(2**32 - 256)] * 4 + [1], [2**32 - 256] * 8 + [-1])]
    for trial in range(150):
        k = int(rng.integers(1, 40))
        if trial % 2:
            a = rng.integers(-7, 8, size=k) * (1 << rng.integers(0, 29, size=k))
            b = rng.integers(-7, 8, size=k) * (1 << rng.integers(0, 29, size=k))
        else:
            a = rng.integers(-2**20, 2**20, size=k); b = rng.integers(-2**20, 2**20, size=k)
        cases.append((a, b))
    for a, b in cases:
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        pairs = [(int(p), int(q)) for p, q in zip(a, b)]
        for policy in ('wide', 'control'):
            got = native.conv(a.reshape(1, -1, 1, 1), b.reshape(1, -1, 1, 1), {}, 5, policy)[0, 0, 0, 0]
            assert np.float64(expected(pairs, 5, policy)).tobytes() == got.tobytes()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', ['int8', 'fp8_e4m3fn', 'fp8_e5m2', 'posit8_es1'])
def test_depthwise_and_grouped_geometry(backend, name):
    native = Native(backend); rng = np.random.default_rng(5); book = Codebook(scalar_codebook(name))
    for groups, stride, padding in ((4, [1, 1], [1, 1]), (2, [2, 1], [1, 0]), (1, [1, 2], [0, 1])):
        x = rng.choice(book.units, size=(2, 4, 5, 6)); w = rng.choice(book.units, size=(4, 4 // groups, 3, 3))
        attrs = {'groups': groups, 'stride': stride, 'padding': padding, 'dilation': [1, 1]}
        for policy in ('wide', 'control'):
            y = native.conv(x, w, attrs, 2 * book.shift, policy)
            for n, oc, oy, ox in [(0, 0, 0, 0), (1, 3, y.shape[2] - 1, y.shape[3] - 1), (1, 2, 1, 1)]:
                cg = 4 // groups; pairs = []
                for c in range(cg):
                    for ky in range(3):
                        for kx in range(3):
                            iy = oy * stride[0] - padding[0] + ky; ix = ox * stride[1] - padding[1] + kx
                            a = int(x[n, (oc // (4 // groups)) * cg + c, iy, ix]) if 0 <= iy < 5 and 0 <= ix < 6 else 0
                            pairs.append((a, int(w[oc, c, ky, kx])))
                assert np.float64(expected(pairs, 2 * book.shift, policy)).tobytes() == y[n, oc, oy, ox].tobytes()


def test_uncertified_inputs_are_refused():
    native = Native('cpp')
    with pytest.raises(ValueError):
        native.conv(np.array([[[[2**32]]]], dtype=np.int64), np.ones((1, 1, 1, 1), dtype=np.int64), {}, 0, 'wide')
    with pytest.raises(ValueError):
        native.conv(np.array([[[[2**25 + 1]]]], dtype=np.int64), np.ones((1, 1, 1, 1), dtype=np.int64), {}, 0, 'control')
    with pytest.raises(ValueError):
        native.conv(np.array([[[[0.5]]]]), np.ones((1, 1, 1, 1), dtype=np.int64), {}, 0, 'wide')
