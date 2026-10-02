"""Stored test vectors of the existing contract 2.0/2.1 oracles (tools/scaled_bridge_v2/oracles.py).

`python -m tools.scaled_bridge_ext.vectors` writes artifacts/scaled_bridge_ext_v1/vectors/existing-policies-v1.json.gz
once (seeded; an existing file must hold the same cases). Each case: policy, shift_a, shift_w, a, w, the oracle's
binary64 value as hex and its event word. The tests check that the 2.3 reference reproduces every stored value and
word, and that the live oracle still does.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'artifacts/scaled_bridge_ext_v1/vectors/existing-policies-v1.json.gz'
SEED = 20261002
ORACLE_SOURCES = ('tools/scaled_bridge_v2/oracles.py', 'tools/scaled_bridge_v2/accumulators.py',
                  'public/inference/reference/arithmetic.py')


def _cases():
    rng = np.random.default_rng(SEED)
    cases = []

    def add(spec, a, w, sa=0, sw=0):
        cases.append([spec, int(sa), int(sw), [int(v) for v in a], [int(v) for v in w]])

    # Saturating: end points, exact landing, recovery, both ends, 64-bit operands (as the 2.1 witnesses).
    for width in (2, 3, 8, 16, 24, 32, 48, 63):
        hi = (1 << (width - 1)) - 1; spec = f'sat.w{width}'
        A, B = ([hi], [1]) if width <= 32 else ([2**31 - 1, (1 << (width - 32)) - 1], [1 << (width - 32), 1])
        N = [-v for v in A]
        for sa in (0, 3):
            add(spec, A, B, sa); add(spec, A + [1], B + [1], sa); add(spec, A + [1, -1], B + [1, 1], sa)
            add(spec, N + [-1], B + [1], sa); add(spec, N + [-1, -1, 3], B + [1, 1, 1], sa)
            add(spec, A * 3 + N * 5 + A * 4, B * 12, sa)
        for _ in range(12):
            k = int(rng.integers(1, 60)); span = int(rng.integers(1, max(2, min(width + 2, 31))))
            add(spec, rng.integers(-(1 << span), (1 << span) + 1, size=k), rng.integers(-(1 << span), (1 << span) + 1, size=k),
                int(rng.integers(0, 12)), int(rng.integers(0, 8)))
    # Float accumulators: overflow, sticky infinity, subnormal ties, scale exponents, cancellation.
    add('fp16', [65504], [1]); add('fp16', [65504, 15], [1, 1]); add('fp16', [65504, 16, -65504, 7], [1] * 4)
    add('fp16', [-65504, -16, 65504], [1] * 3); add('fp16', [255, 255], [255, 255]); add('fp16.x-12', [255, 255], [255, 255])
    for a in ([1], [32], [33], [96], [160], [64] * 10, [32, 32], [33, 31], [33, 32], [33, -33], [65535], [2**16 - 16, 16, 1]):
        add('fp16', a, [1] * len(a), 30)
    for a in ([1], [2**7], [2**7 + 1], [3 * 2**7], [2**20 - 1, 1, 1], [2**12 + 1] * 9, [5, -5], [2**8, -2**7, 2**7 - 1]):
        add('f21.x-20', a, [1] * len(a), 126)
    add('f21.x64', [2**32 - 1] * 2, [2**32 - 1] * 2); add('f21.x64', [-(2**32 - 1)] * 3, [2**32 - 1] * 3)
    add('f21', [2**32 - 1, 1, -(2**32 - 1), 4097, 4097, -1], [2**32 - 1, 1, 2**32 - 1, 1, 1, 1])
    specs = ('fp16', 'f21', 'fp16.x-12', 'f21.x7', 'fp16.x9', 'f21.x-20')
    for trial in range(240):
        k = int(rng.integers(1, 60)); mode = trial % 6; spec = specs[(trial // 6) % 6]
        if mode == 0:
            a = rng.integers(-255, 256, size=k); b = rng.integers(-128, 128, size=k)
        elif mode == 1:
            a = rng.integers(-15, 16, size=k) << rng.integers(0, 12, size=k); b = rng.integers(-15, 16, size=k) << rng.integers(0, 12, size=k)
        elif mode == 2:
            a = rng.integers(-2**31 + 1, 2**31, size=k); b = rng.integers(-2**31 + 1, 2**31, size=k)
        elif mode == 3:
            a = rng.integers(-3, 4, size=k); b = rng.integers(-3, 4, size=k)
        elif mode == 4:
            half = rng.integers(-2**20, 2**20, size=(k + 1) // 2); a = np.concatenate([half, -half])[:k]; b = np.full(k, int(rng.integers(1, 2**10)))
        else:
            a = rng.integers(-2**12, 2**12, size=k); b = rng.integers(-2**12, 2**12, size=k)
        add(spec, a, b, int(rng.choice([0, 0, 6, 12, 20, 30, 44, 60])), 0)
    # Exact and binary32 control arms (operands exact in binary32).
    for trial in range(120):
        k = int(rng.integers(1, 80)); span = int(rng.choice([3, 8, 12, 16, 23]))
        a = rng.integers(-(1 << span), 1 << span, size=k); b = rng.integers(-(1 << span), 1 << span, size=k)
        add('wide' if trial % 2 else 'control', a, b, int(rng.integers(0, 10)), int(rng.integers(0, 10)))
    for trial in range(20):
        k = int(rng.integers(1, 40))
        add('wide', rng.integers(-(2**32) + 1, 2**32, size=k), rng.integers(-(2**32) + 1, 2**32, size=k), int(rng.integers(0, 60)))
    return cases


def generate():
    from tools.scaled_bridge_v2.oracles import expected_dot
    out = []
    for spec, sa, sw, a, w in _cases():
        value, word = expected_dot(list(zip(a, w)), sa, sw, spec)
        out.append({'policy': spec, 'shift_a': sa, 'shift_w': sw, 'a': a, 'w': w,
                    'value_hex': float(value).hex(), 'word': word})
    return out


def oracle_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ORACLE_SOURCES}


def load():
    with gzip.open(PATH, 'rt') as f:
        return json.load(f)


def write():
    payload = {'schema': 'scaled-bridge-ext-vectors-1', 'seed': SEED, 'oracle_sources_sha256': oracle_hashes(),
               'origin': 'tools/scaled_bridge_v2/oracles.py expected_dot (contract 2.0/2.1 policies)',
               'cases': generate()}
    if PATH.exists():
        old = load()
        if old['cases'] != payload['cases']:
            raise ValueError(f'{PATH} exists with different cases; it is written once')
        return PATH
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(PATH, 'wb', mtime=0) as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode())
    return PATH


# -- supplement 1 (2026-10-02, review 1): relative saturating widths and event counters at the 65,535 cap ---------
SUPPLEMENT_PATH = ROOT / 'artifacts/scaled_bridge_ext_v1/vectors/existing-policies-v1-supplement-1.json.gz'


def _supplement_cases():
    """(policy, shift_a, shift_w, a, w, certificate fields) for 2.1 names the first file lacks."""
    rng = np.random.default_rng(SEED + 1)
    cases = []
    for trial in range(48):
        k = int(rng.integers(1, 50)); span = int(rng.integers(2, 12))
        a = [int(v) for v in rng.integers(-(1 << span), (1 << span) + 1, size=k)]
        w = [int(v) for v in rng.integers(-(1 << span), (1 << span) + 1, size=k)]
        absolute = max(sum(abs(x) * abs(y) for x, y in zip(a, w)), 1)
        cert = {'signed_bits_absolute': absolute.bit_length() + 1 + int(rng.integers(0, 3)),
                'signed_bits_structural': absolute.bit_length() + int(rng.integers(-2, 2))}
        d = int(rng.integers(0, 12))
        spec = ('sat.abs-' if trial % 2 else 'sat.struct-') + str(d)
        cases.append([spec, int(rng.integers(0, 10)), int(rng.integers(0, 6)), a, w, cert])
    n = 70000                                            # more clamps than the 16-bit counters hold
    for spec, a in (('sat.w2', [1] * n), ('sat.w2', [-1] * n), ('sat.w3', [3, -3] * (n // 2)),
                    ('sat.w8', [127] * 65535 + [1, 1]), ('sat.w8', [-128] + [-1] * 65534 + [5])):
        cases.append([spec, 0, 0, a, [1] * len(a), None])
    return cases


def generate_supplement():
    from tools.scaled_bridge_v2.oracles import expected_dot
    out = []
    for spec, sa, sw, a, w, cert in _supplement_cases():
        value, word = expected_dot(list(zip(a, w)), sa, sw, spec, cert)
        out.append({'policy': spec, 'shift_a': sa, 'shift_w': sw, 'a': a, 'w': w, 'certificate': cert,
                    'value_hex': float(value).hex(), 'word': word})
    return out


def load_supplement():
    with gzip.open(SUPPLEMENT_PATH, 'rt') as f:
        return json.load(f)


def write_supplement():
    payload = {'schema': 'scaled-bridge-ext-vectors-1', 'seed': SEED + 1, 'oracle_sources_sha256': oracle_hashes(),
               'origin': 'tools/scaled_bridge_v2/oracles.py expected_dot: sat.abs-<d>/sat.struct-<d> with 2.1 '
                         'certificate fields, and saturation counters beyond the 65,535 cap of the event word',
               'cases': generate_supplement()}
    if SUPPLEMENT_PATH.exists():
        if load_supplement()['cases'] != payload['cases']:
            raise ValueError(f'{SUPPLEMENT_PATH} exists with different cases; it is written once')
        return SUPPLEMENT_PATH
    with gzip.GzipFile(SUPPLEMENT_PATH, 'wb', mtime=0) as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode())
    return SUPPLEMENT_PATH


if __name__ == '__main__':
    path = write()
    print(path, len(load()['cases']), 'cases', path.stat().st_size, 'bytes')
    path = write_supplement()
    print(path, len(load_supplement()['cases']), 'cases', path.stat().st_size, 'bytes')
