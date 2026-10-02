"""Independent rational witnesses for one codebook and both native backends."""
from __future__ import annotations
from fractions import Fraction as Q
import json
import time
import numpy as np
from public.inference.reference.arithmetic import format_named, model_c
from .common import ROOT, immutable, reference, unseal, run_root, engine_sources, digest, under_gpu_lock, Stopwatch
from .codebooks import Codebook, scalar_codebook, quantize, reconstructed
from .native import Native
from .engine import oracle_dots, window_max
from .oracles import expected_dot, saturating


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rounded64(value):
    return float(format_named('fp64_e11m52_accumulator').rounded(value))


def manifest_levels(name):
    """Finite levels straight from the manifest oracle, independent of Codebook."""
    from public.formats.oracle.number_format import NumberFormat
    fmt = NumberFormat(json.loads((ROOT / f'public/formats/manifests/accepted/{name}.json').read_text()))
    book = {}
    for code in range(1 << fmt.bits):
        value = fmt.decode(code)
        if value.is_finite():
            book[Q(value)] = min(code, book.get(Q(value), code))
    return book


def python_dot(pairs, shift_a, shift_w, policy):
    if policy == 'wide':
        return rounded64(Q(sum(a * b for a, b in pairs), 1 << (shift_a + shift_w)))
    fmt = format_named('fp32_e8m23_accumulator')
    return float(fmt.decode(model_c(((Q(a, 1 << shift_a), Q(b, 1 << shift_w)) for a, b in pairs), fmt)))


def conversion_checks(native, rng):
    """Directed and random witnesses of the exact accumulators and RNE conversion."""
    checks = 0
    directed = [([2**27, 1], [2**26, 1]), ([2**27, 3], [2**26, 1]), ([2**27, 1, 1], [2**26, 1, 1]),
                ([2**31, 1], [2**31, 1]), ([-(2**31), -1], [2**31, 1]), ([2**32 - 2**8, 1], [2**32 - 2**8, 1]),
                ([2**32 - 2**8] * 3 + [-(2**32 - 2**8)] * 3 + [1], [2**32 - 2**8] * 6 + [-1]),
                ([2**30, 2**30, 1], [2**23, 2**23, 1]), ([2**30, 2**30, 3], [2**23, 2**23, 1])]
    cases = [(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)) for a, b in directed]
    for trial in range(120):
        k = int(rng.integers(1, 48))
        if trial % 3 == 0:
            a = rng.integers(-2**20, 2**20, size=k); b = rng.integers(-2**20, 2**20, size=k)
        elif trial % 3 == 1:
            a = rng.integers(-2**31 + 1, 2**31, size=k); b = rng.integers(-2**31 + 1, 2**31, size=k)
        else:
            a = rng.integers(-7, 8, size=k) * (1 << rng.integers(0, 29, size=k))
            b = rng.integers(-7, 8, size=k) * (1 << rng.integers(0, 29, size=k))
        cases.append((a.astype(np.int64), b.astype(np.int64)))
    for a, b in cases:
        x = a.reshape(1, -1, 1, 1); w = b.reshape(1, -1, 1, 1)
        pairs = [(int(p), int(q)) for p, q in zip(a, b)]
        float32_exact = all(int(np.float32(v)) == v for pair in pairs for v in pair)
        for policy in ('wide', 'control'):
            if policy == 'control' and not float32_exact:
                continue
            expect = python_dot(pairs, 2, 3, policy)
            for kernel in native.values():
                got = kernel.conv(x, w, {}, 5, policy)
                require(np.float64(expect).tobytes() == got.tobytes(), 'accumulator/conversion witness mismatch')
                checks += 1
    return checks


def export_codebook(name, case):
    """Codebook data of an export and an oracle of its levels that does not read that data.

    Signed books must be the accepted manifest codebook. The B2 unsigned integer variant
    (`<format>.unsigned`, variant b2_unsigned_integer_v1) is defined as levels 0 .. 2^bits-1, code = level.
    """
    from .worker import locate
    data = unseal(locate(case))['codebooks'][name]
    if data.get('variant') == 'b2_unsigned_integer_v1':
        base = data['base_format']
        bits = int(json.loads((ROOT / f'public/formats/manifests/accepted/{base}.json').read_text())['bits'])
        require(name == base + '.unsigned' and data['id'] == name, 'unsigned codebook name')
        return data, {Q(k): k for k in range(1 << bits)}
    require(data == scalar_codebook(name), 'export codebook differs from the accepted manifest codebook')
    return data, manifest_levels(name)


def same(expected, actual):
    return np.float64(expected).tobytes() == np.float64(actual).tobytes()


def unary_oracle(op, value):
    """Contract 2.2 relu6 / hardsigmoid / hardswish of an exact binary64 operand value, in rationals."""
    v = Q(value)
    if op == 'relu6':
        return float(min(max(v, Q(0)), Q(6))) + 0.
    t = min(max(Q(rounded64(v + 3)), Q(0)), Q(6))
    if op == 'hardsigmoid':
        return rounded64(t / 6) + 0.
    return rounded64(Q(rounded64(v * t)) / 6) + 0.


OPERATOR_SCALES = (1., float(np.float32(.0137)), float(np.float32(.1)), float(np.float32(2. ** -7)), float(np.float32(.73)))


def operator_checks(book, partners, rng):
    """Contract 2.2 operators (relu6, hardsigmoid, hardswish, mul) against rational RNE64 oracles.

    The operators run on the host in binary64 for either native backend (they have no kernel); these witnesses
    cover every code level of `book` at several scales, directed and random unstored values, and code pairs of
    `book` with each partner codebook (exhaustive up to 70,000 pairs per scale pair, otherwise 20,000 random).
    """
    from .engine import Coded, Raw, unary, tensor_product
    counts = {'unary_stored': 0, 'unary_unstored': 0, 'mul_stored': 0, 'mul_unstored': 0, 'refusals': 0}
    units = book.units.reshape(1, -1, 1, 1)
    for scale in OPERATOR_SCALES:
        state = Coded(book.codes.reshape(units.shape), units, scale, book)
        for op in ('relu6', 'hardsigmoid', 'hardswish'):
            got = unary(op, state).ravel()
            for u, g in zip(book.units, got):
                level = max(int(u), 0) if op == 'relu6' else int(u)
                r = rounded64(Q(level, 1 << book.shift) * Q(scale))
                require(same(unary_oracle(op, r), g), f'{op} stored-operand witness mismatch')
                counts['unary_stored'] += 1
    special = [-1e300, -6., -3., -1.5, -1., -0.375, -0., 0., 5e-324, 1., 3., 6., 1e300]
    special += [float(np.nextafter(v, d)) for v in (-3., 3., 6., -1.5) for d in (-np.inf, np.inf)]
    values = np.array(special + list(rng.normal(scale=4., size=400)) + list(rng.uniform(-3.5, 6.5, size=400)))
    for op in ('relu6', 'hardsigmoid', 'hardswish'):
        got = unary(op, Raw(values.reshape(1, -1, 1, 1))).ravel()
        for v, g in zip(values, got):
            require(same(unary_oracle(op, float(v)), g), f'{op} unstored-operand witness mismatch')
            counts['unary_unstored'] += 1
    for partner in partners:
        cert = {'stored_operands': True, 'input_codebooks': [book.id, partner.id]}
        pairs = len(book.units) * len(partner.units)
        for index, (sa, sb) in enumerate(((float(np.float32(.0137)), float(np.float32(.731))),
                                          (float(np.float32(2. ** -5)), float(np.float32(.1))))):
            if pairs <= 70000 and index == 0:
                ua = book.units.reshape(1, -1, 1, 1); ub = np.tile(partner.units.reshape(1, 1, 1, -1), (1, ua.shape[1], 1, 1))
            else:
                ua = rng.choice(book.units, size=(1, 200, 1, 1)); ub = rng.choice(partner.units, size=(1, 200, 1, 100))
            a = Coded(np.zeros(ua.shape, dtype=np.uint8), ua.astype(np.int64), sa, book)
            b = Coded(np.zeros(ub.shape, dtype=np.uint8), ub.astype(np.int64), sb, partner)
            got = tensor_product(a, b, cert, 'witness')
            shift = book.shift + partner.shift
            for (_, i, _, j), g in np.ndenumerate(got):
                p = int(ua[0, i, 0, 0]) * int(ub[0, i, 0, j])
                expect = rounded64(Q(rounded64(Q(p, 1 << shift) * Q(sa))) * Q(sb)) + 0.
                require(same(expect, g), 'mul stored-operand witness mismatch')
                counts['mul_stored'] += 1
        x = rng.normal(scale=3., size=(1, 8, 1, 1)); y = rng.normal(scale=3., size=(1, 8, 5, 7)); y[0, 0, 0, :3] = (-0., 0., 1e-300)
        got = tensor_product(Raw(x), Raw(y), {'stored_operands': False}, 'witness')
        for (n, c, i, j), g in np.ndenumerate(got):
            require(same(rounded64(Q(float(x[n, c, 0, 0])) * Q(float(y[n, c, i, j]))) + 0., g), 'mul unstored-operand witness mismatch')
            counts['mul_unstored'] += 1
        for bad in ((Coded(np.zeros((1, 2, 3, 3), np.uint8), np.zeros((1, 2, 3, 3), np.int64), 1., book),
                     Coded(np.zeros((1, 3, 1, 1), np.uint8), np.zeros((1, 3, 1, 1), np.int64), 1., partner), cert),
                    (Coded(np.zeros((1, 2, 1, 1), np.uint8), np.zeros((1, 2, 1, 1), np.int64), 1., book),
                     Coded(np.zeros((1, 2, 3, 3), np.uint8), np.zeros((1, 2, 3, 3), np.int64), 1., partner),
                     {'stored_operands': False}),
                    (Raw(np.zeros((1, 2, 1, 1))), Raw(np.zeros((1, 2, 3, 3))), cert)):
            try:
                tensor_product(*bad, 'witness')
            except ValueError:
                counts['refusals'] += 1
            else:
                raise ValueError('an inadmissible tensor product was accepted')
    return counts


def codebook_checks(name, case=None):
    watch = Stopwatch('conformance', [name] + ([case] if case else [])); start = time.perf_counter(); checks = 0
    native = {b: Native(b) for b in ('cpp', 'cuda')}
    rng = np.random.default_rng(20261001)
    data, oracle = (scalar_codebook(name), manifest_levels(name)) if case is None else export_codebook(name, case)
    book = Codebook(data)
    levels, units, bounds, shift = book.levels, book.units, book.bounds, book.shift
    require(sorted(oracle) == [Q(int(u), 1 << shift) for u in units], 'codebook levels differ from the manifest oracle')
    require([oracle[q] for q in sorted(oracle)] == [int(c) for c in book.codes], 'codebook codes differ from the manifest oracle')
    samples = np.concatenate([levels, bounds, np.nextafter(bounds, -np.inf), np.nextafter(bounds, np.inf),
                              [-1e300, 1e300, -0., 0.]])
    for scale in (1., float(np.float32(.0137)), float(np.float32(2. ** -126)), float(np.float32(2. ** 120))):
        actual_codes, _, _ = quantize(samples * scale if scale == 1. else samples, scale, book)
        for value, actual in zip(samples, actual_codes):
            raw = float(value)
            with np.errstate(over='ignore'):
                actual_y = np.float64(raw) / np.float64(scale)
            expected_y = rounded64(Q(raw) / Q(scale))
            require(actual_y == expected_y, 'binary64 normalization oracle')
            if np.isinf(expected_y):
                expected = oracle[max(oracle) if expected_y > 0 else min(oracle)]
            else:
                v = Q(expected_y)
                expected = oracle[min(oracle, key=lambda k: (abs(k - v), oracle[k] & 1, oracle[k]))]
            require(int(actual) == expected, 'rational scaled store mismatch'); checks += 1
    for invalid in (np.nan, np.inf, -np.inf):
        try:
            quantize([invalid], 1., book)
        except ValueError:
            checks += 1
        else:
            raise ValueError('nonfinite store admitted')
    for scale in (0., -1., np.inf, np.nan):
        try:
            quantize([1.], scale, book)
        except ValueError:
            checks += 1
        else:
            raise ValueError('invalid scale admitted')
    # Every finite code pair: exact product, and the one-step FMA witness.
    pairs_exact = [[Q(int(a) * int(b), 1 << (2 * shift)) for b in units] for a in units]
    wide = np.array([[rounded64(q) for q in row] for row in pairs_exact]).T.reshape(1, len(units), 1, len(units))
    wide = wide + 0.
    if all(len(bin(abs(int(a) * int(b))).rstrip('0')) - 2 <= 24 for a in units for b in units):
        control = wide
    else:
        control = np.array([[python_dot([(int(a), int(b))], shift, shift, 'control') for b in units] for a in units]
                           ).T.reshape(wide.shape) + 0.
    x = units.reshape(1, 1, 1, -1); w = units.reshape(-1, 1, 1, 1)
    for kernel in native.values():
        for policy, expected in (('wide', wide), ('control', control)):
            got = kernel.conv(x, w, {}, 2 * shift, policy)
            require(np.array_equal(expected.view('u8'), got.view('u8')), 'exhaustive pair product mismatch')
            checks += got.size
    # Independently indexed small convolutions: groups, stride, dilation, padding.
    for attrs in ({'groups': 2, 'stride': [2, 1], 'padding': [1, 2], 'dilation': [1, 2]},
                  {'groups': 1, 'stride': [1, 2], 'padding': [0, 1], 'dilation': [2, 1]},
                  {'groups': 4, 'stride': [1, 1], 'padding': [1, 1], 'dilation': [1, 1]}):
        g = attrs['groups']; x = rng.choice(units, size=(2, 4, 6, 7)); w = rng.choice(units, size=(4, 4 // g, 2, 3))
        for policy in ('wide', 'control'):
            cpu = native['cpp'].conv(x, w, attrs, 2 * shift, policy); gpu = native['cuda'].conv(x, w, attrs, 2 * shift, policy)
            require(np.array_equal(cpu.view('u8'), gpu.view('u8')), 'geometry CPU/CUDA mismatch')
            for n in range(2):
                for oc in range(4):
                    for oy in range(cpu.shape[2]):
                        for ox in range(cpu.shape[3]):
                            pairs = []
                            for c in range(4 // g):
                                for ky in range(2):
                                    for kx in range(3):
                                        iy = oy * attrs['stride'][0] - attrs['padding'][0] + ky * attrs['dilation'][0]
                                        ix = ox * attrs['stride'][1] - attrs['padding'][1] + kx * attrs['dilation'][1]
                                        a = int(x[n, oc // (4 // g) * (4 // g) + c, iy, ix]) if 0 <= iy < 6 and 0 <= ix < 7 else 0
                                        pairs.append((a, int(w[oc, c, ky, kx])))
                            expect = python_dot(pairs, shift, shift, policy)
                            require(np.float64(expect).tobytes() == cpu[n, oc, oy, ox].tobytes(), 'independent geometry oracle mismatch')
                            checks += 1
    # Long cancellation and rounding-sensitive FMA sequence in the admitted grid.
    hi = int(units.max()); small = int(units[units > 0].min())
    seq = np.array([hi] * 2048 + [small] * 17 + [-hi] * 2048, dtype=np.int64)
    x = seq.reshape(1, -1, 1, 1); w = np.full_like(x, hi).reshape(1, -1, 1, 1)
    for policy in ('wide', 'control'):
        for kernel in native.values():
            y = kernel.conv(x, w, {}, 2 * shift, policy)
            checks += oracle_dots(x, w, {}, y, shift, shift, policy)
    checks += conversion_checks(native, rng)
    # Residual, MAC scale/bias and average sequences against rational RNE64.
    for _ in range(160):
        a, b = (int(v) for v in rng.choice(units, size=2))
        sa = float(np.float32(rng.uniform(.0001, 3))); sb = float(np.float32(rng.uniform(.0001, 3)))
        ra = rounded64(Q(a, 1 << shift) * Q(sa)); rb = rounded64(Q(b, 1 << shift) * Q(sb))
        expect = rounded64(Q(ra) + Q(rb)); got = float(reconstructed(a, sa, book) + reconstructed(b, sb, book))
        require(np.float64(expect).tobytes() == np.float64(got).tobytes(), 'residual alignment mismatch'); checks += 1
        bias = float(np.float32(rng.normal())); dot = rounded64(Q(a * b, 1 << (2 * shift)))
        expect = rounded64(Q(rounded64(Q(rounded64(Q(dot) * Q(sa))) * Q(sb))) + Q(bias))
        got = np.float64(dot) * np.float64(sa); got = got * np.float64(sb); got = got + np.float64(bias)
        require(np.float64(expect).tobytes() == got.tobytes(), 'scale/bias sequence mismatch'); checks += 1
        expect = rounded64(Q(rounded64(Q(a + b, 1 << shift) / 49)) * Q(sa))
        got = np.ldexp(np.float64(a + b), -shift) / np.float64(49); got = got * np.float64(sa)
        require(np.float64(expect).tobytes() == got.tobytes(), 'average sequence mismatch'); checks += 1
    p = np.array([[[[-1, -2], [-3, -4]]]], dtype=np.int64)
    require(np.array_equal(window_max(p, {'kernel_size': [2, 2], 'stride': [1, 1], 'padding': [1, 1], 'dilation': [1, 1]},
                                      np.iinfo(np.int64).min),
                           np.array([[[[-1, -1, -2], [-1, -1, -2], [-3, -3, -4]]]], dtype=np.int64)), 'negative padded pool mismatch')
    checks += 1
    for kernel in native.values():
        for invalid in (np.array([[[[2**32]]]], dtype=np.int64), np.array([[[[.5]]]])):
            try:
                kernel.conv(invalid, np.ones((1, 1, 1, 1), dtype=np.int64), {}, 3, 'wide')
            except ValueError:
                checks += 1
            else:
                raise ValueError('native invalid grid input accepted')
        try:
            kernel.conv(np.array([[[[2**25 + 1]]]], dtype=np.int64), np.ones((1, 1, 1, 1), dtype=np.int64), {}, 3, 'control')
        except ValueError:
            checks += 1
        else:
            raise ValueError('inexact binary32 operand accepted by the control arm')
    if case is None:
        partners = [book]
    else:
        from .worker import locate
        partners = [Codebook(v) for _, v in sorted(unseal(locate(case))['codebooks'].items())]
    operators = operator_checks(book, partners, rng)
    checks += sum(operators.values())
    record = {'status': 'pass', 'codebook': data, 'checks': checks, 'seconds': time.perf_counter() - start,
              'operator_checks_2_2': operators, 'operator_partners': [b.id for b in partners],
              'engine_sources': digest(engine_sources()), 'native': {b: reference(n.path) for b, n in native.items()},
              'gpu_lock_declared': under_gpu_lock(),
              'scope': 'manifest-enumerated store oracle; exhaustive finite code pairs; independently indexed small '
                       'convolutions incl. depthwise; long FMA cancellation; two-limb and RNE conversion witnesses; '
                       'rational post-operation sequences; special-value failures; contract 2.2 operators '
                       '(relu6, hardsigmoid, hardswish on every level and directed values; mul on code pairs with '
                       'every codebook of the export; inadmissible products refused)'}
    path = run_root() / 'conformance' / f'{name}.json'
    if not path.exists():
        immutable(path, record)
    watch.close(checks=checks)
    print(f'{name}: primitive conformance {checks} checks passed', flush=True)
    return record


POLICY_GEOMETRY = ('sat.w12', 'sat.w20', 'fp16', 'fp16.x-6', 'f21')


def policy_checks():
    """Primitive rational witnesses of the parameterised accumulator policies, both backends.

    Independent of any codebook: sequences of grid integers on a chosen product
    grid. Every kernel value and event word is compared with oracles.py.
    """
    watch = Stopwatch('conformance', ['policies']); start = time.perf_counter()
    native = {b: Native(b) for b in ('cpp', 'cuda')}
    rng = np.random.default_rng(20261002)
    count = {'saturating_directed': 0, 'saturating_random': 0, 'saturating_int32_manifest': 0, 'float_directed': 0,
             'float_random': 0, 'geometry': 0, 'observed_high_clamp': 0, 'observed_low_clamp': 0,
             'observed_positive_infinity': 0, 'observed_negative_infinity': 0, 'observed_subnormal_result': 0}

    def one(a, b, shift, spec, group):
        a = [int(v) for v in a]; b = [int(v) for v in b]
        x = np.array(a, dtype=np.int64).reshape(1, -1, 1, 1); w = np.array(b, dtype=np.int64).reshape(1, -1, 1, 1)
        expect, word = expected_dot(list(zip(a, b)), shift, 0, spec)
        for kernel in native.values():
            y, events = kernel.conv_stat(x, w, {}, shift, spec)
            require(np.float64(expect).tobytes() == y.tobytes(), f'policy witness value mismatch {spec}')
            require(int(events.flat[0]) == word, f'policy witness event mismatch {spec}')
            count[group] += 1
        if spec.startswith('sat'):
            count['observed_high_clamp'] += bool(word & 0xFFFF); count['observed_low_clamp'] += bool(word >> 16)
        else:
            count['observed_positive_infinity'] += expect == np.inf; count['observed_negative_infinity'] += expect == -np.inf
        return expect, word

    # Saturating integer: both end points, exact landing on an end point, recovery, no wrap.
    for width in (2, 3, 8, 16, 24, 32, 48, 63):
        hi = (1 << (width - 1)) - 1; lo = -hi - 1; spec = f'sat.w{width}'
        if width <= 32:
            A, B = [hi], [1]
        else:
            step = 1 << (width - 32); A, B = [2**31 - 1, step - 1], [step, 1]      # sums to hi exactly
        N = [-v for v in A]
        for shift in (0, 3):
            scale = 2. ** -shift
            value, word = one(A, B, shift, spec, 'saturating_directed')
            require(value == float(hi) * scale and word == 0, 'landing on the high end point is not a saturation event')
            value, word = one(A + [1], B + [1], shift, spec, 'saturating_directed')
            require(value == float(hi) * scale and word == 1, 'high end point witness')
            value, word = one(A + [1, -1], B + [1, 1], shift, spec, 'saturating_directed')
            require(value == float(hi - 1) * scale and word == 1, 'no wrap and recovery after a high clamp')
            value, word = one(N + [-1], B + [1], shift, spec, 'saturating_directed')
            require(value == float(lo) * scale and word == 0, 'landing on the low end point is not a saturation event')
            value, word = one(N + [-1, -1, 3], B + [1, 1, 1], shift, spec, 'saturating_directed')
            require(value == float(lo + 3) * scale and word == 1 << 16, 'low end point witness and recovery')
            value, word = one(A * 3 + N * 5 + A * 4, B * 3 + B * 5 + B * 4, shift, spec, 'saturating_directed')
            require(word & 0xFFFF and word >> 16, 'both end points in one reduction')
        if width > 32:
            for a, b in (([2**31 - 1] * 6, [2**31 - 1] * 6), ([2**32 - 1] * 5 + [-(2**32 - 1)] * 11, [2**32 - 1] * 16),
                         ([2**32 - 1, 1, -(2**32 - 1)], [2**32 - 1, 1, 2**32 - 1])):
                one(a, b, 3, spec, 'saturating_directed')
        for trial in range(60):
            k = int(rng.integers(1, 80)); span = int(rng.integers(1, max(2, min(width + 2, 31))))
            a = rng.integers(-(1 << span), (1 << span) + 1, size=k); b = rng.integers(-(1 << span), (1 << span) + 1, size=k)
            one(a, b, int(rng.integers(0, 20)), spec, 'saturating_random')
    fmt32 = format_named('int32_accumulator')
    for trial in range(80):
        k = int(rng.integers(1, 40))
        a = [int(v) for v in rng.integers(-2**24, 2**24, size=k)]; b = [int(v) for v in rng.integers(-2**12, 2**12, size=k)]
        public = fmt32.decode(model_c(((Q(p), Q(q)) for p, q in zip(a, b)), fmt32))
        require(public == saturating(list(zip(a, b)), 32)[0], 'saturating oracle differs from the public int32 manifest')
        one(a, b, 0, 'sat.w32', 'saturating_int32_manifest')
    # Float accumulators. binary16: largest finite 65504, overflow threshold 65520 (tie to even -> infinity),
    # subnormal spacing 2^-24. 21-bit float: largest finite (2-2^-12)*2^127, subnormal spacing 2^-138.
    one([65504], [1], 0, 'fp16', 'float_directed')
    value, word = one([65504, 15], [1, 1], 0, 'fp16', 'float_directed'); require(value == 65504 and word == 0, 'below the overflow threshold')
    value, word = one([65504, 16, -65504, 7], [1, 1, 1, 1], 0, 'fp16', 'float_directed'); require(value == np.inf and word == 3, 'overflow to +infinity stays infinite')
    value, word = one([-65504, -16, 65504], [1, 1, 1], 0, 'fp16', 'float_directed'); require(value == -np.inf and word == 2, 'overflow to -infinity')
    value, word = one([255, 255], [255, 255], 0, 'fp16', 'float_directed'); require(value == np.inf, 'integer code products overflow binary16')
    value, word = one([255, 255], [255, 255], 0, 'fp16.x-12', 'float_directed'); require(np.isfinite(value) and word == 0, 'scale exponent')
    for a, expected in (([1], 0.), ([32], 0.), ([33], 2. ** -24), ([96], 2. ** -23), ([160], 2. ** -23), ([64] * 10, 10 * 2. ** -24),
                        ([32, 32], 0.), ([33, 31], 2. ** -24), ([33, 32], 2. ** -23), ([33, -33], 0.), ([65535], None), ([2**16 - 16, 16, 1], None)):
        value, word = one(a, [1] * len(a), 30, 'fp16', 'float_directed')
        require(expected is None or value == expected, 'binary16 subnormal witness'); count['observed_subnormal_result'] += 1
    for a in ([1], [2**7], [2**7 + 1], [3 * 2**7], [2**20 - 1, 1, 1], [2**12 + 1] * 9, [5, -5], [2**8, -2**7, 2**7 - 1]):
        one(a, [1] * len(a), 126, 'f21.x-20', 'float_directed'); count['observed_subnormal_result'] += 1
    value, word = one([2**32 - 1, 2**32 - 1], [2**32 - 1, 2**32 - 1], 0, 'f21.x64', 'float_directed'); require(value == np.inf, '21-bit overflow')
    value, word = one([-(2**32 - 1)] * 3, [2**32 - 1] * 3, 0, 'f21.x64', 'float_directed'); require(value == -np.inf, '21-bit negative overflow')
    one([2**32 - 1, 1, -(2**32 - 1), 4097, 4097, -1], [2**32 - 1, 1, 2**32 - 1, 1, 1, 1], 0, 'f21', 'float_directed')
    one([2**32 - 1] * 40 + [3] * 40, [2**32 - 1] * 40 + [5] * 40, 9, 'f21.x-9', 'float_directed')
    for trial in range(900):
        k = int(rng.integers(1, 90)); mode = trial % 6
        spec = ('fp16', 'f21', 'fp16.x-12', 'f21.x7', 'fp16.x9', 'f21.x-20')[trial % 6 if trial % 5 else (trial // 5) % 6]
        if mode == 0:
            a = rng.integers(-255, 256, size=k); b = rng.integers(-128, 128, size=k)
        elif mode == 1:
            a = rng.integers(-15, 16, size=k) * (1 << rng.integers(0, 12, size=k)); b = rng.integers(-15, 16, size=k) * (1 << rng.integers(0, 12, size=k))
        elif mode == 2:
            a = rng.integers(-2**31 + 1, 2**31, size=k); b = rng.integers(-2**31 + 1, 2**31, size=k)
        elif mode == 3:
            a = rng.integers(-(2**32) + 1, 2**32, size=k); b = rng.integers(-(2**32) + 1, 2**32, size=k)
        elif mode == 4:
            a = rng.integers(-3, 4, size=k); b = rng.integers(-3, 4, size=k)
        else:
            half = rng.integers(-2**20, 2**20, size=(k + 1) // 2); a = np.concatenate([half, -half])[:k]; b = np.ones(k, dtype=np.int64) * int(rng.integers(1, 2**10))
        shift = int(rng.choice([0, 0, 6, 12, 20, 30, 44, 60, 100, 126]))
        value, word = one(a, b, shift, spec, 'float_random')
    # Geometry with padding, stride, dilation and groups: every output against the oracle.
    for spec in POLICY_GEOMETRY:
        for attrs in ({'groups': 2, 'stride': [2, 1], 'padding': [1, 2], 'dilation': [1, 2]},
                      {'groups': 1, 'stride': [1, 2], 'padding': [1, 1], 'dilation': [1, 1]}):
            g = attrs['groups']; x = rng.integers(-128, 256, size=(2, 4, 6, 7)); w = rng.integers(-128, 128, size=(4, 4 // g, 3, 3))
            cpu, ecpu = native['cpp'].conv_stat(x, w, attrs, 4, spec); gpu, egpu = native['cuda'].conv_stat(x, w, attrs, 4, spec)
            require(np.array_equal(cpu.view('u8'), gpu.view('u8')) and np.array_equal(ecpu, egpu), 'policy geometry CPU/CUDA mismatch')
            for n in range(2):
                for oc in range(4):
                    for oy in range(cpu.shape[2]):
                        for ox in range(cpu.shape[3]):
                            pairs = []
                            for c in range(4 // g):
                                for ky in range(3):
                                    for kx in range(3):
                                        iy = oy * attrs['stride'][0] - attrs['padding'][0] + ky * attrs['dilation'][0]
                                        ix = ox * attrs['stride'][1] - attrs['padding'][1] + kx * attrs['dilation'][1]
                                        a = int(x[n, oc // (4 // g) * (4 // g) + c, iy, ix]) if 0 <= iy < 6 and 0 <= ix < 7 else 0
                                        pairs.append((a, int(w[oc, c, ky, kx])))
                            expect, word = expected_dot(pairs, 4, 0, spec)
                            require(np.float64(expect).tobytes() == cpu[n, oc, oy, ox].tobytes()
                                    and int(ecpu[n, oc, oy, ox]) == word, 'policy geometry oracle mismatch')
                            count['geometry'] += 1
    for key in ('observed_high_clamp', 'observed_low_clamp', 'observed_positive_infinity', 'observed_negative_infinity'):
        require(count[key] > 0, 'a required witness class was never exercised: ' + key)
    for kernel in native.values():
        for bad in ('sat.w64', 'sat.w1', 'fp16.x65', 'fp32', 'sat.w08', 'sat.abs-1'):
            try:
                kernel.conv_stat(np.ones((1, 1, 1, 1), dtype=np.int64), np.ones((1, 1, 1, 1), dtype=np.int64), {}, 0, bad)
            except ValueError:
                count['saturating_directed'] += 1
            else:
                raise ValueError('invalid policy accepted: ' + bad)
    checks = sum(v for k, v in count.items() if not k.startswith('observed'))
    record = {'status': 'pass', 'checks': checks, 'by_class': {k: int(v) for k, v in count.items()},
              'seconds': time.perf_counter() - start, 'engine_sources': digest(engine_sources()),
              'native': {b: reference(n.path) for b, n in native.items()}, 'gpu_lock_declared': under_gpu_lock(),
              'scope': 'saturating integer widths 2..63 (both end points, exact landing, recovery, 64-bit operand '
                       'products, public int32 manifest cross-check); binary16 and 21-bit float (overflow to both '
                       'infinities, sticky infinity, overflow tie, subnormal ties and underflow to zero, scale '
                       'exponents, cancellation, operands up to 2^32); padded/strided/dilated/grouped geometry'}
    path = run_root() / 'conformance' / 'policies.json'
    if not path.exists():
        seal_volatile(path, record)
    watch.close(checks=checks)
    print(f'policies: primitive conformance {checks} checks passed {count}', flush=True)
    return record


def seal_volatile(path, record):
    from .common import seal
    path.parent.mkdir(parents=True, exist_ok=True)
    seal(path, record)
