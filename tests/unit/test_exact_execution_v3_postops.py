from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from public.inference.operators.dispatch import Operators
from public.inference.reference import operators as ref
from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import Encoding, Tensor, QUANTIZATION_OBSERVER
from tools.exact_execution_v3.postops import DIRECT_FORMATS, FP64, FP64Stores, FloatingResiduals


def observed(call):
    events = []
    token = QUANTIZATION_OBSERVER.set(lambda values, tensor: events.append((tuple(values), tensor)))
    try:
        result = call()
    finally:
        QUANTIZATION_OBSERVER.reset(token)
    return result, events


def bits(values):
    return tuple(np.asarray(values, dtype=np.float64).view(np.uint64).tolist())


@pytest.mark.parametrize('name', sorted(DIRECT_FORMATS))
@pytest.mark.parametrize('activation', ['identity', 'relu', 'relu6'])
def test_fp64_store_preserves_all_low_bit_boundaries_bias_and_events(name, activation):
    fmt = format_named(name)
    finite = sorted(set(float(fmt.decode(code)) for code in range(1 << fmt.bits)
                        if fmt.decode(code).is_finite()))
    midpoints = [(a + b) / 2 for a, b in zip(finite, finite[1:])]
    samples = [np.float64(0), np.float64(-0.0), np.float64(1e-320), np.float64(-1e-320),
               np.float64(1e300), np.float64(-1e300)]
    for value in finite + midpoints:
        samples.extend((np.nextafter(value, -np.inf), np.float64(value), np.nextafter(value, np.inf)))
    values = np.asarray(samples, dtype=np.float64)
    states = bits(np.concatenate((values, values)))
    shape = (1, 2, len(values))
    args = (states, shape, FP64, (Fraction(1), Fraction(1)), ('0.125', '-0'), activation, Encoding(name))
    expected = observed(lambda: Operators()._store(*args))
    actual = observed(lambda: FP64Stores().store(*args))
    assert actual[0] is not None
    assert actual[0].codes == expected[0].codes
    assert len(actual[1]) == len(expected[1]) == 1
    assert actual[1][0][1] == expected[1][0][1]
    for actual_raw, expected_raw in zip(actual[1][0][0], expected[1][0][0]):
        a, b = real(actual_raw), real(expected_raw)
        assert a == b
        if a == 0 and b == 0:
            assert (isinstance(a, Decimal) and a.is_signed()) == (isinstance(b, Decimal) and b.is_signed())


def test_fp64_store_specials_and_unsupported_domains_fall_back():
    store = FP64Stores()
    args = (bits([float('nan')]), (1, 1), FP64, (Fraction(1),), None, 'identity', Encoding('fp6_e2m3'))
    assert store.store(*args) is None
    assert store.store(bits([float('inf')]), *args[1:]) is None
    assert store.store(bits([1.0]), (1, 1), FP64, (Fraction(2),), None, 'identity', Encoding('fp6_e2m3')) is None
    assert store.store(bits([1.0]), (1, 1), FP64, (Fraction(1),), None, 'identity', Encoding('int8')) is None
    with pytest.raises(ValueError, match='bias count'):
        store.store(bits([1.0]), (1, 1), FP64, (Fraction(1),), ('0', '1'), 'identity', Encoding('fp6_e2m3'))


@pytest.mark.parametrize('name', sorted(DIRECT_FORMATS))
def test_float_residual_all_finite_code_pairs_and_exact_three_events(name):
    fmt = format_named(name)
    finite = [code for code in range(1 << fmt.bits) if fmt.decode(code).is_finite()]
    n = len(finite)
    encoding = Encoding(name)
    left = Tensor((1, n*n), tuple(a for a in finite for _ in finite), encoding)
    right = Tensor(left.shape, tuple(b for _ in finite for b in finite), encoding)
    attrs = {'operation': 'add', 'output': encoding, 'accumulator': FP64, 'alignment': encoding}
    expected = observed(lambda: ref.elementwise(left, right, **attrs))
    accelerator = FloatingResiduals()
    actual = observed(lambda: accelerator.add(left, right, **attrs))
    assert actual == expected
    assert len(actual[1]) == 3
    assert observed(lambda: accelerator.add(left, right, **attrs)) == expected


def test_float_residual_special_code_routes_to_unchanged_oracle():
    encoding = Encoding('fp6_e3m2')
    fmt = format_named(encoding.format)
    special = next(code for code in range(1 << fmt.bits) if not fmt.decode(code).is_finite())
    left = Tensor((1, 2), (0, 1), encoding)
    right = Tensor((1, 2), (special, 0), encoding)
    attrs = {'operation': 'add', 'output': encoding, 'accumulator': FP64, 'alignment': encoding}
    actual, actual_events = observed(lambda: FloatingResiduals().add(left, right, **attrs))
    expected, expected_events = observed(lambda: ref.elementwise(left, right, **attrs))
    assert actual == expected
    assert [(tuple(str(v) for v in raw), tensor) for raw, tensor in actual_events] == [
        (tuple(str(v) for v in raw), tensor) for raw, tensor in expected_events]
