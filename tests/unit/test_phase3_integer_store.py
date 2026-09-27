from decimal import Decimal
from fractions import Fraction
import random

import pytest

from public.inference.operators.dispatch import Operators
from public.inference.tensor import Encoding, QUANTIZATION_OBSERVER
from tools.phase3.integer_store import integer_store, supported_scale


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("output_bits", [4, 6, 8])
@pytest.mark.parametrize("activation", ["identity", "relu", "relu6"])
def test_exact_integer_stores_match_reference_codes_and_all_observer_values(bits, output_bits, activation):
    rng = random.Random(370 + bits + output_bits)
    signed = list(range(-160, 161)) + [-(1 << (bits - 1)), (1 << (bits - 1)) - 1]
    signed += [rng.randrange(-(1 << (bits - 1)), 1 << (bits - 1)) for _ in range(17)]
    states = tuple(n & ((1 << bits) - 1) for n in signed * 2)
    shape = (1, 2, 1, len(signed))
    args = (states, shape, f"int{bits}_accumulator", (Fraction(1, 10), Fraction(3, 50)),
            [Fraction(7, 20), Fraction(-29, 100)], activation, Encoding(f"int{output_bits}", scales=("0.2",)))
    seen = []
    token = QUANTIZATION_OBSERVER.set(lambda values, tensor: seen.append(tuple(values)))
    try:
        expected = Operators()._store(*args)
        actual = integer_store(*args)
    finally:
        QUANTIZATION_OBSERVER.reset(token)
    assert actual == expected
    assert seen[0] == seen[1]


@pytest.mark.parametrize("bias", [[str(2**100), str(-(2**100))], [Decimal("Infinity"), Decimal("-Infinity")], None])
def test_saturating_int64_bias_addition_does_not_wrap(bias):
    bits = 64
    values = [-(2**63), -(2**63) + 1, -1, 0, 1, 2**63 - 2, 2**63 - 1] * 2
    states = tuple(v & ((1 << bits) - 1) for v in values)
    args = (states, (1, 2, 7), "int64_accumulator", (Fraction(1), Fraction(1)), bias, "identity", Encoding("int8"))
    assert integer_store(*args) == Operators()._store(*args)


@pytest.mark.parametrize("scale,output_scale", [("1", "2"), ("2", "1"), ("0.000000000000001", "100000000000000"),
                                              ("100000000000000", "0.000000000000001"), ("0.7", "0.3")])
def test_threshold_ties_duplicates_and_unreachable_codes(scale, output_scale):
    states = tuple(n & ((1 << 32) - 1) for n in range(-260, 261))
    args = (states, (1, 1, len(states)), "int32_accumulator", (Fraction(scale),), None, "identity", Encoding("int8", scales=(output_scale,)))
    assert integer_store(*args) == Operators()._store(*args)


def test_unsupported_domains_fall_back_instead_of_approximating():
    assert supported_scale(Fraction(1, 3)) is False
    assert supported_scale(Fraction(1, 2**200)) is False
    assert integer_store((0,), (1, 1), "fp32_e8m23_accumulator", (Fraction(1),), None, "identity", Encoding("int8")) is None
    assert integer_store((0,), (1, 1), "int32_accumulator", (Fraction(1, 3),), None, "identity", Encoding("int8")) is None


def test_rank_batch_channel_addressing_matches_reference():
    states = tuple(range(2 * 3 * 4 * 5))
    args = (states, (2, 3, 4, 5), "int32_accumulator", (Fraction(1, 10), Fraction(1, 5), Fraction(1, 2)),
            [1, 2, 3], "relu6", Encoding("int4", scales=("0.7",)))
    assert integer_store(*args) == Operators()._store(*args)
