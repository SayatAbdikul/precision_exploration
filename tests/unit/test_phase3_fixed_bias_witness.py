from fractions import Fraction
from itertools import product

import pytest

from public.analysis.phase3.fixed_bias_witness import SparseTwoSum, discrepant_dot_targets
from public.inference.reference.arithmetic import encode, format_named, round_integer


@pytest.mark.parametrize("weights", [(2,), (1, 1), (1, 2, -1), (0, 2, 2), (0, 0)])
def test_sparse_search_matches_exhaustive_distinct_position_vectors(weights):
    values = {0: Fraction(0), 1: Fraction(1, 2), 2: Fraction(-1), 3: Fraction(2)}
    search = SparseTwoSum(weights, values)
    possible = {
        sum((values[c] * w for c, w in zip(codes, weights)), Fraction(0))
        for codes in product(values, repeat=len(weights)) if sum(c != 0 for c in codes) <= 2
    }
    for target in (Fraction(n, 2) for n in range(-20, 21)):
        witness = search.find(target)
        assert (witness is not None) == (target in possible)
        if witness is not None:
            assert sum(c != 0 for c in witness) <= 2
            assert sum(values[c] * w for c, w in zip(witness, weights)) == target


def test_one_weight_position_cannot_be_used_twice():
    assert SparseTwoSum([1], {0: 0, 1: 1}).find(2) is None
    assert SparseTwoSum([1, 1], {0: 0, 1: 1}).find(2) == [1, 1]


def test_target_enumeration_matches_exhaustive_small_lattices_including_ties():
    fmt = format_named("posit4_es0")
    q, step, maximum = Fraction(1, 16), Fraction(1, 64), 2
    for bias in (Fraction(1, 1000), Fraction(-1, 1000), Fraction(1, 7), Fraction(0)):
        stored = round_integer(bias / step) * step
        expected = {n * q for n in range(-32, 33) if encode(fmt, n*q+bias) != encode(fmt, n*q+stored)}
        assert set(discrepant_dot_targets(q, maximum, bias, step, fmt.name)) == expected


def test_sparse_search_rejects_nonzero_only_codebooks():
    with pytest.raises(ValueError, match="exact activation zero"):
        SparseTwoSum([1], {0: -1, 1: 1})
