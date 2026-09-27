from fractions import Fraction
from itertools import product

import pytest

from development.acceptance_proofs.mapped_mean import prove_mean
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, Tensor


def test_odd_binary_count_excludes_zero_tie_but_even_count_does_not():
    encoding = Encoding('binary_pm1', (Fraction(1, 10),))
    assert prove_mean(encoding, encoding, 49)['status'] == 'exact_output_codes'
    assert prove_mean(encoding, encoding, 784)['status'] == 'pending'


def test_ternary_bound_covers_all_small_reduction_orders():
    source, target = Encoding('ternary', (Fraction(1, 10),)), Encoding('ternary', (Fraction(1, 7),))
    count = 5
    proof = prove_mean(source, target, count)
    assert proof['status'] == 'exact_output_codes'
    fp = format_named('fp64_e11m52_accumulator')
    for values in product((Fraction(-1, 10), Fraction(0), Fraction(1, 10)), repeat=count):
        total = Fraction(0)
        for value in values:
            total = fp.rounded(total+value)
        actual = fp.rounded(total/count)
        expected = sum(values)/count
        assert abs(Fraction(actual)-expected) <= Fraction(proof['mean_error_bound'])
        assert Tensor.quantize([actual], (1,), target).codes == Tensor.quantize([expected], (1,), target).codes


def test_proof_does_not_accept_an_exact_output_threshold_tie():
    source = Encoding('ternary', (1,))
    target = Encoding('ternary', (Fraction(2, 5),))
    assert prove_mean(source, target, 5)['status'] == 'pending'


def test_uncovered_domains_fail_closed():
    with pytest.raises(ValueError, match='scalar mapped'):
        prove_mean(Encoding('nf4', (1,)), Encoding('nf4', (1,)), 49)
