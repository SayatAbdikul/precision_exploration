from fractions import Fraction
from itertools import product

from public.analysis.phase3.fixed_nonmac import finite_values, mean_thresholds, prove_nonmac, hard_codes
from public.analysis.phase3.fixed_mac_bounds import quantum
from public.inference.reference.arithmetic import format_named
from public.inference.reference.operators import adaptive_average_pool2d
from public.inference.tensor import Encoding, Tensor


ACC = "posit8_es1_quire64_accumulator"


def pool_graph():
    domain = Encoding("posit4_es0").document()
    return {"inputs": {"x": domain}, "nodes": [{"name": "mean", "op": "adaptive_average_pool2d", "inputs": ["x"],
             "attrs": {"output_size": 1, "output": domain, "accumulator": ACC}}]}


def test_mean_lattice_margin_matches_enumeration_and_preserves_exact_ties():
    values = finite_values(Encoding("posit4_es0"))
    q, count = quantum(values), 3
    report = mean_thresholds(values, count, values, Fraction(1, 2**24))
    distances = [abs(i*q/count-(a+b)/2)
                 for i in range(int(count*min(values)/q), int(count*max(values)/q)+1)
                 for a,b in zip(values, values[1:])]
    assert report["minimum_nonzero_margin"] == min(d for d in distances if d)
    # With an even reduction length, exact mean/output ties are attainable.
    ties = mean_thresholds(values, 2, values, Fraction(1, 2**24))
    assert ties["exact_ties"] > 0 and ties["ties_on_accumulator_grid"]
    coarse = mean_thresholds(values, 2, values, Fraction(1))
    assert not coarse["ties_on_accumulator_grid"]


def test_proved_mean_matches_all_two_and_three_code_windows():
    encoding = Encoding("posit4_es0")
    fmt = format_named(encoding.format)
    finite = [c for c in range(16) if fmt.decode(c).is_finite()]
    for count in (2, 3):
        cases = list(product(finite, repeat=count))
        inputs = Tensor((len(cases), 1, 1, count), tuple(c for row in cases for c in row), encoding)
        proof = prove_nonmac(pool_graph(), {"x": list(inputs.shape)})
        assert proof["pending_nodes"] == []
        actual = adaptive_average_pool2d(inputs, output=encoding, accumulator=ACC)
        expected = Tensor.quantize([sum((Fraction(fmt.decode(c)) for c in row), Fraction(0))/count for row in cases], actual.shape, encoding)
        assert actual.codes == expected.codes


def test_posit_hard_activation_probes_cover_finite_codes_only():
    encoding = Encoding("posit4_es0")
    for function in ("hard_swish", "hard_sigmoid"):
        result = hard_codes(encoding, encoding, function, ACC)
        assert result["finite_codes_tested"] == 15
        assert result["mismatches"] == 0


def test_finer_grid_resolves_posit8_minimum_positive_hard_swish_failure():
    encoding = Encoding("posit8_es1")
    original = hard_codes(encoding, encoding, "hard_swish", ACC)
    assert original["mismatches"] == 1
    assert original["examples"] == [{"input_code": 1, "actual_output_code": 0, "exact_output_code": 1}]
    for function in ("hard_swish", "hard_sigmoid"):
        revised = hard_codes(encoding, encoding, function, "posit8_es1_quire64_f28_accumulator")
        assert revised["finite_codes_tested"] == 255
        assert revised["mismatches"] == 0
