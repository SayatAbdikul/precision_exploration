from fractions import Fraction

from public.analysis.phase3.fp64_nonmac import mean_boundary_margin, prove_nonmac


def test_nearest_integer_sum_margin_matches_complete_enumeration():
    a, b, count = Fraction(1, 10), Fraction(101, 1007), 3
    brute = min(abs(Fraction(s)*a/count-(Fraction(c)+Fraction(1, 2))*b)
                for s in range(-8*count, 7*count+1) for c in range(-8, 7))
    assert brute > 0
    assert mean_boundary_margin(a, count, b, 4, 4) == brute


def pool_graph(input_scale, output_scale):
    x = {"format": "int4", "scales": [input_scale]}
    return {"inputs": {"x": x}, "nodes": [{"name": "pool", "op": "adaptive_average_pool2d", "inputs": ["x"],
             "attrs": {"output_size": 1, "accumulator": "fp64_e11m52_accumulator",
                       "output": {"format": "int4", "scales": [output_scale]}}}]}


def test_exact_mean_tie_stays_pending_but_separated_boundaries_are_proved():
    tied = prove_nonmac(pool_graph("1", "1"), {"x": [1, 1, 1, 2]})
    assert tied["pending_nodes"] == ["pool"]
    assert tied["records"][0]["boundary_margin"] == "0"
    separated = prove_nonmac(pool_graph("0.1", "0.101"), {"x": [1, 1, 1, 3]})
    assert separated["pending_nodes"] == []
    assert separated["records"][0]["status"] == "exact_output_codes"
