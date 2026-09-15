from fractions import Fraction

from public.analysis.phase3.fixed_mac_bounds import fixed_mac_bounds, quantum


def graph(scale="1", bias="0"):
    return {"inputs": {"x": {"format": "posit4_es0", "scales": [scale]}},
            "constants": {"w": {"shape": [1, 2], "codes": [4, 12], "encoding": {"format": "posit4_es0", "scales": ["1"]}}},
            "nodes": [{"name": "dot", "op": "linear", "inputs": ["x", "w"],
                       "attrs": {"output": {"format": "posit4_es0", "scales": ["1"]}, "bias": [bias],
                                 "accumulator": "posit8_es1_quire64_accumulator"}}]}


def test_quantum_and_actual_cancellation_bound():
    assert quantum([Fraction(0), Fraction(3, 16), Fraction(-9, 32)]) == Fraction(3, 32)
    r = fixed_mac_bounds(graph())["records"][0]
    assert r["status"] == "exact_products_and_sums_after_bias_store"
    # Opposite weights do not erase the required prefix headroom.
    assert Fraction(r["channels"][0]["maximum_absolute_dot_codes"]) > 0
    assert r["channels"][0]["products_on_grid"]


def test_fractional_grid_and_stored_bias_overflow_stay_pending():
    mapped = graph(scale="0.1")
    mapped["inputs"]["x"]["format"] = "int4"
    r = fixed_mac_bounds(mapped)
    assert r["pending_nodes"] == ["dot"]
    assert not r["records"][0]["channels"][0]["products_on_grid"]
    r = fixed_mac_bounds(graph(bias=str(2**40)))
    assert r["pending_nodes"] == ["dot"]
    assert not r["records"][0]["channels"][0]["overflow_excluded"]


def test_bias_rounding_error_is_retained_instead_of_claimed_exact():
    r = fixed_mac_bounds(graph(bias="0.1"))["records"][0]
    assert r["status"] == "exact_products_and_sums_after_bias_store"
    assert 0 < Fraction(r["channels"][0]["bias_storage_error"]) <= Fraction(1, 2**25)
