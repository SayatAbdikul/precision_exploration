from copy import deepcopy
from fractions import Fraction

import pytest

from public.analysis.phase3.fp64_bounds import error_bound, HALF_MIN_SUBNORMAL, MAX_FINITE
from public.inference.reference.arithmetic import format_named, model_c
from tools.analysis.phase3_fp64_revisions import verify_graph_revision


@pytest.mark.parametrize("pairs,bias", [
    ([(Fraction(1, 10), Fraction(3, 7))]*37, Fraction(17, 13)),
    ([(Fraction(2**52), Fraction(1)), (Fraction(1, 3), Fraction(1)),
      (Fraction(-2**52), Fraction(1))], None),
    ([(HALF_MIN_SUBNORMAL, Fraction(1))]*4, HALF_MIN_SUBNORMAL),
    ([(Fraction(1, 10**40), Fraction(1, 10**40))]*7, Fraction(-1, 10**80)),
])
def test_bound_contains_actual_model_c_rounding_with_cancellation_and_subnormals(pairs, bias):
    exact = sum(a*b for a, b in pairs)+(bias or 0)
    acc = format_named("fp64_e11m52_accumulator")
    actual = acc.decode(model_c(pairs, acc, bias=bias))
    bound = error_bound(sum(abs(a*b) for a, b in pairs), len(pairs), absolute_bias=None if bias is None else abs(bias))
    assert bound["overflow_excluded"]
    assert abs(actual-exact) <= bound["absolute_error_bound"]


def test_overflow_and_invalid_rounding_count_cannot_pass_bound():
    assert not error_bound(MAX_FINITE*2, 1)["overflow_excluded"]
    with pytest.raises(ValueError, match="n.u"):
        error_bound(1, 2**53)


def test_revision_verifier_rejects_weight_or_output_scale_changes():
    before = {"schema_version": "2.0.0", "inputs": {"x": {}}, "constants": {"w": {"codes": [1, 2]}},
              "outputs": ["layer"], "provenance": {}, "manifest_hashes": {"int8": "operand", "int32_accumulator": "old"},
              "nodes": [{"name": "layer", "op": "linear", "inputs": ["x", "w"],
                         "attrs": {"accumulator": "int32_accumulator", "output": {"scales": ["1"]}}}]}
    after = deepcopy(before)
    after["nodes"][0]["attrs"]["accumulator"] = "fp64_e11m52_accumulator"
    after["manifest_hashes"] = {"int8": "operand", "fp64_e11m52_accumulator": "new"}
    assert verify_graph_revision(before, after, "int32_accumulator") == 1
    after["constants"]["w"]["codes"][0] = 3
    with pytest.raises(ValueError, match="constants"):
        verify_graph_revision(before, after, "int32_accumulator")
    after["constants"] = deepcopy(before["constants"])
    after["nodes"][0]["attrs"]["output"]["scales"] = ["2"]
    with pytest.raises(ValueError, match="operator attributes"):
        verify_graph_revision(before, after, "int32_accumulator")
