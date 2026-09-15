from itertools import product
from fractions import Fraction

import pytest

from public.analysis.phase3.finite_fp64_bounds import finite_codebook, finite_mac_bounds
from public.analysis.phase3.fp64_bounds import integer_mac_bounds
from public.inference.reference.arithmetic import format_named, model_c
from public.inference.tensor import Encoding, Tensor


def graph_for(name, codes, bias=Fraction(1, 3), scale="1"):
    encoding = Encoding(name, (scale,))
    return {"inputs": {"x": encoding.document()},
            "constants": {"w": Tensor((1, len(codes)), tuple(codes), encoding).document()},
            "nodes": [{"name": "y", "op": "linear", "inputs": ["x", "w"],
                       "attrs": {"accumulator": "fp64_e11m52_accumulator", "bias": [bias], "output": encoding.document()}}]}


@pytest.mark.parametrize("name", ["fp4_e2m1", "log4", "nf4", "binary_pm1", "ternary"])
def test_all_two_operand_sequences_fit_bound_for_nonuniform_codebooks(name):
    encoding = Encoding(name)
    values = finite_codebook(encoding)
    ordered = sorted(values, key=values.get)
    graph = graph_for(name, [ordered[-1], ordered[0]])
    result = finite_mac_bounds(graph)
    assert not result["pending"]
    bound = result["macs"][0]
    assert bound["overflow_excluded"]
    acc = format_named("fp64_e11m52_accumulator")
    weights = [values[ordered[-1]], values[ordered[0]]]
    for inputs in product(values.values(), repeat=2):
        pairs = list(zip(inputs, weights))
        exact = sum(a*b for a, b in pairs)+Fraction(1, 3)
        actual = Fraction(acc.decode(model_c(pairs, acc, bias=Fraction(1, 3))))
        assert abs(actual-exact) <= Fraction(bound["maximum_absolute_error_bound"])


def test_integer_specialization_agrees_with_existing_independent_bound():
    graph = graph_for("int4", [7, 8, 3], scale="0.375")
    actual = finite_mac_bounds(graph)["macs"][0]
    previous = integer_mac_bounds(graph)["macs"][0]
    assert actual["maximum_absolute_error_bound"] == previous["maximum_absolute_error_bound"]
    assert actual["bound_in_minimum_output_spacings"] == previous["bound_in_output_steps"]


def test_nonfinite_weights_cannot_receive_finite_bound():
    fmt = format_named("fp8_e5m2")
    nonfinite = next(c for c in range(256) if not fmt.decode(c).is_finite())
    result = finite_mac_bounds(graph_for("fp8_e5m2", [nonfinite]))
    assert not result["macs"]
    assert "nonfinite" in result["pending"][0]["reason"]


def test_shared_block_mac_is_pending_not_silently_counted_as_covered():
    graph = graph_for("int4", [1, 2])
    graph["nodes"][0]["op"] = "block_conv2d"
    result = finite_mac_bounds(graph)
    assert not result["macs"]
    assert result["pending"] == [{"node": "y", "reason": "shared block MAC needs scale-aware bounds"}]
