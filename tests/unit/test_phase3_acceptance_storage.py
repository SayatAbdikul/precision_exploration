import pytest
from public.analysis.phase3.storage import tensor_storage, shared_scale_count
from public.inference.tensor import Tensor, Encoding
from public.quantization.graph.executable import freeze_graph
from tools.phase3.acceptance import prove_integer_graph


def test_shared_metadata_counts_partial_blocks_for_each_row():
    assert shared_scale_count([3, 33], 1, 32) == 6
    storage = tensor_storage([3, 33], 4, scale_count=6, scale_bits=8)
    assert storage["nominal_payload_bits"] == 396
    assert storage["metadata_bits"] == 48
    assert storage["padding_bits"] == 4
    assert storage["byte_aligned_bits"] == 448


def graph(function="relu"):
    return freeze_graph(inputs={"x": Encoding("int8").document()},
        constants={"w": Tensor((1, 1), (1,), Encoding("int8"))},
        nodes=[{"name": "linear", "op": "linear", "inputs": ["x", "w"],
                "attrs": {"accumulator": "int32_accumulator", "output": Encoding("int8").document()}},
               {"name": "activation", "op": "activation", "inputs": ["linear"],
                "attrs": {"accumulator": "int32_accumulator", "function": function, "output": Encoding("int8").document()}}],
        outputs=["activation"], provenance={"kind": "test"})


def test_finite_int32_mac_and_relu_have_static_proof():
    result = prove_integer_graph(graph(), {"x": [1, 1]}, {"linear": [1, 1], "activation": [1, 1]})
    assert result["status"] == "accepted"


def test_integer_hard_swish_cannot_inherit_mac_width_acceptance():
    result = prove_integer_graph(graph("hard_swish"), {"x": [1, 1]}, {"linear": [1, 1], "activation": [1, 1]})
    assert result["status"] == "pending"
    assert "outside the proven" in result["reasons"][0]
