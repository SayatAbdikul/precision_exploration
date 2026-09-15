import pytest

from public.analysis.phase3.resources import graph_resources, encoded_storage


def encoding(name="int4"):
    return {"format": name, "scales": ["1"]}


def test_residual_liveness_retains_skip_until_join_and_counts_views_as_copies():
    graph = {"inputs": {"x": encoding()}, "constants": {}, "nodes": [
        {"name": "a", "op": "activation", "inputs": ["x"],
         "attrs": {"function": "relu", "output": encoding()}},
        {"name": "b", "op": "activation", "inputs": ["a"],
         "attrs": {"function": "relu", "output": encoding()}},
        {"name": "join", "op": "elementwise", "inputs": ["x", "b"],
         "attrs": {"operation": "add", "output": encoding(), "accumulator": "int64_accumulator"}},
        {"name": "view", "op": "flatten", "inputs": ["join"], "attrs": {"start_dim": 1}}],
        "outputs": ["view"]}
    result = graph_resources(graph, {name: [1, 16] for name in ("x", "a", "b", "join", "view")})
    # Each buffer: 16*4 payload + 64-bit mapping metadata = 128 bits.
    assert result["peak_live_activation_bits"] == 384
    assert result["peak_live_buffers"] == ["a", "b", "x"]
    assert result["layers"][2]["live_activation_bits_before_release"] == 384
    assert result["layers"][3]["live_activation_bits_before_release"] == 256
    assert result["maximum_sequential_accumulator_bits"] == 64
    assert result["maximum_output_accumulator_bank_bits"] == 1024
    assert result["arithmetic_counts"] == {"relu_evaluations": 32, "elementwise_add": 16}


def test_partial_shared_blocks_and_flatten_scale_layout():
    domain = {"format": "mxfp4_e2m1", "axis": 2, "block_size": 32, "policy": "intrinsic_mse_v1"}
    graph = {"inputs": {"x": domain}, "constants": {}, "nodes": [
        {"name": "flat", "op": "flatten", "inputs": ["x"], "attrs": {"start_dim": 1}}], "outputs": ["flat"]}
    result = graph_resources(graph, {"x": [1, 3, 33], "flat": [1, 99]})
    assert encoded_storage([1, 3, 33], domain, 64)["metadata_bits"] == 48
    assert result["layers"][0]["activation"]["metadata_bits"] == 32
    assert result["peak_live_activation_bits"] == 448+432


def test_grouped_mac_counts_and_full_patch_materialization():
    domain = {"format": "bfp6", "axis": 1, "block_size": 32, "policy": "intrinsic_mse_v1"}
    graph = {"inputs": {"x": domain}, "constants": {"w": {"shape": [4, 18]}}, "nodes": [
        {"name": "conv", "op": "block_conv2d", "inputs": ["x", "w"],
         "attrs": {"output": domain, "accumulator": "fp64_e11m52_accumulator", "groups": 2,
                   "patch_format": "bfp6", "bias": ["0"]*4}}], "outputs": ["conv"]}
    result = graph_resources(graph, {"x": [1, 4, 4, 4], "conv": [1, 4, 2, 2]})
    assert result["arithmetic_counts"]["model_c_products"] == 16*18
    assert result["arithmetic_counts"]["stored_bias_adds"] == 16
    patch = result["layers"][0]["fully_materialized_patch_buffer"]
    assert patch["values"] == 1*2*2*2*18
    assert patch["metadata_bits"] == 8*8


def test_missing_shape_fails_instead_of_underestimating():
    with pytest.raises(ValueError, match="missing observed"):
        graph_resources({"inputs": {"x": encoding()}, "constants": {}, "nodes": [], "outputs": ["x"]}, {})
