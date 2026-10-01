"""Hand-calculated packing, metadata and bandwidth witnesses."""

import json
from pathlib import Path

import pytest

from public.analysis.memory_model import Organization, manifest, tensor_layout, throughput_roof
from tools.analysis.memory_model import format_rows


def test_all_accepted_manifests_are_covered():
    index, rows, roofs, hypothetical, mapped_per_row = format_rows()
    assert index["accepted_count"] == 25
    assert len(rows) == 50
    assert len(roofs) == 25
    assert {r["format"] for r in rows} == {m["name"] for m in index["manifests"]}
    assert {r["format"] for r in hypothetical} == {"bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3"}
    assert all(r["status"] == "hypothetical_not_accepted_configuration" for r in hypothetical)
    assert len(mapped_per_row) == sum(m["family"] in ("integer", "codebook", "binary", "ternary") for m in index["manifests"])


def test_nondivisible_width_and_row_alignment():
    x = tensor_layout([2, 3], "int5", Organization(word_bits=32, banks=2),
                      packing_axis=1, scale_count=2)
    assert x["ideal_payload_bits"] == 30
    assert x["payload"]["words"] == 2  # one 15-bit row in each word
    assert x["payload"]["allocated_bits"] == 64
    assert x["metadata"]["raw_bits"] == 128  # two assumed 64-bit mapped scales
    assert x["metadata"]["allocated_bits"] == 128
    assert x["effective_encoded_bits_per_value"] == 32
    assert not x["cross_word_value_handling_required"]
    long_row = tensor_layout([2, 7], "int5", Organization(word_bits=32),
                             packing_axis=1, scale_count=0)
    assert long_row["cross_word_value_handling_required"]


def test_intrinsic_block_tails_and_axis_count():
    x = tensor_layout([2, 33], "mxfp4_e2m1", Organization(word_bits=64, banks=2),
                      packing_axis=1, scale_axis=1)
    assert x["block"]["block_size"] == 32
    assert x["block"]["tail_values_in_last_block"] == 1
    assert x["scale_count"] == 4
    assert x["metadata"]["raw_bits"] == 32
    assert x["payload"]["words"] == 6  # each 33-code row needs three words
    assert x["encoded_allocated_bits"] == 512
    alt = tensor_layout([2, 33], "mxfp4_e2m1", Organization(word_bits=64, banks=2),
                        packing_axis=1, scale_axis=1, block_size=16)
    assert alt["scale_count"] == 6
    assert alt["block"]["alternate_block_size"]


def test_non_last_axis_metadata_and_reorder():
    x = tensor_layout([2, 3, 5], "bfp6", scale_axis=1, packing_axis=1)
    assert x["scale_count"] == 10
    assert x["metadata"]["raw_bits"] == 80
    assert x["axis_reorder_required"]


def test_ports_banks_capacity_and_double_buffer():
    one = tensor_layout([100], "int8", Organization(word_bits=32, banks=2, read_ports=1,
                         write_ports=1, bank_depth_words=20), scale_count=0)
    two = tensor_layout([100], "int8", Organization(word_bits=32, banks=2, read_ports=2,
                         write_ports=1, bank_depth_words=20), scale_count=0)
    assert one["read_words"] == 25
    assert one["read_cycles"] == 13
    assert two["read_cycles"] == 7
    assert two["write_cycles"] == one["write_cycles"] == 13
    assert one["fits_capacity"]
    doubled = tensor_layout([100], "int8", Organization(word_bits=32, banks=2,
                            bank_depth_words=20), scale_count=0, double_buffer=True)
    assert doubled["encoded_allocated_bits"] == 2 * one["encoded_allocated_bits"]
    assert doubled["fits_capacity"] is False


def test_payload_metadata_bank_collision_costs_two_accesses():
    x = tensor_layout([4], "int4", Organization(word_bits=64, banks=2, read_ports=1),
                      scale_count=1)
    assert x["payload"]["words"] == x["metadata"]["words"] == 1
    assert x["bank_transfer_words"] == [2, 0]  # both separate arrays start at bank zero
    assert x["read_cycles"] == 2
    dual = tensor_layout([4], "int4", Organization(word_bits=64, banks=2, read_ports=2),
                         scale_count=1)
    assert dual["read_cycles"] == 1


def test_bias_and_accumulator_are_separate_storage():
    x = tensor_layout([4], "binary_pm1", Organization(word_bits=32), scale_count=0, bias_values=3,
                      bias_bits=32, accumulator_values=4, accumulator_bits=32)
    assert x["payload"]["allocated_bits"] == 32
    assert x["bias"]["allocated_bits"] == 96
    assert x["accumulator"]["allocated_bits"] == 128
    assert x["total_allocated_bits"] == 256
    assert x["physical_area_um2"] is None
    assert x["read_energy_pj"] is None


def test_total_padding_includes_bias_and_accumulator_and_zero_block_rejected():
    x = tensor_layout([4], "binary_pm1", Organization(word_bits=64), scale_count=0,
                      bias_values=3, bias_bits=32, accumulator_values=4,
                      accumulator_bits=32)
    assert x["padding_fraction"] == 0.2875
    with pytest.raises(ValueError, match="block_size"):
        tensor_layout([2, 33], "mxfp4_e2m1", block_size=0)


def test_same_bandwidth_roof_hand_example():
    x = tensor_layout([4], "int8", Organization(word_bits=32), scale_count=0)
    roof = throughput_roof(x, x, weight_reuse=1, activation_reuse=1,
                           compute_mac_per_cycle=10, compute_clock_mhz=500,
                           bandwidth_GBps=8)
    assert roof["bits_per_mac"] == 16
    assert roof["memory_mac_per_s"] == 4e9
    assert roof["compute_mac_per_s"] == 5e9
    assert roof["limiter"] == "memory"


@pytest.mark.parametrize("shape", [[], [0], [2, -1]])
def test_bad_boundary_shapes(shape):
    with pytest.raises(ValueError):
        tensor_layout(shape, "int4", scale_count=0)


def test_required_mapping_needs_configuration_scale_count():
    with pytest.raises(ValueError):
        tensor_layout([1], "int4")


def test_manifest_widths_drive_table():
    for name in ("int4", "int5", "int6", "fp7_e3m3", "int8", "ternary"):
        x = tensor_layout([8], name, scale_count=0 if manifest(name)["scaling"]["mode"] == "required_mapping" else None)
        assert x["ideal_payload_bits"] == 8 * manifest(name)["bits"]
