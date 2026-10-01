"""Hand-worked schedule, contention, spill and retained-source checks."""

from dataclasses import replace
from pathlib import Path

import pytest

from public.analysis.memory_model import Organization
from public.analysis.memory_schedule import ConvLayer, Schedule, evaluate_layer
from tools.analysis.memory_schedule import retained_layers, scenarios


SMALL = ConvLayer("hand", "one", "pointwise", (1, 1, 1, 2),
                  (1, 1, 1, 2), (1, 1, 1, 1), True)


def hand_setup(**kwargs):
    base = Schedule(channels_per_tile=1, positions_per_tile=2, reduction_per_tile=1,
                    organization=Organization(word_bits=32, banks=1, bank_depth_words=16,
                                              clock_mhz=100), compute_mac_per_cycle=2,
                    external_bandwidth_GBps=1, double_buffer_inputs=False,
                    cache_mapped_scales=False)
    return replace(base, **kwargs)


def test_hand_counted_tile_and_shared_port_contention():
    result = evaluate_layer(SMALL, "int8", hand_setup())
    # One W word, two row-aligned A words, one O word, one bias word,
    # plus three separate 64-bit scale words (two 32-bit transfers each).
    assert result["weight_bits"] == 32
    assert result["input_bits"] == 64
    assert result["output_bits"] == result["bias_bits"] == 32
    assert result["scale_bits"] == 192
    assert result["external_bits"] == result["peak_local_bank_bits"] == 352
    assert result["peak_single_bank_bits"] == 352
    assert result["peak_accumulator_register_bits"] == 128
    assert result["local_read_cycles"] == result["local_write_cycles"] == 10
    assert result["compute_cycles"] == 1
    assert result["external_cycles"] == 5
    assert result["limiter"] == "local_read"
    dual = evaluate_layer(SMALL, "int8", hand_setup(
        organization=Organization(word_bits=32, banks=1, read_ports=2,
                                  write_ports=2, bank_depth_words=16, clock_mhz=100)))
    assert dual["local_read_cycles"] == dual["local_write_cycles"] == 5


def test_capacity_and_scale_cache():
    tight = evaluate_layer(SMALL, "int8", hand_setup(
        organization=Organization(word_bits=32, banks=1, bank_depth_words=10,
                                  clock_mhz=100)))
    assert not tight["fits_local_bank"]
    assert tight["analytical_mac_per_s"] is None
    cached = evaluate_layer(SMALL, "int8", hand_setup(cache_mapped_scales=True))
    assert cached["external_bits"] == 352  # one tile: caching saves nothing


def test_depthwise_input_is_per_channel_and_spill_adds_traffic():
    normal = ConvLayer("hand", "normal", "pointwise", (1, 2, 1, 2),
                       (1, 2, 1, 2), (2, 2, 1, 1))
    depth = ConvLayer("hand", "depth", "depthwise", (1, 2, 1, 2),
                      (1, 2, 1, 2), (2, 1, 1, 2))
    setup = hand_setup(channels_per_tile=2, reduction_per_tile=2)
    regular = evaluate_layer(normal, "int4", setup)
    separate = evaluate_layer(depth, "int4", setup)
    assert separate["input_bits"] > regular["input_bits"]
    split = replace(setup, reduction_per_tile=1)
    no_spill = evaluate_layer(normal, "int4", split)
    spill = evaluate_layer(normal, "int4", replace(split, spill_accumulator_after_k_tile=True))
    assert spill["spill_bits"] == 2 * 4 * 64  # one boundary, four accumulators
    assert spill["external_bits"] == no_spill["external_bits"] + spill["spill_bits"]
    assert spill["local_read_cycles"] > no_spill["local_read_cycles"]


def test_common_packing_sensitivity_and_roof_limits():
    layer = ConvLayer("hand", "wide", "pointwise", (1, 32, 1, 8),
                      (1, 8, 1, 8), (8, 32, 1, 1))
    setup = scenarios()["base_64b_8x8"]
    four = evaluate_layer(layer, "int4", setup)
    five = evaluate_layer(layer, "int5", setup)
    assert five["external_bits"] > four["external_bits"]
    assert four["model_c_products"] == five["model_c_products"] == 8 * 8 * 32
    small = evaluate_layer(layer, "int4", scenarios()["low_reuse_64b_1x1"])
    assert small["external_bits_per_mac"] > four["external_bits_per_mac"]


def test_invalid_layers_and_schedule():
    with pytest.raises(ValueError):
        ConvLayer("x", "bad", "depthwise", (1, 2, 1, 1), (1, 3, 1, 1),
                  (3, 1, 1, 1))
    with pytest.raises(ValueError):
        Schedule(channels_per_tile=0)


def test_retained_shapes_are_verified_read_only():
    root = Path("/home/maveric/precision_exploration")
    if not (root / "results/summaries/phase3-hardware-priors.json").exists():
        pytest.skip("retained observation root is unavailable on this host")
    layers, source = retained_layers(root)
    assert len(layers) == len(source["layers"]) == 5
    assert {x.kind for x in layers} == {"standard", "pointwise", "depthwise"}
