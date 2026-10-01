"""Versioned analytical packed-memory and bandwidth model.

Level A is raw encoded payload. Level B is a declared word/bank organization.
Neither level implies an available physical memory macro or area/energy result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil, gcd, prod
from pathlib import Path

from public.analysis.phase3.storage import shared_scale_count


SCHEMA_VERSION = "memory-model-1.0.0"
MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "formats/manifests/accepted"


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def manifest(name):
    """Load the accepted datatype definition, never a private copy of its width."""
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        raise ValueError("invalid format name")
    data = json.loads((MANIFEST_ROOT / f"{name}.json").read_text())
    if data["name"] != name:
        raise ValueError("manifest name mismatch")
    return data


@dataclass(frozen=True)
class Organization:
    word_bits: int = 64
    banks: int = 1
    read_ports: int = 1
    write_ports: int = 1
    clock_mhz: float = 500.0
    bank_depth_words: int | None = None

    def __post_init__(self):
        for name in ("word_bits", "banks", "read_ports", "write_ports"):
            _positive(getattr(self, name), name)
        if self.word_bits % 8:
            raise ValueError("word_bits must be byte aligned")
        if not 0 < self.clock_mhz < float("inf"):
            raise ValueError("clock_mhz must be finite and positive")
        if self.bank_depth_words is not None:
            _positive(self.bank_depth_words, "bank_depth_words")


def _array(raw_bits, word_bits, banks, lines=1):
    """Pack each declared contiguous line independently, stripe words across banks."""
    if raw_bits and raw_bits % lines:
        raise ValueError("array bits must split into equal lines")
    words = lines * ((raw_bits // lines + word_bits - 1) // word_bits) if raw_bits else 0
    words_per_bank = ceil(words / banks)
    allocated_bits = words_per_bank * banks * word_bits
    return {"raw_bits": raw_bits, "words": words,
            "words_per_bank": words_per_bank, "allocated_bits": allocated_bits,
            "padding_bits": allocated_bits - raw_bits}


def _bank_loads(words, banks):
    """Each separate array starts striping at bank zero."""
    q, r = divmod(words, banks)
    return [q + (bank < r) for bank in range(banks)]


def tensor_layout(shape, format_name, organization=Organization(), *, packing_axis=None,
                  scale_count=None, scale_axis=None, block_size=None,
                  mapped_scale_bits=64, bias_values=0, bias_bits=0,
                  accumulator_values=0, accumulator_bits=0, double_buffer=False):
    """Return Level A/B storage for one tensor and optional bias/accumulator banks.

    `packing_axis` places one word boundary at each axis line. An axis other
    than the last presumes an explicit reorder/gather in hardware. `None` packs
    one flat contiguous stream. Intrinsic shared metadata uses the manifest's
    block size unless `block_size` is explicitly overridden as a hypothetical.
    Required-mapping metadata count is configuration-specific and must be passed.
    The 64-bit mapped scale default is analytical; it is not an exact-rational
    implementation claim.
    """
    if not shape or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("shape needs positive dimensions")
    if packing_axis is not None and (type(packing_axis) is not int or not 0 <= packing_axis < len(shape)):
        raise ValueError("invalid packing_axis")
    fmt = manifest(format_name)
    bits = fmt["bits"]
    n = prod(shape)
    lines = prod(shape[:packing_axis] + shape[packing_axis + 1:]) if packing_axis is not None else 1
    line_values = shape[packing_axis] if packing_axis is not None else n
    payload = _array(lines * line_values * bits, organization.word_bits, organization.banks, lines)
    mode = fmt["scaling"]["mode"]
    if mode == "intrinsic_shared":
        block_size = _positive(fmt["block"]["block_size"] if block_size is None else block_size, "block_size")
        axis = scale_axis if scale_axis is not None else len(shape) - 1
        count = shared_scale_count(list(shape), axis, block_size)
        if scale_count is not None and (type(scale_count) is not int or scale_count != count):
            raise ValueError("intrinsic scale_count must match the manifest block layout")
        scale_bits = fmt["scaling"]["scale_bits"]
        tail = shape[axis] % block_size
        block_info = {"axis": axis, "block_size": block_size, "blocks": count,
                      "tail_values_in_last_block": tail or block_size,
                      "alternate_block_size": block_size != fmt["block"]["block_size"]}
    elif mode == "required_mapping":
        if block_size is not None or scale_axis is not None:
            raise ValueError("required mapping has no intrinsic block layout")
        if scale_count is None:
            raise ValueError("required-mapping scale_count must be supplied from a configuration")
        if type(scale_count) is not int or scale_count < 0:
            raise ValueError("invalid scale_count")
        count, scale_bits, block_info = scale_count, mapped_scale_bits, None
        _positive(scale_bits, "mapped_scale_bits")
    else:
        if block_size is not None or scale_axis is not None:
            raise ValueError("unscaled format has no block layout")
        if scale_count not in (None, 0):
            raise ValueError("unscaled format cannot carry scale metadata")
        count, scale_bits, block_info = 0, 0, None
    metadata = _array(count * scale_bits, organization.word_bits, organization.banks)
    for v, b, label in ((bias_values, bias_bits, "bias"),
                        (accumulator_values, accumulator_bits, "accumulator")):
        if type(v) is not int or v < 0 or type(b) is not int or b < 0 or (v and not b):
            raise ValueError(f"invalid {label} storage")
    bias = _array(bias_values * bias_bits, organization.word_bits, organization.banks)
    accumulator = _array(accumulator_values * accumulator_bits, organization.word_bits, organization.banks)
    activation_copies = 2 if double_buffer else 1
    encoded_bits = (payload["allocated_bits"] + metadata["allocated_bits"]) * activation_copies
    total_bits = encoded_bits + bias["allocated_bits"] + accumulator["allocated_bits"]
    capacity_bits = None if organization.bank_depth_words is None else organization.banks * organization.bank_depth_words * organization.word_bits
    if capacity_bits is not None and total_bits > capacity_bits:
        fits = False
    else:
        fits = None if capacity_bits is None else True
    transfer_words = payload["words"] + metadata["words"]
    bank_loads = [p + m for p, m in zip(_bank_loads(payload["words"], organization.banks),
                                         _bank_loads(metadata["words"], organization.banks))]
    read_cycles = max(ceil(load / organization.read_ports) for load in bank_loads)
    write_cycles = max(ceil(load / organization.write_ports) for load in bank_loads)
    # Useful encoded values per second; metadata and padding consume access slots.
    delivered_values_s = n * organization.clock_mhz * 1e6 / read_cycles if read_cycles else None
    crossing_period = organization.word_bits // gcd(organization.word_bits, bits)
    crosses_word = any((i * bits) % organization.word_bits + bits > organization.word_bits
                       for i in range(min(line_values, crossing_period)))
    return {
        "schema_version": SCHEMA_VERSION, "evidence_level": "B_packed_logical_analytical",
        "format": format_name, "family": fmt["family"], "shape": list(shape), "values": n,
        "word_bits": organization.word_bits, "banks": organization.banks,
        "read_ports": organization.read_ports, "write_ports": organization.write_ports,
        "clock_mhz": organization.clock_mhz, "packing_axis": packing_axis,
        "axis_reorder_required": packing_axis is not None and packing_axis != len(shape) - 1,
        "cross_word_value_handling_required": crosses_word,
        "packing_unpacking_required": bits < organization.word_bits,
        "scale_addressing_required": count > 0,
        "double_buffer": double_buffer, "scale_mode": mode, "scale_count": count,
        "scale_bits": scale_bits, "block": block_info,
        "ideal_payload_bits": n * bits, "ideal_bits_per_value": bits,
        "payload": payload, "metadata": metadata, "bias": bias, "accumulator": accumulator,
        "encoded_allocated_bits": encoded_bits, "total_allocated_bits": total_bits,
        "effective_encoded_bits_per_value": encoded_bits / n,
        "metadata_fraction": metadata["raw_bits"] * activation_copies / total_bits if total_bits else 0,
        "padding_fraction": (sum(x["padding_bits"] for x in (payload, metadata)) * activation_copies
                             + bias["padding_bits"] + accumulator["padding_bits"]) / total_bits if total_bits else 0,
        "capacity_bits": capacity_bits, "fits_capacity": fits,
        "capacity_utilization": total_bits / capacity_bits if capacity_bits else None,
        "whole_tensor_capacity_count": capacity_bits // total_bits if capacity_bits is not None else None,
        "usable_value_capacity_for_identical_tensors": (capacity_bits // total_bits) * n if capacity_bits is not None else None,
        "free_capacity_bits": capacity_bits - total_bits if capacity_bits is not None and fits else None,
        "read_words": transfer_words, "write_words": transfer_words,
        "bank_transfer_words": bank_loads,
        "read_cycles": read_cycles, "write_cycles": write_cycles,
        "read_traffic_bits": transfer_words * organization.word_bits,
        "write_traffic_bits": transfer_words * organization.word_bits,
        "read_traffic_bits_per_value": transfer_words * organization.word_bits / n,
        "delivered_values_per_s": delivered_values_s,
        "delivered_payload_gbps": delivered_values_s * bits / 1e9 if delivered_values_s is not None else None,
        "port_cost_status": "uncharacterized", "physical_memory_status": "unverified",
        "physical_area_um2": None, "read_energy_pj": None, "write_energy_pj": None,
        "physical_latency_ns": None,
        "mapped_scale_status": "analytical_width_not_exact_rational_implementation" if mode == "required_mapping" else "manifest_intrinsic_or_none",
    }


def throughput_roof(weight, activation, *, weight_reuse, activation_reuse,
                    compute_mac_per_cycle, compute_clock_mhz, bandwidth_GBps):
    """Analytical same-bandwidth MAC roof with declared value reuse factors."""
    for name, val in (("weight_reuse", weight_reuse), ("activation_reuse", activation_reuse),
                      ("compute_mac_per_cycle", compute_mac_per_cycle), ("compute_clock_mhz", compute_clock_mhz),
                      ("bandwidth_GBps", bandwidth_GBps)):
        if not isinstance(val, (int, float)) or not 0 < val < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    bits_per_mac = (weight["read_traffic_bits_per_value"] / weight_reuse
                    + activation["read_traffic_bits_per_value"] / activation_reuse)
    memory_mac_s = bandwidth_GBps * 8e9 / bits_per_mac
    compute_mac_s = compute_mac_per_cycle * compute_clock_mhz * 1e6
    return {"weight_reuse": weight_reuse, "activation_reuse": activation_reuse,
            "bits_per_mac": bits_per_mac, "bandwidth_GBps": bandwidth_GBps,
            "compute_mac_per_s": compute_mac_s, "memory_mac_per_s": memory_mac_s,
            "limited_mac_per_s": min(compute_mac_s, memory_mac_s),
            "limiter": "memory" if memory_mac_s < compute_mac_s else "compute",
            "evidence_level": "analytical_roof"}
