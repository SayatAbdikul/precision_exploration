"""Analytical, format-comparable tiled convolution schedule (version 1).

The output, weight and gathered-input tile shapes are common across formats.
No conversion, halo reuse, SRAM implementation, or graph-quality equivalence is
implied. All traffic is logical word traffic under the declared schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

from public.analysis.memory_model import Organization, manifest, tensor_layout


SCHEMA_VERSION = "memory-schedule-1.0.0"


@dataclass(frozen=True)
class ConvLayer:
    model: str
    name: str
    kind: str  # standard, pointwise, or depthwise
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    bias_present: bool = True

    def __post_init__(self):
        if self.kind not in ("standard", "pointwise", "depthwise"):
            raise ValueError("unsupported convolution kind")
        if any(len(s) != 4 or any(type(v) is not int or v <= 0 for v in s)
               for s in (self.input_shape, self.output_shape, self.weight_shape)):
            raise ValueError("invalid NCHW/OIHW shapes")
        ni, ci, _, _ = self.input_shape
        no, co, _, _ = self.output_shape
        wo, wi, kh, kw = self.weight_shape
        if ni != no or co != wo or (self.kind == "depthwise" and (ci != co or wi != 1)):
            raise ValueError("inconsistent convolution shapes")
        if self.kind != "depthwise" and wi != ci:
            raise ValueError("grouped convolution is outside this common schedule")
        if self.kind == "pointwise" and (kh, kw) != (1, 1):
            raise ValueError("pointwise must have a 1x1 kernel")

    @property
    def output_channels(self):
        return self.output_shape[1]

    @property
    def positions(self):
        return self.output_shape[0] * prod(self.output_shape[2:])

    @property
    def reduction(self):
        return prod(self.weight_shape[1:])


@dataclass(frozen=True)
class Schedule:
    channels_per_tile: int = 8
    positions_per_tile: int = 8
    reduction_per_tile: int = 32
    organization: Organization = Organization(word_bits=64, banks=4, read_ports=1,
                                               write_ports=1, clock_mhz=500,
                                               bank_depth_words=256)  # 8 KiB
    compute_mac_per_cycle: int = 64
    external_bandwidth_GBps: float = 8.0
    accumulator_bits: int = 64
    bias_bits: int = 32
    double_buffer_inputs: bool = True
    cache_mapped_scales: bool = True
    spill_accumulator_after_k_tile: bool = False

    def __post_init__(self):
        for name in ("channels_per_tile", "positions_per_tile", "reduction_per_tile",
                     "compute_mac_per_cycle", "accumulator_bits", "bias_bits"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive integer")
        if not 0 < self.external_bandwidth_GBps < float("inf"):
            raise ValueError("external_bandwidth_GBps must be finite and positive")


def _parts(length, tile):
    q, r = divmod(length, tile)
    return [(tile, q)] * bool(q) + ([(r, 1)] if r else [])


def _layout(shape, fmt, org, *, bias_values=0, accumulator_values=0,
            bias_bits=0, accumulator_bits=0):
    mode = manifest(fmt)["scaling"]["mode"]
    kwargs = {"packing_axis": len(shape) - 1, "bias_values": bias_values,
              "bias_bits": bias_bits, "accumulator_values": accumulator_values,
              "accumulator_bits": accumulator_bits}
    if mode == "required_mapping":
        kwargs["scale_count"] = 1
    elif mode == "intrinsic_shared":
        kwargs["scale_axis"] = len(shape) - 1
    return tensor_layout(list(shape), fmt, org, **kwargs)


def _bank_words(layout, fields, banks):
    loads = [0] * banks
    for field in fields:
        words = layout[field]["words"]
        q, r = divmod(words, banks)
        for bank in range(banks):
            loads[bank] += q + (bank < r)
    return loads


def _sum_loads(*loads):
    return [sum(parts) for parts in zip(*loads)]


def evaluate_layer(layer: ConvLayer, format_name: str, schedule: Schedule = Schedule()):
    """Evaluate a tile-loop schedule with no reuse between spatial/channel tiles.

    A standard/pointwise input patch is gathered once per spatial/reduction
    tile and broadcast across output channels in that tile. Depthwise has one
    independent input patch per channel. Weights reload for each spatial tile;
    inputs reload for each output-channel tile. Input halo overlap is not reused.
    Static mapped scales are fetched once externally when caching is enabled,
    but still read through local bank ports on each tile. An accumulator tile is
    held in registers, reported separately from packed bank occupancy. Optional
    reduction-tile spills add explicit read/write traffic and port accesses.
    """
    org = schedule.organization
    if format_name != manifest(format_name)["name"]:
        raise ValueError("invalid format")
    banks = org.banks
    total_mac = layer.output_channels * layer.positions * layer.reduction
    totals = {k: 0 for k in ("external_bits", "weight_bits", "input_bits", "output_bits",
                            "bias_bits", "scale_bits", "spill_bits", "local_read_cycles",
                            "local_write_cycles", "compute_cycles", "tile_instances")}
    max_bank_bits = max_acc_bits = peak_single_bank_bits = 0
    peak_input_bank_bits = peak_weight_bank_bits = peak_output_bank_bits = 0
    all_fit = True
    mapped = manifest(format_name)["scaling"]["mode"] == "required_mapping"
    full_input = _layout(layer.input_shape, format_name, org)
    full_output = _layout(layer.output_shape, format_name, org)
    full_activation_pair_bits = (full_input["encoded_allocated_bits"]
                                 + full_output["encoded_allocated_bits"])
    for m, nm in _parts(layer.output_channels, schedule.channels_per_tile):
        for p, np in _parts(layer.positions, schedule.positions_per_tile):
            outer_multiplicity = nm * np
            out = _layout((m, p), format_name, org,
                          bias_values=m if layer.bias_present else 0,
                          bias_bits=schedule.bias_bits if layer.bias_present else 0)
            acc_bits = m * p * schedule.accumulator_bits
            max_acc_bits = max(max_acc_bits, acc_bits)
            k_parts = _parts(layer.reduction, schedule.reduction_per_tile)
            # Output conversion needs the stored output scale after reduction.
            output_read = _bank_words(out, ("metadata", "bias"), banks)
            output_write = _bank_words(out, ("payload", "metadata"), banks)
            output_external = sum(out[x]["words"] for x in ("payload", "metadata", "bias")) * org.word_bits
            totals["output_bits"] += outer_multiplicity * out["payload"]["words"] * org.word_bits
            totals["bias_bits"] += outer_multiplicity * out["bias"]["words"] * org.word_bits
            totals["scale_bits"] += outer_multiplicity * out["metadata"]["words"] * org.word_bits
            for k, nk in k_parts:
                w = _layout((m, k), format_name, org)
                a = _layout((m, p, k) if layer.kind == "depthwise" else (p, k), format_name, org)
                instances = outer_multiplicity * nk
                totals["tile_instances"] += instances
                w_bits = sum(w[x]["words"] for x in ("payload", "metadata")) * org.word_bits
                a_bits = sum(a[x]["words"] for x in ("payload", "metadata")) * org.word_bits
                totals["weight_bits"] += instances * w["payload"]["words"] * org.word_bits
                totals["input_bits"] += instances * a["payload"]["words"] * org.word_bits
                totals["scale_bits"] += instances * (w["metadata"]["words"] + a["metadata"]["words"]) * org.word_bits
                # W/A tiles are independently packed and restart bank striping.
                copies = 2 if schedule.double_buffer_inputs else 1
                bank_bits = (w["encoded_allocated_bits"] + a["encoded_allocated_bits"]) * copies
                bank_bits += out["encoded_allocated_bits"] + out["bias"]["allocated_bits"]
                w_bank = _bank_words(w, ("payload", "metadata"), banks)
                a_bank = _bank_words(a, ("payload", "metadata"), banks)
                out_bank = _bank_words(out, ("payload", "metadata", "bias"), banks)
                bank_occupancy = [copies * (wb + ab) + ob
                                  for wb, ab, ob in zip(w_bank, a_bank, out_bank)]
                peak_single_bank_bits = max(peak_single_bank_bits, max(bank_occupancy) * org.word_bits)
                peak_input_bank_bits = max(peak_input_bank_bits, a["encoded_allocated_bits"] * copies)
                peak_weight_bank_bits = max(peak_weight_bank_bits, w["encoded_allocated_bits"] * copies)
                peak_output_bank_bits = max(peak_output_bank_bits,
                                            out["encoded_allocated_bits"] + out["bias"]["allocated_bits"])
                max_bank_bits = max(max_bank_bits, bank_bits)
                if org.bank_depth_words is not None and max(bank_occupancy) > org.bank_depth_words:
                    all_fit = False
                reads = _sum_loads(_bank_words(w, ("payload", "metadata"), banks),
                                   _bank_words(a, ("payload", "metadata"), banks))
                writes = list(reads)  # loading W/A into the local banks
                # Count final bias, output scale, output write exactly once per
                # output tile, on the last reduction segment.
                base_read = max(ceil(x / org.read_ports) for x in reads)
                base_write = max(ceil(x / org.write_ports) for x in writes)
                totals["local_read_cycles"] += instances * base_read
                totals["local_write_cycles"] += instances * base_write
                if k == k_parts[-1][0]:
                    # Exactly one last K segment per M/P tile combines its
                    # W/A accesses with bias, output scale, and output write.
                    combined_read = _sum_loads(reads, output_read)
                    combined_write = _sum_loads(writes, output_write)
                    totals["local_read_cycles"] += outer_multiplicity * (
                        max(ceil(x / org.read_ports) for x in combined_read) - base_read)
                    totals["local_write_cycles"] += outer_multiplicity * (
                        max(ceil(x / org.write_ports) for x in combined_write) - base_write)
                totals["compute_cycles"] += instances * ceil(m * p * k / schedule.compute_mac_per_cycle)
                totals["external_bits"] += instances * (w_bits + a_bits)
            totals["external_bits"] += outer_multiplicity * output_external
            if schedule.spill_accumulator_after_k_tile:
                spills = sum(nk for _, nk in k_parts) - 1
                if spills:
                    acc_words = ceil(acc_bits / org.word_bits)
                    spill_bits = outer_multiplicity * 2 * spills * acc_words * org.word_bits
                    totals["spill_bits"] += spill_bits
                    totals["external_bits"] += spill_bits
                    loads = [q + (b < r) for b in range(banks)
                             for q, r in [divmod(acc_words, banks)]]
                    totals["local_read_cycles"] += outer_multiplicity * spills * max(ceil(x / org.read_ports) for x in loads)
                    totals["local_write_cycles"] += outer_multiplicity * spills * max(ceil(x / org.write_ports) for x in loads)
    if mapped and schedule.cache_mapped_scales:
        # Three globally cached tensor scales: W, A, and O. The local-port
        # metadata reads above remain, while repeated external fetches vanish.
        totals["external_bits"] -= totals["scale_bits"]
        totals["external_bits"] += 3 * ceil(64 / org.word_bits) * org.word_bits
    clock_hz = org.clock_mhz * 1e6
    external_cycles = ceil(totals["external_bits"] * clock_hz /
                           (schedule.external_bandwidth_GBps * 8e9))
    roofs = {"compute": totals["compute_cycles"], "local_read": totals["local_read_cycles"],
             "local_write": totals["local_write_cycles"], "external": external_cycles}
    limiting = max(roofs, key=roofs.get)
    cycles = roofs[limiting]
    return {"schema_version": SCHEMA_VERSION, "evidence_level": "analytical_schedule_upper_bound",
            "model": layer.model, "layer": layer.name, "kind": layer.kind,
            "format": format_name, "format_bits": manifest(format_name)["bits"],
            "output_channels": layer.output_channels, "positions": layer.positions,
            "reduction": layer.reduction, "model_c_products": total_mac,
            "tile_m": schedule.channels_per_tile, "tile_p": schedule.positions_per_tile,
            "tile_k": schedule.reduction_per_tile, "word_bits": org.word_bits,
            "banks": banks, "read_ports": org.read_ports, "write_ports": org.write_ports,
            "clock_mhz": org.clock_mhz, "compute_mac_per_cycle": schedule.compute_mac_per_cycle,
            "external_bandwidth_budget_GBps": schedule.external_bandwidth_GBps,
            "bank_capacity_bits": None if org.bank_depth_words is None else banks * org.bank_depth_words * org.word_bits,
            "peak_local_bank_bits": max_bank_bits,
            "peak_single_bank_bits": peak_single_bank_bits,
            "full_input_packed_bits": full_input["encoded_allocated_bits"],
            "full_output_packed_bits": full_output["encoded_allocated_bits"],
            "full_activation_pair_packed_bits": full_activation_pair_bits,
            "full_activation_pair_fits_local_bank": (full_activation_pair_bits <= banks * org.bank_depth_words * org.word_bits)
                                                     if org.bank_depth_words is not None else None,
            "peak_input_bank_bits": peak_input_bank_bits,
            "peak_weight_bank_bits": peak_weight_bank_bits,
            "peak_output_and_bias_bank_bits": peak_output_bank_bits,
            "peak_accumulator_register_bits": max_acc_bits,
            "fits_local_bank": all_fit if org.bank_depth_words is not None else None,
            "double_buffer_inputs": schedule.double_buffer_inputs,
            "cache_mapped_scales": schedule.cache_mapped_scales,
            "spill_accumulator_after_k_tile": schedule.spill_accumulator_after_k_tile,
            **totals, "external_cycles": external_cycles,
            "external_GB": totals["external_bits"] / 8e9,
            "external_bits_per_mac": totals["external_bits"] / total_mac,
            "cycle_roofs": roofs, "limiter": limiting if all_fit else "capacity",
            "delivered_external_GBps": (totals["external_bits"] / 8 * clock_hz / cycles / 1e9)
                                        if all_fit else None,
            "analytical_mac_per_s": total_mac * clock_hz / cycles if all_fit else None}
