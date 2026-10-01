"""Generate all-format and retained-graph packed-memory evidence.

Run: python3 -m tools.analysis.memory_model --observation-root /path/to/precision_exploration
The observation root is read only. Outputs are written under this checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from public.analysis.memory_model import Organization, manifest, tensor_layout, throughput_roof


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "public/formats/manifests/accepted/index.json"


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _svg(rows, path):
    # A compact static publication figure; the CSV remains authoritative.
    w, h = 880, 65 + 25 * len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="12" y="22" font-family="sans-serif" font-size="15">Effective bits/value, 32×33 weight matrix, 64-bit words</text>',
             '<text x="12" y="40" font-family="sans-serif" font-size="11">Blue: encoded payload + mapped/intrinsic metadata + padding; scale mapping is analytical where required</text>']
    for i, row in enumerate(rows):
        y = 65 + i * 25
        length = min(560, row["effective_bits_per_value"] * 55)
        ideal = row["ideal_bits_per_value"] * 55
        parts.append(f'<text x="10" y="{y+12}" font-family="monospace" font-size="12">{row["format"]}</text>')
        parts.append(f'<rect x="230" y="{y}" width="{length:.1f}" height="15" fill="#2166ac"/>')
        parts.append(f'<path d="M {230+ideal:.1f} {y-2} v 19" stroke="#d6604d" stroke-width="2"/>')
        parts.append(f'<text x="{240+length:.1f}" y="{y+12}" font-family="monospace" font-size="11">{row["effective_bits_per_value"]:.2f}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def format_rows():
    index = json.loads(INDEX.read_text())
    rows, roofs, hypotheticals, mapped_per_row = [], [], [], []
    for item in index["manifests"]:
        name = item["name"]
        fmt = manifest(name)
        count = 1 if fmt["scaling"]["mode"] == "required_mapping" else None
        for word_bits in (32, 64):
            layout = tensor_layout([32, 33], name, Organization(word_bits=word_bits, banks=2,
                                   read_ports=1, write_ports=1, clock_mhz=500,
                                   bank_depth_words=16384 // (2 * word_bits)),
                                   packing_axis=1, scale_axis=1 if fmt["scaling"]["mode"] == "intrinsic_shared" else None,
                                   scale_count=count)
            row = {"format": name, "family": item["family"], "manifest_sha256": item["sha256"],
                   "width_bits": item["bits"], "word_bits": word_bits,
                   "scale_mode": layout["scale_mode"], "scale_count": layout["scale_count"],
                   "scale_bits": layout["scale_bits"], "scale_block_size": layout["block"]["block_size"] if layout["block"] else None,
                   "values": layout["values"], "ideal_bits_per_value": layout["ideal_bits_per_value"],
                   "effective_bits_per_value": layout["effective_encoded_bits_per_value"],
                   "payload_bits": layout["payload"]["raw_bits"],
                   "metadata_bits": layout["metadata"]["raw_bits"],
                   "padding_bits": layout["payload"]["padding_bits"] + layout["metadata"]["padding_bits"],
                   "allocated_bits": layout["encoded_allocated_bits"],
                   "capacity_bits": layout["capacity_bits"],
                   "usable_value_capacity_for_identical_tensors": layout["usable_value_capacity_for_identical_tensors"],
                   "capacity_utilization": layout["capacity_utilization"],
                   "metadata_fraction": layout["metadata_fraction"], "padding_fraction": layout["padding_fraction"],
                   "cross_word_value_handling_required": layout["cross_word_value_handling_required"],
                   "read_words": layout["read_words"], "read_cycles": layout["read_cycles"],
                   "read_traffic_bits": layout["read_traffic_bits"],
                   "read_traffic_bits_per_value": layout["read_traffic_bits_per_value"],
                   "delivered_payload_gbps": layout["delivered_payload_gbps"],
                   "same_value_count": 1056, "level": "B_packed_logical_analytical"}
            rows.append(row)
            if word_bits == 64:
                roof = throughput_roof(layout, layout, weight_reuse=4, activation_reuse=4,
                                       compute_mac_per_cycle=64, compute_clock_mhz=500, bandwidth_GBps=8)
                roofs.append({"format": name, "effective_bits_per_value": row["effective_bits_per_value"],
                              "read_traffic_bits_per_value": layout["read_traffic_bits_per_value"],
                              **roof})
        if fmt["scaling"]["mode"] == "intrinsic_shared":
            for block_size in (16, 64):
                alt = tensor_layout([32, 33], name, Organization(word_bits=64, banks=2),
                                    packing_axis=1, scale_axis=1, block_size=block_size)
                hypotheticals.append({"format": name, "block_size": block_size,
                                      "status": "hypothetical_not_accepted_configuration",
                                      "scale_count": alt["scale_count"],
                                      "effective_bits_per_value": alt["effective_encoded_bits_per_value"]})
        if fmt["scaling"]["mode"] == "required_mapping":
            per_row = tensor_layout([32, 33], name, Organization(word_bits=64, banks=2),
                                    packing_axis=1, scale_count=32)
            mapped_per_row.append({"format": name, "scale_count": 32,
                                   "status": "illustrative_per_output_channel_graph_policy_not_manifest_default",
                                   "mapped_scale_bits_assumption": 64,
                                   "effective_bits_per_value": per_row["effective_encoded_bits_per_value"]})
    if len({row["format"] for row in rows}) != 25:
        raise ValueError("accepted format coverage is not 25")
    return index, rows, roofs, hypotheticals, mapped_per_row


def graph_rows(observation_root):
    source = observation_root / "results/summaries/phase3-hardware-priors.json"
    if not source.exists():
        return [], [], {"status": "missing", "path": "results/summaries/phase3-hardware-priors.json"}
    raw = source.read_bytes()
    report = json.loads(raw)
    rows, layers = [], []
    graph_hashes_checked = 0
    shape_hashes_checked = set()
    for record in report["prepared_graph_storage"]:
        if "resources" not in record:
            continue
        model, name = record["configuration"].split("/", 1)
        graph_path = observation_root / record["graph"]["path"]
        if not graph_path.exists():
            raise ValueError(f"missing graph: {record['graph']['path']}")
        graph_bytes = graph_path.read_bytes()
        if hashlib.sha256(graph_bytes).hexdigest() != record["graph"]["sha256"]:
            raise ValueError(f"changed graph: {record['graph']['path']}")
        graph = json.loads(graph_bytes)
        graph_hashes_checked += 1
        shape_ref = record["shape_evidence"]
        if shape_ref["sha256"] not in shape_hashes_checked:
            shape_path = observation_root / shape_ref["path"]
            if not shape_path.exists() or hashlib.sha256(shape_path.read_bytes()).hexdigest() != shape_ref["sha256"]:
                raise ValueError(f"missing or changed shape evidence: {shape_ref['path']}")
            shape_hashes_checked.add(shape_ref["sha256"])
        constants = record["storage"]["constants"]
        resources = record["resources"]
        # The prepared encoding records the actual scale count and axis.
        # The last tensor dimension is the declared contiguous packing line.
        recorded_constants = {c["name"]: c for c in constants}
        if set(recorded_constants) != set(graph["constants"]):
            raise ValueError("prepared graph/constants summary disagreement")
        packed_weights = 0
        flat_weights = 0
        metadata_bits = 0
        weight_values = 0
        weight_scale_count = 0
        scale_axes = set()
        for constant_name, tensor in graph["constants"].items():
            c = recorded_constants[constant_name]
            enc = tensor["encoding"]
            fmt = manifest(enc["format"])
            mode = fmt["scaling"]["mode"]
            count = len(enc["scales"]) if mode != "none" else 0
            kwargs = {"packing_axis": len(tensor["shape"]) - 1}
            if mode == "intrinsic_shared":
                kwargs.update(scale_axis=enc["axis"], block_size=enc["block_size"], scale_count=count)
            elif mode == "required_mapping":
                kwargs.update(scale_count=count)
            else:
                # Some unscaled graph encodings retain a unity scale token;
                # it is not stored metadata under the accepted manifest.
                pass
            layout = tensor_layout(tensor["shape"], fmt["name"], Organization(word_bits=64), **kwargs)
            flat_kwargs = dict(kwargs)
            flat_kwargs["packing_axis"] = None
            flat_layout = tensor_layout(tensor["shape"], fmt["name"], Organization(word_bits=64), **flat_kwargs)
            if layout["ideal_payload_bits"] != c["nominal_payload_bits"] or layout["metadata"]["raw_bits"] != c["metadata_bits"]:
                raise ValueError("graph encoding disagrees with retained storage summary")
            packed_weights += layout["encoded_allocated_bits"]
            flat_weights += flat_layout["encoded_allocated_bits"]
            metadata_bits += layout["metadata"]["raw_bits"]
            weight_values += layout["values"]
            weight_scale_count += count
            if count:
                scale_axes.add("per_tensor" if enc["axis"] is None else str(enc["axis"]))
        bias_bits = record["storage"]["nominal_bias_bits"]
        candidate_layers = [r for r in resources["layers"] if r["arithmetic"].get("model_c_products")]
        representative = max(candidate_layers, key=lambda r: r["arithmetic"]["model_c_products"]) if candidate_layers else None
        rows.append({"model": model, "format": name, "graph_sha256": record["graph"]["sha256"],
                     "shape_sha256": record["shape_evidence"]["sha256"],
                     "weight_values": weight_values, "weight_metadata_bits": metadata_bits,
                     "weight_scale_count": weight_scale_count,
                     "recorded_scale_axes": ";".join(sorted(scale_axes)),
                     "packing_policy": "last_dimension_lines_64bit_one_bank_per_tensor",
                     "weight_packed_64b_bits": packed_weights,
                     "weight_flat_64b_bits": flat_weights,
                     "bias_raw_bits": bias_bits,
                     "bias_packed_64b_bits": (bias_bits + 63) // 64 * 64,
                     "peak_activation_byte_packed_bits": resources["peak_live_activation_bits"],
                     "peak_node": resources["peak_node"],
                     "max_output_accumulator_bank_bits": resources["maximum_output_accumulator_bank_bits"],
                     "one_sequential_accumulator_bits": resources["maximum_sequential_accumulator_bits"],
                     "model_c_products": resources["arithmetic_counts"].get("model_c_products", 0),
                     "representative_layer": representative["name"] if representative else None,
                     "level": "B_graph_estimate_from_retained_shapes_and_resources"})
        if representative:
            layers.append({"model": model, "format": name, "layer": representative["name"],
                           "op": representative["op"], "shape": "x".join(map(str, representative["shape"])),
                           "reduction_length": representative["reduction_length"],
                           "model_c_products": representative["arithmetic"]["model_c_products"],
                           "output_activation_bits": representative["activation"]["byte_aligned_bits"],
                           "output_metadata_bits": representative["activation"]["metadata_bits"],
                           "output_accumulator_bank_bits": representative["one_accumulator_per_output_bits"]})
    coverage = {"status": "available", "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_schema_version": report["schema_version"],
                "configurations": len(rows), "models": sorted({r["model"] for r in rows}),
                "verified_graph_files": graph_hashes_checked,
                "verified_shape_evidence_files": len(shape_hashes_checked),
                "formats_per_model": {m: sum(r["model"] == m for r in rows) for m in sorted({r["model"] for r in rows})},
                "source_path": "results/summaries/phase3-hardware-priors.json",
                "derivation": "retained Phase 3 graph and observed-shape resource summary; no inference rerun"}
    return rows, layers, coverage


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-root", type=Path, default=Path("/home/maveric/precision_exploration"))
    args = parser.parse_args(argv)
    index, rows, roofs, hypotheticals, mapped_per_row = format_rows()
    graphs, layers, coverage = graph_rows(args.observation_root)
    fields = list(rows[0])
    _write_csv(ROOT / "results/tables/memory-all-formats.csv", rows, fields)
    _write_csv(ROOT / "results/tables/memory-throughput-roofs.csv", roofs, list(roofs[0]))
    _write_csv(ROOT / "results/tables/memory-hypothetical-blocks.csv", hypotheticals, list(hypotheticals[0]))
    _write_csv(ROOT / "results/tables/memory-mapped-per-row.csv", mapped_per_row, list(mapped_per_row[0]))
    if graphs:
        _write_csv(ROOT / "results/tables/memory-graph-examples.csv", graphs, list(graphs[0]))
        _write_csv(ROOT / "results/tables/memory-layer-examples.csv", layers, list(layers[0]))
    _svg([r for r in rows if r["word_bits"] == 64], ROOT / "results/figures/memory-effective-bits.svg")
    output_names = ["memory-all-formats.csv", "memory-throughput-roofs.csv",
                    "memory-hypothetical-blocks.csv", "memory-mapped-per-row.csv"]
    if graphs:
        output_names += ["memory-graph-examples.csv", "memory-layer-examples.csv"]
    output_hashes = {name: hashlib.sha256((ROOT / "results/tables" / name).read_bytes()).hexdigest()
                     for name in output_names}
    _write_json(ROOT / "results/summaries/memory-model-1.0.0.json", {
        "schema_version": "memory-model-results-1.0.0", "accepted_manifest_aggregate_sha256": index["aggregate_sha256"],
        "model_source_sha256": hashlib.sha256((ROOT / "public/analysis/memory_model.py").read_bytes()).hexdigest(),
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "table_sha256": output_hashes,
        "review_corrections": ["per-line actual cross-word detection", "all-array padding fraction",
                               "explicit zero block-size rejection"],
        "format_count": len({r["format"] for r in rows}), "format_rows": len(rows),
        "graph_coverage": coverage, "graph_rows": len(graphs),
        "scenario": {"shape": [32, 33], "packing_axis": 1, "banks": 2, "read_ports": 1,
                     "write_ports": 1, "clock_mhz": 500, "word_bits": [32, 64],
                     "same_capacity_bits": 16384,
                     "required_mapping_scales": "one per-tensor mapped scale at assumed 64 bits; per-row graph policy shown separately",
                     "intrinsic_scales": "manifest 8-bit scales, axis 1, manifest block size",
                     "roof": "same 1056 values, 8 GB/s shared budget, 64 MAC/cycle at 500 MHz, W/A reuse 4 each"},
        "level_A": "nominal payload bits only", "level_B": "packed words/banks, metadata, padding and port-limited access cycles",
        "level_C": {"status": "unavailable_for_SRAM", "ICS55": "No characterized SRAM or validated generated macro in available public collateral; a synthesized register array would be identified separately and has not been measured here"},
        "limits": ["64-bit mapped scale storage is analytical and does not prove arbitrary rational scale hardware",
                   "ports are assumed available; port implementation/area/energy is uncharacterized",
                   "graph peak liveness is the existing byte-packed schedule estimate, not physical occupancy or host memory",
                   "graph weight table reports both flat 64-bit words and conservative last-axis-line alignment per tensor; neither is a realized physical memory",
                   "traffic assumes one sequential read/write per tensor; roofs assume explicitly declared reuse, not measured accelerator traffic",
                   "double buffering, accumulator banks and bias can be modeled by API but are not included in the illustrative format table",
                   "no physical SRAM area, read/write energy or latency has been awarded"]})
    print(f"Wrote {len(rows)} format rows, {len(graphs)} graph rows, {len(layers)} layer rows")


if __name__ == "__main__":
    main()
