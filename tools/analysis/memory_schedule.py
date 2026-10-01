"""Generate common-layout convolution schedule evidence from retained shapes.

Run: python3 -m tools.analysis.memory_schedule --observation-root /path/to/precision_exploration
The observation root is read only. No graph execution or inference occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from public.analysis.memory_model import Organization
from public.analysis.memory_schedule import ConvLayer, Schedule, evaluate_layer


ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = (
    ("resnet18", "conv1", "standard"),
    ("mobilenet_v2", "features_1_conv_0_0", "depthwise"),
    ("mobilenet_v2", "features_1_conv_1", "pointwise"),
    ("mobilenet_v3_large", "features_1_block_1_0", "pointwise"),
    ("yolov8n", "model_0_conv", "standard"),
)


def retained_layers(observation_root: Path):
    summary_path = observation_root / "results/summaries/phase3-hardware-priors.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    layers, sources = [], []
    for model, name, kind in SELECTIONS:
        record = next(r for r in summary["prepared_graph_storage"]
                      if r["configuration"] == f"{model}/int4")
        def read_verified(ref):
            path = observation_root / ref["path"]
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != ref["sha256"]:
                raise ValueError(f"source identity changed: {path}")
            return json.loads(data)
        graph = read_verified(record["graph"])
        shapes = read_verified(record["shape_evidence"])["shapes"]
        node = next(n for n in graph["nodes"] if n["name"] == name)
        weight = graph["constants"][node["inputs"][1]]
        layer = ConvLayer(model, name, kind, tuple(shapes[node["inputs"][0]]),
                          tuple(shapes[name]), tuple(weight["shape"]),
                          node["attrs"].get("bias") is not None)
        if kind == "depthwise" and node["attrs"].get("groups") != layer.input_shape[1]:
            raise ValueError("expected depthwise groups")
        if kind == "standard" and node["attrs"].get("groups", 1) != 1:
            raise ValueError("expected ungrouped standard convolution")
        layers.append(layer)
        sources.append({"model": model, "layer": name,
                        "graph_sha256": record["graph"]["sha256"],
                        "shape_sha256": record["shape_evidence"]["sha256"],
                        "input_shape": layer.input_shape,
                        "output_shape": layer.output_shape,
                        "weight_shape": layer.weight_shape})
    return layers, {"summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                    "layers": sources}


def scenarios():
    base = Schedule()
    return {
        "base_64b_8x8": base,
        "large_tile_64b_16x16": replace(base, channels_per_tile=16, positions_per_tile=16),
        "word_32b_8x8": replace(base, organization=Organization(word_bits=32, banks=4,
                                  read_ports=1, write_ports=1, clock_mhz=500,
                                  bank_depth_words=512)),
        "low_reuse_64b_1x1": replace(base, channels_per_tile=1, positions_per_tile=1),
        "spill_64b_8x8": replace(base, spill_accumulator_after_k_tile=True),
    }


def generate(observation_root: Path):
    layers, sources = retained_layers(observation_root)
    index = json.loads((ROOT / "public/formats/manifests/accepted/index.json").read_text())
    rows = []
    for scenario, setup in scenarios().items():
        for layer in layers:
            for fmt in (item["name"] for item in index["manifests"]):
                result = evaluate_layer(layer, fmt, setup)
                row = {"scenario": scenario, **result}
                row.pop("cycle_roofs")
                rows.append(row)
    path = ROOT / "results/tables/memory-scheduled-layers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    source_script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_model_sha = hashlib.sha256((ROOT / "public/analysis/memory_schedule.py").read_bytes()).hexdigest()
    summary = {
        "schema_version": "memory-schedule-results-1.0.0",
        "evidence_level": "analytical_schedule_upper_bound",
        "accepted_manifest_aggregate_sha256": index["aggregate_sha256"],
        "sources": sources, "generator_sha256": source_script_sha,
        "model_sha256": source_model_sha,
        "rows": len(rows), "formats": len(index["manifests"]),
        "layers": len(layers), "scenarios": list(scenarios()),
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capacity_infeasible_rows": sum(r["fits_local_bank"] is False for r in rows),
        "physical_sram_area_energy": None,
        "quality_equivalence": "not_established_for_hypothetical_common_tile_layout",
    }
    output = ROOT / "results/summaries/memory-schedule-1.0.0.json"
    output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-root", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = generate(args.observation_root)
    print(f"Wrote {len(rows)} rows; {summary['capacity_infeasible_rows']} exceed assumed local-bank capacity")


if __name__ == "__main__":
    main()
