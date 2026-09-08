#!/usr/bin/env python3
"""Normalize and ingest the completed Phase 1 numerical and hardware evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import shutil
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from public.analysis.hardware import parse_opensta_report, parse_yosys_stat
from public.experiments.registry import ArtifactRecord, ExperimentRegistry, experiment_sha256
from public.experiments.registry.identity import repository_state


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "results/databases/phase1.sqlite"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(relative: str, value: Any) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(relative: str, semantic_type: str) -> ArtifactRecord:
    path = ROOT / relative
    return ArtifactRecord(digest(path), relative, path.stat().st_size, semantic_type)


def hardware_evidence() -> dict[str, Any]:
    run_root = ROOT / "artifacts/ppa/ics55-pilot"
    repeat_root = ROOT / "artifacts/ppa/ics55-pilot-repeat"
    run_manifest = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
    yosys_report = json.loads((run_root / "yosys-stat.json").read_text(encoding="utf-8"))
    top = run_manifest["config"]["top"]
    core_names = [name for name in yosys_report["modules"] if "smoke_arithmetic" in name and top not in name]
    if len(core_names) != 1:
        raise RuntimeError(f"expected one arithmetic core in Yosys report, found {core_names}")
    base = {
        "rtl_sha256": run_manifest["rtl_sha256"],
        "harness_version": run_manifest["config"]["harness_version"],
        "pdk": "ics55",
        "pdk_version": "v1.10.102",
        "pdk_repository_commit": "68d89edb47847671e18f9e65d66c0cd883995e05",
        "library": "ics55_LLSC_H7CR",
        "liberty_sha256": run_manifest["external_resource_hashes"]["liberty_sha256"],
        "corner": run_manifest["config"]["corner"],
        "vt": "rvt",
        "harness": {
            "launch": "registered inputs",
            "capture": "registered result",
            "clock_uncertainty_ns": 0.10,
            "input_delay_ns": 0.20,
            "output_delay_ns": 0.20,
            "output_load_pf": 0.01,
            "clock_network": "ideal",
        },
        "latency_cycles": 2,
        "initiation_interval_cycles": 1,
        "pipeline_stages": run_manifest["config"]["pipeline_stages"],
        "accounting_boundary": "all-in registered wrapper and combinational DUT; core-only metrics also reported",
    }
    synthesis_identity = {
        **base,
        "tool": "Yosys",
        "tool_version": yosys_report["creator"].removeprefix("Yosys "),
        "target_clock": run_manifest["config"]["target_clock"],
        "core_module": core_names[0],
    }
    synthesis = parse_yosys_stat(run_root / "yosys-stat.json", top=top, identity=synthesis_identity)

    timing = []
    for period in (10, 5, 2):
        timing_identity = {
            **base,
            "tool": "OpenSTA",
            "tool_version": "3.1.0 2996e37a3a",
            "target_clock": f"{period}ns",
            "time_unit": "ns",
            "netlist_sha256": run_manifest["outputs"]["netlist.v"]["sha256"],
        }
        timing.append(parse_opensta_report(run_root / f"opensta-{period}ns.txt", identity=timing_identity))

    stable_outputs = ("netlist.v", "yosys-stat.json")
    repeated_hashes = {
        name: {
            "primary_sha256": digest(run_root / name),
            "repeat_sha256": digest(repeat_root / name),
            "identical": digest(run_root / name) == digest(repeat_root / name),
        }
        for name in stable_outputs
    }
    if not all(value["identical"] for value in repeated_hashes.values()):
        raise RuntimeError("ICsprout55 synthesis repeat changed a stable output")

    output = {
        "schema_version": "1.0.0",
        "scope": "generic Phase 1 smoke flow; no candidate format or private MANT implementation",
        "availability": {
            "status": "public_preview_usable",
            "pdk_repository": "https://github.com/openecos-projects/icsprout55-pdk",
            "ecc_repository": "https://github.com/openecos-projects/ecc",
            "ecos_studio_repository": "https://github.com/openecos-projects/ecos-studio",
            "host_strategy": "native Yosys plus OpenSTA in Linux container",
            "ecc_packaged_installer": "Linux x86_64 only; not used on this macOS arm64 host",
            "sky130_fallback": "not_exercised_because_ics55_public_rvt_collateral_was_usable",
        },
        "pdk": {
            "version": "v1.10.102",
            "repository_commit": base["pdk_repository_commit"],
            "liberty_archive_sha256": "ef33eec4cd5f617d3dd0073122e556df06560dd65eebf805648640038dedc2b7",
            "typical_rvt_liberty_sha256": base["liberty_sha256"],
            "corner": base["corner"],
            "vt": "rvt",
        },
        "flow_stage_status": {
            "elaboration": "completed",
            "mapped_logic_synthesis": "completed",
            "post_synthesis_sta": "completed_without_interconnect_parasitics",
            "placement": "unavailable_in_validated_host_setup",
            "routing": "unavailable_in_validated_host_setup",
            "parasitic_extraction": "unavailable_in_public_preview_collateral",
            "power": "unavailable_in_public_preview_collateral",
        },
        "synthesis": synthesis,
        "timing_sweep": timing,
        "repeatability": {"run_count": 2, "stable_outputs": repeated_hashes, "runtime_logs": "retained but excluded from byte-stability claims"},
        "raw_evidence": {
            "run_manifest": "artifacts/ppa/ics55-pilot/run-manifest.json",
            "repeat_run_manifest": "artifacts/ppa/ics55-pilot-repeat/run-manifest.json",
            "opensta_script_sha256": digest(ROOT / "public/pdk_flow/common/opensta_pilot.tcl"),
        },
        "memory_evidence": {
            "strongest_level": 3,
            "level_1_characterized_sram": "unavailable",
            "level_2_validated_generated_memory": "unavailable",
            "level_3_synthesized_register_memory": "available through RVT standard cells",
            "level_4_analytical": "available but weaker and not used as physical SRAM evidence",
            "decision_D9": "open",
        },
    }
    dump("results/summaries/ics55-phase1-pilot.json", output)
    return output


def classifier_rows(relative: str) -> list[dict[str, Any]]:
    rows = []
    with (ROOT / relative).open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            rows.append({
                "sample_id": source["sample_id"],
                "ordinal": source["ordinal"],
                "ground_truth": source["ground_truth"],
                "fp32_prediction": source["fp32_prediction"],
                "fp32_correct": source["fp32_correct"],
                "summary": {"top5_predictions": source["top5_predictions"], "row_version": "1.0.0"},
            })
    return rows


def detector_rows() -> list[dict[str, Any]]:
    predictions = json.loads(
        (ROOT / "artifacts/per_image_predictions/yolov8n_coco2017_val5k_coco_predictions.json").read_text()
    )
    annotations = json.loads((ROOT / "data/raw/coco2017/annotations/instances_val2017.json").read_text())
    predicted: dict[int, list[dict[str, Any]]] = defaultdict(list)
    truth: dict[int, list[int]] = defaultdict(list)
    for item in predictions:
        predicted[int(item["image_id"])].append(item)
    for item in annotations["annotations"]:
        truth[int(item["image_id"])].append(int(item["category_id"]))
    rows = []
    with (ROOT / "data/manifests/evaluation/coco2017_val_5k.tsv").open(encoding="utf-8", newline="") as stream:
        for ordinal, item in enumerate(csv.DictReader(stream, delimiter="\t")):
            image_id = int(item["image_id"])
            detections = sorted(predicted.get(image_id, []), key=lambda value: -float(value["score"]))
            rows.append({
                "sample_id": str(image_id),
                "ordinal": ordinal,
                "ground_truth": {"category_ids": sorted(truth.get(image_id, [])), "annotation_count": len(truth.get(image_id, []))},
                "fp32_prediction": {"detection_count": len(detections), "top_detections": detections[:5]},
                "fp32_correct": None,
                "summary": {"correctness": "not_defined_per_image_for_coco_map", "row_version": "1.0.0"},
            })
    return rows


def register_baselines(registry: ExperimentRegistry) -> dict[str, str]:
    dataset_index = json.loads((ROOT / "data/manifests/index.json").read_text())
    specifications = [
        ("resnet18", "imagenet_screen_1k", "public/workloads/datasets/imagenet_resnet18_preprocessing.json", "results/summaries/resnet18_fp32_imagenet1k_screen.json", "artifacts/per_image_predictions/resnet18_imagenet1k_screen.jsonl"),
        ("mobilenet_v2", "imagenet_screen_1k", "public/workloads/datasets/imagenet_mobilenet_preprocessing.json", "results/summaries/mobilenet_v2_fp32_imagenet1k_screen.json", "artifacts/per_image_predictions/mobilenet_v2_imagenet1k_screen.jsonl"),
        ("mobilenet_v3_large", "imagenet_screen_1k", "public/workloads/datasets/imagenet_mobilenet_preprocessing.json", "results/summaries/mobilenet_v3_large_fp32_imagenet1k_screen.json", "artifacts/per_image_predictions/mobilenet_v3_large_imagenet1k_screen.jsonl"),
        ("yolov8n", "coco_evaluation_5k", "public/workloads/datasets/coco2017.json", "results/summaries/yolov8n_fp32_coco2017_val5k.json", "artifacts/per_image_predictions/yolov8n_coco2017_val5k_coco_predictions.json"),
    ]
    ids: dict[str, str] = {}
    for model, dataset_key, preprocessing_path, summary_path, prediction_path in specifications:
        model_path = f"public/workloads/models/manifests/{model}.json"
        model_manifest = json.loads((ROOT / model_path).read_text())
        dataset = dataset_index["records"][dataset_key]
        summary = json.loads((ROOT / summary_path).read_text())
        configuration = {
            "schema_version": "phase1-fp32-baseline-1.0.0",
            "model": {
                "name": model,
                "manifest_sha256": digest(ROOT / model_path),
                "checkpoint_sha256": model_manifest["checkpoint_sha256"],
                "deployment_graph_sha256": model_manifest["deployment_graph_sha256"],
            },
            "dataset": {"name": dataset_key, "list_sha256": dataset["sha256"], "count": dataset["count"]},
            "preprocessing_sha256": digest(ROOT / preprocessing_path),
            "evaluator": model_manifest["evaluator"],
            "precision": "fp32",
            "seed": 0,
        }
        experiment_id = experiment_sha256(configuration)
        ids[model] = experiment_id
        run_id, status = registry.submit(configuration)
        if status == "PENDING":
            claimed = registry.claim_next("phase1-local")
            if claimed is None or int(claimed["run_id"]) != run_id:
                raise RuntimeError(f"could not claim baseline run {run_id}")
            metrics = summary["metrics"]
            registry.add_metrics(
                run_id,
                {key: (float(value), "count" if key == "count" else ("percent" if "percent" in key else "ratio"))
                 for key, value in metrics.items()},
                lease_token=claimed["lease_token"],
            )
            rows = classifier_rows(prediction_path) if model != "yolov8n" else detector_rows()
            registry.store_per_image(experiment_id, rows)
            registry.add_run_metadata(run_id, {
                "repository_state": repository_state(ROOT),
                "summary_path": summary_path,
                "prediction_path": prediction_path,
                "per_sample_schema": "1.0.0",
            })
            registry.finish(run_id, "COMPLETED", lease_token=claimed["lease_token"])
        elif status != "COMPLETED":
            raise RuntimeError(f"baseline {model} has unexpected registry status {status}")

        inputs = [record(model_path, "model_manifest"), record(preprocessing_path, "preprocessing_manifest"),
                  record(dataset["path"], "dataset_sample_list")]
        prediction = record(prediction_path, "fp32_per_sample_predictions")
        result_summary = record(summary_path, "fp32_metric_summary")
        for item in inputs:
            registry.register_artifact(item, producer_experiment_id=None, metadata={})
        for item in (prediction, result_summary):
            registry.register_artifact(item, producer_experiment_id=experiment_id, metadata={})
            registry.attach_artifact(run_id, item.sha256, "prediction" if item is prediction else "summary")
            for dependency in inputs:
                registry.add_artifact_dependency(item.sha256, dependency.sha256, dependency.semantic_type)
    return ids


def main() -> None:
    hardware = hardware_evidence()
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(prefix="phase1-rebuild-", suffix=".sqlite", dir=DATABASE.parent)
    os.close(descriptor)
    staged = Path(staged_name)
    with ExperimentRegistry(staged) as registry:
        baseline_ids = register_baselines(registry)
        accepted_manifest = record("public/formats/manifests/accepted/fp8_e4m3fn.json", "datatype_manifest")
        truth_index = record("public/formats/conformance/truth-table-index.json", "truth_table_index")
        truth_table = record("artifacts/datatype_truth_tables/fp8_e4m3fn.add.jsonl", "datatype_truth_table")
        for item in (accepted_manifest, truth_index, truth_table):
            registry.register_artifact(item, producer_experiment_id=None, metadata={})
        registry.add_artifact_dependency(truth_table.sha256, accepted_manifest.sha256, "source_manifest")
        registry.add_artifact_dependency(truth_index.sha256, accepted_manifest.sha256, "accepted_manifest_set_member")
        for item in [hardware["synthesis"], *hardware["timing_sweep"]]:
            registry.register_hardware_run(
                item["hardware_run_id"],
                experiment_id=None,
                identity=item["identity"],
                metrics=item["metrics"],
                evidence_level=item["evidence_level"],
                parser_version=item["parser_version"],
            )
            report_path = (
                "artifacts/ppa/ics55-pilot/yosys-stat.json"
                if item is hardware["synthesis"]
                else f"artifacts/ppa/ics55-pilot/opensta-{item['identity']['target_clock']}.txt"
            )
            report = record(report_path, "hardware_tool_report")
            registry.register_artifact(report, producer_experiment_id=None, metadata={"hardware_run_id": item["hardware_run_id"]})
            registry.attach_hardware_artifact(item["hardware_run_id"], report.sha256, "source_report")
            # Exercise the production idempotency path, not only a synthetic test.
            registry.register_hardware_run(
                item["hardware_run_id"],
                experiment_id=None,
                identity=item["identity"],
                metrics=item["metrics"],
                evidence_level=item["evidence_level"],
                parser_version=item["parser_version"],
            )
        snapshot = registry.export_snapshot()
    if DATABASE.exists():
        archive = DATABASE.with_name("phase1-before-corrections.sqlite")
        if not archive.exists():
            shutil.copyfile(DATABASE, archive)
    staged.replace(DATABASE)
    dump("results/summaries/phase1-registry-export.json", {
        "schema_version": "1.0.0",
        "authoritative_database": "results/databases/phase1.sqlite",
        "baseline_experiment_ids": baseline_ids,
        "snapshot": snapshot,
    })
    print(json.dumps({"baselines": len(baseline_ids), "hardware_runs": 1 + len(hardware["timing_sweep"]),
                      "database": DATABASE.relative_to(ROOT).as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
