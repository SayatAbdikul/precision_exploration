"""Derive Phase 2 runtime and workload decisions from retained measurements.

Budget projections are arithmetic scenarios, never measured full-suite timings.
Missing counters and source mismatches remain explicit evidence limitations.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from public.experiments.registry.identity import canonical_json_bytes
from public.inference.conformance_job import source_identity
from tools.analysis.phase2_counter_evidence import profile_errors as validate_profile

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = "phase2-yolov8n-workload-02870391c93c.json"


def strategy_candidates(report):
    groups = defaultdict(list)
    for row in report["result"]["records"]:
        groups[(row["format"], row["shape_role"])].append(row)
    result = []
    for (fmt, shape), rows in groups.items():
        if len({row["output_sha256"] for row in rows}) != 1:
            raise ValueError("strategy measurements have different numerical outputs")
        if any(not math.isfinite(row["median_seconds"]) or row["median_seconds"] <= 0 for row in rows):
            raise ValueError("strategy latency must be positive and finite")
        fastest = min(rows, key=lambda row: row["median_seconds"])
        alternatives = [row for row in rows if row["strategy"] != "predecoded"]
        prepared = next((row for row in rows if row["strategy"] == "predecoded"), None)
        breakeven = None
        if prepared is not None and alternatives:
            saving = min(row["median_seconds"] for row in alternatives) - prepared["median_seconds"]
            setup = prepared["predecode_setup_seconds"]
            if saving > 0 and setup is not None:
                breakeven = max(1, math.ceil(setup / saving))
        result.append({"backend": report["backend"], "format": fmt, "shape_role": shape,
                       "measured_fastest_strategy": fastest["strategy"],
                       "median_seconds": fastest["median_seconds"],
                       "mac_per_second": fastest["mac_per_second"],
                       "sample_count": len(fastest["samples_seconds"]),
                       "strategies_compared": [row["strategy"] for row in rows],
                       "same_numerical_output": True,
                       "predecode_setup_amortization_calls": breakeven})
    return result


def workload_cost(records, candidate_count):
    """Project measured backend execution time without scaling image resolution."""
    if not records:
        raise ValueError("workload cost needs native-resolution image measurements")
    result = []
    for backend in ("cpp", "cuda"):
        seconds = [row["backends"][backend]["seconds"] for row in records]
        if any(not math.isfinite(value) or value <= 0 for value in seconds):
            raise ValueError("workload latency must be positive and finite")
        median = statistics.median(seconds)
        result.append({"backend": backend, "measured_images": len(records),
                       "median_seconds_per_image": median,
                       "observed_seconds_range": [min(seconds), max(seconds)],
                       "projected_hours_one_format_1k": median * 1000 / 3600,
                       "projected_hours_candidate_count_1k": median * 1000 * candidate_count / 3600})
    return result


def build_report(root=ROOT):
    artifacts = {}

    def read(relative):
        path = root / relative
        payload = path.read_bytes()
        artifacts[str(relative)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    def verify(relative, expected):
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"evidence hash mismatch: {relative}")
        artifacts[str(relative)] = actual

    current_source = source_identity()
    benchmarks = [read(f"results/summaries/phase2-benchmark-{backend}-final.json")
                  for backend in ("cpp", "cuda")]
    profile = read("results/summaries/phase2-cuda-profile.json")
    profile_errors = validate_profile(root, profile, current_source)
    for relative, expected in profile["artifacts"].items():
        try:
            verify(relative, expected)
        except (ValueError, FileNotFoundError) as exc:
            profile_errors.append(str(exc))
    hardware = read("results/summaries/phase2-hardware-pilot.json")
    timing = read("results/summaries/phase2-sta-power-pilot.json")
    for row in hardware["runs"]:
        verify(row["run_manifest"], row["run_manifest_sha256"])
        verify(str(Path(row["run_manifest"]).parent / "primitive.sv"), row["rtl_sha256"])
        if f"PASS {row['exhaustive_pairs']} pairs" not in row["simulation"]:
            raise ValueError("hardware primitive lacks exhaustive simulation evidence")
    for row in timing["runs"]:
        if row["status"] != "measured" or row["returncode"] != 0:
            raise ValueError("hardware timing/power run did not complete")
        for relative, expected in row["reports"].items():
            verify(relative, expected)

    workload = read(f"results/summaries/{WORKLOAD}")
    directory = Path(workload["artifacts"])
    config = read(directory / "configuration.json")
    summary = read(directory / "summary.json")
    records = []
    for path in sorted((root / directory).glob("*.json")):
        if len(path.stem) != 64:
            continue
        row = read(path.relative_to(root))
        digest = row.pop("record_sha256")
        if hashlib.sha256(canonical_json_bytes(row)).hexdigest() != digest:
            raise ValueError(f"workload record content mismatch: {path.name}")
        if row["job_sha256"] != workload["job_sha256"] or row["graph_sha256"] != summary["graph_sha256"]:
            raise ValueError("workload timing belongs to another configuration")
        if row["backends"]["cpp"]["layers"] != row["backends"]["cuda"]["layers"]:
            raise ValueError("workload backends differ")
        records.append(row)
    if len(records) != summary["metrics"]["image_count"][0]:
        raise ValueError("workload image count does not match retained records")
    candidate_count = len(read("public/formats/manifests/accepted/index.json")["manifests"])
    wide = read("results/summaries/phase2-wide-policy-candidates.json")
    benchmarks_current = all(report["source_sha256"] == current_source for report in benchmarks)
    profiling_complete = not profile_errors and not profile["missing_metrics"]
    candidates = [row for report in benchmarks for row in strategy_candidates(report)]
    return {
        "schema_version": "2.0.0", "audit_source_sha256": current_source,
        "D2": {"status": "accepted" if benchmarks_current and profiling_complete else "open",
               "scope": "measured FP6 E3M2 and INT8 GEMM shapes; other formats retain correctness-validated implementations pending strategy measurements",
               "runtime_recommendation": "use each measured fastest strategy, accounting for operand preparation and reuse",
               "strategy_candidates": candidates,
               "benchmark_sources_current": benchmarks_current,
               "profiling_source_sha256": profile["source_sha256"],
               "profiling_source_current": profile["source_sha256"] == current_source,
               "profiling_artifact_errors": profile_errors,
               "missing_metrics": profile["missing_metrics"],
               "counter_records": profile.get("counter_records", []),
               "limits": ["three latency samples per strategy, with predecoded inputs reused",
                          "amortization includes preparation of both operands; weights-only reuse differs",
                          "depthwise has no measured alternative strategy",
                          "the other accepted families have conformance pilots, not strategy comparisons",
                          "Nsight Systems registers and transfer traces do not measure achieved occupancy or kernel DRAM traffic"]},
        "D3": {"status": "accepted", "decision": "retain EfficientNet as optional; main suite remains the four core workloads",
               "core_workloads": ["resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n"],
               "workload_measurement_source_sha256": workload["source_sha256"],
               "workload_measurement_source_current": workload["source_sha256"] == current_source,
               "measurement_format": config["formats"], "candidate_count_scenario": candidate_count,
               "cost_scenarios": workload_cost(records, candidate_count),
               "project_wall_clock_budget_hours": None,
               "budget_policy": "allocate no additional mandatory workload while core screening already takes substantial measured compute and no incremental budget is committed",
               "limits": ["projections use eight native-resolution YOLO images at one FP6/FP32 configuration",
                          "the 25-candidate scenario assumes identical throughput; actual families differ",
                          "calibration, graph setup, preprocessing, comparison-backend execution, and scoring add time",
                          "classifier 32x32 synthetic timings are not extrapolated to native resolution",
                          "no EfficientNet runtime, available project budget, or full-suite completion date is asserted"],
               "reopen_when": "core screening throughput and an explicit incremental compute budget support staging and timing EfficientNet"},
        "hardware_pilots": {"status": "complete_for_H2_pilot_scope", "primitive_runs": len(hardware["runs"]),
                            "timing_power_runs": len(timing["runs"]),
                            "formats": sorted({row["format"] for row in hardware["runs"]}),
                            "internal_pipeline_stages": sorted({row["internal_stages"] for row in hardware["runs"]}),
                            "periods_ns": sorted({row["period_ns"] for row in timing["runs"]}),
                            "scope": timing["scope"]},
        "accumulator_policy": {"status": "explicit_execution_supported; Experiment_A_family_wide_acceptance_open",
                               "phase2_requirement": "independent explicit accumulator manifests, correct stored bias and Model C; no unresolved policy aliases execute",
                               "phase3_requirement": "N3.2 requires sufficiently wide family-appropriate accumulators before each Experiment A configuration executes",
                               "remaining": wide["remaining"]},
        "artifacts": artifacts,
    }


def main():
    output = ROOT / "results/summaries/phase2-decision-evidence.json"
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps(build_report(), indent=2) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
