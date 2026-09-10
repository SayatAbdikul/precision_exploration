"""Write the source-frozen Phase 2 verification record."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from public.inference.conformance_job import source_identity
from public.experiments.registry.identity import canonical_json_bytes, experiment_sha256
from public.quantization.graph.executable import graph_sha256
from public.workloads.datasets.identity import load_tsv, verify_payload_record
from public.quantization.calibration.artifact import validate as validate_calibration

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root.find("testsuite")
    if suite is None:
        suite = root
    return {"tests": int(suite.attrib.get("tests", 0)), "failures": int(suite.attrib.get("failures", 0)),
            "errors": int(suite.attrib.get("errors", 0)), "skipped": int(suite.attrib.get("skipped", 0)),
            "path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}


def payload_status(root: Path, record: dict) -> dict:
    """Count verified bytes, not merely filenames left by a partial recovery."""
    manifest = root / record["path"]
    rows = load_tsv(manifest)
    if file_hash(manifest) != record["sha256"] or len(rows) != record["count"]:
        raise ValueError("frozen dataset list identity/count mismatch")
    payload = root / "data/raw" / record["logical_payload_root"]
    verified, missing, invalid = 0, 0, []
    for row in rows:
        path = payload / row.get("relative_path", row.get("file_name", ""))
        if not path.resolve().is_relative_to(payload.resolve()):
            raise ValueError("image path escapes frozen payload root")
        if not path.is_file():
            missing += 1
        elif file_hash(path) == row["sha256"]:
            verified += 1
        else:
            invalid.append(str(path.relative_to(root)))
    return {"expected": record["count"], "verified_present": verified, "missing": missing,
            "invalid": invalid, "status": "verified" if verified == record["count"] else "incomplete"}


def detector_evidence(root: Path, report_path: Path, current_source: str) -> dict:
    """Recheck the saved execution records before claiming network equality."""
    report = json.loads(report_path.read_text())
    work = (root / report["artifacts"]).resolve()
    if not work.is_relative_to(root.resolve()):
        raise ValueError("workload artifact path escapes repository")
    configuration = json.loads((work / "configuration.json").read_text())
    identity = experiment_sha256(configuration)
    if identity != report["job_sha256"]:
        raise ValueError("workload configuration identity mismatch")
    execution_source = configuration["runtime"]["source_sha256"]
    if execution_source != report["source_sha256"]:
        raise ValueError("workload source identity mismatch")
    graph = json.loads((work / "graph.json").read_text())
    graph_identity = graph_sha256(graph)
    summary = json.loads((work / "summary.json").read_text())
    if summary["job_sha256"] != identity or summary["graph_sha256"] != graph_identity:
        raise ValueError("workload summary identity mismatch")
    evaluation = configuration["evaluation"]
    index = json.loads((root / "data/manifests/index.json").read_text())["records"]
    record = index[evaluation["name"]]
    manifest = root / record["path"]
    if file_hash(manifest) != evaluation["sha256"] or evaluation["sha256"] != record["sha256"]:
        raise ValueError("workload evaluation list identity mismatch")
    rows = sorted(load_tsv(manifest), key=lambda row: row["sha256"])
    if len(rows) != record["count"]:
        raise ValueError("workload evaluation list count mismatch")
    if evaluation["selection"] == "first_eight_by_sha256_v1":
        rows = rows[:8]
    elif evaluation["selection"] != "all":
        raise ValueError("unsupported workload selection")
    layers = set()
    for row in rows:
        document = json.loads((work / f"{row['sha256']}.json").read_text())
        digest = document.pop("record_sha256")
        if hashlib.sha256(canonical_json_bytes(document)).hexdigest() != digest:
            raise ValueError("workload image record hash mismatch")
        if (document["job_sha256"] != identity or document["graph_sha256"] != graph_identity
                or document["sample"] != row):
            raise ValueError("workload image record identity mismatch")
        backends = document["backends"]
        if not {"cpp", "cuda"}.issubset(backends) or backends["cpp"]["layers"] != backends["cuda"]["layers"]:
            raise ValueError("workload backend layer evidence mismatch")
        layers.add(len(backends["cpp"]["layers"]))
    if len(layers) != 1 or not rows:
        raise ValueError("inconsistent workload layer counts")
    metrics = summary["metrics"]
    if metrics["image_count"][0] != len(rows) or metrics["bit_exact"][0] != 1:
        raise ValueError("workload summary contradicts image records")
    return {"status": "completed" if report["status"] == "COMPLETED" and execution_source == current_source else "stale_or_incomplete",
            "source_sha256": execution_source, "layers": layers.pop(), "images": len(rows),
            "backend_equality": True, "metrics": metrics, "report": str(report_path.relative_to(root)),
            "report_sha256": file_hash(report_path), "summary_sha256": file_hash(work / "summary.json")}


def fixed_image_evidence(root: Path, model: str, current_source: str) -> dict:
    path = root / "results/summaries" / f"phase2-{model}-fixed-images.json"
    if not path.exists():
        return {"status": "missing"}
    document = json.loads(path.read_text())
    from tools.run.phase2_fixed_images import select_samples, verify_images
    selection, payload_root = select_samples(root)
    verify_images(selection["samples"], payload_root)
    if document["model"] != model or document["selection"] != selection:
        raise ValueError("fixed-image report selection mismatch")
    if document["model_manifest_sha256"] != file_hash(root / f"public/workloads/models/manifests/{model}.json"):
        raise ValueError("fixed-image model identity mismatch")
    artifact = (root / document["graph_artifact"]).resolve()
    if not artifact.is_relative_to(root.resolve()):
        raise ValueError("fixed-image graph escapes repository")
    if graph_sha256(json.loads(artifact.read_text())) != document["graph_sha256"]:
        raise ValueError("fixed-image graph identity mismatch")
    records = document["records"]
    if [row["sample"] for row in records] != selection["samples"]:
        raise ValueError("fixed-image report is incomplete")
    layer_counts = set()
    for row in records:
        cpp, cuda = row["backends"]["cpp"], row["backends"]["cuda"]
        if cpp["layers"] != cuda["layers"] or cpp["outputs"] != cuda["outputs"]:
            raise ValueError("fixed-image backend evidence mismatch")
        layer_counts.add(len(cpp["layers"]))
    if len(layer_counts) != 1:
        raise ValueError("fixed-image layer count mismatch")
    return {"status": "completed" if document["source_sha256"] == current_source else "stale",
            "source_sha256": document["source_sha256"], "images": len(records),
            "layers": layer_counts.pop(), "backend_equality": True,
            "path": str(path.relative_to(root)), "sha256": file_hash(path)}


def baseline_evidence(root: Path) -> dict:
    baseline = json.loads((root / "results/summaries/phase1-fp32-baselines.json").read_text())
    records = {}
    for run in baseline["runs"]:
        model = run["model"]
        name = "yolov8n_coco2017_val5k_coco_predictions.json" if model == "yolov8n" else f"{model}_imagenet1k_screen.jsonl"
        path = root / "artifacts/per_image_predictions" / name
        actual = file_hash(path) if path.exists() else None
        records[model] = {"expected_sha256": run["prediction_sha256"], "actual_sha256": actual,
                          "status": "verified" if actual == run["prediction_sha256"] else "missing_or_changed"}
    return records


def main() -> None:
    current_source = source_identity()
    index = json.loads((ROOT / "data/manifests/index.json").read_text())["records"]
    coco = {}
    for name in ("coco_calibration_2k", "coco_screen_1k", "coco_evaluation_5k"):
        coco[name] = {"count": verify_payload_record(ROOT, index[name]), "list_sha256": index[name]["sha256"]}
    calibrations = []
    for path in sorted((ROOT / "artifacts/calibration").glob("yolov8n-*.json")):
        if "observations" in path.name:
            continue
        document = validate_calibration(json.loads(path.read_text()), ROOT)
        calibrations.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash(path),
                             "format": document["format"], "images": document["observations"]["image_count"],
                             "nodes": len(document["encodings"]), "sampling": document["observations"]["sampling"]})
    image_counts = {name: payload_status(ROOT, index[name]) for name in
                    ("imagenet_calibration_2k", "imagenet_screen_1k", "imagenet_evaluation_10k")}
    smoke = {}
    for name in ("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n"):
        path = ROOT / "results/summaries" / f"phase2-{name}-synthetic-smoke.json"
        if path.exists():
            document = json.loads(path.read_text())
            smoke[name] = {"source_sha256": document.get("source_sha256"),
                           "layers": len(document.get("records", [{}])[0].get("layers", {})),
                           "backend_equality_checked": document.get("backend_equality_checked"),
                           "status": "current" if document.get("source_sha256") == current_source else "stale"}
    detector_workload = sorted((ROOT / "results/summaries").glob("phase2-yolov8n-workload-*.json"))
    detector_runs = [detector_evidence(ROOT, path, current_source) for path in detector_workload
                     if json.loads(path.read_text()).get("status") == "COMPLETED"]
    detector = next((run for run in reversed(detector_runs) if run["status"] == "completed"),
                    detector_runs[-1] if detector_runs else {"status": "missing"})
    final_reports = {}
    evidence_sources = {}
    for path in sorted((ROOT / "results/summaries").glob("phase2-*-final.json")):
        final_reports[str(path.relative_to(ROOT))] = file_hash(path)
        document = json.loads(path.read_text())
        evidence_sources[str(path.relative_to(ROOT))] = {"source_sha256": document.get("source_sha256"),
            "status": "current" if document.get("source_sha256") == current_source else "stale"}
    classifiers = {name: fixed_image_evidence(ROOT, name, current_source)
                   for name in ("resnet18", "mobilenet_v2", "mobilenet_v3_large")}
    decisions_path = ROOT / "results/summaries/phase2-decision-evidence.json"
    decisions = json.loads(decisions_path.read_text())
    diagnosis_path = ROOT / "results/summaries/phase2-detector-diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text())
    for reference in diagnosis["evidence"]:
        path = (ROOT / reference["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or file_hash(path) != reference["sha256"]:
            raise ValueError("detector diagnosis evidence changed")
    if diagnosis["source_sha256"] != current_source:
        raise ValueError("detector diagnosis source is stale")
    matrix_path = ROOT / "results/summaries/phase2-family-matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix_formats = {row["format"] for row in matrix["records"]}
    accepted_formats = {row["name"] for row in json.loads((ROOT / "public/formats/manifests/accepted/index.json").read_text())["manifests"]}
    if matrix_formats != accepted_formats or any(set(row["backends"]) != {"cpp", "cuda"} for row in matrix["records"]):
        raise ValueError("family matrix coverage is incomplete")
    profile_path = ROOT / "results/summaries/phase2-cuda-profile.json"
    profile = json.loads(profile_path.read_text())
    baselines = baseline_evidence(ROOT)
    report = {
        "schema_version": "2.0.0",
        "status": "in_progress_external_gates_open",
        "source_sha256": current_source,
        "tests": {"phase2_cpu": test_counts(ROOT / "artifacts/conformance/phase2/pytest-phase2-final.xml"),
                  "full_cpu": test_counts(ROOT / "artifacts/conformance/phase2/pytest-cpu-final.xml"),
                  "cuda": test_counts(ROOT / "artifacts/conformance/phase2/pytest-cuda-final.xml"),
                  "cpu_known_external_failures": ["ImageNet payload verification", "frozen Phase 1 prediction artifacts"]},
        "coco": coco,
        "calibration": {"artifacts": calibrations, "policy": "mse_numpy_f64_100coarse_50fine_v1"},
        "imagenet": {"payloads": image_counts, "status": "verified" if all(item["status"] == "verified" for item in image_counts.values()) else "incomplete"},
        "synthetic_networks": smoke,
        "classifier_native_resolution": classifiers,
        "frozen_baseline_predictions": baselines,
        "detector_native_resolution": detector,
        "detector_quality_diagnosis": {"status": diagnosis["status"], "path": str(diagnosis_path.relative_to(ROOT)),
                                       "sha256": file_hash(diagnosis_path), "conclusion": diagnosis["conclusion"]},
        "all_format_matrix": {"path": "results/summaries/phase2-family-matrix.json",
                               "sha256": file_hash(matrix_path), "formats": len(matrix_formats),
                               "status": "current" if matrix["source_sha256"] == current_source else "stale",
                               "backends": ["cpp", "cuda"]},
        "wide_policy": {"path": "results/summaries/phase2-wide-policy-candidates.json",
                         "sha256": file_hash(ROOT / "results/summaries/phase2-wide-policy-candidates.json"),
                         "status": "candidate policies; per-graph Experiment A acceptance required before Phase 3 sweeps"},
        "profiling": {"status": profile["counter_status"], "missing_metrics": profile["missing_metrics"],
                      "source_sha256": profile["source_sha256"], "source_current": profile["source_sha256"] == current_source,
                      "path": str(profile_path.relative_to(ROOT)), "sha256": file_hash(profile_path)},
        "decisions": {"D2": decisions["D2"]["status"], "D3": decisions["D3"]["status"],
                      "path": str(decisions_path.relative_to(ROOT)), "sha256": file_hash(decisions_path)},
        "final_report_hashes": final_reports,
        "evidence_sources": evidence_sources,
    }
    remaining = []
    for name, value in image_counts.items():
        if value["status"] != "verified":
            remaining.append({"gate": name, "reason": f"{value['missing']} missing and {len(value['invalid'])} invalid frozen payloads"})
    for name, value in classifiers.items():
        if value["status"] != "completed":
            remaining.append({"gate": f"{name}_native_resolution", "reason": value["status"]})
    for name, value in baselines.items():
        if value["status"] != "verified":
            remaining.append({"gate": f"{name}_frozen_predictions", "reason": value["status"]})
    if decisions["D2"]["status"] != "accepted":
        remaining.append({"gate": "D2", "reason": "achieved occupancy and kernel DRAM counter evidence unavailable"})
    if decisions["D3"]["status"] != "accepted":
        remaining.append({"gate": "D3", "reason": decisions["D3"]["status"]})
    for name in ("phase2_cpu", "full_cpu", "cuda"):
        counts = report["tests"][name]
        if counts["errors"] or counts["failures"] or not counts["tests"]:
            remaining.append({"gate": name, "reason": "test failures/errors or no executed tests"})
    if detector["status"] != "completed":
        remaining.append({"gate": "detector_native_resolution", "reason": detector["status"]})
    for name, value in {**evidence_sources, **smoke, "family_matrix": report["all_format_matrix"]}.items():
        if value["status"] != "current":
            remaining.append({"gate": name, "reason": "evidence uses a different engine source"})
    report["remaining_gates"] = remaining
    report["status"] = "in_progress_gates_open" if remaining else "complete"
    output = ROOT / "results/summaries/phase2-final-verification.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": file_hash(output), "source_sha256": report["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
