"""Closure evidence must fail when bytes, layer checks or source identities drift."""
import copy
import hashlib
import json

import pytest

from public.experiments.registry.identity import canonical_json_bytes, experiment_sha256
from public.quantization.graph.executable import graph_sha256
from tools.analysis.phase2_finalize import calibration_coverage, detector_evidence, payload_status
from tools.run.phase2_engine import graph_witness


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document))


def dataset(root):
    payload = b"frozen sample"
    digest = hashlib.sha256(payload).hexdigest()
    row = {"relative_path": "image.jpeg", "sha256": digest}
    manifest = root / "samples.tsv"
    manifest.write_text("relative_path\tsha256\nimage.jpeg\t" + digest + "\n")
    record = {"path": "samples.tsv", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "count": 1, "logical_payload_root": "images"}
    image = root / "data/raw/images/image.jpeg"
    image.parent.mkdir(parents=True)
    image.write_bytes(payload)
    return row, record, image


def test_partial_payload_verification_distinguishes_missing_and_corrupt_bytes(tmp_path):
    _, record, image = dataset(tmp_path)
    assert payload_status(tmp_path, record)["status"] == "verified"
    image.write_bytes(b"corrupt")
    status = payload_status(tmp_path, record)
    assert status["verified_present"] == 0
    assert status["invalid"] == ["data/raw/images/image.jpeg"]
    image.unlink()
    status = payload_status(tmp_path, record)
    assert status["missing"] == 1 and not status["invalid"]


def test_detector_calibration_alone_cannot_close_classifier_calibration_gate():
    artifacts = [{"model": "yolov8n", "format": name} for name in ("fp6_e3m2", "int8")]
    coverage = calibration_coverage(artifacts)
    assert coverage["status"] == "incomplete"
    assert len(coverage["missing"]) == 6
    assert {item["model"] for item in coverage["missing"]} == {
        "resnet18", "mobilenet_v2", "mobilenet_v3_large"}
    assert calibration_coverage(artifacts + coverage["missing"])["status"] == "completed"


@pytest.fixture
def workload(tmp_path):
    row, record, _ = dataset(tmp_path)
    write_json(tmp_path / "data/manifests/index.json", {"records": {"screen": record}})
    graph, _ = graph_witness()
    graph_identity = graph_sha256(graph)
    config = {"runtime": {"source_sha256": "current"},
              "evaluation": {"name": "screen", "sha256": record["sha256"], "selection": "all"}}
    identity = experiment_sha256(config)
    work = tmp_path / "artifacts/run"
    write_json(work / "configuration.json", config)
    write_json(work / "graph.json", graph)
    write_json(work / "summary.json", {"job_sha256": identity, "graph_sha256": graph_identity,
               "metrics": {"image_count": [1, "count"], "bit_exact": [1, "boolean"], "map50_95": [0, "fraction"]}})
    layers = {node["name"]: {"sha256": "same"} for node in graph["nodes"]}
    document = {"job_sha256": identity, "graph_sha256": graph_identity, "sample": row,
                "backends": {"cpp": {"layers": layers}, "cuda": {"layers": copy.deepcopy(layers)}}}
    image_record = work / f"{row['sha256']}.json"
    def save(document):
        write_json(image_record, {**document, "record_sha256": hashlib.sha256(canonical_json_bytes(document)).hexdigest()})
    save(document)
    report = tmp_path / "report.json"
    write_json(report, {"artifacts": "artifacts/run", "job_sha256": identity,
                       "source_sha256": "current", "status": "COMPLETED"})
    return tmp_path, report, document, image_record, save


def test_detector_completion_requires_current_source_and_preserves_zero_quality(workload):
    root, report, *_ = workload
    result = detector_evidence(root, report, "current")
    assert result["status"] == "completed" and result["images"] == 1
    assert result["metrics"]["map50_95"][0] == 0
    assert detector_evidence(root, report, "changed")["status"] == "stale_or_incomplete"


def test_detector_completion_rejects_corrupted_record(workload):
    root, report, _, image_record, _ = workload
    document = json.loads(image_record.read_text())
    document["backends"]["cuda"]["layers"]["conv"]["sha256"] = "changed"
    write_json(image_record, document)
    with pytest.raises(ValueError, match="record hash mismatch"):
        detector_evidence(root, report, "current")


def test_detector_completion_rechecks_backend_equality_after_valid_record_rehash(workload):
    root, report, document, _, save = workload
    document["backends"]["cuda"]["layers"]["conv"]["sha256"] = "changed"
    save(document)
    with pytest.raises(ValueError, match="backend layer evidence mismatch"):
        detector_evidence(root, report, "current")


def test_matching_backends_do_not_hide_omitted_graph_layers(workload):
    root, report, document, _, save = workload
    for backend in document["backends"].values():
        del backend["layers"]["pool"]
    save(document)
    with pytest.raises(ValueError, match="layer evidence is incomplete"):
        detector_evidence(root, report, "current")
