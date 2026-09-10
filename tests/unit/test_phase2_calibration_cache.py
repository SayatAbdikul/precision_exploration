"""Resumed observations must equal a fresh pass over the frozen image order."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from public.experiments.registry.identity import canonical_json_bytes
from public.quantization.calibration.observer import Observer
from tools.run.phase2_calibration_cache import append_record, image_record, observe_classifier, read_record


def observe(observer, index):
    observer.record("features", torch.arange(37, dtype=torch.float32) * (index + 1))
    observer.record("small", torch.tensor([-0.0, float(index)], dtype=torch.float32))
    observer.finish_image()


def test_reordered_cache_assembly_matches_fresh_full_observations():
    fresh, resumed = Observer(), Observer()
    records = {}
    for index in (2, 0, 1):
        single = Observer()
        observe(single, index)
        records[index] = image_record("identity", {"index": index}, single)
    for index in range(3):
        observe(fresh, index)
        append_record(resumed, read_record(records[index], "identity", {"index": index}))
    assert resumed.summary() == fresh.summary()
    for name, values in fresh.arrays().items():
        assert resumed.arrays()[name].tobytes() == values.tobytes()


def test_cache_rejects_modified_bytes_and_provenance():
    single = Observer()
    observe(single, 0)
    row = {"sha256": "image"}
    record = image_record("identity", row, single)
    with pytest.raises(ValueError, match="provenance"):
        read_record(record, "changed source", row)
    with pytest.raises(ValueError, match="provenance"):
        read_record(record, "identity", {"sha256": "different image"})
    record["nodes"]["small"]["observed_elements"] += 1
    with pytest.raises(ValueError, match="record hash"):
        read_record(record, "identity", row)


def test_matching_digest_cannot_hide_invalid_samples():
    single = Observer()
    observe(single, 0)
    record = image_record("identity", {}, single)
    record["nodes"]["small"]["values_f32le"] = np.array([np.nan, 0], dtype="<f4").tobytes().hex()
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(
        {key: value for key, value in record.items() if key != "record_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="samples/counts"):
        read_record(record, "identity", {})


def test_resume_rejects_changed_node_coverage():
    combined, single = Observer(), Observer()
    observe(single, 0)
    nodes = read_record(image_record("id", {}, single), "id", {})
    append_record(combined, nodes)
    with pytest.raises(ValueError, match="node coverage"):
        append_record(combined, {"small": nodes["small"]})


@pytest.fixture
def classifier(tmp_path, monkeypatch):
    from PIL import Image
    from torchvision import models
    from tools.run import phase2_calibration_cache as cache
    monkeypatch.setattr(cache, "source_identity", lambda: "engine source")
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1)).eval()
    monkeypatch.setattr(models, "resnet18", lambda **kwargs: copy.deepcopy(model))
    torch.save(model.state_dict(), tmp_path / "checkpoint.pth")
    manifest = tmp_path / "public/workloads/models/manifests/resnet18.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"checkpoint_path": "checkpoint.pth"}))
    payload = tmp_path / "images"
    payload.mkdir()
    rows = []
    for index in range(2):
        path = payload / f"{index}.png"
        Image.new("RGB", (8, 8), color=(index * 80, 10, 30)).save(path)
        rows.append({"relative_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return tmp_path, {"model": "resnet18", "calibration_list": {"count": 2}}, rows, payload


def test_partial_observation_resumes_without_recomputing_verified_images(classifier, monkeypatch):
    root, context, rows, payload = classifier
    absent = payload / rows[1]["relative_path"]
    saved_bytes = absent.read_bytes()
    absent.unlink()
    assert observe_classifier(root, context, rows, payload, available_only=True) is None
    progress = json.loads((root / "artifacts/calibration_progress/resnet18.json").read_text())
    assert (progress["observed_images"], progress["missing_images"]) == (1, 1)
    assert not (root / "artifacts/calibration").exists()
    absent.write_bytes(saved_bytes)
    resumed, arrays = observe_classifier(root, context, rows, payload)
    progress = json.loads((root / "artifacts/calibration_progress/resnet18.json").read_text())
    assert (progress["reused_images"], progress["new_images"]) == (1, 1)
    from tools.run import phase2_calibration_cache as cache
    def no_inference(*args, **kwargs):
        raise AssertionError("completed cache must not execute model again")
    monkeypatch.setattr(cache, "classifier_interpreter", no_inference)
    again, reused_arrays = observe_classifier(root, context, rows, payload)
    assert again == resumed
    for name in arrays:
        assert arrays[name].tobytes() == reused_arrays[name].tobytes()
    absent.write_bytes(b"corrupted payload")
    with pytest.raises(ValueError, match="frozen bytes"):
        observe_classifier(root, context, rows, payload)


def test_partial_cache_never_substitutes_for_missing_training_payload(classifier):
    root, context, rows, payload = classifier
    observe_classifier(root, context, rows, payload)
    (payload / rows[0]["relative_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        observe_classifier(root, context, rows, payload)
