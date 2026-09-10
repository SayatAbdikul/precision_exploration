"""Resume classifier calibration from verified per-image FP32 observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from public.experiments.registry.identity import canonical_json_bytes
from public.inference.conformance_job import source_identity
from public.quantization.calibration.observer import Observer, SAMPLING, classifier_interpreter
from public.workloads.datasets.identity import sha256

SCHEMA = "phase2-calibration-image-1.0.0"


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(canonical_json_bytes(document))
    temporary.replace(path)


def image_record(identity, row, observer):
    if observer.images != 1:
        raise ValueError("cache entry must contain exactly one completed image")
    arrays = observer.arrays()
    document = {"schema_version": SCHEMA, "cache_sha256": identity, "sample": row,
                "nodes": {name: {"values_f32le": values.astype("<f4").tobytes().hex(),
                                 "observed_elements": observer.counts[name]}
                          for name, values in arrays.items()}}
    return {**document, "record_sha256": hashlib.sha256(canonical_json_bytes(document)).hexdigest()}


def read_record(document, identity, row):
    if set(document) != {"schema_version", "cache_sha256", "sample", "nodes", "record_sha256"}:
        raise ValueError("invalid calibration cache schema")
    payload = {key: value for key, value in document.items() if key != "record_sha256"}
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != document["record_sha256"]:
        raise ValueError("calibration cache record hash mismatch")
    if document["schema_version"] != SCHEMA or document["cache_sha256"] != identity or document["sample"] != row:
        raise ValueError("calibration cache provenance mismatch")
    nodes = {}
    for name, node in document["nodes"].items():
        if set(node) != {"values_f32le", "observed_elements"}:
            raise ValueError("invalid calibration cache node")
        values = np.frombuffer(bytes.fromhex(node["values_f32le"]), dtype="<f4").copy()
        count = node["observed_elements"]
        if type(count) is not int or count <= 0 or len(values) != min(16, count) or not np.isfinite(values).all():
            raise ValueError("invalid calibration cache samples/counts")
        nodes[name] = (values, count)
    if not nodes:
        raise ValueError("empty calibration cache record")
    return nodes


def append_record(combined, nodes):
    if combined.expected_nodes is not None and set(nodes) != combined.expected_nodes:
        raise ValueError("calibration node coverage changed between images")
    combined.expected_nodes = set(nodes)
    for name, (values, count) in nodes.items():
        combined.samples.setdefault(name, []).append(values)
        combined.counts[name] = combined.counts.get(name, 0) + count
    combined.images += 1


class ImageObserver:
    """Keep the interpreter fixed while starting a fresh observer per image."""
    current = None

    def record(self, name, tensor):
        self.current.record(name, tensor)


def observe_classifier(root, context, rows, payload_root, *, available_only=False):
    import PIL
    from PIL import Image
    import torch
    import torchvision
    from torchvision import models
    from public.workloads.models.torchvision_eval import MODEL_SPECS

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model_name = context["model"]
    source = source_identity()
    provenance = {"context": context, "source_sha256": source, "sampling": SAMPLING,
                  "collector_sha256": sha256(Path(__file__)),
                  "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                              "numpy": np.__version__, "pillow": PIL.__version__,
                              "system": platform.system(), "machine": platform.machine(),
                              "torch_build": torch.__config__.show(), "threads": 4,
                              "mkldnn": torch.backends.mkldnn.enabled}}
    identity = hashlib.sha256(canonical_json_bytes(provenance)).hexdigest()
    cache = root / "artifacts/calibration_observations" / model_name / identity
    cache.mkdir(parents=True, exist_ok=True)
    # The long-running data continuation can reach calibration while this
    # precomputation is finishing. Serialize a model's cache writers.
    import fcntl
    with (cache / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(cache / "provenance.json", provenance)
        combined = Observer()
        proxy = ImageObserver()
        run = transform = None
        reused = computed = missing = 0
        with torch.inference_mode():
            for row in sorted(rows, key=lambda item: item["sha256"]):
                image_path = payload_root / row["relative_path"]
                if not image_path.resolve().is_relative_to(payload_root.resolve()):
                    raise ValueError("calibration payload path escapes frozen root")
                if not image_path.exists() and available_only:
                    missing += 1
                    continue
                if sha256(image_path) != row["sha256"]:
                    raise ValueError("calibration image differs from frozen bytes")
                record_path = cache / f"{row['sha256']}.json"
                if record_path.exists():
                    document = json.loads(record_path.read_text())
                    reused += 1
                else:
                    if run is None:
                        manifest = json.loads((root / f"public/workloads/models/manifests/{model_name}.json").read_text())
                        _, weights_class, weights_name = MODEL_SPECS[model_name]
                        transform = getattr(getattr(models, weights_class), weights_name).transforms()
                        model = getattr(models, model_name)(weights=None)
                        model.load_state_dict(torch.load(root / manifest["checkpoint_path"], map_location="cpu", weights_only=True))
                        run = classifier_interpreter(model.eval(), proxy).run
                    proxy.current = Observer()
                    with Image.open(image_path) as image:
                        inputs = transform(image.convert("RGB")).unsqueeze(0)
                    run(inputs)
                    proxy.current.finish_image()
                    document = image_record(identity, row, proxy.current)
                    write_json(record_path, document)
                    computed += 1
                append_record(combined, read_record(document, identity, row))
                if combined.images % 100 == 0:
                    print(f"{model_name}: observed {combined.images}/{len(rows)} frozen images ({reused} cached, {computed} new)", flush=True)
        if source_identity() != source or sha256(Path(__file__)) != provenance["collector_sha256"]:
            raise ValueError("calibration source changed while observing images")
        progress = {"schema_version": "phase2-calibration-progress-1.0.0", "model": model_name,
                    "cache_sha256": identity, "source_sha256": source,
                    "observed_images": combined.images, "expected_images": len(rows),
                    "reused_images": reused, "new_images": computed, "missing_images": missing,
                    "status": "observations_complete" if combined.images == len(rows) else "waiting_for_frozen_images",
                    "cache": str(cache.relative_to(root))}
        write_json(root / "artifacts/calibration_progress" / f"{model_name}.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
        if available_only:
            return None
        if combined.images != context["calibration_list"]["count"]:
            raise ValueError("calibration requires the complete frozen training population")
        return combined.summary(), combined.arrays()
