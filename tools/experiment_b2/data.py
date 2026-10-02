"""Read-only access to the v1 calibration observations and a preprocessed-input cache.

The FP32 calibration observations do not depend on the format or on the B2
recipe, so the v1 cache is reused instead of being recomputed.  It is verified
(seal, per-batch array hashes, list identity) and never written.

The input cache stores the exact FP32 tensors that v1's ``image_batch``
produces (same PIL decode and torchvision transform, every image re-hashed
when the cache is built), so evaluation jobs spend their GPU-lock time on
inference instead of on JPEG decoding.
"""
from __future__ import annotations

import os

import numpy as np

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, seal, unseal
from tools.experiment_b import common as v1
from .common import BASE

V1_CALIBRATION = ROOT / "artifacts/experiment_b/calibration"


def v1_calibration(model):
    """Load the v1 observations of ``model``: ``(arrays, maxima, identity record)``."""
    record, rows, _ = dataset("imagenet_calibration_2k")
    folders = [p for p in (V1_CALIBRATION / model).iterdir() if p.is_dir()]
    matching = []
    for folder in folders:
        provenance = unseal(folder / "provenance.json")
        if (digest(provenance) == folder.name and provenance["source_sha256"] == v1.source_identity()
                and provenance["protocol"] == v1.PROTOCOL and provenance["context"] == v1.frozen_inputs(model)):
            matching.append((folder, provenance))
    if len(matching) != 1:
        raise ValueError(f"expected exactly one current v1 calibration cache for {model}, found {len(matching)}")
    folder, provenance = matching[0]
    summary = unseal(folder / "summary.json")
    if summary["identity"] != folder.name or summary["images"] != len(rows) or summary["list_sha256"] != record["sha256"]:
        raise ValueError("v1 calibration summary does not match the frozen calibration list")
    combined, maxima = {}, {}
    batch = v1.PROTOCOL["calibration_batch_size"]
    for start in range(0, len(rows), batch):
        meta = unseal(folder / f"{start:05d}.json")
        npz = folder / f"{start:05d}.npz"
        if (meta["calibration_sha256"] != folder.name or meta["samples"] != rows[start:start + batch]
                or file_hash(npz) != meta["arrays_sha256"]):
            raise ValueError("v1 calibration checkpoint mismatch")
        with np.load(npz, allow_pickle=False) as saved:
            for key in saved.files:
                combined.setdefault(key, []).append(saved[key].ravel())
        for key, value in meta["maxima"].items():
            maxima[key] = max(maxima.get(key, 0), value)
    arrays = {key: np.concatenate(chunks) for key, chunks in combined.items()}
    for key, values in arrays.items():
        node = summary["nodes"][key]
        if node["sample_count"] != values.size or node["maxabs"] != maxima[key]:
            raise ValueError("v1 calibration arrays do not reproduce the sealed summary")
    identity = {"identity": folder.name, "summary_sha256": digest(summary), "graph": provenance["graph"],
                "runtime": provenance["runtime"], "reused_from": "artifacts/experiment_b/calibration (read-only)"}
    return arrays, maxima, identity


def transform_identity(transform):
    import PIL
    import torch
    import torchvision
    return {"transform": repr(transform), "pillow": PIL.__version__, "torch": torch.__version__,
            "torchvision": torchvision.__version__}


def cached_inputs(list_name, limit, transform, *, build=True):
    """FP32 input tensors ``[limit, 3, H, W]`` of the first ``limit`` sha256-sorted rows of a frozen list."""
    from tools.experiment_b.classifier import image_batch
    record, rows, payload = dataset(list_name)
    rows = rows[:limit]
    key = {"list": list_name, "list_sha256": record["sha256"], "rows_sha256": digest(rows),
           "count": len(rows), **transform_identity(transform)}
    folder = BASE / "input_cache" / digest(key)
    data, meta_path = folder / "inputs.npy", folder / "meta.json"
    if meta_path.exists():
        meta = unseal(meta_path)
        if meta["key"] != key or file_hash(data) != meta["array_sha256"]:
            raise ValueError("input cache integrity failure")
        return np.load(data, mmap_mode="r"), rows
    if not build:
        raise FileNotFoundError(f"input cache missing for {list_name}[:{limit}]")
    folder.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(rows), 32):
        chunks.append(image_batch(rows[start:start + 32], payload, transform, "cpu").numpy())
    array = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    if array.shape[0] != len(rows) or not np.isfinite(array).all():
        raise ValueError("invalid preprocessed inputs")
    temporary = data.with_name(data.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.save(stream, array)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(data)
    seal(meta_path, {"key": key, "array_sha256": file_hash(data), "shape": list(array.shape),
                     "built_by": "tools.experiment_b.classifier.image_batch (every image re-hashed)"})
    return np.load(data, mmap_mode="r"), rows
