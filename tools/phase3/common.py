"""Atomic, content-addressed Phase 3 evidence and frozen campaign definitions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from public.experiments.registry.identity import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n")


def digest(document):
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            payload = canonical_json_bytes(document) if isinstance(document, dict) else (
                json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def reference(path, root=ROOT):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(root.resolve())), "sha256": file_hash(path)}


def checked(reference, root=ROOT):
    if set(reference) != {"path", "sha256"}:
        raise ValueError("invalid artifact reference")
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != reference["sha256"]:
        raise ValueError("artifact path/hash mismatch")
    return path


def campaign(root=ROOT):
    plan = read(root / "public/experiments/configs/experiment_a/phase3-screen-v1.json")
    for item in plan["inputs"].values():
        checked(item, root)
    return plan
