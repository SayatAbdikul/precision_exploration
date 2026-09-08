"""Stable checkpoint, graph, and per-sample baseline evidence helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ModelIdentityError(ValueError):
    """Raised when a frozen model payload or graph no longer matches."""


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_identity(
    *,
    architecture: str,
    graph_version: str,
    nodes: Iterable[Mapping[str, Any]],
) -> str:
    value = {
        "architecture": architecture,
        "graph_version": graph_version,
        "nodes": list(nodes),
    }
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_prediction_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, int | str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    seen: set[str] = set()
    with temporary.open("wb") as stream:
        for ordinal, source in enumerate(rows):
            row = dict(source)
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate prediction sample: {sample_id}")
            seen.add(sample_id)
            row.setdefault("ordinal", ordinal)
            encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
            count += 1
    temporary.replace(output)
    return {"path": output.as_posix(), "count": count, "sha256": digest.hexdigest()}


def classification_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    count = 0
    top1 = 0
    top5 = 0
    for row in rows:
        count += 1
        target = row["ground_truth"]
        predictions = list(row["top5_predictions"])
        top1 += int(bool(predictions) and predictions[0] == target)
        top5 += int(target in predictions[:5])
    if count == 0:
        raise ValueError("cannot calculate metrics for zero samples")
    return {"count": count, "top1_percent": 100.0 * top1 / count, "top5_percent": 100.0 * top5 / count}


def verify_model_manifest(manifest_path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(repository_root).resolve()
    def payload_path(raw):
        if not isinstance(raw, str) or "\\" in raw or any(part in {"", ".", ".."} for part in raw.split("/")):
            raise ModelIdentityError("model payload path is not portable")
        path = root / raw
        if Path(raw).is_absolute() or not path.resolve().is_relative_to(root):
            raise ModelIdentityError("model payload path escapes repository")
        return path
    checkpoint = payload_path(manifest["checkpoint_path"])
    if not checkpoint.is_file():
        raise ModelIdentityError(f"checkpoint is missing: {manifest['checkpoint_path']}")
    if checkpoint.stat().st_size != manifest["checkpoint_size_bytes"]:
        raise ModelIdentityError(f"checkpoint size mismatch: {manifest['name']}")
    actual = checkpoint_sha256(checkpoint)
    if actual != manifest["checkpoint_sha256"]:
        raise ModelIdentityError(f"checkpoint hash mismatch: {manifest['name']}")
    graph_path = payload_path(manifest["deployment_graph_path"])
    matches = [graph_path]
    if not graph_path.is_file():
        raise ModelIdentityError(f"deployment graph is missing or ambiguous: {manifest['name']}")
    graph = json.loads(matches[0].read_text(encoding="utf-8"))
    actual_graph = graph_identity(
        architecture=graph["architecture"],
        graph_version=graph["graph_version"],
        nodes=graph["nodes"],
    )
    if actual_graph != manifest["deployment_graph_sha256"] or checkpoint_sha256(graph_path) != actual_graph:
        raise ModelIdentityError(f"deployment graph hash mismatch: {manifest['name']}")
    if manifest.get("deployment_graph_version") != "folded-aten-fixed-input-1.1.0":
        raise ModelIdentityError("unsupported deployment graph version")
    if graph["graph_version"] != manifest["deployment_graph_version"] or graph["architecture"] != manifest["name"]:
        raise ModelIdentityError("deployment graph metadata mismatch")
    if matches[0].read_bytes() != (json.dumps(graph, sort_keys=True, separators=(",", ":")) + "\n").encode():
        raise ModelIdentityError("deployment graph must use canonical serialized bytes")
    prep = payload_path(manifest["preprocessing_path"])
    if checkpoint_sha256(prep) != manifest["preprocessing_sha256"]:
        raise ModelIdentityError("preprocessing hash mismatch")
    preprocessing = json.loads(prep.read_text())
    for annotation, expected in preprocessing.get("annotation_sha256", {}).items():
        path = payload_path(f"data/raw/coco2017/annotations/{annotation}.json")
        if checkpoint_sha256(path) != expected:
            raise ModelIdentityError(f"annotation hash mismatch: {annotation}")
    return manifest
