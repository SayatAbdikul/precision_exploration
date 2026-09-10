"""Synthetic engine jobs in the existing lease-fenced experiment registry.

This schema cannot be mistaken for a workload quality result. Real workload
jobs still require the Phase 1 frozen model/data reference validation.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from public.experiments.registry.identity import canonical_json_bytes
from public.inference.tensor import Tensor
from public.quantization.graph.executable import execute, validate_graph

SCHEMA = "phase2-conformance-2.0.0"


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = []
    for folder in ("public/inference", "public/quantization", "public/cuda", "public/formats/oracle"):
        paths.extend(p for p in (root / folder).rglob("*") if p.suffix in {".py", ".h", ".cpp", ".cu"})
    record = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def validate_job(configuration):
    value = copy.deepcopy(configuration)
    if set(value) != {"schema_version", "purpose", "graph", "inputs", "runtime"}:
        raise ValueError("invalid conformance job fields")
    if value["schema_version"] != SCHEMA or value["purpose"] != "engine_conformance":
        raise ValueError("conformance job cannot claim workload quality")
    runtime = value["runtime"]
    if set(runtime) != {"backend", "semantic_version", "source_sha256"} or runtime["semantic_version"] != "2.0.0":
        raise ValueError("unsupported conformance runtime")
    if runtime["backend"] not in {"reference", "cpp", "cuda"} or runtime["source_sha256"] != source_identity():
        raise ValueError("runtime backend/source identity mismatch")
    validate_graph(value["graph"])
    if value["graph"]["provenance"] != {"kind": "synthetic_conformance"}:
        raise ValueError("this job schema requires a synthetic conformance graph")
    if set(value["inputs"]) != set(value["graph"]["inputs"]):
        raise ValueError("conformance graph inputs mismatch")
    for name, document in value["inputs"].items():
        tensor = Tensor.from_document(document)
        from public.inference.tensor import SharedEncoding, parse_encoding
        expected = parse_encoding(value["graph"]["inputs"][name])
        compatible = (tensor.encoding.format == expected.format and tensor.encoding.axis == expected.axis
                      and tensor.encoding.block_size == expected.block_size) if isinstance(expected,SharedEncoding) else tensor.encoding == expected
        if not compatible:
            raise ValueError("conformance input domain mismatch")
    canonical_json_bytes(value)
    return value


def run_job(configuration):
    value = validate_job(configuration)
    inputs = {name: Tensor.from_document(doc) for name, doc in value["inputs"].items()}
    expected = execute(value["graph"], inputs)
    actual = execute(value["graph"], inputs, backend=value["runtime"]["backend"])
    for name, layer in expected["layers"].items():
        if actual["layers"][name] != layer:
            raise ValueError(f"first mismatching layer: {name}")
    if actual["outputs"] != expected["outputs"]:
        raise ValueError("graph output mismatch")
    return {"matching_layers": (len(actual["layers"]), "count"), "bit_exact": (1, "boolean")}
