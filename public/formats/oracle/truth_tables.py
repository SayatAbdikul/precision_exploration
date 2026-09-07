"""Truth-table generation for manifest-driven primitive conformance."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .manifest import manifest_sha256
from .number_format import NumberFormat, OracleError


TABLE_SCHEMA_VERSION = "1.0.0"


def decode_rows(number_format: NumberFormat) -> Iterable[dict[str, Any]]:
    for code in range(number_format.code_count):
        value = number_format.decode(code)
        yield {"code": code, "value": str(value)}


def encode_boundary_rows(number_format: NumberFormat) -> Iterable[dict[str, Any]]:
    """Enumerate every finite value and every adjacent rounding boundary."""

    values = sorted({number_format.decode(code) for code in range(number_format.code_count)
                     if number_format.decode(code).is_finite()})
    probes: list[tuple[str, Decimal]] = [("representable", value) for value in values]
    probes.extend(("adjacent_midpoint", (left + right) / 2) for left, right in zip(values, values[1:]))
    if values:
        span = max(Decimal(1), values[-1] - values[0])
        probes.extend((("below_minimum", values[0] - span), ("above_maximum", values[-1] + span)))
    probes.extend((("positive_zero", Decimal("0")), ("negative_zero", Decimal("-0")),
                   ("positive_infinity", Decimal("Infinity")), ("negative_infinity", Decimal("-Infinity")),
                   ("nan", Decimal("NaN"))))
    for index, (probe, value) in enumerate(probes):
        row: dict[str, Any] = {"ordinal": index, "probe": probe, "value": str(value)}
        try:
            row["result"] = number_format.encode(value)
        except OracleError as exc:
            row["error"] = str(exc)
        yield row


def operation_rows(number_format: NumberFormat, operation: str) -> Iterable[dict[str, Any]]:
    if operation not in {"add", "mul"}:
        raise ValueError(f"unsupported binary operation: {operation}")
    function = getattr(number_format, operation)
    for left in range(number_format.code_count):
        for right in range(number_format.code_count):
            yield {"a": left, "b": right, "result": function(left, right)}


def conversion_rows(source: NumberFormat, destination: NumberFormat) -> Iterable[dict[str, Any]]:
    for code in range(source.code_count):
        try:
            yield {"source": code, "result": source.convert(code, destination)}
        except OracleError as exc:
            yield {"source": code, "error": str(exc)}


def write_jsonl_table(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    kind: str,
    source: NumberFormat,
    destination: NumberFormat | None = None,
    completeness: str = "exhaustive",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("wb") as stream:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
            count += 1
        stream.flush()
    temporary.replace(output)
    metadata = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "kind": kind,
        "source_manifest": source.name,
        "source_manifest_sha256": manifest_sha256(source.manifest),
        "destination_manifest": destination.name if destination else None,
        "destination_manifest_sha256": manifest_sha256(destination.manifest) if destination else None,
        "completeness": completeness,
        "context": context or {"scale": "1"},
        "row_count": count,
        "sha256": digest.hexdigest(),
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
