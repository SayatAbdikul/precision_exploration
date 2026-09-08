#!/usr/bin/env python3
"""Generate exhaustive feasible Phase 1 oracle tables and a tracked index."""

from __future__ import annotations

import json
from pathlib import Path

from public.formats.oracle import NumberFormat, load_manifest
from public.formats.oracle.number_format import ORACLE_VERSION
from public.formats.oracle.truth_tables import conversion_rows, decode_rows, encode_boundary_rows, operation_rows, write_jsonl_table


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    manifest_root = ROOT / "public/formats/manifests/accepted"
    accepted = json.loads((manifest_root / "index.json").read_text(encoding="utf-8"))
    artifact_root = ROOT / "artifacts/datatype_truth_tables"
    records = []
    formats = []
    for item in accepted["manifests"]:
        number_format = NumberFormat(load_manifest(ROOT / item["path"]))
        formats.append(number_format)
        context = {"scale": "1", "block_size": 32} if "block" in number_format.manifest else {"scale": "1"}
        jobs = [
            ("decode", decode_rows(number_format), None),
            ("encode_boundaries", encode_boundary_rows(number_format), None),
            ("add", operation_rows(number_format, "add"), None),
            ("mul", operation_rows(number_format, "mul"), None),
        ]
        for kind, rows, destination in jobs:
            path = artifact_root / f"{number_format.name}.{kind}.jsonl"
            metadata = write_jsonl_table(path, rows, kind=kind, source=number_format,
                                         destination=destination, context=context)
            records.append({"format": number_format.name, "artifact_path": path.relative_to(ROOT).as_posix(), **metadata})
    for source in formats:
        for destination in formats:
            kind = "convert"
            path = artifact_root / f"{source.name}.to.{destination.name}.jsonl"
            metadata = write_jsonl_table(path, conversion_rows(source, destination), kind=kind,
                                         source=source, destination=destination,
                                         context={"source_scale": "1", "destination_scale": "1"})
            records.append({"format": source.name, "artifact_path": path.relative_to(ROOT).as_posix(), **metadata})
    index = {"schema_version": "1.0.0", "oracle": "public.formats.oracle.NumberFormat", "oracle_version": ORACLE_VERSION,
             "generator": "tools/setup/generate_truth_tables.py", "accepted_manifest_set_sha256": accepted["aggregate_sha256"],
             "coverage": "exhaustive decode, representable-value and adjacent-midpoint encoding boundaries, ADD, MUL, and every ordered accepted-format conversion; scale=1 scalar context",
             "records": records}
    output = ROOT / "public/formats/conformance/truth-table-index.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"formats": accepted["accepted_count"], "tables": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
