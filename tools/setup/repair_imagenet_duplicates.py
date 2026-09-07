#!/usr/bin/env python3
"""Replace duplicate-content rows in a finalized ImageNet subset in place."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

import pyarrow
import pyarrow.dataset
from datasets import Image, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    return parser.parse_args()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest_path = root / "manifest.tsv"
    metadata_path = root / "selection.json"
    rows = manifest_rows(manifest_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_hash[row["sha256"]].append(index)
    replacements = [index for indices in by_hash.values() for index in indices[1:]]
    if not replacements:
        print("manifest already has unique content")
        return

    excluded = set(by_hash)
    for excluded_manifest in args.exclude_manifest:
        excluded.update(row["sha256"] for row in manifest_rows(excluded_manifest))

    scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=1,
            range_size_limit=int(metadata.get("range_size_mib", 4)) << 20,
        )
    )
    dataset = load_dataset(
        metadata["dataset"],
        split=metadata["split"],
        revision=metadata["revision"],
        streaming=True,
        token=True,
        batch_size=int(metadata.get("read_batch", 128)),
        fragment_scan_options=scan_options,
    )
    dataset = dataset.cast_column("image", Image(decode=False))
    dataset = dataset.shuffle(
        seed=int(metadata["seed"]), buffer_size=int(metadata["shuffle_buffer"])
    )

    pending: dict[int, list[int]] = defaultdict(list)
    for index in replacements:
        pending[int(rows[index]["label"])].append(index)
    repaired: list[dict[str, str | int]] = []

    for stream_position, sample in enumerate(dataset):
        label = int(sample["label"])
        if not pending.get(label):
            continue
        image = sample["image"]
        payload = image.get("bytes")
        if payload is None:
            continue
        payload = bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if digest in excluded:
            continue

        row_index = pending[label].pop(0)
        old_row = dict(rows[row_index])
        relative = Path(f"{label:04d}") / f"{digest[:20]}.JPEG"
        destination = root / relative
        destination.write_bytes(payload)
        rows[row_index] = {
            "relative_path": relative.as_posix(),
            "label": str(label),
            "source_name": image.get("path") or "",
            "stream_position": str(stream_position),
            "sha256": digest,
        }
        excluded.add(digest)
        repaired.append(
            {
                "label": label,
                "old_relative_path": old_row["relative_path"],
                "old_sha256": old_row["sha256"],
                "new_relative_path": relative.as_posix(),
                "new_sha256": digest,
            }
        )
        if not pending[label]:
            del pending[label]
        if not pending:
            break

    if pending:
        raise RuntimeError(f"unable to replace duplicates for labels {sorted(pending)}")

    temporary_manifest = manifest_path.with_suffix(".tsv.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(manifest_path)

    metadata["content_deduplication"] = {
        "policy": "keep first manifest occurrence; replace later occurrence from pinned stream",
        "replacements": repaired,
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)

    for repair in repaired:
        old_path = root / str(repair["old_relative_path"])
        if old_path != root / str(repair["new_relative_path"]):
            old_path.unlink()
    print(f"replaced {len(repaired)} duplicate-content row(s)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
