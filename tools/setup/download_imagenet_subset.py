#!/usr/bin/env python3
"""Stream a small, deterministic, class-balanced ImageNet-1K subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import pyarrow
import pyarrow.dataset
from datasets import Image, load_dataset
from huggingface_hub import HfApi


REPOSITORY = "ILSVRC/imagenet-1k"
EXPECTED_CLASSES = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_250_904)
    parser.add_argument("--shuffle-buffer", type=int, default=2_048)
    parser.add_argument("--read-batch", type=int, default=128)
    parser.add_argument("--range-size-mib", type=int, default=4)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="TSV manifest whose sha256 values must not enter the new subset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_class < 1:
        raise SystemExit("--per-class must be positive")
    if args.shuffle_buffer < 1:
        raise SystemExit("--shuffle-buffer must be positive")
    if args.read_batch < 1 or args.range_size_mib < 1:
        raise SystemExit("--read-batch and --range-size-mib must be positive")

    output = args.output.resolve()
    partial = output.with_name(f"{output.name}.partial")
    if output.exists():
        raise SystemExit(f"refusing to overwrite completed output: {output}")
    partial.mkdir(parents=True, exist_ok=True)

    revision = HfApi().dataset_info(REPOSITORY, token=True).sha
    if not revision:
        raise RuntimeError("Hugging Face did not return an immutable dataset revision")

    scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=1,
            range_size_limit=args.range_size_mib << 20,
        )
    )
    dataset = load_dataset(
        REPOSITORY,
        split=args.split,
        revision=revision,
        streaming=True,
        token=True,
        batch_size=args.read_batch,
        fragment_scan_options=scan_options,
    )
    dataset = dataset.cast_column("image", Image(decode=False))
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    counts = [0] * EXPECTED_CLASSES
    rows: list[dict[str, str | int]] = []
    selected_hashes: set[str] = set()
    target = EXPECTED_CLASSES * args.per_class
    excluded_hashes: set[str] = set()
    if args.exclude_manifest:
        with args.exclude_manifest.open("r", encoding="utf-8", newline="") as handle:
            excluded_hashes = {row["sha256"] for row in csv.DictReader(handle, delimiter="\t")}

    for stream_position, sample in enumerate(dataset):
        label = int(sample["label"])
        if not 0 <= label < EXPECTED_CLASSES:
            raise RuntimeError(f"unexpected ImageNet label {label}")
        if counts[label] >= args.per_class:
            continue

        image = sample["image"]
        payload = image.get("bytes")
        if payload is None:
            raise RuntimeError(f"sample at stream position {stream_position} has no image bytes")
        payload = bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if digest in excluded_hashes or digest in selected_hashes:
            continue
        relative_path = Path(f"{label:04d}") / f"{digest[:20]}.JPEG"
        destination = partial / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

        counts[label] += 1
        selected_hashes.add(digest)
        rows.append(
            {
                "relative_path": relative_path.as_posix(),
                "label": label,
                "source_name": image.get("path") or "",
                "stream_position": stream_position,
                "sha256": digest,
            }
        )
        if len(rows) % 100 == 0:
            print(f"saved {len(rows)}/{target} images", flush=True)
        if len(rows) == target:
            break

    incomplete = [label for label, count in enumerate(counts) if count != args.per_class]
    if incomplete:
        raise RuntimeError(
            f"selection ended with {len(incomplete)} incomplete classes; first labels: {incomplete[:10]}"
        )

    rows.sort(key=lambda row: (int(row["label"]), str(row["relative_path"])))
    with (partial / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("relative_path", "label", "source_name", "stream_position", "sha256"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    selection = {
        "dataset": REPOSITORY,
        "revision": revision,
        "split": args.split,
        "class_count": EXPECTED_CLASSES,
        "per_class": args.per_class,
        "image_count": len(rows),
        "selection": "streaming shuffle followed by first per-class quota",
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "read_batch": args.read_batch,
        "range_size_mib": args.range_size_mib,
        "excluded_manifest": args.exclude_manifest.as_posix() if args.exclude_manifest else None,
        "excluded_sha256_count": len(excluded_hashes),
    }
    (partial / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.rename(output)
    print(f"completed {len(rows)} images at {output}", flush=True)
    # Remote Parquet readers may leave non-daemon prefetch workers waiting on
    # already-unneeded shards.  The finalized directory is durable at this
    # point, so terminate those workers instead of delaying command completion.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
