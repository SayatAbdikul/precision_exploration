#!/usr/bin/env python3
"""Download a reproducible COCO image set without the full training archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org"


@dataclass(frozen=True)
class Record:
    image_id: int
    file_name: str
    width: int
    height: int
    stratum: tuple[int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", choices=("train2017", "val2017"), required=True)
    parser.add_argument("--count", type=int, help="omit to download every annotated image")
    parser.add_argument("--screen-count", type=int, help="write a nested screen manifest")
    parser.add_argument("--seed", type=int, default=20_250_904)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def size_bin(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    return "10+"


def stable_rank(seed: int, image_id: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{image_id}".encode()).digest()


def stable_stratum_rank(seed: int, key: tuple[int, str, str]) -> bytes:
    return hashlib.sha256(f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode()).digest()


def inventory(document: dict[str, Any]) -> list[Record]:
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in document["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations[int(annotation["image_id"])].append(annotation)

    records = []
    for image in document["images"]:
        image_id = int(image["id"])
        objects = annotations[image_id]
        if objects:
            primary = max(objects, key=lambda item: (float(item["area"]), -int(item["category_id"])))
            category = int(primary["category_id"])
            object_size = size_bin(float(primary["area"]))
        else:
            category = -1
            object_size = "none"
        records.append(
            Record(
                image_id=image_id,
                file_name=str(image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
                stratum=(category, object_size, count_bin(len(objects))),
            )
        )
    return records


def proportional_sample(records: list[Record], count: int, seed: int) -> list[Record]:
    if not 0 < count <= len(records):
        raise ValueError(f"count must be in 1..{len(records)}")
    if count == len(records):
        return sorted(records, key=lambda item: item.image_id)

    groups: dict[tuple[int, str, str], list[Record]] = defaultdict(list)
    for record in records:
        groups[record.stratum].append(record)

    exact = {key: len(group) * count / len(records) for key, group in groups.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = count - sum(quotas.values())
    remainder_order = sorted(
        groups,
        key=lambda key: (-(exact[key] - quotas[key]), stable_stratum_rank(seed, key)),
    )
    for key in remainder_order[:remainder]:
        quotas[key] += 1

    selected = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda item: (stable_rank(seed, item.image_id), item.image_id))
        selected.extend(ranked[: quotas[key]])
    return sorted(selected, key=lambda item: (stable_rank(seed, item.image_id), item.image_id))


def download_one(split: str, partial: Path, record: Record) -> tuple[Record, str]:
    destination = partial / record.file_name
    if destination.exists():
        payload = destination.read_bytes()
        with Image.open(destination) as image:
            image.verify()
        return record, hashlib.sha256(payload).hexdigest()

    url = f"{BASE_URL}/{split}/{record.file_name}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "precision-exploration/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            temporary.write_bytes(payload)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(destination)
            return record, hashlib.sha256(payload).hexdigest()
        except Exception as error:  # transport and image validation both retry
            last_error = error
            if temporary.exists():
                temporary.unlink()
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def write_manifest(path: Path, records: list[Record], digests: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("file_name", "image_id", "width", "height", "sha256"))
        for record in records:
            writer.writerow(
                (record.file_name, record.image_id, record.width, record.height, digests[record.image_id])
            )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    annotations_path = args.annotations.resolve()
    annotation_bytes = annotations_path.read_bytes()
    document = json.loads(annotation_bytes)
    records = inventory(document)
    selected = proportional_sample(records, args.count or len(records), args.seed)
    selected_ids = {record.image_id for record in selected}

    output = args.output.resolve()
    partial = output.with_name(f"{output.name}.partial")
    if output.exists():
        if any(output.iterdir()):
            raise SystemExit(f"refusing to overwrite completed output: {output}")
        if partial.exists():
            raise SystemExit(f"both empty output and partial output exist: {output}, {partial}")
        output.rename(partial)
    else:
        partial.mkdir(parents=True, exist_ok=True)

    digests: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, args.split, partial, record) for record in selected]
        for completed, future in enumerate(as_completed(futures), 1):
            record, digest = future.result()
            digests[record.image_id] = digest
            if completed % 100 == 0 or completed == len(futures):
                print(f"downloaded/verified {completed}/{len(futures)} images", flush=True)

    write_manifest(partial / "manifest.tsv", selected, digests)
    screen_ids: set[int] = set()
    if args.screen_count:
        if args.screen_count > len(selected):
            raise ValueError("--screen-count cannot exceed --count")
        screen = proportional_sample(selected, args.screen_count, args.seed)
        screen_ids = {record.image_id for record in screen}
        if not screen_ids <= selected_ids:
            raise RuntimeError("screen set is not nested in the downloaded set")
        write_manifest(partial / "screen_manifest.tsv", screen, digests)

    population_strata = Counter(record.stratum for record in records)
    selection_strata = Counter(record.stratum for record in selected)
    metadata = {
        "dataset": "COCO 2017",
        "split": args.split,
        "annotation_file": annotations_path.name,
        "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "population_count": len(records),
        "image_count": len(selected),
        "screen_count": len(screen_ids),
        "seed": args.seed,
        "selection": "proportional primary-category/object-size/object-count stratification",
        "strata_population": {str(key): value for key, value in sorted(population_strata.items())},
        "strata_selection": {str(key): value for key, value in sorted(selection_strata.items())},
    }
    (partial / "selection.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.rename(output)
    print(f"completed {len(selected)} images at {output}", flush=True)


if __name__ == "__main__":
    main()
