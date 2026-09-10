#!/usr/bin/env python3
"""Create deterministic YOLO labels and a dataset file from COCO annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def prepare(annotation_path: Path, dataset_root: Path) -> dict[str, int]:
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in source["images"]}
    categories = sorted(source["categories"], key=lambda row: int(row["id"]))
    category_index = {int(row["id"]): index for index, row in enumerate(categories)}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for annotation in source["annotations"]:
        if not annotation.get("iscrowd", 0):
            grouped[int(annotation["image_id"])].append(annotation)

    # The dataset list uses images/val2017; img2label_paths replaces that
    # component with labels before opening payloads through the symlink.
    label_root = dataset_root / "labels" / "val2017"
    label_root.mkdir(parents=True, exist_ok=True)
    image_link = dataset_root / "images" / "val2017"
    image_link.parent.mkdir(parents=True, exist_ok=True)
    if image_link.is_symlink() and image_link.resolve() != (dataset_root / "val2017").resolve():
        image_link.unlink()
    if not image_link.exists():
        image_link.symlink_to(Path("../val2017"), target_is_directory=True)
    for image_id, image in images.items():
        width, height = float(image["width"]), float(image["height"])
        lines = []
        for annotation in sorted(grouped.get(image_id, []), key=lambda row: int(row["id"])):
            x, y, w, h = map(float, annotation["bbox"])
            lines.append(
                f"{category_index[int(annotation['category_id'])]} "
                f"{(x + w / 2) / width:.12g} {(y + h / 2) / height:.12g} "
                f"{w / width:.12g} {h / height:.12g}"
            )
        (label_root / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    names = {index: row["name"] for index, row in enumerate(categories)}
    list_path = dataset_root / "val2017.txt"
    list_path.write_text(
        "".join(f"./images/val2017/{images[key]['file_name']}\n" for key in sorted(images)), encoding="utf-8"
    )
    cache = dataset_root / "labels" / "val2017.cache"
    if cache.exists():
        cache.unlink()
    yaml_path = dataset_root / "coco2017-local.yaml"
    yaml_path.write_text(
        f"path: {dataset_root.resolve()}\ntrain: images/train2017\nval: val2017.txt\n" +
        "names:\n" + "".join(f"  {key}: {json.dumps(value)}\n" for key, value in names.items()),
        encoding="utf-8",
    )
    return {"images": len(images), "annotations": sum(map(len, grouped.values())), "categories": len(categories)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.annotations, args.dataset_root), sort_keys=True))


if __name__ == "__main__":
    main()
