"""Deterministic selection and hashing for fixed workload subsets."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class ImageNetSample:
    sample_id: str
    label: int


@dataclass(frozen=True, order=True)
class CocoSample:
    sample_id: str
    categories: tuple[int, ...]
    object_sizes: tuple[str, ...]


def _rank(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()


def _unique_ids(samples: Iterable[ImageNetSample | CocoSample]) -> None:
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample ID: {sample.sample_id}")
        seen.add(sample.sample_id)


def select_imagenet_class_balanced(
    samples: Sequence[ImageNetSample], *, per_class: int, seed: int
) -> list[ImageNetSample]:
    """Select an exact number per class using a stable content-based rank."""

    if per_class < 1:
        raise ValueError("per_class must be positive")
    _unique_ids(samples)
    grouped: dict[int, list[ImageNetSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    selected: list[ImageNetSample] = []
    for label in sorted(grouped):
        ordered = sorted(grouped[label], key=lambda item: (_rank(seed, item.sample_id), item.sample_id))
        if len(ordered) < per_class:
            raise ValueError(f"class {label} has {len(ordered)} samples; {per_class} required")
        selected.extend(ordered[:per_class])
    return selected


def select_imagenet_nested_evaluation(
    samples: Sequence[ImageNetSample], *, per_class_counts: Sequence[int], seed: int
) -> dict[int, list[ImageNetSample]]:
    """Create explicitly nested class-balanced evaluation sets.

    A count of 1 gives 1,000 samples on ImageNet-1K and a count of 10 gives
    10,000. Each smaller set is a prefix by class of every larger set.
    """

    counts = sorted(set(per_class_counts))
    if not counts or counts[0] < 1:
        raise ValueError("per_class_counts must contain positive values")
    _unique_ids(samples)
    grouped: dict[int, list[ImageNetSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    maximum = counts[-1]
    ordered_by_label: dict[int, list[ImageNetSample]] = {}
    for label, values in grouped.items():
        ordered = sorted(values, key=lambda item: (_rank(seed, item.sample_id), item.sample_id))
        if len(ordered) < maximum:
            raise ValueError(f"class {label} has {len(ordered)} samples; {maximum} required")
        ordered_by_label[label] = ordered
    return {
        count: [sample for label in sorted(ordered_by_label) for sample in ordered_by_label[label][:count]]
        for count in counts
    }


def _coco_stratum(sample: CocoSample) -> tuple[int, str, int]:
    primary = min(sample.categories) if sample.categories else -1
    size = min(sample.object_sizes) if sample.object_sizes else "none"
    object_count_bin = min(len(sample.categories), 4)
    return primary, size, object_count_bin


def select_coco_stratified(samples: Sequence[CocoSample], *, count: int, seed: int) -> list[CocoSample]:
    """Deterministic round-robin across category/size/object-count strata."""

    if not 0 < count <= len(samples):
        raise ValueError("count must be positive and no larger than the sample inventory")
    _unique_ids(samples)
    grouped: dict[tuple[int, str, int], deque[CocoSample]] = {}
    temporary: dict[tuple[int, str, int], list[CocoSample]] = defaultdict(list)
    for sample in samples:
        temporary[_coco_stratum(sample)].append(sample)
    for key, values in temporary.items():
        grouped[key] = deque(sorted(values, key=lambda item: (_rank(seed, item.sample_id), item.sample_id)))

    selected: list[CocoSample] = []
    keys = sorted(grouped)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if grouped[key]:
                selected.append(grouped[key].popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("stratified selection exhausted unexpectedly")
    return selected


def write_ordered_list(path: str | Path, sample_ids: Iterable[str]) -> dict[str, int | str]:
    identifiers = list(sample_ids)
    if not identifiers:
        raise ValueError("cannot write an empty subset list")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("subset list contains duplicate sample IDs")
    payload = ("\n".join(identifiers) + "\n").encode("utf-8")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return {
        "count": len(identifiers),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "path": output.as_posix(),
    }


def assert_disjoint(*groups: Iterable[str]) -> None:
    materialized = [set(group) for group in groups]
    for left_index, left in enumerate(materialized):
        for right in materialized[left_index + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(f"sample lists overlap; example: {sorted(overlap)[0]}")
