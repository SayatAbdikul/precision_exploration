from __future__ import annotations

from pathlib import Path

import pytest

from public.workloads.subsets.selection import (
    CocoSample,
    ImageNetSample,
    assert_disjoint,
    select_coco_stratified,
    select_imagenet_class_balanced,
    select_imagenet_nested_evaluation,
    write_ordered_list,
)


def test_imagenet_selection_is_balanced_nested_and_deterministic() -> None:
    samples = [ImageNetSample(f"class-{label}/image-{index}.jpg", label) for label in range(4) for index in range(12)]
    calibration = select_imagenet_class_balanced(samples, per_class=2, seed=17)
    repeated = select_imagenet_class_balanced(samples, per_class=2, seed=17)
    assert calibration == repeated
    assert len(calibration) == 8

    nested = select_imagenet_nested_evaluation(samples, per_class_counts=[1, 5, 10], seed=91)
    assert len(nested[1]) == 4
    assert set(nested[1]) < set(nested[5]) < set(nested[10])


def test_coco_selection_is_deterministic_and_unique() -> None:
    samples = [
        CocoSample(str(index), (index % 3,), (("small", "medium", "large")[index % 3],))
        for index in range(30)
    ]
    selected = select_coco_stratified(samples, count=15, seed=23)
    assert selected == select_coco_stratified(samples, count=15, seed=23)
    assert len({sample.sample_id for sample in selected}) == 15


def test_list_write_hash_and_overlap_detection(tmp_path: Path) -> None:
    first = write_ordered_list(tmp_path / "first.txt", ["a", "b"])
    second = write_ordered_list(tmp_path / "second.txt", ["c", "d"])
    assert first["count"] == second["count"] == 2
    assert first["sha256"] != second["sha256"]
    assert_disjoint(["a", "b"], ["c", "d"])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(["a", "b"], ["b", "c"])
