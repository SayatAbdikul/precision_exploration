import numpy as np
import pytest

from public.analysis.phase3.coco_cache import CocoImageCache, paired_coco_cached
from public.analysis.phase3.statistics import coco_metrics, paired_coco, resampled_coco


def fixture():
    annotations = {"images": [{"id": i, "width": 100, "height": 100} for i in (9, 3, 8)],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "absent"}],
        "annotations": [
            {"id": 10, "image_id": 9, "category_id": 1, "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 11, "image_id": 3, "category_id": 1, "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 12, "image_id": 3, "category_id": 2, "bbox": [20, 20, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 13, "image_id": 9, "category_id": 2, "bbox": [20, 20, 50, 50], "area": 2500, "iscrowd": 1}]}
    # Equal scores, duplicate detections, a crowd match, and an empty image
    # make draw ordering, ignore masks and multiplicity observable in AP.
    predictions = [
        {"image_id": 9, "category_id": 1, "bbox": [1, 1, 10, 10], "score": .5},
        {"image_id": 9, "category_id": 1, "bbox": [1, 1, 10, 10], "score": .5},
        {"image_id": 9, "category_id": 2, "bbox": [20, 20, 50, 50], "score": .5},
        {"image_id": 3, "category_id": 1, "bbox": [40, 40, 10, 10], "score": .5},
        {"image_id": 3, "category_id": 2, "bbox": [20, 20, 50, 50], "score": .5},
        {"image_id": 8, "category_id": 1, "bbox": [1, 1, 10, 10], "score": .5}]
    return annotations, predictions


@pytest.mark.parametrize("draws", [[9, 9, 3], [3, 9, 9], [8, 3, 3, 3, 9], [9], [3, 8]])
def test_cached_matching_equals_official_repeated_image_evaluation(draws):
    annotations, predictions = fixture()
    population, (cloned,) = resampled_coco(annotations, [predictions], draws)
    expected = coco_metrics(population, cloned)
    actual = CocoImageCache(annotations, predictions).metrics(draws)
    np.testing.assert_array_equal(actual, expected)


def test_bootstrap_intervals_are_unchanged_by_cached_matching():
    annotations, predictions = fixture()
    candidate = predictions[2:]
    # Avoid all-empty draws here; the separate test checks that failure case.
    args = (annotations, predictions, candidate, [9, 3])
    assert paired_coco_cached(*args, resamples=12, seed=31) == paired_coco(*args, resamples=12, seed=31)


def test_empty_predictions_and_all_empty_population_match_reference_rules():
    annotations, _ = fixture()
    cache = CocoImageCache(annotations, [])
    np.testing.assert_array_equal(cache.metrics([9, 3, 9]), [0, 0])
    with pytest.raises(ValueError, match="no evaluable"):
        cache.metrics([8, 8, 8])
