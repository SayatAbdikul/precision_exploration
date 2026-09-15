import copy
import pytest

from public.analysis.phase3.statistics import paired_classification, resampled_coco, paired_coco, screening_label


def rows():
    return [{"sample_id": str(i), "ground_truth": i, "fp32_prediction": [i, 10, 11, 12, 13],
             "candidate_prediction": [i, 10, 11, 12, 13]} for i in range(4)]


def test_identical_classifiers_have_zero_paired_intervals():
    result = paired_classification(rows(), ["0", "1", "2", "3"], resamples=100)
    for metric in result["metrics"].values():
        assert metric["delta"] == 0 and metric["delta_interval"] == [0, 0]
        assert metric["paired_counts"] == {"both_correct": 4, "fp32_only": 0, "candidate_only": 0, "both_wrong": 0}


def test_paired_deltas_sign_and_determinism():
    data = rows()
    data[0]["candidate_prediction"] = [10, 0, 11, 12, 13]
    first = paired_classification(data, ["0", "1", "2", "3"], resamples=100)
    assert first == paired_classification(data, ["0", "1", "2", "3"], resamples=100)
    assert first["metrics"]["top1"]["delta"] == -.25
    assert first["metrics"]["top5"]["delta"] == 0
    assert first["metrics"]["top1"]["paired_counts"]["fp32_only"] == 1


@pytest.mark.parametrize("ids", [["0", "1", "2"], ["0", "0", "2", "3"], ["1", "0", "2", "3"]])
def test_bootstrap_refuses_incomplete_duplicate_or_misaligned_pairs(ids):
    with pytest.raises(ValueError, match="unique ordered"):
        paired_classification(rows(), ids)


def test_catastrophic_quality_cannot_be_pruned_without_diagnosis():
    stats = {"metrics": {"top1": {"delta_interval": [-.9, -.8]}}}
    policy = {"catastrophic_min_loss": .2, "promising_max_loss": .01}
    assert screening_label(stats, policy)["label"] == "UNCERTAIN"
    assert screening_label(stats, policy)["diagnosis_required"]
    assert screening_label(stats, policy, diagnosis_verified=True)["label"] == "CATASTROPHIC/BROKEN"


def coco():
    annotations = {"images": [{"id": 7, "height": 100, "width": 100}, {"id": 9, "height": 100, "width": 100}],
                   "categories": [{"id": 1, "name": "object"}],
                   "annotations": [{"id": i, "image_id": image, "category_id": 1, "bbox": [0, 0, 10, 10],
                                     "area": 100, "iscrowd": 0} for i, image in enumerate((7, 9), 1)]}
    predictions = [{"image_id": image, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .9} for image in (7, 9)]
    return annotations, predictions


def test_coco_resampling_retains_duplicate_draw_multiplicity_without_mutating_inputs():
    annotations, predictions = coco()
    original = copy.deepcopy(annotations)
    output, (a, b) = resampled_coco(annotations, [predictions, predictions], [9, 9, 7])
    assert [row["id"] for row in output["images"]] == [1, 2, 3]
    assert [row["image_id"] for row in output["annotations"]] == [1, 2, 3]
    assert len({row["id"] for row in output["annotations"]}) == 3
    assert [row["image_id"] for row in a] == [1, 2, 3] and a == b
    assert original == annotations


def test_coco_subset_reference_and_identical_predictions_have_zero_intervals():
    pytest.importorskip("pycocotools")
    annotations, predictions = coco()
    result = paired_coco(annotations, predictions, predictions, [9], resamples=3)
    assert result["images"] == 1
    assert result["metrics"]["map50_95"]["fp32"] == pytest.approx(1)
    assert result["metrics"]["map50_95"]["delta_interval"] == [0, 0]


def test_coco_empty_candidate_is_zero_quality_not_missing_population():
    pytest.importorskip("pycocotools")
    annotations, predictions = coco()
    result = paired_coco(annotations, predictions, [], [7, 9], resamples=3)
    assert result["metrics"]["map50_95"]["candidate"] == 0
    assert result["metrics"]["map50_95"]["delta"] == pytest.approx(-1)
