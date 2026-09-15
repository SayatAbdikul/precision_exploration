"""Seeded paired image resampling; mAP is always a dataset-level metric."""
from __future__ import annotations

import contextlib
import copy
import io

import numpy as np


def settings(confidence, resamples, seed):
    if not 0 < confidence < 1 or type(resamples) is not int or resamples < 2 or type(seed) is not int or seed < 0:
        raise ValueError("invalid bootstrap settings")
    return np.random.default_rng(seed)


def interval(values, confidence):
    alpha = (1 - confidence) / 2
    return [float(v) for v in np.quantile(values, [alpha, 1-alpha], method="linear")]


def paired_classification(rows, expected_ids, *, confidence=.95, resamples=5000, seed=310911):
    rng = settings(confidence, resamples, seed)
    ids = [row["sample_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or len(set(expected_ids)) != len(expected_ids) or ids != list(expected_ids):
        raise ValueError("paired rows must exactly match the unique ordered frozen image IDs")
    baseline, candidate = [], []
    for row in rows:
        truth = row["ground_truth"]
        if type(truth) is not int or truth < 0:
            raise ValueError("invalid classification ground truth")
        outcomes = []
        for key in ("fp32_prediction", "candidate_prediction"):
            prediction = row[key]
            if len(prediction) != 5 or len(set(prediction)) != 5 or any(type(v) is not int or v < 0 for v in prediction):
                raise ValueError("paired predictions need five distinct class IDs")
            outcomes.append((prediction[0] == truth, truth in prediction))
        baseline.append(outcomes[0]); candidate.append(outcomes[1])
    fp32, quantized = np.asarray(baseline, dtype=np.int8), np.asarray(candidate, dtype=np.int8)
    delta = quantized - fp32
    sampled = np.empty((resamples, 2), dtype=np.float64)
    for start in range(0, resamples, 128):
        draws = rng.integers(0, len(rows), size=(min(128, resamples-start), len(rows)))
        sampled[start:start+len(draws)] = delta[draws].mean(axis=1)
    metrics = {}
    for column, name in enumerate(("top1", "top5")):
        a, b = fp32[:, column].astype(bool), quantized[:, column].astype(bool)
        metrics[name] = {"fp32": float(a.mean()), "candidate": float(b.mean()), "delta": float(delta[:, column].mean()),
                         "delta_interval": interval(sampled[:, column], confidence),
                         "paired_counts": {"both_correct": int((a & b).sum()), "fp32_only": int((a & ~b).sum()),
                                           "candidate_only": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum())}}
    return {"method": "paired_image_percentile_bootstrap_v1", "images": len(rows), "confidence": confidence,
            "resamples": resamples, "seed": seed, "units": "fraction", "metrics": metrics}


def resampled_coco(annotations, prediction_sets, draws):
    """Clone repeated images and annotations so COCOeval cannot deduplicate draws."""
    images = {row["id"]: row for row in annotations["images"]}
    if len(images) != len(annotations["images"]) or not draws or any(value not in images for value in draws):
        raise ValueError("invalid COCO image population/draws")
    truth_by_image = {key: [] for key in images}
    seen_annotations = set()
    for row in annotations["annotations"]:
        if row["id"] in seen_annotations or row["image_id"] not in images:
            raise ValueError("invalid COCO annotation identity")
        seen_annotations.add(row["id"])
        truth_by_image[row["image_id"]].append(row)
    predictions_by_image = []
    for predictions in prediction_sets:
        grouped = {key: [] for key in images}
        for row in predictions:
            if row["image_id"] not in grouped:
                raise ValueError("prediction outside the COCO population")
            grouped[row["image_id"]].append(row)
        predictions_by_image.append(grouped)
    output = {key: copy.deepcopy(value) for key, value in annotations.items() if key not in {"images", "annotations"}}
    output.update(images=[], annotations=[])
    results = [[] for _ in prediction_sets]
    for new_id, old_id in enumerate(draws, 1):
        output["images"].append({**images[old_id], "id": new_id})
        for row in truth_by_image[old_id]:
            output["annotations"].append({**row, "id": len(output["annotations"])+1, "image_id": new_id})
        for index, grouped in enumerate(predictions_by_image):
            results[index].extend({**row, "image_id": new_id} for row in grouped[old_id])
    return output, results


def coco_metrics(annotations, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = copy.deepcopy(annotations)
        truth.createIndex()
        if predictions:
            result = truth.loadRes(copy.deepcopy(predictions))
        else:
            result = COCO()
            result.dataset = {**copy.deepcopy(annotations), "annotations": []}
            result.createIndex()
        evaluation = COCOeval(truth, result, "bbox")
        evaluation.params.imgIds = [row["id"] for row in annotations["images"]]
        evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
    metrics = np.asarray([evaluation.stats[0], evaluation.stats[1]], dtype=np.float64)
    if not np.isfinite(metrics).all() or np.any(metrics < 0):
        raise ValueError("COCO population has no evaluable ground truth")
    return metrics


def paired_coco(annotations, fp32, candidate, expected_ids, *, confidence=.95, resamples=2000, seed=310911):
    rng = settings(confidence, resamples, seed)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("COCO screen IDs must be nonempty and unique")
    ids = list(expected_ids)
    # This also restricts baseline evaluation to exactly the candidate subset.
    subset, (a, b) = resampled_coco(annotations, [fp32, candidate], ids)
    baseline, quantized = coco_metrics(subset, a), coco_metrics(subset, b)
    sampled = np.empty((resamples, 2))
    for index in range(resamples):
        draws = [ids[int(i)] for i in rng.integers(0, len(ids), size=len(ids))]
        population, (left, right) = resampled_coco(annotations, [fp32, candidate], draws)
        sampled[index] = coco_metrics(population, right) - coco_metrics(population, left)
    return {"method": "paired_coco_image_clone_percentile_bootstrap_v1", "images": len(ids),
            "confidence": confidence, "resamples": resamples, "seed": seed, "units": "fraction",
            "metrics": {name: {"fp32": float(baseline[i]), "candidate": float(quantized[i]),
                               "delta": float(quantized[i]-baseline[i]), "delta_interval": interval(sampled[:, i], confidence)}
                        for i, name in enumerate(("map50_95", "map50"))}}


def screening_label(statistics, policy, *, diagnosis_verified=False):
    metrics = statistics["metrics"]
    primary = metrics["top1"] if "top1" in metrics else metrics["map50_95"]
    if primary["delta_interval"][1] < -policy["catastrophic_min_loss"]:
        return {"label": "CATASTROPHIC/BROKEN" if diagnosis_verified else "UNCERTAIN",
                "diagnosis_required": not diagnosis_verified, "reason": "primary quality loss exceeds catastrophic threshold"}
    promising = all(row["delta_interval"][0] >= -policy["promising_max_loss"] for row in metrics.values())
    return {"label": "PROMISING" if promising else "UNCERTAIN", "diagnosis_required": False,
            "reason": "all quality intervals satisfy the preservation tolerance" if promising else "retain for more evidence"}
