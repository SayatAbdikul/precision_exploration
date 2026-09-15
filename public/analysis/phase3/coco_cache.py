"""Reuse image-local COCO matching while recomputing AP for every joint draw.

COCOeval.accumulate reads scores, match presence and ignore masks, not the
numeric identity of matched annotations. Repeating an image's evaluated rows
therefore equals cloning that image/its annotations and evaluating again. Rows
remain in draw order, preserving stable score-tie ordering and multiplicity.
"""
import contextlib
import copy
import io

import numpy as np

from public.analysis.phase3.statistics import interval, resampled_coco, settings


class CocoImageCache:
    def __init__(self, annotations, predictions):
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
            evaluation.evaluate()
        self.parameters = copy.deepcopy(evaluation._paramsEval)
        self.ids = list(self.parameters.imgIds)
        self.positions = {image_id: i for i, image_id in enumerate(self.ids)}
        self.rows = evaluation.evalImgs
        self.group_count = len(self.parameters.catIds)*len(self.parameters.areaRng)
        if not self.ids or len(self.rows) != self.group_count*len(self.ids):
            raise ValueError("COCO image cache has an unexpected evaluation layout")

    def metrics(self, draws):
        from pycocotools.cocoeval import COCOeval
        if not draws or any(image_id not in self.positions for image_id in draws):
            raise ValueError("invalid cached COCO image draw")
        indices = [self.positions[image_id] for image_id in draws]
        evaluation = COCOeval()
        evaluation.params = copy.deepcopy(self.parameters)
        # Unique virtual image positions prevent COCO's set membership checks
        # from deduplicating a repeated source image.
        evaluation.params.imgIds = list(range(1, len(draws)+1))
        evaluation._paramsEval = copy.deepcopy(evaluation.params)
        evaluation.evalImgs = [self.rows[group*len(self.ids)+index]
                               for group in range(self.group_count) for index in indices]
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.accumulate()
            evaluation.summarize()
        metrics = np.asarray([evaluation.stats[0], evaluation.stats[1]], dtype=np.float64)
        if not np.isfinite(metrics).all() or np.any(metrics < 0):
            raise ValueError("COCO population has no evaluable ground truth")
        return metrics


def paired_coco_cached(annotations, fp32, candidate, expected_ids, *, confidence=.95, resamples=2000, seed=310911):
    rng = settings(confidence, resamples, seed)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("COCO screen IDs must be nonempty and unique")
    subset, (a, b) = resampled_coco(annotations, [fp32, candidate], list(expected_ids))
    left, right = CocoImageCache(subset, a), CocoImageCache(subset, b)
    ids = list(range(1, len(expected_ids)+1))
    baseline, quantized = left.metrics(ids), right.metrics(ids)
    sampled = np.empty((resamples, 2), dtype=np.float64)
    for index in range(resamples):
        draws = (rng.integers(0, len(ids), size=len(ids))+1).tolist()
        sampled[index] = right.metrics(draws)-left.metrics(draws)
    return {"method": "paired_coco_image_clone_percentile_bootstrap_v1", "images": len(ids),
            "confidence": confidence, "resamples": resamples, "seed": seed, "units": "fraction",
            "metrics": {name: {"fp32": float(baseline[i]), "candidate": float(quantized[i]),
                               "delta": float(quantized[i]-baseline[i]), "delta_interval": interval(sampled[:, i], confidence)}
                        for i, name in enumerate(("map50_95", "map50"))}}
