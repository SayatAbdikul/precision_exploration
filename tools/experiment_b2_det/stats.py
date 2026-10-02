"""COCO statistics for the detector study: the real evaluator, the project's paired image bootstrap,
and the dependence of AP on the order of equal scores.

``Evaluated`` runs ``COCOeval.evaluate`` once (through the project's ``CocoImageCache``) and then calls
``COCOeval.accumulate`` per draw, restricted to the area range 'all' and maxDets 100 - the only slices
that mAP50-95 and mAP50 read.  ``Evaluated.check`` asserts equality with the unrestricted evaluator.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json

import numpy as np

from tools.experiment_b.common import ROOT, file_hash

RESAMPLES, SEED, CONFIDENCE = 2000, 310911, 0.95


def annotations():
    manifest = json.loads((ROOT / "public/workloads/datasets/coco2017.json").read_text())
    path = ROOT / "data/raw/coco2017/annotations/instances_val2017.json"
    if file_hash(path) != manifest["annotation_sha256"]["instances_val2017"]:
        raise ValueError("COCO annotation drift")
    return json.loads(path.read_text())


class Evaluated:
    def __init__(self, truth, detections, image_ids):
        """``image_ids`` in evaluation order (this order breaks score ties across images)."""
        from public.analysis.phase3.coco_cache import CocoImageCache
        from public.analysis.phase3.statistics import resampled_coco
        subset, (results,) = resampled_coco(truth, [detections], list(image_ids))
        self.cache = CocoImageCache(subset, results)
        self.images = len(image_ids)
        parameters = self.cache.parameters
        self.categories, self.areas = len(parameters.catIds), len(parameters.areaRng)
        if tuple(parameters.areaRng[0]) != (0, 1e10) or parameters.maxDets[-1] != 100 or parameters.iouThrs[0] != 0.5:
            raise ValueError("unexpected COCOeval parameter layout")

    def _rows(self, indices):
        rows, n = self.cache.rows, self.images
        return [rows[k * self.areas * n + i] for k in range(self.categories) for i in indices]

    def metrics(self, draws=None):
        """[mAP50-95, mAP50] for 1-based image positions ``draws`` (default: every image once)."""
        from pycocotools.cocoeval import COCOeval
        indices = range(self.images) if draws is None else [d - 1 for d in draws]
        evaluation = COCOeval()
        evaluation.params = copy.deepcopy(self.cache.parameters)
        evaluation.params.imgIds = list(range(1, len(indices) + 1))
        evaluation.params.areaRng = [evaluation.params.areaRng[0]]
        evaluation.params.areaRngLbl = ["all"]
        evaluation.params.maxDets = [100]
        evaluation._paramsEval = copy.deepcopy(evaluation.params)
        evaluation.evalImgs = self._rows(indices)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.accumulate()
        precision = evaluation.eval["precision"][:, :, :, 0, 0]
        if not (precision > -1).any():
            raise ValueError("COCO population has no evaluable ground truth")
        return np.array([precision[precision > -1].mean(), precision[0][precision[0] > -1].mean()])

    def check(self):
        full = self.cache.metrics(list(range(1, self.images + 1)))
        fast = self.metrics()
        if not np.array_equal(full, fast):
            raise ValueError(f"restricted accumulate differs from COCOeval: {full} {fast}")
        return fast

    def bootstrap(self, resamples=RESAMPLES, seed=SEED):
        """Per-draw metrics with the draw sequence of ``paired_coco_cached`` (same generator, same calls)."""
        from public.analysis.phase3.statistics import settings
        rng = settings(CONFIDENCE, resamples, seed)
        sampled = np.empty((resamples, 2))
        for index in range(resamples):
            sampled[index] = self.metrics((rng.integers(0, self.images, size=self.images) + 1).tolist())
        return sampled

    def tie_statistics(self):
        """Envelope of AP over every order of equal scores across detections, at fixed image-local matching."""
        rows = self._rows(range(self.images))
        thresholds = np.linspace(0.0, 1.0, 101)
        result = {mode: [] for mode in ("stable", "true_positives_first", "false_positives_first")}
        detections = tied = 0
        for k in range(self.categories):
            entries = [e for e in rows[k * self.images:(k + 1) * self.images] if e is not None]
            if not entries:
                continue
            scores = np.concatenate([e["dtScores"][:100] for e in entries])
            matched = np.concatenate([e["dtMatches"][:, :100] for e in entries], axis=1) != 0
            ignored = np.concatenate([e["dtIgnore"][:, :100] for e in entries], axis=1) != 0
            positives = np.count_nonzero(np.concatenate([e["gtIgnore"] for e in entries]) == 0)
            if positives == 0:
                continue
            _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
            detections += len(scores)
            tied += int((counts[inverse] > 1).sum())
            for mode in result:
                values = np.zeros((matched.shape[0], 101))
                for t in range(matched.shape[0]):
                    tp, fp = matched[t] & ~ignored[t], ~matched[t] & ~ignored[t]
                    secondary = {"stable": np.zeros(len(scores)), "true_positives_first": ~tp,
                                 "false_positives_first": tp}[mode]
                    order = np.lexsort((np.arange(len(scores)), secondary, -scores))
                    tps, fps = np.cumsum(tp[order]).astype(float), np.cumsum(fp[order]).astype(float)
                    recall, precision = tps / positives, tps / (fps + tps + np.spacing(1))
                    precision = np.maximum.accumulate(precision[::-1])[::-1]
                    at = np.searchsorted(recall, thresholds, side="left")
                    inside = at < len(precision)
                    values[t, inside] = precision[at[inside]]
                result[mode].append(values)
        summary = {mode: [float(np.mean(np.stack(v))), float(np.mean(np.stack(v)[:, 0]))] for mode, v in result.items()}
        summary.update(detections_scored=int(detections), detections_with_a_tied_score=int(tied))
        return summary


def interval(values, confidence=CONFIDENCE):
    alpha = (1 - confidence) / 2
    return [float(v) for v in np.quantile(values, [alpha, 1 - alpha], method="linear")]


def paired(left, right):
    """``left``/``right``: ``{"point": [2], "draws": [R, 2]}`` from the same draw sequence. right minus left, in points."""
    if left["draws"].shape != right["draws"].shape:
        raise ValueError("bootstrap vectors are not paired")
    difference = 100 * (right["draws"] - left["draws"])
    return {name: {"left": float(100 * left["point"][i]), "right": float(100 * right["point"][i]),
                   "delta": float(100 * (right["point"][i] - left["point"][i])), "interval": interval(difference[:, i])}
            for i, name in enumerate(("map50_95", "map50"))}
