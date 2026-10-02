"""Parts B (detector), C (detector tie orders) and D (agreement and error decomposition) from stored 1k detections.

One worker per configuration (``job``): it builds the project's COCOeval cache once (``stats.Evaluated``, the fixed
evaluation order = the 1k screen rows in ascending sha256) and writes one compact result file. Nothing is run on a
model; only the 1,000 screen rows of ``coco_screen_1k`` are evaluated.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, dataset, unseal
from . import SEED

RUNS = ROOT / "artifacts/experiment_b2_det/runs/1000"
WORK = ROOT / "artifacts/quality_metrics_v1/work/det"
ORDERS = 200
SUB_N = (128, 256, 512)
SUB_PERMS = 3
SUB_DRAWS = 500
AREA_DRAWS = 500
THRESHOLD = 0.25
SENTINELS = ("int8", "int6", "fp8_e4m3fn", "fp7_e3m3", "fp6_e2m3", "log8", "posit8_es1", "bfp6", "mxfp8_e4m3")
BLOCK = ("bfp6", "mxfp8_e4m3")
SUBSAMPLE = {("fp32", "fp32"), ("int8", "default"), ("int8", "conformant"), ("int6", "default"),
             ("fp8_e4m3fn", "default"), ("posit8_es1", "default")}
AREA_BOOT = {("fp32", "fp32"), ("int8", "default"), ("int6", "default")}


def configurations():
    """(format, recipe) of every unit of parts C and D, in a fixed order."""
    units = [("fp32", "fp32")]
    for fmt in SENTINELS:
        units.append((fmt, "default"))
        units.append((fmt, "default_fp32_box_logits" if fmt in BLOCK else "conformant"))
    units += [("int8", "default_head_logits"), ("int6", "default_head_logits")]
    return units


def run_record(fmt, recipe):
    found = sorted(RUNS.glob(f"{fmt}--{recipe}--*.json"))
    if len(found) != 1:
        raise ValueError(f"expected one run record for {fmt} {recipe}, found {len(found)}")
    payload = unseal(found[0])
    if payload["images"] != 1000 or payload["format"] != fmt or payload["recipe_name"] != recipe:
        raise ValueError(f"run record mismatch: {found[0]}")
    return payload, found[0]


def load(fmt, recipe):
    from tools.experiment_b2_det import runner
    payload, path = run_record(fmt, recipe)
    arrays = runner.load_detections(runner.detection_file(payload["configuration_sha256"], 1000))
    if len(arrays["score_e5"]) != payload["detections"]["index"]["detections"]:
        raise ValueError("detection count differs from the run record")
    return payload, path, arrays


def screen_rows():
    _, rows, _ = dataset("coco_screen_1k")
    if len(rows) != 1000:
        raise ValueError("coco_screen_1k is not 1,000 rows")
    return rows


def evaluated(arrays, rows, truth):
    from tools.experiment_b2_det.post import records
    from tools.experiment_b2_det.stats import Evaluated
    return Evaluated(truth, records(arrays, rows), [int(r["image_id"]) for r in rows])


# ----------------------------------------------------------------------------------------------- accumulate (C)
def tables(ev):
    """Per category (area 'all', maxDets 100): scores, matched / ignored (T x D), image and in-image position."""
    rows, n = ev._rows(range(ev.images)), ev.images
    out = []
    for k in range(ev.categories):
        entries = [(i, e) for i, e in enumerate(rows[k * n:(k + 1) * n]) if e is not None]
        if not entries:
            continue
        gt_ignore = np.concatenate([e["gtIgnore"] for _, e in entries]) if entries else np.zeros(0)
        positives = int(np.count_nonzero(gt_ignore == 0))
        if positives == 0:
            continue
        scores = np.concatenate([np.asarray(e["dtScores"][:100], dtype=np.float64) for _, e in entries])
        matched = np.concatenate([e["dtMatches"][:, :100] for _, e in entries], axis=1) != 0
        ignored = np.concatenate([e["dtIgnore"][:, :100] for _, e in entries], axis=1) != 0
        image = np.concatenate([np.full(len(e["dtScores"][:100]), i) for i, e in entries]).astype(np.int64)
        position = np.concatenate([np.arange(len(e["dtScores"][:100])) for _, e in entries]).astype(np.int64)
        out.append({"scores": scores, "tp": matched & ~ignored, "fp": ~matched & ~ignored, "image": image,
                    "position": position, "positives": positives})
    return out


RECALL = np.linspace(0.0, 1.0, 101)


def ap(tables_, key):
    """[mAP50-95, mAP50] with the order ``key(table)`` (indices) among each category's detections."""
    values = []
    for t in tables_:
        order = key(t)
        tps = np.cumsum(t["tp"][:, order], axis=1).astype(float)
        fps = np.cumsum(t["fp"][:, order], axis=1).astype(float)
        q = np.zeros((tps.shape[0], len(RECALL)))
        if tps.shape[1]:
            recall = tps / t["positives"]
            precision = tps / (fps + tps + np.spacing(1))
            precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
            for i in range(tps.shape[0]):
                at = np.searchsorted(recall[i], RECALL, side="left")
                inside = at < tps.shape[1]
                q[i, inside] = precision[i, at[inside]]
        values.append(q)
    stacked = np.stack(values)  # categories x T x R
    return np.array([stacked.mean(), stacked[:, 0].mean()])


def stable_key(t, rank=None):
    image = t["image"] if rank is None else rank[t["image"]]
    return np.lexsort((t["position"], image, -t["scores"]))


def tie_orders(tables_, orders=ORDERS, seed=SEED):
    image_order = np.empty((orders, 2))
    uniform = np.empty((orders, 2))
    images = 1 + max(int(t["image"].max()) for t in tables_ if len(t["image"]))
    for r in range(orders):
        rank = np.random.default_rng([seed, 1, r]).permutation(images)
        image_order[r] = ap(tables_, lambda t: stable_key(t, rank))
        rng = np.random.default_rng([seed, 2, r])
        uniform[r] = ap(tables_, lambda t: np.lexsort((rng.random(len(t["scores"])), -t["scores"])))
    return image_order, uniform


# ----------------------------------------------------------------------------------------------- full COCOeval (D)
def full_stats(ev, draws=None):
    """The 12 COCOeval summary statistics (areas small/medium/large included) over 1-based positions ``draws``."""
    from pycocotools.cocoeval import COCOeval
    cache, n = ev.cache, ev.images
    indices = range(n) if draws is None else [d - 1 for d in draws]
    evaluation = COCOeval()
    evaluation.params = copy.deepcopy(cache.parameters)
    evaluation.params.imgIds = list(range(1, len(indices) + 1))
    evaluation._paramsEval = copy.deepcopy(evaluation.params)
    evaluation.evalImgs = [cache.rows[g * n + i] for g in range(cache.group_count) for i in indices]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation.accumulate()
        evaluation.summarize()
    return np.asarray(evaluation.stats, dtype=np.float64)


def matches(ev):
    """Area 'all' matching: per ground truth id (ignored excluded) the matched detection id at IoU 0.5 and 0.75."""
    rows, n = ev._rows(range(ev.images)), ev.images
    gt = {}
    for k in range(ev.categories):
        for i, e in enumerate(rows[k * n:(k + 1) * n]):
            if e is None:
                continue
            for g, gid in enumerate(e["gtIds"]):
                if e["gtIgnore"][g]:
                    continue
                gt[int(gid)] = (i, int(e["gtMatches"][0, g]), int(e["gtMatches"][5, g]))
    det = {}
    for k in range(ev.categories):
        for i, e in enumerate(rows[k * n:(k + 1) * n]):
            if e is None:
                continue
            for d, did in enumerate(e["dtIds"]):
                det[int(did)] = (int(e["dtMatches"][0, d]), bool(e["dtIgnore"][0, d]))
    return gt, det


def xywh_iou(a, b):
    """IoU matrix between boxes a (m x 4) and b (n x 4) in xywh."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    w = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    h = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = w * h
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def owners(arrays):
    return np.repeat(np.arange(len(arrays["counts"])), arrays["counts"].astype(np.int64))


def agreement(fp32, q, gt_boxes, gt_area):
    """Part D metrics of configuration ``q`` against FP32; both: dicts with arrays, gt and det match tables."""
    from scipy.stats import kendalltau, spearmanr
    out = {}
    fs, qs = fp32["arrays"]["score_e5"] / 1e5, q["arrays"]["score_e5"] / 1e5
    for tau_name, tau in (("all", 0.0), ("s025", THRESHOLD)):
        for t, column in (("50", 1), ("75", 2)):
            kept = total = 0
            by_area = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
            for gid, entry in fp32["gt"].items():
                did = entry[column]
                if did == 0 or fs[did - 1] < tau:
                    continue
                total += 1
                qd = q["gt"][gid][column]
                hit = qd != 0 and qs[qd - 1] >= tau
                kept += hit
                area = gt_area[gid]
                bucket = "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")
                by_area[bucket][0] += hit
                by_area[bucket][1] += 1
            out[f"tp_kept_{tau_name}_iou{t}"] = kept / total if total else None
            out[f"fp32_tp_{tau_name}_iou{t}"] = total
            for bucket, (h, n) in by_area.items():
                out[f"tp_kept_{tau_name}_iou{t}_{bucket}"] = h / n if n else None
        # matched-in-Q objects that FP32 missed
        gained = sum(1 for gid, e in q["gt"].items() if e[1] != 0 and qs[e[1] - 1] >= tau
                     and (fp32["gt"][gid][1] == 0 or fs[fp32["gt"][gid][1] - 1] < tau))
        out[f"tp_gained_{tau_name}_iou50"] = gained
    # false positives at score >= 0.25, IoU 0.5 (detections beyond the 100 per image/category are not evaluated)
    fo, qo = owners(fp32["arrays"]), owners(q["arrays"])
    fcat, qcat = fp32["arrays"]["category"], q["arrays"]["category"]
    fbox, qbox = fp32["arrays"]["box_milli"] / 1000.0, q["arrays"]["box_milli"] / 1000.0
    def false_positives(side, scores):
        return [d for d, (m, ig) in side["det"].items() if m == 0 and not ig and scores[d - 1] >= THRESHOLD]
    qfp, ffp = false_positives(q, qs), false_positives(fp32, fs)
    fkeep = np.flatnonzero(fs >= THRESHOLD)
    index = {}
    for d in fkeep:
        index.setdefault((fo[d], fcat[d]), []).append(d)
    new = 0
    for did in qfp:
        d = did - 1
        cand = index.get((qo[d], qcat[d]), [])
        if not cand or xywh_iou(qbox[d], fbox[cand]).max() < 0.5:
            new += 1
    out.update(fp_q_s025=len(qfp), fp_fp32_s025=len(ffp), fp_new_s025=new,
               det_q_s025=int((qs >= THRESHOLD).sum()), det_fp32_s025=int((fs >= THRESHOLD).sum()))
    # box drift on objects matched by both at IoU 0.5, score >= 0.25
    ious, delta = [], []
    for gid, entry in fp32["gt"].items():
        fd, qd = entry[1], q["gt"][gid][1]
        if fd == 0 or qd == 0 or fs[fd - 1] < THRESHOLD or qs[qd - 1] < THRESHOLD:
            continue
        ious.append(xywh_iou(qbox[qd - 1], fbox[fd - 1])[0, 0])
        g = gt_boxes[gid]
        delta.append(xywh_iou(qbox[qd - 1], g)[0, 0] - xywh_iou(fbox[fd - 1], g)[0, 0])
    ious, delta = np.array(ious), np.array(delta)
    out.update(drift_pairs=len(ious), drift_iou_median=float(np.median(ious)) if len(ious) else None,
               drift_iou_mean=float(ious.mean()) if len(ious) else None,
               drift_iou_below_0_9=float((ious < 0.9).mean()) if len(ious) else None,
               gt_iou_change_mean=float(delta.mean()) if len(delta) else None)
    # score rank: greedy matching of FP32 detections (score >= 0.25, descending) to quantized detections, IoU >= 0.5
    qindex = {}
    for d in range(len(qs)):
        qindex.setdefault((qo[d], qcat[d]), []).append(d)
    pairs = []
    for key, fl in index.items():
        cand = qindex.get(key, [])
        if not cand:
            continue
        fl = sorted(fl, key=lambda d: (-fs[d], d))
        iou = xywh_iou(fbox[fl], qbox[cand])
        used = np.zeros(len(cand), bool)
        for a, d in enumerate(fl):
            row = np.where(used, -1.0, iou[a])
            b = int(row.argmax())
            if row[b] >= 0.5:
                used[b] = True
                pairs.append((fs[d], qs[cand[b]]))
    pairs = np.array(pairs)
    if len(pairs) > 2:
        rho = spearmanr(pairs[:, 0], pairs[:, 1]).statistic
        tau = kendalltau(pairs[:, 0], pairs[:, 1]).statistic
    else:
        rho = tau = None
    out.update(score_pairs=len(pairs), score_matched_share=len(pairs) / len(fkeep) if len(fkeep) else None,
               score_spearman=float(rho) if rho is not None else None, score_kendall_tau_b=float(tau) if tau is not None else None)
    return out


# ----------------------------------------------------------------------------------------------- score step (D)
def score_units(scores_e5, tolerance=2):
    """Infer the per-level score code units (in 1e-5) from stored quantized scores: each value is c * unit."""
    values = np.unique(scores_e5[scores_e5 > 0]).astype(np.int64)
    units = []
    for v in values:
        if not any(abs(v - round(v / u) * u) <= tolerance * max(1, round(v / u)) for u in units):
            units.append(int(v))
    unexplained = [int(v) for v in values if not any(abs(v - round(v / u) * u) <= tolerance * max(1, round(v / u)) for u in units)]
    return units, unexplained


def requantised(arrays, unit_e5, levels):
    """Scores rounded to multiples of ``unit_e5`` (codes 0..levels-1); code 0 dropped. Boxes and order unchanged."""
    s = arrays["score_e5"].astype(np.float64)
    code = np.clip(np.round(s / unit_e5), 0, levels - 1)
    keep = code > 0
    o = owners(arrays)
    counts = np.bincount(o[keep], minlength=len(arrays["counts"])).astype(np.uint16)
    return {"counts": counts, "category": arrays["category"][keep], "box_milli": arrays["box_milli"][keep],
            "score_e5": np.round(code[keep] * unit_e5).astype(np.int32)}


# ----------------------------------------------------------------------------------------------- the worker
def side(fmt, recipe, rows, truth):
    payload, path, arrays = load(fmt, recipe)
    ev = evaluated(arrays, rows, truth)
    point = ev.check()
    if abs(point[0] - payload["map50_95"]) > 1e-9 or abs(point[1] - payload["map50"]) > 1e-9:
        raise ValueError(f"stored detections do not reproduce the run record: {fmt} {recipe}")
    gt, det = matches(ev)
    return {"payload": payload, "path": path, "arrays": arrays, "ev": ev, "point": point, "gt": gt, "det": det}


def subsample(ev):
    """Block metrics: [perm][n] -> array (blocks, 1 + draws, 2); draw sequences are shared by all configurations."""
    result = {}
    for r in range(SUB_PERMS):
        perm = np.random.default_rng([SEED, 3, r]).permutation(ev.images)
        for n in SUB_N:
            blocks = []
            for b in range(ev.images // n):
                block = perm[b * n:(b + 1) * n]
                draws = np.random.default_rng([SEED, 4, r, n, b]).integers(0, n, size=(SUB_DRAWS, n))
                values = [ev.metrics((block + 1).tolist())]
                values += [ev.metrics((block[d] + 1).tolist()) for d in draws]
                blocks.append(values)
            result[f"r{r}_n{n}"] = np.array(blocks)
    return result


def area_bootstrap(ev):
    draws = np.random.default_rng([SEED, 5]).integers(0, ev.images, size=(AREA_DRAWS, ev.images))
    return np.array([full_stats(ev, (d + 1).tolist()) for d in draws])


def job(spec):
    fmt, recipe = spec
    from tools.experiment_b2_det.stats import annotations
    WORK.mkdir(parents=True, exist_ok=True)
    target = WORK / f"{fmt}--{recipe}.npz"
    if target.exists():
        return target.name + " (exists)"
    truth = annotations()
    rows = screen_rows()
    q = side(fmt, recipe, rows, truth)
    tabs = tables(q["ev"])
    fixed = ap(tabs, stable_key)
    if np.abs(fixed - q["point"]).max() > 1e-12:
        raise ValueError(f"own accumulate differs from COCOeval: {fixed} {q['point']}")
    image_order, uniform = tie_orders(tabs)
    stats12 = full_stats(q["ev"])
    if abs(stats12[0] - q["point"][0]) > 1e-12:
        raise ValueError("full COCOeval summary differs from the restricted accumulate")
    info = {"format": fmt, "recipe": recipe, "configuration_sha256": q["payload"]["configuration_sha256"],
            "run_record": str(q["path"].relative_to(ROOT)), "point": q["point"].tolist(), "stats12": stats12.tolist(),
            "detections": int(len(q["arrays"]["score_e5"])), "distinct_scores": int(len(np.unique(q["arrays"]["score_e5"])))}
    arrays = {"image_order": image_order, "uniform": uniform}
    if (fmt, recipe) != ("fp32", "fp32"):
        f = side("fp32", "fp32", rows, truth)
        # ground-truth boxes and areas by the resampled (1..n) annotation ids, identical for every configuration
        from public.analysis.phase3.statistics import resampled_coco
        subset, _ = resampled_coco(truth, [[]], [int(r["image_id"]) for r in rows])
        boxes = {a["id"]: a["bbox"] for a in subset["annotations"]}
        areas = {a["id"]: a["area"] for a in subset["annotations"]}
        info["agreement"] = agreement(f, q, boxes, areas)
    if spec in AREA_BOOT:
        arrays["area_draws"] = area_bootstrap(q["ev"])
    if spec in SUBSAMPLE:
        arrays.update(subsample(q["ev"]))
    if recipe == "default_head_logits":
        default = load(fmt, "default")[2]
        units, unexplained = score_units(default["score_e5"])
        levels = 64 if fmt == "int6" else 256
        info["score_units_e5"], info["score_units_unexplained"] = units, unexplained
        steps = {}
        for u in units:
            synthetic = requantised(q["arrays"], u, levels)
            ev = evaluated(synthetic, rows, truth)
            steps[str(u)] = {"point": ev.metrics().tolist(), "detections": int(len(synthetic["score_e5"])),
                             "distinct_scores": int(len(np.unique(synthetic["score_e5"])))}
            keep = q["arrays"]["score_e5"] >= u / 2.0
            o = owners(q["arrays"])
            dropped = {"counts": np.bincount(o[keep], minlength=1000).astype(np.uint16),
                       "category": q["arrays"]["category"][keep], "box_milli": q["arrays"]["box_milli"][keep],
                       "score_e5": q["arrays"]["score_e5"][keep]}
            ev2 = evaluated(dropped, rows, truth)
            steps[str(u)]["threshold_only_point"] = ev2.metrics().tolist()
            steps[str(u)]["threshold_only_detections"] = int(keep.sum())
        info["score_step"] = steps
    temporary = target.with_suffix(".partial.npz")
    np.savez_compressed(temporary, info=json.dumps(info), **arrays)
    temporary.replace(target)
    return target.name
