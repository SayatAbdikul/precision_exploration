"""Detector postprocessing with an explicit order among equal scores, and compact detection storage.

Fixed rule (``order="index"``): candidates are ordered by (score descending, anchor index ascending, class
index ascending).  Greedy NMS, the 300-detection cut and the emitted order all follow that total order.
The other orders exist to measure how much AP depends on the choice.
"""
from __future__ import annotations

import numpy as np
import torch

CONF, IOU, MAX_DET, MAX_NMS, MAX_WH = 0.001, 0.7, 300, 30000, 7680
ORDERS = ("index", "reverse", "random")
RULE = {"version": "b2_detector_postprocess_v1", "conf": CONF, "iou": IOU, "max_det": MAX_DET, "max_nms": MAX_NMS,
        "multi_label": True, "class_offset": MAX_WH,
        "total_order": "score descending, then anchor index ascending, then class index ascending",
        "emission": "kept detections in that total order", "boxes": "v1 scale_boxes, xywh, 3 decimals; score 5 decimals"}


def tie_key(count, order, seed, image_ordinal, device):
    if order == "index":
        return None
    if order == "reverse":
        return torch.arange(count - 1, -1, -1, device=device)
    if order == "random":
        generator = torch.Generator().manual_seed(1_000_003 * (seed + 1) + image_ordinal)
        return torch.randperm(count, generator=generator).to(device)
    raise ValueError(f"unknown tie order: {order}")


def ordered_nms(output, *, order="index", seed=0, first_ordinal=0):
    """``output`` [N, 84, 8400] (xywh pixels, class scores) -> list of [n, 6] (xyxy, score, class)."""
    import torchvision
    from ultralytics.utils import ops
    results = []
    for ordinal, image in enumerate(output):
        prediction = image.transpose(0, 1)
        prediction = prediction[prediction[:, 4:].amax(1) > CONF]
        box, scores = ops.xywh2xyxy(prediction[:, :4]), prediction[:, 4:]
        i, j = torch.where(scores > CONF)
        box, score, label = box[i], scores[i, j], j
        count = len(score)
        if not count:
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        key = tie_key(count, order, seed, first_ordinal + ordinal, output.device)
        first = torch.arange(count, device=output.device) if key is None else torch.argsort(key, stable=True)
        ranked = first[torch.sort(score[first], descending=True, stable=True).indices][:MAX_NMS]
        shifted = box[ranked] + (label[ranked].float() * MAX_WH)[:, None]
        surrogate = torch.arange(len(ranked), 0, -1, device=output.device, dtype=torch.float32)
        keep = torchvision.ops.nms(shifted, surrogate, IOU)
        keep = ranked[torch.sort(keep).values[:MAX_DET]]
        results.append(torch.cat((box[keep], score[keep, None], label[keep, None].float()), 1))
    return results


def legacy_nms(output):
    """The v1 call (ultralytics non_max_suppression); it converts ``output`` in place, so a clone is passed."""
    from ultralytics.utils import ops
    return ops.non_max_suppression(output.clone(), conf_thres=CONF, iou_thres=IOU, multi_label=True, max_det=MAX_DET)


def pack(detections, shapes):
    """v1 box scaling and rounding -> per image ``(category uint8, box int32 milli-pixel, score int32 1e-5)``."""
    from ultralytics.utils import ops
    from ultralytics.data.converter import coco80_to_coco91_class
    categories = coco80_to_coco91_class()
    packed = []
    for detection, shape in zip(detections, shapes):
        detection = detection.clone()
        ops.scale_boxes((640, 640), detection[:, :4], shape)
        boxes = ops.xyxy2xywh(detection[:, :4])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes, rest = boxes.cpu().numpy(), detection[:, 4:6].cpu().numpy()
        packed.append((np.array([categories[int(c)] for c in rest[:, 1]], dtype=np.uint8),
                       np.array([[int(round(round(float(v), 3) * 1000)) for v in box] for box in boxes],
                                dtype=np.int32).reshape(-1, 4),
                       np.array([int(round(round(float(s), 5) * 100000)) for s in rest[:, 0]], dtype=np.int32)))
    return packed


def merge(packed):
    return {"counts": np.array([len(p[0]) for p in packed], dtype=np.uint16),
            "category": np.concatenate([p[0] for p in packed]) if packed else np.zeros(0, np.uint8),
            "box_milli": np.concatenate([p[1] for p in packed]) if packed else np.zeros((0, 4), np.int32),
            "score_e5": np.concatenate([p[2] for p in packed]) if packed else np.zeros(0, np.int32)}


def records(arrays, rows):
    """Compact arrays -> COCO result dictionaries in stored order (the values v1 writes to JSON)."""
    if len(arrays["counts"]) != len(rows):
        raise ValueError("detection store does not match the image list")
    owner = np.repeat(np.arange(len(rows)), arrays["counts"].astype(np.int64))
    ids = [int(row["image_id"]) for row in rows]
    return [{"image_id": ids[o], "category_id": int(c), "bbox": [int(v) / 1000 for v in b], "score": int(s) / 100000}
            for o, c, b, s in zip(owner, arrays["category"], arrays["box_milli"], arrays["score_e5"])]


def from_sealed(detections_per_image):
    """Sealed v1 per-image detection lists -> the compact arrays (exact)."""
    return merge([(np.array([d["category_id"] for d in dets], dtype=np.uint8),
                   np.array([[int(round(v * 1000)) for v in d["bbox"]] for d in dets], dtype=np.int32).reshape(-1, 4),
                   np.array([int(round(d["score"] * 100000)) for d in dets], dtype=np.int32))
                  for dets in detections_per_image])


def same(left, right):
    return all(left[k].shape == right[k].shape and np.array_equal(left[k], right[k])
               for k in ("counts", "category", "box_milli", "score_e5"))
