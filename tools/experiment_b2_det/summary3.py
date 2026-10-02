"""Versioned summaries after review 2 (protocol addendum 4); CPU, no model.

``.venv-b/bin/python -m tools.experiment_b2_det.summary3 [--tag r3]`` writes, under
``results/summaries/b2-detector-v1/``, new files beside the earlier versions (which stay as written):

- ``ties-1000-r3``: tie-order sensitivity of both recipes.  For int8, int6, fp6_e2m3 (``conformant``) and bfp6
  (``default_fp32_box_logits``) seven orders: the fixed rule, the five orders of the first study and ``imageid``
  (fixed within-image rule, images in ascending COCO image_id, as pycocotools orders them when the original ids are
  kept).  For every other sentinel format and FP32 the fixed rule and ``imageid``.  Per format, whether the
  conformant-minus-default difference is within the tie-order sensitivity (addendum 4 reading rule).
- ``conformant-1000-r3``: the r2 conformant table with a column ``rule_conformance``; the block-format rows are
  the nearest available recipe, not the rule-conformant one (their joins are not stored).

Everything is development evidence (COCO screen list).
"""
from __future__ import annotations

import argparse

import numpy as np

from . import summary, summary2
from .arms4 import conformant_arm
from .engine import BLOCK_FORMATS
from .tieorders import TIE_FORMATS, load_imageid

ORDERS = summary.TIE_TAGS + ("imageid",)
RULE_CONFORMANT = "rule-conformant: box logits not quantized, score group of the joins stored (signed codes)"
NEAREST = ("nearest available, NOT rule-conformant at the joins: box logits not quantized as the rule prescribes, "
           "but the joins are not stored although the rule keeps them quantized (the engine has no score-only "
           "join store for block formats)")


def rule_conformance(name):
    return NEAREST if name in BLOCK_FORMATS else RULE_CONFORMANT


def order_points(record, images):
    """``{order: [mAP50-95, mAP50]}`` in points for every order whose statistics exist."""
    points = {}
    for tag in summary.TIE_TAGS:
        path = summary.BASE / "bootstrap" / f"{summary.stem(record, images, tag)}.npz"
        if path.exists():
            points[tag] = [100 * float(x) for x in summary.vector(path.stem)["point"]]
    imageid = load_imageid(summary.stem(record, images))
    if imageid is not None:
        points["imageid"] = [100 * float(x) for x in imageid["point"]]
    return points


def sensitivity(points):
    values = np.array([points[tag] for tag in ORDERS if tag in points])
    fixed = np.array(points["index"])
    return {"orders_measured": [tag for tag in ORDERS if tag in points],
            "order_range_map50_95": float(values[:, 0].max() - values[:, 0].min()),
            "order_range_map50": float(values[:, 1].max() - values[:, 1].min()),
            "largest_move_from_fixed_map50_95": float(np.abs(values[:, 0] - fixed[0]).max()),
            "largest_move_from_fixed_map50": float(np.abs(values[:, 1] - fixed[1]).max())}


def tie_row(name, recipe, record, images, role):
    index = summary.vector(summary.stem(record, images))
    points = order_points(record, images)
    row = {"source": "fixed rule, new engine", "format": name, "recipe": recipe, "role": role,
           "configuration": record["configuration_sha256"][:12], **summary.envelope(index)}
    if "imageid" in points:
        row["imageid"] = {"map50_95": points["imageid"][0], "map50": points["imageid"][1]}
        row["imageid_minus_fixed"] = {"map50_95": points["imageid"][0] - points["index"][0],
                                      "map50": points["imageid"][1] - points["index"][1]}
    row["orders"] = {tag: {"map50_95": v[0], "map50": v[1]} for tag, v in points.items()}
    row.update(sensitivity(points))
    return row


def ties(images=1000):
    found = summary.records(images)
    rows, comparison = [], []
    rows.append(tie_row("fp32", "fp32", found[("fp32", "fp32")], images, "reference"))
    for name in summary.SENTINELS:
        arm = conformant_arm(name)
        pair = {}
        for recipe, role in (("default", "frozen `default`"),
                             (arm, "conformant" if arm == "conformant" else "nearest available to conformant")):
            if (name, recipe) in found:
                pair[recipe] = tie_row(name, recipe, found[(name, recipe)], images, role)
                rows.append(pair[recipe])
        if len(pair) == 2:
            left = summary.vector(summary.stem(found[(name, "default")], images))
            right = summary.vector(summary.stem(found[(name, arm)], images))
            difference = summary.difference(left, right)
            moves = max(pair["default"]["largest_move_from_fixed_map50_95"], pair[arm]["largest_move_from_fixed_map50_95"])
            entry = {"format": name, "conformant_arm": arm, "rule_conformance": rule_conformance(name),
                     "conformant_minus_default": difference,
                     "orders_measured_default": len(pair["default"]["orders_measured"]),
                     "orders_measured_conformant": len(pair[arm]["orders_measured"]),
                     "tie_sensitivity": moves,
                     "reading": ("within tie-order sensitivity" if abs(difference["delta"]) <= moves
                                 else "larger than tie-order sensitivity")}
            if "imageid" in pair["default"] and "imageid" in pair[arm]:
                entry["conformant_minus_default_imageid_order"] = (pair[arm]["imageid"]["map50_95"]
                                                                   - pair["default"]["imageid"]["map50_95"])
            per_order = [pair[arm]["orders"][t]["map50_95"] - pair["default"]["orders"][t]["map50_95"]
                         for t in ORDERS if t in pair[arm]["orders"] and t in pair["default"]["orders"]]
            entry["conformant_minus_default_over_common_orders"] = {
                "orders": len(per_order), "min": float(min(per_order)), "max": float(max(per_order))}
            comparison.append(entry)
    for v1 in ("maxabs", "percentile_99_9"):
        stem = summary.sealed_stem("int8", v1, images)
        path = summary.BASE / "bootstrap" / f"{stem}.npz"
        if path.exists():
            row = {"source": "sealed v1 predictions", "format": "int8", "recipe": f"v1_{v1}", "role": "old recipe",
                   **summary.envelope(summary.vector(stem))}
            imageid = load_imageid(stem)
            if imageid is not None:
                fixed = [100 * float(x) for x in summary.vector(stem)["point"]]
                row["imageid"] = {"map50_95": 100 * float(imageid["point"][0]), "map50": 100 * float(imageid["point"][1])}
                row["imageid_minus_fixed"] = {"map50_95": row["imageid"]["map50_95"] - fixed[0],
                                              "map50": row["imageid"]["map50"] - fixed[1]}
            rows.append(row)
    return {"panel": f"screen{images}", "interpretation": "development evidence (COCO screen list)",
            "protocol": "addendum 4 (public/experiments/configs/breadth-study/b2-detector-protocol-v1-addendum-4.json)",
            "rule": {"index": "fixed rule: score descending, then anchor index ascending, then class index ascending, "
                              "for NMS, the 300-detection cut and emission; evaluation images in ascending image sha256; "
                              "COCOeval's stable sort completes the order",
                     "reverse": "tie key reversed and evaluation image order reversed",
                     "randomN": "random tie key per image (seed N) and random evaluation image order (seed 7919+N)",
                     "imageid": "fixed within-image rule (stored detections) and evaluation images in ascending COCO "
                                "image_id: the order pycocotools uses when the original image ids are kept",
                     "envelope": "all orders of equal scores across detections at fixed image-local matching, fixed-rule "
                                 "detections: true positives first (upper) and false positives first (lower)",
                     "tie_formats": list(TIE_FORMATS),
                     "reading": "a conformant-minus-default difference is 'within tie-order sensitivity' when its absolute "
                                "value is at most the larger of the two recipes' largest moves from the fixed rule over "
                                "the orders measured (descriptive; decides nothing)"},
            "rows": rows, "recipe_comparison": comparison}


def conformant_table(images=1000):
    document = summary2.conformant_table(images)
    for row in document["rows"]:
        row["rule_conformance"] = rule_conformance(row["format"])
        row["joins_stored"] = row["format"] not in BLOCK_FORMATS
    document["recipes"]["conformant"] = (
        "the recipe the written freeze rule prescribes (addendum 3): `default` with the box logits not quantized and "
        "the joins quantized (score group, signed codes). For bfp6 and mxfp8_e4m3 the registered arm "
        "default_fp32_box_logits, which is the nearest available recipe and NOT rule-conformant at the joins (they "
        "are not stored), because the engine has no score-only join store for block formats (addendum 4)")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="r3")
    parser.add_argument("--only", nargs="*", default=None, choices=("ties-1000", "conformant-1000"))
    args = parser.parse_args()
    wanted = lambda key: args.only is None or key in args.only  # noqa: E731
    if wanted("ties-1000"):
        document = ties()
        summary.write(f"ties-1000-{args.tag}", document, document["rows"])  # recipe_comparison: JSON only
    if wanted("conformant-1000"):
        document = conformant_table()
        summary.write(f"conformant-1000-{args.tag}", document, document["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
