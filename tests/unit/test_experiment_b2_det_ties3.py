"""Experiment B2 detector, addendum 4: tie orders of the conformant recipe, the `imageid` evaluation order and the
r3 summaries.

CPU only (``.venv/bin/python -m pytest tests/unit/test_experiment_b2_det_ties3.py -q``).
"""
import contextlib
import io
import json

import numpy as np
import pytest

from tools.experiment_b.common import ROOT, dataset, seal
from tools.experiment_b2_det import runner, summary, summary3, tieorders


def test_new_modules_are_not_numeric_sources():
    assert not any(name in path for path in runner.NUMERIC for name in ("tieorders", "summary3"))
    assert summary3.ORDERS == ("index", "reverse", "random0", "random1", "random2", "random3", "imageid")
    assert tuple(runner.order_tag(*o) for o in runner.TIE_ORDERS) == summary.TIE_TAGS[1:]


def test_same_arrays():
    a = {"counts": np.array([1, 2], np.uint16), "score_e5": np.array([3, 4, 5], np.int32)}
    assert tieorders.same_arrays(a, {k: v.copy() for k, v in a.items()})
    assert not tieorders.same_arrays(a, {**a, "score_e5": np.array([3, 5, 4], np.int32)})
    assert not tieorders.same_arrays(a, {"counts": a["counts"]})


def test_imageid_order_equals_plain_pycocotools():
    """The `imageid` evaluation (lane evaluator, images in ascending image_id) equals pycocotools with the original
    image ids on stored INT6 detections of the first 40 screen images."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from tools.experiment_b2_det.post import records
    from tools.experiment_b2_det.stats import Evaluated, annotations
    stored = sorted((runner.BASE / "runs" / "1000").glob("int6--default--*.json"))
    if not stored:
        pytest.skip("stored INT6 detections not available")
    identity = json.loads(stored[0].read_text())["payload"]["configuration_sha256"]
    arrays = runner.load_detections(runner.detection_file(identity, 1000))
    n = 40
    keep = int(arrays["counts"][:n].astype(np.int64).sum())
    subset = {"counts": arrays["counts"][:n], "category": arrays["category"][:keep],
              "box_milli": arrays["box_milli"][:keep], "score_e5": arrays["score_e5"][:keep]}
    _, rows, _ = dataset("coco_screen_1k")
    rows = rows[:n]
    truth, detections = annotations(), records(subset, rows)
    ids = [int(r["image_id"]) for r in rows]
    ours = Evaluated(truth, detections, sorted(ids)).check()
    with contextlib.redirect_stdout(io.StringIO()):
        ground = COCO()
        ground.dataset = {"images": [im for im in truth["images"] if im["id"] in set(ids)],
                          "categories": truth["categories"],
                          "annotations": [a for a in truth["annotations"] if a["image_id"] in set(ids)]}
        ground.createIndex()
        evaluation = COCOeval(ground, ground.loadRes(detections), "bbox")
        evaluation.params.imgIds = ids
        evaluation.evaluate(), evaluation.accumulate(), evaluation.summarize()
    assert ours[0] == pytest.approx(evaluation.stats[0], abs=1e-12)
    assert ours[1] == pytest.approx(evaluation.stats[1], abs=1e-12)


def _vector(folder, stem, point, draws=None, ties=True):
    data = {"point": np.array(point)}
    if draws is not None:
        data["draws"] = draws
    if ties:
        data["ties"] = json.dumps({"stable": list(point), "true_positives_first": [p + 0.01 for p in point],
                                   "false_positives_first": [p - 0.01 for p in point], "detections_scored": 10,
                                   "detections_with_a_tied_score": 5, "detections": 12, "distinct_scores": 9})
    np.savez_compressed(folder / f"{stem}.npz", **data)


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    base = tmp_path / "evidence"
    (base / "bootstrap").mkdir(parents=True)
    (base / "imageorder").mkdir()
    rng = np.random.default_rng(5)
    noise = 0.003 * rng.standard_normal((40, 2))
    table = {("fp32", "fp32"): [0.39, 0.54], ("int8", "default"): [0.380, 0.537],
             ("int8", "conformant"): [0.3835, 0.539], ("int6", "default"): [0.3094, 0.46],
             ("int6", "conformant"): [0.3105, 0.45], ("bfp6", "default"): [0.34, 0.49],
             ("bfp6", "default_fp32_box_logits"): [0.3422, 0.495]}
    # Tie orders (moves in points): int8 both recipes <= 0.12; int6 default 0.19 under imageid only.
    moves = {("int8", "default"): [0.0, 0.1, -0.12, 0.05, 0.0, 0.02, 0.03],
             ("int8", "conformant"): [0.0, 0.08, 0.0, -0.05, 0.01, 0.0, -0.02],
             ("int6", "default"): [0.0, 0.11, -0.05, 0.02, 0.0, 0.03, 0.19],
             ("int6", "conformant"): [0.0, 0.02, 0.01, 0.0, -0.04, 0.05, 0.19]}
    (base / "runs" / "1000").mkdir(parents=True)
    for index, ((name, recipe), point) in enumerate(table.items()):
        identity = f"{index:064x}"
        seal(base / "runs" / "1000" / f"{name}--{recipe}--{index:012x}.json",
             {"configuration_sha256": identity, "format": name, "recipe_name": recipe, "images": 1000})
        _vector(base / "bootstrap", f"{identity}-1000-index", point, np.array(point) + noise)
        shift = moves.get((name, recipe), [0.0] * 7)
        if (name, recipe) in moves:
            for tag, move in zip(summary.TIE_TAGS[1:], shift[1:6]):
                _vector(base / "bootstrap", f"{identity}-1000-{tag}", [point[0] + move / 100, point[1]], np.zeros((0, 2)))
        _vector(base / "imageorder", f"{identity}-1000-index-imageid", [point[0] + shift[6] / 100, point[1]])
    for name in ("int8", "int6", "bfp6"):
        for recipe, point in (("maxabs", [0.31, 0.47]), ("percentile_99_9", [0.03, 0.05])):
            _vector(base / "bootstrap", f"v1--{name}--{recipe}-1000-sealed", point, np.array(point) + noise)
    _vector(base / "imageorder", "v1--int8--percentile_99_9-1000-sealed-imageid", [0.05, 0.07])
    monkeypatch.setattr(summary, "BASE", base)
    monkeypatch.setattr(summary, "OUT", tmp_path / "out")
    monkeypatch.setattr(tieorders, "ORDER_DIR", base / "imageorder")
    monkeypatch.setattr(summary, "SENTINELS", ("int8", "int6", "bfp6", "log8"))
    return tmp_path


def test_ties_r3(tree):
    document = summary3.ties()
    rows = {(r["format"], r["recipe"]): r for r in document["rows"]}
    assert rows[("int8", "default")]["orders_measured"] == list(summary3.ORDERS)
    assert rows[("int8", "default")]["largest_move_from_fixed_map50_95"] == pytest.approx(0.12)
    assert rows[("int8", "default")]["order_range_map50_95"] == pytest.approx(0.22)
    assert rows[("int6", "default")]["largest_move_from_fixed_map50_95"] == pytest.approx(0.19)
    assert rows[("int6", "default")]["imageid_minus_fixed"]["map50_95"] == pytest.approx(0.19)
    # Formats outside the tie study: the fixed rule and imageid only.
    assert rows[("bfp6", "default")]["orders_measured"] == ["index", "imageid"]
    assert rows[("bfp6", "default_fp32_box_logits")]["role"] == "nearest available to conformant"
    assert rows[("int8", "v1_percentile_99_9")]["imageid_minus_fixed"]["map50_95"] == pytest.approx(2.0)
    assert "imageid" not in rows[("int8", "v1_maxabs")]
    comparison = {r["format"]: r for r in document["recipe_comparison"]}
    assert set(comparison) == {"int8", "int6", "bfp6"}
    assert comparison["int8"]["conformant_minus_default"]["delta"] == pytest.approx(0.35)
    assert comparison["int8"]["reading"] == "larger than tie-order sensitivity"
    assert comparison["int6"]["tie_sensitivity"] == pytest.approx(0.19)
    assert comparison["int6"]["reading"] == "within tie-order sensitivity"  # 0.11 <= 0.19
    assert comparison["int6"]["conformant_minus_default_imageid_order"] == pytest.approx(0.11)
    assert comparison["int6"]["conformant_minus_default_over_common_orders"]["orders"] == 7
    assert comparison["bfp6"]["rule_conformance"] == summary3.NEAREST
    assert comparison["bfp6"]["orders_measured_default"] == 2


def test_conformant_r3_labels_block_rows(tree):
    document = summary3.conformant_table()
    rows = {r["format"]: r for r in document["rows"]}
    assert rows["int8"]["rule_conformance"] == summary3.RULE_CONFORMANT and rows["int8"]["joins_stored"]
    assert rows["bfp6"]["rule_conformance"] == summary3.NEAREST and not rows["bfp6"]["joins_stored"]
    assert "NOT rule-conformant" in document["recipes"]["conformant"]


def test_r3_files_are_write_once(tree):
    assert summary3.main.__module__ == "tools.experiment_b2_det.summary3"
    document = summary3.ties()
    assert summary.write("ties-1000-r3", document, document["rows"]) is True
    assert summary.write("ties-1000-r3", document, document["rows"]) is False
    with pytest.raises(SystemExit):
        summary.write("ties-1000-r3", {**document, "panel": "other"}, document["rows"])
    assert (tree / "out" / "ties-1000-r3.csv").exists() and ROOT.exists()
