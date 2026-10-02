"""Experiment B2 detector: plan analysis, v1 regression on a sample, bias correction, tie order, statistics.

CPU only (``.venv/bin/python -m pytest tests/unit/test_experiment_b2_det.py -q``).  The real folded YOLOv8n
graph and two frozen screen images are used; calibration observations are taken from those two images
with v1's own observer, so the test does not depend on any artifact directory.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from tools.experiment_b.common import dataset
from tools.experiment_b2_det import post, stats
from tools.experiment_b2_det.engine import B2DetectorEngine, analyze, bias_correct, prepare
from tools.experiment_b2_det.frozen import DEFAULT
from tools.experiment_b2_det.recipe import NAMED, DetRecipe, hardware_semantics, named


@pytest.fixture(scope="module")
def world():
    from tools.experiment_b.classifier import configure
    from tools.experiment_b_ext.detector import DetectorEngine, DetectorObserver, image_batch, load_detector
    configure("cpu")
    graph, constants, _ = load_detector("cpu")
    _, rows, payload = dataset("coco_screen_1k")
    rows = rows[:2]
    inputs, shapes = image_batch(rows, payload, "cpu")
    engine, observer = DetectorEngine(graph, constants, "cpu"), DetectorObserver()
    engine.observer = observer
    with torch.inference_mode():
        fp32 = engine.run(inputs)
    samples = {k: np.concatenate(v) for k, v in observer.samples.items()}
    return {"graph": graph, "constants": constants, "rows": rows, "inputs": inputs, "shapes": shapes,
            "samples": samples, "maxima": dict(observer.maxima), "fp32": fp32}


def test_named_recipes_are_valid_and_described():
    assert named("v1_maxabs").v1_equivalent() == "maxabs"
    assert named("v1_percentile_99_9").v1_equivalent() == "percentile_99_9"
    assert DEFAULT.v1_equivalent() is None and named("default") == DEFAULT
    for recipe in NAMED.values():
        assert len(hardware_semantics(recipe)["head"]) == 7
    with pytest.raises(ValueError):
        DetRecipe(boundaries="fused_relu")


def test_plan_v1_quantizes_everything_but_flatten(world):
    plan = analyze(world["graph"], DetRecipe(), "int8")
    off = [k for k, v in plan.items() if not v["quantizes"]]
    assert sorted(off) == sorted(n["name"] for n in world["graph"]["nodes"] if n["op"] == "flatten")
    assert all(v["signedness"] == "signed" for v in plan.values() if v["quantizes"])
    assert plan["model_22"]["groups"] == ["boxes", "scores"]


def test_plan_switches(world):
    graph = world["graph"]
    plan = analyze(graph, DEFAULT, "int8")
    assert plan["model_0_conv"]["reason"] == "fused_into:model_0_act" and plan["model_0_act"]["quantizes"]
    assert plan["images"]["signedness"] == "unsigned" and plan["model_0_act"]["signedness"] == "signed"
    for name in ("model_2_split_0", "model_9_pool_0", "model_9_pool_2", "model_10"):
        assert plan[name]["reason"] == "code_passthrough"
    assert plan["model_9_cat"]["quantizes"] and plan["model_2_m_0"]["quantizes"]
    assert plan["model_22_level_0_box_logits_2"]["quantizes"] and plan["model_22_level_0_class_logits_2"]["quantizes"]
    assert plan["model_22_level_0_dfl"]["signedness"] == "unsigned"
    assert plan["model_22_level_0_scores"]["signedness"] == "unsigned"
    for name in ("model_22_level_0_boxes", "model_22_level_0_cat", "model_22"):
        assert not plan[name]["quantizes"] and plan[name]["reason"] == "head_exempt"
    concat = analyze(graph, replace(DEFAULT, passthrough="nonarith_concat"), "int8")
    assert concat["model_9_cat"]["reason"] == "concat_passthrough" and concat["model_2_m_0"]["quantizes"]
    # Non-integer and block formats have no unsigned variant; block formats take no pass-through.
    assert analyze(graph, DEFAULT, "fp8_e4m3fn")["images"]["signedness"] == "signed"
    block = analyze(graph, DEFAULT, "bfp6")
    assert block["model_9_pool_0"]["quantizes"] and block["images"]["signedness"] == "signed"
    logits = analyze(graph, named("default_head_logits"), "int8")
    assert not logits["model_22_level_0_dfl"]["quantizes"] and not logits["model_22_level_0_scores"]["quantizes"]


@pytest.mark.parametrize("name,v1", [("int8", "maxabs"), ("int8", "percentile_99_9"), ("fp6_e2m3", "maxabs"),
                                     ("bfp6", "maxabs")])
def test_all_switches_off_reproduces_v1_bit_for_bit(world, name, v1):
    from tools.experiment_b_ext.detector import prepare_detector
    recipe = named("v1_maxabs" if v1 == "maxabs" else "v1_percentile_99_9")
    args = (world["graph"], world["constants"], name)
    mine, _ = prepare(*args, recipe, world["samples"], world["maxima"], "cpu")
    theirs, _ = prepare_detector(*args, v1, world["samples"], world["maxima"], "cpu")
    with torch.inference_mode():
        left, right = mine.run(world["inputs"]), theirs.run(world["inputs"])
    assert torch.equal(left, right)
    assert not torch.equal(left, world["fp32"])


def test_bias_correction_matches_channel_means(world):
    graph, constants, inputs = world["graph"], world["constants"], world["inputs"]
    engine, _ = prepare(graph, constants, "int8", named("cum5_weight_mse"), world["samples"], world["maxima"], "cpu")
    reference = B2DetectorEngine(graph, constants, "cpu")
    before = {k: v.clone() for k, v in engine.bias.items()}
    with torch.inference_mode():
        report = bias_correct(reference, engine, inputs, chunk=1)
        convs = [n["name"] for n in graph["nodes"] if n["op"] == "conv2d"]
        want, got = {k: None for k in convs}, {k: None for k in convs}
        reference.run(inputs, capture=want)
    assert set(report) == set(convs) and max(r["max_abs_residual_after"] for r in report.values()) < 1e-3
    assert any(not torch.equal(before[k], engine.bias[k]) for k in convs)
    # The executed engine sees corrected inputs layer by layer: the first convolution's mean now matches FP32.
    with torch.inference_mode():
        raw = engine.op(graph["nodes"][0], {"images": engine.qdq("images", inputs)}.__getitem__)
    assert torch.allclose(raw.mean(dim=(0, 2, 3)), want["model_0_conv"].mean(dim=(0, 2, 3)), atol=1e-4)


def test_ordered_nms_index_order_equals_v1_and_orders_differ_only_in_ties(world):
    output = world["fp32"]
    index = post.merge(post.pack(post.ordered_nms(output), world["shapes"]))
    legacy = post.merge(post.pack(post.legacy_nms(output), world["shapes"]))
    assert post.same(index, legacy) and int(index["counts"].sum()) > 0
    coarse = output.clone()
    coarse[:, 4:] = torch.round(coarse[:, 4:] * 16) / 16
    variants = {tag: post.merge(post.pack(post.ordered_nms(coarse, order=order, seed=seed), world["shapes"]))
                for tag, order, seed in (("index", "index", 0), ("reverse", "reverse", 0), ("random", "random", 1))}
    again = post.merge(post.pack(post.ordered_nms(coarse, order="random", seed=1), world["shapes"]))
    assert post.same(variants["random"], again)
    assert not post.same(variants["index"], variants["reverse"])
    for arrays in variants.values():
        start = 0
        for count in arrays["counts"]:
            scores = arrays["score_e5"][start:start + int(count)]
            assert (np.diff(scores) <= 0).all()
            start += int(count)


def test_compact_store_round_trip(world):
    arrays = post.merge(post.pack(post.ordered_nms(world["fp32"]), world["shapes"]))
    rows = post.records(arrays, world["rows"])
    assert len(rows) == int(arrays["counts"].sum()) and arrays["category"].dtype == np.uint8
    per_image = [[r for r in rows if r["image_id"] == int(row["image_id"])] for row in world["rows"]]
    assert post.same(post.from_sealed(per_image), arrays)
    assert all(r["score"] == round(r["score"], 5) and r["bbox"] == [round(v, 3) for v in r["bbox"]] for r in rows)


def test_statistics_equal_the_project_evaluator(world):
    from public.analysis.phase3.coco_cache import paired_coco_cached
    truth = stats.annotations()
    ids = [int(r["image_id"]) for r in world["rows"]]
    coarse = world["fp32"].clone()
    coarse[:, 4:] = torch.round(coarse[:, 4:] * 16) / 16
    left = post.records(post.merge(post.pack(post.ordered_nms(world["fp32"]), world["shapes"])), world["rows"])
    right = post.records(post.merge(post.pack(post.ordered_nms(coarse), world["shapes"])), world["rows"])
    a, b = stats.Evaluated(truth, left, ids), stats.Evaluated(truth, right, ids)
    points = {"left": a.check(), "right": b.check()}
    for evaluated, point in ((a, points["left"]), (b, points["right"])):
        ties = evaluated.tie_statistics()
        assert np.allclose(ties["stable"], point, atol=1e-12)
        assert ties["false_positives_first"][0] <= ties["stable"][0] <= ties["true_positives_first"][0]
    mine = stats.paired({"point": points["left"], "draws": a.bootstrap(resamples=6, seed=11)},
                        {"point": points["right"], "draws": b.bootstrap(resamples=6, seed=11)})
    theirs = paired_coco_cached(truth, left, right, ids, resamples=6, seed=11)["metrics"]["map50_95"]
    assert abs(mine["map50_95"]["delta"] - 100 * theirs["delta"]) < 1e-9
    assert np.allclose(mine["map50_95"]["interval"], [100 * x for x in theirs["delta_interval"]], atol=1e-9)
