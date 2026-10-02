"""Experiment B2 detector: check arms, the frozen recipe pin, and the summary tables on a synthetic evidence tree.

CPU only (``.venv/bin/python -m pytest tests/unit/test_experiment_b2_det_summary.py -q``).
"""
import csv
import json

import numpy as np
import pytest

from tools.experiment_b.common import seal
from tools.experiment_b2_det import report, summary
from tools.experiment_b2_det.arms3 import ARMS3
from tools.experiment_b2_det.engine import analyze
from tools.experiment_b2_det.frozen import DEFAULT, FROZEN
from tools.experiment_b2_det.recipe import NAMED, hardware_semantics, named


@pytest.fixture(scope="module")
def graph():
    from tools.experiment_b_ext.detector import load_detector
    return load_detector("cpu")[0]


def test_frozen_recipe_is_pinned():
    assert DEFAULT.as_dict() == {
        "boundaries": "fused_silu", "passthrough": "nonarith", "unsigned": True, "activation_range": "mse",
        "weight_range": "mse", "bias_correction": "none", "q_projection": True, "q_box_logits": True, "q_dfl": True,
        "q_boxes": False, "q_class_logits": True, "q_scores": True, "q_joins": False, "quantize_input": True,
        "quantize_weights": True, "quantize_activations": True}
    semantics = hardware_semantics(DEFAULT)
    assert semantics["diagnostic"] == [] and "folded FP32 bias unchanged" in semantics["bias_constants"]
    assert sum(text.startswith("NOT quantized") for text in semantics["head"]) == 2
    # Every ablation differs from the frozen recipe; every check arm is registered under its name.
    assert all(recipe != DEFAULT for name, recipe in FROZEN.items() if name != "default")
    assert all(NAMED[name] == recipe and recipe != DEFAULT for name, recipe in ARMS3.items())
    assert named("default_head_logits") == named("cum5_fused_silu")


def test_check_arms_change_only_the_named_boundaries(graph):
    base = analyze(graph, DEFAULT, "int8")
    joins = analyze(graph, named("default_q_joins"), "int8")
    changed = {k for k in base if base[k]["quantizes"] != joins[k]["quantizes"]}
    assert changed and all(base[k]["reason"] == "head_exempt" and joins[k]["groups"] == ["scores"] for k in changed)
    assert "model_22" in changed and all(joins[k]["signedness"] == "signed" for k in changed)
    box = analyze(graph, named("default_fp32_box_logits"), "int8")
    changed = {k for k in base if base[k]["quantizes"] != box[k]["quantizes"]}
    assert changed and all(base[k]["kind"] == "conv" and not box[k]["quantizes"] for k in changed)
    assert all("box_logits" in k for k in changed)
    cls = analyze(graph, named("default_fp32_class_logits"), "int8")
    changed = {k for k in base if base[k]["quantizes"] != cls[k]["quantizes"]}
    assert changed and all("class_logits" in k and not cls[k]["quantizes"] for k in changed)


def test_reference_arms():
    assert report.reference("cum4_act_mse") == "cum3_unsigned" and report.reference("cum1_head_logits") == "v1_maxabs"
    assert report.reference("cum5_q_boxes") == "cum5_weight_mse" and report.reference("default_q_joins") == "default"
    assert report.reference("a_fused_silu") == "v1_maxabs" and report.reference("default") == "v1_maxabs"
    assert report.reference("fp32") is None and report.reference("v1_maxabs") is None


def test_flat_and_interval_helpers():
    row = summary.flat("", {"a": {"delta": 1.23456, "interval": [0.1, 0.25]}, "n": 3, "orders": {"index": {"m": 2.0}}}, {})
    assert row == {"a.delta": 1.235, "a.interval.low": 0.1, "a.interval.high": 0.25, "n": 3, "orders.index.m": 2.0}
    assert summary.excludes_zero([0.1, 0.2]) and summary.excludes_zero([-2, -1]) and not summary.excludes_zero([-1, 1])


def _vector(folder, stem, point, draws, ties=None):
    ties = ties or {"stable": list(point), "true_positives_first": [point[0] + 0.004, point[1] + 0.006],
                    "false_positives_first": [point[0] - 0.002, point[1] - 0.003], "detections_scored": 1000,
                    "detections_with_a_tied_score": 250, "detections": 1200, "distinct_scores": 77}
    np.savez_compressed(folder / f"{stem}.npz", point=np.array(point), draws=draws, ties=json.dumps(ties))


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A synthetic evidence tree: FP32, INT8 (frozen + two ablations + tie orders), one more format, sealed v1."""
    base = tmp_path / "evidence"
    (base / "runs/1000").mkdir(parents=True)
    (base / "bootstrap").mkdir()
    rng = np.random.default_rng(5)
    noise = 0.004 * rng.standard_normal((50, 2))
    points = {("fp32", "fp32"): [0.39, 0.54], ("int8", "default"): [0.3805, 0.535],
              ("int8", "default_weights_only"): [0.3905, 0.541], ("int8", "default_head_logits"): [0.3815, 0.536],
              ("int6", "default"): [0.31, 0.46]}
    for index, ((name, recipe), point) in enumerate(points.items()):
        identity = f"{index:064x}"
        seal(base / "runs/1000" / f"{name}--{recipe}--{identity[:12]}.json",
             {"configuration_sha256": identity, "format": name, "recipe_name": recipe, "images": 1000})
        _vector(base / "bootstrap", f"{identity}-1000-index", point, np.array(point) + noise + 0.0005 * index)
        if (name, recipe) == ("int8", "default"):
            for shift, tag in enumerate(summary.TIE_TAGS[1:], start=1):
                _vector(base / "bootstrap", f"{identity}-1000-{tag}", [point[0] - 0.0001 * shift, point[1]], np.zeros((0, 2)))
    for name in ("int8", "int6"):
        for recipe, point in (("maxabs", [0.31, 0.47]), ("percentile_99_9", [0.03, 0.05])):
            _vector(base / "bootstrap", f"v1--{name}--{recipe}-1000-sealed", point, np.array(point) + noise)
    _vector(base / "bootstrap", "v1--fp32--baseline-1000-sealed", [0.39, 0.54], np.array([0.39, 0.54]) + noise)
    for module in (summary, report):
        monkeypatch.setattr(module, "BASE", base)
        monkeypatch.setattr(module, "OUT", tmp_path / "out")
    monkeypatch.setattr(summary, "SENTINELS", ("int8", "int6", "log8"))
    return tmp_path


def test_exit_statement_and_residual_account(tree):
    document = summary.exit_test()
    assert document["int8_minus_fp32"]["delta"] == pytest.approx(-0.95)
    low, high = document["int8_minus_fp32"]["interval"]
    assert low <= high and low == pytest.approx(-0.9, abs=1e-6) and high == pytest.approx(-0.9, abs=1e-6)
    assert document["point_estimate_within_one_point"] and document["interval_within_one_point"]
    assert [r["arm"] for r in document["residual_account"]] == ["default_weights_only", "default_head_logits"]
    assert "default_q_boxes" in document["not_available"]
    assert document["residual_account"][0]["minus_frozen_recipe"]["delta"] == pytest.approx(1.0)
    assert document["v1"]["maxabs"]["frozen_recipe_minus_v1"]["delta"] == pytest.approx(7.05)


def test_sentinel_and_tie_tables_and_write_once(tree):
    document = summary.sentinels()
    rows = {r["format"]: r for r in document["rows"]}
    assert rows["log8"] == {"format": "log8", "missing": True}
    assert rows["int8"]["frozen_minus_int8_frozen"] is None and rows["int8"]["head_logits"]["map50_95"] == pytest.approx(38.15)
    assert rows["int6"]["frozen_minus_int8_frozen"]["delta"] == pytest.approx(-7.05) and "head_logits" not in rows["int6"]
    assert rows["int6"]["frozen_minus_v1_percentile_99_9"]["delta"] == pytest.approx(28.0)
    tie_rows = summary.ties()["rows"]
    int8 = next(r for r in tie_rows if r["format"] == "int8" and r["source"].startswith("fixed"))
    assert int8["order_range_map50_95"] == pytest.approx(0.05) and int8["largest_move_from_fixed_map50_95"] == pytest.approx(0.05)
    assert int8["envelope_width_map50_95"] == pytest.approx(0.6) and int8["share_of_scored_detections_with_a_tied_score"] == 0.25
    assert "orders" not in next(r for r in tie_rows if r["format"] == "int6" and r["source"].startswith("fixed"))
    assert sum(r["source"] == "sealed v1 predictions" for r in tie_rows) == 5
    assert summary.write("sentinels-test", document, document["rows"]) is True
    assert summary.write("sentinels-test", document, document["rows"]) is False
    with (tree / "out/sentinels-test.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 3 and table[1]["frozen_minus_fp32.interval.low"] != ""
    with pytest.raises(SystemExit):
        summary.write("sentinels-test", {**document, "panel": "other"}, document["rows"])
    effects = report.rows(1000)
    row = next(r for r in effects if r["recipe"] == "default_weights_only")
    assert row["reference"] == "default" and row["delta_reference"] == pytest.approx(1.0)


def test_freeze_tests_apply_the_two_rules(tree):
    """dev128 decision pairs: classifier-default switch dropped iff its leave-one-out arm is better with an interval
    excluding zero; contract option adopted iff it gains more than 0.5 point with an interval excluding zero."""
    for _, _, _, dev, screen, _ in summary.FREEZE:
        assert all(name in NAMED for name in dev + (screen or ()))
    base = summary.BASE
    (base / "runs/128").mkdir(parents=True)
    rng = np.random.default_rng(11)
    noise = 0.002 * rng.standard_normal((50, 2))
    points = {"fp32": [0.41, 0.56], "cum5_weight_mse": [0.392, 0.55], "cum6_bias_correction": [0.28, 0.45],
              "cum5_fused_silu": [0.399, 0.552], "cum5_concat_passthrough": [0.396, 0.551],
              "cum5_signed": [0.3918, 0.55], "default": [0.405, 0.555], "default_q_joins": [0.403, 0.554]}
    for index, (recipe, point) in enumerate(points.items()):
        identity = f"{index + 100:064x}"
        name = "fp32" if recipe == "fp32" else "int8"
        seal(base / "runs/128" / f"{name}--{recipe}--{identity[:12]}.json",
             {"configuration_sha256": identity, "format": name, "recipe_name": recipe, "images": 128})
        _vector(base / "bootstrap", f"{identity}-128-index", point, np.array(point) + noise)
    rows = {r["decision"]: r for r in summary.freeze_tests()["rows"]}
    bias = rows["bias correction"]
    assert bias["kind"] == "default" and bias["test_met_on_dev128"] is True
    assert bias["dev128"]["delta"] == pytest.approx(11.2) and bias["screen1k"] is None
    assert rows["unsigned codes"]["test_met_on_dev128"] is False          # leave-one-out worse (-0.02)
    assert rows["fused SiLU boundary"]["test_met_on_dev128"] is True       # +0.7 > 0.5, interval excludes zero
    assert rows["concatenation pass-through"]["test_met_on_dev128"] is False  # +0.4 < 0.5 although it excludes zero
    joins = rows["head: 84-channel joins not quantized"]
    assert joins["dev128"]["delta"] == pytest.approx(0.2) and joins["test_met_on_dev128"] is False
    assert rows["head: decoded boxes not quantized"]["dev128"] is None
    assert rows["head: decoded boxes not quantized"]["test_met_on_dev128"] is None
    scores = rows["head: sigmoid scores not quantized"]["screen1k"]       # from the 1000-image tree
    assert scores["base"] == "default" and scores["arm"] == "default_head_logits" and scores["delta"] == pytest.approx(0.1)
