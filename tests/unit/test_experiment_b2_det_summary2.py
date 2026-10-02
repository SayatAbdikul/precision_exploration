"""Experiment B2 detector, addendum 3: the protocol-conformant recipe and the versioned (r2) summaries.

CPU only (``.venv/bin/python -m pytest tests/unit/test_experiment_b2_det_summary2.py -q``).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.experiment_b.common import ROOT, seal
from tools.experiment_b2_det import runner, summary, summary2
from tools.experiment_b2_det.arms4 import ARMS4, conformant_arm
from tools.experiment_b2_det.engine import analyze
from tools.experiment_b2_det.frozen import DEFAULT
from tools.experiment_b2_det.recipe import NAMED, named


@pytest.fixture(scope="module")
def graph():
    from tools.experiment_b_ext.detector import load_detector
    return load_detector("cpu")[0]


def test_conformant_recipe_is_pinned_and_outside_the_numeric_sources():
    recipe = named("conformant")
    assert NAMED["conformant"] == ARMS4["conformant"] == recipe
    changed = {k for k, v in recipe.as_dict().items() if DEFAULT.as_dict()[k] != v}
    assert changed == {"q_box_logits", "q_joins"} and not recipe.q_box_logits and recipe.q_joins
    assert not any("arms4" in path for path in runner.NUMERIC)
    assert conformant_arm("int8") == conformant_arm("posit8_es1") == "conformant"
    assert conformant_arm("bfp6") == conformant_arm("mxfp8_e4m3") == "default_fp32_box_logits"


def test_conformant_plan_and_block_formats(graph):
    base = analyze(graph, DEFAULT, "int8")
    plan = analyze(graph, named("conformant"), "int8")
    changed = {k for k in base if base[k]["quantizes"] != plan[k]["quantizes"]}
    box_logits = {k for k in changed if base[k]["kind"] == "conv"}
    joins = changed - box_logits
    assert box_logits and all(base[k]["quantizes"] and not plan[k]["quantizes"] for k in box_logits)
    assert len(joins) == 4 and "model_22" in joins
    assert all(plan[k]["groups"] == ["scores"] and plan[k]["signedness"] == "signed" for k in joins)
    # Block formats have no score-only join store: the substitution in arms4.conformant_arm is forced.
    with pytest.raises(ValueError):
        analyze(graph, named("conformant"), "bfp6")
    block = analyze(graph, named("default_fp32_box_logits"), "bfp6")
    assert not any(row["groups"] for row in block.values() if row["quantizes"])


def _rows(**override):
    row = {"kind": "contract", "test_met_on_dev128": True, "decision_pair_measured": "before"}
    row.update(override)
    return row


def test_rule_outcome():
    assert summary2.rule_outcome(_rows()) is True
    assert summary2.rule_outcome(_rows(decision_pair_measured="after (addendum 2)")) is False
    assert summary2.rule_outcome(_rows(test_met_on_dev128=False)) is False
    assert summary2.rule_outcome(_rows(test_met_on_dev128=None)) is False
    assert summary2.rule_outcome(_rows(kind="default")) is False          # leave-one-out better: dropped
    assert summary2.rule_outcome(_rows(kind="default", test_met_on_dev128=False)) is True


def test_freeze_deviation_on_the_stored_evidence(monkeypatch):
    stored = ROOT / "results/summaries/b2-detector-v1/freeze-tests.json"
    if not stored.exists():
        pytest.skip("stored freeze tests not available")
    monkeypatch.setattr(summary, "freeze_tests", lambda: json.loads(stored.read_text())["payload"])
    rows = {r["decision"]: r for r in summary2.freeze_tests()["rows"]}
    deviations = {name for name, row in rows.items() if row["default_deviates_from_rule"]}
    assert deviations == {"head: box logits not quantized", "head: 84-channel joins not quantized"}
    assert all(row["conformant_carries"] == row["rule_prescribes"] for row in rows.values())
    assert rows["bias correction"]["rule_prescribes"] is False and rows["fused SiLU boundary"]["rule_prescribes"]


def _vector(folder, stem, point, draws):
    ties = {"stable": list(point), "true_positives_first": list(point), "false_positives_first": list(point),
            "detections_scored": 10, "detections_with_a_tied_score": 5, "detections": 12, "distinct_scores": 9}
    np.savez_compressed(folder / f"{stem}.npz", point=np.array(point), draws=draws, ties=json.dumps(ties))


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    base = tmp_path / "evidence"
    (base / "bootstrap").mkdir(parents=True)
    rng = np.random.default_rng(3)
    noise = 0.003 * rng.standard_normal((40, 2))
    points = {1000: {("fp32", "fp32"): [0.39, 0.54], ("int8", "default"): [0.380, 0.537],
                     ("int8", "conformant"): [0.3846, 0.540], ("int6", "default"): [0.31, 0.46],
                     ("int6", "conformant"): [0.318, 0.47], ("bfp6", "default"): [0.34, 0.49],
                     ("bfp6", "default_fp32_box_logits"): [0.345, 0.495],
                     ("int8", "cum5_fused_silu"): [0.3816, 0.538], ("int8", "default_head_logits"): [0.3816, 0.538]},
              128: {("fp32", "fp32"): [0.409, 0.56], ("int8", "default"): [0.405, 0.557],
                    ("int8", "conformant"): [0.406, 0.558]}}
    for images, table in points.items():
        (base / "runs" / str(images)).mkdir(parents=True)
        for index, ((name, recipe), point) in enumerate(table.items()):
            # The two aliases share one configuration identity, as in the real evidence.
            alias = recipe in ("cum5_fused_silu", "default_head_logits")
            identity = f"{images:04d}{999 if alias else index:060x}"
            seal(base / "runs" / str(images) / f"{name}--{recipe}--{index:012x}.json",
                 {"configuration_sha256": identity, "format": name, "recipe_name": recipe, "images": images})
            _vector(base / "bootstrap", f"{identity}-{images}-index", point, np.array(point) + noise)
    for name in ("int8", "int6", "bfp6"):
        for recipe, point in (("maxabs", [0.31, 0.47]), ("percentile_99_9", [0.03, 0.05])):
            _vector(base / "bootstrap", f"v1--{name}--{recipe}-1000-sealed", point, np.array(point) + noise)
    monkeypatch.setattr(summary, "BASE", base)
    monkeypatch.setattr(summary, "OUT", tmp_path / "out")
    from tools.experiment_b2_det import report
    monkeypatch.setattr(report, "BASE", base)
    monkeypatch.setattr(summary, "SENTINELS", ("int8", "int6", "bfp6", "log8"))
    return tmp_path


def test_conformant_table(tree):
    document = summary2.conformant_table()
    rows = {r["format"]: r for r in document["rows"]}
    assert rows["log8"]["missing"] and rows["bfp6"]["conformant_arm"] == "default_fp32_box_logits"
    assert rows["int8"]["conformant_minus_frozen"]["delta"] == pytest.approx(0.46)
    assert rows["int8"]["conformant_minus_int8_conformant"] is None
    assert rows["int6"]["conformant_minus_int8_conformant"]["delta"] == pytest.approx(-6.66)
    assert rows["bfp6"]["conformant_minus_v1_maxabs"]["delta"] == pytest.approx(3.5)
    exits = document["exit_statement_int8"]
    assert exits["default"]["minus_fp32"]["delta"] == pytest.approx(-1.0)
    assert exits["conformant"]["minus_fp32"]["delta"] == pytest.approx(-0.54)
    assert exits["conformant"]["point_estimate_within_one_point"] and exits["conformant"]["interval_within_one_point"]
    assert document["dev128_int8"]["conformant_minus_default"]["delta"] == pytest.approx(0.1)


def test_effects_mark_aliases_and_reference_conformant_to_default(tree):
    groups = summary2.aliases(1000)
    assert groups == [{"format": "int8", "configuration": groups[0]["configuration"],
                       "names": ["cum5_fused_silu", "default_head_logits"]}]
    rows = {(r["format"], r["recipe"]): r for r in summary2.effects(1000)}
    assert rows[("int8", "default_head_logits")]["same_configuration_as"] == "cum5_fused_silu"
    assert rows[("int8", "default")]["same_configuration_as"] == ""
    assert rows[("int8", "conformant")]["reference"] == "default"
    assert rows[("int8", "conformant")]["delta_reference"] == pytest.approx(0.46)
    assert Path(summary.OUT).name == "out"
