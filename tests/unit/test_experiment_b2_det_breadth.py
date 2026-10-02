"""Lane Q6 (detector breadth): wrapper invariants, CPU only (``.venv/bin/python -m pytest``)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pytest

from tools.experiment_b.common import unseal
from tools.experiment_b2_det import engine as det_engine
from tools.experiment_b2_det import runner
from tools.experiment_b2_det.recipe import named
from tools.experiment_b2_det_breadth import core

L7_INT6_DEFAULT = "1209460b7248c402b7f432ce3fc9fed0cb74424c98cb4ab1ada07592f2ed58c5"


@lru_cache(maxsize=1)
def detector():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from tools.experiment_b_ext.detector import load_detector
    graph, constants, _ = load_detector("cpu")
    return graph, constants


def l7_calibration_identity():
    configuration = unseal(runner.BASE / "configurations" / "1834351b1f740296e04f455a7853e654fdca14c92cd9810b574c96fb7a348a7f.json")
    return configuration["calibration"]


def test_relative_tie_store_round_trips_on_stored_l7_orders():
    index = runner.load_detections(runner.detection_file(L7_INT6_DEFAULT, 1000, "index"))
    for tag in ("reverse", "random0"):
        other = runner.load_detections(runner.detection_file(L7_INT6_DEFAULT, 1000, tag))
        encoded = core.encode_relative(index, other)
        decoded = core.decode_relative(index, encoded)
        assert all(np.array_equal(decoded[k], other[k]) for k in other)
        assert len(encoded["extra_category"]) < 0.6 * len(other["category"])


def test_relative_tie_store_handles_duplicates_and_new_rows():
    reference = {"counts": np.array([3, 1], np.uint16), "category": np.array([1, 1, 2, 5], np.uint8),
                 "box_milli": np.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 9, 9, 9], [0, 0, 1, 1]], np.int32),
                 "score_e5": np.array([50, 50, 40, 7], np.int32)}
    other = {"counts": np.array([3, 2], np.uint16), "category": np.array([2, 1, 1, 5, 6], np.uint8),
             "box_milli": np.array([[9, 9, 9, 9], [1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 1, 1], [3, 3, 3, 3]], np.int32),
             "score_e5": np.array([40, 50, 50, 7, 3], np.int32)}
    encoded = core.encode_relative(reference, other)
    assert len(encoded["extra_category"]) == 1
    decoded = core.decode_relative(reference, encoded)
    assert all(np.array_equal(decoded[k], other[k]) for k in other)


def test_subset_partition_is_disjoint_and_covers_the_list():
    parts = core.subset_batches()
    assert len(parts) == 5 and all(len(p) == 50 for p in parts)
    flat = [x for p in parts for x in p]
    assert sorted(flat) == list(range(250))
    assert parts == core.subset_batches()  # seeded


def test_all_batches_reproduce_the_sealed_ranges_and_calibration_identity():
    """The union of all 250 batches equals what ``engine.v1_calibration`` returns: same per-node sample counts and
    maxima as the sealed summary (the checks v1_calibration itself applies), concatenated in the same order; and the
    calibration entry of the full set is L7's (so the configuration identity is unchanged; the GPU gate compares the
    arrays with the session's own observations as well)."""
    calibration = l7_calibration_identity()
    summary = unseal(det_engine.V1_CALIBRATION / calibration["identity"] / "summary.json")
    per_batch = core.batches(calibration["identity"])
    assert len(per_batch) == 250 and [b[0] for b in per_batch] == list(range(0, 2000, 8))
    union_samples, union_maxima = core.subset_observations(per_batch, range(250))
    assert set(union_samples) == set(summary["nodes"])
    for key, row in summary["nodes"].items():
        assert union_samples[key].size == row["sample_count"] and union_maxima[key] == row["maxabs"]
    assert core.subset_calibration(calibration, per_batch, "all", list(range(250))) == calibration
    part = core.subset_batches()[0]
    entry = core.subset_calibration(calibration, per_batch, "s0", part)
    assert entry["subset"]["images"] == 400 and entry["identity"] == calibration["identity"]
    sub_samples, sub_maxima = core.subset_observations(per_batch, part)
    assert all(sub_maxima[k] <= union_maxima[k] for k in union_maxima)
    assert all(sub_samples[k].size * 5 == union_samples[k].size for k in union_samples)


def test_block_formats_do_not_read_the_calibration_observations():
    graph, constants = detector()
    recipe = named("default")
    import torch
    for name in ("mxfp8_e4m3", "bfp6"):
        empty, _ = det_engine.prepare(graph, constants, name, recipe, {}, {}, "cpu")
        assert empty.block is not None and empty.scales == {}
    full, _ = det_engine.prepare(graph, constants, "bfp6", recipe, {"x": np.zeros(1)}, {"x": 1.0}, "cpu")
    assert all(torch.equal(full.constants[k], empty.constants[k]) for k in full.constants)


def test_q1_6_is_not_an_alias_of_int8_under_default():
    graph, _ = detector()
    recipe = named("default")
    int8 = det_engine.analyze(graph, recipe, "int8")
    q16 = det_engine.analyze(graph, recipe, "q1_6")
    unsigned8 = sorted(k for k, v in int8.items() if v["signedness"] == "unsigned")
    assert unsigned8 and not any(v["signedness"] == "unsigned" for v in q16.values())
    assert {k for k, v in int8.items() if v["quantizes"]} == {k for k, v in q16.items() if v["quantizes"]}


def test_attribution_plans():
    graph, _ = detector()
    recipe = named("default")
    summary = unseal(det_engine.V1_CALIBRATION / l7_calibration_identity()["identity"] / "summary.json")
    base = det_engine.analyze(graph, recipe, "int6")
    assert core.attribution_plan(graph, base, recipe, "int6", frozenset()) is base
    seen = set()
    for group in core.GROUP_ORDER:
        wide = core.group_nodes(graph, group)
        assert wide and not (wide & seen)
        seen |= wide
        plan = core.attribution_plan(graph, base, recipe, "int6", wide)
        assert not any(plan[n]["quantizes"] for n in wide)
        for name, row in plan.items():
            if row["quantizes"]:
                assert name in summary["nodes"] or all(f"{name}:{g}" in summary["nodes"] for g in row["groups"] or ())
            if name not in wide and base[name]["quantizes"]:
                assert row["quantizes"] and row["signedness"] == base[name]["signedness"]
        units = core.fine_units(graph, group)
        assert all(units.values()) and set().union(*units.values()) == wide
        assert sum(len(u) for u in units.values()) == len(wide)
    # SPPF wide: the upsample after it was a code pass-through and must now quantize with its own range.
    plan = core.attribution_plan(graph, base, recipe, "int6", core.group_nodes(graph, "sppf"))
    assert base["model_10"]["reason"] == "code_passthrough" and plan["model_10"]["quantizes"]
    assert core.wide_constants(graph, core.group_nodes(graph, "dfl"))  # projection constants
    names = {n["name"] for n in graph["nodes"]} | {"images"}
    assert seen <= names


def test_gate_record_if_present():
    path = core.BASE / "reproduction" / "int8-default-gate.json"
    if not path.exists():
        pytest.skip("gate not run yet")
    record = unseal(path)
    assert record["passed"] and record["all_identities_equal_l7"] and record["detections_bit_identical_to_l7_screen1k"]
    from tools.experiment_b.common import file_hash
    assert record["l7_detection_file_sha256"] == file_hash(runner.detection_file(record["l7_configuration"], 1000))


def test_part_a_records_follow_the_stop_rule():
    records = [r for r in core.runs() if r["study_part"] == "A"]
    if not records:
        pytest.skip("part A not run")
    labels = {(r["format"], r["label"]) for r in records}
    for r in records:
        if r["label"] != "default":
            continue
        arm = core.conformant_arm(r["format"])
        if r["at_chance"]:
            assert (r["format"], arm) not in labels and list(r["detections"]) == ["index"]
        else:
            assert (r["format"], arm) in labels and len(r["detections"]) == 6
            assert all(core.detection_file(r["configuration_sha256"], 1000, t).exists() for t in r["detections"])


def test_summary_files_are_write_once(tmp_path, monkeypatch):
    from tools.experiment_b2_det_breadth import summary
    monkeypatch.setattr(summary, "OUT", tmp_path)
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    summary.write("x", {"rows": []}, [{"a": 1.0, "d": {"delta": 1.0, "interval": [0.5, 1.5]}}], "")
    assert (tmp_path / "x.csv").read_text().splitlines()[0] == "a,d,d_low,d_high"
    with pytest.raises(SystemExit):
        summary.write("x", {"rows": []}, [{"a": 1.0}], "")
