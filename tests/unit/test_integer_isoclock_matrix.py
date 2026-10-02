"""Pure-Python parts of tools/hardware/integer_isoclock_matrix.py (no tools run)."""
import math

import pytest

from tools.hardware import integer_isoclock_matrix as m

PATH = """Startpoint: a (rising edge-triggered flip-flop clocked by clk)
Endpoint: b (rising edge-triggered flip-flop clocked by clk)
        Cap        Slew       Delay        Time   Description
---------------------------------------------------------------------------------------
               0.000000    0.000000    0.000000 ^ a/CK (DFFQX1H7R)
   0.005317    0.067776    0.128868    0.128868 ^ a/Q (DFFQX1H7R)
               0.067776    0.000000    0.128868 ^ dut/_1_/B (NAND2X0P5H7R)
   0.047857    1.038587    0.612050    0.740917 v dut/_1_/Y (NAND2X0P5H7R)
               1.038587    0.000000    0.740917 v dut/_2_/D (DFFQX1H7R)
                                       0.740917   data arrival time
                          -0.035166    9.864834   library setup time

worst slack max 2.793602
tns max 0.000000
"""
VIOL = """max slew

Pin                                    Limit    Slew   Slack
------------------------------------------------------------
dut/_1_/Y                               0.80    1.04   -0.24 (VIOLATED)
dut/_9_/A                               0.80    1.04   -0.24 (VIOLATED)

max capacitance

Pin                                    Limit     Cap   Slack
------------------------------------------------------------
other/_3_/Y                             0.03    0.12   -0.08 (VIOLATED)
"""


def test_abc_delay_is_picoseconds_minus_overhead():
    assert m.abc_delay_ps(10) == 10000 - m.ABC_OVERHEAD_PS
    assert m.abc_delay_ps(2.5) == 2500 - m.ABC_OVERHEAD_PS
    with pytest.raises(ValueError):
        m.abc_delay_ps(0.4)


def test_min_period_is_period_minus_slack():
    assert math.isclose(m.min_period_ns(10, 2.793602), 7.206398)
    assert math.isclose(m.min_period_ns(2, -5.2), 7.2)


def test_parse_sta_report_extracts_path_and_slack():
    parsed = m.parse_sta_report(PATH)
    assert parsed["worst_slack_ns"] == 2.793602 and parsed["tns_ns"] == 0.0
    assert parsed["arrival_ns"] == 0.740917 and parsed["setup_ns"] == -0.035166
    names = [p["pin"] for p in parsed["path_pins"]]
    assert "dut/_1_/Y" in names and "dut/_1_/B" in names
    assert parsed["path_cells"] == 2  # cap column present on the Q and Y output pins only


def test_parse_sta_report_rejects_incomplete_reports():
    with pytest.raises(ValueError):
        m.parse_sta_report(PATH.replace("worst slack max 2.793602\n", ""))
    with pytest.raises(ValueError):
        m.parse_sta_report("worst slack max 1.0\ntns max 0.0\n")


def test_parse_violations_by_check():
    v = m.parse_violations(VIOL)
    assert [x["pin"] for x in v["max_slew"]] == ["dut/_1_/Y", "dut/_9_/A"]
    assert [x["pin"] for x in v["max_capacitance"]] == ["other/_3_/Y"]
    assert v["max_fanout"] == []
    assert m.parse_violations("")["max_slew"] == []


def test_validity_rule_uses_the_critical_path_only():
    pins = m.parse_sta_report(PATH)["path_pins"]
    violating = m.parse_violations(VIOL)
    assert not m.point_is_valid(pins, violating)            # dut/_1_/Y is on the path
    assert m.critical_path_violations(pins, violating)["max_slew"] == ["dut/_1_/Y"]
    off_path = {"max_slew": [{"pin": "dut/_9_/A"}], "max_capacitance": [], "max_fanout": []}
    assert m.point_is_valid(pins, off_path)                  # violators elsewhere do not invalidate
    assert not m.point_is_valid([], {"max_slew": [], "max_capacitance": [], "max_fanout": []})


def _point(core, total, slack, valid=True):
    return {"core_area": core, "total_area": total, "worst_slack_ns": slack, "valid": valid}


def test_common_targets_and_ratios():
    pts = {}
    for t in (8, 5):
        pts[((4, 32), t)] = _point(70, 80, 0.1)
        pts[((8, 32), t)] = _point(100, 100, 0.1)
        pts[((8, 64), t)] = _point(190, 190, 0.1)
    pts[((8, 64), 5)] = _point(190, 190, -0.1)               # misses timing at 5 ns
    main = [(4, 32), (8, 32), (8, 64)]
    assert m.common_targets(pts, main, [8, 5]) == [8]
    out = m.iso_clock_ratios(pts, [8, 5], main)
    assert out["all_main_common_targets"] == [8]
    mult = out["multiplier_effect_int4_over_int8_acc32"]
    assert [r["target_ns"] for r in mult] == [8, 5]          # pair-common targets are wider
    assert math.isclose(mult[0]["core_ratio"], 0.7) and math.isclose(mult[0]["total_ratio"], 0.8)
    acc = out["accumulator_effect_int8_acc64_over_acc32"]
    assert [r["target_ns"] for r in acc] == [8] and math.isclose(acc[0]["total_ratio"], 1.9)


def test_invalid_points_are_excluded_from_comparison():
    pts = {((4, 32), 8): _point(1, 1, 1.0, valid=False), ((8, 32), 8): _point(1, 1, 1.0)}
    assert m.common_targets(pts, [(4, 32), (8, 32)], [8]) == []
    with pytest.raises(ValueError):
        m.ratio(1.0, 0.0)


def test_pareto_front_drops_dominated_points():
    pts = [{"min_period_ns": 3.87, "total_area": 2767.0, "id": "4ns"},
           {"min_period_ns": 3.80, "total_area": 2758.0, "id": "3ns"},   # better on both
           {"min_period_ns": 5.44, "total_area": 2642.0, "id": "6ns"},
           {"min_period_ns": 5.67, "total_area": 2642.0, "id": "8ns"},   # same area, slower
           {"min_period_ns": 5.44, "total_area": 2642.0, "id": "dup"}]    # exact duplicate is kept
    ids = {p["id"] for p in m.pareto_front(pts)}
    assert ids == {"3ns", "6ns", "dup"}


def test_abc_load_constraint_is_in_femtofarads():
    assert m.ABC_LOAD_FF == 10      # equals the 0.01 pF STA load (1 pF = 1000 fF)
    assert m.ABC_LOAD_FF == round(m.OUTPUT_LOAD_PF * 1000)
