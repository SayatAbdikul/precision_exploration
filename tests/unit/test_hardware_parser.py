from __future__ import annotations

import json
from pathlib import Path

import pytest

from public.analysis.hardware import HardwareParseError, parse_opensta_report, parse_yosys_stat


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = {
    "rtl_sha256": "1" * 64,
    "harness_version": "1.0.0",
    "pdk": "ics55",
    "library": "rvt",
    "corner": "tt_25c",
    "vt": "rvt",
    "tool": "yosys",
    "tool_version": "fixture",
    "target_clock": "10 ns",
    "pipeline_stages": 1,
}


def test_yosys_fixture_normalizes_and_has_stable_identity() -> None:
    fixture = ROOT / "tests" / "conformance" / "fixtures" / "hardware" / "yosys-stat-smoke.json"
    first = parse_yosys_stat(fixture, top="registered_smoke_arithmetic", identity=IDENTITY)
    second = parse_yosys_stat(fixture, top="registered_smoke_arithmetic", identity=IDENTITY)
    assert first == second
    assert first["metrics"]["cell_count"] == {"value": 12, "unit": "count"}
    assert first["metrics"]["design_cell_count"] == {"value": 12, "unit": "count"}
    assert first["metrics"]["register_count"] == {"value": 8, "unit": "count"}
    assert first["identity"]["pdk"] == "ics55"


def test_missing_top_and_identity_are_rejected() -> None:
    fixture = ROOT / "tests" / "conformance" / "fixtures" / "hardware" / "yosys-stat-smoke.json"
    with pytest.raises(HardwareParseError, match="top module"):
        parse_yosys_stat(fixture, top="missing", identity=IDENTITY)
    incomplete = dict(IDENTITY)
    incomplete.pop("corner")
    with pytest.raises(HardwareParseError, match="corner"):
        parse_yosys_stat(fixture, top="registered_smoke_arithmetic", identity=incomplete)


def test_malformed_metric_is_rejected() -> None:
    fixture = json.loads(
        (ROOT / "tests" / "conformance" / "fixtures" / "hardware" / "yosys-stat-smoke.json").read_text()
    )
    fixture["modules"]["\\registered_smoke_arithmetic"]["num_cells"] = "twelve"
    with pytest.raises(HardwareParseError, match="num_cells"):
        parse_yosys_stat(fixture, top="registered_smoke_arithmetic", identity=IDENTITY)


def test_opensta_fixture_normalizes_units_and_timing_status() -> None:
    fixture = ROOT / "tests" / "conformance" / "fixtures" / "hardware" / "opensta-smoke.txt"
    identity = {**IDENTITY, "tool": "OpenSTA", "tool_version": "fixture", "target_clock": "5ns", "time_unit": "ns"}
    result = parse_opensta_report(fixture, identity=identity)
    assert result["metrics"]["worst_slack"] == {"value": 2.963997, "unit": "ns"}
    assert result["metrics"]["minimum_clock_period"] == {"value": 2.04, "unit": "ns"}
    assert result["metrics"]["fmax"] == {"value": 491.16, "unit": "MHz"}
    assert result["metrics"]["timing_met"] == {"value": 1, "unit": "boolean"}


def test_opensta_missing_or_ambiguous_summary_is_rejected(tmp_path: Path) -> None:
    identity = {**IDENTITY, "tool": "OpenSTA", "tool_version": "fixture", "target_clock": "5ns", "time_unit": "ns"}
    report = tmp_path / "bad.txt"
    report.write_text("worst slack max 1.0\ntns max 0.0\n")
    with pytest.raises(HardwareParseError, match="minimum period"):
        parse_opensta_report(report, identity=identity)
    with pytest.raises(HardwareParseError, match="time_unit"):
        parse_opensta_report(
            ROOT / "tests" / "conformance" / "fixtures" / "hardware" / "opensta-smoke.txt",
            identity={**identity, "time_unit": "ps"},
        )


@pytest.mark.parametrize("area", [float("nan"), float("inf"), -1, True])
def test_nonfinite_or_negative_area_is_rejected(area):
    fixture = json.loads((ROOT / "tests/conformance/fixtures/hardware/yosys-stat-smoke.json").read_text())
    fixture["modules"]["\\registered_smoke_arithmetic"]["area"] = area
    with pytest.raises(HardwareParseError):
        parse_yosys_stat(fixture, top="registered_smoke_arithmetic", identity=IDENTITY)


@pytest.mark.parametrize("count", [-1, 1.8, True, "2"])
def test_malformed_per_cell_count_is_rejected(count):
    fixture = json.loads((ROOT / "tests/conformance/fixtures/hardware/yosys-stat-smoke.json").read_text())
    fixture["design"]["num_cells_by_type"] = {"DFF": count}
    with pytest.raises(HardwareParseError):
        parse_yosys_stat(fixture, top="registered_smoke_arithmetic", identity=IDENTITY)
