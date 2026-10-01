"""Requested physical stages must determine CLI status without erasing synthesis."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from public.pdk_flow.ics55 import validate_integer as driver


def _fake_synthesis(*, config_path, work_dir, **_kwargs):
    folder = Path(work_dir)
    stat = {
        "modules": {
            "\\integer_mac_harness": {"num_wires": 4, "num_wire_bits": 4,
                "num_pub_wire_bits": 4, "num_cells": 2, "num_submodules": 1,
                "num_cells_by_type": {"DFFQX1H7R": 2, "\\integer_mac": 1}, "area": 2.0},
            "\\integer_mac": {"num_wires": 1, "num_wire_bits": 1,
                "num_pub_wire_bits": 1, "num_cells": 1, "num_submodules": 0,
                "num_cells_by_type": {"DFFQX1H7R": 1}, "area": 10.0}},
        "design": {"num_cells": 3, "num_submodules": 0,
                   "num_cells_by_type": {"DFFQX1H7R": 3}, "area": 12.0}}
    (folder / "yosys-stat.json").write_text(json.dumps(stat))
    netlist = folder / "netlist.v"
    netlist.write_text("module integer_mac_harness; endmodule\n")
    (folder / "run-manifest.json").write_text("{}\n")
    return {"outputs": {"netlist.v": {"sha256": hashlib.sha256(netlist.read_bytes()).hexdigest()}}}


@pytest.mark.parametrize("sta", ("not_requested", "completed", "failed", "unavailable"))
@pytest.mark.parametrize("place", ("not_requested", "completed", "failed", "unavailable"))
def test_requested_stage_matrix_preserves_synthesis_and_sets_exit(monkeypatch, tmp_path, sta, place):
    monkeypatch.setattr(driver, "CASES", ((4, 32),))
    monkeypatch.setattr(driver, "run", _fake_synthesis)
    real_which = shutil.which
    def fake_which(name):
        if name == "fake-sta":
            return None if sta == "unavailable" else "/fake/sta"
        if name == "fake-road":
            return None if place == "unavailable" else "/fake/openroad"
        return real_which(name)
    monkeypatch.setattr(driver.shutil, "which", fake_which)

    def fake_sta(*, output, period, **_kwargs):
        if sta == "failed":
            raise RuntimeError("fixture STA failure")
        report = output / f"opensta-{period}ns.txt"
        report.write_bytes((driver.ROOT / "tests/conformance/fixtures/hardware/opensta-smoke.txt").read_bytes())
        log = output / f"opensta-{period}ns.log"
        log.write_text("fixture STA completed\n")
        return report, log

    def fake_place(*, output, **_kwargs):
        if place == "failed":
            raise RuntimeError("fixture placement failure")
        placed = output / "placed.def"
        placed.write_text("VERSION 5.8 ;\n")
        log = output / "openroad.log"
        log.write_text("fixture placement completed\n")
        return placed, log

    monkeypatch.setattr(driver, "run_sta", fake_sta)
    monkeypatch.setattr(driver, "run_placement", fake_place)
    liberty = tmp_path / "typ.lib"
    liberty.write_text("fixture liberty\n")
    tech = tmp_path / "tech.lef"
    tech.write_text("fixture tech LEF\n")
    cells = tmp_path / "cells.lef"
    cells.write_text("fixture cells LEF\n")
    output = tmp_path / "summary.json"
    args = ["--liberty", str(liberty), "--work-dir", str(tmp_path / "runs"),
            "--summary", str(output), "--table", str(tmp_path / "table.csv")]
    if sta != "not_requested":
        args += ["--opensta", "fake-sta"]
    if place != "not_requested":
        args += ["--openroad", "fake-road", "--tech-lef", str(tech), "--cells-lef", str(cells)]
    expected_failure = sta in ("failed", "unavailable") or place in ("failed", "unavailable")
    if expected_failure:
        with pytest.raises(SystemExit, match="incomplete stages"):
            driver.main(args)
    else:
        driver.main(args)
    summary = json.loads(output.read_text())
    row = summary["records"][0]
    assert row["synthesis_status"] == "completed"
    assert row["netlist_sha256"]
    assert row["placement_status"] == place
    assert [entry["status"] for entry in row["timing"]] == [sta] * 3
    assert summary["flow_capabilities"]["post_synthesis_sta"]["status"] == sta
    assert summary["flow_capabilities"]["placement"]["status"] == place
    assert summary["flow_source_hashes"]["public/pdk_flow/ics55/validate_integer.py"] == driver.digest(Path(driver.__file__))


def test_synthesis_failure_marks_requested_downstream_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "CASES", ((4, 32),))
    def fail_run(**_kwargs):
        raise RuntimeError("fixture synthesis failure")
    monkeypatch.setattr(driver, "run", fail_run)
    liberty = tmp_path / "typ.lib"
    liberty.write_text("fixture liberty\n")
    summary = tmp_path / "summary.json"
    args = ["--liberty", str(liberty), "--work-dir", str(tmp_path / "runs"),
            "--summary", str(summary), "--table", str(tmp_path / "table.csv"),
            "--opensta", "missing-sta", "--openroad", "missing-road"]
    with pytest.raises(SystemExit, match="synthesis"):
        driver.main(args)
    row = json.loads(summary.read_text())["records"][0]
    assert row["synthesis_status"] == "failed"
    assert row["placement_status"] == "unavailable"
    assert {entry["status"] for entry in row["timing"]} == {"unavailable"}
