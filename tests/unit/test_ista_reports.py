from pathlib import Path

import pytest

from public.analysis.hardware.ista import parse_reports


def test_ista_report_units_failures_and_power_totals(tmp_path):
    timing, power = tmp_path / "pilot.rpt", tmp_path / "pilot.pwr"
    timing.write_text("GitVersion: " + "a" * 40 + "\n"
                      "| endpoint | clk | max | 1.20r | 1.00 | 0.00 | -0.20 | 833.33 |\n"
                      "| endpoint | clk | min | 0.20r | 0.10 | 0.00 | 0.10 | NA |\n"
                      "| clk | max | -0.40 |\n| clk | min | 0.00 |\n")
    valid_power = "Net Switch Power == 1e-5 (10%)\nCell Internal Power == 8e-5 (80%)\nCell Leakage Power == 1e-5 (10%)\nTotal Power == 1e-4 W\n"
    power.write_text(valid_power)
    result = parse_reports(timing, power)
    assert result["metrics"]["setup_worst_slack"] == {"value": -0.2, "unit": "ns"}
    assert result["metrics"]["total_power"] == {"value": 0.0001, "unit": "W"}
    assert result["timing_met"] is False
    for bad in (valid_power.replace(" W", " mW"), valid_power.replace("1e-4", "2e-4"),
                valid_power + "Total Power == 1e-4 W\n"):
        power.write_text(bad)
        with pytest.raises(ValueError):
            parse_reports(timing, power)
    power.write_text(valid_power)
    timing.write_text(timing.read_text().replace("| clk | min | 0.00 |", ""))
    with pytest.raises(ValueError):
        parse_reports(timing, power)
