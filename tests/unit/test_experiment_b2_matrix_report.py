"""Report tables added after review 1 of the B2 matrix (intrinsic arm against INT8, chance-level reversals)."""
import subprocess
import sys

import pytest

from tools.analysis.b2_matrix import classify
from tools.experiment_b.common import ROOT

SUMMARY = ROOT / "results/summaries/b2-matrix-v1/summary.json"


def test_classify_boundaries():
    assert classify(-1.0, -2.0, 0.5) == "within_one_point"
    assert classify(-1.02, -2.13, 0.07) == "not_resolved"
    assert classify(-1.49, -2.75, -0.23) == "separated_below"
    assert classify(1.8, -0.2, 3.7) == "not_resolved"


@pytest.mark.skipif(not SUMMARY.exists(), reason="matrix summary tables not present")
def test_report_has_intrinsic_arm_and_chance_column():
    text = subprocess.run([sys.executable, "-m", "tools.analysis.b2_matrix_report"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout
    section = text.split("<!-- vs-int8-intrinsic -->", 1)[1].split("<!--", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "max-abs activation blocks" in line]
    assert len(rows) == 8  # 4 shared formats x 2 recipes
    mxfp8 = next(line for line in rows if line.startswith("| mxfp8_e4m3") and "cum5_act_maxabs" in line)
    assert "-1.5 [-3.4, +0.3] unresolved" in mxfp8 and "-5.7 [-7.7, -3.9] BELOW" in mxfp8
    changes = text.split("<!-- ordering-changes -->", 1)[1].split("<!--", 1)[0]
    assert "both formats below 1 percent" in changes
    row = next(line for line in changes.splitlines() if line.startswith("| MBv3-L | minimal | default |"))
    assert row.endswith("| 27 | 10 |")


# Revision 3 (review 2): 4-decimal rendering and the "interval excludes 0" column of the shared-arms table.
SUMMARY_D4 = ROOT / "results/summaries/b2-matrix-v1/summary-d4.json"


def test_separated_reads_bounds_not_rounded_display():
    from tools.analysis.b2_matrix_report import separated
    assert separated("0.0049", "1.735") == "search better"
    assert separated("0.0", "1.74") == "no"
    assert separated("-2.36", "-0.47") == "search worse"


def test_rounding_digits_default_and_wide():
    from tools.analysis import b2_matrix as analysis
    assert analysis.DIGITS == 2 and analysis.r(0.85549) == 0.86 and analysis.r(0.12345, 3) == 0.123
    old = analysis.DIGITS
    try:
        analysis.DIGITS = 4
        assert analysis.r(0.85549) == 0.8555 and analysis.r(0.123456, 3) == 0.1235
        assert analysis.rounded([{"a": 0.123456, "b": "x"}]) == [{"a": 0.1235, "b": "x"}]
    finally:
        analysis.DIGITS = old


@pytest.mark.skipif(not SUMMARY_D4.exists(), reason="4-decimal matrix tables not present")
def test_report_d4_shows_fifth_separated_arm_pair():
    text = subprocess.run([sys.executable, "-m", "tools.analysis.b2_matrix_report", "--tag", "d4"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout
    arms = text.split("<!-- shared-arms -->", 1)[1].split("<!--", 1)[0]
    rows = [line for line in arms.splitlines() if line.startswith("| ") and not line.startswith("| model")]
    assert len(rows) == 24
    assert sum("| search better |" in line or "| search worse |" in line for line in rows) == 5
    mx = next(line for line in rows if line.startswith("| MBv3-L | mxfp8_e4m3 | default |"))
    assert "+0.9 [+0.0, +1.7] | search better" in mx
    intrinsic = text.split("<!-- vs-int8-intrinsic -->", 1)[1].split("<!--", 1)[0]
    mxfp8 = next(line for line in intrinsic.splitlines() if line.startswith("| mxfp8_e4m3") and "cum5_act_maxabs" in line)
    assert "-1.5 [-3.4, +0.4] unresolved" in mxfp8
