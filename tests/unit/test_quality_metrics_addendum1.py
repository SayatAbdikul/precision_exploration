"""Lane Q7 addendum 1: near-equal subsets, power table, coverage knife edge (reads the v1 summaries, read-only)."""
import csv
import math
from pathlib import Path

import pytest

from tools.analysis.quality_metrics_v1 import addendum1 as A, resolution

V1 = Path(__file__).resolve().parents[2] / "results/summaries/quality-metrics-v1"


def _row(kind, diff, pd, width):
    return {"kind": kind, "diff_expected_pp": str(diff), "discordance": str(pd), "width_expected_median_n1000": str(width)}


def test_near_equal_selection_and_projection():
    rows = [_row("format_pair", 0.5, 0.06, 2.4), _row("format_pair", -1.5, 0.08, 2.8), _row("format_pair", 4.0, 0.3, 6.0),
            _row("recipe_pair", 0.1, 0.1, 3.0)]
    out = {(r["threshold_pp"], r["kind"]): r for r in A.near_equal(rows)}
    fp1, fp2 = out[(1.0, "format_pair")], out[(2.0, "format_pair")]
    assert fp1["contrasts"] == 1 and fp2["contrasts"] == 2 and fp2["contrasts_all"] == 3
    assert fp2["discordance_median"] == pytest.approx(0.07)
    assert fp2["halfwidth_n1000_median_pp"] == pytest.approx(1.3)
    assert fp2["projected_halfwidth_n9000_pp"] == pytest.approx(1.3 / 3)
    assert fp2["mdd_pp_n1000"] == pytest.approx(100 * 2.8016 * math.sqrt(0.07 / 1000), rel=1e-4)
    assert fp2["post_hoc"] and not fp1["post_hoc"]


def test_power_formula_matches_document_table():
    # v1 document numbers (near-equal format pairs 7.05 %, accumulator arms 1.6 %, recipe pairs 10.2 %)
    assert resolution.images_needed(0.0705, 0.01, "difference") == 5534
    assert resolution.images_needed(0.0705, 0.01, "equivalence") == 6038
    assert resolution.images_needed(0.016, 0.005, "noninferiority") == 3957
    assert resolution.images_needed(0.102, 0.01, "beyond_margin") == 8006


@pytest.mark.skipif(not (V1 / "partB-coverage.csv").exists(), reason="v1 summaries not present")
def test_knife_edge_redraw_reproduces_v1_coverage():
    rows = [r for r in csv.DictReader((V1 / "partB-coverage.csv").open()) if r["scenario"] == "pd_q5_asym_q95" and r["n"] == "128"]
    p, a = float(rows[0]["discordance"]), float(rows[0]["right_only_share"])
    share, zero = A._coverage_counts(p * (1 - a), p * a, 128, 4, (p * (2 * a - 1), 0.0))
    assert share[0, 0] == pytest.approx(float(rows[0]["percentile"]))
    assert share[0, 1] == pytest.approx(float(rows[0]["bca"]))
    assert share[1, 0] > share[0, 0]  # truth 0 is covered more often: intervals end exactly at 0


@pytest.mark.skipif(not (V1 / "partA-units.csv").exists(), reason="v1 summaries not present")
def test_agreement_correlation_selection():
    rows = list(csv.DictReader((V1 / "partA-units.csv").open()))
    c = A.agreement_correlation(rows)
    assert c["cells"] == 96 and c["pearson_r"] > 0.95
