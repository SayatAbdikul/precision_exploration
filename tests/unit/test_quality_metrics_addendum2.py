"""Lane Q7 addendum 2: identical contrasts, power grid, knife-edge end points, exact-cell coverage."""
import csv
from pathlib import Path

import numpy as np
import pytest

from tools.analysis.quality_metrics_v1 import addendum2 as A, resolution

V1 = Path(__file__).resolve().parents[2] / "results/summaries/quality-metrics-v1"


def _row(kind, contrast, diff, pd, width):
    r = {"kind": kind, "contrast": contrast, "diff_expected_pp": str(diff), "discordance": str(pd),
         "width_expected_median_n1000": str(width)}
    r.update({f"width_expected_median_n{n}": str(width * (1000 / n) ** 0.5) for n in resolution.NS})
    return r


def test_identical_contrasts_are_separated():
    rows = [_row("accumulator_vs_wide", "x/control", 0.0, 0.0, 0.0), _row("accumulator_vs_wide", "x/sat.w30", 0.0, 0.0, 0.0),
            _row("accumulator_vs_wide", "x/sat.w20", -0.5, 0.03, 1.6), _row("accumulator_vs_wide", "x/sat.w18", -5.0, 0.1, 4.0)]
    assert [r["label"] for r in A.identical_contrasts(rows)] == ["control", "event_free_width"]
    subs = {r["subset"]: r for r in A.subsets(rows) if r["kind"] == "accumulator_vs_wide"}
    assert subs["near_equal_2"]["contrasts"] == 3 and subs["near_equal_2"]["identical"] == 2
    assert subs["near_equal_2_differing"]["contrasts"] == 1
    assert subs["near_equal_2_differing"]["discordance_median"] == pytest.approx(0.03)
    assert subs["near_equal_2"]["discordance_median"] == 0.0


def test_count_cells_capture_full_mass_and_match_multinomial():
    cells = A.count_cells(128, 0.01, 0.015)
    assert sum(c[2] for c in cells) == pytest.approx(1.0, abs=1e-6)
    # mean of right-only counts equals n * p_right
    assert sum(c[1] * c[2] for c in cells) == pytest.approx(128 * 0.015, rel=1e-5)


def test_seed0_intervals_are_the_lane_intervals():
    assert A.intervals_seeded(256, 3, 4, 0) == resolution.intervals(256, 3, 4)
    assert A.intervals_seeded(256, 3, 4, 1)[2] == resolution.intervals(256, 3, 4)[2]  # Wald does not depend on the seed


@pytest.mark.skipif(not (V1 / "partB-coverage.csv").exists(), reason="v1 summaries not present")
def test_knife_edge_endpoint_split_reproduces_v1_and_adds_up():
    cov = list(csv.DictReader((V1 / "partB-coverage.csv").open()))
    rows = A.knife_edge_endpoints([r for r in cov if r["n"] in ("128",)])
    for r in rows:
        assert r["coverage_recomputed"] == pytest.approx(r["coverage_v1"])
        assert r["coverage_recomputed"] + r["miss_lower_end_at_zero"] + r["miss_other"] == pytest.approx(1.0)
    pct = next(r for r in rows if r["method"] == "percentile")
    assert pct["miss_lower_end_at_zero"] == pytest.approx(pct["share_lower_end_at_zero_only"] + pct["share_both_ends_at_zero"])


@pytest.mark.skipif(not (V1 / "partB-subsample.csv").exists(), reason="v1 summaries not present")
def test_review2_recounts():
    sub = list(csv.DictReader((V1 / "partB-subsample.csv").open()))
    assert len(A.identical_contrasts(sub)) == 21
    acc = A.subset(sub, "accumulator_vs_wide", "near_equal_2_differing")
    pd = float(np.median([float(r["discordance"]) for r in acc]))
    assert len(acc) == 36 and pd == pytest.approx(0.028)
    assert resolution.images_needed(pd, 0.005, "noninferiority") == 6925
    assert resolution.images_needed(pd, 0.005, "equivalence") == 9592
    assert resolution.images_needed(0.102, 0.01, "equivalence") == 8736
