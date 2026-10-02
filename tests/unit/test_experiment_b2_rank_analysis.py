"""Lane Q2 analysis helpers: Shapley decomposition, rank correlations, reversal criterion (CPU, no records)."""
import itertools

import numpy as np
import pytest
from scipy.stats import kendalltau, spearmanr

from tools.experiment_b2_rank import analysis


@pytest.mark.parametrize("k", [2, 3])
def test_shapley_contributions_sum_to_default_minus_minimal(k):
    rng = np.random.default_rng(k)
    credit = {corner: rng.random(50) for corner in itertools.product((0, 1), repeat=k)}
    parts = analysis.shapley(credit, tuple(f"f{i}" for i in range(k)))
    total = sum(plus - minus for plus, minus in parts.values())
    assert np.allclose(total, credit[(1,) * k] - credit[(0,) * k])


def test_shapley_additive_model_recovers_each_switch():
    effects = (0.3, -0.1, 0.05)
    credit = {c: np.full(4, sum(e * on for e, on in zip(effects, c))) for c in itertools.product((0, 1), repeat=3)}
    parts = analysis.shapley(credit, ("u", "w", "b"))
    for (plus, minus), effect in zip(parts.values(), effects):
        assert np.allclose(plus - minus, effect)


def test_tau_b_and_spearman_match_scipy_with_ties():
    rng = np.random.default_rng(7)
    for _ in range(50):
        a = rng.integers(0, 4, size=7).astype(float)
        b = rng.integers(0, 4, size=7).astype(float)
        if len(set(a)) < 2 or len(set(b)) < 2:
            continue
        assert analysis.tau_b(a, b) == pytest.approx(kendalltau(a, b).statistic)
        assert analysis.spearman(a, b) == pytest.approx(spearmanr(a, b).statistic)


def test_tau_b_vectorised_over_bootstrap_rows():
    rng = np.random.default_rng(3)
    a, b = rng.random((5, 6)), rng.random((5, 6))
    got = analysis.tau_b(a, b)
    assert got.shape == (5,)
    for row in range(5):
        assert got[row] == pytest.approx(kendalltau(a[row], b[row]).statistic)


def test_reversal_criterion_point_and_interval():
    # (d, low, high, chance) per view; delta = 1 pp
    diffs = {("8", "x", "y", "default"): (3.0, 1.5, 4.5, False),
             ("8", "x", "y", "minimal"): (-2.0, -3.0, -0.5, False),
             ("8", "x", "y", "v1_maxabs"): (-5.0, -6.0, -4.0, False),
             ("8", "x", "y", "default_signed"): (0.5, -0.5, 1.5, False)}
    rows = {(r["view_a"], r["view_b"]): r for r in analysis.reversals("m", diffs)}
    assert set(rows) == {("default", "minimal"), ("default", "v1_maxabs")}
    assert rows[("default", "minimal")]["point_reversal"] and not rows[("default", "minimal")]["interval_reversal"]
    assert rows[("default", "v1_maxabs")]["interval_reversal"]
    assert rows[("default", "v1_maxabs")]["category"] == "involves_weak"
    assert rows[("default", "minimal")]["category"] == "strong_vs_other"


def test_pair_summary_verdicts():
    def rev(x, cat, interval, chance=False):
        return {"model": "m", "class": "8", "x": x, "y": "y", "category": cat, "interval_reversal": interval,
                "chance_pair": chance}
    rows = {r["x"]: r for r in analysis.pair_summary([
        rev("a", "strong_vs_strong", True), rev("a", "involves_weak", True),
        rev("b", "involves_weak", True), rev("b", "strong_vs_strong", False),
        rev("c", "strong_vs_other", True, chance=True),
        rev("d", "strong_vs_other", True), rev("d", "involves_weak", True),
        rev("e", "involves_searched_sensitivity", False)])}
    assert rows["a"]["verdict"] == "survives_among_strong"
    assert rows["b"]["verdict"] == "maxabs_percentile_only"
    assert rows["c"]["verdict"] == "interval_only_with_chance_format"
    assert rows["d"]["verdict"] == "minimal_side"
    assert rows["e"]["verdict"] == "point_only"
    assert analysis.tier("default_searched") == "searched" and analysis.tier("default_unsigned") == "strong"
