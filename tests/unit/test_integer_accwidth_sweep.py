import json

import pytest

from tools.hardware import integer_accwidth_sweep as s


def point(target, period, slack=None, h="aaa", valid=True):
    return {"target_ns": target, "min_period_ns": period, "worst_slack_ns": target - period if slack is None else slack,
            "netlist_sha256": h, "valid": valid, "core_area": 100.0, "total_area": 150.0}


def test_period_comparison_uses_tolerance_not_exact_floats():
    assert s.same_period(3.887763, 3.887764)
    assert not s.same_period(3.887763, 3.9)


def test_one_netlist_with_periods_differing_in_the_sixth_decimal_is_one_netlist():
    # stage 2 labelled this wrongly: same netlist hash, periods 3.887763 vs 3.887764
    pts = [point(4, 3.887763, h="n1"), point(3, 3.887764, h="n1"), point(2, 3.887764, h="n1"),
           point(5, 4.8, h="n2"), point(6, 5.5, h="n3")]
    best = s.smallest_valid_period(pts)
    assert best["tied_hashes"] == ["n1"] and best["netlist_sha256"] == "n1"
    assert best["targets_ns"] == [4, 3, 2]
    assert best["first_target_ns"] == 4 and best["meets_first_target"] is True
    assert len(s.netlist_groups(pts)) == 3


def test_different_hashes_within_tolerance_are_both_reported():
    pts = [point(4, 4.0, h="x"), point(3, 4.00005, h="y")]
    best = s.smallest_valid_period(pts)
    assert best["tied_hashes"] == ["x", "y"]


def test_same_period_far_apart_hashes_are_not_merged():
    pts = [point(5, 4.5, h="a"), point(4, 4.5 + 0.01, h="b")]
    assert len(s.netlist_groups(pts)) == 2
    assert s.smallest_valid_period(pts)["tied_hashes"] == ["a"]


def test_invalid_points_never_give_the_smallest_period():
    pts = [point(4, 3.0, valid=False), point(5, 4.0)]
    assert s.smallest_valid_period(pts)["period_ns"] == 4.0
    assert s.smallest_valid_period([point(4, 3.0, valid=False)]) is None


def test_meets_requires_valid_and_nonnegative_slack():
    assert s.meets(point(5, 4.0))
    assert not s.meets(point(4, 4.5))
    assert not s.meets(point(5, 4.0, valid=False))
    assert s.meets(point(5, 5.0, slack=0.0))


def test_common_targets_and_tightest_met():
    by = {(4, 16): [point(12, 3), point(8, 3), point(4, 5)],
          (8, 64): [point(12, 9), point(8, 7.8), point(4, 9)]}
    assert s.common_met_targets(by, list(by), (12, 8, 4)) == [12, 8]
    assert s.tightest_met_target(by[(4, 16)]) == 8


def test_ols_exact_line_and_residuals():
    fit = s.ols([16, 20, 24], [100, 140, 180])
    assert fit["slope"] == pytest.approx(10) and fit["intercept"] == pytest.approx(-60)
    assert fit["max_abs_residual"] < 1e-9 and fit["r2"] == pytest.approx(1)
    noisy = s.ols([0, 1, 2, 3], [0, 1.2, 1.8, 3.0])
    assert sum(noisy["residuals"]) == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        s.ols([1, 1], [2, 3])


def test_required_signed_bits_boundaries():
    assert s.required_signed_bits(0) == 1
    assert s.required_signed_bits(127) == 8 and s.required_signed_bits(128) == 9
    assert s.required_signed_bits(2**31 - 1) == 32 and s.required_signed_bits(2**31) == 33


def test_covering_width_and_not_established():
    assert s.covering_width(26) == 28 and s.covering_width(28) == 28 and s.covering_width(29) == 32
    assert s.covering_width(62) == 64 and s.covering_width(65) is None and s.covering_width(None) is None


def test_read_requirement_checks_headroom_and_takes_worst_reduction():
    def red(node, dot, bias, acc="int32_accumulator"):
        bits = int(acc[3:5])
        return {"node": node, "dot_bound": str(dot), "maximum_stored_bias_codes": bias, "accumulator": acc,
                "remaining_headroom": str((1 << (bits - 1)) - 1 - dot - bias)}
    r = s.read_requirement({"reductions": [red("a", 100, 5), red("b", 1000, 2**20)]})
    assert r["node"] == "b" and r["required_bits"] == s.required_signed_bits(1000 + 2**20)
    bad = red("a", 100, 5)
    bad["remaining_headroom"] = "1"
    with pytest.raises(ValueError):
        s.read_requirement({"reductions": [bad]})


def test_manifest_hook_equals_shipped_manifests():
    s.check_hook_matches_shipped()
    assert s.accumulator_manifest(20)["numeric"]["integer_bits"] == 19
