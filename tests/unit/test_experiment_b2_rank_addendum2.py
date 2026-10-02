"""Lane Q2 addendum-2: tie-tolerant rank statistics and reversal switch labels (CPU)."""
import numpy as np

from tools.experiment_b2_rank import addendum2 as a2
from tools.experiment_b2_rank import analysis as an


def test_tie_tolerance_removes_summation_order_ties():
    # six formats at chance whose means differ only by floating-point summation order
    chance = np.array([0.001, 0.001 + 1e-17, 0.001 - 2e-17, 0.001, 0.001 + 3e-18, 0.001])
    a = np.concatenate([[0.7, 0.6, 0.5], chance])
    b = np.concatenate([[0.71, 0.58, 0.52], chance[::-1] + np.array([0, 2e-17, 0, -1e-17, 0, 0])])
    with a2.tie_tolerant():
        tied = float(an.tau_b(a, b))
        rho = float(an.spearman(a, b))
    assert tied == 1.0 and rho == 1.0
    # the patch is undone outside the context
    assert an.tau_b is not None and an.tau_b.__name__ == "tau_b"


def test_tie_tolerance_keeps_real_differences():
    a, b = np.array([0.5, 0.4, 0.3]), np.array([0.3, 0.4, 0.5])
    with a2.tie_tolerant():
        assert float(an.tau_b(a, b)) == -1.0
        boot = np.vstack([a, b])
        assert np.allclose(an.tau_b(boot, boot), 1.0)


def test_switch_labels():
    assert a2.switched("default", "default_no_bias_correction") == "B"
    assert a2.switched("default_no_bias_correction", "default_weight_maxabs") == "W+B"
    assert a2.switched("default_no_bias_correction", "default_unsigned") == "B+S"
    assert a2.switched("default", "default_signed") == "U"
    assert a2.switched("default_unsigned", "default_weight_maxabs") == "W+S"


def test_switch_summary_counts():
    rows = [dict(model="m", **{"class": "6"}, x="a", y="b", view_a="default", view_b="default_no_bias_correction"),
            dict(model="m", **{"class": "6"}, x="a", y="c", view_a="default_no_bias_correction",
                 view_b="default_unsigned"),
            dict(model="m", **{"class": "6"}, x="a", y="c", view_a="default", view_b="default_weight_maxabs")]
    for r in rows:
        r.update(switches=a2.switched(r["view_a"], r["view_b"]),
                 involves_no_bias_correction="default_no_bias_correction" in (r["view_a"], r["view_b"]))
    s = a2.switch_summary(rows)
    assert s["rows"] == 3 and s["rows_with_no_bias_correction"] == 2 and s["pairs"] == 2
    assert s["pairs_with_no_bias_correction_row"] == 2
    assert s["pairs_reversing_under_single_switch"] == {"B": ["m/6/a/b"], "W": ["m/6/a/c"]}
    assert s["pairs_only_under_combined_switches"] == []
