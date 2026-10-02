import json
import math

import pytest

from tools.experiment_b2 import engine, matrix, matrix_finite


def test_finite_rows_replaces_only_nonfinite_sqnr():
    rows = {"a": {"sqnr_db": -math.inf, "entropy_bits": 1.0}, "b": {"sqnr_db": 12.5}, "c": {"sqnr_db": None}}
    out = matrix_finite.finite_rows(rows)
    assert out["a"] == {"sqnr_db": None, "sqnr_db_nonfinite": "-inf", "entropy_bits": 1.0}
    assert out["b"] == {"sqnr_db": 12.5} and out["c"] == {"sqnr_db": None}
    assert rows["a"]["sqnr_db"] == -math.inf  # input untouched
    json.dumps(out, allow_nan=False)


def test_summary_of_finite_rows_is_json_compliant():
    rows = matrix_finite.finite_rows({"a": {"sqnr_db": -math.inf, "entropy_bits": 0.5, "levels_used": 2, "levels_total": 2,
                                            "fraction_at_extremes": 1.0, "fraction_zero": 0.0},
                                      "b": {"sqnr_db": 3.0, "entropy_bits": 1.0, "levels_used": 2, "levels_total": 2,
                                            "fraction_at_extremes": 1.0, "fraction_zero": 0.0}})
    summary = matrix.occupancy_summary(rows)
    assert summary["min_sqnr_db"] == 3.0 and summary["median_sqnr_db"] == 3.0
    json.dumps(summary, allow_nan=False)


def test_patch_is_scoped_and_marks_own_sources():
    before_audit, before_own = engine.audit_summary, matrix.own_sources
    with matrix_finite.patched():
        assert engine.audit_summary is not before_audit
        sources = matrix.own_sources()
        assert matrix_finite.THIS in sources and set(before_own()) < set(sources)
    assert engine.audit_summary is before_audit and matrix.own_sources is before_own


def test_main_refuses_campaigns():
    with pytest.raises(SystemExit):
        matrix_finite.main(["campaign", "matrix"])
