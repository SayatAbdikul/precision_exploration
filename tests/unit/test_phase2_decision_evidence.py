"""Prevent decision summaries from accepting incompatible or invalid timings."""
import math

import pytest

from tools.analysis.phase2_decision_evidence import strategy_candidates, workload_cost


def test_strategy_decision_rejects_different_scientific_results():
    rows = [{"format": "fp6_e3m2", "shape_role": "conv", "output_sha256": digest}
            for digest in ("a" * 64, "b" * 64)]
    with pytest.raises(ValueError, match="different numerical outputs"):
        strategy_candidates({"result": {"records": rows}})


def test_predecode_selection_accounts_for_preparation():
    common = {"format": "fp6_e3m2", "shape_role": "conv", "output_sha256": "a" * 64,
              "mac_per_second": 1, "samples_seconds": [1, 1, 1]}
    rows = [{**common, "strategy": "algorithmic", "median_seconds": 5},
            {**common, "strategy": "predecoded", "median_seconds": 3, "predecode_setup_seconds": 7}]
    record = strategy_candidates({"backend": "cpp", "result": {"records": rows}})[0]
    assert record["measured_fastest_strategy"] == "predecoded"
    assert record["predecode_setup_amortization_calls"] == 4


@pytest.mark.parametrize("seconds", [0, -1, math.inf, math.nan])
def test_budget_cannot_be_derived_from_invalid_duration(seconds):
    with pytest.raises(ValueError, match="positive and finite"):
        workload_cost([{"backends": {"cpp": {"seconds": seconds}}}], 25)


def test_budget_projects_each_backend_without_rescaling_resolution():
    records = [{"backends": {"cpp": {"seconds": 36}, "cuda": {"seconds": 18}}}]
    cpp, cuda = workload_cost(records, 25)
    assert cpp["projected_hours_one_format_1k"] == 10
    assert cuda["projected_hours_one_format_1k"] == 5
    assert cuda["projected_hours_candidate_count_1k"] == 125
