import pytest
from public.analysis.phase3.promotion import promote


def row(name, label="PROMISING", **extras):
    return {"model": "net", "format": name, "family": "integer", "scope": "screen", "images": 1000,
            "label": label, **extras}


def test_all_promising_and_uncertain_candidates_advance_without_top_n():
    result = promote([row("a"), row("b", "UNCERTAIN")], [("net", "a"), ("net", "b")])
    assert result["status"] == "ready_for_D4_review"
    assert len(result["promote"]) == 2 and result["hard_top_n"] is None


def test_missing_results_and_undiagnosed_failures_keep_D4_open():
    result = promote([row("a", "CATASTROPHIC/BROKEN")], [("net", "a"), ("net", "b")])
    assert result["status"] == "incomplete"
    assert len(result["promote"]) == 1 and result["configuration_removals"] == []
    assert result["missing_configurations"] == [{"model": "net", "format": "b"}]


def test_diagnosed_configuration_failure_does_not_eliminate_family():
    result = promote([row("a", "CATASTROPHIC/BROKEN", diagnosis_verified=True)], [("net", "a")])
    assert len(result["configuration_removals"]) == 1
    assert result["family_eliminations"] == [] and result["status"] == "incomplete"


def test_native_pilot_cannot_make_promotion_decision():
    with pytest.raises(ValueError, match="fixed-1k"):
        promote([row("a", scope="pilot", images=8)], [("net", "a")])
