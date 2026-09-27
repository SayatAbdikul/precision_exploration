import pytest

from tools.analysis.phase3_review import apply_review
from tools.phase3.common import digest, reference, write


def fixture(root, *, extra=False, severe=False):
    plan = {"models": ["model"], "formats": ["int8"] + (["int4"] if extra else []),
            "statistics": {"catastrophic_min_loss": 0.2, "promising_max_loss": 0.01}}
    metrics = {"top1": {"delta_interval": [-0.75, -0.69] if severe else [-0.02, 0.0]}}
    analysis = {"model": "model", "format": "int8", "scope": "screen", "images": 1000,
                "configuration_sha256": "configuration", "statistics": {"metrics": metrics}}
    write(root / "analysis.json", analysis)
    ref = reference(root / "analysis.json", root)
    record = {**analysis, "family": "integer", "label": "UNCERTAIN", "analysis": ref, "metrics": metrics}
    review = {"schema_version": "phase3-d4-review-1.0.0", "campaign_sha256": digest(plan),
              "decision": "approve_D4", "reviewer": "scientific reviewer", "reviewed_at": "2026-09-22T12:00:00+00:00",
              "configurations": [{"model": "model", "format": "int8", "analysis": ref, "preservation_signals": []}],
              "reviews": {name: {"status": "completed", "reasoning": "reviewed retained evidence", "evidence": [ref]}
                          for name in ("sensitivity", "hardware_preservation", "family_preservation", "calibration")}}
    return plan, [record], review


def test_explicit_complete_review_can_close_d4(tmp_path):
    plan, records, review = fixture(tmp_path)
    result = apply_review(records, plan, review, tmp_path)
    assert result["decision"] == "D4_ACCEPTED"
    assert result["promote"][0]["reasons"] == ["uncertain"]


def test_review_cannot_turn_missing_screen_into_completion(tmp_path):
    plan, records, review = fixture(tmp_path, extra=True)
    result = apply_review(records, plan, review, tmp_path)
    assert result["decision"] == "D4_OPEN"
    assert result["missing_configurations"] == [{"model": "model", "format": "int4"}]


def test_catastrophic_config_without_diagnosis_is_preserved(tmp_path):
    plan, records, review = fixture(tmp_path, severe=True)
    result = apply_review(records, plan, review, tmp_path)
    assert result["decision"] == "D4_OPEN"
    assert result["configuration_removals"] == []
    assert result["pending_review"] == [{"model": "model", "format": "int8",
                                         "reason": "catastrophic quality loss still needs verified diagnosis; preserve"}]


def test_verified_catastrophic_diagnosis_can_remove_only_its_configuration(tmp_path):
    plan, records, review = fixture(tmp_path, severe=True)
    analysis_ref = review["configurations"][0]["analysis"]
    diagnosis = {"schema_version": "phase3-configuration-diagnosis-1.0.0",
                 "analysis": analysis_ref, "configuration_sha256": "configuration",
                 "conclusion": "catastrophic_configuration", "explanation": "verified specific cause",
                 "evidence": [analysis_ref]}
    write(tmp_path / "diagnosis.json", diagnosis)
    review["configurations"][0]["diagnosis"] = reference(tmp_path / "diagnosis.json", tmp_path)
    result = apply_review(records, plan, review, tmp_path)
    assert result["configuration_removals"][0]["format"] == "int8"
    assert result["decision"] == "D4_OPEN"  # no preserved representative of its family
    review["configurations"][0]["preservation_signals"] = ["family_representative"]
    preserved = apply_review(records, plan, review, tmp_path)
    assert preserved["decision"] == "D4_ACCEPTED"
    assert preserved["configuration_removals"] == []
    assert preserved["promote"][0]["reasons"] == ["family_representative"]


def test_review_rejects_catastrophic_diagnosis_for_nonsevere_screen(tmp_path):
    plan, records, review = fixture(tmp_path)
    review["configurations"][0]["diagnosis"] = review["configurations"][0]["analysis"]
    with pytest.raises(ValueError, match="unsupported"):
        apply_review(records, plan, review, tmp_path)


def test_review_rejects_missing_hash_evidence(tmp_path):
    plan, records, review = fixture(tmp_path)
    review["reviews"]["calibration"]["evidence"][0] = {"path": "absent.json", "sha256": "0" * 64}
    with pytest.raises(OSError):
        apply_review(records, plan, review, tmp_path)


@pytest.mark.parametrize("mutation", [lambda r: r.update(reviewer=""), lambda r: r["reviews"].pop("sensitivity"),
                                       lambda r: r.update(configurations=[]), lambda r: r.update(campaign_sha256="wrong")])
def test_review_rejects_unattributed_incomplete_or_wrong_campaign(tmp_path, mutation):
    plan, records, review = fixture(tmp_path)
    mutation(review)
    with pytest.raises(ValueError):
        apply_review(records, plan, review, tmp_path)
