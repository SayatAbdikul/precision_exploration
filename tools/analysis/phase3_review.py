"""Verify explicit scientific review evidence before closing decision D4.

The review cannot manufacture missing screens, infer family elimination, or
convert a pilot/local bound into graph acceptance. Default reports remain open.
"""
from datetime import datetime

from public.analysis.phase3.promotion import promote
from public.analysis.phase3.statistics import screening_label
from tools.phase3.common import ROOT, checked, digest, read


def apply_review(records, plan, review, root=ROOT):
    if review.get("schema_version") != "phase3-d4-review-1.0.0" or review.get("campaign_sha256") != digest(plan):
        raise ValueError("D4 review belongs to another campaign/schema")
    if review.get("decision") != "approve_D4" or not str(review.get("reviewer", "")).strip():
        raise ValueError("D4 requires an explicit attributed review decision")
    when = datetime.fromisoformat(review["reviewed_at"])
    if when.tzinfo is None:
        raise ValueError("review timestamp needs a timezone")
    indexed = {(row["model"], row["format"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("duplicate screen evidence")
    choices = review.get("configurations", [])
    keys = [(row["model"], row["format"]) for row in choices]
    if len(set(keys)) != len(keys) or set(keys) != set(indexed):
        raise ValueError("review must cover each measured configuration exactly once")
    reviewed = []
    pending_diagnoses = []
    for row in choices:
        record = dict(indexed[(row["model"], row["format"])])
        if row["analysis"] != record["analysis"]:
            raise ValueError("review references an old/different analysis")
        analysis = read(checked(row["analysis"], root))
        if (analysis["model"] != row["model"] or analysis["format"] != row["format"] or
                analysis["scope"] != "screen" or analysis["images"] != 1000):
            raise ValueError("review requires a full matching screen")
        classification = screening_label(analysis["statistics"], plan["statistics"])
        if record["label"] != classification["label"] or record["metrics"] != analysis["statistics"]["metrics"]:
            raise ValueError("review classification/metrics differ from the measured analysis")
        record["preservation_signals"] = row.get("preservation_signals", [])
        record["diagnosis_verified"] = False
        if row.get("diagnosis") is not None:
            if not classification["diagnosis_required"]:
                raise ValueError("catastrophic diagnosis is unsupported by the measured interval")
            diagnosis = read(checked(row["diagnosis"], root))
            if (diagnosis.get("schema_version") != "phase3-configuration-diagnosis-1.0.0" or
                    diagnosis.get("analysis") != row["analysis"] or
                    diagnosis.get("configuration_sha256") != analysis["configuration_sha256"] or
                    diagnosis.get("conclusion") != "catastrophic_configuration" or
                    not str(diagnosis.get("explanation", "")).strip() or not diagnosis.get("evidence")):
                raise ValueError("diagnosis is missing specific configuration/evidence/reasoning")
            for evidence in diagnosis["evidence"]:
                checked(evidence, root)
            record["diagnosis_verified"] = True
        classification = screening_label(analysis["statistics"], plan["statistics"],
                                         diagnosis_verified=record["diagnosis_verified"])
        record["label"] = classification["label"]
        if classification["diagnosis_required"]:
            pending_diagnoses.append({"model": row["model"], "format": row["format"],
                                      "reason": "catastrophic quality loss still needs verified diagnosis; preserve"})
        reviewed.append(record)
    required = {"sensitivity", "hardware_preservation", "family_preservation", "calibration"}
    reviews = review.get("reviews", {})
    if set(reviews) != required:
        raise ValueError("D4 requires sensitivity, hardware, family and calibration reviews")
    for name, item in reviews.items():
        if item.get("status") != "completed" or not str(item.get("reasoning", "")).strip() or not item.get("evidence"):
            raise ValueError(f"incomplete {name} review")
        for evidence in item["evidence"]:
            checked(evidence, root)
    expected = [(m, f) for m in plan["models"] for f in plan["formats"]]
    result = promote(reviewed, expected)
    if pending_diagnoses:
        result["pending_review"].extend(pending_diagnoses)
        result["status"] = "incomplete"
    result["decision"] = "D4_ACCEPTED" if result["status"] == "ready_for_D4_review" else "D4_OPEN"
    if result["decision"] == "D4_ACCEPTED":
        result["status"] = "complete"
    result["reviewer"], result["reviewed_at"] = review["reviewer"], review["reviewed_at"]
    for row in result["promote"]:
        record = indexed[(row["model"], row["format"])]
        row.update(analysis=record["analysis"], metrics=record["metrics"])
    return result
