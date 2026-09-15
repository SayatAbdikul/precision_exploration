from copy import deepcopy
import pytest

from tools.phase3.common import digest
from tools.phase3.pilot_reuse import compatible_jobs, merge_record


def fixture():
    job = {"scope": "pilot", "images": 1, "backends": ["cpp"], "prepared": "same",
           "baseline": "same", "source_sha256": "same", "pipeline_sha256": "same"}
    payload = {"job_sha256": digest(job), "graph_sha256": "graph", "sample": {"sha256": "image"}, "paired": {"truth": 2},
               "backends": {"cpp": {"layers": {"node": "trace"}, "prediction": [2], "output_sha256": "output", "seconds": 19}}}
    return job, {**payload, "record_sha256": digest(payload)}


def test_extension_preserves_actual_evidence_and_records_origin_idempotently():
    job, source = fixture()
    target = {**job, "images": 8, "backends": ["cpp", "cuda"]}
    merged, added = merge_record(source, None, job, target, {"source_image": "original"})
    assert added == ["cpp"] and merged["backends"] == source["backends"]
    assert merged["job_sha256"] == digest(target) != source["job_sha256"]
    assert merged["reuse_provenance"] == [{"source_image": "original", "backends": ["cpp"]}]
    again, added = merge_record(source, merged, job, target, {"source_image": "original"})
    assert not added and again == merged


@pytest.mark.parametrize("field", ["prepared", "baseline", "source_sha256", "pipeline_sha256", "scope"])
def test_reuse_cannot_cross_numerical_configuration_or_evaluation_identity(field):
    job, _ = fixture()
    with pytest.raises(ValueError):
        compatible_jobs(job, {**job, "images": 8, field: "different"})


def test_existing_target_conflict_and_corrupt_source_are_rejected():
    job, source = fixture(); target = {**job, "images": 8}
    merged, _ = merge_record(source, None, job, target, {})
    payload = {k: v for k, v in deepcopy(merged).items() if k != "record_sha256"}
    payload["backends"]["cpp"]["prediction"] = [3]
    with pytest.raises(ValueError, match="disagrees"):
        merge_record(source, {**payload, "record_sha256": digest(payload)}, job, target, {})
    source["paired"]["truth"] = 3
    with pytest.raises(ValueError, match="identity mismatch"):
        merge_record(source, None, job, target, {})
