"""Reuse verified pilot image evidence under an explicitly compatible job."""
from copy import deepcopy

from tools.phase3.common import digest


def compatible_jobs(source, target):
    if source["scope"] != "pilot" or target["scope"] != "pilot":
        raise ValueError("image reuse is limited to pilots")
    if any(source[key] != target[key] for key in source if key not in {"images", "backends"}):
        raise ValueError("pilot reuse requires identical configuration, source, pipeline and baseline")
    if set(source) != set(target) or not source["images"] <= target["images"] <= 8:
        raise ValueError("pilot reuse cannot shorten or exceed the eight-image pilot")
    if not set(source["backends"]) & set(target["backends"]):
        raise ValueError("pilot reuse requires a common backend")


def merge_record(source_record, existing, source_job, target_job, provenance):
    compatible_jobs(source_job, target_job)
    payload = {k: v for k, v in source_record.items() if k != "record_sha256"}
    if digest(payload) != source_record["record_sha256"] or payload["job_sha256"] != digest(source_job):
        raise ValueError("source pilot checkpoint identity mismatch")
    context = {k: payload[k] for k in ("graph_sha256", "sample", "paired")}
    context["job_sha256"] = digest(target_job)
    if existing is not None:
        result = {k: v for k, v in existing.items() if k != "record_sha256"}
        if digest(result) != existing["record_sha256"] or any(result.get(k) != v for k, v in context.items()):
            raise ValueError("target pilot checkpoint identity mismatch")
        if not set(result["backends"]) <= set(target_job["backends"]):
            raise ValueError("target checkpoint contains an undeclared backend")
        result = deepcopy(result)
    else:
        result = {**deepcopy(context), "backends": {}}
    added = []
    for backend in target_job["backends"]:
        if backend not in payload["backends"]:
            continue
        source_result = payload["backends"][backend]
        # Preserve a target's own timings and diagnostics if that work already
        # exists; require its numerical result to agree with the source.
        if backend in result["backends"]:
            if any(result["backends"][backend][key] != source_result[key]
                   for key in ("layers", "prediction", "output_sha256")):
                raise ValueError("existing target backend disagrees with reused evidence")
        else:
            result["backends"][backend] = deepcopy(source_result)
            added.append(backend)
    if added:
        result.setdefault("reuse_provenance", []).append({**provenance, "backends": added})
    return {**result, "record_sha256": digest(result)}, added
