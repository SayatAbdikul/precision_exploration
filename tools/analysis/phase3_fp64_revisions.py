"""Verify operand preservation and local numerical bounds of FP64 revisions."""
from public.analysis.phase3.accumulator_probes import probe_graph
from public.analysis.phase3.fp64_bounds import integer_mac_bounds
from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def verify_graph_revision(before, after, original_accumulator, *, proposed_accumulator="fp64_e11m52_accumulator"):
    for key in ("schema_version", "inputs", "constants", "outputs", "provenance"):
        if before[key] != after[key]:
            raise ValueError(f"accumulator revision altered {key}")
    if len(before["nodes"]) != len(after["nodes"]):
        raise ValueError("accumulator revision altered node count")
    changes = 0
    for old, new in zip(before["nodes"], after["nodes"]):
        expected = {**old, "attrs": dict(old["attrs"])}
        if "accumulator" in expected["attrs"]:
            if expected["attrs"]["accumulator"] != original_accumulator:
                raise ValueError("original graph accumulator policy is inconsistent")
            expected["attrs"]["accumulator"] = proposed_accumulator
            changes += 1
        if expected != new:
            raise ValueError("accumulator revision changed operands, topology or operator attributes")
    if changes == 0:
        raise ValueError("accumulator revision contains no accumulator nodes")
    # Manifest identities for operand formats must survive the revision too.
    operand_hashes = {k: v for k, v in before["manifest_hashes"].items() if k != original_accumulator}
    if operand_hashes != {k: v for k, v in after["manifest_hashes"].items() if k != proposed_accumulator}:
        raise ValueError("accumulator revision changed operand manifests")
    return changes


def main():
    plan, source = campaign(), source_identity()
    records = []
    for key, prepared in preparation_records().items():
        if "configuration" not in prepared:
            continue
        config = read(checked(prepared["configuration"]))
        resolution_ref = config.get("accumulator_resolution")
        if config["formats"]["accumulator"]["name"] != "fp64_e11m52_accumulator" or resolution_ref is None:
            continue
        if config != configuration(config["model"], prepared["format"], plan):
            raise ValueError("FP64 revision is not the active prepared definition")
        resolution = read(checked(resolution_ref))
        evidence = read(checked(resolution["evidence"][0]))
        if evidence["schema_version"] != "phase3-nonlinear-resolution-evidence-1.0.0" or evidence["source_sha256"] != source:
            raise ValueError("FP64 revision lacks current nonlinear resolution evidence")
        old_config = read(checked(evidence["configuration"]))
        if config["calibration"] != old_config["calibration"]:
            raise ValueError("FP64 revision changed calibration")
        before, after = read(checked(evidence["graph"])), read(checked(prepared["graph"]))
        changed = verify_graph_revision(before, after, old_config["formats"]["accumulator"]["name"])
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"],
                        "graph": prepared["graph"], "resolution": resolution_ref,
                        "preserved_operand_graph": True, "revised_accumulator_nodes": changed,
                        "nonlinear_probes": probe_graph(after), "local_mac_bounds": integer_mac_bounds(after)})
    if source_identity() != source:
        raise ValueError("engine changed during FP64 revision checks")
    report = {"schema_version": "phase3-fp64-revision-checks-1.0.0", "campaign_sha256": digest(plan), "source_sha256": source,
              "status": "pending_native_and_whole_graph_acceptance", "records": records,
              "implementations": [source_reference(ROOT / path) for path in (
                  "tools/analysis/phase3_fp64_revisions.py", "public/analysis/phase3/fp64_bounds.py", "public/analysis/phase3/accumulator_probes.py")]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"fp64-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-fp64-revisions.json", {**report, "immutable_evidence": reference(path)})
    print({"verified_operand_preserving_revisions": len(records),
           "macs_with_local_bounds": sum(len(r["local_mac_bounds"]["macs"]) for r in records)})


if __name__ == "__main__":
    main()
