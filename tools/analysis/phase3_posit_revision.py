"""Verify the prepared finer-grid posit revision and selected native MAC rows."""
from math import prod

from public.analysis.phase3.fixed_mac_bounds import fixed_mac_bounds
from public.analysis.phase3.fixed_nonmac import prove_nonmac
from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend
from tools.analysis.phase3_fp64_revisions import verify_graph_revision
from tools.run.phase3_family_mac_pilot import check_layer
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    prepared = preparation_records()["mobilenet_v3_large/posit8_es1"]
    config = read(checked(prepared["configuration"]))
    if config != configuration("mobilenet_v3_large", "posit8_es1", plan) or config["runtime"]["source_sha256"] != source:
        raise ValueError("posit revision verification requires current preparation")
    resolution = read(checked(config["accumulator_resolution"]))
    evidence = read(checked(resolution["evidence"][0]))
    if evidence["schema_version"] != "phase3-posit-resolution-evidence-1.0.0" or evidence["source_sha256"] != source:
        raise ValueError("posit revision has stale diagnostic evidence")
    old_config = read(checked(evidence["configuration"]))
    if config["calibration"] != old_config["calibration"]:
        raise ValueError("posit revision changed calibration")
    before, after = read(checked(evidence["original_graph"])), read(checked(prepared["graph"]))
    changed = verify_graph_revision(before, after, resolution["replaces"], proposed_accumulator=resolution["accumulator"])
    shape_ref = evidence["shape_evidence"]
    shapes = read(checked(shape_ref))
    mac, nonmac = fixed_mac_bounds(after), prove_nonmac(after, shapes["shapes"])
    if mac["pending_nodes"] or nonmac["pending_nodes"]:
        raise ValueError("prepared posit revision does not reproduce local precision checks")
    macs = [n for n in after["nodes"] if n["op"] in {"linear", "conv2d", "depthwise_conv2d"}]
    longest = max(macs, key=lambda n: prod(after["constants"][n["inputs"][1]]["shape"][1:]))
    backend = NativeBackend("cpp")
    native = [check_layer(after, n, backend) for n in {n["name"]: n for n in (macs[0], longest)}.values()]
    if any(r["status"] != "matching_finite_selected_rows" for r in native):
        raise ValueError("new fixed accumulator differs from the native reference on selected rows")
    if source_identity() != source:
        raise ValueError("engine changed during posit revision validation")
    report = {"schema_version": "phase3-posit-revision-checks-1.0.0", "campaign_sha256": digest(plan), "source_sha256": source,
              "status": "candidate_pending_native_images_and_graph_acceptance", "configuration": prepared["configuration"],
              "graph": prepared["graph"], "resolution": config["accumulator_resolution"], "shape_evidence": shape_ref,
              "preserved_operand_graph": True, "revised_accumulator_nodes": changed,
              "local_mac_bounds": mac, "local_nonmac_checks": nonmac, "native_mac_checks": native,
              "native_library": reference(ROOT / "build/phase2/cpp/libprecision_cpp.so"),
              "native_build": reference(ROOT / "build/phase2/cpp/libprecision_cpp.json"),
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_posit_revision.py", "tools/analysis/phase3_fp64_revisions.py",
                  "tools/run/phase3_family_mac_pilot.py", "public/analysis/phase3/fixed_mac_bounds.py", "public/analysis/phase3/fixed_nonmac.py")]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"posit-revision-check-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-posit-revision.json", {**report, "immutable_evidence": reference(path)})
    print({"preserved_operand_graph": True, "revised_accumulator_nodes": changed,
           "macs": len(mac["records"]), "nonmacs": len(nonmac["records"]), "native_rows": sum(len(x["checks"]) for x in native)})


if __name__ == "__main__":
    main()
