"""Join current local proof coverage without turning it into graph acceptance."""
from collections import Counter

from public.inference.conformance_job import source_identity
from tools.phase3.acceptance import verify_acceptance
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    reports = {}
    for name in ("integer-readiness", "fixed-mac-bounds", "fixed-nonmac", "finite-fp64-mac-bounds", "finite-fp64-nonmac", "fp64-nonmac", "shared-nonmac"):
        summary = read(ROOT / f"results/summaries/phase3-{name}.json")
        ref = summary["immutable_evidence"]
        evidence = read(checked(ref))
        if evidence["source_sha256"] != source or evidence["campaign_sha256"] != digest(plan):
            raise ValueError(f"stale local proof inventory: {name}")
        for implementation in evidence["implementations"]:
            checked(implementation)
        reports[name] = {"reference": ref, "records": {r["configuration"]: r for r in evidence["records"]}}
    records = []
    for key, prepared in preparation_records().items():
        config = read(checked(prepared["configuration"]))
        if config != configuration(prepared["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("gate inventory requires current preparation")
        graph = read(checked(prepared["graph"]))
        macs = {n["name"] for n in graph["nodes"] if n["op"] in {"linear", "conv2d", "depthwise_conv2d", "block_conv2d"}}
        others = {n["name"] for n in graph["nodes"]}-macs
        mac_covered, nonmac_covered, evidence_refs, loose, discrepancies = set(), set(), {}, [], []
        for name, report in reports.items():
            row = report["records"].get(key)
            if row is None:
                continue
            if row["graph"] != prepared["graph"] or row["configuration_artifact"] != prepared["configuration"]:
                raise ValueError(f"local proof belongs to an old graph: {key}/{name}")
            evidence_refs[name] = report["reference"]
            if name == "integer-readiness":
                mac_covered.update(r["node"] for r in row["proof"]["mac_bounds"] if r["status"] == "exact_mac_bound")
                if row["status"] == "static_proof_passed":
                    nonmac_covered.update(others)
            elif name == "fixed-mac-bounds":
                mac_covered.update(r["node"] for r in row["records"] if r["status"] == "exact_products_and_sums_after_bias_store")
            elif name == "finite-fp64-mac-bounds":
                mac_covered.update(r["node"] for r in row["macs"] if r["overflow_excluded"])
                loose.extend(r["node"] for r in row["macs"] if not r["below_half_minimum_output_spacing"])
            else:
                nonmac_covered.update(r["node"] for r in row["records"] if r["status"] in {"exact_output_codes", "no_accumulator_rounding"})
                discrepancies.extend({"node": r["node"], "evidence": name,
                                      "mismatches": r.get("mismatches", r.get("evidence", {}).get("mismatches", 0))}
                                     for r in row["records"] if r.get("mismatches") or r.get("evidence", {}).get("mismatches"))
        if not mac_covered <= macs or not nonmac_covered <= others:
            raise ValueError("proof coverage contains a node of the wrong operation class")
        accepted_path = ROOT / "artifacts/phase3/acceptance" / prepared["configuration_sha256"] / "prepared-accepted.json"
        accepted = None
        if accepted_path.exists():
            verify_acceptance(read(accepted_path))
            accepted = reference(accepted_path)
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"],
                        "accumulator": config["formats"]["accumulator"]["name"], "local_evidence": evidence_refs,
                        "mac_nodes": len(macs), "mac_nodes_with_local_finite_bounds": len(mac_covered),
                        "nonmac_nodes": len(others), "nonmac_nodes_with_local_coverage": len(nonmac_covered),
                        "pending_mac_nodes": sorted(macs-mac_covered), "pending_nonmac_nodes": sorted(others-nonmac_covered),
                        "loose_minimum_spacing_mac_bounds": loose, "retained_local_discrepancies": discrepancies,
                        "verified_screen_acceptance": accepted,
                        "status": "screen_accepted" if accepted else "pending_graph_acceptance"})
    expected = {f"{model}/{fmt}" for model in plan["models"] for fmt in plan["formats"]}
    if {r["configuration"] for r in records} != expected:
        raise ValueError("gate inventory does not cover the full frozen matrix")
    if source_identity() != source:
        raise ValueError("engine changed during gate inventory")
    report = {"schema_version": "phase3-gate-inventory-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "phase3_open", "records": records,
              "counts": {"configurations": len(records), "statuses": dict(Counter(r["status"] for r in records)),
                         "mac_nodes_with_local_finite_bounds": sum(r["mac_nodes_with_local_finite_bounds"] for r in records),
                         "mac_nodes": sum(r["mac_nodes"] for r in records),
                         "nonmac_nodes_with_local_coverage": sum(r["nonmac_nodes_with_local_coverage"] for r in records),
                         "nonmac_nodes": sum(r["nonmac_nodes"] for r in records)},
              "implementation": source_reference(__file__),
              "limits": ["coverage counts union current node identities; overlapping proofs are not added twice",
                         "FP64 MAC coverage means a finite-domain local overflow/error bound, not output-code equality",
                         "only existing verified native/accumulator acceptance authorizes a complete screen",
                         "calibration representativeness, nonfinite reachability, native pilots, quality screens and D4 remain separate gates",
                         "uncovered or discrepant configurations remain pending; no family or candidate is eliminated"]}
    write(ROOT / "results/summaries/phase3-gate-inventory.json", report)
    print(report["counts"], flush=True)


if __name__ == "__main__":
    main()
