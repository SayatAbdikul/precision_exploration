"""Version a finer 64-bit fixed grid for a reproduced posit8 hard-swish failure."""
from copy import deepcopy

from public.analysis.phase3.fixed_mac_bounds import fixed_mac_bounds
from public.analysis.phase3.fixed_nonmac import prove_nonmac
from public.inference.conformance_job import source_identity
from public.inference.tensor import Tensor
from public.quantization.graph.executable import freeze_graph
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference

POLICY = "posit8_es1_quire64_f28_accumulator"
KEY = "mobilenet_v3_large/posit8_es1"


def main():
    plan, source = campaign(), source_identity()
    directory = ROOT / "public/experiments/configs/experiment_a"
    index_path = directory / "phase3-accumulator-resolutions-v1.json"
    before = read(index_path)
    previous = before["resolutions"].get(KEY)
    if previous is not None:
        prior = read(checked(previous))
        if prior["accumulator"] == POLICY:
            for item in prior["evidence"]:
                checked(item)
            print({"status": "already_versioned", "resolution": previous})
            return
        raise ValueError("another posit8 resolution needs explicit evidence review")
    prepared = preparation_records()[KEY]
    config = read(checked(prepared["configuration"]))
    if config != configuration("mobilenet_v3_large", "posit8_es1", plan) or config["runtime"]["source_sha256"] != source:
        raise ValueError("posit resolution requires current preparation")
    graph = read(checked(prepared["graph"]))
    shape_ref = read(ROOT / "results/summaries/phase3-shapes.json")["models"][config["model"]]
    shapes = read(checked(shape_ref))
    if shapes["campaign_sha256"] != digest(plan) or shapes["model_manifest"] != plan["inputs"][config["model"]]:
        raise ValueError("shape observation identity mismatch")
    original = prove_nonmac(graph, shapes["shapes"])
    failures = [r for r in original["records"] if r.get("evidence", {}).get("mismatches", 0)]
    if not failures:
        raise ValueError("no reproduced finite-code precision failure")
    nodes = deepcopy(graph["nodes"])
    for node in nodes:
        if "accumulator" in node["attrs"]:
            node["attrs"]["accumulator"] = POLICY
    proposed = freeze_graph(inputs=graph["inputs"], constants={k: Tensor.from_document(v) for k,v in graph["constants"].items()},
                            nodes=nodes, outputs=graph["outputs"],
                            provenance={"kind": "diagnostic_accumulator_revision", "original_graph": prepared["graph"], "policy": POLICY})
    mac, nonmac = fixed_mac_bounds(proposed), prove_nonmac(proposed, shapes["shapes"])
    if mac["pending_nodes"] or nonmac["pending_nodes"]:
        raise ValueError("finer fixed policy does not pass all local MAC/non-MAC checks")
    proposal_path = ROOT / "artifacts/phase3/accumulator-probes" / f"posit-proposal-{digest(proposed)}.json"
    write(proposal_path, proposed)
    evidence = {"schema_version": "phase3-posit-resolution-evidence-1.0.0", "source_sha256": source,
                "campaign_sha256": digest(plan), "configuration": prepared["configuration"], "original_graph": prepared["graph"],
                "proposed_graph": reference(proposal_path), "shape_evidence": shape_ref,
                "original_nonmac": original, "proposed_mac": mac, "proposed_nonmac": nonmac,
                "scope": "same operand encodings/constants; finer fixed grid throughout graph; local checks only, native acceptance pending",
                "implementations": [source_reference(ROOT / p) for p in (
                    "tools/run/phase3_resolve_posit.py", "public/analysis/phase3/fixed_mac_bounds.py", "public/analysis/phase3/fixed_nonmac.py")]}
    evidence_path = ROOT / "artifacts/phase3/accumulator-probes" / f"posit-resolution-{digest(evidence)}.json"
    write(evidence_path, evidence)
    resolution = {"schema_version": "phase3-accumulator-resolution-1.0.0", "campaign_sha256": digest(plan),
                  "model": config["model"], "format": "posit8_es1", "replaces": config["formats"]["accumulator"]["name"],
                  "accumulator": POLICY, "status": "candidate_pending_native_acceptance",
                  "reason": "24 fractional bits erase posit8 code 1 through hard-swish; 28 fractional bits resolve every finite hard-activation code and retain graph MAC headroom within 64 bits",
                  "evidence": [reference(evidence_path), prepared["configuration"], prepared["graph"],
                               reference(ROOT / f"public/formats/manifests/accumulators/{POLICY}.json")]}
    target = directory / "accumulator-resolutions" / f"mobilenet_v3_large-posit8_es1-{digest(resolution)[:12]}.json"
    write(target, resolution)
    if source_identity() != source or read(index_path) != before:
        raise ValueError("engine or accumulator index changed during resolution")
    index = deepcopy(before)
    index["resolutions"][KEY] = reference(target)
    write(index_path, index)
    print({"status": "candidate_versioned", "resolution": reference(target), "affected_nodes": len(failures),
           "macs_checked": len(mac["records"]), "nonmacs_checked": len(nonmac["records"])})


if __name__ == "__main__":
    main()
