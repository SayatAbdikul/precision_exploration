"""Inventory static integer proofs without issuing native/workload acceptance."""
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from tools.phase3.acceptance import prove_integer_graph
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    shape_index = read(ROOT / "results/summaries/phase3-shapes.json")
    if shape_index["campaign_sha256"] != digest(plan):
        raise ValueError("shape inventory belongs to another campaign")
    records = []
    for key, prepared in preparation_records().items():
        config = read(checked(prepared["configuration"]))
        if format_named(config["formats"]["accumulator"]["name"]).family != "integer":
            continue
        if config != configuration(config["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("static readiness requires current preparation")
        shape_ref = shape_index["models"][config["model"]]
        observed = read(checked(shape_ref))
        if observed["campaign_sha256"] != digest(plan) or observed["model_manifest"] != plan["inputs"][config["model"]]:
            raise ValueError("shape observation belongs to another model or campaign")
        graph = read(checked(prepared["graph"]))
        manifest = read(checked(plan["inputs"][config["model"]]))
        proof = prove_integer_graph(graph, {name: manifest["input_shape"] for name in graph["inputs"]}, observed["shapes"])
        passed = proof.pop("status") == "accepted"
        record = {"configuration": key, "configuration_artifact": prepared["configuration"],
                  "graph": prepared["graph"], "shape_evidence": shape_ref,
                  "status": "static_proof_passed" if passed else "pending_static_proof", "proof": proof}
        records.append(record)
        print(key, record["status"], flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during static integer proof inventory")
    report = {"schema_version": "phase3-integer-readiness-1.0.0", "campaign_sha256": digest(plan),
              "source_sha256": source, "status": "static_inventory_only", "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_integer_readiness.py", "tools/phase3/acceptance.py", "tools/phase3/graphs.py")],
              "limits": ["fixed FP32-observed shapes used here still need native confirmation",
                         "passing the existing finite-integer proof does not issue graph acceptance",
                         "full screens still require eight-image C++/CUDA agreement and complete diagnostics",
                         "unsupported operations remain pending, not rejected or eliminated"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"integer-readiness-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-integer-readiness.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
