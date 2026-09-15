"""Check remaining local FP64 arithmetic against exact output boundaries."""
from public.analysis.phase3.fp64_nonmac import prove_nonmac
from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.phase3.graphs import configuration


def main():
    plan, source = campaign(), source_identity()
    shape_index = read(ROOT / "results/summaries/phase3-shapes.json")
    if shape_index["campaign_sha256"] != digest(plan):
        raise ValueError("shape evidence belongs to another campaign")
    records = []
    for key, prepared in preparation_records().items():
        if "configuration" not in prepared:
            continue
        config = read(checked(prepared["configuration"]))
        if config["formats"]["accumulator"]["name"] != "fp64_e11m52_accumulator" or not config.get("accumulator_resolution"):
            continue
        if config != configuration(config["model"], prepared["format"], plan):
            raise ValueError("non-MAC checks require the active prepared definition")
        shapes_ref = shape_index["models"][config["model"]]
        observed = read(checked(shapes_ref))
        if observed["campaign_sha256"] != digest(plan) or observed["model_manifest"] != plan["inputs"][config["model"]]:
            raise ValueError("shape observation model mismatch")
        result = prove_nonmac(read(checked(prepared["graph"])), observed["shapes"])
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"],
                        "graph": prepared["graph"], "shape_evidence": shapes_ref, **result})
    if source_identity() != source:
        raise ValueError("engine changed during non-MAC validation")
    report = {"schema_version": "phase3-fp64-nonmac-checks-1.0.0", "campaign_sha256": digest(plan), "source_sha256": source,
              "status": "local_checks_only_not_graph_acceptance", "records": records,
              "implementations": [source_reference(ROOT / path) for path in (
                  "tools/analysis/phase3_fp64_nonmac.py", "public/analysis/phase3/fp64_nonmac.py",
                  "public/analysis/phase3/box_bounds.py",
                  "public/analysis/phase3/fp64_bounds.py", "public/analysis/phase3/accumulator_probes.py")]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"nonmac-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-fp64-nonmac.json", {**report, "immutable_evidence": reference(path)})
    print({r["configuration"]: {"nodes": len(r["records"]), "pending": r["pending_nodes"]} for r in records})


if __name__ == "__main__":
    main()
