"""Retain ordinary finite FP64 non-MAC checks without issuing acceptance."""
from public.analysis.phase3.finite_fp64_nonmac import prove_nonmac
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    index = read(ROOT / "results/summaries/phase3-shapes.json")
    if index["campaign_sha256"] != digest(plan):
        raise ValueError("shape inventory belongs to another campaign")
    records = []
    for key, prepared in preparation_records().items():
        config = read(checked(prepared["configuration"]))
        if (config["formats"]["accumulator"]["name"] != "fp64_e11m52_accumulator"
                or format_named(prepared["format"]).manifest["scaling"]["mode"] == "intrinsic_shared"):
            continue
        if config != configuration(config["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("finite FP64 non-MAC checks require current preparation")
        shape_ref = index["models"][config["model"]]
        observed = read(checked(shape_ref))
        if observed["campaign_sha256"] != digest(plan) or observed["model_manifest"] != plan["inputs"][config["model"]]:
            raise ValueError("shape observation identity mismatch")
        result = prove_nonmac(read(checked(prepared["graph"])), observed["shapes"])
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"],
                        "shape_evidence": shape_ref, **result})
        print(key, len(result["records"]), "non-MACs; pending", result["pending_nodes"], flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during finite FP64 non-MAC checks")
    report = {"schema_version": "phase3-finite-fp64-nonmac-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "local_checks_only_not_graph_acceptance", "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_finite_fp64_nonmac.py", "public/analysis/phase3/finite_fp64_nonmac.py",
                  "public/analysis/phase3/fp64_bounds.py", "public/analysis/phase3/fixed_nonmac.py",
                  "public/analysis/phase3/fixed_mac_bounds.py")]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"finite-fp64-nonmac-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-finite-fp64-nonmac.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
