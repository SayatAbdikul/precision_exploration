"""Bound ordinary FP64 MACs and explicitly retain pending shared-block MACs."""
from public.analysis.phase3.finite_fp64_bounds import finite_mac_bounds
from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    records = []
    for key, prepared in preparation_records().items():
        config = read(checked(prepared["configuration"]))
        if config["formats"]["accumulator"]["name"] != "fp64_e11m52_accumulator":
            continue
        if config != configuration(prepared["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("finite FP64 bounds require current preparation")
        result = finite_mac_bounds(read(checked(prepared["graph"])))
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"],
                        "graph": prepared["graph"], **result})
        print(key, len(result["macs"]), "bounded MACs;", len(result["pending"]), "pending", flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during finite FP64 checks")
    report = {"schema_version": "phase3-finite-fp64-mac-bounds-1.0.0", "source_sha256": source,
              "campaign_sha256": digest(plan), "status": "local_bounds_only_not_graph_acceptance", "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_finite_fp64_bounds.py", "public/analysis/phase3/finite_fp64_bounds.py",
                  "public/analysis/phase3/fp64_bounds.py")]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"finite-fp64-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-finite-fp64-mac-bounds.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
