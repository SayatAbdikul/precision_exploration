"""Retain local quire/fixed MAC grid and headroom proofs for current candidates."""
from public.analysis.phase3.fixed_mac_bounds import fixed_mac_bounds
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    records = []
    for key, prepared in preparation_records().items():
        config = read(checked(prepared["configuration"]))
        if format_named(config["formats"]["accumulator"]["name"]).family != "fixed_point":
            continue
        if config != configuration(config["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("fixed MAC bounds require current preparation")
        result = fixed_mac_bounds(read(checked(prepared["graph"])))
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"], **result})
        print(key, len(result["records"]), "MACs; pending", len(result["pending_nodes"]), flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during fixed MAC validation")
    report = {"schema_version": "phase3-fixed-mac-bounds-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "local_proofs_only_not_graph_acceptance", "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_fixed_mac_bounds.py", "public/analysis/phase3/fixed_mac_bounds.py")]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"fixed-mac-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-fixed-mac-bounds.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
