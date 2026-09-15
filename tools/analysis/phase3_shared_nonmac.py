"""Retain common-scale residual proofs and shared non-accumulator operations."""
from public.analysis.phase3.shared_nonmac import prove_nonmac
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
        if format_named(prepared["format"]).manifest["scaling"]["mode"] != "intrinsic_shared":
            continue
        config = read(checked(prepared["configuration"]))
        if config != configuration(prepared["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("shared non-MAC checks require current preparation")
        result = prove_nonmac(read(checked(prepared["graph"])))
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"], **result})
        print(key, len(result["records"]), "nodes; pending", result["pending_nodes"], flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during shared non-MAC checks")
    report = {"schema_version": "phase3-shared-nonmac-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "conditional_local_coverage_not_acceptance", "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_shared_nonmac.py", "public/analysis/phase3/shared_nonmac.py",
                  "public/analysis/phase3/finite_fp64_nonmac.py", "public/analysis/phase3/fp64_bounds.py",
                  "public/analysis/phase3/fixed_mac_bounds.py")]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"shared-nonmac-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-shared-nonmac.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
