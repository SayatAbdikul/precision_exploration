"""Audit integer nonlinear rounding before expensive MobileNetV3/YOLO pilots."""
from public.analysis.phase3.accumulator_probes import probe_graph
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from tools.phase3.common import ROOT, campaign, checked, digest, read, write
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, records = campaign(), []
    source = source_identity()
    for key, prepared in preparation_records().items():
        if "graph" not in prepared:
            continue
        config = read(checked(prepared["configuration"]))
        if format_named(config["formats"]["activation"]["name"]).family != "integer":
            continue
        if config["runtime"]["source_sha256"] != source:
            raise ValueError("nonlinear audit requires current engine graph definitions")
        graph = read(checked(prepared["graph"]))
        probes = probe_graph(graph)
        if probes:
            records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"],
                            "probes": probes, "requires_precision_revision": any(p["requires_precision_revision"] for p in probes)})
    if source_identity() != source:
        raise ValueError("engine changed during nonlinear accumulator audit")
    report = {"schema_version": "phase3-nonlinear-accumulator-audit-1.0.0", "campaign_sha256": digest(plan),
              "source_sha256": source, "records": records, "status": "diagnostic_only_no_acceptances",
              "implementations": [source_reference(ROOT / path) for path in (
                  "tools/analysis/phase3_nonlinear_accumulators.py", "public/analysis/phase3/accumulator_probes.py")],
              "limits": ["these probes preserve candidate operand encodings and do not revise active configurations",
                         "whole-graph policy resolution and native validation remain required; failures do not reject an operand family"]}
    write(ROOT / "results/summaries/phase3-nonlinear-accumulator-audit.json", report)
    print({"probed_configurations": len(records), "precision_revision_required": sum(r["requires_precision_revision"] for r in records)})


if __name__ == "__main__":
    main()
