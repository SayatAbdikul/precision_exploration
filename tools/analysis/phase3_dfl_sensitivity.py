"""Retain deterministic YOLO DFL checks against wider exponential arithmetic."""
from public.analysis.phase3.dfl_sensitivity import vectors, check_vectors
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Tensor, parse_encoding
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    prepared = preparation_records()["yolov8n/int8"]
    config = read(checked(prepared["configuration"]))
    if config != configuration("yolov8n", "int8", plan) or config["runtime"]["source_sha256"] != source:
        raise ValueError("DFL sensitivity requires the current prepared definition")
    graph = read(checked(prepared["graph"]))
    domains, rows = dict(graph["inputs"]), []
    for node in graph["nodes"]:
        attrs = node["attrs"]
        domain = domains[node["inputs"][0]]
        domains[node["name"]] = attrs.get("output", domain)
        if node["op"] != "dfl":
            continue
        x, output = parse_encoding(domain), parse_encoding(attrs["output"])
        if x.axis is not None or output.axis is not None or format_named(x.format).family != "integer":
            raise ValueError("DFL checks require scalar integer domains")
        weights = Tensor.from_document(graph["constants"][node["inputs"][1]])
        cases = vectors(format_named(x.format).bits, attrs["bins"])
        print(f"checking {node['name']}: {len(cases)} vectors", flush=True)
        result = check_vectors(cases, x, output, weights, attrs["accumulator"])
        rows.append({"node": node["name"], "input_codes": cases, **result})
        print({key: value for key, value in rows[-1].items() if key not in {"input_codes", "results", "examples"}}, flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during DFL checks")
    report = {"schema_version": "phase3-dfl-sensitivity-1.0.0", "source_sha256": source,
              "campaign_sha256": digest(plan), "configuration": prepared["configuration"], "graph": prepared["graph"],
              "status": "bounded_sensitivity_only_not_graph_acceptance", "records": rows,
              "case_policy": {"seed": 310914, "random_cases": 128,
                              "systematic": "all integer code gaps, 1/half/bins-1 high logits in forward/reverse order; peak positions and uniform endpoints"},
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_dfl_sensitivity.py", "public/analysis/phase3/dfl_sensitivity.py")],
              "limits": ["same stored logits, quantized projection weights, and prescribed probability/output stores",
                         "400/800-digit exp references remove accumulator rounding; agreement is not a transcendental error proof",
                         "projection compared separately with an exact rational sum on identical stored probabilities",
                         "selected vectors are not exhaustive for 16 logits and do not represent image frequencies",
                         "reference operator path only; native MAC conformance and whole-graph acceptance remain separate"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"dfl-sensitivity-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-dfl-sensitivity.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
