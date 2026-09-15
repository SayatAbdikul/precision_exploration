"""Probe all three posit detector domains using retained finite code vectors."""
from public.analysis.phase3.posit_dfl_cases import selected_vectors
from public.analysis.phase3.dfl_sensitivity import check_vectors
from public.inference.conformance_job import source_identity
from public.inference.tensor import Tensor, parse_encoding
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    inventory, records = preparation_records(), []
    for name in ("posit4_es0", "posit6_es1", "posit8_es1"):
        prepared = inventory[f"yolov8n/{name}"]
        config = read(checked(prepared["configuration"]))
        if config != configuration("yolov8n", name, plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("posit DFL checks require current preparation")
        graph = read(checked(prepared["graph"]))
        domains, groups = dict(graph["inputs"]), {}
        for node in graph["nodes"]:
            attrs, domain = node["attrs"], domains[node["inputs"][0]]
            domains[node["name"]] = attrs.get("output", domain)
            if node["op"] != "dfl":
                continue
            payload = {"input": domain, "attrs": attrs, "weights": graph["constants"][node["inputs"][1]]}
            key = digest(payload)
            groups.setdefault(key, {"domain": payload, "nodes": []})["nodes"].append(node["name"])
        results = []
        for group in groups.values():
            payload = group["domain"]
            x, out = parse_encoding(payload["input"]), parse_encoding(payload["attrs"]["output"])
            cases = selected_vectors(x, payload["attrs"]["bins"])
            print(f"checking {name}: {len(cases)} vectors shared by {len(group['nodes'])} identical DFL domains", flush=True)
            result = check_vectors(cases, x, out, Tensor.from_document(payload["weights"]), payload["attrs"]["accumulator"])
            results.append({**group, "input_codes": cases, **result})
            print({k:v for k,v in result.items() if k not in {"results", "examples"}}, flush=True)
        records.append({"configuration": f"yolov8n/{name}", "configuration_artifact": prepared["configuration"],
                        "graph": prepared["graph"], "groups": results})
    if source_identity() != source:
        raise ValueError("engine changed during posit DFL checks")
    report = {"schema_version": "phase3-posit-dfl-sensitivity-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "bounded_sensitivity_only_not_acceptance", "records": records,
              "case_policy": {"seed": 310915, "random_cases": 128,
                              "systematic": "uniform finite codes; zero-versus-each-code with selected counts/orders; selected peak positions; no NaR"},
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_posit_dfl.py", "public/analysis/phase3/posit_dfl_cases.py", "public/analysis/phase3/dfl_sensitivity.py")],
              "limits": ["identical DFL domains are deduplicated only after hashing every input/output/accumulator/weight attribute",
                         "actual stored projection weights and prescribed probability/output stores are retained",
                         "400/800-digit reference agreement is not a proof over all logit vectors or transcendental error",
                         "selected finite inputs, not native image frequencies, NaR behavior or whole-graph acceptance"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"posit-dfl-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-posit-dfl-sensitivity.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
