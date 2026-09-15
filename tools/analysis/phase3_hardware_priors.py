"""Record nominal bit-storage and limited Phase 2 synthesis preservation priors."""
from public.analysis.phase3.storage import tensor_storage, shared_scale_count, graph_storage
from public.analysis.phase3.resources import graph_resources, format_support_prior
from public.inference.reference.arithmetic import format_named
from tools.phase3.common import ROOT, campaign, read, checked, reference, write, digest
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan = campaign()
    specs = read(checked(plan["inputs"]["formats"]))["manifests"]
    rows = []
    shape = [32, 33]  # Deliberately includes incomplete blocks and per-row metadata.
    for spec in specs:
        fmt = format_named(spec["name"])
        mode = fmt.manifest["scaling"]["mode"]
        count = shared_scale_count(shape, 1, fmt.manifest["block"]["block_size"]) if mode == "intrinsic_shared" else shape[0] if mode == "required_mapping" else 0
        width = fmt.manifest["scaling"]["scale_bits"] if mode == "intrinsic_shared" else 64 if count else 0
        rows.append({"format": fmt.name, "family": fmt.family, "example_shape": shape,
                     "scope": "illustrative weight matrix, not a model storage total", "scale_bits_assumption": width,
                     **tensor_storage(shape, fmt.bits, scale_count=count, scale_bits=width),
                     "support_arithmetic": format_support_prior(fmt.name)})
    graphs = []
    shape_path = ROOT / "results/summaries/phase3-shapes.json"
    shapes = read(shape_path) if shape_path.exists() else {"models": {}}
    if shape_path.exists() and shapes["campaign_sha256"] != digest(plan):
        raise ValueError("shape inventory belongs to another campaign")
    for key, row in preparation_records().items():
        if "graph" in row:
            graph = read(checked(row["graph"]))
            item = {"configuration": key, "configuration_artifact": row["configuration"],
                    "graph": row["graph"], "storage": graph_storage(graph)}
            if row["model"] in shapes["models"]:
                shape_ref = shapes["models"][row["model"]]
                observed = read(checked(shape_ref))
                if (observed["campaign_sha256"] != digest(plan) or observed["model"] != row["model"]
                        or observed["model_manifest"] != plan["inputs"][row["model"]]):
                    raise ValueError("shape observation identity mismatch")
                checked(observed["model_manifest"])
                for implementation in observed["implementations"]:
                    checked(implementation)
                item.update(shape_evidence=shape_ref, resources=graph_resources(graph, observed["shapes"]))
            else:
                item["resources_status"] = "pending_shape_observation"
            graphs.append(item)
    snapshot = checked(plan["inputs"]["phase2"]).parent
    pilot_path = snapshot / "phase2-hardware-pilot.json"
    report = {"schema_version": "phase3-hardware-priors-1.1.0", "campaign_sha256": digest(plan), "format_priors": rows,
              "prepared_graph_storage": graphs, "phase2_primitive_evidence": reference(pilot_path),
              "implementations": [source_reference(ROOT / path) for path in (
                  "tools/analysis/phase3_hardware_priors.py", "public/analysis/phase3/storage.py", "public/analysis/phase3/resources.py")],
              "scope": "rough packed storage, graph liveness, logical arithmetic and primitive preservation signals",
              "limits": ["Phase 2 ROM primitive pilots are not complete Model C MAC architectures",
                         "no routed area, extracted parasitics, measured workload energy, or measured peak memory is claimed",
                         "mapped 64-bit scale storage is an assumption, not a proof of exact rational scale implementation",
                         "operator counts and alternative buffer allocations do not determine the cost of decode, conversion or control",
                         "these priors cannot alone eliminate any configuration or family"]}
    write(ROOT / "results/summaries/phase3-hardware-priors.json", report)
    print(f"Recorded {len(rows)} format storage priors and {len(graphs)} prepared graph estimates")


if __name__ == "__main__":
    main()
