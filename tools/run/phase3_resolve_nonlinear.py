"""Version FP64 candidates for reproduced integer nonlinear precision failures."""
from public.analysis.phase3.accumulator_probes import probe_graph
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def candidate_resolution(config, prepared, evidence, previous, plan):
    policies = read(checked(plan["inputs"]["accumulator_candidates"]))["records"]
    name = config["formats"]["activation"]["name"]
    original = next(row["candidate_accumulator"] for row in policies if row["format"] == name)
    result = {"schema_version": "phase3-accumulator-resolution-1.0.0", "campaign_sha256": digest(plan),
              "model": config["model"], "format": name, "replaces": original,
              "accumulator": "fp64_e11m52_accumulator", "status": "candidate_pending_native_acceptance",
              "reason": "reproduced fractional rounding in integer nonlinear arithmetic; evaluate FP64 throughout the graph with unchanged operand encodings",
              "evidence": [evidence, prepared["configuration"], prepared["accumulator_bounds"]]}
    if previous is not None:
        result["supersedes"] = previous
        result["evidence"].append(previous)
    return result


def main():
    plan, source = campaign(), source_identity()
    destination = ROOT / "public/experiments/configs/experiment_a"
    index_path = destination / "phase3-accumulator-resolutions-v1.json"
    original_index = read(index_path)
    index = read(index_path)
    revisions = []
    for key, prepared in preparation_records().items():
        if "configuration" not in prepared:
            continue
        config = read(checked(prepared["configuration"]))
        if (format_named(config["formats"]["activation"]["name"]).family != "integer"
                or format_named(config["formats"]["accumulator"]["name"]).family != "integer"):
            continue
        if config != configuration(config["model"], prepared["format"], plan):
            raise ValueError("nonlinear resolution needs the current prepared definition")
        probes = probe_graph(read(checked(prepared["graph"])))
        if not any(p["requires_precision_revision"] for p in probes):
            continue
        if any(p["variants"]["fp64_e11m52_accumulator"]["mismatching_codes"] for p in probes):
            raise ValueError("proposed FP64 policy does not resolve every bounded probe")
        evidence = {"schema_version": "phase3-nonlinear-resolution-evidence-1.0.0", "campaign_sha256": digest(plan),
                    "source_sha256": source, "configuration": prepared["configuration"], "graph": prepared["graph"],
                    "probes": probes, "implementations": [source_reference(ROOT / path) for path in (
                        "tools/run/phase3_resolve_nonlinear.py", "public/analysis/phase3/accumulator_probes.py")],
                    "scope": "bounded nonlinear evidence only; FP64 MAC, bias, reduction and whole-graph acceptance pending"}
        path = ROOT / "artifacts/phase3/accumulator-probes" / f"{digest(evidence)}.json"
        write(path, evidence)
        resolution = candidate_resolution(config, prepared, reference(path), index["resolutions"].get(key), plan)
        target = destination / "accumulator-resolutions" / f"{config['model']}-{prepared['format']}-{digest(resolution)[:12]}.json"
        if target.exists() and read(target) != resolution:
            raise ValueError("accumulator resolution is immutable")
        write(target, resolution)
        index["resolutions"][key] = reference(target)
        revisions.append({"configuration": key, "resolution": reference(target)})
    if source_identity() != source or read(index_path) != original_index:
        raise ValueError("engine or accumulator index changed during resolution")
    write(index_path, index)
    # Immutable event record preserves revisions even after an idempotent rerun.
    event = {"schema_version": "phase3-nonlinear-resolution-event-1.0.0", "campaign_sha256": digest(plan), "revisions": revisions}
    write(ROOT / "artifacts/phase3/accumulator-probes" / f"resolutions-{digest(event)}.json", event)
    print({"new_fp64_candidates": len(revisions), "configurations": [row["configuration"] for row in revisions]})


if __name__ == "__main__":
    main()
