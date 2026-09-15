"""Relate retained native FP64 states to local bounds and output stores."""
from fractions import Fraction

from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend
from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import Tensor, parse_encoding
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.run.phase3_family_mac_pilot import check_layer


def compare(layer, node, bound):
    inputs = Tensor.from_document(layer["input"]).values()
    weights = Tensor.from_document(layer["selected_weights"]).values()
    k = layer["k"]
    acc = format_named(layer["accumulator"])
    output = parse_encoding(node["attrs"]["output"])
    if output.axis is not None or output.block_size is not None:
        raise ValueError("spacing sensitivity requires scalar output encoding")
    records = []
    for ordinal, check in enumerate(layer["checks"]):
        if not all(check[key] for key in ("pre_store_accumulator_equal", "post_bias_equal", "finite_post_bias")):
            raise ValueError("native family state failed its reference comparison")
        exact = sum((Fraction(a)*Fraction(b) for a, b in zip(inputs, weights[ordinal*k:(ordinal+1)*k])), Fraction(0))
        post = acc.decode(check["native_accumulator_code"])
        if layer["selected_bias"] is not None:
            bias = real(layer["selected_bias"][ordinal])
            exact += bias
            post = acc.rounded(post+acc.rounded(bias))
        error = abs(Fraction(post)-exact)
        if error != Fraction(check["error_against_exact_sum"]) or Fraction(post) != Fraction(check["post_bias_value"]):
            raise ValueError("retained native post-bias error does not reproduce")
        exact_code = Tensor.quantize([exact], (1,), output).codes[0]
        actual_code = Tensor.quantize([post], (1,), output).codes[0]
        records.append({"channel": check["channel"], "native_accumulator_code": check["native_accumulator_code"],
                        "exact_sum_plus_bias": str(exact), "fp64_post_bias": str(post), "absolute_error": str(error),
                        "within_local_bound": error <= Fraction(bound["maximum_absolute_error_bound"]),
                        "error_in_minimum_output_spacings": str(error/Fraction(bound["minimum_finite_output_spacing"])),
                        "exact_output_code": exact_code, "fp64_output_code": actual_code,
                        "output_codes_equal": exact_code == actual_code})
    return {"node": layer["node"], "k": k, "checks": records}


def main():
    plan, source = campaign(), source_identity()
    inventory = preparation_records()
    summary = read(ROOT / "results/summaries/phase3-finite-fp64-mac-bounds.json")
    bounds_ref = summary["immutable_evidence"]
    bounds = read(checked(bounds_ref))
    if bounds["source_sha256"] != source or bounds["campaign_sha256"] != digest(plan):
        raise ValueError("FP64 local bounds are stale")
    by_key = {r["configuration"]: r for r in bounds["records"] if r["macs"]}
    sources, records = [], []
    for model in plan["models"]:
        family = read(ROOT / f"results/summaries/phase3-family-mac-pilot-{model}.json")
        family_ref = family["immutable_evidence"]
        family = read(checked(family_ref))
        if family["source_sha256"] != source or family["campaign_sha256"] != digest(plan):
            raise ValueError("retained family native evidence is stale")
        checked(family["native_library"])
        checked(family["native_build"])
        checked(family["implementation"])
        sources.append(family_ref)
        for row in family["records"]:
            key = row["configuration"]
            if key not in by_key:
                continue
            prepared = inventory[key]
            config = read(checked(prepared["configuration"]))
            if config != configuration(model, prepared["format"], plan):
                raise ValueError("spacing checks require the active configuration")
            if row["graph"] != prepared["graph"] or row["configuration_artifact"] != prepared["configuration"]:
                raise ValueError("family native evidence does not match the active graph")
            graph = read(checked(prepared["graph"]))
            nodes = {n["name"]: n for n in graph["nodes"]}
            bound_rows = {b["node"]: b for b in by_key[key]["macs"]}
            records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"],
                            "layers": [compare(layer, nodes[layer["node"]], bound_rows[layer["node"]]) for layer in row["layers"]]})
    # Exercise every layer whose generic worst-case spacing bound was loose,
    # including layers not selected by the original first/longest-K policy.
    native, targeted = NativeBackend("cpp"), []
    for key, row in by_key.items():
        loose = [b for b in row["macs"] if not b["below_half_minimum_output_spacing"]]
        if not loose:
            continue
        graph = read(checked(row["graph"]))
        nodes = {n["name"]: n for n in graph["nodes"]}
        for bound in loose:
            layer = check_layer(graph, nodes[bound["node"]], native)
            targeted.append({"configuration": key, "graph": row["graph"], "bound": bound,
                             "native_case": layer, "comparison": compare(layer, nodes[bound["node"]], bound)})
            print(key, bound["node"], "targeted selected rows complete", flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during FP64 spacing sensitivity")
    report = {"schema_version": "phase3-fp64-spacing-sensitivity-1.0.0", "source_sha256": source,
              "campaign_sha256": digest(plan), "status": "selected_rows_only_not_acceptance", "bounds": bounds_ref,
              "family_native_evidence": sources, "records": records, "targeted_loose_bounds": targeted,
              "native_library": reference(ROOT / "build/phase2/cpp/libprecision_cpp.so"),
              "native_build": reference(ROOT / "build/phase2/cpp/libprecision_cpp.json"),
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_fp64_spacing_sensitivity.py", "tools/run/phase3_family_mac_pilot.py")],
              "limits": ["selected finite synthetic activation patterns and real prepared weight rows; not image frequencies",
                         "targeted cases overlap the retained first/longest-K cases and must not be added as unique coverage",
                         "post-bias arithmetic uses the common Python path, not an independent native bias kernel",
                         "matching stored outputs here cannot tighten the all-input analytical bound or accept a graph"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"fp64-spacing-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-fp64-spacing-sensitivity.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
