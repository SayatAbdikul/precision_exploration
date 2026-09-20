"""Seek actual finite-code MAC witnesses for pending posit bias boundaries."""
import argparse
from fractions import Fraction
from math import prod
from time import perf_counter

from public.analysis.phase3.fixed_bias_witness import SparseTwoSum, discrepant_dot_targets
from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend
from public.inference.reference.arithmetic import binary, encode, format_named, model_c, real
from public.inference.tensor import Encoding, Tensor, parse_encoding
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def check_witness(node, weight_codes, encoding, codes, channel, bias, target, native):
    k = len(codes)
    inputs = Tensor((1, k), tuple(codes), encoding)
    weights = Tensor((1, k), tuple(weight_codes), encoding)
    left, right = inputs.values(), weights.values()
    pairs = list(zip(left, right))
    exact_dot = sum((Fraction(a) * Fraction(b) for a, b in pairs), Fraction(0))
    if exact_dot != target:
        raise ValueError("constructed witness does not produce the claimed exact sum")
    acc = format_named(node["attrs"]["accumulator"])
    native_dot = native.flex_gemm(left, right, batch=1, channels=1, k=k, accumulator=acc.name)[0]
    reference_dot = model_c(pairs, acc)
    native_post = acc.rounded(binary(acc.decode(native_dot), acc.rounded(bias), "add"))
    reference_post = model_c(pairs, acc, bias=bias)
    if native_dot != reference_dot or acc.encode(native_post) != reference_post:
        raise ValueError("native/reference accumulator disagreement on bias witness")
    output = format_named(node["attrs"]["output"]["format"])
    exact_code = encode(output, exact_dot + bias)
    stored_code = encode(output, native_post)
    if exact_code == stored_code:
        raise ValueError("sparse witness does not reproduce an output discrepancy")
    return {
        "status": "finite_mac_bias_boundary_witness", "channel": channel,
        "input": inputs.document(), "selected_weights": weights.document(), "bias": str(bias),
        "exact_dot": str(exact_dot), "exact_sum_plus_original_bias": str(exact_dot + bias),
        "prescribed_stored_bias": str(acc.rounded(bias)), "post_bias_value": str(native_post),
        "exact_output_code": exact_code, "prescribed_output_code": stored_code,
        "native_accumulator_code": native_dot, "reference_accumulator_code": reference_dot,
        "native_post_bias_code": acc.encode(native_post), "reference_post_bias_code": reference_post,
        "nonzero_activation_positions": [i for i, value in enumerate(left) if value != 0],
        "native_strategy": native.last_strategy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--format", default="posit8_es1")
    parser.add_argument("--channels", type=int, default=4, help="first pending channels searched per layer")
    parser.add_argument("--targets", type=int, default=32, help="smallest-magnitude discrepant targets searched per channel")
    args = parser.parse_args()
    if args.channels < 1 or args.targets < 1:
        raise ValueError("search limits must be positive")
    plan, source = campaign(), source_identity()
    proof_ref = read(ROOT / "results/summaries/phase3-fixed-bias-sensitivity.json")["immutable_evidence"]
    proof = read(checked(proof_ref))
    if proof["source_sha256"] != source or proof["campaign_sha256"] != digest(plan):
        raise ValueError("bias witness search requires current boundary evidence")
    key = f"{args.model}/{args.format}"
    prepared = preparation_records()[key]
    config = read(checked(prepared["configuration"]))
    if config != configuration(args.model, args.format, plan) or config["runtime"]["source_sha256"] != source:
        raise ValueError("bias witness search requires current preparation")
    record = next(row for row in proof["records"] if row["configuration"] == key)
    if record["graph"] != prepared["graph"] or record["configuration_artifact"] != prepared["configuration"]:
        raise ValueError("boundary proof refers to a different graph")
    graph = read(checked(prepared["graph"]))
    nodes = {node["name"]: node for node in graph["nodes"]}
    domains = dict(graph["inputs"])
    for node in graph["nodes"]:
        domains[node["name"]] = node["attrs"].get("output", domains[node["inputs"][0]])
    paths = ("tools/run/phase3_fixed_bias_witness.py", "public/analysis/phase3/fixed_bias_witness.py",
             "public/analysis/phase3/fixed_bias_sensitivity.py", "public/analysis/phase3/fixed_mac_bounds.py")
    implementations = [source_reference(ROOT / p) for p in paths]
    native = NativeBackend("cpp")
    study = {"schema_version": "phase3-fixed-bias-witness-job-1.0.0", "source_sha256": source,
             "campaign_sha256": digest(plan), "configuration": prepared["configuration"], "graph": prepared["graph"],
             "boundary_proof": proof_ref, "implementations": implementations,
             "native_library": reference(ROOT / "build/phase2/cpp/libprecision_cpp.so"),
             "native_build": reference(ROOT / "build/phase2/cpp/libprecision_cpp.json"),
             "search": {"channels_per_layer": args.channels, "targets_per_channel": args.targets,
                        "target_order": "absolute_value_then_signed_value", "maximum_nonzero_activations": 2}}
    work = ROOT / "artifacts/phase3/bias-witnesses" / digest(study)
    write(work / "job.json", study)
    layers = []
    for layer in record["layers"]:
        if layer["status"] != "pending":
            continue
        node = nodes[layer["node"]]
        x, output = (parse_encoding(d) for d in (domains[node["inputs"][0]], node["attrs"]["output"]))
        weight = graph["constants"][node["inputs"][1]]
        w = parse_encoding(weight["encoding"])
        if (any(not isinstance(e, Encoding) or e.axis is not None or e.block_size is not None or
                e.scales != (Fraction(1),) for e in (x, w, output)) or x != w or x.format != args.format):
            raise ValueError("witness search requires identical intrinsic scalar posit operands/output")
        fmt, acc = format_named(x.format), format_named(node["attrs"]["accumulator"])
        if fmt.family != "posit" or acc.family != "fixed_point":
            raise ValueError("witness search requires a finite posit/fixed-grid MAC")
        finite = {code: Fraction(v) for code in range(1 << fmt.bits) if (v := fmt.decode(code)).is_finite()}
        k = prod(weight["shape"][1:])
        step = Fraction(1, 1 << acc.manifest["numeric"]["fractional_bits"])
        started = perf_counter()
        attempts, witness = [], None
        for channel in [c for c in layer["channels"] if c["status"] == "pending"][:args.channels]:
            c = channel["channel"]
            wc = weight["codes"][c*k:(c+1)*k]
            if len(wc) != k:
                raise ValueError("weight channel length mismatch")
            bias = real(node["attrs"]["bias"][c]) if "bias" in node["attrs"] else Fraction(0)
            search = SparseTwoSum([finite[code] for code in wc], finite)
            if search.quantum != Fraction(channel["dot_quantum"]):
                raise ValueError("witness product lattice differs from retained bound")
            targets = discrepant_dot_targets(search.quantum, channel["maximum_absolute_dot"], bias, step, output.format)
            attempted = 0
            for target in targets[:args.targets]:
                attempted += 1
                codes = search.find(target)
                if codes is not None:
                    witness = check_witness(node, wc, x, codes, c, bias, target, native)
                    break
            attempts.append({"channel": c, "available_targets": len(targets), "searched_targets": attempted})
            if witness is not None:
                break
        row = {"node": node["name"], "k": k, "accumulator": acc.name, "attempts": attempts,
               "witness": witness, "status": "witness_found" if witness else "no_sparse_witness_found",
               "seconds": perf_counter() - started}
        write(work / f"{node['name']}.json", row)
        layers.append({**row, "evidence": reference(work / f"{node['name']}.json")})
        print(key, node["name"], row["status"], f"{row['seconds']:.2f}s", flush=True)
    if source_identity() != source or any(reference(ROOT / p)["sha256"] != ref["sha256"] for p, ref in zip(paths, implementations)):
        raise ValueError("witness sources changed during execution")
    result = {"schema_version": "phase3-fixed-bias-witness-1.0.0", "status": "local_sensitivity_only",
              "job": reference(work / "job.json"), "configuration": key, "layers": layers,
              "layers_with_witnesses": sum(r["status"] == "witness_found" for r in layers),
              "limits": ["finite activation vectors with at most two nonzero positions and actual encoded weight rows",
                         "native C++ and exact Model C agree under the declared accumulator, including prescribed bias storage",
                         "output differences compare original frozen bias with prescribed stored bias; no backend bug is implied",
                         "local encoded-vector reachability does not prove upstream graph or natural-image reachability",
                         "bounded searches without witnesses do not establish output-code stability",
                         "no configuration revision, graph acceptance or D4 ranking is issued"]}
    write(work / "summary.json", result)
    write(ROOT / f"results/summaries/phase3-fixed-bias-witness-{args.model}-{args.format}.json", result)


if __name__ == "__main__":
    main()
