"""Small actual-K C++/reference checks for prepared non-shared wide policies."""
import argparse
from fractions import Fraction
from math import prod
from time import perf_counter

from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend
from public.inference.reference.arithmetic import format_named, model_c, real
from public.inference.tensor import Encoding, Tensor, parse_encoding
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def check_layer(graph, node, native):
    domains = dict(graph["inputs"])
    for item in graph["nodes"]:
        if item["name"] == node["name"]:
            break
        domains[item["name"]] = item["attrs"].get("output", domains[item["inputs"][0]])
    encoding = parse_encoding(domains[node["inputs"][0]])
    if not isinstance(encoding, Encoding) or encoding.axis is not None or encoding.block_size is not None:
        raise ValueError("family pilot requires scalar non-shared activation encoding")
    fmt = format_named(encoding.format)
    finite = sorted((real(fmt.decode(code)), code) for code in range(1 << fmt.bits) if fmt.decode(code).is_finite())
    near_zero = min(finite, key=lambda pair: (abs(pair[0]), pair[1]))[1]
    pattern = tuple(dict.fromkeys((near_zero, finite[0][1], finite[-1][1], finite[len(finite)//3][1], finite[2*len(finite)//3][1])))
    weights = Tensor.from_document(graph["constants"][node["inputs"][1]])
    if weights.encoding.block_size is not None or weights.encoding.axis not in (None, 0):
        raise ValueError("family pilot requires ordinary scalar or per-channel weights")
    k, channels = prod(weights.shape[1:]), weights.shape[0]
    bias = node["attrs"].get("bias")
    selected = sorted({0, channels-1, max(range(channels), key=lambda c: abs(real(bias[c]))) if bias else 0})
    inputs = Tensor((1, k), tuple(pattern[i % len(pattern)] for i in range(k)), encoding)
    w_encoding = Encoding(weights.encoding.format,
                          tuple(weights.encoding.scales[c] for c in selected) if weights.encoding.axis == 0 else weights.encoding.scales,
                          weights.encoding.axis)
    rows = Tensor((len(selected), k), tuple(code for c in selected for code in weights.codes[c*k:(c+1)*k]), w_encoding)
    left, right = inputs.values(), rows.values()
    accumulator = node["attrs"]["accumulator"]
    acc = format_named(accumulator)
    if acc.family not in {"float", "fixed_point"}:
        raise ValueError("integer product-domain policies have separate checks")
    started = perf_counter()
    actual = native.flex_gemm(left, right, batch=1, channels=len(selected), k=k, accumulator=accumulator)
    seconds = perf_counter()-started
    checks = []
    for ordinal, channel in enumerate(selected):
        pairs = list(zip(left, right[ordinal*k:(ordinal+1)*k]))
        expected = model_c(pairs, acc)
        bias_value = real(bias[channel]) if bias else None
        post = acc.rounded(acc.decode(actual[ordinal])+acc.rounded(bias_value)) if bias else acc.decode(actual[ordinal])
        expected_post = acc.decode(model_c(pairs, acc, bias=bias_value))
        finite_post = all(isinstance(value, Fraction) or value.is_finite() for value in (post, expected_post))
        checks.append({"channel": channel, "native_accumulator_code": actual[ordinal], "reference_accumulator_code": expected,
                       "pre_store_accumulator_equal": actual[ordinal] == expected,
                       "post_bias_equal": acc.encode(post) == acc.encode(expected_post), "finite_post_bias": finite_post,
                       "post_bias_value": str(post),
                       "error_against_exact_sum": str(abs(Fraction(post)-(sum((Fraction(a)*Fraction(b) for a,b in pairs), Fraction(0))+Fraction(bias_value or 0)))) if finite_post else None})
    return {"node": node["name"], "k": k, "accumulator": accumulator, "input": inputs.document(),
            "selected_weights": rows.document(), "selected_bias": [bias[c] for c in selected] if bias else None,
            "native_seconds": seconds, "native_strategy": native.last_strategy, "checks": checks,
            "status": "matching_finite_selected_rows" if all(r["pre_store_accumulator_equal"] and r["post_bias_equal"] and r["finite_post_bias"] for r in checks)
                      else "requires_diagnosis"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet18")
    args = parser.parse_args()
    plan, source = campaign(), source_identity()
    if args.model not in plan["models"]:
        raise ValueError("model is outside the campaign")
    native, records, skipped = NativeBackend("cpp"), [], []
    for key, prepared in preparation_records().items():
        if prepared["model"] != args.model:
            continue
        config = read(checked(prepared["configuration"]))
        fmt = format_named(prepared["format"])
        if fmt.manifest["scaling"]["mode"] == "intrinsic_shared" or format_named(config["formats"]["accumulator"]["name"]).family == "integer":
            skipped.append({"configuration": key, "reason": "shared block layout or integer product-domain policy needs its separate pilot"})
            continue
        if config != configuration(args.model, prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("family pilot requires current preparation")
        graph = read(checked(prepared["graph"]))
        macs = [n for n in graph["nodes"] if n["op"] in {"linear", "conv2d", "depthwise_conv2d"}]
        longest = max(macs, key=lambda n: prod(graph["constants"][n["inputs"][1]]["shape"][1:]))
        row = {"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"], "layers": []}
        for node in {n["name"]: n for n in (macs[0], longest)}.values():
            print(f"checking {key}/{node['name']}", flush=True)
            row["layers"].append(check_layer(graph, node, native))
        records.append(row)
        print(key, [r["status"] for r in row["layers"]], flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during family MAC checks")
    report = {"schema_version": "phase3-family-mac-pilot-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "selected_rows_only_not_acceptance", "model": args.model, "backend": "cpp", "records": records, "skipped": skipped,
              "native_library": reference(ROOT / "build/phase2/cpp/libprecision_cpp.so"),
              "native_build": reference(ROOT / "build/phase2/cpp/libprecision_cpp.json"),
              "implementation": source_reference(__file__),
              "limits": ["first and longest-K MAC; first, last and largest-bias channels, deduplicated",
                         "deterministic finite code pattern includes extrema; not real-image activation frequencies",
                         "C++/reference accumulator codes are compared before output quantization",
                         "bias application is checked in the common Python path; no independent native bias kernel is claimed",
                         "exact sums use the oracle's declared finite values, including its approximations for irrational formats",
                         "not full graph shape/lowering checks, CUDA conformance, a precision sufficiency proof, or a quality ranking"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"family-mac-{digest(report)}.json"
    write(path, report)
    write(ROOT / f"results/summaries/phase3-family-mac-pilot-{args.model}.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
