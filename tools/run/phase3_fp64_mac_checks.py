"""Check selected actual-K FP64 MAC rows against the rational reference on CPU."""
from fractions import Fraction
from math import prod
from time import perf_counter

from public.analysis.phase3.fp64_bounds import error_bound
from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend
from public.inference.reference.arithmetic import format_named, model_c, real
from public.inference.tensor import Encoding, Tensor, parse_encoding
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def check_rows(graph, node, native):
    domains = dict(graph["inputs"])
    for item in graph["nodes"]:
        if item["name"] == node["name"]:
            break
        domains[item["name"]] = item["attrs"].get("output", domains[item["inputs"][0]])
    domain = parse_encoding(domains[node["inputs"][0]])
    fmt = format_named(domain.format)
    if fmt.family != "integer" or domain.axis is not None:
        raise ValueError("MAC check requires scalar integer activation mapping")
    weight = Tensor.from_document(graph["constants"][node["inputs"][1]])
    k, channels = prod(weight.shape[1:]), weight.shape[0]
    bias = node["attrs"].get("bias")
    selected = sorted({0, channels-1, max(range(channels), key=lambda c: abs(real(bias[c]))) if bias else 0})
    pattern = (0, (1 << (fmt.bits-1))-1, 1 << (fmt.bits-1), (1 << fmt.bits)-1, 1)
    inputs = Tensor((1, k), tuple(pattern[i % len(pattern)] for i in range(k)), domain)
    left = inputs.values()
    if weight.encoding.block_size is not None or weight.encoding.axis not in {None, 0}:
        raise ValueError("MAC check requires scalar or per-output-channel weight mapping")
    selected_encoding = Encoding(weight.encoding.format,
        tuple(weight.encoding.scales[c] for c in selected) if weight.encoding.axis == 0 else weight.encoding.scales,
        weight.encoding.axis)
    selected_weights = Tensor((len(selected), k), tuple(code for c in selected for code in weight.codes[c*k:(c+1)*k]), selected_encoding)
    right = selected_weights.values()
    accumulator = node["attrs"]["accumulator"]
    if accumulator != "fp64_e11m52_accumulator":
        raise ValueError("MAC check requires the FP64 revision")
    acc = format_named(accumulator)
    start = perf_counter()
    actual = native.flex_gemm(left, right, batch=1, channels=len(selected), k=k, accumulator=accumulator)
    native_seconds = perf_counter()-start
    results = []
    for ordinal, channel in enumerate(selected):
        pairs = list(zip(left, right[ordinal*k:(ordinal+1)*k]))
        expected = model_c(pairs, acc)
        if actual[ordinal] != expected:
            raise ValueError("native FP64 accumulator state differs from exact Model C reference")
        real_bias = real(bias[channel]) if bias else None
        post_bias = acc.rounded(acc.decode(actual[ordinal])+acc.rounded(real_bias)) if bias else acc.decode(actual[ordinal])
        reference_post_bias = acc.decode(model_c(pairs, acc, bias=real_bias))
        if post_bias != reference_post_bias:
            raise ValueError("stored bias application differs from Model C reference")
        exact = sum((a*b for a, b in pairs), Fraction(0))+(real_bias or 0)
        bound = error_bound(sum(abs(a*b) for a, b in pairs), k, absolute_bias=None if real_bias is None else abs(real_bias))
        observed_error = abs(post_bias-exact)
        if not bound["overflow_excluded"] or observed_error > bound["absolute_error_bound"]:
            raise ValueError("observed FP64 error exceeds the conservative bound")
        results.append({"channel": channel, "native_accumulator_code": actual[ordinal],
                        "reference_accumulator_code": expected, "post_bias_value": str(post_bias),
                        "absolute_error": str(observed_error), "absolute_error_bound": str(bound["absolute_error_bound"])})
    return {"node": node["name"], "k": k, "selected_channels": selected,
            "input": inputs.document(), "selected_weights_sha256": digest(selected_weights.document()),
            "native_strategy": native.last_strategy, "native_seconds": native_seconds, "rows": results}


def main():
    source, records = source_identity(), []
    native = NativeBackend("cpp")
    for key, prepared in preparation_records().items():
        if "configuration" not in prepared:
            continue
        config = read(checked(prepared["configuration"]))
        if config["formats"]["accumulator"]["name"] != "fp64_e11m52_accumulator" or not config.get("accumulator_resolution"):
            continue
        if config["runtime"]["source_sha256"] != source:
            raise ValueError("native MAC check requires current-source preparation")
        graph = read(checked(prepared["graph"]))
        macs = [n for n in graph["nodes"] if n["op"] in {"linear", "conv2d", "depthwise_conv2d"}]
        widest = max(macs, key=lambda n: prod(graph["constants"][n["inputs"][1]]["shape"][1:]))
        chosen = {n["name"]: n for n in (macs[0], widest)}
        cases = []
        for node in chosen.values():
            print(f"checking {key}/{node['name']}", flush=True)
            cases.append(check_rows(graph, node, native))
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"], "cases": cases})
    if source_identity() != source:
        raise ValueError("engine changed during native FP64 checks")
    report = {"schema_version": "phase3-fp64-mac-checks-1.0.0", "campaign_sha256": digest(campaign()), "source_sha256": source,
              "status": "verified_selected_rows", "backend": "cpp", "records": records,
              "native_library": reference(ROOT / "build/phase2/cpp/libprecision_cpp.so"),
              "native_build": reference(ROOT / "build/phase2/cpp/libprecision_cpp.json"),
              "implementations": [source_reference(ROOT / path) for path in (
                  "tools/run/phase3_fp64_mac_checks.py", "public/analysis/phase3/fp64_bounds.py")],
              "limits": ["synthetic stored-code pattern with actual weight rows, mapping scales, bias and full reduction length",
                         "first and longest-K MAC; first, last and largest-bias output channels, deduplicated",
                         "C++ accumulator-state agreement is checked before output quantization can hide errors",
                         "not a native-resolution image pilot, CUDA comparison or whole-graph acceptance"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"native-fp64-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-fp64-mac-checks.json", {**report, "immutable_evidence": reference(path)})
    print({"configurations": len(records), "selected_mac_layers": sum(len(r["cases"]) for r in records)})


if __name__ == "__main__":
    main()
