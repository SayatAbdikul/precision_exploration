"""Check selected native MAC rows with explicit shared scales across full K."""
from fractions import Fraction
from math import ceil
from time import perf_counter

from public.analysis.phase3.fp64_bounds import error_bound
from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend
from public.inference.reference.arithmetic import binary, format_named, model_c, real
from public.inference.tensor import Encoding, Tensor
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def check_layer(graph, node, native):
    weights = Tensor.from_document(graph["constants"][node["inputs"][1]])
    if len(weights.shape) != 2 or weights.encoding.axis != 1 or weights.encoding.block_size != 32:
        raise ValueError("shared pilot requires output-channel by K block-32 weights")
    channels, k = weights.shape
    fmt = format_named(weights.encoding.format)
    finite = sorted((Fraction(fmt.decode(c)), c) for c in range(1 << fmt.bits) if fmt.decode(c).is_finite())
    zero = min(finite, key=lambda x: (abs(x[0]), x[1]))[1]
    small = min((v, c) for v, c in finite if v > 0)[1]
    pattern = (zero, finite[0][1], finite[-1][1], small, finite[len(finite)//3][1])
    blocks = ceil(k/32)
    exponents = (-127, -8, 0, 8, 127)
    scales = tuple(Fraction(2**e) if e >= 0 else Fraction(1, 2**(-e)) for e in (exponents[i % len(exponents)] for i in range(blocks)))
    inputs = Tensor((1, k), tuple(pattern[i % len(pattern)] for i in range(k)),
                    Encoding(weights.encoding.format, scales, axis=1, block_size=32))
    biases = node["attrs"].get("bias")
    selected = sorted({0, channels-1, max(range(channels), key=lambda c: abs(real(biases[c]))) if biases else 0})
    rows = Tensor((len(selected), k), tuple(code for c in selected for code in weights.codes[c*k:(c+1)*k]),
                  Encoding(weights.encoding.format, tuple(scale for c in selected for scale in weights.encoding.scales[c*blocks:(c+1)*blocks]),
                           axis=1, block_size=32))
    started = perf_counter()
    left, right = inputs.values(), rows.values()
    preparation_seconds = perf_counter()-started
    accumulator = node["attrs"]["accumulator"]
    if accumulator != "fp64_e11m52_accumulator":
        raise ValueError("shared pilot bound requires declared FP64")
    acc = format_named(accumulator)
    started = perf_counter()
    actual = native.flex_gemm(left, right, batch=1, channels=len(selected), k=k, accumulator=accumulator)
    native_seconds = perf_counter()-started
    checks = []
    for ordinal, channel in enumerate(selected):
        pairs = list(zip(left, right[ordinal*k:(ordinal+1)*k]))
        expected = model_c(pairs, acc)
        bias = real(biases[channel]) if biases else None
        post = acc.decode(actual[ordinal])
        if bias is not None:
            post = acc.rounded(binary(post, acc.rounded(bias), "add"))
        reference_post = acc.decode(model_c(pairs, acc, bias=bias))
        exact = sum((Fraction(a)*Fraction(b) for a, b in pairs), Fraction(0))+(bias or 0)
        magnitude = sum((abs(Fraction(a)*Fraction(b)) for a, b in pairs), Fraction(0))
        bound = error_bound(magnitude, k, absolute_bias=None if bias is None else abs(bias))
        error = abs(Fraction(post)-exact)
        checks.append({"channel": channel, "native_accumulator_code": actual[ordinal], "reference_accumulator_code": expected,
                       "pre_store_accumulator_equal": actual[ordinal] == expected,
                       "post_bias_equal": acc.encode(post) == acc.encode(reference_post),
                       "exact_sum_plus_bias": str(exact), "post_bias_value": str(post),
                       "absolute_error": str(error), "absolute_error_bound": str(bound["absolute_error_bound"]),
                       "overflow_excluded_for_case": bound["overflow_excluded"], "within_case_error_bound": error <= bound["absolute_error_bound"]})
    return {"node": node["name"], "k": k, "blocks": blocks, "last_block_valid_values": k-32*(blocks-1),
            "input": inputs.document(), "selected_weights": rows.document(),
            "selected_bias": [biases[c] for c in selected] if biases else None,
            "accumulator": accumulator, "preparation_seconds": preparation_seconds,
            "native_seconds": native_seconds, "native_strategy": native.last_strategy, "checks": checks,
            "status": "matching_selected_rows" if all(c["pre_store_accumulator_equal"] and c["post_bias_equal"] and
                c["overflow_excluded_for_case"] and c["within_case_error_bound"] for c in checks) else "requires_diagnosis"}


def main():
    plan, source = campaign(), source_identity()
    native, records = NativeBackend("cpp"), []
    for key, prepared in preparation_records().items():
        if format_named(prepared["format"]).manifest["scaling"]["mode"] != "intrinsic_shared":
            continue
        config = read(checked(prepared["configuration"]))
        if config != configuration(prepared["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("shared MAC pilot requires current preparation")
        graph = read(checked(prepared["graph"]))
        macs = [n for n in graph["nodes"] if n["op"] in {"block_conv2d", "linear"}]
        longest = max(macs, key=lambda n: graph["constants"][n["inputs"][1]]["shape"][1])
        layers = []
        for node in {n["name"]: n for n in (macs[0], longest)}.values():
            print(key, node["name"], "checking shared row", flush=True)
            layers.append(check_layer(graph, node, native))
        records.append({"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"], "layers": layers})
    if source_identity() != source:
        raise ValueError("engine changed during shared native checks")
    report = {"schema_version": "phase3-shared-mac-pilot-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "selected_encoded_patches_only_not_acceptance", "records": records,
              "native_library": reference(ROOT / "build/phase2/cpp/libprecision_cpp.so"),
              "native_build": reference(ROOT / "build/phase2/cpp/libprecision_cpp.json"),
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/run/phase3_shared_mac_pilot.py", "public/analysis/phase3/fp64_bounds.py")],
              "limits": ["first and longest-K layers; first, last and largest-bias channels, deduplicated",
                         "explicit valid encoded patches alternate E8M0 scales including extremes; not observed image distributions",
                         "actual prepared weight block scales and full sequential K are retained, including final partial blocks",
                         "no block subtotal reset; native/reference states compared before output storage",
                         "bias uses the common Python path; intrinsic patch/output scale search and convolution geometry are not exercised",
                         "case-specific bounds do not cover all runtime scale assignments, CUDA or graph acceptance"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"shared-mac-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-shared-mac-pilot.json", {**report, "immutable_evidence": reference(path)})
    print(len(records), "shared configurations checked", flush=True)


if __name__ == "__main__":
    main()
