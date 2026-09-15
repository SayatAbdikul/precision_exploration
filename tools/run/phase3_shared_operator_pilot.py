"""Compare original and accelerated shared operators on one receptive field."""
from copy import deepcopy
import argparse
from fractions import Fraction
from math import ceil, prod
from time import perf_counter

from public.inference.conformance_job import source_identity
from public.inference.operators.dispatch import Operators
from public.inference.reference.arithmetic import format_named, real
from public.inference.reference.operators import pair
from public.inference.tensor import Encoding, Tensor, QUANTIZATION_OBSERVER, parse_encoding
from public.quantization.calibration import mse
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.phase3.shared_scale_dyadic import admitted, mse_scale_dyadic
from tools.phase3.shared_scale_pruned import mse_scale_pruned


def one_field(graph):
    node = next(n for n in graph["nodes"] if n["op"] == "block_conv2d")
    attrs = deepcopy(node["attrs"])
    weights = Tensor.from_document(graph["constants"][node["inputs"][1]])
    if attrs.get("groups", 1) != 1 or len(weights.shape) != 2 or weights.encoding.axis != 1 or weights.encoding.block_size != 32:
        raise ValueError("first shared convolution needs ordinary groups and K-owned blocks")
    co, k = weights.shape
    kh, kw = pair(attrs["kernel_size"])
    dh, dw = pair(attrs.get("dilation", 1))
    if k % (kh*kw):
        raise ValueError("first-layer K is inconsistent with kernel geometry")
    shape = (1, k//(kh*kw), (kh-1)*dh+1, (kw-1)*dw+1)
    fmt = format_named(weights.encoding.format)
    finite = sorted((Fraction(v), c) for c in range(1 << fmt.bits) if (v := fmt.decode(c)).is_finite())
    zero = min(finite, key=lambda v: (abs(v[0]), v[1]))[1]
    pattern = (zero, finite[0][1], finite[-1][1], finite[len(finite)//3][1])
    scale_count = ceil(shape[1]/32)*shape[2]*shape[3]
    scales = tuple((Fraction(1, 4), Fraction(1), Fraction(4))[i % 3] for i in range(scale_count))
    inputs = Tensor(shape, tuple(pattern[i % len(pattern)] for i in range(prod(shape))),
                    Encoding(weights.encoding.format, scales, axis=1, block_size=32))
    bias = attrs.get("bias")
    selected = sorted({0, co-1, max(range(co), key=lambda c: abs(real(bias[c]))) if bias else 0})
    blocks = ceil(k/32)
    rows = Tensor((len(selected), k), tuple(v for c in selected for v in weights.codes[c*k:(c+1)*k]),
                  Encoding(weights.encoding.format, tuple(s for c in selected for s in weights.encoding.scales[c*blocks:(c+1)*blocks]),
                           axis=1, block_size=32))
    if bias:
        attrs["bias"] = [bias[c] for c in selected]
    attrs["padding"] = 0
    return {"node": node["name"], "selected_channels": selected, "inputs": inputs.document(), "weights": rows.document(),
            "attrs": attrs, "geometry_change": "one complete receptive field, zero padding, selected output channels"}


def execute_case(case, backend, accelerated, *, search_fn=mse_scale_dyadic):
    traces, calls = [], []
    original = mse.mse_scale
    def search(values, name, **options):
        values = tuple(values)
        calls.append({"format": name, "elements": len(values), "admitted": all(admitted(Fraction(real(v))) for v in values)})
        return search_fn(values, name, **options)
    def observe(values, tensor):
        traces.append({"input_values_sha256": digest({"values": [str(v) for v in values]}), "stored": tensor.document()})
    token = QUANTIZATION_OBSERVER.set(observe)
    try:
        if accelerated:
            mse.mse_scale = search
        attrs = deepcopy(case["attrs"])
        attrs["output"] = parse_encoding(attrs["output"])
        operators = Operators(backend)
        started = perf_counter()
        result = operators.block_conv2d(Tensor.from_document(case["inputs"]), Tensor.from_document(case["weights"]), **attrs)
        seconds = perf_counter()-started
        return {"output": result.document(), "quantizer_trace": traces, "seconds": seconds, "accelerated_scale_calls": calls}
    finally:
        mse.mse_scale = original
        QUANTIZATION_OBSERVER.reset(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=("dyadic", "pruned"), default="dyadic")
    args = parser.parse_args()
    plan, source = campaign(), source_identity()
    benchmark_ref = read(ROOT / f"results/summaries/phase3-shared-{args.accelerator}-benchmark.json")["immutable_evidence"]
    benchmark = read(checked(benchmark_ref))
    if benchmark["status"] != "prototype_verified" or benchmark["source_sha256"] != source:
        raise ValueError("shared operator acceleration requires current equivalence evidence")
    for implementation in benchmark["implementations"]:
        archived = checked(implementation)
        if archived.name in {"shared_scale_dyadic.py", "shared_scale_fast.py", "shared_scale_pruned.py"} and reference(ROOT / "tools/phase3" / archived.name)["sha256"] != implementation["sha256"]:
            raise ValueError("scale accelerator changed after its benchmark")
    implementations = [source_reference(ROOT / p) for p in (
        "tools/run/phase3_shared_operator_pilot.py", "tools/phase3/shared_scale_dyadic.py", "tools/phase3/shared_scale_fast.py",
        "tools/phase3/shared_scale_pruned.py")]
    records = []
    for key, prepared in preparation_records().items():
        if format_named(prepared["format"]).manifest["scaling"]["mode"] != "intrinsic_shared":
            continue
        config = read(checked(prepared["configuration"]))
        if config != configuration(prepared["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("shared operator pilot requires current preparation")
        case = one_field(read(checked(prepared["graph"])))
        context = {"configuration": key, "configuration_artifact": prepared["configuration"], "graph": prepared["graph"],
                   "case": case, "source_sha256": source, "accelerator": args.accelerator,
                   "accelerator_benchmark": benchmark_ref, "implementations": implementations}
        path = ROOT / "artifacts/phase3/shared-operator-pilot" / f"{digest(context)}.json"
        if path.exists():
            record = read(path)
            if record["context"] != context or digest({k:v for k,v in record.items() if k != "record_sha256"}) != record["record_sha256"]:
                raise ValueError("shared operator checkpoint identity mismatch")
        else:
            print(key, "checking original reference/C++ and accelerated C++", flush=True)
            original_ref = execute_case(case, "reference", False)
            original_cpp = execute_case(case, "cpp", False)
            accelerated_cpp = execute_case(case, "cpp", True, search_fn=mse_scale_pruned if args.accelerator == "pruned" else mse_scale_dyadic)
            equal_outputs = original_ref["output"] == original_cpp["output"] == accelerated_cpp["output"]
            equal_traces = original_ref["quantizer_trace"] == original_cpp["quantizer_trace"] == accelerated_cpp["quantizer_trace"]
            record = {"context": context, "reference": original_ref, "cpp": original_cpp, "accelerated_cpp": accelerated_cpp,
                      "outputs_equal": equal_outputs, "quantizer_traces_equal": equal_traces,
                      "status": "matching_small_operator" if equal_outputs and equal_traces else "requires_diagnosis"}
            record["record_sha256"] = digest(record)
            write(path, record)
        records.append({"configuration": key, "evidence": reference(path), "status": record["status"],
                        "seconds": {name: record[name]["seconds"] for name in ("reference", "cpp", "accelerated_cpp")},
                        "accelerated_scale_calls": len(record["accelerated_cpp"]["accelerated_scale_calls"]),
                        "admitted_scale_calls": sum(c["admitted"] for c in record["accelerated_cpp"]["accelerated_scale_calls"])})
        if source_identity() != source:
            raise ValueError("engine changed during shared operator pilot")
    report = {"schema_version": "phase3-shared-operator-pilot-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "small_operator_coverage_only", "records": records, "accelerator": args.accelerator, "accelerator_benchmark": benchmark_ref,
              "implementations": implementations,
              "native_library": reference(ROOT / "build/phase2/cpp/libprecision_cpp.so"),
              "native_build": reference(ROOT / "build/phase2/cpp/libprecision_cpp.json"),
              "limits": ["first convolution uses one receptive field and selected actual weight rows; no native-resolution claim",
                         "original reference and C++ retain the oracle search; acceleration is installed only inside this diagnostic process",
                         "all intermediate quantizer inputs/stored tensors and final output encodings are compared",
                         "per-case timings include intrinsic patch/output scale searches; no production speedup or graph acceptance"]}
    output_name = "phase3-shared-operator-pilot.json" if args.accelerator == "dyadic" else "phase3-shared-operator-pruned-pilot.json"
    write(ROOT / "results/summaries" / output_name, report)
    print(len(records), "small shared operators retained", flush=True)


if __name__ == "__main__":
    main()
