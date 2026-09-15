"""Retain native-resolution shared-format diagnostics with an explicit accelerator.

This separate pilot does not modify the frozen registry runner or issue screen
acceptance. The source graph, accelerator proof, image records and layer
diagnostics are retained together for subsequent review.
"""
import argparse
import fcntl
from time import perf_counter

import numpy as np

from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
from public.analysis.phase3.statistics import paired_classification
from public.inference.conformance_job import source_identity
from public.inference.tensor import SharedEncoding, Tensor, QUANTIZATION_OBSERVER, parse_encoding
from public.quantization.calibration import mse
from public.quantization.graph.executable import execute, graph_sha256
from tools.phase3.baselines import screen_rows
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.phase3.runtime import load_runtime
from tools.phase3.shared_scale_pruned import mse_scale_pruned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "mobilenet_v2", "mobilenet_v3_large"), required=True)
    parser.add_argument("--format", choices=("bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3"), required=True)
    parser.add_argument("--images", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.images <= 8:
        raise ValueError("diagnostic native pilots require 1–8 images")
    plan, source = campaign(), source_identity()
    prepared = preparation_records()[f"{args.model}/{args.format}"]
    config = read(checked(prepared["configuration"]))
    if config != configuration(args.model, args.format, plan) or config["runtime"]["source_sha256"] != source:
        raise ValueError("shared native pilot requires current preparation")
    graph = read(checked(prepared["graph"]))
    benchmark_ref = read(ROOT / "results/summaries/phase3-shared-pruned-benchmark.json")["immutable_evidence"]
    benchmark = read(checked(benchmark_ref))
    if benchmark["status"] != "prototype_verified" or benchmark["source_sha256"] != source or benchmark["campaign_sha256"] != digest(plan):
        raise ValueError("native pilot requires current scale equivalence evidence")
    helper_paths = ("tools/phase3/shared_scale_pruned.py", "tools/phase3/shared_scale_dyadic.py", "tools/phase3/shared_scale_fast.py")
    expected_helpers = {checked(ref).name: ref["sha256"] for ref in benchmark["implementations"]}
    if any(reference(ROOT / p)["sha256"] != expected_helpers.get(p.rsplit("/", 1)[-1]) for p in helper_paths):
        raise ValueError("scale accelerator changed after its benchmark")
    operator_ref = reference(ROOT / "results/summaries/phase3-shared-operator-pruned-pilot.json")
    operators = read(checked(operator_ref))
    if operators["source_sha256"] != source or operators["accelerator_benchmark"] != benchmark_ref:
        raise ValueError("small operator evidence is stale")
    operator = next(r for r in operators["records"] if r["configuration"] == f"{args.model}/{args.format}")
    if operator["status"] != "matching_small_operator":
        raise ValueError("small shared operator has an unresolved mismatch")
    checked(operator["evidence"])
    baseline_ref = read(ROOT / "results/summaries/phase3-baselines.json")["models"][args.model]
    baseline = read(checked(baseline_ref))
    paths = ("tools/run/phase3_shared_native_pilot.py", "tools/phase3/runtime.py", "public/analysis/phase3/diagnostics.py",
             "public/analysis/phase3/statistics.py", *helper_paths)
    implementations = [source_reference(ROOT / p) for p in paths]
    job = {"schema_version": "phase3-shared-native-diagnostic-1.0.0", "scope": "diagnostic_accelerated_native",
           "model": args.model, "format": args.format, "images": args.images, "backend": "cpp", "source_sha256": source,
           "campaign_sha256": digest(plan), "configuration": prepared["configuration"], "graph": prepared["graph"],
           "baseline": baseline_ref, "accelerator_benchmark": benchmark_ref, "small_operator_evidence": operator_ref,
           "implementations": implementations}
    identity = digest(job)
    work = ROOT / "artifacts/phase3/shared-native-pilot" / identity
    lock_path = ROOT / "artifacts/phase3/locks/cpu-shared-native-pilot.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(work / "job.json", job)
        population, payload, _ = screen_rows(args.model, plan, ROOT)
        sample, observe, manifest = load_runtime(args.model, plan, ROOT)
        input_name = next(iter(graph["inputs"]))
        encoding = parse_encoding(graph["inputs"][input_name])
        if not isinstance(encoding, SharedEncoding):
            raise ValueError("native shared pilot needs intrinsic input scaling")
        records, image_refs = [], []
        for ordinal, row in enumerate(population[:args.images]):
            paired = baseline["records"][ordinal]
            if paired["sample_sha256"] != row["sha256"]:
                raise ValueError("shared native image order differs from baseline")
            path = work / f"{row['sha256']}.json"
            if path.exists():
                record = read(path)
                if (record["job_sha256"] != identity or record["sample"] != row or record["paired"] != paired
                        or digest({k:v for k,v in record.items() if k != "record_sha256"}) != record["record_sha256"]):
                    raise ValueError("shared native checkpoint identity mismatch")
            else:
                started = perf_counter()
                fp32, _ = sample(payload / row["relative_path"])
                if list(fp32.shape) != manifest["input_shape"]:
                    raise ValueError("native shared preprocessing shape mismatch")
                references = ReferenceSamples(plan["diagnostics"]["sample_elements_per_layer_image"])
                observe(fp32, references)
                fp32_seconds = perf_counter()-started
                counts = {"calls": 0, "fallback_calls": 0, "evaluated_scales": 0, "excluded_scales": 0}
                node_state = {"current": "input", "completed": 0}
                original = mse.mse_scale
                def search(values, name, **options):
                    details = {}
                    result = mse_scale_pruned(values, name, diagnostics=details, **options)
                    counts["calls"] += 1
                    if details["fallback"]:
                        counts["fallback_calls"] += 1
                    else:
                        counts["evaluated_scales"] += details["evaluated_scales"]
                        counts["excluded_scales"] += 255-details["evaluated_scales"]
                    return result
                class Diagnostics(LayerDiagnostics):
                    def begin_node(self, node):
                        node_state["current"] = node["name"]
                        super().begin_node(node)
                    def __call__(self, node, tensor):
                        super().__call__(node, tensor)
                        node_state["completed"] += 1
                        write(work / "progress.json", {"job_sha256": identity, "image": ordinal+1, **node_state,
                              "elapsed_seconds": perf_counter()-started, "scale_search": dict(counts)})
                        print(f"image {ordinal+1}: {node['name']} saved diagnostics; {counts['calls']} scale searches", flush=True)
                diagnostics = Diagnostics(references)
                token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
                try:
                    mse.mse_scale = search
                    preparation_start = perf_counter()
                    tensor = Tensor.quantize(fp32.numpy().ravel().tolist(), tuple(fp32.shape), encoding)
                    input_seconds = perf_counter()-preparation_start
                    inference_start = perf_counter()
                    result = execute(graph, {input_name: tensor}, backend="cpp", observer=diagnostics)
                    inference_seconds = perf_counter()-inference_start
                except Exception as error:
                    write(work / "failure.json", {"job_sha256": identity, "sample": row, **node_state,
                          "error": f"{type(error).__name__}: {error}", "scale_search": counts})
                    raise
                finally:
                    mse.mse_scale = original
                    QUANTIZATION_OBSERVER.reset(token)
                if result["graph_sha256"] != graph_sha256(graph) or set(diagnostics.records) != {n["name"] for n in graph["nodes"]}:
                    raise ValueError("shared native graph or diagnostic coverage mismatch")
                output = next(iter(result["outputs"].values()))
                values = output.values()
                if not all(np.isfinite(float(v)) for v in values):
                    raise ValueError("shared native output contains nonfinite values")
                prediction = sorted(range(len(values)), key=lambda i: (-values[i], i))[:5]
                record = {"job_sha256": identity, "sample": row, "paired": paired, "graph_sha256": result["graph_sha256"],
                          "prediction": prediction, "output_sha256": digest(output.document()), "output": output.document(),
                          "layers": result["layers"], "diagnostics": diagnostics.records, "execution_modes": result["execution_modes"],
                          "scale_search": counts, "timings_seconds": {"fp32_observation": fp32_seconds,
                              "input_encoding": input_seconds, "inference_with_diagnostics": inference_seconds,
                              "total": perf_counter()-started}}
                record["record_sha256"] = digest(record)
                write(path, record)
            if source_identity() != source or any(reference(ROOT / p)["sha256"] != ref["sha256"] for p, ref in zip(paths, implementations)):
                raise ValueError("shared native pilot sources changed")
            records.append(record)
            image_refs.append(reference(path))
        pairs = [{**r["paired"], "candidate_prediction": r["prediction"]} for r in records]
        statistics = paired_classification(pairs, [r["sample_id"] for r in pairs],
            **{k:plan["statistics"][k] for k in ("confidence", "seed")}, resamples=plan["statistics"]["classification_resamples"])
        report = {"schema_version": "phase3-shared-native-diagnostic-summary-1.0.0", "status": "completed", "scope": job["scope"],
                  "job": reference(work / "job.json"), "images": len(records), "image_records": image_refs, "statistics": statistics,
                  "classification": "PILOT_ONLY", "limits": ["native-resolution C++ diagnostic with explicitly identified exact scale-search acceleration",
                      "full original graph/encoded weights and frozen images; no current production runner mutation",
                      "layer diagnostics and per-image checkpoints retained; per-layer progress is not an inference-resume checkpoint",
                      "C++/CUDA image agreement and shared accumulator acceptance remain pending; not a fixed-1k screen or D4 label"]}
        write(work / "summary.json", report)
        write(ROOT / f"results/summaries/phase3-shared-native-{args.model}-{args.format}-{args.images}.json", report)
        print(statistics, flush=True)


if __name__ == "__main__":
    main()
