"""Build, exercise, and benchmark Phase 2 exact backends on synthetic graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from public.experiments.registry import ExperimentRegistry
from public.experiments.registry.identity import canonical_json_bytes, experiment_sha256
from public.experiments.registry.artifacts import ArtifactStore
from public.experiments.scheduler import LocalScheduler
from public.inference.native import build, NativeBackend, prepare_tensor
from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import Encoding, Tensor
from public.inference.conformance_job import SCHEMA, source_identity, run_job
from public.quantization.graph.executable import execute, freeze_graph

ROOT = Path(__file__).resolve().parents[2]


def graph_witness():
    encoding = Encoding("fp6_e3m2")
    weights = Tensor.quantize([1, -1], (2, 1, 1, 1), encoding)
    graph = freeze_graph(inputs={"x": encoding.document()}, constants={"w": weights}, nodes=[
        {"name": "conv", "op": "conv2d", "inputs": ["x", "w"], "attrs": {
            "accumulator": "fp16_e5m10_accumulator", "output": encoding.document(), "bias": ["0.5", "-0.5"], "activation": "relu"}},
        {"name": "pool", "op": "adaptive_average_pool2d", "inputs": ["conv"], "attrs": {
            "accumulator": "fp16_e5m10_accumulator", "output": encoding.document(), "output_size": 1}},
        {"name": "logits", "op": "flatten", "inputs": ["pool"], "attrs": {"start_dim": 1}}],
        outputs=["logits"], provenance={"kind": "synthetic_conformance"})
    inputs = {"x": Tensor.quantize([1, -2, 3, 4], (1, 1, 2, 2), encoding)}
    return graph, inputs


def conformance(backend):
    graph, inputs = graph_witness()
    configuration = {"schema_version": SCHEMA, "purpose": "engine_conformance", "graph": graph,
        "inputs": {name: value.document() for name, value in inputs.items()},
        "runtime": {"backend": backend, "semantic_version": "2.0.0", "source_sha256": source_identity()}}
    store = ArtifactStore(ROOT / "artifacts/encoded_tensors/phase2")
    with ExperimentRegistry(ROOT / "results/databases/phase2.sqlite") as registry:
        run_id, status = registry.submit(configuration)
        if status not in {"PENDING", "COMPLETED"}:
            raise RuntimeError(f"conformance job did not submit: {status}")
        if status == "PENDING":
            scheduler = LocalScheduler(registry, worker_id=f"phase2-{backend}")
            scheduler.run_once(run_job)
        repeated_id, status = registry.submit(configuration)
        if status != "COMPLETED" or repeated_id != run_id:
            raise RuntimeError(f"conformance execution failed: {status}; inspect phase2.sqlite")
        result = execute(graph, inputs, backend=backend)
        payload = {**result, "outputs": {name: tensor.document() for name, tensor in result["outputs"].items()}}
        artifact = store.put_bytes(canonical_json_bytes(payload), semantic_type="phase2_conformance_layer_outputs")
        experiment_id = experiment_sha256(configuration)
        registry.register_artifact(artifact, producer_experiment_id=experiment_id,
                                   metadata={"graph_sha256": result["graph_sha256"], "backend": backend})
        registry.attach_artifact(run_id, artifact.sha256, "conformance_outputs")
    return {"experiment_id": experiment_id, "run_id": run_id, "status": status, "deduplicated": True,
            "output_sha256": artifact.sha256, "output_path": str(Path(artifact.path).relative_to(ROOT)), "layers": result["layers"]}


def benchmark(backend, repeats):
    if backend == "reference":
        raise ValueError("microbenchmark requires cpp or cuda")
    native, rng, records = NativeBackend(backend), random.Random(201), []
    for name in ("fp6_e3m2", "int8"):
        fmt = format_named(name)
        finite = [c for c in range(1 << fmt.bits) if fmt.decode(c).is_finite()]
        for label, batch, channels, k in (("conv_gemm", 196, 64, 288), ("pointwise_gemm", 784, 32, 32)):
            left = [rng.choice(finite) for _ in range(batch*k)]
            right = [rng.choice(finite) for _ in range(channels*k)]
            prepared_at = time.perf_counter()
            lv = prepare_tensor(Tensor((batch,k), tuple(left), Encoding(name)))
            rv = prepare_tensor(Tensor((channels,k), tuple(right), Encoding(name)))
            preparation_seconds = time.perf_counter() - prepared_at
            args = dict(batch=batch, channels=channels, k=k, accumulator="int32_accumulator" if name == "int8" else "fp32_e8m23_accumulator")
            expected = native.gemm(lv, rv, **args)
            for strategy in ("algorithmic", "lookup", "predecoded"):
                def invoke():
                    if strategy == "predecoded":
                        return native.gemm(lv, rv, **args)
                    return native.encoded_gemm(left, right, activation_format=name, weight_format=name, strategy=strategy, **args)
                if invoke() != expected:
                    raise RuntimeError("benchmark strategy changes numerical output")
                durations = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    actual = invoke()
                    durations.append(time.perf_counter() - start)
                    if actual != expected:
                        raise RuntimeError("benchmark execution is nondeterministic")
                elapsed = statistics.median(durations)
                records.append({"format": name, "shape_role": label, "batch": batch, "channels": channels, "k": k,
                                "strategy": strategy, "median_seconds": elapsed, "mac_per_second": batch*channels*k/elapsed,
                                "samples_seconds": durations, "accumulator": args["accumulator"],
                                "predecode_setup_seconds": preparation_seconds if strategy == "predecoded" else None,
                                "output_sha256": hashlib.sha256(canonical_json_bytes({"codes": list(actual)})).hexdigest()})
        geometry = dict(n=1, ci=32, h=28, w=28, co=32, kh=3, kw=3, oh=28, ow=28,
                        sh=1, sw=1, ph=1, pw=1, dh=1, dw=1, groups=32)
        inputs = prepare_tensor(Tensor((1,32,28,28), tuple(rng.choice(finite) for _ in range(32*28*28)), Encoding(name)))
        weights = prepare_tensor(Tensor((32,1,3,3), tuple(rng.choice(finite) for _ in range(32*9)), Encoding(name)))
        accumulator = "int32_accumulator" if name == "int8" else "fp32_e8m23_accumulator"
        expected = native.conv2d(inputs, weights, geometry=geometry, accumulator=accumulator)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            actual = native.conv2d(inputs, weights, geometry=geometry, accumulator=accumulator)
            durations.append(time.perf_counter()-start)
            if actual != expected:
                raise RuntimeError("depthwise benchmark is nondeterministic")
        elapsed = statistics.median(durations)
        records.append({"format": name, "shape_role": "depthwise_conv", "geometry": geometry,
                        "strategy": "predecoded", "median_seconds": elapsed,
                        "mac_per_second": 32*28*28*9/elapsed, "samples_seconds": durations,
                        "accumulator": accumulator,
                        "output_sha256": hashlib.sha256(canonical_json_bytes({"codes": list(actual)})).hexdigest()})
    return {"scope": "synthetic layer-sized shapes; host binding/allocations/transfers included; predecoded tensors reused; no network runtime prediction",
            "D2": "open; occupancy, bandwidth and network-scale profiling still required", "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "conformance", "benchmark"))
    parser.add_argument("--backend", choices=("reference", "cpp", "cuda"), default="cpp")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.action == "build":
        print(build(args.backend))
        return
    execution_source = source_identity()
    result = conformance(args.backend) if args.action == "conformance" else benchmark(args.backend, args.repeats)
    if source_identity() != execution_source:
        raise RuntimeError("engine sources changed during execution; rerun before publishing evidence")
    report = {"schema_version": "2.0.0", "backend": args.backend, "source_sha256": execution_source, "result": result}
    output = args.output or ROOT / "results/summaries" / f"phase2-{args.action}-{args.backend}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.action} {args.backend}: {output}")


if __name__ == "__main__":
    main()
