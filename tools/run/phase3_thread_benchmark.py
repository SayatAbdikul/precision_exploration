"""Isolated native-image CPU/CUDA thread benchmark; never a screening result.

Fresh processes set and verify thread counts after the frozen FP32 loader has
run. This diagnostic override is explicit in the job and reports. Production
sources, registry jobs, accepted pilots and completed screens remain unchanged.
"""
import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter, process_time

from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write


def summarize(records):
    groups, expected = {}, {}
    for row in records:
        sample = row["sample_sha256"]
        signature = (row["output_sha256"], row["layers_sha256"])
        if sample in expected and signature != expected[sample]:
            raise ValueError("thread/backend results disagree")
        expected[sample] = signature
        groups.setdefault((row["backend"], row["threads"]), []).append(row)
    result = []
    for (backend, threads), rows in sorted(groups.items()):
        if len({r["sample_sha256"] for r in rows}) != len(rows):
            raise ValueError("duplicate image within a thread/backend group")
        if {r["sample_sha256"] for r in rows} != set(expected):
            raise ValueError("thread/backend groups must use identical images")
        seconds = [r["inference_with_diagnostics_seconds"] for r in rows]
        result.append({"backend": backend, "threads": threads, "images": len(rows),
                       "seconds_per_image": seconds, "mean_seconds": statistics.mean(seconds),
                       "median_seconds": statistics.median(seconds), "min_seconds": min(seconds),
                       "max_seconds": max(seconds),
                       "mean_native_dispatch_seconds": statistics.mean(r["native_dispatch_seconds"] for r in rows),
                       "mean_store_seconds": statistics.mean(r["store_seconds"] for r in rows),
                       "mean_diagnostic_seconds": statistics.mean(r["diagnostic_seconds"] for r in rows),
                       "mean_cpu_seconds": statistics.mean(r["cpu_seconds"] for r in rows)})
    return sorted(result, key=lambda row: row["mean_seconds"])


def probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"command": command, "returncode": result.returncode, "output": result.stdout + result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def set_threads(count):
    import torch
    torch.set_num_threads(count)
    pools = []
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "/" in line and any(name in line for name in ("libgomp", "libomp.so", "libiomp"))}
    for path in sorted(paths):
        lib = ctypes.CDLL(path)
        lib.omp_set_dynamic.argtypes = [ctypes.c_int]
        lib.omp_set_num_threads.argtypes = [ctypes.c_int]
        lib.omp_get_max_threads.restype = ctypes.c_int
        lib.omp_set_dynamic(0)
        lib.omp_set_num_threads(count)
        effective = lib.omp_get_max_threads()
        if effective != count:
            raise ValueError("OpenMP thread setting was not applied")
        pools.append({"library": path, "max_threads": effective})
    if not pools or torch.get_num_threads() != count:
        raise ValueError("effective CPU thread counts could not be verified")
    return {"torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(), "openmp": pools,
            "environment": {key: os.environ.get(key) for key in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_DYNAMIC")}}


def trial(plan_path, ordinal):
    import torch
    from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
    from public.inference.conformance_job import source_identity
    from public.inference.native import NativeBackend
    from public.inference.operators.dispatch import Operators
    from public.inference.tensor import QUANTIZATION_OBSERVER, SharedEncoding, parse_encoding
    from public.quantization.graph.executable import execute
    from public.quantization.ptq.encoding import float_tensor
    from tools.phase3.baselines import screen_rows
    from tools.phase3.evidence import verify_complete
    from tools.phase3.runtime import load_runtime
    from tools.phase3.screening import pipeline_identity

    plan = read(plan_path)
    setting = plan["settings"][ordinal]
    backend, threads = setting["backend"], setting["threads"]
    if (source_identity() != plan["source_sha256"] or pipeline_identity() != plan["pipeline_sha256"] or
            reference(__file__)["sha256"] != plan["implementation"]["sha256"]):
        raise ValueError("benchmark source identity changed")
    frozen = campaign()
    if digest(frozen) != plan["campaign_sha256"]:
        raise ValueError("benchmark campaign changed")
    prepared = read(checked(plan["prepared"]))
    graph = read(checked(prepared["graph"]))
    _, _, (_, _, accepted_graph, _, accepted_records, _) = verify_complete(checked(plan["accepted_pilot"]), current_execution=True)
    if graph != accepted_graph:
        raise ValueError("benchmark graph differs from accepted pilot")
    population, payload, _ = screen_rows(plan["model"], frozen, ROOT)
    torch.set_num_interop_threads(1)
    sample, observe, manifest = load_runtime(plan["model"], frozen)
    native = NativeBackend(backend)
    if reference(ROOT / f"build/phase2/{backend}/libprecision_{backend}.so") != plan["libraries"][backend]:
        raise ValueError("native library changed")
    # The loader sets four threads. Override in this diagnostic process only,
    # after loading all native runtimes, and record the effective settings.
    effective = set_threads(threads)
    warm_start = perf_counter()
    warm = native.flex_gemm([1] * 64, [1] * 64, batch=1, channels=1, k=64, accumulator="int64_accumulator")
    if tuple(warm) != (64,):
        raise ValueError("native context warmup returned an incorrect result")
    warm_seconds = perf_counter() - warm_start
    input_name = next(iter(graph["inputs"]))
    encoding = parse_encoding(graph["inputs"][input_name])
    if isinstance(encoding, SharedEncoding):
        raise ValueError("thread benchmark currently requires ordinary encoded inputs")
    work = plan_path.parent
    rows = []
    for image_index, selected in enumerate(population[:plan["images"]]):
        reference_record = accepted_records[image_index]
        if reference_record["sample"] != selected:
            raise ValueError("benchmark images differ from accepted pilot")
        start = perf_counter()
        fp32, _ = sample(payload / selected["relative_path"])
        if list(fp32.shape) != manifest["input_shape"]:
            raise ValueError("benchmark preprocessing shape mismatch")
        references = ReferenceSamples(frozen["diagnostics"]["sample_elements_per_layer_image"])
        observe(fp32, references)
        tensor = float_tensor(fp32.numpy(), encoding)
        preparation_seconds = perf_counter() - start
        counters = {"native_dispatch_seconds": 0.0, "store_seconds": 0.0, "diagnostic_seconds": 0.0}
        # Inclusive component timers only; diagnostics inside stores overlap.
        # flex_* timing includes host admission/marshalling, not GPU-only time.
        originals = []
        def instrument(cls, name, counter):
            original = getattr(cls, name)
            def wrapped(self, *args, **kwargs):
                begin = perf_counter()
                try:
                    return original(self, *args, **kwargs)
                finally:
                    counters[counter] += perf_counter() - begin
            originals.append((cls, name, original))
            setattr(cls, name, wrapped)
        class TimedDiagnostics(LayerDiagnostics):
            def quantization(self, *args, **kwargs):
                begin = perf_counter()
                try:
                    return super().quantization(*args, **kwargs)
                finally:
                    counters["diagnostic_seconds"] += perf_counter() - begin
            def __call__(self, node, value):
                begin = perf_counter()
                try:
                    return super().__call__(node, value)
                finally:
                    counters["diagnostic_seconds"] += perf_counter() - begin
        diagnostics = TimedDiagnostics(references)
        token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
        instrument(NativeBackend, "flex_gemm", "native_dispatch_seconds")
        instrument(NativeBackend, "flex_conv2d", "native_dispatch_seconds")
        instrument(Operators, "_store", "store_seconds")
        print(f"{backend} threads={threads} image={image_index+1}/{plan['images']} executing", flush=True)
        gpu_before = probe(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,power.draw", "--format=csv"])
        begin, cpu_begin = perf_counter(), process_time()
        try:
            result = execute(graph, {input_name: tensor}, backend=backend, observer=diagnostics)
        finally:
            elapsed, cpu_seconds = perf_counter() - begin, process_time() - cpu_begin
            QUANTIZATION_OBSERVER.reset(token)
            for cls, name, original in reversed(originals):
                setattr(cls, name, original)
        output = next(iter(result["outputs"].values()))
        expected = reference_record["backends"][backend]
        if result["layers"] != expected["layers"] or digest(output.document()) != expected["output_sha256"]:
            raise ValueError("benchmark output differs from the accepted native pilot")
        if set(diagnostics.records) != set(result["layers"]):
            raise ValueError("benchmark diagnostics are incomplete")
        row = {"backend": backend, "threads": threads, "image_index": image_index,
               "sample_sha256": selected["sha256"], "input_sha256": digest(tensor.document()),
               "output_sha256": digest(output.document()), "layers_sha256": digest(result["layers"]),
               "matches_accepted_pilot": True, "diagnostic_layers": len(diagnostics.records),
               "inference_with_diagnostics_seconds": elapsed, "cpu_seconds": cpu_seconds,
               "preparation_seconds": preparation_seconds, "native_warmup_seconds": warm_seconds,
               "thread_settings": effective, "gpu_before": gpu_before, **counters}
        write(work / f"{ordinal:02d}-{image_index:02d}.json", row)
        rows.append(row)
        print(f"{backend} threads={threads} image={image_index+1}: {elapsed:.3f}s; native dispatch "
              f"{counters['native_dispatch_seconds']:.3f}s; stores {counters['store_seconds']:.3f}s; "
              f"diagnostics {counters['diagnostic_seconds']:.3f}s; output MATCH", flush=True)
    if source_identity() != plan["source_sha256"] or pipeline_identity() != plan["pipeline_sha256"]:
        raise ValueError("production source identity changed during benchmark")
    write(work / f"{ordinal:02d}-completed.json", {"setting": setting, "records": rows})


def run(args):
    from public.inference.conformance_job import source_identity
    from tools.phase3.evidence import verify_complete
    from tools.phase3.preparation_inventory import preparation_records
    from tools.phase3.provenance import source_reference
    from tools.phase3.screening import pipeline_identity
    from contextlib import ExitStack

    if not 1 <= args.images <= 8 or not args.threads or any(n < 1 for n in args.threads) or len(set(args.threads)) != len(args.threads):
        raise ValueError("use 1–8 frozen images and unique positive thread counts")
    with ExitStack() as stack:
        for name in ("native-worker.lock", "cpu-native-worker.lock"):
            path = ROOT / "artifacts/phase3/locks" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = campaign()
        prepared = preparation_records()[f"{args.model}/{args.format}"]
        config = read(checked(prepared["configuration"]))
        acceptance = read(ROOT / "artifacts/phase3/acceptance" / prepared["configuration_sha256"] / "acceptance.json")
        verify_complete(checked(acceptance["pilot"]), current_execution=True)
        if config["runtime"]["source_sha256"] != source_identity():
            raise ValueError("benchmark preparation is stale")
        settings = [{"backend": backend, "threads": n} for backend in ("cpp", "cuda") for n in args.threads]
        random.Random(310920).shuffle(settings)
        job = {"schema_version": "phase3-thread-benchmark-1.0.0", "scope": "diagnostic_thread_benchmark",
               "model": args.model, "format": args.format, "images": args.images, "settings": settings,
               "source_sha256": source_identity(), "pipeline_sha256": pipeline_identity(), "campaign_sha256": digest(plan),
               "prepared": reference(checked(prepared["configuration"]).parent / "prepared.json"),
               "accepted_pilot": acceptance["pilot"], "implementation": source_reference(__file__),
               "libraries": {b: reference(ROOT / f"build/phase2/{b}/libprecision_{b}.so") for b in ("cpp", "cuda")},
               "machine": {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
                           "cpu": probe(["lscpu"]), "gpu": probe(["nvidia-smi"])},
               "thread_policy": "fresh process; environment and explicit torch/OpenMP counts; interop=1; OMP_DYNAMIC=FALSE",
               "timing_policy": "native-resolution execute with frozen diagnostics; initialization and FP32/input preparation separate; tiny native context warmup only"}
        work = ROOT / "artifacts/phase3/thread-benchmark" / digest(job)
        write(work / "job.json", job)
        print("Benchmark:", work, flush=True)
        rows = []
        for index, setting in enumerate(settings):
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_DYNAMIC": "FALSE"}
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[key] = str(setting["threads"])
            print("Starting", index, setting, flush=True)
            subprocess.run([sys.executable, "-m", "tools.run.phase3_thread_benchmark", "--trial", str(work / "job.json"),
                            "--ordinal", str(index)], env=env, cwd=ROOT, check=True)
            rows.extend(read(work / f"{index:02d}-completed.json")["records"])
            write(work / "progress.json", {"completed_settings": index+1, "planned_settings": len(settings), "records": rows})
        if source_identity() != job["source_sha256"] or pipeline_identity() != job["pipeline_sha256"]:
            raise ValueError("production sources changed during benchmark")
        report = {"schema_version": "phase3-thread-benchmark-summary-1.0.0", "status": "completed", "scope": job["scope"],
                  "job": reference(work / "job.json"), "rows": summarize(rows), "records": rows,
                  "limits": ["small identical-image timing comparison, not quality screening or graph acceptance",
                             "first inference in each process has cold graph caches; subsequent images use warm caches",
                             "native dispatch includes CPU admission/marshalling; not GPU-only kernel timing",
                             "store and diagnostic timers overlap and must not be added as disjoint costs",
                             "FP32 observations use the requested thread count; exact candidate layers/output must match accepted pilot",
                             "environment changes are confined to benchmark subprocesses; production runtime still defaults to four threads"]}
        write(work / "summary.json", report)
        write(ROOT / "results/summaries/phase3-thread-benchmark.json", report)
        for row in report["rows"]:
            print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mobilenet_v2")
    parser.add_argument("--format", default="int8")
    parser.add_argument("--images", type=int, default=2)
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--ordinal", type=int)
    args = parser.parse_args()
    if args.trial:
        trial(args.trial, args.ordinal)
    else:
        run(args)


if __name__ == "__main__":
    main()
