"""Prepare shared-format graphs with durable block-scale search checkpoints."""
import argparse
import fcntl
from pathlib import Path

from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from public.quantization.calibration.mse import mse_scale
from tools.phase3.common import ROOT, campaign, checked, file_hash, read, write
from tools.phase3.graphs import build
from tools.phase3.provenance import source_reference
from tools.phase3.shared_scale_cache import SharedScaleCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fast-fp32", action="store_true", help="use the verified exact integer-grid scale evaluator for admitted FP32 weights")
    args = parser.parse_args()
    plan, source = campaign(), source_identity()
    if args.model not in plan["models"] or any(name not in plan["formats"] or
            format_named(name).manifest["scaling"]["mode"] != "intrinsic_shared" for name in args.formats):
        raise ValueError("cached preparation only admits frozen shared-format candidates")
    implementations = [source_reference(ROOT / path) for path in (
        "tools/run/phase3_shared_graphs.py", "tools/phase3/shared_scale_cache.py")]
    acceleration_evidence = None
    if args.fast_fp32:
        benchmark_path = ROOT / "results/summaries/phase3-shared-scale-benchmark.json"
        benchmark = read(benchmark_path)
        expected = file_hash(ROOT / "tools/phase3/shared_scale_fast.py")
        if (benchmark["status"] != "verified_equivalent" or benchmark["source_sha256"] != source
                or not any(item["sha256"] == expected for item in benchmark["implementations"])
                or {r["format"] for r in benchmark["records"]} != {"bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"}
                or any(not r["responses_identical"] or r["blocks"] < 64 for r in benchmark["records"])):
            raise ValueError("exact FP32 acceleration lacks matching validated benchmark evidence")
        for item in benchmark["implementations"]:
            checked(item)
        implementations.append(source_reference(ROOT / "tools/phase3/shared_scale_fast.py"))
        acceleration_evidence = source_reference(benchmark_path)
    lock_path = ROOT / "artifacts/phase3/locks" / f"shared-preparation-{args.model}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = read(args.report) if args.report.exists() else {"schema_version": "phase3-preparation-1.0.0", "records": {}}
        for name in args.formats:
            key = f"{args.model}/{name}"
            cache_path = ROOT / "artifacts/phase3/shared-scale-cache" / source / f"{name}.sqlite"
            progress_path = ROOT / "artifacts/phase3/preparation-progress" / f"{args.model}-{name}.json"
            def progress(stats):
                write(progress_path, {"configuration": key, "cache": str(cache_path.relative_to(ROOT)),
                                      "implementations": implementations, **stats})
                print(f"{key}: {stats['hits']} cached blocks reused, {stats['misses']} new blocks saved", flush=True)
            cache = SharedScaleCache(cache_path, source_sha256=source, oracle=mse_scale, progress=progress, fast_fp32=args.fast_fp32)
            print(f"preparing {key} with durable shared-scale cache", flush=True)
            try:
                with cache.installed():
                    result = build(args.model, name, plan)
                if source_identity() != source:
                    raise ValueError("engine changed during shared preparation")
                report["records"][key] = result
            except Exception as error:
                report["records"][key] = {"status": "preparation_failed", "error": f"{type(error).__name__}: {error}"}
            finally:
                progress(cache.stats())
                cache.close()
            report["preparation_implementations"] = implementations
            if acceleration_evidence is not None:
                report["acceleration_evidence"] = acceleration_evidence
            write(args.report, report)
            print(key, report["records"][key]["status"], flush=True)


if __name__ == "__main__":
    main()
