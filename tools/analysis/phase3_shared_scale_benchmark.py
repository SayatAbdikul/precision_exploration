"""Validate and time exact FP32 scale search on retained preparation blocks."""
from fractions import Fraction
import json
import sqlite3
from time import perf_counter

from public.inference.conformance_job import source_identity
from public.quantization.calibration.mse import mse_scale
from tools.phase3.common import ROOT, campaign, digest, write
from tools.phase3.provenance import source_reference
from tools.phase3.shared_scale_fast import admitted, mse_scale_fp32


def main():
    source = source_identity()
    path = ROOT / "artifacts/phase3/shared-scale-cache" / source / "bfp6.sqlite"
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = list(connection.execute("SELECT identity,request,response,response_sha256 FROM scales ORDER BY identity LIMIT 64"))
    finally:
        connection.close()
    if len(rows) != 64:
        raise ValueError("benchmark needs 64 completed preparation blocks")
    blocks = []
    for identity, request, response, expected_hash in rows:
        request, response = json.loads(request), json.loads(response)
        if digest(request) != identity or request["source_sha256"] != source or digest(response) != expected_hash:
            raise ValueError("invalid cached benchmark input provenance")
        values = [Fraction(v) for v in request["values"]]
        if not all(admitted(v) for v in values):
            raise ValueError("benchmark contains values outside the proved fast domain")
        blocks.append(values)
    records = []
    for name in ("bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"):
        start = perf_counter()
        expected = [mse_scale(block, name) for block in blocks]
        oracle_seconds = perf_counter()-start
        start = perf_counter()
        actual = [mse_scale_fp32(block, name) for block in blocks]
        fast_seconds = perf_counter()-start
        if actual != expected:
            raise ValueError(f"FP32 integer-grid search differs from oracle for {name}")
        records.append({"format": name, "blocks": len(blocks), "responses_identical": True,
                        "oracle_seconds": oracle_seconds, "integer_grid_seconds": fast_seconds,
                        "speedup": oracle_seconds/fast_seconds})
        print(records[-1], flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during shared-scale benchmark")
    report = {"schema_version": "phase3-shared-scale-benchmark-1.0.0", "campaign_sha256": digest(campaign()),
              "source_sha256": source, "status": "verified_equivalent", "records": records,
              "input_blocks": [[str(v) for v in block] for block in blocks],
              "implementations": [source_reference(ROOT / path) for path in (
                  "tools/analysis/phase3_shared_scale_benchmark.py", "tools/phase3/shared_scale_fast.py")],
              "limits": ["256 format/block comparisons on retained preparation inputs, plus separate edge/tie tests",
                         "timings include host contention and are scale-search timings, not graph preparation or inference speedups",
                         "the exact integer-grid implementation only admits bounded finite FP32 values and retains oracle fallback"]}
    write(ROOT / "results/summaries/phase3-shared-scale-benchmark.json", report)


if __name__ == "__main__":
    main()
