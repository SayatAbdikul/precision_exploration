"""Verify exact pruning against the retained block suite and fresh oracle calls."""
from fractions import Fraction
from time import perf_counter

from public.inference.conformance_job import source_identity
from public.quantization.calibration.mse import mse_scale
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.provenance import source_reference
from tools.phase3.shared_scale_pruned import mse_scale_pruned


def main():
    plan, source = campaign(), source_identity()
    suite_ref = read(ROOT / "results/summaries/phase3-shared-dyadic-benchmark.json")["immutable_evidence"]
    suite = read(checked(suite_ref))
    if suite["status"] != "prototype_verified" or suite["source_sha256"] != source or suite["campaign_sha256"] != digest(plan):
        raise ValueError("pruning requires a verified current block suite")
    records = []
    for row in suite["records"]:
        name = row["format"]
        checks, old_seconds, new_seconds = [], 0.0, 0.0
        mse_scale_pruned([Fraction(0)], name)
        for case in row["cases"]:
            values = [Fraction(v) for v in case["values"]]
            started = perf_counter()
            expected = mse_scale(values, name)
            elapsed_old = perf_counter()-started
            details = {}
            started = perf_counter()
            actual = mse_scale_pruned(values, name, diagnostics=details)
            elapsed_new = perf_counter()-started
            old_seconds += elapsed_old
            new_seconds += elapsed_new
            checks.append({"kind": case["kind"], "values": case["values"], "oracle": expected, "accelerated": actual,
                           "complete_response_equal": actual == expected == case["oracle"], "search": details,
                           "oracle_seconds": elapsed_old, "accelerated_seconds": elapsed_new})
        records.append({"format": name, "cases": checks, "oracle_seconds": old_seconds, "accelerated_seconds": new_seconds,
                        "search_only_speedup": old_seconds/new_seconds,
                        "complete_responses_equal": all(c["complete_response_equal"] for c in checks)})
        print(name, len(checks), "cases", records[-1]["complete_responses_equal"], "speedup", old_seconds/new_seconds, flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during pruning verification")
    report = {"schema_version": "phase3-shared-pruned-benchmark-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "prototype_verified" if all(r["complete_responses_equal"] for r in records) else "requires_diagnosis",
              "input_suite": suite_ref, "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_shared_scale_pruned_benchmark.py", "tools/phase3/shared_scale_pruned.py",
                  "tools/phase3/shared_scale_dyadic.py", "tools/phase3/shared_scale_fast.py")],
              "limits": ["candidate_count remains the 255 legal scales accounted for by evaluation or exact exclusion",
                         "actual evaluated/excluded counts are retained separately and preserve smaller-scale ties",
                         "fresh original oracle responses and retained responses both agree; no policy or codebook change",
                         "search-only warm-cache timings under concurrent load, not production inference speedups",
                         "prototype is not installed in frozen execution"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"shared-pruned-benchmark-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-shared-pruned-benchmark.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
