"""Compare bounded-dyadic scale search with all original oracle response fields."""
from fractions import Fraction
from random import Random
from time import perf_counter

from public.inference.conformance_job import source_identity
from public.quantization.calibration.mse import mse_scale
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.provenance import source_reference
from tools.phase3.shared_scale_dyadic import admitted, mse_scale_dyadic
from tools.phase3.shared_scale_fast import FORMATS


def main():
    plan, source = campaign(), source_identity()
    refs, native_values = [], []
    for model in plan["models"]:
        summary = read(ROOT / f"results/summaries/phase3-family-mac-pilot-{model}.json")
        ref = summary["immutable_evidence"]
        evidence = read(checked(ref))
        if evidence["source_sha256"] != source or evidence["campaign_sha256"] != digest(plan):
            raise ValueError("dyadic benchmark needs current native state evidence")
        refs.append(ref)
        for row in evidence["records"]:
            for layer in row["layers"]:
                if layer["accumulator"] == "fp64_e11m52_accumulator":
                    native_values.extend(Fraction(c["post_bias_value"]) for c in layer["checks"] if c["finite_post_bias"])
    pool = [v for v in native_values if admitted(v)]
    if not pool:
        raise ValueError("no admitted retained native values")
    rng = Random(310915)
    cases = [{"kind": "retained_native_values_mixed_block", "values": [rng.choice(pool) for _ in range(32)]} for _ in range(32)]
    for _ in range(8):
        values = []
        for _ in range(32):
            numerator = (1 << 52) | rng.getrandbits(52) | 1
            values.append(Fraction(numerator, 2**rng.randrange(43, 63))*(-1 if rng.randrange(2) else 1))
        cases.append({"kind": "seeded_53_bit_dyadic_block", "values": values})
    cases.extend([
        {"kind": "zero_block", "values": [Fraction(0)]*32},
        {"kind": "partial_block_domain_edges", "values": [Fraction(2**52+1, 2**132), Fraction(2**52+1, 2**52)*2**79,
            Fraction(1, 2**80), Fraction(2**80), -Fraction(2**52+3, 2**132)]},
        {"kind": "single_value_partial_block", "values": [Fraction(2**53-1, 2**54)]}])
    records = []
    for name in sorted(FORMATS):
        mse_scale_dyadic([Fraction(0)], name)
        mse_scale([Fraction(0)], name)
        checks, old_seconds, new_seconds = [], 0.0, 0.0
        for case in cases:
            if not all(admitted(v) for v in case["values"]):
                raise ValueError("benchmark case is outside the proven domain")
            started = perf_counter()
            expected = mse_scale(case["values"], name)
            elapsed_old = perf_counter()-started
            started = perf_counter()
            actual = mse_scale_dyadic(case["values"], name)
            elapsed_new = perf_counter()-started
            old_seconds += elapsed_old
            new_seconds += elapsed_new
            checks.append({"kind": case["kind"], "values": [str(v) for v in case["values"]],
                           "oracle": expected, "accelerated": actual, "complete_response_equal": actual == expected,
                           "oracle_seconds": elapsed_old, "accelerated_seconds": elapsed_new})
        records.append({"format": name, "cases": checks, "oracle_seconds": old_seconds, "accelerated_seconds": new_seconds,
                        "search_only_speedup": old_seconds/new_seconds, "complete_responses_equal": all(c["complete_response_equal"] for c in checks)})
        print(name, len(checks), "cases", records[-1]["complete_responses_equal"], "speedup", old_seconds/new_seconds, flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during dyadic benchmark")
    report = {"schema_version": "phase3-shared-dyadic-benchmark-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "prototype_verified" if all(r["complete_responses_equal"] for r in records) else "requires_diagnosis",
              "native_state_evidence": refs, "retained_native_values": len(native_values), "admitted_native_values": len(pool),
              "seed": 310915, "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_shared_scale_dyadic_benchmark.py", "tools/phase3/shared_scale_dyadic.py",
                  "tools/phase3/shared_scale_fast.py")],
              "limits": ["all 255 scales, original smallest-scale tie rule and complete response fields are compared",
                         "mixed blocks contain retained native values but are not actual layer activation blocks",
                         "53-bit dyadic admission is bounded in magnitude; other inputs delegate to existing verified behavior",
                         "warm-cache search-only timings with concurrent work; not native image or end-to-end speedups",
                         "prototype is not installed into frozen production execution or used to change completed evidence"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"shared-dyadic-benchmark-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-shared-dyadic-benchmark.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
