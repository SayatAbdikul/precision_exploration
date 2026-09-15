"""Audit mapped bias headroom and optionally version wider integer candidates."""
import argparse

from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.preparation_inventory import preparation_records


def audit(*, resolve=False, root=ROOT):
    plan = campaign(root)
    preparation = preparation_records(root)
    destination = root / "public/experiments/configs/experiment_a"
    index_path = destination / "phase3-accumulator-resolutions-v1.json"
    index = read(index_path) if index_path.exists() else {"schema_version": "phase3-accumulator-resolution-index-1.0.0", "resolutions": {}}
    rows = []
    for key, prepared in sorted(preparation.items()):
        if "accumulator_bounds" not in prepared:
            continue
        bounds = read(checked(prepared["accumulator_bounds"], root))
        failures = []
        for item in bounds["reductions"]:
            if item["status"] == "insufficient_headroom":
                maximum = int(item["dot_bound"]) + item["maximum_stored_bias_codes"]
                failures.append({**item, "required_signed_bits": maximum.bit_length()+1})
        rows.append({"model": prepared["model"], "format": prepared["format"], "configuration": prepared["configuration"],
                     "bounds": prepared["accumulator_bounds"], "failures": failures,
                     "status": "accumulator_revision_required" if failures else "static_checks_only_not_accepted"})
        if not resolve or not failures or key in index["resolutions"]:
            continue
        if any(item["accumulator"] != "int32_accumulator" or item["required_signed_bits"] > 64 for item in failures):
            continue
        resolution = {"schema_version": "phase3-accumulator-resolution-1.0.0", "campaign_sha256": digest(plan),
                      "model": prepared["model"], "format": prepared["format"], "replaces": "int32_accumulator",
                      "accumulator": "int64_accumulator", "status": "candidate_pending_native_acceptance",
                      "reason": "mapped bias plus conservative reduction bound exceeds INT32; preserve canonical calibration and widen the accumulator",
                      "evidence": [prepared["accumulator_bounds"], prepared["configuration"]]}
        path = destination / "accumulator-resolutions" / f"{prepared['model']}-{prepared['format']}-{digest(resolution)[:12]}.json"
        if path.exists() and read(path) != resolution:
            raise ValueError("accumulator resolution is immutable")
        write(path, resolution)
        index["resolutions"][key] = reference(path, root)
    if resolve:
        write(index_path, index)
    report = {"schema_version": "phase3-accumulator-audit-1.0.0", "campaign_sha256": digest(plan), "records": rows,
              "resolutions": index["resolutions"], "limits": ["absence of static overflow is not accumulator acceptance",
                  "FP64 and quire rounding sensitivity remains pending", "pending accumulator engineering does not reject an operand format"]}
    write(root / "results/summaries/phase3-accumulator-audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true", help="version INT64 candidates for proven INT32 headroom failures")
    args = parser.parse_args()
    report = audit(resolve=args.resolve)
    print({"audited_graphs": len(report["records"]), "revision_required": sum(bool(row["failures"]) for row in report["records"]),
           "versioned_resolutions": len(report["resolutions"])})


if __name__ == "__main__":
    main()
