"""Count verified Phase 3 evidence without mistaking preparation for screening."""
from datetime import datetime, timezone
import sqlite3
from public.inference.conformance_job import source_identity
from public.quantization.calibration.artifact import validate as validate_calibration, frozen_context
from tools.phase3.common import ROOT, campaign, read, checked, write, reference, digest
from public.experiments.registry import ExperimentRegistry
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.graphs import configuration


def main():
    plan = campaign()
    source = source_identity()
    calibrations, graphs, runs, acceptances, sensitivity = [], [], [], [], []
    for model in plan["models"]:
        context = frozen_context(ROOT, model)
        inventory = read(ROOT / f"artifacts/phase3/calibration/{model}.json")
        if inventory["campaign_sha256"] != digest(plan):
            raise ValueError("calibration belongs to another campaign")
        for row in inventory["records"]:
            artifact = validate_calibration(read(checked(row["calibration"])), ROOT, verify_payloads=False)
            if artifact["context"] != context or artifact["format"] != row["format"]:
                raise ValueError("calibration inventory identity mismatch")
            calibrations.append({"model": model, "format": row["format"], "artifact": row["calibration"]})
    pairs = [(row["model"], row["format"]) for row in calibrations]
    if len(pairs) != len(set(pairs)) or set(pairs) != {(model, name) for model in plan["models"] for name in plan["formats"]}:
        raise ValueError("calibration coverage does not match the frozen candidate matrix")
    for row in preparation_records().values():
        if "configuration" not in row:
            continue
        config = read(checked(row["configuration"]))
        checked(row["graph"]); checked(row["accumulator_bounds"])
        graphs.append({"model": row["model"], "format": row["format"], "configuration_sha256": row["configuration_sha256"],
                       "current_source": config["runtime"]["source_sha256"] == source,
                       "current_definition": digest(configuration(row["model"], row["format"], plan)) == row["configuration_sha256"],
                       "status": row["status"]})
    for path in sorted((ROOT / "artifacts/phase3/runs").glob("*/job.json")):
        job = read(path)
        prepared = read(checked(job["prepared"]))
        image_records = []
        for image in sorted(path.parent.glob("*.json")):
            if len(image.stem) != 64:
                continue
            record = read(image)
            if digest({key: value for key, value in record.items() if key != "record_sha256"}) != record["record_sha256"]:
                raise ValueError("Phase 3 image checkpoint hash mismatch")
            image_records.append(record)
        summary_path = path.parent / "summary.json"
        runs.append({"job_sha256": digest(job), "scope": job["scope"], "requested_images": job["images"],
                     "model": prepared["model"], "format": prepared["format"],
                     "current_source": job["source_sha256"] == source,
                     "completed_images": sum(set(row["backends"]) == set(job["backends"]) for row in image_records),
                     "completed_image_backends": sum(len(row["backends"]) for row in image_records),
                     "summary": reference(summary_path) if summary_path.exists() else None})
    database = ROOT / "results/databases/phase3.sqlite"
    if database.exists():
        with ExperimentRegistry(database, validator=lambda value: value) as registry:
            for run in runs:
                row = registry.connection.execute("SELECT status,error,heartbeat_at FROM runs WHERE experiment_id=? ORDER BY attempt DESC LIMIT 1",
                                                  (run["job_sha256"],)).fetchone()
                run["registry"] = dict(row) if row else None
    for path in sorted((ROOT / "artifacts/phase3/acceptance").glob("*/acceptance.json")):
        value = read(path)
        proof = read(checked(value["proof"]))
        checked(value["pilot"])
        checked(proof["proof_implementation"])
        acceptances.append({"configuration_sha256": value["configuration_sha256"], "status": value["status"],
                            "current_source": value["source_sha256"] == source, "evidence": reference(path)})
    for path in sorted((ROOT / "results/summaries").glob("phase3-sensitivity-*.json")):
        value = read(path)
        job = read(checked(value["job"]))
        if value["scope"] != "diagnostic_one_layer" or value["status"] != "completed" or digest(job) != value["job_sha256"]:
            raise ValueError("invalid sensitivity completion record")
        for item in value["image_records"]:
            checked(item)
        if len(value["image_records"]) != value["images"]:
            raise ValueError("incomplete sensitivity image evidence")
        sensitivity.append({key: value[key] for key in ("model", "format", "target", "images")})
        sensitivity[-1].update(current_source=job["source_sha256"] == source, evidence=reference(path))
    cache_counts = []
    for path in sorted((ROOT / "artifacts/phase3/shared-scale-cache" / source).glob("*.sqlite")):
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            count = connection.execute("SELECT COUNT(*) FROM scales").fetchone()[0]
        cache_counts.append({"format": path.stem, "completed_block_searches": count,
                             "scope": "operational cache count; entries are hash-checked when reused, not graph acceptance"})
    report = {"schema_version": "phase3-progress-1.0.0", "status": "in_progress", "updated_at": datetime.now(timezone.utc).isoformat(), "campaign": reference(ROOT / "public/experiments/configs/experiment_a/phase3-screen-v1.json"),
              "source_sha256": source, "planned_configurations": plan["configuration_count"],
              "verified_calibrations": len(calibrations), "calibrations": calibrations, "graphs": graphs, "runs": runs,
              "current_prepared_graphs": sum(row["current_source"] and row["current_definition"] for row in graphs), "acceptances": acceptances,
              "completed_sensitivity_studies": sensitivity,
              "shared_scale_cache": cache_counts,
              "completed_fixed_1k_screens": sum(row["scope"] == "screen" and row["requested_images"] == 1000 and row["completed_images"] == 1000
                  and row["summary"] is not None and (row.get("registry") or {}).get("status") == "COMPLETED" for row in runs),
              "remaining": ["complete encoded graphs and accumulator acceptance for the candidate matrix",
                            "complete native runtime/diagnostics pilots", "complete every valid fixed-1k paired screen",
                            "selected layer sensitivity and catastrophic diagnosis", "hardware preservation priors and D4 decision"]}
    write(ROOT / "results/summaries/phase3-progress.json", report)
    print({"calibrations": len(calibrations), "prepared_graphs": len(graphs), "runs": len(runs), "status": report["status"]})


if __name__ == "__main__":
    main()
