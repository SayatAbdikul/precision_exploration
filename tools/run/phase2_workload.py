"""Submit/resume strict real-image validation, retaining registry failures."""
import argparse
import json
from pathlib import Path

from public.experiments.registry import ExperimentRegistry
from public.experiments.scheduler import LocalScheduler
from public.experiments.registry.identity import experiment_sha256
from public.inference.conformance_job import source_identity
from public.inference.workload_job import SCHEMA,run_job,ROOT
from public.inference.reference.arithmetic import format_named
from public.formats.oracle.manifest import manifest_sha256
from public.workloads.datasets.identity import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--calibration",required=True,type=Path)
    parser.add_argument("--weight-format",default="fp6_e3m2")
    parser.add_argument("--accumulator",default="fp32_e8m23_accumulator")
    parser.add_argument("--backend",choices=("reference","cpp","cuda"),default="cuda")
    parser.add_argument("--compare-backend",choices=("reference","cpp","cuda"))
    parser.add_argument("--full",action="store_true",help="Run every image in the frozen screening list")
    parser.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    path = args.calibration.resolve()
    cal = json.loads(path.read_text())
    records = json.loads((ROOT/"data/manifests/index.json").read_text())["records"]
    name = "coco_screen_1k" if args.model == "yolov8n" else "imagenet_screen_1k"
    formats = {role:{"name":f,"sha256":manifest_sha256(format_named(f).manifest)} for role,f in {
        "weight":args.weight_format,"activation":cal["format"],"accumulator":args.accumulator}.items()}
    config = {"schema_version":SCHEMA,"purpose":"strict_workload_validation","model":args.model,
              "calibration":{"path":str(path.relative_to(ROOT)),"sha256":sha256(path)},"formats":formats,
              "evaluation":{"name":name,"sha256":records[name]["sha256"],"selection":"all" if args.full else "first_eight_by_sha256_v1"},
              "runtime":{"backend":args.backend,"compare_backend":args.compare_backend,"semantic_version":"2.0.0","source_sha256":source_identity()}}
    from public.inference.workload_job import validate_job
    validate_job(config)
    if args.preflight:
        print("Verified frozen model, calibration, all image payloads and runtime identities")
        return
    identity = experiment_sha256(config)
    # Each command owns one queue, so another pending experiment cannot be
    # accidentally executed with this workload executor.
    database = ROOT/"results/databases"/f"phase2-workload-{identity}.sqlite"
    with ExperimentRegistry(database) as registry:
        run_id,status = registry.submit(config)
        if status == "PENDING":
            LocalScheduler(registry,worker_id="phase2-workload").run_once(run_job)
        row = dict(registry.connection.execute("SELECT * FROM runs WHERE run_id=?",(run_id,)).fetchone())
        report = {"schema_version":"2.0.0","job_sha256":identity,"run_id":run_id,"status":row["status"],"error":row["error"],
                  "source_sha256":config["runtime"]["source_sha256"],"model":args.model,"evaluation":config["evaluation"],
                  "artifacts":f"artifacts/workload_runs/{identity}","database":str(database.relative_to(ROOT))}
        summary = ROOT/"results/summaries"/f"phase2-{args.model}-workload-{identity[:12]}.json"
        summary.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
        if row["status"] != "COMPLETED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
