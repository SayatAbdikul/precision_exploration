"""Continue explicitly listed integer pilots/screens under frozen identities."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.evidence import verify_complete
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.phase3.screening import pipeline_identity, validate_job


def check_plan(plan):
    if (plan["schema_version"] != "phase3-integer-continuation-1.0.0" or plan["source_sha256"] != source_identity()
            or plan["pipeline_sha256"] != pipeline_identity() or plan["campaign_sha256"] != digest(campaign())):
        raise ValueError("continuation plan campaign/engine/pipeline changed")
    if plan["implementation_sha256"] != reference(__file__)["sha256"]:
        raise ValueError("continuation implementation changed")
    for task in plan["tasks"]:
        if (task["kind"] not in {"pilot", "screen", "analyze"} or task["model"] not in {"resnet18", "mobilenet_v2"}
                or task["format"] not in {"int4", "int5", "int6", "int8"}):
            raise ValueError("continuation task is outside the finite-integer workload scope")


def job_for(task):
    prepared = preparation_records()[f"{task['model']}/{task['format']}"]
    directory = checked(prepared["configuration"]).parent
    path = directory / "prepared.json"
    if task["kind"] == "screen":
        path = ROOT / "artifacts/phase3/acceptance" / prepared["configuration_sha256"] / "prepared-accepted.json"
    baselines = read(ROOT / "results/summaries/phase3-baselines.json")
    job = {"schema_version": "phase3-job-1.0.0", "purpose": "experiment_a_screening", "prepared": reference(path),
           "baseline": baselines["models"][task["model"]], "scope": task["kind"],
           "images": 8 if task["kind"] == "pilot" else 1000,
           "backends": ["cpp", "cuda"] if task["kind"] == "pilot" else ["cuda"],
           "campaign_sha256": digest(campaign()), "source_sha256": source_identity(), "pipeline_sha256": pipeline_identity()}
    validate_job(job)
    return job, directory / "prepared.json"


def run(plan_path):
    plan = read(plan_path)
    check_plan(plan)
    work = ROOT / "artifacts/phase3/integer-continuation" / digest(plan)
    work.mkdir(parents=True, exist_ok=True)
    with (work / "controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(work / "plan.json", plan)
        source_reference(__file__)
        for ordinal, task in enumerate(plan["tasks"]):
            # Wait for the current native worker without taking its experiment
            # lease. The runner subsequently acquires its own exclusive lock.
            with (ROOT / "artifacts/phase3/locks/native-worker.lock").open("a") as worker:
                fcntl.flock(worker, fcntl.LOCK_EX)
            check_plan(plan)
            started = perf_counter()
            event_path = work / f"{ordinal:03d}.json"
            log_path = work / f"{ordinal:03d}.log"
            event = {"task": task, "plan_sha256": digest(plan), "status": "running"}
            write(event_path, event)
            print(ordinal, task, "starting", flush=True)
            try:
                with log_path.open("a") as log:
                    def command(module, *args):
                        check_plan(plan)
                        subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    if task["kind"] == "analyze":
                        previous_job = read(checked(task["job"]))
                        summary = checked(task["job"]).parent / "summary.json"
                        _, _, (_, config, _, _, _, _) = verify_complete(summary, current_execution=True)
                        if previous_job["scope"] != "screen" or config["model"] != task["model"] or config["formats"]["activation"]["name"] != task["format"]:
                            raise ValueError("analysis task references another workload")
                    else:
                        job, prepared_path = job_for(task)
                        directory = ROOT / "artifacts/phase3/runs" / digest(job)
                        summary = directory / "summary.json"
                        if task["kind"] == "pilot":
                            source_summary = checked(task["pilot_source"])
                            source_job, _, (_, config, _, _, _, _) = verify_complete(source_summary, current_execution=True)
                            if source_job["scope"] != "pilot" or config["model"] != task["model"] or config["formats"]["activation"]["name"] != task["format"]:
                                raise ValueError("pilot reuse source references another workload")
                            if not (directory / "job.json").exists():
                                command("tools.run.phase3_extend_pilot", "--summary", source_summary, "--images", 8, "--backends", "cpp", "cuda")
                            if read(directory / "job.json") != job:
                                raise ValueError("extended pilot differs from planned native job")
                        command("tools.run.phase3_screen", "--model", task["model"], "--format", task["format"],
                                "--scope", job["scope"], "--images", job["images"], "--backends", *job["backends"],
                                "--recover-stale-seconds", 300)
                    verify_complete(summary, current_execution=True)
                    command("tools.analysis.phase3_results", "--summary", summary)
                    command("tools.analysis.phase3_diagnostics", "--summary", summary)
                    if task["kind"] == "pilot":
                        command("tools.analysis.phase3_shape_check", "--pilot", summary)
                        command("tools.run.phase3_accept", "--prepared", prepared_path, "--pilot", summary)
                    event.update(status="completed", summary=reference(summary))
            except Exception as error:
                event.update(status="failed", error=f"{type(error).__name__}: {error}")
            event.update(seconds=perf_counter()-started, log=reference(log_path) if log_path.exists() else None)
            write(event_path, event)
            print(ordinal, event["status"], event.get("error", ""), flush=True)
            # A changed definition stops subsequent work; other task failures
            # remain recorded while independent accepted work can continue.
            check_plan(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan)


if __name__ == "__main__":
    main()
