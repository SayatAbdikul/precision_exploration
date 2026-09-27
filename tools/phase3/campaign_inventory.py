"""Discover ready work from frozen definitions and validated image checkpoints."""
from pathlib import Path

from public.inference.conformance_job import source_identity
from public.quantization.graph.executable import graph_sha256
from tools.phase3.baselines import screen_rows
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.evidence import verify_execution_provenance
from tools.phase3.screening import checkpoint, pipeline_identity


def make_job(task, root=ROOT):
    prepared = task["prepared"]
    if task["kind"] == "screen":
        prepared = reference(root / "artifacts/phase3/acceptance" / task["configuration_sha256"] /
                             "prepared-accepted.json", root)
    return {"schema_version": "phase3-job-1.0.0", "purpose": "experiment_a_screening",
            "prepared": prepared, "baseline": task["baseline"], "scope": task["kind"],
            "images": 1000 if task["kind"] == "screen" else 8,
            "backends": ["cuda"] if task["kind"] == "screen" else ["cpp", "cuda"],
            "campaign_sha256": task["campaign_sha256"], "source_sha256": task["source_sha256"],
            "pipeline_sha256": task["pipeline_sha256"]}


def saved_progress(job, root=ROOT):
    """Hash-check partial checkpoints without requiring a completed summary."""
    identity = digest(job)
    directory = root / "artifacts/phase3/runs" / identity
    if not directory.exists():
        return {"saved_images": 0, "requested_images": job["images"], "remaining_images": job["images"]}
    if (directory / "job.json").exists() and read(directory / "job.json") != job:
        raise ValueError("saved job differs from its identity")
    prepared = read(checked(job["prepared"], root))
    config = read(checked(prepared["configuration"], root))
    graph = read(checked(prepared["graph"], root))
    graph_id = graph_sha256(graph)
    baseline = read(checked(job["baseline"], root))
    population, _, _ = screen_rows(config["model"], campaign(root), root, verify_images=False)
    layers = {node["name"] for node in graph["nodes"]}
    complete, seconds = 0, []
    for index, sample in enumerate(population[:job["images"]]):
        path = directory / f"{sample['sha256']}.json"
        if not path.exists():
            continue
        record = checkpoint(path, {"job_sha256": identity, "graph_sha256": graph_id,
                                  "sample": sample, "paired": baseline["records"][index]})
        if not set(record["backends"]) <= set(job["backends"]):
            raise ValueError("checkpoint contains an undeclared backend")
        verify_execution_provenance(record, prepared["configuration_sha256"], root)
        for backend in record["backends"].values():
            if set(backend["layers"]) != layers or set(backend["diagnostics"]) != layers:
                raise ValueError("checkpoint layer/diagnostic evidence is incomplete")
        if set(record["backends"]) == set(job["backends"]):
            complete += 1
            seconds.append(sum(v["seconds"] for v in record["backends"].values()))
    return {"saved_images": complete, "requested_images": job["images"],
            "remaining_images": job["images"] - complete,
            "historical_mean_seconds": sum(seconds) / len(seconds) if seconds else None,
            "timing_scope": "historical per-image timings may span devices; not a current-host ETA"}


def discover(root=ROOT):
    plan, source, pipeline = campaign(root), source_identity(), pipeline_identity(root)
    campaign_id = digest(plan)
    baselines = read(root / "results/summaries/phase3-baselines.json")["models"]
    prepared_rows = preparation_records(root)
    readiness = read(root / "results/summaries/phase3-integer-readiness.json")
    evidence = read(checked(readiness["immutable_evidence"], root))
    if evidence["source_sha256"] != source or evidence["campaign_sha256"] != campaign_id:
        raise ValueError("static readiness inventory is stale")
    for item in evidence["implementations"]:
        checked(item, root)
    ready = {row["configuration"]: row for row in evidence["records"] if row["status"] == "static_proof_passed"}
    tasks, blocked = [], []
    for model in plan["models"]:
        for fmt in plan["formats"]:
            key = f"{model}/{fmt}"
            if key not in prepared_rows:
                blocked.append({"configuration": key, "reason": "preparation missing"})
                continue
            prepared = prepared_rows[key]
            config = read(checked(prepared["configuration"], root))
            if config["runtime"]["source_sha256"] != source:
                blocked.append({"configuration": key, "reason": "prepared engine is stale"})
                continue
            prepared_ref = reference(checked(prepared["configuration"], root).parent / "prepared.json", root)
            approved = root / "artifacts/phase3/acceptance" / prepared["configuration_sha256"] / "prepared-accepted.json"
            accepted = approved.exists()
            if accepted:
                approval = read(approved)
                decision = read(checked(approval["screen_acceptance"], root))
                if (decision["status"] != "accepted" or decision["source_sha256"] != source or
                        decision["configuration_sha256"] != prepared["configuration_sha256"]):
                    raise ValueError(f"invalid acceptance: {key}")
            static = ready.get(key)
            if not accepted and not (static and static["graph"] == prepared["graph"] and
                                     static["configuration_artifact"] == prepared["configuration"]):
                blocked.append({"configuration": key,
                                "reason": "family/graph acceptance proof still required; not a rejected candidate"})
                continue
            common = {"model": model, "format": fmt, "configuration_sha256": prepared["configuration_sha256"],
                      "prepared": prepared_ref, "baseline": baselines[model], "campaign_sha256": campaign_id,
                      "source_sha256": source, "pipeline_sha256": pipeline}
            prefix = prepared["configuration_sha256"]
            dependencies = []
            if not accepted:
                tasks.append({**common, "id": prefix + "-pilot", "kind": "pilot", "dependencies": [], "resource": "cuda"})
                tasks.append({**common, "id": prefix + "-accept", "kind": "accept",
                              "dependencies": [prefix + "-pilot"], "resource": "cpu"})
                dependencies = [prefix + "-accept"]
            screen = {**common, "id": prefix + "-screen", "kind": "screen", "dependencies": dependencies, "resource": "cuda"}
            if accepted:
                job = make_job(screen, root)
                screen["progress"] = saved_progress(job, root)
            if screen.get("progress", {}).get("remaining_images") != 0:
                tasks.append({**common, "id": prefix + "-validate-store", "kind": "validate-store",
                              "dependencies": dependencies, "resource": "cuda"})
                screen["dependencies"] = [prefix + "-validate-store"]
                screen["accelerated_store"] = True
            tasks.append(screen)
            tasks.append({**common, "id": prefix + "-analyze", "kind": "analyze",
                          "dependencies": [prefix + "-screen"], "resource": "cpu"})
    return {"schema_version": "phase3-controller-inventory-1.0.0", "campaign_sha256": campaign_id,
            "source_sha256": source, "pipeline_sha256": pipeline,
            "planned_configurations": plan["configuration_count"], "tasks": tasks,
            "blocked": blocked, "phase3_complete": False}
