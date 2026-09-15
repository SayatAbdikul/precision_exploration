"""Verify retained numerical evidence independently of summary status labels."""
from pathlib import Path

from public.inference.conformance_job import source_identity
from public.quantization.graph.executable import graph_sha256
from tools.phase3.common import ROOT, campaign, checked, digest, file_hash, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.baselines import screen_rows


def archive_pipeline(root=ROOT):
    from tools.phase3.screening import pipeline_identity
    paths = [root / "tools/phase3" / f"{name}.py" for name in ("common", "graphs", "baselines", "runtime", "screening")]
    paths += [root / "public/analysis/phase3" / f"{name}.py" for name in ("statistics", "diagnostics")]
    identity = pipeline_identity(root)
    directory = root / "artifacts/phase3/pipelines" / identity
    files = {}
    for path in paths:
        relative = path.relative_to(root)
        target = directory / "files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = path.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("pipeline archive is immutable")
        target.write_bytes(payload)
        files[str(relative)] = reference(target, root)
    record = {"pipeline_sha256": identity, "files": files}
    if digest({name: item["sha256"] for name, item in files.items()}) != identity:
        raise ValueError("pipeline changed during archival")
    write(directory / "index.json", record)
    return reference(directory / "index.json", root)


def verify_pipeline(job, root=ROOT, *, current_execution=False):
    path = root / "artifacts/phase3/pipelines" / job["pipeline_sha256"] / "index.json"
    index = read(path)
    if index["pipeline_sha256"] != job["pipeline_sha256"] or digest(
            {name: item["sha256"] for name, item in index["files"].items()}) != job["pipeline_sha256"]:
        raise ValueError("execution pipeline archive identity mismatch")
    for name, item in index["files"].items():
        checked(item, root)
        # Common JSON writing and later statistical analysis do not change
        # native outputs. All graph/runtime/diagnostic code must still match
        # when accepting historical pilot images for current screening.
        if current_execution and name not in {"tools/phase3/common.py", "public/analysis/phase3/statistics.py"}:
            if file_hash(root / name) != item["sha256"]:
                raise ValueError("pilot execution/diagnostic implementation has changed")
    return reference(path, root)


def read_records(job, image_refs=None, root=ROOT):
    plan = campaign(root)
    if job["campaign_sha256"] != digest(plan):
        raise ValueError("execution belongs to another frozen campaign")
    if job["scope"] not in {"pilot", "screen"} or type(job["images"]) is not int or (
            job["scope"] == "pilot" and not 1 <= job["images"] <= 8) or (job["scope"] == "screen" and job["images"] != 1000):
        raise ValueError("invalid native pilot/screen population")
    backends = job["backends"]
    if not backends or len(set(backends)) != len(backends) or not set(backends) <= {"reference", "cpp", "cuda"}:
        raise ValueError("invalid declared execution backends")
    prepared = read(checked(job["prepared"], root))
    config = read(checked(prepared["configuration"], root))
    graph = read(checked(prepared["graph"], root))
    if digest(config) != prepared["configuration_sha256"] or config["runtime"]["source_sha256"] != job["source_sha256"]:
        raise ValueError("execution configuration/source mismatch")
    expected_config = configuration(config["model"], config["formats"]["activation"]["name"], plan, root,
                                    accumulator_resolution=config.get("accumulator_resolution"))
    # Historical numerical results retain their original source identity.
    expected_config["runtime"]["source_sha256"] = job["source_sha256"]
    if config != expected_config:
        raise ValueError("configuration differs from frozen candidate definition")
    baseline = read(checked(job["baseline"], root))
    if baseline["model"] != config["model"] or baseline["campaign_sha256"] != digest(plan) or len(baseline["records"]) != 1000:
        raise ValueError("paired baseline identity/population mismatch")
    checked(baseline["predictions"], root)
    population, _, _ = screen_rows(config["model"], plan, root, verify_images=False)
    selected = population[:job["images"]]
    if len(selected) != job["images"]:
        raise ValueError("frozen population is incomplete")
    identity = digest(job)
    directory = root / "artifacts/phase3/runs" / identity
    if image_refs is None:
        image_refs = [reference(directory / f"{row['sha256']}.json", root) for row in selected]
    if len(image_refs) != len(selected) or len({item["path"] for item in image_refs}) != len(selected):
        raise ValueError("incomplete or duplicated image evidence")
    records = []
    layers = {node["name"] for node in graph["nodes"]}
    graph_identity = graph_sha256(graph)
    for ordinal, (item, sample) in enumerate(zip(image_refs, selected)):
        record = read(checked(item, root))
        paired = baseline["records"][ordinal]
        if (digest({k: v for k, v in record.items() if k != "record_sha256"}) != record["record_sha256"]
                or record["job_sha256"] != identity or record["graph_sha256"] != graph_identity
                or record["sample"] != sample or record["paired"] != paired or paired["sample_sha256"] != sample["sha256"]):
            raise ValueError("image evidence hash/provenance mismatch")
        if set(record["backends"]) != set(backends):
            raise ValueError("incomplete backend evidence")
        first = record["backends"][backends[0]]
        for result in record["backends"].values():
            if set(result["layers"]) != layers or set(result["diagnostics"]) != layers:
                raise ValueError("incomplete layer/diagnostic evidence")
            if result["layers"] != first["layers"] or result["output_sha256"] != first["output_sha256"] or result["prediction"] != first["prediction"]:
                raise ValueError("retained native backends disagree")
        records.append(record)
    return prepared, config, graph, baseline, records, image_refs


def paired_records(records, model, scope):
    rows = []
    for record in records:
        baseline = record["paired"]
        candidate = next(iter(record["backends"].values()))["prediction"]
        rows.append({**baseline, "candidate_prediction": candidate,
                     "fp32_correct": None if model == "yolov8n" else baseline["fp32_prediction"][0] == baseline["ground_truth"],
                     "candidate_correct": None if model == "yolov8n" else candidate[0] == baseline["ground_truth"],
                     "summary": {"scope": scope, "sample_sha256": baseline["sample_sha256"]}})
    return rows


def verify_complete(summary_path, root=ROOT, *, current_execution=False):
    summary_path = Path(summary_path)
    summary = read(summary_path)
    job = read(summary_path.parent / "job.json")
    if (summary["status"] != "completed" or summary["job_sha256"] != digest(job)
            or summary["source_sha256"] != job["source_sha256"] or summary["images"] != job["images"] or summary["scope"] != job["scope"]):
        raise ValueError("completed summary identity/population mismatch")
    if current_execution and job["source_sha256"] != source_identity():
        raise ValueError("numerical engine changed since native execution")
    verify_pipeline(job, root, current_execution=current_execution)
    evidence = read_records(job, summary["image_records"], root)
    prepared, config, _, _, records, _ = evidence
    if summary["configuration_sha256"] != prepared["configuration_sha256"]:
        raise ValueError("summary belongs to another configuration")
    if summary["backend_equality"] != ({"cpp", "cuda"} <= set(job["backends"])) or not summary["diagnostics_complete"]:
        raise ValueError("summary overstates native/diagnostic coverage")
    if read(checked(summary["paired"], root)) != paired_records(records, config["model"], job["scope"]):
        raise ValueError("paired analysis rows differ from retained predictions")
    expected_timings = {backend: [record["backends"][backend]["seconds"] for record in records] for backend in job["backends"]}
    if summary["timings_seconds"] != expected_timings:
        raise ValueError("summary timings differ from retained execution")
    return job, summary, evidence
