"""Lease-fenced Phase 3 pilot/screen execution with paired image checkpoints."""
from __future__ import annotations

import time
from pathlib import Path

from public.experiments.registry import ExperimentRegistry
from public.inference.conformance_job import source_identity
from public.inference.tensor import Tensor, SharedEncoding, parse_encoding, QUANTIZATION_OBSERVER
from public.inference.workload_job import detector_predictions
from public.quantization.graph.executable import execute, graph_sha256
from public.quantization.ptq.encoding import float_tensor
from public.analysis.phase3.diagnostics import ReferenceSamples, LayerDiagnostics
from tools.phase3.baselines import screen_rows
from tools.phase3.common import ROOT, campaign, read, write, checked, reference, digest, file_hash
from tools.phase3.runtime import load_runtime


def pipeline_identity(root=ROOT):
    files = [root / "tools/phase3" / f"{name}.py" for name in ("common", "graphs", "baselines", "runtime", "screening")]
    files += [root / "public/analysis/phase3" / f"{name}.py" for name in ("statistics", "diagnostics")]
    return digest({str(path.relative_to(root)): file_hash(path) for path in sorted(files)})


def checkpoint(path, context):
    if not path.exists():
        return {**context, "backends": {}}
    document = read(path)
    payload = {key: value for key, value in document.items() if key != "record_sha256"}
    if digest(payload) != document.get("record_sha256"):
        raise ValueError("image checkpoint hash mismatch")
    if any(payload.get(key) != value for key, value in context.items()):
        raise ValueError("image checkpoint provenance mismatch")
    return payload


def save_checkpoint(path, payload):
    write(path, {**payload, "record_sha256": digest(payload)})


def require_screen_acceptance(prepared, root=ROOT):
    from tools.phase3.acceptance import verify_acceptance
    return verify_acceptance(prepared, root)


def validate_job(job, root=ROOT):
    if set(job) != {"schema_version", "purpose", "prepared", "baseline", "scope", "images", "backends",
                   "campaign_sha256", "pipeline_sha256", "source_sha256"} or job["schema_version"] != "phase3-job-1.0.0":
        raise ValueError("invalid Phase 3 job schema")
    if job["purpose"] != "experiment_a_screening" or job["scope"] not in {"pilot", "screen"}:
        raise ValueError("invalid Phase 3 execution scope")
    if type(job["images"]) is not int or (job["scope"] == "pilot" and not 1 <= job["images"] <= 8) or (job["scope"] == "screen" and job["images"] != 1000):
        raise ValueError("pilots use 1–8 images; full screening requires all 1000")
    if not job["backends"] or len(set(job["backends"])) != len(job["backends"]) or not set(job["backends"]) <= {"cpp", "cuda", "reference"}:
        raise ValueError("invalid execution backends")
    plan = campaign(root)
    if job["campaign_sha256"] != digest(plan) or job["source_sha256"] != source_identity() or job["pipeline_sha256"] != pipeline_identity(root):
        raise ValueError("Phase 3 campaign/source/pipeline identity mismatch")
    prepared = read(checked(job["prepared"], root))
    config = read(checked(prepared["configuration"], root))
    if digest(config) != prepared["configuration_sha256"] or config["campaign_sha256"] != digest(plan) or config["runtime"]["source_sha256"] != source_identity():
        raise ValueError("prepared configuration identity/source mismatch")
    from tools.phase3.graphs import configuration
    if config != configuration(config["model"], config["formats"]["activation"]["name"], plan, root):
        raise ValueError("configuration differs from the frozen uniform-W/A candidate definition")
    checked(config["calibration"], root)
    checked(prepared["graph"], root)
    baseline = read(checked(job["baseline"], root))
    if baseline["model"] != config["model"] or baseline["campaign_sha256"] != digest(plan) or len(baseline["records"]) != 1000:
        raise ValueError("paired baseline identity/population mismatch")
    checked(baseline["predictions"], root)
    if job["scope"] == "screen":
        require_screen_acceptance(prepared, root)
    return job


def run_job(job, root=ROOT):
    validate_job(job, root)
    import numpy as np
    plan = campaign(root)
    prepared = read(checked(job["prepared"], root))
    config = read(checked(prepared["configuration"], root))
    graph = read(checked(prepared["graph"], root))
    identity, graph_identity = digest(job), graph_sha256(graph)
    work = root / "artifacts/phase3/runs" / identity
    write(work / "job.json", job)
    population, payload_root, _ = screen_rows(config["model"], plan, root)
    baseline = read(checked(job["baseline"], root))
    selected = population[:job["images"]]
    sample, observe, manifest = load_runtime(config["model"], plan, root)
    inputs_name = next(iter(graph["inputs"]))
    encoding = parse_encoding(graph["inputs"][inputs_name])
    records = []
    for ordinal, row in enumerate(selected):
        paired = baseline["records"][ordinal]
        if paired["sample_sha256"] != row["sha256"]:
            raise ValueError("FP32 pairs do not follow frozen image order")
        context = {"job_sha256": identity, "graph_sha256": graph_identity, "sample": row, "paired": paired}
        path = work / f"{row['sha256']}.json"
        stored = checkpoint(path, context)
        if not set(stored["backends"]) <= set(job["backends"]):
            raise ValueError("checkpoint includes an undeclared backend")
        if set(stored["backends"]) != set(job["backends"]):
            fp32, original_shape = sample(payload_root / row.get("relative_path", row.get("file_name", "")))
            if list(fp32.shape) != manifest["input_shape"]:
                raise ValueError("native preprocessing shape mismatch")
            references = ReferenceSamples(plan["diagnostics"]["sample_elements_per_layer_image"])
            observe(fp32, references)
            if not {node["name"] for node in graph["nodes"]} <= set(references.records):
                raise ValueError("FP32 diagnostic plan omits candidate nodes")
            tensor = Tensor.quantize(fp32.numpy().ravel().tolist(), tuple(fp32.shape), encoding) if isinstance(encoding, SharedEncoding) else float_tensor(fp32.numpy(), encoding)
            for backend in job["backends"]:
                if backend in stored["backends"]:
                    continue
                diagnostics = LayerDiagnostics(references)
                token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
                start = time.perf_counter()
                print(f"{config['model']} {config['formats']['activation']['name']} image {ordinal+1}/{len(selected)} {backend}: executing", flush=True)
                try:
                    result = execute(graph, {inputs_name: tensor}, backend=backend, observer=diagnostics)
                finally:
                    QUANTIZATION_OBSERVER.reset(token)
                seconds = time.perf_counter()-start
                output = next(iter(result["outputs"].values()))
                values = output.values()
                if not all(np.isfinite(float(value)) for value in values):
                    raise ValueError("nonfinite candidate output; retain failure for diagnosis")
                prediction = detector_predictions(output, original_shape, row["image_id"]) if config["model"] == "yolov8n" else sorted(range(len(values)), key=lambda index: (-values[index], index))[:5]
                record = {"seconds": seconds, "layers": result["layers"], "output_sha256": digest(output.document()),
                          "prediction": prediction, "diagnostics": diagnostics.records, "execution_modes": result["execution_modes"]}
                for previous in stored["backends"].values():
                    if previous["layers"] != record["layers"] or previous["output_sha256"] != record["output_sha256"]:
                        raise ValueError("candidate native backends disagree")
                stored["backends"][backend] = record
                save_checkpoint(path, stored)
                print(f"image {ordinal+1} {backend}: saved {seconds:.2f}s", flush=True)
        if source_identity() != job["source_sha256"] or pipeline_identity(root) != job["pipeline_sha256"]:
            raise ValueError("execution sources changed; completed records remain under their original identity")
        backend_rows = list(stored["backends"].values())
        if any(result["layers"] != backend_rows[0]["layers"] or result["output_sha256"] != backend_rows[0]["output_sha256"]
               for result in backend_rows[1:]):
            raise ValueError("resumed candidate backends disagree")
        records.append(stored)
    paired_rows = []
    for record in records:
        baseline_row = record["paired"]
        candidate = record["backends"][job["backends"][0]]["prediction"]
        paired_rows.append({**baseline_row, "candidate_prediction": candidate,
                            "fp32_correct": None if config["model"] == "yolov8n" else baseline_row["fp32_prediction"][0] == baseline_row["ground_truth"],
                            "candidate_correct": None if config["model"] == "yolov8n" else candidate[0] == baseline_row["ground_truth"],
                            "summary": {"scope": job["scope"], "sample_sha256": baseline_row["sample_sha256"]}})
    write(work / "paired.json", paired_rows)
    with ExperimentRegistry(root / "results/databases/phase3.sqlite", validator=lambda item: validate_job(item, root)) as registry:
        registry.store_per_image(identity, paired_rows)
    summary = {"schema_version": "phase3-run-summary-1.0.0", "job_sha256": identity, "configuration_sha256": prepared["configuration_sha256"],
               "source_sha256": job["source_sha256"], "status": "completed", "scope": job["scope"], "images": len(records),
               "backend_equality": {"cpp", "cuda"} <= set(job["backends"]),
               "diagnostics_complete": all(set(result["diagnostics"]) == {node["name"] for node in graph["nodes"]}
                                           for record in records for result in record["backends"].values()),
               "paired": reference(work / "paired.json", root),
               "image_records": [reference(work / f"{row['sha256']}.json", root) for row in selected],
               "timings_seconds": {backend: [record["backends"][backend]["seconds"] for record in records] for backend in job["backends"]}}
    write(work / "summary.json", summary)
    return {"images": (len(records), "count"), "completed": (1, "boolean")}
