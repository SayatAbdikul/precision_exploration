"""Explicit one-MAC-layer interventions in an otherwise folded FP32 classifier.

This diagnostic is deliberately separate from strict end-to-end screening.
Inputs, weights and output of exactly one selected MAC are quantized; its
decoded result resumes the FP32 graph. No other layer uses candidate arithmetic.
"""
from __future__ import annotations

import time

from public.analysis.phase3.diagnostics import ReferenceSamples, LayerDiagnostics
from public.analysis.phase3.statistics import paired_classification
from public.inference.conformance_job import source_identity
from public.inference.tensor import Tensor, SharedEncoding, parse_encoding, QUANTIZATION_OBSERVER
from public.quantization.graph.executable import execute, freeze_graph
from public.quantization.ptq.encoding import float_tensor
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.baselines import screen_rows
from tools.phase3.runtime import load_runtime
from tools.phase3.screening import checkpoint, save_checkpoint, pipeline_identity


class OneLayer:
    def __init__(self, graph, target, *, backend="cpp", sample_limit=4096):
        domains = dict(graph["inputs"])
        selected = None
        for node in graph["nodes"]:
            if node["name"] == target:
                selected = node
                break
            domains[node["name"]] = node["attrs"].get("output", domains[node["inputs"][0]])
        if selected is None or selected["op"] not in {"linear", "conv2d", "depthwise_conv2d"}:
            raise ValueError("one-layer intervention requires a named MAC layer")
        constants = {name: Tensor.from_document(graph["constants"][name]) for name in selected["inputs"] if name in graph["constants"]}
        inputs = {name: domains[name] for name in selected["inputs"] if name not in constants}
        self.graph = freeze_graph(inputs=inputs, constants=constants, nodes=[selected], outputs=[target],
                                  provenance={"kind": "diagnostic_one_mac_layer", "target": target})
        self.target, self.backend, self.sample_limit = target, backend, sample_limit
        self.values, self.result = {}, None

    def reset(self):
        self.values, self.result = {}, None

    def record(self, name, value):
        if name in self.graph["inputs"]:
            # A later in-place FP32 activation must not alter the captured input.
            self.values[name] = value.detach().clone()
        if name != self.target:
            return
        if self.result is not None or set(self.values) != set(self.graph["inputs"]):
            raise ValueError("one-layer observation order/coverage mismatch")
        import numpy as np
        import torch
        references = ReferenceSamples(self.sample_limit)
        references.record(name, value)
        tensors = {}
        for source, domain in self.graph["inputs"].items():
            encoding = parse_encoding(domain)
            array = self.values[source].contiguous().numpy()
            tensors[source] = (Tensor.quantize(array.ravel().tolist(), tuple(array.shape), encoding)
                               if isinstance(encoding, SharedEncoding) else float_tensor(array, encoding))
        diagnostics = LayerDiagnostics(references)
        token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
        start = time.perf_counter()
        try:
            execution = execute(self.graph, tensors, backend=self.backend, observer=diagnostics)
        finally:
            QUANTIZATION_OBSERVER.reset(token)
        seconds = time.perf_counter()-start
        output = execution["outputs"][self.target]
        decoded = np.asarray([float(v) for v in output.values()], dtype=np.float32).reshape(output.shape)
        if tuple(value.shape) != output.shape or not np.isfinite(decoded).all():
            raise ValueError("one-layer intervention has invalid shape/nonfinite output")
        self.result = {"layer_trace": execution["layers"][self.target], "output_sha256": digest(output.document()),
                       "diagnostics": diagnostics.records[self.target], "native_seconds": seconds,
                       "execution_modes": execution["execution_modes"]}
        # MAC outputs are fresh tensors (unlike identity/reshape views, which
        # are intentionally not admitted as intervention targets).
        value.copy_(torch.from_numpy(decoded).to(device=value.device, dtype=value.dtype))


class NoIntervention:
    def record(self, name, value):
        pass


def top5(tensor):
    values = tensor.detach().cpu().numpy().ravel()
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))[:5]


def run(prepared_path, target, *, backend="cpp", images=8, root=ROOT):
    if type(images) is not int or not 1 <= images <= 8:
        raise ValueError("selected sensitivity studies use 1–8 frozen images")
    prepared = read(prepared_path)
    config = read(checked(prepared["configuration"], root))
    checked(config["calibration"], root)
    graph = read(checked(prepared["graph"], root))
    if config["model"] == "yolov8n":
        raise ValueError("detector interventions require a separate FP32 propagation implementation")
    if config["runtime"]["source_sha256"] != source_identity() or digest(config) != prepared["configuration_sha256"]:
        raise ValueError("sensitivity preparation is stale or mismatched")
    plan = campaign(root)
    if config["campaign_sha256"] != digest(plan):
        raise ValueError("sensitivity candidate belongs to a different campaign")
    selected = OneLayer(graph, target, backend=backend, sample_limit=plan["diagnostics"]["sample_elements_per_layer_image"])
    job = {"schema_version": "phase3-one-layer-1.0.0", "scope": "diagnostic_one_layer", "campaign_sha256": digest(plan),
           "prepared": reference(prepared_path, root), "target": target, "images": images, "backend": backend,
           "source_sha256": source_identity(), "pipeline_sha256": pipeline_identity(root), "implementation": reference(__file__, root),
           "intervention": "quantize inputs/weights, execute one declared Model C MAC, quantize output, resume folded FP32"}
    identity = digest(job)
    work = root / "artifacts/phase3/sensitivity" / identity
    write(work / "job.json", job)
    write(work / "intervention-graph.json", selected.graph)
    population, payload_root, _ = screen_rows(config["model"], plan, root)
    baseline_ref = read(root / "results/summaries/phase3-baselines.json")["models"][config["model"]]
    baseline = read(checked(baseline_ref, root))
    sample, observe, _ = load_runtime(config["model"], plan, root)
    records, paired = [], []
    for ordinal, row in enumerate(population[:images]):
        frozen = baseline["records"][ordinal]
        if frozen["sample_sha256"] != row["sha256"]:
            raise ValueError("sensitivity image order differs from the frozen baseline")
        path = work / f"{row['sha256']}.json"
        context = {"job_sha256": identity, "sample": row, "frozen_reference": frozen}
        document = checkpoint(path, context)
        if backend not in document["backends"]:
            started = time.perf_counter()
            inputs, _ = sample(payload_root / row["relative_path"])
            fp32_prediction = top5(observe(inputs, NoIntervention()))
            selected.reset()
            prediction = top5(observe(inputs, selected))
            if selected.result is None:
                raise ValueError("selected layer was not executed")
            document["backends"][backend] = {**selected.result, "fp32_prediction": fp32_prediction,
                                            "candidate_prediction": prediction, "total_seconds": time.perf_counter()-started}
            save_checkpoint(path, document)
            print(f"{config['model']} {prepared['format']} {target}: saved image {ordinal+1}/{images}", flush=True)
        if (source_identity() != job["source_sha256"] or pipeline_identity(root) != job["pipeline_sha256"]
                or reference(__file__, root) != job["implementation"]):
            raise ValueError("sensitivity sources changed; retain completed checkpoints")
        result = document["backends"][backend]
        paired.append({**frozen, "fp32_prediction": result["fp32_prediction"], "candidate_prediction": result["candidate_prediction"]})
        records.append(reference(path, root))
    options = {key: plan["statistics"][key] for key in ("confidence", "seed")}
    statistics = paired_classification(paired, [row["sample_id"] for row in paired],
                                       resamples=plan["statistics"]["classification_resamples"], **options)
    report = {"schema_version": "phase3-one-layer-summary-1.0.0", "status": "completed", "scope": "diagnostic_one_layer",
              "job_sha256": identity, "job": reference(work / "job.json", root), "model": config["model"], "format": prepared["format"],
              "target": target, "backend": backend, "images": images, "image_records": records, "statistics": statistics,
              "intervention_graph": reference(work / "intervention-graph.json", root),
              "frozen_fp32_top5_matches": sum(row["fp32_prediction"] == baseline["records"][i]["fp32_prediction"] for i, row in enumerate(paired)),
              "paired_reference": "fresh folded-model FP32 predictions on identical images", "frozen_baseline": baseline_ref,
              "limits": ["one selected MAC layer only; not strict end-to-end screening and never a D4 quality label",
                         "input/output representation stores and declared accumulator are part of the intervention",
                         "small fixed diagnostic subset; timing may overlap other authorized work"]}
    write(work / "summary.json", report)
    write(root / "results/summaries" / f"phase3-sensitivity-{identity[:12]}.json", report)
    return report
