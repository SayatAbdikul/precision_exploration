"""Vendor anchor: PyTorch FX graph-mode static INT8 on the frozen checkpoints (CPU).

Uses only what ships with the installed torch: ``torch.ao.quantization``
``prepare_fx``/``convert_fx`` with the default x86 qconfig mapping
(HistogramObserver activations with reduce_range=True, i.e. 7-bit quint8;
per-channel symmetric qint8 weights), plus a full-range variant
(reduce_range=False, 8-bit activations) so the anchor is not handicapped by
the legacy non-VNNI saturation margin.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, frozen_inputs, seal
from tools.experiment_b.runner import metrics, runtime
from .common import BASE, PROTOCOL

QCONFIGS = {
    "x86_default": "get_default_qconfig_mapping('x86'): HistogramObserver(reduce_range=True) quint8 affine "
                   "activations (7-bit range), per-channel symmetric qint8 weights",
    "x86_fullrange": "same mapping with the global qconfig using HistogramObserver(reduce_range=False): 8-bit "
                     "quint8 affine activations, per-channel symmetric qint8 weights",
    "qnnpack_fullrange": "QNNPACK engine (no 16-bit saturating multiply-add): HistogramObserver(reduce_range=False) "
                         "8-bit quint8 affine activations, per-channel symmetric qint8 weights; added after the "
                         "x86 full-range arm showed kernel saturation",
}
ENGINES = {"x86_default": "x86", "x86_fullrange": "x86", "qnnpack_fullrange": "qnnpack"}
CALIBRATION_BATCH = 32
THREADS = 8


def qconfig_mapping(name):
    from torch.ao.quantization import QConfig, get_default_qconfig_mapping
    from torch.ao.quantization.observer import HistogramObserver, default_per_channel_weight_observer
    mapping = get_default_qconfig_mapping("x86")
    if name in ("x86_fullrange", "qnnpack_fullrange"):
        mapping = mapping.set_global(QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                                             weight=default_per_channel_weight_observer))
    elif name != "x86_default":
        raise ValueError(f"unknown vendor qconfig: {name}")
    return mapping


def quantize_static(model, name, batches):
    """prepare_fx -> calibrate -> convert_fx.  ``batches`` yields FP32 input tensors."""
    import torch
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = ENGINES[name]
    first = next(iter(batches))
    prepared = prepare_fx(copy.deepcopy(model).eval(), qconfig_mapping(name), example_inputs=(first,))
    with torch.inference_mode():
        for batch in batches:
            prepared(batch)
    return convert_fx(prepared)


def graph_census(converted):
    counts = Counter()
    for node in converted.graph.nodes:
        if node.op == "call_module":
            counts[type(converted.get_submodule(node.target)).__module__.split(".")[-1] + "." +
                   type(converted.get_submodule(node.target)).__name__] += 1
        elif node.op in ("call_function", "call_method"):
            counts[f"{node.op}:{getattr(node.target, '__name__', node.target)}"] += 1
    return dict(sorted(counts.items()))


def run_vendor(args):
    import torch
    from tools.experiment_b.classifier import configure, load_model
    from .data import cached_inputs
    from .runner import evaluate, record_run
    start = time.monotonic()
    configure("cpu")
    torch.set_num_threads(THREADS)
    # quantized::linear resizes its output with an op that has no flagged-deterministic kernel.
    torch.use_deterministic_algorithms(False)
    env = {**runtime("cpu"), "threads": THREADS, "quantized_engine": ENGINES[args.qconfig], "deterministic": False}
    context = frozen_inputs(args.model)
    _, transform, original = load_model(args.model, "cpu")
    inputs, rows = cached_inputs("imagenet_screen_1k", 1000, transform)
    calibration, calibration_rows = cached_inputs("imagenet_calibration_2k", args.calibration_images, transform)

    def batches():
        for index in range(0, len(calibration_rows), CALIBRATION_BATCH):
            yield torch.from_numpy(np.array(calibration[index:index + CALIBRATION_BATCH], dtype=np.float32))

    class Replay:
        def __iter__(self):
            return batches()

    tick = time.monotonic()
    converted = quantize_static(original, args.qconfig, Replay())
    preparation = time.monotonic() - tick
    configuration = {"model_context": context, "vendor": {
        "flow": "torch.ao.quantization.quantize_fx.prepare_fx/convert_fx (FX graph mode static PTQ)",
        "qconfig": args.qconfig, "qconfig_description": QCONFIGS[args.qconfig],
        "model": "unfolded torchvision module with the frozen checkpoint; FX fuses conv-bn(-relu)",
        "graph_census": graph_census(converted)},
        "calibration": {"list": "imagenet_calibration_2k", "images": len(calibration_rows),
                        "rows_sha256": digest(calibration_rows), "batch": CALIBRATION_BATCH},
        "protocol": PROTOCOL, "runtime": env, "vendor_source_sha256": file_hash(ROOT / "tools/experiment_b2/vendor.py"),
        "data_source_sha256": file_hash(ROOT / "tools/experiment_b2/data.py")}
    identity = digest(configuration)
    seal(BASE / "configurations" / f"{identity}.json", configuration)
    reference = {"model_context": context, "mode": "unfolded_fp32_cpu_reference", "runtime": env, "protocol": PROTOCOL}
    reference_id = digest(reference)
    seal(BASE / "configurations" / f"{reference_id}.json", reference)
    baseline, _ = evaluate(original, inputs, rows, reference_id, "cpu", args.images)
    predictions, timing = evaluate(converted, inputs, rows, identity, "cpu", args.images)
    result = metrics(predictions, baseline)
    seal(BASE / "summaries" / f"{identity}-{args.images}.json",
         {"configuration_sha256": identity, "panel_images": args.images, "metrics": result, "timing": timing,
          "baseline_sha256": reference_id, "prediction_digest": digest(predictions)})
    wall = time.monotonic() - start
    record_run(args.stage, args.model, "int8", f"vendor_{args.qconfig}", args.images, identity, {
        "metrics": result, "baseline_sha256": reference_id, "device": "cpu",
        "calibration_images": len(calibration_rows),
        "cost": {"job_wall_seconds": wall, "gpu_seconds": 0.0, "preparation_seconds": preparation,
                 "evaluation_seconds": timing["seconds_this_invocation"], "cpu_threads": THREADS}})
    print(json.dumps({"model": args.model, "vendor": args.qconfig, "images": args.images,
                      "top1": result["top1_percent"], "top5": result["top5_percent"],
                      "fp32_top1": result["fp32_top1_percent"], "configuration": identity[:12],
                      "wall_seconds": round(wall, 1)}), flush=True)
    return 0
