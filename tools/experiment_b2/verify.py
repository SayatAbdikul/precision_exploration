"""Read-only verification jobs for B2 (added after independent review 1, 2026-10-01).

Nothing here writes to ``configurations/``, ``predictions/``, ``summaries/`` or ``runs/``.
Each job re-computes something from the frozen inputs, compares it with the
sealed evidence and seals one record under ``artifacts/experiment_b2/verification/``.

- ``bias-residual``: how exact the empirical bias correction is when the
  finished engine is executed again on the bias-correction images, and whether
  any negative value ever reaches an unsigned quantizer.
- ``vendor-check``: rebuild a vendor (FX static INT8) model with the current
  ``vendor.py`` and compare with the sealed per-image predictions; record a
  census of the converted graph with full module paths.
- ``vendor-probe``: unsealed-prediction probes of vendor variants that are not
  anchors (currently a MinMax full-range arm), with a per-node code census that
  locates where a collapsing variant loses the signal.
- ``batch-sensitivity`` (r3): the sealed configuration re-evaluated at other
  batch sizes, compared with its sealed batch-8 predictions.
- ``relu6-operator`` (r3): the quantized ReLU6, Hardswish and Hardsigmoid
  operators on their own, both engines and both memory layouts.

Records are never overwritten (r3): a job refuses to write to an existing
path; pass ``--tag`` to write a new record beside the old one.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, frozen_inputs, seal, unseal
from tools.experiment_b.runner import metrics, runtime
from .common import BASE, BIAS_CORRECTION_IMAGES, MODELS, PROTOCOL, source_identity

OUT = BASE / "verification"
START = time.monotonic()


def target(path, tag):
    """The record path for this job; refuses a path that exists (verification records are never replaced)."""
    if tag:
        path = path.with_name(f"{path.stem}--{tag}.json")
    if path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} exists; verification records are not overwritten, pass a new --tag")
    return path


def _probe_class():
    import torch

    class Probe(torch.fx.Interpreter):
        """Executes a graph as B2 does and records raw conv/linear channel sums.

        With ``engine`` the quantizers and scales of that B2 engine are applied
        at its quantizing boundaries (same call as ``B2Interpreter.run_node``);
        without it the graph runs in FP32.
        """

        def __init__(self, module, plan, engine=None):
            super().__init__(module)
            self.plan, self.engine = plan, engine
            self.sums, self.counts, self.negative = {}, {}, {}

        def run_node(self, node):
            result = super().run_node(node)
            row = self.plan.get(node.name)
            if row is None:
                return result
            if row["kind"] in ("conv", "linear"):
                dims = [0] + list(range(2, result.ndim))
                self.sums[node.name] = self.sums.get(node.name, 0) + result.double().sum(dim=dims)
                self.counts[node.name] = self.counts.get(node.name, 0) + result.numel() // result.shape[1]
            if self.engine is not None and row["quantizes"]:
                if row["signedness"] == "unsigned":
                    self.negative[node.name] = self.negative.get(node.name, 0) + int((result < 0).sum())
                result = self.engine.quantizers[row["signedness"]](result, self.engine.scales[node.name])
            return result

        def means(self):
            return {name: self.sums[name] / self.counts[name] for name in self.sums}

    return Probe


def _channel_means(probe, inputs, chunk, device):
    import torch
    with torch.inference_mode():
        for start in range(0, len(inputs), chunk):
            probe.run(torch.from_numpy(np.array(inputs[start:start + chunk], dtype=np.float32)).to(device))
    return probe.means()


def bias_residual(args):
    """Residual per-channel mean gap of a bias-corrected B2 engine, re-executed on the correction images."""
    import torch
    from . import frozen  # noqa: F401
    from .data import cached_inputs
    from .engine import bias_correct
    from .runner import build
    setup, recipe, engine, metadata, configuration, identity = build(args, seal_configuration=False)
    if recipe.bias_correction != "empirical":
        raise SystemExit("bias-residual needs a recipe with empirical bias correction")
    sealed = (BASE / "configurations" / f"{identity}.json").exists()
    path = target(OUT / "bias-residual" / f"{args.model}--{args.format}--{args.recipe}--{identity[:12]}.json", args.tag)
    graph, device = setup.get("engine_graph", setup["graph"]), args.device
    inputs, rows = cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"], build=False)
    Probe = _probe_class()
    probe = torch.from_numpy(np.array(inputs[:8], dtype=np.float32)).to(device)
    with torch.inference_mode():
        if not torch.equal(Probe(engine.module, engine.plan, engine).run(probe), engine.run(probe)):
            raise ValueError("the verification interpreter does not reproduce the B2 engine")
    reference = _channel_means(Probe(graph, engine.plan), inputs, 32, device)
    residual = {}
    negative = {}
    for chunk in (32, PROTOCOL["inference_batch_size"]):
        run = Probe(engine.module, engine.plan, engine)
        means = _channel_means(run, inputs, chunk, device)
        residual[chunk] = {name: (reference[name] - means[name]) for name in reference}
        negative[chunk] = int(sum(run.negative.values()))
        unsigned_boundaries = len(run.negative)
    with torch.inference_mode():
        second = bias_correct(graph, copy.deepcopy(engine.module), engine.plan, engine.quantizers,
                              metadata["activation_scales"], inputs)
    layers = {}
    for name, first in metadata["bias_correction"].items():
        row = {"applied_max_abs_correction": first["max_abs_correction"],
               "applied_rms_correction": first["rms_correction"],
               "rms_fp32_channel_mean": first["rms_fp32_channel_mean"],
               "second_pass_max_abs_correction": second[name]["max_abs_correction"]}
        for chunk, values in residual.items():
            row[f"residual_max_abs_batch{chunk}"] = float(values[name].abs().max())
            row[f"residual_rms_batch{chunk}"] = float(values[name].pow(2).mean().sqrt())
        layers[name] = row
    batch = PROTOCOL["inference_batch_size"]
    key = f"residual_max_abs_batch{batch}"
    worst = max(layers, key=lambda name: layers[name][key])
    ratios = sorted(layers[name][key] / layers[name]["applied_max_abs_correction"] for name in layers
                    if layers[name]["applied_max_abs_correction"] > 0)
    wall = time.monotonic() - START
    record = {
        "check": "bias_correction_residual", "model": args.model, "format": args.format, "recipe_name": args.recipe,
        "configuration_sha256": identity, "configuration_is_a_sealed_b2_configuration": sealed,
        "images": len(rows), "rows_sha256": digest(rows), "source_sha256": source_identity(),
        "method": "FP32 and quantized per-output-channel means of every conv/linear output (before its own "
                  "quantization) over the bias-correction images; residual = FP32 mean - quantized mean after "
                  "correction, with the finished engine re-executed at batch 32 (the staged pass used 32) and at "
                  f"the evaluation batch size {batch}; second pass = corrections a second sequential bias_correct "
                  "would apply to the corrected graph",
        "summary": {"layers": len(layers), "worst_layer": worst, "worst_residual": layers[worst][key],
                    "applied_correction_at_worst_layer": layers[worst]["applied_max_abs_correction"],
                    "layers_with_residual_above_1e-3": sum(layers[n][key] > 1e-3 for n in layers),
                    "max_residual_batch32": max(layers[n]["residual_max_abs_batch32"] for n in layers),
                    "median_residual_to_correction_ratio": ratios[len(ratios) // 2] if ratios else None,
                    "max_second_pass_correction": max(layers[n]["second_pass_max_abs_correction"] for n in layers),
                    "unsigned_boundaries": unsigned_boundaries,
                    "negative_elements_reaching_unsigned_quantizers": negative[batch]},
        "layers": layers, "runtime": setup["env"],
        "cost": {"job_wall_seconds": wall, "gpu_seconds": wall if device == "cuda" else 0.0,
                 "note": "gpu_seconds is the whole job wall-clock while the GPU lock was held"}}
    seal(path, record)
    print(json.dumps({"record": str(path.relative_to(ROOT)), **record["summary"], "sealed_configuration": sealed,
                      "wall_seconds": round(wall, 1)}), flush=True)
    return 0


def detailed_census(converted):
    """Module census with full paths, so float and quantized modules cannot be confused."""
    from torch import nn
    modules, functions = Counter(), Counter()
    float_mac, standalone = [], Counter()
    for node in converted.graph.nodes:
        if node.op == "call_module":
            module = converted.get_submodule(node.target)
            cls = type(module)
            path = f"{cls.__module__}.{cls.__name__}"
            modules[path] += 1
            if cls.__module__.startswith("torch.nn.modules") and isinstance(module, (nn.Conv2d, nn.Linear)):
                float_mac.append(node.name)
            if cls.__name__ in ("ReLU", "ReLU6", "Hardswish", "Hardsigmoid", "Hardtanh"):
                standalone[cls.__name__] += 1
        elif node.op in ("call_function", "call_method"):
            functions[f"{node.op}:{getattr(node.target, '__name__', node.target)}"] += 1
    return {"modules": dict(sorted(modules.items())), "functions": dict(sorted(functions.items())),
            "float_conv_or_linear_nodes": float_mac,
            "standalone_activation_modules": dict(sorted(standalone.items()))}


def code_census(converted, tensor):
    """Per quantized node output on ``tensor``: scale, zero point, distinct codes used and the largest code."""
    import torch
    rows = {}

    class Census(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if isinstance(result, torch.Tensor) and result.is_quantized:
                codes = result.int_repr()
                per_tensor = result.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric)
                rows[node.name] = {"scale": float(result.q_scale()) if per_tensor else None,
                                   "zero_point": int(result.q_zero_point()) if per_tensor else None,
                                   "distinct_codes": int(torch.unique(codes).numel()),
                                   "max_code": int(codes.max()), "min_code": int(codes.min()),
                                   "fraction_at_zero_point": float((codes == result.q_zero_point()).float().mean())
                                   if per_tensor else None}
            return result

    with torch.inference_mode():
        Census(converted).run(tensor)
    return rows


def _layout(codes, expected):
    """How a wrong ReLU6 output relates to the exact clamp: same multiset of codes, and the clamp read in the other layout."""
    import torch
    histogram = torch.equal(torch.bincount(codes.flatten(), minlength=256), torch.bincount(expected.flatten(), minlength=256))
    permuted = False
    if codes.ndim == 4:
        flat = codes.contiguous().flatten()
        permuted = bool(torch.equal(flat, expected.permute(0, 2, 3, 1).contiguous().flatten())
                        or torch.equal(codes.permute(0, 2, 3, 1).contiguous().flatten(), expected.contiguous().flatten()))
    return {"code_histogram_equal_to_clamp": bool(histogram), "equals_clamp_read_in_other_layout": permuted}


def _channels_last(tensor):
    import torch
    return bool(tensor.ndim == 4 and tensor.is_contiguous(memory_format=torch.channels_last) and not tensor.is_contiguous())


def kernel_audit(converted, tensor):
    """Does every quantized kernel compute what its own quantized operands say?

    For each quantized conv or linear module the actual quantized input and the
    quantized weights are dequantized, the layer is evaluated in FP32 and the
    result is quantized with the module's output scale and zero point.  A
    correct integer kernel differs from that by at most one code (rounding of
    the requantization).  Larger differences are arithmetic failures of the
    kernel, not quantization error.  A ReLU6 applied to a quantized tensor must
    return exactly ``clamp(codes, zero_point, code_of(6.0))``; its input codes
    are copied before the call because the module works in place.  Since r3
    Hardswish, Hardsigmoid, pooling modules and the quantized add / mul /
    adaptive-average-pool functions are audited the same way (FP32 evaluation of
    the dequantized operands, requantized with the output parameters; kinds
    ``hardswish``, ``hardsigmoid``, ``pool``, ``add``, ``mul``), conv rows carry
    the number of distinct output codes, and ReLU6 rows say whether the input
    was channels-last and how a wrong output relates to the clamp.  Returns
    rows in topological order.
    """
    import torch
    from torch.ao.nn import quantized as nnq
    from torch.ao.nn.intrinsic import quantized as nniq
    from torch.nn import functional as F
    rows = []
    elementwise = {"Hardswish": ("hardswish", F.hardswish), "Hardsigmoid": ("hardsigmoid", F.hardsigmoid)}
    pools = {"AdaptiveAvgPool2d", "MaxPool2d", "AvgPool2d"}

    def generic(name, kind, out, reference, operands):
        expected = torch.quantize_per_tensor(reference, float(out.q_scale()), int(out.q_zero_point()), out.dtype)
        error = (out.int_repr().int() - expected.int_repr().int()).abs()
        rows.append({"node": name, "kind": kind, "groups": None, "max_code_error": int(error.max()),
                     "fraction_code_error_above_1": float((error > 1).float().mean()),
                     "input_channels_last": [_channels_last(x) for x in operands]})

    class Audit(torch.fx.Interpreter):
        def run_node(self, node):
            self.current = node.name
            return super().run_node(node)

        def call_function(self, target, args, kwargs):
            out = super().call_function(target, args, kwargs)
            name = getattr(target, "__name__", str(target))
            if not (isinstance(out, torch.Tensor) and out.is_quantized):
                return out
            quantized = [x for x in args if isinstance(x, torch.Tensor) and x.is_quantized]
            if name in ("add", "add_relu", "mul", "mul_relu") and len(quantized) == 2:
                left, right = (x.dequantize() for x in quantized)
                reference = left + right if name.startswith("add") else left * right
                generic(self.current, name.split("_")[0], out, torch.relu(reference) if name.endswith("relu") else reference,
                        quantized)
            elif name == "adaptive_avg_pool2d" and len(quantized) == 1:
                generic(self.current, "pool", out, F.adaptive_avg_pool2d(quantized[0].dequantize(), *args[1:], **kwargs),
                        quantized)
            return out

        def call_module(self, target, args, kwargs):
            module = self.fetch_attr(target)
            label = type(module).__name__
            quantized_input = isinstance(args[0], torch.Tensor) and args[0].is_quantized
            if label == "ReLU6" and quantized_input:
                scale, zero = float(args[0].q_scale()), int(args[0].q_zero_point())
                layout = _channels_last(args[0])
                before = args[0].int_repr().clone().int()
                out = super().call_module(target, args, kwargs)
                expected = before.clamp(zero, min(255, zero + int(round(6.0 / scale))))
                codes = out.int_repr().int()
                error = (codes - expected).abs()
                rows.append({"node": target.replace(".", "_"), "kind": "relu6", "groups": None,
                             "max_code_error": int(error.max()),
                             "fraction_code_error_above_1": float((error > 1).float().mean()),
                             "fraction_codes_not_equal": float((error > 0).float().mean()),
                             "input_zero_point": zero, "input_max_code": int(before.max()),
                             "input_fraction_codes_above_127": float((before > 127).float().mean()),
                             "input_channels_last": layout, **_layout(codes, expected)})
                return out
            if quantized_input and (label in elementwise or label in pools):
                reference_input = args[0].dequantize()
                out = super().call_module(target, args, kwargs)
                if isinstance(out, torch.Tensor) and out.is_quantized:
                    if label in elementwise:
                        kind, reference = elementwise[label][0], elementwise[label][1](reference_input)
                    else:
                        float_module = {"AdaptiveAvgPool2d": lambda x: F.adaptive_avg_pool2d(x, module.output_size),
                                        "MaxPool2d": lambda x: F.max_pool2d(x, module.kernel_size, module.stride,
                                                                            module.padding, module.dilation,
                                                                            module.ceil_mode),
                                        "AvgPool2d": lambda x: F.avg_pool2d(x, module.kernel_size, module.stride,
                                                                            module.padding, module.ceil_mode)}[label]
                        kind, reference = "pool", float_module(reference_input)
                    generic(target.replace(".", "_"), kind, out, reference, [args[0]])
                return out
            out = super().call_module(target, args, kwargs)
            if isinstance(module, nnq.Conv2d):
                reference = F.conv2d(args[0].dequantize(), module.weight().dequantize(), module.bias(), module.stride,
                                     module.padding, module.dilation, module.groups)
                fused = isinstance(module, nniq.ConvReLU2d)
            elif isinstance(module, nnq.Linear):
                reference = F.linear(args[0].dequantize(), module.weight().dequantize(), module.bias())
                fused = isinstance(module, nniq.LinearReLU)
            else:
                return out
            if fused:
                reference = torch.relu(reference)
            expected = torch.quantize_per_tensor(reference, float(module.scale), int(module.zero_point), torch.quint8)
            error = (out.int_repr().int() - expected.int_repr().int()).abs()
            codes = args[0].int_repr()
            rows.append({"node": target.replace(".", "_"), "kind": "linear" if isinstance(module, nnq.Linear) else "conv",
                         "groups": getattr(module, "groups", None),
                         "max_code_error": int(error.max()), "fraction_code_error_above_1": float((error > 1).float().mean()),
                         "input_zero_point": int(args[0].q_zero_point()), "input_max_code": int(codes.max()),
                         "input_fraction_codes_above_127": float((codes > 127).float().mean()),
                         "input_scale": float(args[0].q_scale()),
                         "distinct_output_codes": int(torch.unique(out.int_repr()).numel()),
                         "distinct_expected_codes": int(torch.unique(expected.int_repr()).numel())})
            return out

    with torch.inference_mode():
        Audit(converted).run(tensor)
    return rows


def _vendor_setup(model, calibration_images):
    import torch
    from tools.experiment_b.classifier import configure, load_model
    from .data import cached_inputs
    from .vendor import CALIBRATION_BATCH, THREADS
    configure("cpu")
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(False)
    _, transform, original = load_model(model, "cpu")
    inputs, rows = cached_inputs("imagenet_screen_1k", 1000, transform, build=False)
    calibration, calibration_rows = cached_inputs("imagenet_calibration_2k", calibration_images, transform, build=False)

    class Replay:
        def __iter__(self):
            for index in range(0, len(calibration_rows), CALIBRATION_BATCH):
                yield torch.from_numpy(np.array(calibration[index:index + CALIBRATION_BATCH], dtype=np.float32))

    return original, inputs, rows, calibration_rows, Replay()


def _top5(model, inputs, rows, limit, *, with_logits=False):
    import torch
    batch_size = PROTOCOL["inference_batch_size"]
    result, parts = [], []
    with torch.inference_mode():
        for start in range(0, limit, batch_size):
            tensor = torch.from_numpy(np.array(inputs[start:start + batch_size], dtype=np.float32))
            output = model(tensor)
            result.extend(output.topk(5, dim=1).indices.tolist())
            if with_logits:
                parts.append(output.clone())
    records = [{"sample": row, "top5": classes} for row, classes in zip(rows[:limit], result)]
    return (records, torch.cat(parts)) if with_logits else records


def vendor_check(args):
    """Rebuild one sealed vendor configuration with the current ``vendor.py`` and compare predictions."""
    import torch
    from .runner import load_predictions
    from .vendor import CALIBRATION_BATCH, ENGINES, THREADS, graph_census, quantize_static
    matches = sorted((BASE / "runs" / "vendor").glob(f"{args.model}--int8--vendor_{args.qconfig}--1000--*.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one sealed vendor run, found {len(matches)}")
    run = unseal(matches[0])
    identity = run["configuration_sha256"]
    path = target(OUT / "vendor" / f"{args.model}--{args.qconfig}--{identity[:12]}.json", args.tag)
    sealed = unseal(BASE / "configurations" / f"{identity}.json")
    original, inputs, rows, calibration_rows, replay = _vendor_setup(args.model, sealed["calibration"]["images"])
    env = {**runtime("cpu"), "threads": THREADS, "quantized_engine": ENGINES[args.qconfig], "deterministic": False}
    tick = time.monotonic()
    converted = quantize_static(original, args.qconfig, replay)
    preparation = time.monotonic() - tick
    current_hash = file_hash(ROOT / "tools/experiment_b2/vendor.py")
    fields = {"model_context": sealed["model_context"] == frozen_inputs(args.model),
              "graph_census": sealed["vendor"]["graph_census"] == graph_census(converted),
              "calibration_rows": sealed["calibration"] == {"list": "imagenet_calibration_2k",
                                                            "images": len(calibration_rows),
                                                            "rows_sha256": digest(calibration_rows),
                                                            "batch": CALIBRATION_BATCH},
              "protocol": sealed["protocol"] == PROTOCOL, "runtime": sealed["runtime"] == env,
              "data_source": sealed["data_source_sha256"] == file_hash(ROOT / "tools/experiment_b2/data.py")}
    saved = load_predictions(identity, rows)
    fresh = _top5(converted, inputs, rows, 1000)
    equal = sum(a["top5"] == b["top5"] for a, b in zip(saved, fresh))
    probe = torch.from_numpy(np.array(inputs[:8], dtype=np.float32))
    codes = code_census(converted, probe)
    wall = time.monotonic() - START
    record = {
        "check": "vendor_rebuild", "model": args.model, "qconfig": args.qconfig,
        "sealed_configuration_sha256": identity, "sealed_run_recorded_at": run["recorded_at"],
        "sealed_vendor_source_sha256": sealed["vendor_source_sha256"], "current_vendor_source_sha256": current_hash,
        "vendor_source_hash_is_current": sealed["vendor_source_sha256"] == current_hash,
        "every_other_identity_field_equal": all(fields.values()), "identity_fields_equal": fields,
        "identity_with_current_vendor_source": digest({**sealed, "vendor_source_sha256": current_hash}),
        "images": len(fresh), "top5_lists_equal_to_sealed": equal,
        "top1_percent": metrics(fresh, fresh)["top1_percent"], "sealed_top1_percent": run["metrics"]["top1_percent"],
        "census": detailed_census(converted),
        "codes_on_first_8_screen_images": {
            "quantized_nodes": len(codes), "largest_code_anywhere": max(v["max_code"] for v in codes.values()),
            "smallest_code_anywhere": min(v["min_code"] for v in codes.values()),
            "fewest_distinct_codes": min(v["distinct_codes"] for v in codes.values())},
        "runtime": env,
        "cost": {"job_wall_seconds": wall, "gpu_seconds": 0.0, "preparation_seconds": preparation,
                 "cpu_threads": THREADS}}
    seal(path, record)
    print(json.dumps({"record": str(path.relative_to(ROOT)), "top5_equal": equal,
                      "hash_current": record["vendor_source_hash_is_current"],
                      "other_fields_equal": record["every_other_identity_field_equal"],
                      "top1": record["top1_percent"], "standalone": record["census"]["standalone_activation_modules"],
                      "float_mac": len(record["census"]["float_conv_or_linear_nodes"]),
                      "largest_code": record["codes_on_first_8_screen_images"]["largest_code_anywhere"],
                      "wall_seconds": round(wall, 1)}), flush=True)
    return 0 if equal == len(fresh) and all(fields.values()) else 1


PROBES = {
    "x86_minmax_fullrange": ("x86", "MinMaxObserver(reduce_range=False) quint8 affine activations, per-channel "
                                    "symmetric qint8 weights, x86 engine"),
    "qnnpack_minmax_fullrange": ("qnnpack", "the same observers on the QNNPACK engine"),
}


# Sealed vendor arms re-run with the QNNPACK ReLU6 kernel replaced by an exact clamp of the codes.
RELU6_FIX = {"qnnpack_fullrange_relu6fix": "qnnpack_fullrange"}


def replace_relu6_by_code_clamp(converted):
    """Swap every ReLU6 that acts on a quantized tensor for an exact integer clamp (same scale, same zero point).

    This is what the operator is defined to compute; the QNNPACK kernel shipped with torch 2.3.0 on this x86
    machine does not (see ``kernel_audit``).  Returns the number of modules replaced.
    """
    import torch
    from torch import nn

    class CodeClampReLU6(nn.Module):
        def forward(self, x):
            scale, zero = float(x.q_scale()), int(x.q_zero_point())
            codes = x.int_repr().clamp(zero, min(255, zero + int(round(6.0 / scale))))
            return torch._make_per_tensor_quantized_tensor(codes, scale, zero)

    replaced = 0
    for node in converted.graph.nodes:
        if node.op == "call_module" and type(converted.get_submodule(node.target)).__name__ == "ReLU6":
            parent, _, leaf = node.target.rpartition(".")
            setattr(converted.get_submodule(parent) if parent else converted, leaf, CodeClampReLU6())
            replaced += 1
    return replaced


def _paired(model, rows, fresh, baseline):
    """Paired top-1 statistics of a probe against FP32, the B2 default and the vendor anchor of record."""
    from tools.analysis.b_stage_balanced_comparisons import compare_records
    from .runner import load_predictions

    def brief(left, right):
        top1 = compare_records(left, right)["top1"]
        return {"left_percent": top1["left_percent"], "right_percent": top1["right_percent"],
                "difference_pp": top1["difference_pp"], "pointwise_95_interval_pp": top1["pointwise_95_interval_pp"],
                "mcnemar_exact_p": top1["mcnemar_exact_p"]}

    def sealed(stage, recipe):
        matches = sorted((BASE / "runs" / stage).glob(f"{model}--int8--{recipe}--1000--*.json"))
        if len(matches) != 1:
            raise ValueError(f"expected one sealed {stage} run for {model} {recipe}")
        return load_predictions(unseal(matches[0])["configuration_sha256"], rows)

    return {"left_fp32_cpu__right_probe": brief(baseline, fresh),
            "left_probe__right_b2_default_int8": brief(fresh, sealed("exit", "default")),
            "left_vendor_x86_default__right_probe": brief(sealed("vendor", "vendor_x86_default"), fresh)}


def _probe_mapping():
    from torch.ao.quantization import QConfig, get_default_qconfig_mapping
    from torch.ao.quantization.observer import MinMaxObserver, default_per_channel_weight_observer
    return get_default_qconfig_mapping("x86").set_global(
        QConfig(activation=MinMaxObserver.with_args(reduce_range=False),
                weight=default_per_channel_weight_observer))


def vendor_probe(args):
    """A vendor variant that is not an anchor: accuracy on the 1k screen and a per-node code census.

    ``--qconfig`` is either a probe defined here or one of the sealed vendor arms
    (then the census locates where that arm loses the signal).  Predictions are
    not written; the record keeps their digest.
    """
    import torch
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from .vendor import ENGINES, QCONFIGS, THREADS, qconfig_mapping
    path = target(OUT / "vendor-probe" / f"{args.model}--{args.qconfig}--{args.calibration_images}.json", args.tag)
    original, inputs, rows, calibration_rows, replay = _vendor_setup(args.model, args.calibration_images)
    base = RELU6_FIX.get(args.qconfig)
    if base is not None:
        engine, mapping = ENGINES[base], qconfig_mapping(base)
        description = QCONFIGS[base] + "; every ReLU6 on a quantized tensor replaced by an exact clamp of the codes"
    elif args.qconfig in PROBES:
        engine, description, mapping = PROBES[args.qconfig][0], PROBES[args.qconfig][1], _probe_mapping()
    else:
        engine, description, mapping = ENGINES[args.qconfig], QCONFIGS[args.qconfig], qconfig_mapping(args.qconfig)
    torch.backends.quantized.engine = engine
    tick = time.monotonic()
    prepared = prepare_fx(copy.deepcopy(original).eval(), mapping, example_inputs=(next(iter(replay)),))
    with torch.inference_mode():
        for batch in replay:
            prepared(batch)
    converted = convert_fx(prepared)
    relu6_replaced = replace_relu6_by_code_clamp(converted) if base is not None else 0
    preparation = time.monotonic() - tick
    fresh, logits = _top5(converted, inputs, rows, 1000, with_logits=True)
    baseline = _top5(original, inputs, rows, 1000)
    result = metrics(fresh, baseline)
    paired = _paired(args.model, rows, fresh, baseline) if len(calibration_rows) == 2000 else None
    probe = torch.from_numpy(np.array(inputs[:8], dtype=np.float32))
    codes = code_census(converted, probe)
    order = list(codes)
    collapsed = [name for name in order if codes[name]["distinct_codes"] <= 2]
    kernels = kernel_audit(converted, probe)
    failing = [row["node"] for row in kernels if row["max_code_error"] > 1]
    by_kind = {kind: {"modules": sum(row["kind"] == kind for row in kernels),
                      "modules_with_code_error_above_1": sum(row["kind"] == kind and row["max_code_error"] > 1
                                                             for row in kernels)}
               for kind in sorted({row["kind"] for row in kernels})}
    wall = time.monotonic() - START
    record = {"check": "vendor_probe", "model": args.model, "qconfig": args.qconfig, "description": description,
              "quantized_engine": engine, "calibration_images": len(calibration_rows),
              "calibration_rows_sha256": digest(calibration_rows), "images": len(fresh), "metrics": result,
              "prediction_digest": digest(fresh), "distinct_top1_classes": len({p["top5"][0] for p in fresh}),
              "top5_in_screen_row_order": [p["top5"] for p in fresh], "screen_rows_sha256": digest(rows),
              "paired_top1": paired, "relu6_modules_replaced_by_exact_clamp": relu6_replaced,
              "ties": tie_statistics(logits, rows[:1000], per_image=True),
              "census": detailed_census(converted),
              "kernel_audit_first_8_screen_images": {
                  "method": "integer kernel output versus FP32 evaluation of the same dequantized operands, "
                            "requantized with the module's output parameters; a correct kernel differs by at most "
                            "one code",
                  "audited_modules": len(kernels), "modules_with_code_error_above_1": len(failing),
                  "by_kind": by_kind,
                  "first_failing_module": failing[0] if failing else None,
                  "largest_code_error": max(row["max_code_error"] for row in kernels),
                  "largest_fraction_of_outputs_off_by_more_than_1": max(row["fraction_code_error_above_1"]
                                                                        for row in kernels),
                  "rows_in_topological_order": kernels},
              "code_census_first_8_screen_images": {
                  "node_order": order,
                  "quantized_nodes": len(order), "nodes_with_at_most_2_distinct_codes": collapsed,
                  "first_collapsed_node": collapsed[0] if collapsed else None,
                  "position_of_first_collapsed_node": order.index(collapsed[0]) if collapsed else None,
                  "nodes": codes},
              "runtime": {**runtime("cpu"), "threads": THREADS, "quantized_engine": engine, "deterministic": False},
              "vendor_source_sha256": file_hash(ROOT / "tools/experiment_b2/vendor.py"),
              "verify_source_sha256": file_hash(ROOT / "tools/experiment_b2/verify.py"),
              "cost": {"job_wall_seconds": wall, "gpu_seconds": 0.0, "preparation_seconds": preparation,
                       "cpu_threads": THREADS}}
    seal(path, record)
    print(json.dumps({"record": str(path.relative_to(ROOT)), "top1": result["top1_percent"],
                      "fp32_top1": result["fp32_top1_percent"], "distinct_top1": record["distinct_top1_classes"],
                      "relu6_replaced": relu6_replaced,
                      "ties": [record["ties"][key] for key in ("images_with_tied_top1", "top1_percent_strict",
                                                               "top1_percent_expected", "top1_percent_optimistic")],
                      "paired": {k: [round(v["difference_pp"], 2)] + [round(x, 2) for x in v["pointwise_95_interval_pp"]]
                                 for k, v in (paired or {}).items()},
                      "first_collapsed": record["code_census_first_8_screen_images"]["first_collapsed_node"],
                      "collapsed_nodes": len(collapsed), "kernel_failures": len(failing), "kernels": len(kernels), "by_kind": by_kind,
                      "first_failing_kernel": failing[0] if failing else None,
                      "largest_code_error": max(row["max_code_error"] for row in kernels),
                      "wall_seconds": round(wall, 1)}), flush=True)
    return 0


def tie_statistics(logits, rows, *, per_image=False):
    """Top-1 under every tie-breaking rule: quantized logits tie, and ``topk`` resolves ties in an unspecified way.

    ``logits`` is an ``[images, classes]`` tensor.  An image is *tied* when its
    largest logit is attained by more than one class.  ``strict`` counts an
    image as correct only when the label is the unique maximum, ``optimistic``
    when the label is among the maxima, ``expected`` gives credit 1/k for a
    k-way tie (the mean over uniformly random tie-breaking), ``lowest_index``
    lets the smallest class index among the maxima win (a rule an integer
    argmax can implement).  With ``per_image`` the three per-image lists the
    rules are computed from are returned as well, in row order.
    """
    import torch
    logits = logits.detach().cpu()
    labels = torch.tensor([int(row["label"]) for row in rows])
    top = logits.amax(dim=1, keepdim=True)
    maxima = logits == top
    tied = maxima.sum(dim=1)
    among = maxima.gather(1, labels[:, None])[:, 0]
    lowest = maxima.int().argmax(dim=1) == labels  # argmax returns the first maximum of a 0/1 row
    n = len(rows)
    result = {"images_with_tied_top1": int((tied > 1).sum()),
              "images_with_tied_top1_involving_the_label": int((among & (tied > 1)).sum()),
              "largest_tie": int(tied.max()),
              "top1_percent_strict": 100 * float((among & (tied == 1)).sum()) / n,
              "top1_percent_expected": 100 * float((among.double() / tied.double()).sum()) / n,
              "top1_percent_optimistic": 100 * float(among.sum()) / n,
              "top1_percent_lowest_index": 100 * float(lowest.sum()) / n,
              "distinct_logit_values_median_per_image": float(torch.tensor(
                  [row.unique().numel() for row in logits]).double().median())}
    if per_image:
        result["per_image"] = {"tie_size": tied.tolist(), "label_among_maxima": among.int().tolist(),
                               "label_is_lowest_index_maximum": lowest.int().tolist()}
    return result


def batch_sensitivity(args):
    """Does the sealed configuration give the same predictions at another batch size, and how many top-1 are ties?

    The sealed evaluation uses batch 8.  The engine (and the folded FP32 graph)
    is run on the 1000 screen images at the batch sizes of ``--batches``
    (default 8, 32 and 1); top-5 is taken on the device from the FP32 logits of
    each batch, exactly as ``runner.evaluate`` does.  Batch 8 must reproduce
    the sealed top-5 lists; the others are compared with them.  The tie
    statistics say how much of top-1 is decided by the tie-breaking of
    ``topk`` (logits quantized to a small codebook tie often).  Nothing but
    the verification record is written.
    """
    import torch
    from tools.analysis.b_stage_balanced_comparisons import compare_records
    from . import frozen  # noqa: F401
    from .runner import build, load_predictions
    setup, recipe, engine, metadata, configuration, identity = build(args, seal_configuration=False)
    path = target(OUT / "batch-sensitivity" / f"{args.model}--{args.format}--{args.recipe}--{identity[:12]}.json", args.tag)
    rows, inputs, device = setup["rows"][:1000], setup["inputs"], args.device
    sealed = {"b2": load_predictions(identity, rows), "fp32": load_predictions(setup["baseline_sha256"], rows)}
    runners = {"b2": engine.run, "fp32": setup["graph"]}
    protocol_batch = PROTOCOL["inference_batch_size"]
    batches = [int(value) for value in args.batches.split(",")]
    if batches[0] != protocol_batch:
        raise SystemExit(f"the first batch size must be the protocol value {protocol_batch}")

    def evaluate(run, chunk):
        parts, top5 = [], []
        with torch.inference_mode():
            for start in range(0, len(rows), chunk):
                tensor = torch.from_numpy(np.array(inputs[start:start + chunk], dtype=np.float32)).to(device)
                output = run(tensor)
                top5.extend(output.topk(5, dim=1).indices.cpu().tolist())
                parts.append(output.cpu())
        return torch.cat(parts), top5

    result, seconds = {}, {}
    for name, run in runners.items():
        tick = time.monotonic()
        outputs = {chunk: evaluate(run, chunk) for chunk in batches}
        seconds[name] = time.monotonic() - tick
        result[name] = {}
        for chunk, (values, top5) in outputs.items():
            fresh = [{"sample": row, "top5": classes} for row, classes in zip(rows, top5)]
            top1 = compare_records(sealed[name], fresh)["top1"]
            gap = (values.double() - outputs[protocol_batch][0].double()).abs()
            cpu_top1 = values.double().topk(1, dim=1).indices[:, 0].tolist()
            result[name][f"batch{chunk}"] = {
                "top1_percent": metrics(fresh, sealed["fp32"])["top1_percent"],
                "top5_lists_equal_to_sealed_batch8": sum(a["top5"] == b["top5"] for a, b in zip(sealed[name], fresh)),
                "top1_class_changed_images": sum(a["top5"][0] != b["top5"][0] for a, b in zip(sealed[name], fresh)),
                "max_abs_logit_difference_to_batch8": float(gap.max()),
                "images_with_any_logit_difference_to_batch8": int((gap.amax(dim=1) > 0).sum()),
                "paired_top1_sealed_batch8_minus_this": {
                    "difference_pp": top1["difference_pp"], "pointwise_95_interval_pp": top1["pointwise_95_interval_pp"],
                    "mcnemar_exact_p": top1["mcnemar_exact_p"]},
                "ties": tie_statistics(values, rows, per_image=chunk == protocol_batch),
                "top1_class_changed_by_cpu_float64_topk_on_the_same_logits":
                    sum(a != b[0] for a, b in zip(cpu_top1, top5)),
                "prediction_digest": digest(fresh)}
    reproduced = all(result[name][f"batch{protocol_batch}"]["top5_lists_equal_to_sealed_batch8"] == len(rows)
                     for name in result)
    wall = time.monotonic() - START
    record = {"check": "batch_size_sensitivity_and_logit_ties", "model": args.model, "format": args.format,
              "recipe_name": args.recipe, "configuration_sha256": identity, "baseline_sha256": setup["baseline_sha256"],
              "images": len(rows), "rows_sha256": digest(rows), "source_sha256": source_identity(),
              "verify_source_sha256": file_hash(ROOT / "tools/experiment_b2/verify.py"), "batches": batches,
              "method": "engine and folded FP32 graph re-evaluated on the screen at the listed batch sizes (the first is "
                        "the protocol value); top-5 on the device as in runner.evaluate; each compared with the sealed "
                        "batch-8 per-image predictions of the same configuration; tie statistics from the logits",
              "sealed_batch8_predictions_reproduced": reproduced, "results": result, "runtime": setup["env"],
              "cost": {"job_wall_seconds": wall, "gpu_seconds": wall if device == "cuda" else 0.0,
                       "evaluation_seconds": seconds,
                       "note": "gpu_seconds is the whole job wall-clock while the GPU lock was held"}}
    seal(path, record)
    brief = {name: {key: [row["top1_percent"], row["top5_lists_equal_to_sealed_batch8"], row["top1_class_changed_images"],
                          row["max_abs_logit_difference_to_batch8"], row["ties"]["images_with_tied_top1"],
                          row["ties"]["top1_percent_strict"], row["ties"]["top1_percent_expected"],
                          row["ties"]["top1_percent_optimistic"]] for key, row in values.items()}
             for name, values in result.items()}
    print(json.dumps({"record": str(path.relative_to(ROOT)), "reproduced": reproduced,
                      "top1__top5_equal__top1_changed__max_logit_gap__tied__strict__expected__optimistic": brief,
                      "wall_seconds": round(wall, 1)}), flush=True)
    return 0 if reproduced else 1


def relu6_operator(args):
    """Quantized ReLU6, Hardswish and Hardsigmoid on their own: both engines, both layouts, in place or not (CPU)."""
    import torch
    from torch.ao.nn import quantized as nnq
    from torch.nn import functional as F
    from .vendor import THREADS
    path = target(OUT / "operator" / "relu6-layout.json", args.tag)
    torch.set_num_threads(THREADS)
    generator = torch.Generator().manual_seed(20261001)
    values = (torch.rand((2, 16, 12, 12), generator=generator) * 16.0 - 8.0)
    scale, zero = 0.05, 100
    cases = []
    for engine in ("x86", "qnnpack"):
        torch.backends.quantized.engine = engine
        for layout in ("contiguous", "channels_last"):
            def fresh():
                q = torch.quantize_per_tensor(values, scale, zero, torch.quint8)
                return q.contiguous(memory_format=torch.channels_last) if layout == "channels_last" else q
            for inplace in (False, True):
                q = fresh()
                before = q.int_repr().clone().int()
                expected = before.clamp(zero, min(255, zero + int(round(6.0 / scale))))
                with torch.inference_mode():
                    out = nnq.ReLU6(inplace=inplace)(q)
                codes = out.int_repr().int()
                cases.append({"operator": "ReLU6", "engine": engine, "layout": layout, "inplace": inplace,
                              "input_channels_last": _channels_last(fresh()),
                              "fraction_codes_not_equal_to_clamp": float((codes != expected).float().mean()),
                              "max_code_error": int((codes - expected).abs().max()), **_layout(codes, expected)})
            for name, module, function in (("Hardswish", nnq.Hardswish(0.03, 12), F.hardswish),
                                           ("Hardsigmoid", torch.nn.Hardsigmoid(), F.hardsigmoid)):
                q = fresh()
                reference = function(q.dequantize())
                with torch.inference_mode():
                    out = module(q)
                expected = torch.quantize_per_tensor(reference, float(out.q_scale()), int(out.q_zero_point()), out.dtype)
                error = (out.int_repr().int() - expected.int_repr().int()).abs()
                cases.append({"operator": name, "engine": engine, "layout": layout, "inplace": False,
                              "input_channels_last": _channels_last(fresh()), "max_code_error": int(error.max()),
                              "fraction_code_error_above_1": float((error > 1).float().mean())})
    wall = time.monotonic() - START
    record = {"check": "quantized_activation_operators", "tensor": {"shape": list(values.shape), "seed": 20261001,
                                                                    "scale": scale, "zero_point": zero},
              "method": "random quantized tensor; ReLU6 output codes against the exact integer clamp; Hardswish and "
                        "Hardsigmoid against the FP32 function of the dequantized input requantized with the output "
                        "parameters (a correct kernel differs by at most one code)",
              "cases": cases, "runtime": {**runtime("cpu"), "threads": THREADS},
              "verify_source_sha256": file_hash(ROOT / "tools/experiment_b2/verify.py"),
              "cost": {"job_wall_seconds": wall, "gpu_seconds": 0.0}}
    seal(path, record)
    print(json.dumps({"record": str(path.relative_to(ROOT)), "cases": cases}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for label in ("bias-residual", "batch-sensitivity"):
        p = sub.add_parser(label)
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--format", default="int8")
        p.add_argument("--recipe", default="default")
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        p.add_argument("--tag", default="")
        p.add_argument("--batches", default="8,32,1", help="batch-sensitivity only; the first must be the protocol batch")
    for label in ("vendor-check", "vendor-probe"):
        p = sub.add_parser(label)
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--qconfig", default="x86_default")
        p.add_argument("--calibration-images", type=int, default=2000)
        p.add_argument("--tag", default="")
    p = sub.add_parser("relu6-operator")
    p.add_argument("--tag", default="")
    args = parser.parse_args()
    return {"bias-residual": bias_residual, "vendor-check": vendor_check, "vendor-probe": vendor_probe,
            "batch-sensitivity": batch_sensitivity, "relu6-operator": relu6_operator}[args.command](args)
