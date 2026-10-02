"""YOLOv8n quantize-dequantize engine with recipe-controlled boundaries (Experiment B2 detector).

The graph, the operators (``_run_op``), the calibration observations and the block semantics are the v1
ones (``tools.experiment_b_ext``), imported and never modified.  With every switch off the engine performs
the same operations on the same tensors in the same order as v1's ``DetectorEngine``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, unseal
from tools.experiment_b.quantizer import threshold
from tools.experiment_b2.codebook import TableQuantizer, supports_unsigned
from tools.experiment_b2.scales import SEARCH, mse_search, scale_for_top, simple_span
from tools.experiment_b_ext.detector import _run_op
from tools.experiment_b_ext.shared import BlockQuantizer, block_conv2d, quantize_axis

BLOCK_FORMATS = frozenset(("bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"))
BIAS_CORRECTION_IMAGES = 64
V1_CALIBRATION = ROOT / "artifacts/experiment_b_ext/calibration"
FORWARDING = ("slice", "maxpool", "resize")


def kind(node):
    op, attrs = node["op"], node["attrs"]
    if op == "conv2d":
        return "conv"
    if op == "lut":
        return attrs["function"]
    if op == "elementwise":
        return attrs["operation"]
    if op == "pool2d" and attrs["kind"] == "max":
        return "maxpool"
    table = {"concatenate": "concat", "channel_slice": "slice", "resize_nearest": "resize", "flatten": "flatten",
             "dfl": "dfl", "decode_boxes": "decode"}
    if op not in table:
        raise ValueError(f"unsupported detector operation: {op}")
    return table[op]


def analyze(graph, recipe, format_name):
    """``{name: boundary}`` for the input and every node: does it quantize, why not, signedness, scale groups."""
    nodes = graph["nodes"]
    names = {"images"} | {n["name"] for n in nodes}
    kinds = {"images": "input", **{n["name"]: kind(n) for n in nodes}}
    users = {name: [] for name in names}
    for node in nodes:
        for source in node["inputs"]:
            if source in names:
                users[source].append(node["name"])
    block = format_name in BLOCK_FORMATS
    forward = "none" if block else recipe.passthrough
    unsigned_ok = recipe.unsigned and not block and supports_unsigned(format_name)
    plan = {"images": {"kind": "input", "quantizes": recipe.quantize_input and recipe.quantize_activations,
                       "reason": None if recipe.quantize_input and recipe.quantize_activations else "input_exempt",
                       "nonnegative": True, "groups": None}}
    plan["images"]["on_grid"] = plan["images"]["quantizes"]
    for node in nodes:
        name, label = node["name"], kinds[node["name"]]
        sources = [plan[x] for x in node["inputs"] if x in names]
        source_kinds = [kinds[x] for x in node["inputs"] if x in names]
        user_kinds = [kinds[x] for x in users[name]]
        join = label == "concat" and any(k in ("decode", "flatten") for k in source_kinds)
        if label in ("sigmoid", "dfl"):
            nonnegative = True
        elif label in ("maxpool", "resize", "slice", "concat", "flatten", "add", "mul"):
            nonnegative = bool(sources) and all(x["nonnegative"] for x in sources) and not join
        else:
            nonnegative = False
        quantizes, reason, groups, on_grid = True, None, None, None
        if label == "flatten":
            quantizes, reason, on_grid = False, "passthrough", sources[0]["on_grid"]
        elif label == "conv" and "dfl" in user_kinds:
            quantizes, reason = recipe.q_box_logits, "head_exempt"
        elif label == "conv" and "sigmoid" in user_kinds:
            quantizes, reason = recipe.q_class_logits, "head_exempt"
        elif label == "conv" and recipe.boundaries == "fused_silu" and user_kinds == ["silu"]:
            quantizes, reason = False, f"fused_into:{users[name][0]}"
        elif label == "dfl":
            quantizes, reason = recipe.q_dfl, "head_exempt"
        elif label == "decode":
            quantizes, reason = recipe.q_boxes, "head_exempt"
        elif label == "sigmoid":
            quantizes, reason = recipe.q_scores, "head_exempt"
        elif join:
            groups = [g for g, flag in (("boxes", recipe.q_boxes), ("scores", recipe.q_scores)) if flag]
            quantizes, reason = recipe.q_joins and bool(groups), "head_exempt"
            if block and quantizes and len(groups) != 2:
                raise ValueError("block formats quantize an 84-channel join as a whole or not at all")
        elif label in FORWARDING and forward != "none" and sources[0]["on_grid"]:
            quantizes, reason, on_grid = False, "code_passthrough", True
        elif label == "concat" and forward == "nonarith_concat" and all(x["on_grid"] for x in sources):
            quantizes, reason, on_grid = False, "concat_passthrough", True
        if not recipe.quantize_activations and quantizes:
            quantizes, reason = False, "activations_exempt"
        if quantizes:
            reason = None
        plan[name] = {"kind": label, "quantizes": quantizes, "reason": reason, "nonnegative": nonnegative,
                      "groups": groups if quantizes else None,
                      "on_grid": quantizes if on_grid is None else on_grid,
                      "signedness": ("unsigned" if unsigned_ok and nonnegative else "signed") if quantizes else None}
    first = plan["images"]
    first["signedness"] = ("unsigned" if unsigned_ok else "signed") if first["quantizes"] else None
    return plan


def weight_names(graph):
    """Convolution weight constants, and the projection constants of the box expectation."""
    conv = [n["inputs"][1] for n in graph["nodes"] if n["op"] == "conv2d"]
    projection = [n["inputs"][1] for n in graph["nodes"] if n["op"] == "dfl"]
    return conv, projection


def v1_calibration(graph, context):
    """Sealed v1 FP32 observations of the detector: ``(samples, maxima, identity)``; read-only."""
    from tools.experiment_b_ext import runner as ext
    record, rows, _ = dataset("coco_calibration_2k")
    matching = []
    for folder in sorted(p for p in V1_CALIBRATION.iterdir() if p.is_dir()):
        provenance = unseal(folder / "provenance.json")
        if (digest(provenance) == folder.name and provenance["source_sha256"] == ext.source_identity()
                and provenance["protocol"] == ext.PROTOCOL and provenance["model_context"] == context
                and provenance["graph_sha256"] == digest(graph["nodes"]) and provenance["list_sha256"] == record["sha256"]):
            matching.append(folder)
    if len(matching) != 1:
        raise ValueError(f"expected exactly one current v1 detector calibration cache, found {len(matching)}")
    folder = matching[0]
    summary = unseal(folder / "summary.json")
    if summary["identity"] != folder.name or summary["images"] != len(rows):
        raise ValueError("v1 detector calibration summary does not match the frozen calibration list")
    combined, maxima = {}, {}
    for start in range(0, len(rows), 8):
        meta = unseal(folder / f"{start:05d}.json")
        npz = folder / f"{start:05d}.npz"
        if meta["identity"] != folder.name or meta["samples"] != rows[start:start + 8] or file_hash(npz) != meta["npz_sha256"]:
            raise ValueError("v1 detector calibration checkpoint mismatch")
        with np.load(npz, allow_pickle=False) as saved:
            for key in saved.files:
                combined.setdefault(key, []).append(saved[key])
        for key, value in meta["maxima"].items():
            maxima[key] = max(maxima.get(key, 0), value)
    samples = {key: np.concatenate(chunks) for key, chunks in combined.items()}
    for key, values in samples.items():
        if summary["nodes"][key]["sample_count"] != values.size or summary["nodes"][key]["maxabs"] != maxima[key]:
            raise ValueError("v1 detector calibration arrays do not reproduce the sealed summary")
    return samples, maxima, {"identity": folder.name, "summary_sha256": digest(summary),
                             "reused_from": "artifacts/experiment_b_ext/calibration (read-only)"}


class B2DetectorEngine:
    def __init__(self, graph, constants, device, *, plan=None, quantizers=None, scales=None, block=None):
        self.graph, self.constants, self.device = graph, constants, device
        self.plan, self.quantizers, self.scales, self.block = plan, quantizers, scales or {}, block
        self.bias = {n["name"]: torch.tensor([float(v) for v in n["attrs"]["bias"]], device=device)
                     for n in graph["nodes"] if n["op"] == "conv2d" and "bias" in n["attrs"]}

    def qdq(self, name, value):
        if self.plan is None or not self.plan[name]["quantizes"]:
            return value
        if self.block is not None:
            return quantize_axis(value, self.block, 1)
        row, scale = self.plan[name], self.scales[name]
        quantizer = self.quantizers[row["signedness"]]
        if row["groups"] is not None:
            if value.shape[1] != 84:
                raise ValueError("invalid detector head join")
            boxes, scores = value[:, :4], value[:, 4:]
            if "boxes" in scale:
                boxes = quantizer(boxes, scale["boxes"])
            if "scores" in scale:
                scores = quantizer(scores, scale["scores"])
            return torch.cat((boxes, scores), dim=1)
        return quantizer(value, scale)

    def op(self, node, lookup):
        """The raw output of one node; ``lookup(name)`` returns an input tensor."""
        if node["op"] == "conv2d":
            source, weight, attrs = lookup(node["inputs"][0]), self.constants[node["inputs"][1]], node["attrs"]
            bias = self.bias.get(node["name"])
            if self.block is not None:
                module = SimpleNamespace(in_channels=source.shape[1], out_channels=weight.shape[0],
                                         kernel_size=weight.shape[-2:], stride=attrs.get("stride", [1, 1]),
                                         padding=attrs.get("padding", [0, 0]), dilation=attrs.get("dilation", [1, 1]),
                                         groups=attrs.get("groups", 1), padding_mode="zeros", bias=bias)
                return block_conv2d(source, module, self.block, weight)
            return F.conv2d(source, weight, bias, stride=attrs.get("stride", 1), padding=attrs.get("padding", 0),
                            dilation=attrs.get("dilation", 1), groups=attrs.get("groups", 1))
        values = {name: (self.constants[name] if name in self.constants else lookup(name)) for name in node["inputs"]}
        return _run_op(node, values, self.constants, self.device)

    def run(self, inputs, capture=None):
        if tuple(inputs.shape[1:]) != (3, 640, 640):
            raise ValueError("detector requires fixed 640-square input")
        values = {"images": self.qdq("images", inputs)}
        for node in self.graph["nodes"]:
            values[node["name"]] = self.qdq(node["name"], self.op(node, values.__getitem__))
            if capture is not None and node["name"] in capture:
                capture[node["name"]] = values[node["name"]]
        output = values[self.graph["outputs"][0]]
        if tuple(output.shape) != (len(inputs), 84, 8400) or not torch.isfinite(output).all():
            raise ValueError("invalid detector output")
        return output


def bias_correct(reference, engine, inputs, chunk=8):
    """Sequential empirical bias correction, in place on ``engine.bias``.

    Both engines run node by node over all calibration inputs.  At every convolution the per-channel mean
    of the FP32 output minus the mean of the quantized-network output is added to the bias and the
    quantized output is recomputed with the corrected bias, so the correction is exact for the engine
    that is then executed and every later layer sees corrected inputs.
    """
    nodes = engine.graph["nodes"]
    remaining = {"images": 0, **{n["name"]: 0 for n in nodes}}
    for node in nodes:
        for source in node["inputs"]:
            if source in remaining:
                remaining[source] += 1
    chunks = [inputs[i:i + chunk] for i in range(0, len(inputs), chunk)]
    env_fp, env_q, report = {"images": chunks}, {"images": [engine.qdq("images", c) for c in chunks]}, {}
    for node in nodes:
        name = node["name"]
        fp_out = [reference.op(node, lambda key, i=i: env_fp[key][i]) for i in range(len(chunks))]
        q_raw = [engine.op(node, lambda key, i=i: env_q[key][i]) for i in range(len(chunks))]
        if node["op"] == "conv2d":
            if name not in engine.bias:
                raise ValueError("bias correction expects folded biases")
            count = sum(t.numel() // t.shape[1] for t in fp_out)
            mean_fp = sum(t.double().sum(dim=(0, 2, 3)) for t in fp_out) / count
            mean_q = sum(t.double().sum(dim=(0, 2, 3)) for t in q_raw) / count
            delta = (mean_fp - mean_q).to(torch.float32)
            engine.bias[name] = engine.bias[name] + delta
            q_raw = [engine.op(node, lambda key, i=i: env_q[key][i]) for i in range(len(chunks))]
            residual = mean_fp - sum(t.double().sum(dim=(0, 2, 3)) for t in q_raw) / count
            report[name] = {"max_abs_correction": float(delta.abs().max()),
                            "rms_correction": float(delta.double().pow(2).mean().sqrt()),
                            "rms_fp32_channel_mean": float(mean_fp.pow(2).mean().sqrt()),
                            "max_abs_residual_after": float(residual.abs().max())}
        env_fp[name] = fp_out
        env_q[name] = [engine.qdq(name, t) for t in q_raw]
        for source in node["inputs"]:
            if source in remaining:
                remaining[source] -= 1
                if remaining[source] == 0:
                    env_fp.pop(source, None)
                    env_q.pop(source, None)
    return report


def prepare(graph, constants, name, recipe, samples, maxima, device, *, bias_inputs=None):
    """Build the engine of one format and recipe.  Returns ``(engine, metadata)``."""
    plan = analyze(graph, recipe, name)
    conv_weights, projections = weight_names(graph)
    if set(conv_weights) | set(projections) != set(constants):
        raise ValueError("detector constant coverage mismatch")
    quantized = [k for k in conv_weights if recipe.quantize_weights] + [k for k in projections if recipe.q_projection]
    block = name in BLOCK_FORMATS
    metadata = {"boundaries": {k: {f: v[f] for f in ("kind", "quantizes", "reason", "signedness", "groups")}
                               for k, v in plan.items()}}
    if block:
        if recipe.activation_range == "mse" or recipe.weight_range == "mse":
            selection = "maxabs"
        elif recipe.activation_range != recipe.weight_range:
            raise ValueError("block formats take one block scale selection")
        else:
            selection = recipe.activation_range
        quantizer = BlockQuantizer(name, selection, device)
        mapped = {key: (quantizer(weight.reshape(weight.shape[0], -1)).reshape_as(weight) if key in quantized else weight)
                  for key, weight in constants.items()}
        engine = B2DetectorEngine(graph, mapped, device, plan=plan, block=quantizer)
        metadata.update({"block_axis": "reduction_k_for_weights_and_patches; channel_axis_for_stored_activations",
                         "block_size": 32, "scale": "intrinsic_e8m0_per_block", "selection": selection})
    else:
        quantizers = {"signed": TableQuantizer(name, "signed", device)}
        if any(row["signedness"] == "unsigned" for row in plan.values()):
            quantizers["unsigned"] = TableQuantizer(name, "unsigned", device)

        def activation_scale(key, quantizer):
            if recipe.activation_range == "mse":
                values = torch.tensor(samples[key], dtype=torch.float32, device=device).reshape(1, -1)
                found, info = mse_search(values, quantizer, maxima=[maxima[key]])
                search[key] = info["ratio_to_maxabs_scale"][0]
                return float(found[0])
            return scale_for_top(simple_span(samples[key], maxima[key], recipe.activation_range), quantizer.top)

        scales, flat, search = {}, {}, {}
        for node, row in plan.items():
            if not row["quantizes"]:
                continue
            quantizer = quantizers[row["signedness"]]
            if row["groups"] is not None:
                found = {g: activation_scale(f"{node}:{g}", quantizer) for g in row["groups"]}
                scales[node] = {g: torch.tensor(v, device=device) for g, v in found.items()}
                flat[node] = found
            elif f"{node}:boxes" in samples:
                raise ValueError("84-channel node without group scales")
            else:
                flat[node] = activation_scale(node, quantizer)
                scales[node] = torch.tensor(flat[node], device=device)
        signed, weights, weight_scales = quantizers["signed"], {}, {}
        for key, weight in constants.items():
            if key not in quantized:
                weights[key] = weight
                continue
            rows = weight.reshape(weight.shape[0], -1)
            if recipe.weight_range == "mse":
                row_scales = [float(x) for x in mse_search(rows, signed)[0]]
            else:
                row_scales = [scale_for_top(threshold(row.detach().cpu().numpy(), recipe.weight_range), signed.top)
                              for row in rows]
            scale = torch.tensor(row_scales, device=device).reshape((-1,) + (1,) * (weight.ndim - 1))
            weights[key] = signed(weight, scale)
            weight_scales[key] = row_scales
        engine = B2DetectorEngine(graph, weights, device, plan=plan, quantizers=quantizers, scales=scales)
        metadata.update({"activation_scales": flat, "weight_scales_sha256": digest(weight_scales),
                         "weight_scale_count": sum(len(v) for v in weight_scales.values()), "scale_storage": "fp32"})
        if recipe.activation_range == "mse" or recipe.weight_range == "mse":
            metadata["mse_search"] = SEARCH
            metadata["activation_ratio_to_maxabs_scale"] = search
    metadata["quantized_constants"] = len(quantized)
    if recipe.bias_correction == "empirical":
        if bias_inputs is None or len(bias_inputs) != BIAS_CORRECTION_IMAGES:
            raise ValueError("bias correction needs the fixed calibration inputs")
        with torch.inference_mode():
            metadata["bias_correction"] = bias_correct(B2DetectorEngine(graph, constants, device), engine, bias_inputs)
        metadata["bias_sha256"] = digest({k: [float(x) for x in v.cpu().numpy()] for k, v in engine.bias.items()})
    return engine, metadata
