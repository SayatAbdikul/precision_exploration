"""Arms on top of the frozen B2 default: wide (FP32) boundaries, affine and per-channel activation codes.

``prepare_shared`` repeats ``tools.experiment_b2.engine.prepare_b2`` step by step with the same functions on
the same inputs (activation MSE scales of every quantizing node, per-channel weight MSE), but stops before bias
correction, so that many arms can share it.  ``build_arm`` then applies an arm and runs the frozen sequential
bias correction for that arm's boundary plan.  With the empty arm the result is the B2 default engine; the
``verify`` job of ``runner.py`` checks this bit for bit against ``runner.build`` and the sealed matrix cell.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib

import numpy as np
import torch
from torch import nn

from tools.experiment_b.common import digest
from tools.experiment_b2.boundaries import analyze, quantizing_nodes
from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2.engine import B2Interpreter, bias_correct
from tools.experiment_b2.scales import mse_search

from . import quant
from .groups import consumers


@dataclass(frozen=True)
class Arm:
    """One arm: boundaries kept wide, boundaries with an affine code, boundaries with per-channel scales."""
    name: str
    wide: tuple = ()
    affine: tuple = ()
    per_channel: tuple = ()
    kind: str = "diagnostic"          # "reference", "diagnostic" or "recipe"
    note: str = ""

    def spec(self):
        return {"name": self.name, "wide": sorted(self.wide), "affine": sorted(self.affine),
                "per_channel": sorted(self.per_channel), "kind": self.kind}


@dataclass
class Shared:
    model: str
    format: str
    recipe: object
    plan: dict
    quantizers: dict
    activation_scales: dict
    activation_info: dict
    weight_graph: object
    weight_scales: dict
    weight_info: dict
    fp_graph: object
    arrays: dict
    shapes: dict
    device: str
    affine_cache: dict = field(default_factory=dict)
    channel_cache: dict = field(default_factory=dict)

    def fingerprint(self):
        """Digest of everything an arm inherits: activation scales, weight scales and quantized weights."""
        h = hashlib.sha256()
        for name, module in sorted(self.weight_graph.named_modules(), key=lambda kv: kv[0]):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h.update(name.encode())
                h.update(module.weight.detach().cpu().numpy().tobytes())
                h.update(module.bias.detach().cpu().numpy().tobytes())
        return digest({"activation_scales": self.activation_scales, "weight_scales": self.weight_scales,
                       "weights_and_biases_sha256": h.hexdigest()})


def node_shapes(graph, probe):
    """Output shape of every FX node on a probe batch (FP32 graph)."""
    shapes = {}

    class Probe(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if isinstance(result, torch.Tensor):
                shapes[node.name] = tuple(result.shape)
            return result

    with torch.inference_mode():
        Probe(graph).run(probe)
    return shapes


def prepare_shared(model, graph, name, recipe, arrays, maxima, device, probe):
    if (recipe.activation_range, recipe.weight_range, recipe.equalization) != ("mse", "mse", "none"):
        raise ValueError("attribution arms are defined on top of the B2 default recipe only")
    plan = analyze(graph, recipe, name)
    quantizers = {"signed": TableQuantizer(name, "signed", device)}
    if any(row["signedness"] == "unsigned" for row in plan.values()):
        quantizers["unsigned"] = TableQuantizer(name, "unsigned", device)
    result = copy.deepcopy(graph)
    activation_scales, activation_info = {}, {}
    with torch.inference_mode():
        for node in quantizing_nodes(plan):
            quantizer = quantizers[plan[node]["signedness"]]
            values = torch.tensor(arrays[node], dtype=torch.float32, device=device).reshape(1, -1)
            scales, info = mse_search(values, quantizer, maxima=[maxima[node]])
            activation_scales[node] = float(scales[0])
            activation_info[node] = {k: v[0] for k, v in info.items()}
    weight_scales, weight_info = {}, {}
    signed = quantizers["signed"]
    with torch.inference_mode(), torch.no_grad():
        for node in result.graph.nodes:
            if node.op != "call_module":
                continue
            module = result.get_submodule(node.target)
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            found, info = mse_search(module.weight.detach().reshape(module.weight.shape[0], -1), signed)
            scales = [float(x) for x in found]
            weight_info[node.name] = {"median_ratio_to_maxabs_scale": float(np.median(info["ratio_to_maxabs_scale"])),
                                      "min_ratio_to_maxabs_scale": float(np.min(info["ratio_to_maxabs_scale"]))}
            shaped = torch.tensor(scales, dtype=torch.float32, device=device).reshape((-1,) + (1,) * (module.weight.ndim - 1))
            module.weight.copy_(signed(module.weight, shaped))
            weight_scales[node.name] = scales
    shapes = node_shapes(graph, probe)
    return Shared(model, name, recipe, plan, quantizers, activation_scales, activation_info, result, weight_scales,
                  weight_info, graph, arrays, shapes, device)


def affine_params(shared, node):
    if node not in shared.affine_cache:
        row = shared.plan[node]
        if not row["quantizes"] or row["signedness"] != "signed":
            shared.affine_cache[node] = (None, {"reason": "not a signed quantizing boundary"})
        else:
            values = torch.tensor(shared.arrays[node], dtype=torch.float32, device=shared.device)
            with torch.inference_mode():
                shared.affine_cache[node] = quant.affine_search(values, shared.quantizers["signed"],
                                                                shared.activation_scales[node])
    return shared.affine_cache[node]


def channel_params(shared, node, images=2000):
    if node not in shared.channel_cache:
        row = shared.plan[node]
        channels = shared.shapes[node][1]
        samples, info = quant.channel_samples(shared.arrays[node], channels, images)
        with torch.inference_mode():
            scales, search = quant.per_channel_search(samples, shared.quantizers[row["signedness"]], shared.device)
        shared.channel_cache[node] = (scales, {**info, **search, "channels": int(channels),
                                               "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
                                               "scale_spread_log2": float(np.log2(scales.max() / scales.min()))})
    return shared.channel_cache[node]


def _fold(shared, q_graph, consumer, ratio):
    """Fold per-input-channel activation scale ratios into a consumer's weights before weight quantization.

    Hardware stores codes ``q_c`` with scale ``s_c = r_c * s_ref`` and the consumer uses weights
    ``W'[o, c] = Qw(W[o, c] * r_c)`` with one scale per output channel; the QDQ surrogate computes the same
    sum by using ``Qw(W * r) / r`` against the dequantized input.
    """
    node = {n.name: n for n in q_graph.graph.nodes}[consumer]
    module = q_graph.get_submodule(node.target)
    original = shared.fp_graph.get_submodule(node.target).weight.detach()
    ratio = torch.as_tensor(ratio, dtype=torch.float32, device=original.device)
    if isinstance(module, nn.Linear):
        shape = (1, -1)
    elif module.groups == 1:
        shape = (1, -1, 1, 1)
    elif module.groups == module.in_channels == module.out_channels:
        shape = (-1, 1, 1, 1)
    else:
        raise ValueError(f"cannot fold per-channel scales into {consumer}")
    folded = original * ratio.reshape(shape)
    signed = shared.quantizers["signed"]
    found, _ = mse_search(folded.reshape(folded.shape[0], -1), signed)
    shaped = torch.tensor([float(x) for x in found], dtype=torch.float32,
                          device=original.device).reshape((-1,) + (1,) * (folded.ndim - 1))
    quantized = signed(folded, shaped)
    with torch.no_grad():
        module.weight.copy_(quantized / ratio.reshape(shape))
    abs_sum, total = code_sums(quantized, shaped)
    return {"consumer": consumer, "groups": int(getattr(module, "groups", 1)),
            "weight_scales_sha256": hashlib.sha256(np.asarray(found, np.float32).tobytes()).hexdigest(),
            "code_abs_sum": abs_sum, "code_sum": total}


def code_sums(quantized, shaped):
    """Per-output-channel sum of |weight code| and sum of weight codes (codes = quantized weight / its scale)."""
    raw = quantized.double() / shaped.double()
    codes = torch.round(raw)
    if float((codes - raw).abs().max()) > 1e-3:
        return None, None  # non-integer codebook levels: no integer certificate
    dims = tuple(range(1, codes.ndim))
    return [int(v) for v in codes.abs().sum(dim=dims)], [int(v) for v in codes.sum(dim=dims)]


def _input_boundary(graph, plan, node):
    from tools.experiment_b2.boundaries import _producer
    kinds = {k: row["kind"] for k, row in plan.items()}
    return _producer(node.all_input_nodes[0], kinds).name


def accumulator_widths(shared, plan, quantizers, folds):
    """Certified absolute accumulator width of the MACs an arm changes (integer codebooks only).

    Closed form of ``tools/scaled_bridge_v2/certificates.py`` (``signed_bits_absolute``):
    ``bit_length(max|a| * sum|w|) + 1`` per output channel, maximum over channels, with ``a`` the input code
    range and ``w`` the integer weight codes.  An affine input ``x = s*(q - z)`` is accumulated as
    ``sum w*q - z*sum(w)`` with the per-output-channel term added to the accumulator, so its prefix bound is
    ``max|q| * sum|w| + |z| * |sum w|``; subtracting ``z`` from the operand first instead (a ``|q - z|``-wide
    operand) gives ``max|q - z| * sum|w|``.  Both are reported.
    """
    if shared.quantizers["signed"].name not in ("int8", "int6", "int5", "int4"):
        return None
    folded = {f["consumer"]: f for f in folds}
    layers, network_default, network_arm = {}, 0, 0
    for node in shared.weight_graph.graph.nodes:
        if node.op != "call_module" or not isinstance(shared.weight_graph.get_submodule(node.target), (nn.Conv2d, nn.Linear)):
            continue
        source = _input_boundary(shared.weight_graph, shared.plan, node)
        default_row, row = shared.plan[source], plan[source]
        if not default_row["quantizes"]:
            continue  # a MAC fed by a wide boundary has no integer certificate in either case
        module = shared.weight_graph.get_submodule(node.target)
        shaped = torch.tensor(shared.weight_scales[node.name], dtype=torch.float32,
                              device=module.weight.device).reshape((-1,) + (1,) * (module.weight.ndim - 1))
        abs_sum, total = code_sums(module.weight.detach(), shaped)
        levels = shared.quantizers[default_row["signedness"]].levels
        max_a = int(max(abs(float(levels[0])), abs(float(levels[-1]))))
        default_bound = max_a * max(abs_sum)
        bits_default = default_bound.bit_length() + 1
        network_default = max(network_default, bits_default)
        if not row["quantizes"]:
            continue
        key = row["signedness"]
        entry = None
        if key.startswith("affine:"):
            z = int(-quantizers[key].zero)
            top = int(float(levels[-1]))
            bound = max(max_a * a + z * abs(t) for a, t in zip(abs_sum, total))
            pre = (top + z) * max(abs_sum)
            entry = {"input": source, "zero_point": -z, "bits_default": bits_default,
                     "bits_arm": bound.bit_length() + 1, "bits_arm_presubtracted_operand": pre.bit_length() + 1}
        elif node.name in folded and folded[node.name]["code_abs_sum"] is not None:
            bound = max_a * max(folded[node.name]["code_abs_sum"])
            entry = {"input": source, "folded": True, "bits_default": bits_default, "bits_arm": bound.bit_length() + 1}
        if entry is not None:
            layers[node.name] = entry
            network_arm = max(network_arm, entry["bits_arm"])
        else:
            network_arm = max(network_arm, bits_default)
    return {"rule": "signed_bits_absolute (tools/scaled_bridge_v2/certificates.py)", "layers": layers,
            "network_max_bits_default": network_default, "network_max_bits_arm": network_arm}


def build_arm(shared, arm, bias_inputs, audit=False):
    """Interpreter and metadata of one arm (bias correction refitted for the arm's boundary plan)."""
    plan = {k: dict(v) for k, v in shared.plan.items()}
    for node in arm.wide:
        if not plan[node]["quantizes"]:
            raise ValueError(f"{node} is not a quantizing boundary")
        plan[node].update(quantizes=False, reason="attrib_wide", signedness=None)
    if set(arm.affine) & set(arm.per_channel) or (set(arm.affine) | set(arm.per_channel)) & set(arm.wide):
        raise ValueError("a boundary can carry only one arm change")
    quantizers = dict(shared.quantizers)
    scales = {node: shared.activation_scales[node] for node in quantizing_nodes(plan)}
    affine_meta, channel_meta, folds = {}, {}, []
    for node in sorted(arm.affine):
        params, info = affine_params(shared, node)
        affine_meta[node] = {**info, "applied": params is not None}
        if params is None:
            continue
        key = f"affine:{node}"
        quantizers[key] = quant.AffineQuantizer(shared.quantizers["signed"], params["scale"], params["zero"])
        plan[node]["signedness"] = key
        scales[node] = quantizers[key].scale
    q_graph = copy.deepcopy(shared.weight_graph)
    for node in sorted(arm.per_channel):
        row = plan[node]
        if not row["quantizes"]:
            raise ValueError(f"{node} is not a quantizing boundary")
        channel_scales, info = channel_params(shared, node)
        reference = shared.activation_scales[node]
        key = f"per_channel:{node}"
        quantizers[key] = quant.PerChannelQuantizer(shared.quantizers[row["signedness"]], channel_scales, reference)
        plan[node]["signedness"] = key
        scales[node] = quantizers[key].reference
        users = consumers(shared.fp_graph, shared.plan, node)
        for user, label in users:
            if label in ("conv", "linear"):
                folds.append(_fold(shared, q_graph, user, np.asarray(channel_scales, np.float32) / np.float32(reference)))
        channel_meta[node] = {**info, "consumers": users}
    if len({f["consumer"] for f in folds}) != len(folds):
        raise ValueError("a consumer would be folded twice")
    report = None
    if shared.recipe.bias_correction == "empirical":
        with torch.inference_mode():
            report = bias_correct(shared.fp_graph, q_graph, plan, quantizers, scales, bias_inputs)
    interpreter = B2Interpreter(q_graph, quantizers, plan, scales, audit=audit)
    widths = accumulator_widths(shared, plan, quantizers, folds) if (arm.affine or arm.per_channel) else None
    meta = {"arm": arm.spec(), "quantizing_boundaries": len(scales), "wide_boundaries": len(arm.wide),
            "affine": affine_meta, "per_channel": channel_meta, "folds": folds,
            "affine_applied": sorted(k for k, v in affine_meta.items() if v["applied"]),
            "bias_correction_max_abs": max((v["max_abs_correction"] for v in report.values()), default=None) if report else None,
            "bias_correction_digest": digest(report) if report else None, "accumulator": widths}
    for fold in folds:
        fold.pop("code_abs_sum", None), fold.pop("code_sum", None)
    return interpreter, meta
