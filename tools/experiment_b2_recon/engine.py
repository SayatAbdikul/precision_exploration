"""B2 engine with a separate weight format and an optional learned rounding.

``tools.experiment_b2.engine.prepare_b2`` takes one format for weights and
activations and always rounds weights to nearest.  This builder repeats its
steps with the same B2 functions (boundary analysis, quantizers, MSE scale
search, sequential bias correction, interpreter) and adds two things: the
weight codebook may differ from the activation codebook (INT4 weights with
INT8 activations), and the weight of a layer may be replaced by a learned
up/down decision on the same grid.  With one format and no learned rounding
it must give the B2 engine bit for bit (checked by ``evaluate.py`` on every
nearest arm and by the unit tests).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from tools.experiment_b.quantizer import scale_for, threshold
from tools.experiment_b2.boundaries import analyze, kind, quantizing_nodes
from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2.engine import B2Interpreter, bias_correct
from tools.experiment_b2.scales import mse_search, scale_for_top, simple_span

from .adaround import hard_weight, neighbours

SCALE_RULES = ("mse_per_channel", "maxabs_per_channel", "mse_per_layer")
ACTIVATIONS = {"relu": torch.relu, "relu6": nn.functional.relu6, "hardswish": nn.functional.hardswish,
               "hardsigmoid": nn.functional.hardsigmoid}
PASSTHROUGH = ("identity", "flatten")


def weight_layers(graph):
    """Conv/linear nodes of an FX graph in topological order: ``[(node, module)]``."""
    result = []
    for node in graph.graph.nodes:
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                result.append((node, module))
    return result


def consumer_activation(graph, node):
    """``(name, f_a)``: the nonlinearity that is the only consumer of ``node`` (through passthroughs), else identity."""
    kinds = {n.name: kind(graph, n) for n in graph.graph.nodes}

    def users(n):
        found = []
        for user in n.users:
            found.extend(users(user) if kinds[user.name] in PASSTHROUGH else [user])
        return found

    consumers = users(node)
    if len(consumers) == 1 and kinds[consumers[0].name] in ACTIVATIONS:
        label = kinds[consumers[0].name]
        return label, ACTIVATIONS[label]
    return "identity", (lambda value: value)


def weight_scale(module, wformat, rule, quantizer):
    """FP32 scale of every output channel, shaped to broadcast against the weight, as B2 computes it."""
    weight = module.weight.detach()
    channels = weight.shape[0]
    if rule == "mse_per_channel":
        found, _ = mse_search(weight.reshape(channels, -1), quantizer)
        scales = [float(x) for x in found]
    elif rule == "maxabs_per_channel":
        scales = [scale_for(threshold(channel, "maxabs"), wformat) for channel in weight.cpu().numpy()]
    elif rule == "mse_per_layer":
        found, _ = mse_search(weight.reshape(1, -1), quantizer)
        scales = [float(found[0])] * channels
    else:
        raise ValueError(f"unknown weight scale rule: {rule}")
    shaped = torch.tensor(scales, dtype=torch.float32, device=weight.device).reshape((-1,) + (1,) * (weight.ndim - 1))
    return scales, shaped


def recipe_scale_rule(recipe):
    return {"mse": "mse_per_channel", "maxabs": "maxabs_per_channel"}[recipe.weight_range]


def activation_setup(graph, aformat, recipe, arrays, maxima, device):
    """Boundary plan, activation quantizers and activation scales: the first half of ``prepare_b2``."""
    if recipe.equalization != "none":
        raise ValueError("equalization is not supported by the reconstruction builder")
    plan = analyze(graph, recipe, aformat)
    quantizers = {"signed": TableQuantizer(aformat, "signed", device)}
    if any(row["signedness"] == "unsigned" for row in plan.values()):
        quantizers["unsigned"] = TableQuantizer(aformat, "unsigned", device)
    scales = {}
    v1_like = recipe.v1_equivalent() is not None
    for node in quantizing_nodes(plan):
        quantizer = quantizers[plan[node]["signedness"]]
        if recipe.activation_range == "mse":
            values = torch.tensor(arrays[node], dtype=torch.float32, device=device).reshape(1, -1)
            found, _ = mse_search(values, quantizer, maxima=[maxima[node]])
            scales[node] = float(found[0])
        elif v1_like:
            scales[node] = scale_for(maxima[node] if recipe.activation_range == "maxabs"
                                     else threshold(arrays[node], recipe.activation_range), aformat)
        else:
            scales[node] = scale_for_top(simple_span(arrays[node], maxima[node], recipe.activation_range), quantizer.top)
    return plan, quantizers, scales


def quantize_weights(result, wformat, rule, device, learned=None):
    """Replace every conv/linear weight of ``result`` in place; returns the per-layer scales and statistics.

    ``learned``: ``{node name: bool tensor (True = upper neighbour)}`` or ``None`` for round-to-nearest.
    """
    signed = TableQuantizer(wformat, "signed", device)
    scales, report = {}, {}
    with torch.no_grad():
        for node, module in weight_layers(result):
            values, shaped = weight_scale(module, wformat, rule, signed)
            nearest = signed(module.weight, shaped)
            if learned is None:
                module.weight.copy_(nearest)
            else:
                lo, hi, _ = neighbours(module.weight, shaped, signed.levels)
                up = learned[node.name].to(device)
                if up.shape != module.weight.shape or up.dtype != torch.bool:
                    raise ValueError(f"learned rounding of {node.name} has the wrong shape or type")
                chosen = hard_weight(lo, hi, up & (hi > lo), shaped)
                report[node.name] = {"weights": int(chosen.numel()),
                                     "differ_from_nearest": int((chosen != nearest).sum())}
                module.weight.copy_(chosen)
            scales[node.name] = values
    return scales, report


def fp32_plan(graph):
    """A boundary plan in which no tensor quantizes (activations stay FP32)."""
    return {node.name: {"kind": kind(graph, node), "quantizes": False, "reason": "fp32_activations",
                        "nonnegative": False, "signedness": None}
            for node in graph.graph.nodes if node.op != "output"}


def build_engine(graph, wformat, aformat, recipe, arrays, maxima, device, *, weight_rule=None, learned=None,
                 bias_inputs=None, bias_correction=None):
    """A B2 engine: activations per ``recipe`` on ``aformat``; weights on ``wformat``.

    ``aformat=None`` leaves every activation in FP32 (weights only); the weight
    scale rule must then be given, and ``bias_correction`` ("none" or
    "empirical") replaces the recipe switch.  Returns ``(runner, metadata,
    graph)``: ``runner(tensor)`` gives the logits, ``graph`` is the module that
    holds the deployed weights and biases.
    """
    result = copy.deepcopy(graph)
    rule = weight_rule or recipe_scale_rule(recipe)
    with torch.no_grad():
        weight_scales, rounding = quantize_weights(result, wformat, rule, device, learned)
    metadata = {"weight_scales": weight_scales, "weight_scale_rule": rule, "rounding": rounding}
    correction = bias_correction if bias_correction is not None else (recipe.bias_correction if recipe else "none")
    if aformat is None:
        plan, quantizers, activation_scales = fp32_plan(graph), {"signed": TableQuantizer(wformat, "signed", device)}, {}
    else:
        plan, quantizers, activation_scales = activation_setup(graph, aformat, recipe, arrays, maxima, device)
    if correction == "empirical":
        if bias_inputs is None:
            raise ValueError("bias correction needs calibration inputs")
        with torch.inference_mode():
            metadata["bias_correction"] = bias_correct(graph, result, plan, quantizers, activation_scales, bias_inputs)
    elif correction != "none":
        raise ValueError(f"unknown bias correction: {correction}")
    metadata["activation_scales"] = activation_scales
    if aformat is None:
        result.eval()
        return result, metadata, result
    interpreter = B2Interpreter(result, quantizers, plan, activation_scales)
    return interpreter.run, metadata, result


def state_digest(graph):
    """SHA-256 over every conv/linear weight and bias of a graph (FP32 bytes), for identity records."""
    import hashlib
    h = hashlib.sha256()
    for node, module in weight_layers(graph):
        h.update(node.name.encode())
        h.update(np.ascontiguousarray(module.weight.detach().cpu().numpy(), dtype="<f4").tobytes())
        if module.bias is not None:
            h.update(np.ascontiguousarray(module.bias.detach().cpu().numpy(), dtype="<f4").tobytes())
    return h.hexdigest()
