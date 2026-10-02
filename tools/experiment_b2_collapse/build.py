"""Engine builder for lane Q3: the L6 builder plus correction policies and per-layer weight-scale rules.

``tools.experiment_b2_recon.engine.build_engine`` (read-only) is the B2 engine with a
separate weight format and optional learned rounding; with one format and nearest
rounding it is bit-identical to ``prepare_b2``.  This module

* calls it unchanged for the reference arm (``correction = "b2"``: the recipe's own switch,
  i.e. ``tools.experiment_b2.engine.bias_correct``), and
* otherwise calls it with ``bias_correction="none"`` and applies
  ``correct.staged_correct`` with one policy on the deployed graph, or
* for weight-anchor arms (mode 2) quantizes each layer with its own scale rule
  (``maxabs_per_channel``, ``mse_per_channel``, ``anchor_one``: per-channel max-abs mapped to
  the level 1.0) by the same steps as ``recon.engine.quantize_weights``.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2.engine import B2Interpreter
from tools.experiment_b2.scales import scale_for_top
from tools.experiment_b2_recon.engine import (activation_setup, build_engine, fp32_plan, weight_layers,
                                              weight_scale)

from .correct import staged_correct

WEIGHT_RULES = ("maxabs_per_channel", "mse_per_channel", "anchor_one")


def anchored_scale(module, value):
    """Per-channel scale that maps the channel's max-abs weight to the level ``value``."""
    weight = module.weight.detach()
    span = weight.reshape(weight.shape[0], -1).abs().amax(dim=1).cpu().numpy()
    scales = [scale_for_top(float(s), value) for s in span]
    shaped = torch.tensor(scales, dtype=torch.float32, device=weight.device).reshape((-1,) + (1,) * (weight.ndim - 1))
    return scales, shaped


def quantize_weights_by_rule(result, wformat, rules, device):
    """``rules``: ``{node name: rule}``; nearest rounding on the signed codebook of ``wformat``."""
    signed = TableQuantizer(wformat, "signed", device)
    scales, sqnr = {}, {}
    with torch.no_grad():
        for node, module in weight_layers(result):
            rule = rules[node.name]
            if rule == "anchor_one":
                values, shaped = anchored_scale(module, 1.0)
            else:
                values, shaped = weight_scale(module, wformat, rule, signed)
            original = module.weight.detach().clone()
            module.weight.copy_(signed(module.weight, shaped))
            error = (module.weight.double() - original.double()).pow(2).sum()
            sqnr[node.name] = float(10 * torch.log10(original.double().pow(2).sum() / error)) if error > 0 else None
            scales[node.name] = values
    return scales, sqnr


def build(graph, spec, arrays, maxima, device, bias_inputs, learned=None):
    """``(runner, metadata, deployed graph)`` of one arm.

    ``spec`` keys: weight_format, activation_format (None = FP32 activations), recipe (Recipe or
    None), weight_rule (None = the recipe's), correction ("b2" | a ``correct.POLICIES`` member |
    "recipe_none"), correction_images (int), weight_rules (optional ``{node: rule}``).
    """
    correction = spec["correction"]
    recipe = spec["recipe"]
    images = bias_inputs[:spec.get("correction_images") or len(bias_inputs)]
    if spec.get("weight_rules"):
        if correction not in ("none", "recipe_none"):
            raise ValueError("weight-rule arms are defined without bias correction")
        result = copy.deepcopy(graph)
        scales, sqnr = quantize_weights_by_rule(result, spec["weight_format"], spec["weight_rules"], device)
        plan, quantizers, activation_scales = activation_setup(graph, spec["activation_format"], recipe, arrays,
                                                                maxima, device)
        metadata = {"weight_scales": scales, "weight_sqnr_db": sqnr, "activation_scales": activation_scales,
                    "weight_scale_rule": "per_layer_rules"}
        return B2Interpreter(result, quantizers, plan, activation_scales).run, metadata, result
    if correction == "b2":
        if spec.get("correction_images") not in (None, len(bias_inputs)):
            raise ValueError("the reference arm uses the full correction set")
        bias = None if recipe is not None else "empirical"
        return build_engine(graph, spec["weight_format"], spec["activation_format"], recipe, arrays, maxima, device,
                            weight_rule=spec.get("weight_rule"), learned=learned, bias_inputs=bias_inputs,
                            bias_correction=bias)
    run, metadata, result = build_engine(graph, spec["weight_format"], spec["activation_format"], recipe, arrays,
                                         maxima, device, weight_rule=spec.get("weight_rule"), learned=learned,
                                         bias_inputs=None, bias_correction="none")
    if correction == "recipe_none":
        return run, metadata, result
    if spec["activation_format"] is None:
        plan, quantizers, activation_scales = (fp32_plan(graph), {"signed": TableQuantizer(spec["weight_format"],
                                                                                         "signed", device)}, {})
    else:
        plan, quantizers, activation_scales = activation_setup(graph, spec["activation_format"], recipe, arrays,
                                                                maxima, device)
    with torch.inference_mode():
        metadata["correction_report"] = staged_correct(graph, result, plan, quantizers, activation_scales, images,
                                                       policy=correction,
                                                       instrument=correction in ("none", "global"))
    if spec["activation_format"] is None:
        result.eval()
        return result, metadata, result
    return B2Interpreter(result, quantizers, plan, activation_scales).run, metadata, result


def layer_groups(graph):
    """Named layer groups of a classifier graph (conv/linear node names) for the mode-2 swap arms."""
    layers = weight_layers(graph)
    names = [node.name for node, _ in layers]

    def stage(name):
        parts = name.split("_")
        return int(parts[1]) if parts[0] == "features" and len(parts) > 1 and parts[1].isdigit() else None

    depthwise = [node.name for node, module in layers
                 if isinstance(module, nn.Conv2d) and module.groups > 1 and module.groups == module.in_channels]
    groups = {"classifier": [n for n in names if n.startswith("classifier") or n == "fc"],
              "depthwise": depthwise,
              "first": names[:1]}
    stages = [s for s in (stage(n) for n in names) if s is not None]
    if stages:
        top = max(stages)
        groups["last_conv"] = [n for n in names if stage(n) == top]
        groups["early"] = [n for n in names if stage(n) is not None and stage(n) <= top // 3]
        groups["mid"] = [n for n in names if stage(n) is not None and top // 3 < stage(n) <= 2 * top // 3]
        groups["late"] = [n for n in names if stage(n) is not None and stage(n) > 2 * top // 3]
    return groups
