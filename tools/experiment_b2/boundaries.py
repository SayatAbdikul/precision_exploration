"""Which FX tensors quantize, with which signedness, under a B2 recipe.

The analysis is structural (no data): it starts from the v1 rule (every node
output except identity/dropout/flatten quantizes) and removes boundaries
according to the recipe switches.
"""
from __future__ import annotations

import operator

import torch
from torch import nn
from torch.nn import functional as F

from tools.experiment_b.classifier import quantized_node

RELU_LIKE = (nn.ReLU, nn.ReLU6)
ALL_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid)


def kind(graph, node):
    if node.op == "placeholder":
        return "input"
    if node.op == "output":
        return "output"
    if node.op == "call_function":
        if node.target is operator.add:
            return "add"
        if node.target is operator.mul:
            return "mul"
        if node.target is torch.flatten:
            return "flatten"
        if node.target is F.adaptive_avg_pool2d:
            return "avgpool"
        raise ValueError(f"unsupported B2 function: {node.target}")
    if node.op != "call_module":
        raise ValueError(f"unsupported B2 FX op: {node.op}")
    module = graph.get_submodule(node.target)
    table = ((nn.Conv2d, "conv"), (nn.Linear, "linear"), (nn.ReLU6, "relu6"), (nn.ReLU, "relu"),
             (nn.Hardswish, "hardswish"), (nn.Hardsigmoid, "hardsigmoid"), (nn.MaxPool2d, "maxpool"),
             (nn.AdaptiveAvgPool2d, "avgpool"), (nn.AvgPool2d, "avgpool"), (nn.Flatten, "flatten"),
             (nn.Identity, "identity"), (nn.Dropout, "identity"))
    for cls, label in table:
        if isinstance(module, cls):
            return label
    raise ValueError(f"unsupported B2 module: {type(module).__name__}")


PASSTHROUGH = ("identity", "flatten")
FUSABLE_PRODUCERS = ("conv", "linear", "add", "mul")


def _effective_users(graph, node, kinds):
    """Consumers of a tensor, looking through identity/dropout/flatten."""
    result = []
    for user in node.users:
        if kinds[user.name] in PASSTHROUGH:
            result.extend(_effective_users(graph, user, kinds))
        else:
            result.append(user)
    return result


def _producer(node, kinds):
    """The node whose boundary a passthrough chain forwards."""
    while kinds[node.name] in PASSTHROUGH:
        node = node.all_input_nodes[0]
    return node


def analyze(graph, recipe, format_name):
    """Return ``{node name: boundary dict}`` for every FX node except the output."""
    from .codebook import supports_unsigned
    nodes = list(graph.graph.nodes)
    kinds = {node.name: kind(graph, node) for node in nodes}
    fuse = {"v1": (), "fused_relu": ("relu", "relu6"),
            "fused_all": ("relu", "relu6", "hardswish", "hardsigmoid")}[recipe.boundaries]
    unsigned_ok = recipe.unsigned and supports_unsigned(format_name)
    plan = {}
    for node in nodes:
        label = kinds[node.name]
        if label == "output":
            continue
        inputs = [plan[x.name] for x in node.all_input_nodes]
        if label in ("relu", "relu6", "hardsigmoid"):
            nonnegative = True
        elif label in ("maxpool", "avgpool", "identity", "flatten", "add", "mul"):
            nonnegative = bool(inputs) and all(x["nonnegative"] for x in inputs)
        else:
            nonnegative = False
        quantizes, reason = quantized_node(graph, node), None
        if not quantizes:
            reason = "passthrough"
        elif label == "input" and not recipe.quantize_input:
            quantizes, reason = False, "input_exempt"
        elif not recipe.quantize_logits and any(kinds[u.name] == "output" for u in _effective_users(graph, node, kinds)):
            quantizes, reason = False, "logits_exempt"
        elif fuse and label in FUSABLE_PRODUCERS:
            users = _effective_users(graph, node, kinds)
            if len(users) == 1 and kinds[users[0].name] in fuse:
                quantizes, reason = False, f"fused_into:{users[0].name}"
        elif fuse and label == "maxpool":
            source = _producer(node.all_input_nodes[0], kinds)
            if plan[source.name]["quantizes"]:
                quantizes, reason = False, "maxpool_passthrough"
        plan[node.name] = {"kind": label, "quantizes": quantizes, "reason": reason, "nonnegative": nonnegative,
                           "signedness": ("unsigned" if unsigned_ok and nonnegative else "signed") if quantizes else None}
    return plan


def quantizing_nodes(plan):
    return [name for name, row in plan.items() if row["quantizes"]]
