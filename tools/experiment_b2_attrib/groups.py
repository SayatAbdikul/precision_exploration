"""Named groups of quantizing activation boundaries (structural, no data).

Two partitions per model, each covering every quantizing boundary of the B2 default plan exactly once:

* ``role``: what the tensor is (stem, expansion, depthwise, Hardswish output, squeeze-excite parts,
  projection, residual add, classifier ...);
* ``position``: where it is (stem, early / middle / late stages, classifier).

Groups are derived from the FX graph and the B2 boundary plan only; the rules are written per model so that
they can be read and audited, and ``partition`` refuses a node that no rule claims or that two rules claim.
"""
from __future__ import annotations

import re

from torch import nn

from tools.experiment_b2.boundaries import _effective_users, _producer

ROLE_ORDER = {
    "mobilenet_v3_large": ("stem", "expand_conv", "relu_act", "hswish", "dw_conv", "se_pool", "se_reduce",
                           "se_expand", "se_gate", "se_product", "project", "residual_add", "classifier"),
    "mobilenet_v2": ("stem", "expand_act", "dw_act", "project", "residual_add", "classifier"),
    "resnet18": ("stem", "inner_act", "pre_add_conv", "downsample", "block_out", "classifier"),
}
POSITION_ORDER = {
    "mobilenet_v3_large": ("stem", "early", "middle", "late", "classifier"),
    "mobilenet_v2": ("stem", "early", "middle", "late", "classifier"),
    "resnet18": ("stem", "layer1", "layer2", "layer3", "layer4", "classifier"),
}
# Stage boundaries (block index of torchvision ``features``) of the position partition.
STAGES = {"mobilenet_v3_large": ((1, 6, "early"), (7, 12, "middle"), (13, 16, "late")),
          "mobilenet_v2": ((1, 6, "early"), (7, 13, "middle"), (14, 18, "late"))}


def _block_of(names_in_order, name):
    """``features`` block index of a node; function nodes (add, mul) take the last module node before them."""
    match = re.match(r"features_(\d+)_", name)
    if match:
        return int(match.group(1))
    index = names_in_order.index(name)
    for previous in reversed(names_in_order[:index]):
        match = re.match(r"features_(\d+)_", previous)
        if match:
            return int(match.group(1))
    return None


def _module(graph, node):
    return graph.get_submodule(node.target) if node.op == "call_module" else None


def _role_mobilenet_v3(graph, node, plan, kinds, nodes):
    name, label = node.name, plan[node.name]["kind"]
    if label == "input" or name.startswith("features_0_"):
        return "stem"
    if name == "avgpool" or name.startswith("classifier_"):
        return "classifier"
    if label == "add":
        return "residual_add"
    if label == "mul":
        return "se_product"
    if name.endswith("_fc2"):
        return "se_expand"
    if name.endswith("_scale_activation"):
        return "se_gate"
    if re.search(r"_block_2_avgpool$", name):
        return "se_pool"
    if re.search(r"_block_2_activation$", name):
        return "se_reduce"
    if label == "hardswish":
        return "hswish"
    if label in ("relu", "relu6"):
        return "relu_act"
    if label == "conv":
        module = _module(graph, node)
        if module.groups > 1:
            return "dw_conv"
        # Effective users look through the identity that replaced the folded BatchNorm.
        users = _effective_users(graph, node, kinds)
        consumer = kinds[users[0].name] if len(users) == 1 else None
        return "expand_conv" if consumer == "hardswish" else "project"
    return None


def _role_mobilenet_v2(graph, node, plan, kinds, nodes):
    name, label = node.name, plan[node.name]["kind"]
    if label == "input" or name.startswith("features_0_"):
        return "stem"
    if name == "adaptive_avg_pool2d" or name.startswith("classifier_"):
        return "classifier"
    if label == "add":
        return "residual_add"
    if label == "conv":
        return "project"
    if label in ("relu", "relu6"):
        source = node.all_input_nodes[0]
        source = _producer(source, kinds)
        producer = nodes[source.name]
        module = _module(graph, producer) if producer.op == "call_module" else None
        # relu6 <- identity (folded BN) <- conv: look through the identity.
        while module is not None and not isinstance(module, nn.Conv2d):
            producer = producer.all_input_nodes[0]
            module = _module(graph, producer)
        if module is None:
            return None
        return "dw_act" if module.groups > 1 else "expand_act"
    return None


def _role_resnet18(graph, node, plan, kinds, nodes):
    name, label = node.name, plan[node.name]["kind"]
    if label == "input" or name == "relu":
        return "stem"
    if name in ("avgpool", "fc"):
        return "classifier"
    if label == "conv":
        return "downsample" if "downsample" in name else "pre_add_conv"
    if re.fullmatch(r"layer\d_\d_relu", name):
        return "inner_act"
    if re.fullmatch(r"layer\d_\d_relu_1", name):
        return "block_out"
    return None


ROLE_RULES = {"mobilenet_v3_large": _role_mobilenet_v3, "mobilenet_v2": _role_mobilenet_v2,
              "resnet18": _role_resnet18}


def _position(model, name, label, order):
    if model == "resnet18":
        if label == "input" or name == "relu":
            return "stem"
        if name in ("avgpool", "fc"):
            return "classifier"
        match = re.match(r"layer(\d)_", name)
        if match:
            return f"layer{match.group(1)}"
        # adds fused into ReLU are not quantizing nodes in ResNet18; anything else is unexpected.
        return None
    if label == "input" or name.startswith("features_0_"):
        return "stem"
    if name in ("avgpool", "adaptive_avg_pool2d") or name.startswith("classifier_"):
        return "classifier"
    block = _block_of(order, name)
    for low, high, label_ in STAGES[model]:
        if block is not None and low <= block <= high:
            return label_
    return None


def partition(model, graph, plan):
    """``{"role": {group: [nodes]}, "position": {group: [nodes]}}`` over the quantizing nodes of ``plan``."""
    nodes = {n.name: n for n in graph.graph.nodes}
    order = [n.name for n in graph.graph.nodes]
    kinds = {name: row["kind"] for name, row in plan.items()}
    kinds.update({n.name: "output" for n in graph.graph.nodes if n.op == "output"})
    quantizing = [name for name in order if name in plan and plan[name]["quantizes"]]
    result = {}
    for scheme, order_names in (("role", ROLE_ORDER[model]), ("position", POSITION_ORDER[model])):
        groups = {g: [] for g in order_names}
        for name in quantizing:
            if scheme == "role":
                group = ROLE_RULES[model](graph, nodes[name], plan, kinds, nodes)
            else:
                group = _position(model, name, plan[name]["kind"], order)
            if group not in groups:
                raise ValueError(f"{model}: no {scheme} group for boundary {name} (got {group})")
            groups[group].append(name)
        empty = [g for g, members in groups.items() if not members]
        if empty:
            raise ValueError(f"{model}: empty {scheme} groups {empty}")
        covered = [n for members in groups.values() for n in members]
        if sorted(covered) != sorted(quantizing) or len(set(covered)) != len(covered):
            raise ValueError(f"{model}: {scheme} groups are not a partition of the quantizing boundaries")
        result[scheme] = groups
    return result


def consumers(graph, plan, name):
    """Effective consumers (looking through identity/flatten) of a boundary: ``[(node name, kind)]``."""
    kinds = {k: row["kind"] for k, row in plan.items()}
    kinds.update({n.name: "output" for n in graph.graph.nodes if n.op == "output"})
    node = {n.name: n for n in graph.graph.nodes}[name]
    return [(user.name, kinds[user.name]) for user in _effective_users(graph, node, kinds)]
