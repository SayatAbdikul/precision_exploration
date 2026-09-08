"""Deterministic FP32 BN folding and fixed-input deployment graph identity.

The capture records ATen operations, edges, scalar attributes and tensor content
hashes. It is a Phase 1 workload identity, not the Phase 2 low-precision engine.
"""
from __future__ import annotations

import copy
import hashlib
import json
import warnings
from functools import wraps
import math
from typing import Any

GRAPH_VERSION = "folded-aten-fixed-input-1.1.0"


def fold_batchnorm(model):
    import torch.nn as nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    folded = copy.deepcopy(model).eval()

    def visit(module):
        children = list(module.named_children())
        for _, child in children:
            visit(child)
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
    visit(folded)
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in folded.modules()):
        raise ValueError("unfolded BatchNorm remains in the deployment graph")
    return folded


def _tensor_identity(value):
    tensor = value.detach().cpu().contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}


def deterministic_capture(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
                return function(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return wrapped


@deterministic_capture
def capture_graph(model, example) -> list[dict[str, Any]]:
    import torch
    # YOLO caches grids on its first forward. Stabilize before capture.
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        model(example)
        traced = torch.jit.freeze(torch.jit.trace(model.eval(), example, strict=False, check_trace=False))
    graph = traced.inlined_graph
    identities = {}

    def edge(value):
        key = value.unique()
        if key not in identities:
            identities[key] = len(identities)
        return identities[key]

    def attribute(node, name):
        kind = node.kindOf(name)
        if kind == "t":
            return _tensor_identity(node.t(name))
        if kind == "ts":
            return [_tensor_identity(tensor) for tensor in node.ts(name)]
        if kind == "ival":
            value = node.ival(name)
            json.dumps(value, allow_nan=False)
            return value
        if kind not in {"i", "f", "s", "is", "fs", "ss"}:
            raise ValueError(f"unsupported graph attribute {node.kind()}.{name}: {kind}")
        getter = getattr(node, "is" if kind == "is" else kind)
        return getter(name)

    nodes = [{"op": "input", "outputs": [edge(v) for v in graph.inputs()],
              "shape": list(example.shape), "dtype": str(example.dtype)}]
    for node in graph.nodes():
        if list(node.blocks()):
            raise ValueError("fixed-input graph unexpectedly contains control-flow blocks")
        nodes.append({"op": node.kind(), "inputs": [edge(v) for v in node.inputs()],
                      "outputs": [edge(v) for v in node.outputs()],
                      "attributes": {name: attribute(node, name) for name in sorted(node.attributeNames())}})
    nodes.append({"op": "output", "inputs": [edge(v) for v in graph.outputs()]})
    return nodes


def prepare_deployment(model, input_shape: list[int]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    import torch
    model = model.eval()
    folded = fold_batchnorm(model)
    probes = [torch.zeros(input_shape), torch.linspace(-1, 1, steps=math.prod(input_shape)).reshape(input_shape)]
    maximum_error = 0.0

    def compare(left, right):
        nonlocal maximum_error
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-4)
            maximum_error = max(maximum_error, float((left - right).abs().max()))
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise ValueError("fold changed output structure")
            for key in left:
                compare(left[key], right[key])
        else:
            if len(left) != len(right):
                raise ValueError("fold changed output structure")
            for a, b in zip(left, right):
                compare(a, b)
    with torch.inference_mode():
        for probe in probes:
            compare(model(probe), folded(probe))
    nodes = capture_graph(folded, probes[0])
    if any("batch_norm" in node["op"] for node in nodes):
        raise ValueError("captured graph contains BatchNorm")
    # Repeat serialization in the same process: no JIT-generated names enter identity.
    if nodes != capture_graph(folded, probes[0]):
        raise ValueError("graph serialization did not reproduce")
    return folded, nodes, {"status": "passed", "probes": ["zeros", "linear_ramp_minus1_plus1"],
                           "rtol": 1e-4, "atol": 1e-4, "max_absolute_error": maximum_error,
                           "graph_repeat_identical": True, "remaining_batchnorm": 0}
