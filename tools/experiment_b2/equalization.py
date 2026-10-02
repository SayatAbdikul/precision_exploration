"""Cross-layer equalization (Nagel et al. 2019) for exactly function-preserving pairs only.

A pair is ``conv_a -> [identity] -> ReLU -> [identity] -> conv_b`` where every
tensor in between has a single consumer.  ReLU is positively homogeneous, so
dividing output channel ``i`` of ``conv_a`` (weight and bias) by ``s_i`` and
multiplying input channel ``i`` of ``conv_b`` by ``s_i`` leaves the FP32
function unchanged.  ReLU6, Hardswish and Hardsigmoid are not homogeneous and
squeeze-excite branches break the single-consumer rule, so those pairs are
left alone (MobileNetV2 has no eligible pair at all).

The equalized graph is a different FP32 parameterisation, so its calibration
observations are collected again with the v1 observer on the same frozen
calibration images.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch import nn

from tools.experiment_b.common import digest, file_hash, seal, unseal
from .common import BASE

SWEEPS = 20
VERSION = "b2_cle_relu_pairs_v1"


def _single_through_identity(graph, node):
    """Follow single-consumer identity/dropout nodes; return the first other consumer or None."""
    while True:
        users = list(node.users)
        if len(users) != 1:
            return None
        user = users[0]
        if user.op == "call_module" and isinstance(graph.get_submodule(user.target), (nn.Identity, nn.Dropout)):
            node = user
            continue
        return user


def find_pairs(graph):
    pairs = []
    for node in graph.graph.nodes:
        if node.op != "call_module" or not isinstance(graph.get_submodule(node.target), nn.Conv2d):
            continue
        relu = _single_through_identity(graph, node)
        if relu is None or relu.op != "call_module" or type(graph.get_submodule(relu.target)) is not nn.ReLU:
            continue
        nxt = _single_through_identity(graph, relu)
        if nxt is None or nxt.op != "call_module" or not isinstance(graph.get_submodule(nxt.target), nn.Conv2d):
            continue
        a, b = graph.get_submodule(node.target), graph.get_submodule(nxt.target)
        depthwise = b.groups == b.in_channels == b.out_channels and b.weight.shape[1] == 1
        if b.in_channels != a.out_channels or not (b.groups == 1 or depthwise):
            continue
        pairs.append((node.name, node.target, nxt.name, nxt.target, depthwise))
    return pairs


def equalize_graph(graph):
    """Return ``(equalized deep copy, report)``."""
    result = copy.deepcopy(graph)
    pairs = find_pairs(result)
    total = {name: None for name, *_ in pairs}
    with torch.no_grad():
        for _ in range(SWEEPS):
            for name, target_a, _, target_b, depthwise in pairs:
                a, b = result.get_submodule(target_a), result.get_submodule(target_b)
                r1 = a.weight.abs().flatten(1).amax(dim=1).double()
                r2 = (b.weight.abs().flatten(1).amax(dim=1) if depthwise
                      else b.weight.abs().transpose(0, 1).flatten(1).amax(dim=1)).double()
                s = torch.where((r1 > 0) & (r2 > 0), torch.sqrt(r1 / r2), torch.ones_like(r1)).to(a.weight.dtype)
                a.weight.div_(s.reshape(-1, 1, 1, 1))
                if a.bias is not None:
                    a.bias.div_(s)
                b.weight.mul_(s.reshape(-1, 1, 1, 1) if depthwise else s.reshape(1, -1, 1, 1))
                total[name] = s.double() if total[name] is None else total[name] * s.double()
    report = {"version": VERSION, "sweeps": SWEEPS, "pairs": [
        {"first": name, "second": second, "second_depthwise": depthwise,
         "scale_min": float(total[name].min()), "scale_max": float(total[name].max())}
        for name, _, second, _, depthwise in pairs]}
    return result, report


def observe(graph, inputs, device, key):
    """v1 observer over the cached calibration inputs; cached per equalized graph."""
    from tools.experiment_b.classifier import ObservingInterpreter
    from tools.experiment_b.common import PROTOCOL
    folder = BASE / "calibration_cle" / digest(key)
    data, meta_path = folder / "observations.npz", folder / "meta.json"
    if meta_path.exists():
        meta = unseal(meta_path)
        if meta["key"] != key or file_hash(data) != meta["arrays_sha256"]:
            raise ValueError("equalized calibration cache integrity failure")
        with np.load(data, allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}, meta["maxima"], {"identity": folder.name}
    observer = ObservingInterpreter(graph)
    combined, maxima = {}, {}
    batch = PROTOCOL["calibration_batch_size"]
    for start in range(0, len(inputs), batch):
        observer.ordinal, observer.samples, observer.maxima, observer.channel_counts = start, {}, {}, {}
        with torch.inference_mode():
            observer.run(torch.from_numpy(np.array(inputs[start:start + batch], dtype=np.float32)).to(device))
        for name, values in observer.samples.items():
            combined.setdefault(name, []).append(values.ravel())
            maxima[name] = max(maxima.get(name, 0), observer.maxima[name])
    arrays = {name: np.concatenate(chunks) for name, chunks in combined.items()}
    folder.mkdir(parents=True, exist_ok=True)
    temporary = data.with_name(data.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(data)
    seal(meta_path, {"key": key, "arrays_sha256": file_hash(data), "maxima": maxima, "images": len(inputs)})
    return arrays, maxima, {"identity": folder.name}
