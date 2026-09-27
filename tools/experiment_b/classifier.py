"""Frozen classifier loading, channel-covering observation and scalar QDQ FX."""
from __future__ import annotations

import json
import operator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, file_hash
from .quantizer import Quantizer, scale_for, threshold


def configure(device):
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Experiment B requires its CUDA environment/device; no silent CPU fallback")


def load_model(name, device):
    from torchvision import models
    from public.workloads.models.deployment import fold_batchnorm
    from public.workloads.models.torchvision_eval import MODEL_SPECS
    manifest = json.loads((ROOT / f"public/workloads/models/manifests/{name}.json").read_text())
    checkpoint = ROOT / manifest["checkpoint_path"]
    if file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint drift")
    original = getattr(models, name)(weights=None)
    original.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    original.eval()
    graph = torch.fx.symbolic_trace(fold_batchnorm(original)).to(device)
    validate_graph(graph)
    _, weight_class, weight_name = MODEL_SPECS[name]
    transform = getattr(getattr(models, weight_class), weight_name).transforms()
    return graph, transform, original.to(device)


def validate_graph(graph):
    modules = (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid,
               nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d, nn.Flatten, nn.Identity, nn.Dropout)
    functions = (operator.add, operator.mul, torch.flatten, F.adaptive_avg_pool2d)
    for node in graph.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op == "call_module" and isinstance(graph.get_submodule(node.target), modules):
            continue
        if node.op == "call_function" and node.target in functions:
            continue
        raise ValueError(f"unsupported B FX operation: {node.op} {node.target}")


def quantized_node(graph, node):
    if node.op == "output":
        return False
    if node.op == "call_module" and isinstance(graph.get_submodule(node.target), (nn.Identity, nn.Dropout, nn.Flatten)):
        return False
    return not (node.op == "call_function" and node.target is torch.flatten)


def image_batch(rows, payload, transform, device):
    from PIL import Image
    payload = Path(payload).resolve()
    def load(row):
        path = (payload / row["relative_path"]).resolve()
        if not path.is_relative_to(payload) or file_hash(path) != row["sha256"]:
            raise ValueError("image path/content mismatch")
        with Image.open(path) as image:
            return transform(image.convert("RGB"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        tensors = list(pool.map(load, rows))
    return torch.stack(tensors).to(device)


def sample_indices(channels, spatial, ordinal):
    """Rotate channel coverage across images and spatial coverage within channels."""
    count = min(256, channels * spatial)
    k = np.arange(count, dtype=np.int64)
    ch = (k + ordinal * min(256, channels)) % channels
    location = ((k // channels) + ordinal * 997 + ch * 17) % spatial
    return ch, location


class ObservingInterpreter(torch.fx.Interpreter):
    def __init__(self, graph):
        super().__init__(graph)
        self.ordinal = 0
        self.samples = {}
        self.maxima = {}
        self.channel_counts = {}

    def run_node(self, node):
        result = super().run_node(node)
        if quantized_node(self.module, node):
            if not isinstance(result, torch.Tensor) or result.ndim < 2:
                raise ValueError("unexpected calibration tensor")
            shape = result.shape
            values = result.reshape(shape[0], shape[1], -1)
            selections = []
            for i in range(shape[0]):
                ch, pos = sample_indices(shape[1], values.shape[2], self.ordinal + i)
                selections.append(values[i, torch.as_tensor(ch, device=result.device), torch.as_tensor(pos, device=result.device)])
            packed = torch.cat([torch.stack(selections).flatten(), values.abs().amax().reshape(1)]).cpu().numpy()
            if not np.isfinite(packed).all():
                raise ValueError(f"nonfinite calibration node: {node.name}")
            self.samples[node.name] = packed[:-1].reshape(shape[0], -1).copy()
            self.maxima[node.name] = float(packed[-1])
            self.channel_counts[node.name] = shape[1]
        return result


class QDQInterpreter(torch.fx.Interpreter):
    def __init__(self, graph, quantizer, activation_scales):
        super().__init__(graph)
        self.quantizer = quantizer
        self.scales = {k: torch.tensor(v, dtype=torch.float32, device=quantizer.levels.device)
                       for k, v in activation_scales.items()}
        expected = {n.name for n in graph.graph.nodes if quantized_node(graph, n)}
        if expected != set(self.scales):
            raise ValueError("activation scale/node coverage mismatch")

    def run_node(self, node):
        result = super().run_node(node)
        if quantized_node(self.module, node):
            result = self.quantizer(result, self.scales[node.name])
        return result


def prepare_qdq(graph, name, recipe, arrays, maxima, device):
    import copy
    quantizer = Quantizer(name, device)
    activation_scales = {node: scale_for(maxima[node] if recipe == "maxabs" else threshold(values, recipe), name)
                         for node, values in arrays.items()}
    result = copy.deepcopy(graph)
    weight_scales = {}
    with torch.no_grad():
        for node in result.graph.nodes:
            if node.op != "call_module":
                continue
            module = result.get_submodule(node.target)
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            source = module.weight.detach().cpu().numpy()
            scales = [scale_for(threshold(channel, recipe), name) for channel in source]
            shaped = torch.tensor(scales, dtype=torch.float32, device=device).reshape((-1,) + (1,) * (source.ndim - 1))
            module.weight.copy_(quantizer(module.weight, shaped))
            weight_scales[node.name] = scales
    return QDQInterpreter(result, quantizer, activation_scales), {
        "activation_scales": activation_scales, "weight_scales": weight_scales,
        "scale_storage": "fp32; include_scale_metadata_and_application_cost_in_hardware_analysis"}
