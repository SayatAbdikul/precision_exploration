"""FP32 surrogate of intrinsic E8M0/K32 QDQ and block convolution.

This keeps the accepted format's intrinsic scale ownership. It does not claim
the exact engine's sequential rounded accumulator semantics.
"""
from __future__ import annotations

from functools import lru_cache
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.experiment_b.common import ROOT
from tools.experiment_b.classifier import quantized_node


@lru_cache(maxsize=4)
def codebook(name):
    from public.formats.oracle.number_format import NumberFormat
    manifest = json.loads((ROOT / f"public/formats/manifests/accepted/{name}.json").read_text())
    if manifest["scaling"]["mode"] != "intrinsic_shared" or manifest["block"] != {
            "shared_scale_format": "e8m0", "block_size": 32, "block_axis": "reduction_k",
            "incomplete_block_policy": "scale_valid_values_zero_pad"}:
        raise ValueError("unsupported shared format policy")
    fmt = NumberFormat(manifest)
    by_value = {}
    for code in range(1 << fmt.bits):
        value = fmt.decode(code)
        if value.is_finite():
            key = np.float32(float(value))
            by_value[key] = min(code, by_value.get(key, code))
    levels = np.asarray(sorted(by_value), dtype=np.float32)
    codes = np.asarray([by_value[v] for v in levels])
    boundaries = ((levels[:-1].astype(np.float64) + levels[1:].astype(np.float64)) / 2).astype(np.float32)
    ties = np.asarray([(int(b) & 1, int(b)) < (int(a) & 1, int(a)) for a, b in zip(codes[:-1], codes[1:])])
    if manifest["rounding"] != "rne" or not np.all(np.diff(levels) > 0):
        raise ValueError("invalid shared element codebook")
    return levels, boundaries, ties


def numpy_block_qdq(values, name, recipe):
    """Independent NumPy reference on arrays whose final dimension is K."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] < 1 or not np.isfinite(source).all():
        raise ValueError("invalid shared-block input")
    levels, boundaries, ties = codebook(name)
    result = np.empty_like(source)
    for start in range(0, source.shape[-1], 32):
        block = source[..., start:start+32]
        span = np.max(np.abs(block), axis=-1, keepdims=True) if recipe == "maxabs" else np.quantile(
            np.abs(block), .999, axis=-1, keepdims=True, method="linear").astype(np.float32)
        if recipe not in {"maxabs", "percentile_99_9"}:
            raise ValueError("unknown block scale recipe")
        ratio = span / levels[-1]
        exponent = np.ceil(np.log2(np.maximum(ratio, np.float32(2)**-127))).clip(-127, 127).astype(np.int32)
        scale = np.exp2(exponent).astype(np.float32)
        y = block / scale
        positions = np.searchsorted(boundaries, y, side="left")
        adjacent = np.minimum(positions, len(boundaries)-1)
        positions += (y == boundaries[adjacent]) & ties[adjacent]
        reconstructed = levels[positions]
        reconstructed = np.where((reconstructed == 0) & np.signbit(y), np.float32(-0.0), reconstructed)
        result[..., start:start+32] = (reconstructed * scale).astype(np.float32)
    return result


class BlockQuantizer:
    def __init__(self, name, recipe, device):
        if recipe not in {"maxabs", "percentile_99_9"}:
            raise ValueError("unknown block scale recipe")
        self.name, self.recipe = name, recipe
        levels, boundaries, ties = codebook(name)
        self.levels = torch.tensor(levels, device=device)
        self.boundaries = torch.tensor(boundaries, device=device)
        self.ties = torch.tensor(ties, device=device)

    def __call__(self, values):
        if not torch.isfinite(values).all() or values.ndim < 1 or values.shape[-1] < 1:
            raise ValueError("invalid shared-block input")
        outputs = []
        for block in values.split(32, dim=-1):
            magnitudes = block.abs()
            if self.recipe == "maxabs" or block.shape[-1] == 1:
                span = magnitudes.amax(dim=-1, keepdim=True)
            else:
                highest = magnitudes.topk(2, dim=-1).values
                weight = .999 * (block.shape[-1] - 1) - (block.shape[-1] - 2)
                span = highest[..., 1:2] * (1-weight) + highest[..., :1] * weight
            target = span / self.levels[-1]
            mantissa, exponent = torch.frexp(target)
            exponent = torch.where(target == 0, torch.full_like(exponent, -127),
                                   torch.where(mantissa == .5, exponent - 1, exponent)).clamp(-127, 127)
            scale = torch.ldexp(torch.ones_like(target), exponent)
            y = block / scale
            positions = torch.bucketize(y.contiguous(), self.boundaries, right=False)
            adjacent = positions.clamp_max(len(self.boundaries)-1)
            positions += ((y == self.boundaries[adjacent]) & self.ties[adjacent]).long()
            reconstructed = self.levels[positions]
            reconstructed = torch.where(reconstructed == 0, torch.copysign(reconstructed, y), reconstructed)
            outputs.append(reconstructed * scale)
        return torch.cat(outputs, dim=-1)


def quantize_axis(values, quantizer, axis=1):
    return quantizer(values.movedim(axis, -1)).movedim(-1, axis)


def block_conv2d(inputs, module, quantizer, weights):
    if module.padding_mode != "zeros" or isinstance(module.padding, str):
        raise ValueError("unsupported block convolution padding")
    n, channels, height, width = inputs.shape
    out_channels = module.out_channels
    groups = module.groups
    kh, kw = module.kernel_size
    k = channels // groups * kh * kw
    patches = F.unfold(inputs, (kh, kw), dilation=module.dilation, padding=module.padding, stride=module.stride)
    locations = patches.shape[-1]
    patches = patches.reshape(n, groups, k, locations).permute(0, 1, 3, 2).contiguous()
    patches = quantizer(patches).permute(0, 1, 3, 2)
    weight_groups = weights.reshape(groups, out_channels//groups, k)
    output = torch.matmul(weight_groups.unsqueeze(0), patches).reshape(n, out_channels, locations)
    if module.bias is not None:
        output = output + module.bias.reshape(1, -1, 1)
    oh = (height + 2*module.padding[0] - module.dilation[0]*(kh-1) - 1)//module.stride[0] + 1
    ow = (width + 2*module.padding[1] - module.dilation[1]*(kw-1) - 1)//module.stride[1] + 1
    if oh*ow != locations:
        raise ValueError("block convolution shape mismatch")
    return output.reshape(n, out_channels, oh, ow)


class SharedInterpreter(torch.fx.Interpreter):
    def __init__(self, graph, quantizer, weights):
        super().__init__(graph)
        self.quantizer = quantizer
        self.weights = weights

    def run_node(self, node):
        if node.op == "call_module" and isinstance(self.module.get_submodule(node.target), (nn.Conv2d, nn.Linear)):
            args, _ = self.fetch_args_kwargs_from_env(node)
            module = self.module.get_submodule(node.target)
            result = (block_conv2d(args[0], module, self.quantizer, self.weights[node.name])
                      if isinstance(module, nn.Conv2d) else F.linear(args[0], self.weights[node.name], module.bias))
        else:
            result = super().run_node(node)
        if quantized_node(self.module, node):
            result = quantize_axis(result, self.quantizer, 1)
        return result


def prepare_shared(graph, name, recipe, device):
    import copy
    quantizer = BlockQuantizer(name, recipe, device)
    result = copy.deepcopy(graph)
    weights = {}
    for node in result.graph.nodes:
        if node.op != "call_module":
            continue
        module = result.get_submodule(node.target)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            flat = module.weight.detach().reshape(module.out_features if isinstance(module, nn.Linear) else module.out_channels, -1)
            weights[node.name] = quantizer(flat).reshape_as(module.weight)
    return SharedInterpreter(result, quantizer, weights), {
        "block_axis": "reduction_k_for_weights_and_convolution_patches; channel_axis_for_stored_activations",
        "block_size": 32, "scale": "intrinsic_e8m0_per_block", "selection": recipe,
        "accumulator": "fp32_framework_reduction"}
