"""YOLOv8n explicit FP32 graph with exploratory QDQ through the DFL head."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from tools.experiment_b.common import ROOT, file_hash
from tools.experiment_b.classifier import sample_indices
from tools.experiment_b.quantizer import Quantizer, scale_for, threshold
from .shared import BlockQuantizer, block_conv2d, quantize_axis


def load_detector(device):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import ultralytics
    from public.inference.tensor import Encoding
    from public.quantization.graph.yolo import lower
    from public.workloads.models.deployment import fold_batchnorm
    manifest = json.loads((ROOT / "public/workloads/models/manifests/yolov8n.json").read_text())
    checkpoint = ROOT / manifest["checkpoint_path"]
    if file_hash(checkpoint) != manifest["checkpoint_sha256"] or ultralytics.__version__ != manifest["framework_version"]:
        raise ValueError("detector checkpoint or framework drift")
    original = ultralytics.YOLO(checkpoint).model.eval().to(device)
    constants = {}
    # This established lowering supplies the exact named operator/edge plan.
    # Its temporary fp6 constants are discarded; observed FP32 originals are
    # the only weights used by this independently identified B implementation.
    plan = lower(original, input_encoding=Encoding("fp6_e3m2"), weight_format="fp6_e3m2",
                 accumulator="fp32_e8m23_accumulator", provenance={"kind": "experiment_b_structure_only"},
                 constant_observer=lambda name, tensor: constants.__setitem__(name, tensor.detach().clone().to(device)))
    if set(constants) != set(plan["constants"]):
        raise ValueError("detector constant capture coverage mismatch")
    return plan, constants, fold_batchnorm(original).to(device)


def image_batch(rows, payload, device):
    import cv2
    from ultralytics.data.augment import LetterBox
    letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
    payload = Path(payload).resolve()
    images, shapes = [], []
    for row in rows:
        path = (payload / row["file_name"]).resolve()
        if not path.is_relative_to(payload) or file_hash(path) != row["sha256"]:
            raise ValueError("COCO image payload drift")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("COCO image decode failed")
        shapes.append(tuple(image.shape[:2]))
        rgb = letterbox(image=image)[:, :, ::-1].transpose(2, 0, 1)
        images.append(torch.from_numpy(np.ascontiguousarray(rgb)).float()/255)
    return torch.stack(images).to(device), shapes


def _run_op(node, values, constants, device):
    args = [values[name] if name in values else constants[name] for name in node["inputs"]]
    op, attrs = node["op"], node["attrs"]
    x = args[0]
    if op == "conv2d":
        bias = torch.tensor([float(v) for v in attrs["bias"]], device=device) if "bias" in attrs else None
        return F.conv2d(x, args[1], bias, stride=attrs.get("stride", 1), padding=attrs.get("padding", 0),
                        dilation=attrs.get("dilation", 1), groups=attrs.get("groups", 1))
    if op == "lut":
        return {"silu": F.silu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}[attrs["function"]](x)
    if op == "activation":
        return {"relu": F.relu, "relu6": F.relu6}[attrs["function"]](x)
    if op == "elementwise":
        return x + args[1] if attrs["operation"] == "add" else x * args[1]
    if op == "concatenate":
        return torch.cat(args, dim=attrs["axis"])
    if op == "channel_slice":
        return x[:, attrs["start"]:attrs["stop"]]
    if op == "resize_nearest":
        return F.interpolate(x, scale_factor=attrs["factor"], mode="nearest")
    if op == "pool2d" and attrs["kind"] == "max":
        return F.max_pool2d(x, attrs["kernel_size"], attrs.get("stride"), attrs.get("padding", 0))
    if op == "flatten":
        return x.flatten(attrs["start_dim"])
    if op == "dfl":
        batch, _, height, width = x.shape
        probabilities = x.reshape(batch, 4, attrs["bins"], height*width).transpose(1, 2).softmax(1)
        return F.conv2d(probabilities, args[1]).reshape(batch, 4, height, width)
    if op == "decode_boxes":
        height, width = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32, device=device)+.5,
                                torch.arange(width, dtype=torch.float32, device=device)+.5, indexing="ij")
        anchors = torch.stack((xx, yy))[None]
        lo, hi = anchors-x[:, :2], anchors+x[:, 2:]
        return torch.cat(((lo+hi)/2, hi-lo), 1)*attrs["stride"]
    raise ValueError(f"unsupported B detector operation: {op}")


class DetectorEngine:
    def __init__(self, graph, constants, device, *, quantizer=None, activation_scales=None, weight_scales=None):
        self.graph, self.device, self.quantizer = graph, device, quantizer
        self.activation_scales = activation_scales
        self.weight_scales = weight_scales
        self.constants = constants
        self.observer = None
        self.ordinal = 0
        if quantizer is not None and activation_scales is not None:
            expected = {"images"} | {n["name"] for n in graph["nodes"] if n["op"] != "flatten"}
            if set(activation_scales) != expected:
                raise ValueError("detector activation scale/node coverage mismatch")

    def _qdq(self, name, value):
        if isinstance(self.quantizer, BlockQuantizer):
            return quantize_axis(value, self.quantizer, 1)
        scale = self.activation_scales[name]
        if isinstance(scale, list):
            if value.shape[1] != 84 or len(scale) != 2:
                raise ValueError("invalid detector head group scale")
            return torch.cat((self.quantizer(value[:, :4], scale[0]),
                              self.quantizer(value[:, 4:], scale[1])), dim=1)
        return self.quantizer(value, scale)

    def run(self, inputs):
        if tuple(inputs.shape[1:]) != (3, 640, 640):
            raise ValueError("detector requires fixed 640-square input")
        values = {"images": self._qdq("images", inputs) if self.quantizer else inputs}
        if self.observer:
            self.observer.record("images", inputs, self.ordinal)
        for node in self.graph["nodes"]:
            if node["op"] == "conv2d" and isinstance(self.quantizer, BlockQuantizer):
                source = values[node["inputs"][0]]
                weight = self.constants[node["inputs"][1]]
                attrs = node["attrs"]
                module = SimpleNamespace(in_channels=source.shape[1], out_channels=weight.shape[0],
                                         kernel_size=weight.shape[-2:], stride=attrs.get("stride", [1, 1]),
                                         padding=attrs.get("padding", [0, 0]), dilation=attrs.get("dilation", [1, 1]),
                                         groups=attrs.get("groups", 1), padding_mode="zeros",
                                         bias=(torch.tensor([float(v) for v in attrs["bias"]], device=self.device)
                                               if "bias" in attrs else None))
                result = block_conv2d(source, module, self.quantizer, weight)
            else:
                result = _run_op(node, values, self.constants, self.device)
            if self.observer and node["op"] != "flatten":
                self.observer.record(node["name"], result, self.ordinal)
            if self.quantizer and node["op"] != "flatten":
                result = self._qdq(node["name"], result)
            values[node["name"]] = result
        self.ordinal += len(inputs)
        output = values[self.graph["outputs"][0]]
        if tuple(output.shape) != (len(inputs), 84, 8400) or not torch.isfinite(output).all():
            raise ValueError("invalid detector output")
        return output


class DetectorObserver:
    """Channel-covering node samples; isolate boxes and class scores at joins."""
    def __init__(self):
        self.samples, self.maxima, self.channels = {}, {}, {}

    def record(self, name, tensor, ordinal):
        if tensor.ndim < 2:
            raise ValueError("detector observation lacks channel dimension")
        groups = (("boxes", tensor[:, :4]), ("scores", tensor[:, 4:])) if tensor.shape[1] == 84 else ((None, tensor),)
        for suffix, values in groups:
            key = name if suffix is None else f"{name}:{suffix}"
            flat = values.reshape(len(values), values.shape[1], -1)
            sampled = []
            for i in range(len(values)):
                ch, pos = sample_indices(flat.shape[1], flat.shape[2], ordinal+i)
                sampled.append(flat[i, torch.as_tensor(ch, device=flat.device), torch.as_tensor(pos, device=flat.device)])
            sample = torch.cat(sampled).detach().cpu().numpy()
            if not np.isfinite(sample).all():
                raise ValueError("nonfinite detector calibration sample")
            self.samples.setdefault(key, []).append(sample)
            self.maxima[key] = max(self.maxima.get(key, 0), float(values.abs().amax()))
            self.channels[key] = flat.shape[1]


def prepare_detector(graph, constants, name, recipe, samples, maxima, device):
    if name in {"bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"}:
        quantizer = BlockQuantizer(name, recipe, device)
        mapped = {key: quantizer(weight.reshape(weight.shape[0], -1)).reshape_as(weight)
                  for key, weight in constants.items()}
        return DetectorEngine(graph, mapped, device, quantizer=quantizer), {
            "block_axis": "reduction_k_for_weights_and_patches; channel_axis_for_stored_activations",
            "block_size": 32, "scale": "intrinsic_e8m0_per_block", "selection": recipe}
    quantizer = Quantizer(name, device)
    scales = {}
    for key, values in samples.items():
        span = maxima[key] if recipe == "maxabs" else threshold(values, recipe)
        scales[key] = torch.tensor(scale_for(span, name), device=device)
    activation = {}
    for key in {"images"} | {n["name"] for n in graph["nodes"] if n["op"] != "flatten"}:
        if f"{key}:boxes" in scales:
            activation[key] = [scales[f"{key}:boxes"], scales[f"{key}:scores"]]
        else:
            activation[key] = scales[key]
    weights, weight_scales = {}, {}
    for key, weight in constants.items():
        flat = weight.reshape(weight.shape[0], -1)
        row_scales = [scale_for(threshold(row.detach().cpu().numpy(), recipe), name) for row in flat]
        scale = torch.tensor(row_scales, device=device).reshape((-1,) + (1,)*(weight.ndim-1))
        weights[key] = quantizer(weight, scale)
        weight_scales[key] = row_scales
    metadata = {"activation_scales": {k: [float(x) for x in v] if isinstance(v, list) else float(v)
                                       for k, v in activation.items()},
                "weight_scales": weight_scales, "scale_storage": "fp32"}
    return DetectorEngine(graph, weights, device, quantizer=quantizer, activation_scales=activation), metadata
