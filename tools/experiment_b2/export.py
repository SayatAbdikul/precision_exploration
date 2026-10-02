"""Recipe-as-data export (format ``b2-recipe-export-1``) for the exact engine (lane L2).

For every tensor: whether its boundary quantizes, the codebook, signedness and
FP32 scale.  For every conv/linear: weight codes, per-channel scales and the
(possibly corrected) bias constants.
"""
from __future__ import annotations

import os
import struct

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, seal
from .boundaries import kind
from .codebook import codebook_id, describe, level_codes
from .common import BASE
from .recipe import hardware_semantics

VERSION = "b2-recipe-export-1"


def fp32_hex(value):
    return struct.pack(">f", np.float32(value)).hex()


def _pair(value):
    return [int(value)] * 2 if isinstance(value, int) else [int(x) for x in value]


def node_rows(graph, metadata):
    import torch
    from torch import nn
    rows = []
    for node in graph.graph.nodes:
        label = kind(graph, node)
        row = {"name": node.name, "fx_op": node.op, "target": str(node.target), "op": label,
               "inputs": [x.name for x in node.all_input_nodes], "args_repr": repr(node.args),
               "kwargs_repr": repr(node.kwargs), "attrs": {}}
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            if isinstance(module, nn.Conv2d):
                row["attrs"] = {k: _pair(getattr(module, k)) for k in ("stride", "padding", "dilation")}
                row["attrs"]["groups"] = module.groups
            elif isinstance(module, nn.MaxPool2d):
                row["attrs"] = {k: _pair(getattr(module, k)) for k in ("kernel_size", "stride", "padding", "dilation")}
                row["attrs"]["ceil_mode"] = bool(module.ceil_mode)
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid)):
                row["attrs"] = {"inplace": bool(module.inplace)}
        if label != "output":
            boundary = dict(metadata["boundaries"][node.name])
            boundary.pop("kind")
            if boundary["quantizes"]:
                scale = metadata["activation_scales"][node.name]
                boundary.update(scale=scale, scale_fp32_hex=fp32_hex(scale),
                                range_info=metadata.get("activation_search", {}).get(node.name))
            row["boundary"] = boundary
        rows.append(row)
    return rows


def weight_arrays(graph, engine, name, metadata):
    """Weight codes and constants; proves ``level[code] * scale`` is the weight B2 runs with."""
    import torch
    from torch import nn
    quantizer = engine.quantizers["signed"]
    codes_of_level = level_codes(name, "signed")
    levels = quantizer.levels.cpu().numpy()
    arrays, weights = {}, {}
    for node in graph.graph.nodes:
        if node.op != "call_module":
            continue
        original = graph.get_submodule(node.target)
        if not isinstance(original, (nn.Conv2d, nn.Linear)):
            continue
        quantized = engine.module.get_submodule(node.target)
        scales = np.array(metadata["weight_scales"][node.name], dtype=np.float32)
        shaped = torch.tensor(scales, device=original.weight.device).reshape((-1,) + (1,) * (original.weight.ndim - 1))
        index = quantizer.indices(original.weight.detach(), shaped).cpu().numpy()
        reconstructed = quantized.weight.detach().cpu().numpy()
        rebuilt = levels[index] * scales.reshape((-1,) + (1,) * (reconstructed.ndim - 1))
        if not np.array_equal(rebuilt.astype(np.float32), reconstructed):
            raise ValueError(f"exported weight codes do not reproduce the B2 weight of {node.name}")
        key = node.name
        arrays[f"{key}.original_weight"] = original.weight.detach().cpu().numpy()
        arrays[f"{key}.weight_codes"] = codes_of_level[index].astype(np.int32)
        arrays[f"{key}.weight_scales"] = scales
        arrays[f"{key}.weight_reconstructed"] = reconstructed
        arrays[f"{key}.bias_original"] = original.bias.detach().cpu().numpy()
        arrays[f"{key}.bias"] = quantized.bias.detach().cpu().numpy()
        weights[key] = {"codebook": codebook_id(name, "signed"), "channels": int(scales.size),
                        "shape": list(reconstructed.shape),
                        "bias_changed": not np.array_equal(arrays[f"{key}.bias"], arrays[f"{key}.bias_original"])}
    return arrays, weights


def array_hash(array):
    import hashlib
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def write_export(model, name, recipe_name, recipe, graph, engine, metadata, configuration, identity):
    arrays, weights = weight_arrays(graph, engine, name, metadata)
    signedness = sorted({row["signedness"] for row in metadata["boundaries"].values() if row["signedness"]} | {"signed"})
    logical = {"version": VERSION, "model": model, "format": name, "recipe_name": recipe_name,
               "recipe": recipe.as_dict(), "hardware_semantics": hardware_semantics(recipe),
               "configuration_sha256": identity, "model_context": configuration["model_context"],
               "source_sha256": configuration["source_sha256"], "runtime": configuration["runtime"],
               "codebooks": {codebook_id(name, s): describe(name, s) for s in signedness},
               "real_value": "level * scale in FP32 (scale is an FP32 constant; signed zero preserved by B2 only)",
               "nodes": node_rows(graph, metadata), "weights": weights,
               "weight_range_rule": recipe.weight_range, "activation_range_rule": recipe.activation_range,
               "folded_FX_topology": str(graph.graph),
               "array_identities": {k: array_hash(v) for k, v in arrays.items()}}
    folder = BASE / "exports" / digest(logical)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "constants.npz"
    if not path.exists():
        temporary = path.with_name(path.name + f".{os.getpid()}.partial")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    seal(folder / "export.json", {**logical, "constants": {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}})
    return folder


def load_export(folder):
    """Reader for lane L2: returns ``(payload, arrays)`` after integrity checks."""
    from tools.experiment_b.common import unseal
    payload = unseal(folder / "export.json")
    path = ROOT / payload["constants"]["path"]
    if payload["version"] != VERSION or file_hash(path) != payload["constants"]["sha256"]:
        raise ValueError("B2 export integrity failure")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if {k: array_hash(v) for k, v in arrays.items()} != payload["array_identities"]:
        raise ValueError("B2 export array identity failure")
    return payload, arrays


def run_export(args):
    import json
    from .runner import build
    setup, recipe, engine, scales, configuration, identity = build(args)
    folder = write_export(args.model, args.format, args.recipe, recipe, setup["graph"], engine, scales,
                          configuration, identity)
    print(json.dumps({"export": str(folder.relative_to(ROOT)), "configuration_sha256": identity}), flush=True)
    return 0
