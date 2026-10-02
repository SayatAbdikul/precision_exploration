"""Sequential, resumable AdaRound fit of one model on the frozen calibration list.

One fit = one (model, weight format, weight scale rule, reconstruction input,
seed).  Layers are processed in FX topological order; every finished layer is
one small ``npz`` (packed up/down bits, scales, statistics), so a job can stop
after any layer and a later job continues from the files.  No activation is
written to disk: the inputs of the layer being learned are recomputed from
the calibration images and held in memory for that layer only.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import time

import numpy as np
import torch
from torch import nn

from tools.experiment_b.common import ROOT, digest, file_hash, formats, frozen_inputs, seal, unseal
from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2.common import source_identity
from tools.experiment_b2.engine import B2Interpreter

from . import adaround
from .engine import SCALE_RULES, activation_setup, consumer_activation, weight_layers, weight_scale

BASE = ROOT / "artifacts/experiment_b2_recon"
PROTOCOL = "experiment-b2-recon-1.0.0"
CALIBRATION_LIST, CALIBRATION_IMAGES = "imagenet_calibration_2k", 1024
NUMERIC_SOURCES = ("adaround.py", "engine.py", "fit.py")
GIB = float(1 << 30)


def recon_sources():
    folder = ROOT / "tools/experiment_b2_recon"
    return {f"tools/experiment_b2_recon/{name}": file_hash(folder / name) for name in NUMERIC_SOURCES}


class _Stop(Exception):
    pass


class _CaptureMixin:
    stop, captured = None, None

    def run_node(self, node):
        if node.name == self.stop:
            args, _ = self.fetch_args_kwargs_from_env(node)
            self.captured = args[0]
            raise _Stop()
        return super().run_node(node)


class _CaptureFP(_CaptureMixin, torch.fx.Interpreter):
    pass


class _CaptureB2(_CaptureMixin, B2Interpreter):
    pass


def layer_input(interpreter, name, batch):
    """Input tensor of node ``name`` when ``interpreter`` runs ``batch`` (execution stops at that node)."""
    interpreter.stop, interpreter.captured = name, None
    try:
        interpreter.run(batch)
    except _Stop:
        return interpreter.captured
    raise ValueError(f"node {name} was not reached")


def layer_forward(module):
    """``forward(weight, x)`` of a conv/linear module with its own bias and geometry."""
    if isinstance(module, nn.Conv2d):
        if module.padding_mode != "zeros":
            raise ValueError("only zero padding is supported")
        return lambda weight, x: nn.functional.conv2d(x, weight, module.bias, module.stride, module.padding,
                                                      module.dilation, module.groups)
    return lambda weight, x: nn.functional.linear(x, weight, module.bias)


def pack(up):
    return np.packbits(up.detach().cpu().numpy().astype(np.uint8).ravel())


def unpack(bits, shape):
    count = int(np.prod(shape))
    return torch.from_numpy(np.unpackbits(bits)[:count].astype(bool).reshape(shape))


def tensor_sha(tensor):
    return hashlib.sha256(np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4").tobytes()).hexdigest()


def write_layer(path, up, scales, statistics):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, up_bits=pack(up), shape=np.array(up.shape, dtype=np.int64),
                        scales=np.asarray(scales, dtype=np.float32),
                        statistics=np.frombuffer(json.dumps(statistics, sort_keys=True).encode(), dtype=np.uint8))
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"layer record already exists: {path}")
    temporary.replace(path)


def read_layer(path):
    with np.load(path, allow_pickle=False) as saved:
        shape = tuple(int(x) for x in saved["shape"])
        return (unpack(saved["up_bits"], shape), saved["scales"].copy(),
                json.loads(bytes(saved["statistics"]).decode()))


def fit_spec(model, wformat, rule, seed, *, reconstruction_input="fp32_activations", aformat=None, recipe=None,
             settings=None, images=CALIBRATION_IMAGES, env=None, rows=None):
    """Everything that decides the learned rounding; its digest is the fit identity."""
    entries = {row["name"]: row for row in formats()}
    if rule not in SCALE_RULES or wformat not in entries:
        raise ValueError("invalid fit request")
    spec = {"protocol": PROTOCOL, "model_context": frozen_inputs(model), "weight_format": wformat,
            "weight_format_sha256": entries[wformat]["sha256"], "weight_scale_rule": rule, "seed": int(seed),
            "settings": dict(settings or adaround.SETTINGS),
            "calibration": {"list": CALIBRATION_LIST, "images": images, "rows_sha256": digest(rows)},
            "reconstruction_input": reconstruction_input, "b2_source_sha256": source_identity(),
            "recon_sources": recon_sources(), "runtime": env}
    if reconstruction_input == "b2_activations":
        spec["activation_format"] = aformat
        spec["activation_recipe"] = recipe.as_dict()
    elif reconstruction_input != "fp32_activations":
        raise ValueError("unknown reconstruction input")
    return spec


def fit_folder(spec):
    identity = digest(spec)
    mode = "fp32in" if spec["reconstruction_input"] == "fp32_activations" else f"b2in-{spec['activation_format']}"
    name = (f"{spec['model_context']['model']}--{spec['weight_format']}--{spec['weight_scale_rule']}--{mode}"
            f"--s{spec['seed']}--{identity[:12]}")
    return BASE / "fits" / name, identity


def find_fit(model, wformat, rule, seed, mode="fp32in", complete=True):
    """Folder of the finished fit with these properties (exactly one is expected)."""
    pattern = f"{model}--{wformat}--{rule}--{mode}--s{seed}--*"
    matches = [p for p in sorted((BASE / "fits").glob(pattern)) if (p / "complete.json").exists() or not complete]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one fit for {pattern}, found {len(matches)}")
    return matches[0]


def load_fit(folder, graph, device):
    """``{node name: up}`` of a complete fit, checked against the graph it is applied to."""
    summary = unseal(folder / "complete.json")
    learned = {}
    for index, (node, module) in enumerate(weight_layers(graph)):
        up, _, statistics = read_layer(folder / "layers" / f"{index:03d}--{node.name}.npz")
        if statistics["fp32_weight_sha256"] != tensor_sha(module.weight):
            raise ValueError(f"fit {folder.name} was made for a different weight in {node.name}")
        learned[node.name] = up.to(device)
    if len(learned) != summary["layers"]:
        raise ValueError("layer count differs from the fit summary")
    return learned, summary


def _place(total_bytes, device, cap_gib):
    """Hold the layer data on the GPU when it fits beside the other lanes' jobs, else in host memory."""
    if device != "cuda":
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if total_bytes <= cap_gib * GIB and free - total_bytes >= 2.5 * GIB else "cpu"


def learn_layer(index, node, fp_graph, q_graph, fp_run, q_run, inputs, wformat, rule, quantizer, device, seed,
                settings, cap_gib, chunk=64, shared_first_input=True):
    """Learn one layer; sets the learned weight in ``q_graph`` and returns ``(up, scales, statistics)``."""
    tick = time.monotonic()
    fp_module, q_module = fp_graph.get_submodule(node.target), q_graph.get_submodule(node.target)
    label, activation = consumer_activation(fp_graph, node)
    forward = layer_forward(fp_module)
    values, shaped = weight_scale(fp_module, wformat, rule, quantizer)
    weight = fp_module.weight.detach()
    count = len(inputs)
    probe = torch.from_numpy(np.array(inputs[:2], dtype=np.float32)).to(device)
    with torch.no_grad():
        x_probe = layer_input(fp_run, node.name, probe)
        out_probe = activation(forward(weight, x_probe))
    in_bytes = count * x_probe[0].numel() * 4
    out_bytes = count * out_probe[0].numel() * 4
    first = index == 0 and shared_first_input  # the first layer sees the same input on both sides
    keep_target = out_bytes <= in_bytes
    total = in_bytes + (out_bytes if keep_target else (0 if first else in_bytes))
    place = _place(total, device, cap_gib)
    x_hat = torch.empty((count,) + tuple(x_probe.shape[1:]), dtype=torch.float32, device=place)
    side = torch.empty((count,) + tuple((out_probe if keep_target else x_probe).shape[1:]), dtype=torch.float32,
                       device=place) if (keep_target or not first) else x_hat
    with torch.no_grad():
        for start in range(0, count, chunk):
            batch = torch.from_numpy(np.array(inputs[start:start + chunk], dtype=np.float32)).to(device)
            x_fp = layer_input(fp_run, node.name, batch)
            x_hat[start:start + len(batch)] = x_fp if first else layer_input(q_run, node.name, batch)
            if keep_target:
                side[start:start + len(batch)] = activation(forward(weight, x_fp))
            elif not first:
                side[start:start + len(batch)] = x_fp
    collected = time.monotonic() - tick

    def target_of(where):
        rows = side.index_select(0, where.to(side.device)).to(device)
        if keep_target:
            return rows
        with torch.no_grad():
            return activation(forward(weight, rows))

    def error(candidate):
        """Mean reconstruction loss of a fixed weight over all calibration images (the objective without f_reg)."""
        total_loss = 0.0
        with torch.no_grad():
            for start in range(0, count, chunk):
                where = torch.arange(start, min(count, start + chunk))
                output = activation(forward(candidate, x_hat.index_select(0, where.to(x_hat.device)).to(device)))
                total_loss += float(adaround.reconstruction_loss(output, target_of(where))) * len(where)
        return total_loss / count

    lo, hi, _ = adaround.neighbours(weight, shaped, quantizer.levels)
    with torch.no_grad():
        nearest = quantizer(weight, shaped)
    up, statistics = adaround.learn_rounding(weight, shaped, quantizer.levels, forward, activation, x_hat, target_of,
                                             seed=seed, settings=settings, device=device, trace=1000)
    with torch.no_grad():
        learned = adaround.hard_weight(lo, hi, up, shaped)
        statistics.update({
            "node": node.name, "index": index, "kind": type(fp_module).__name__, "activation": label,
            "shape": list(weight.shape), "seed": int(seed), "data_placement": place, "kept": "target" if keep_target else "fp32_input",
            "calibration_error_nearest": error(nearest), "calibration_error_learned": error(learned),
            "calibration_error_fp32_weight": error(weight),
            "differ_from_nearest": int((learned != nearest).sum()),
            "weight_mse_nearest": float((nearest - weight).double().pow(2).mean()),
            "weight_mse_learned": float((learned - weight).double().pow(2).mean()),
            "fp32_weight_sha256": tensor_sha(weight), "learned_weight_sha256": tensor_sha(learned),
            "collect_seconds": collected, "seconds": time.monotonic() - tick})
        q_module.weight.copy_(learned)
    del x_hat, side
    return up, np.asarray(values, dtype=np.float32), statistics


def run_fit(model, wformat, rule, seed, device, *, budget_seconds=600.0, reconstruction_input="fp32_activations",
            aformat=None, recipe_name=None, settings=None, images=CALIBRATION_IMAGES, cap_gib=3.0, max_layers=None,
            log=print):
    """Continue (or start) one fit.  Returns 0 when complete, 3 when the time budget stopped it."""
    from tools.experiment_b.classifier import configure, load_model
    from tools.experiment_b.runner import runtime
    from tools.experiment_b2.data import cached_inputs, v1_calibration
    from tools.experiment_b2.guards import check_graph
    started = time.monotonic()
    settings = dict(settings or adaround.SETTINGS)
    configure(device)
    env = runtime(device)
    graph, transform, _ = load_model(model, device)
    check_graph(graph)
    array, rows = cached_inputs(CALIBRATION_LIST, 2000, transform, build=False)
    inputs, rows = array[:images], rows[:images]
    recipe = None
    if reconstruction_input == "b2_activations":
        from tools.experiment_b2 import frozen  # noqa: F401
        from tools.experiment_b2.recipe import named
        recipe = named(recipe_name)
    spec = fit_spec(model, wformat, rule, seed, reconstruction_input=reconstruction_input, aformat=aformat,
                    recipe=recipe, settings=settings, images=images, env=env, rows=rows)
    folder, identity = fit_folder(spec)
    (folder / "layers").mkdir(parents=True, exist_ok=True)
    if (folder / "spec.json").exists():
        if unseal(folder / "spec.json") != spec:
            raise ValueError("fit folder holds a different specification")
    else:
        seal(folder / "spec.json", spec)
    if (folder / "complete.json").exists():
        log(json.dumps({"fit": folder.name, "status": "already complete"}))
        return 0
    quantizer = TableQuantizer(wformat, "signed", device)
    q_graph = copy.deepcopy(graph)
    fp_run = _CaptureFP(graph)
    if recipe is None:
        q_run = _CaptureFP(q_graph)
    else:
        arrays, maxima, _ = v1_calibration(model)
        plan, quantizers, activation_scales = activation_setup(graph, aformat, recipe, arrays, maxima, device)
        q_run = _CaptureB2(q_graph, quantizers, plan, activation_scales)
    layers = weight_layers(graph)
    done_now, all_statistics = 0, []
    for index, (node, module) in enumerate(layers):
        path = folder / "layers" / f"{index:03d}--{node.name}.npz"
        if path.exists():
            up, scales, statistics = read_layer(path)
            values, shaped = weight_scale(module, wformat, rule, quantizer)
            if not np.array_equal(np.asarray(values, dtype=np.float32), scales):
                raise ValueError(f"stored scales of {node.name} differ from the recomputed ones")
            lo, hi, _ = adaround.neighbours(module.weight.detach(), shaped, quantizer.levels)
            with torch.no_grad():
                learned = adaround.hard_weight(lo, hi, up.to(device), shaped)
                if tensor_sha(learned) != statistics["learned_weight_sha256"]:
                    raise ValueError(f"stored rounding of {node.name} does not reproduce its weight")
                q_graph.get_submodule(node.target).weight.copy_(learned)
            all_statistics.append(statistics)
            continue
        if time.monotonic() - started > budget_seconds or (max_layers is not None and done_now >= max_layers):
            log(json.dumps({"fit": folder.name, "status": "budget reached", "next_layer": index, "layers": len(layers),
                            "wall_seconds": round(time.monotonic() - started, 1)}))
            return 3
        layer_seed = int(seed) * 100003 + index
        up, scales, statistics = learn_layer(index, node, graph, q_graph, fp_run, q_run, inputs, wformat, rule,
                                             quantizer, device, layer_seed, settings, cap_gib,
                                             shared_first_input=recipe is None)
        write_layer(path, up, scales, statistics)
        all_statistics.append(statistics)
        done_now += 1
        log(json.dumps({"layer": index, "of": len(layers), "node": node.name, "act": statistics["activation"],
                        "err_nearest": statistics["calibration_error_nearest"],
                        "err_learned": statistics["calibration_error_learned"],
                        "flipped": statistics["differ_from_nearest"], "weights": statistics["weights"],
                        "unsettled": statistics["far_from_binary_soft_variables"], "place": statistics["data_placement"],
                        "seconds": round(statistics["seconds"], 1)}), flush=True)
    weights = sum(s["weights"] for s in all_statistics)
    summary = {"fit_sha256": identity, "layers": len(layers), "weights": weights,
               "differ_from_nearest": sum(s["differ_from_nearest"] for s in all_statistics),
               "far_from_binary_soft_variables": sum(s["far_from_binary_soft_variables"] for s in all_statistics),
               "clipped_weights": sum(s["clipped_weights"] for s in all_statistics),
               "layers_with_higher_calibration_error_than_nearest":
                   [s["node"] for s in all_statistics if s["calibration_error_learned"] > s["calibration_error_nearest"]],
               "fit_seconds_sum_over_layers": sum(s["seconds"] for s in all_statistics),
               "learned_state_sha256": digest([s["learned_weight_sha256"] for s in all_statistics])}
    seal(folder / "complete.json", summary)
    log(json.dumps({"fit": folder.name, "status": "complete", **{k: v for k, v in summary.items()
                                                                   if k != "layers_with_higher_calibration_error_than_nearest"},
                    "worse_layers": len(summary["layers_with_higher_calibration_error_than_nearest"])}), flush=True)
    return 0
