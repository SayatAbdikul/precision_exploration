"""Low-memory, bit-identical replacement of ``tools.experiment_b2.engine.bias_correct`` (lane Q4, agent r3).

The frozen sequential bias correction keeps the FP32 and the quantized activations of all 256 calibration
images on the GPU (about 5 GB for MobileNetV3-Large).  This module computes the same numbers with the same GPU
operations on the same tensors, in a different schedule:

* FP32 side, once per job: every calibration chunk (32 images) runs through the FP32 graph node by node
  (``_node_values``, as the frozen code), and at every conv/linear node the per-chunk float64 channel sums
  ``t.double().sum(dim=dims)`` are kept (a few kB).  ``mean_fp = sum(per-chunk sums) / count`` is then the very
  expression of the frozen code (Python ``sum`` over the chunks in order, starting at 0).
* Quantized side, per arm: node by node as the frozen code, but each chunk's tensors live in host memory and only
  the chunk in hand is moved to the GPU (copies are exact).  At a conv/linear node the per-chunk float64 sums are
  taken on the GPU before the raw output goes back to the host; after ``delta`` is known each raw chunk returns to
  the GPU, gets ``t + delta`` and the node's quantizer, exactly as the frozen code does per chunk.

Every GPU kernel sees the same input tensor (same values, same shape, chunk 32) as in the frozen code, so with
the frozen deterministic settings the corrections, the report and the corrected weights are bit-identical; the
``gate`` job of ``tools/run/experiment_b2_attrib_lm.py`` checks that against the sealed matrix cells.  An in-place
operator on a tensor that another consumer still needs would make the two schedules differ: that case raises.

``install()`` patches ``bias_correct`` in this process only (in ``tools.experiment_b2_attrib.engine`` and in
``tools.experiment_b2.engine``); no source file of another module is changed.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from tools.experiment_b2.engine import _node_values

OFFLOAD = True              # keep the per-chunk quantized activations in host memory
STATS = {"fp_passes": 0, "fp_seconds": 0.0, "q_passes": 0, "q_seconds": 0.0}
_FP_CACHE = {}


def _chunk(inputs, i, chunk, device):
    return torch.from_numpy(np.array(inputs[i:i + chunk], dtype=np.float32)).to(device)


def _store(t):
    return t.to("cpu") if OFFLOAD else t


def _load(t, device):
    return t.to(device) if OFFLOAD else t


def _guard_inplace(node, out, gpu_inputs, users):
    """Raise if ``out`` aliases an input that another consumer still reads (schedules would then differ)."""
    if not isinstance(out, torch.Tensor):
        raise ValueError(f"{node.name}: non-tensor node output in staged execution")
    for source, tensor in gpu_inputs.items():
        if users[source] > 1 and isinstance(tensor, torch.Tensor) and tensor.data_ptr() == out.data_ptr():
            raise RuntimeError(f"{node.name}: in-place result on {source}, which has other consumers")


def fp_channel_means(fp_graph, plan, inputs, chunk, device):
    """Per conv/linear node: (mean_fp float64 tensor on ``device``, count) with the frozen code's arithmetic."""
    key = (id(fp_graph), id(inputs), len(inputs), chunk, str(device))
    cached = _FP_CACHE.get(key)
    if cached is not None and cached["graph"] is fp_graph and cached["inputs"] is inputs:
        return cached["means"]
    tick = time.monotonic()
    nodes = list(fp_graph.graph.nodes)
    fp_nodes = {n.name: n for n in nodes}
    users = {n.name: len(n.users) for n in nodes}
    targets = [n.name for n in nodes if n.name in plan and plan[n.name]["kind"] in ("conv", "linear")]
    sums = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    for i in range(0, len(inputs), chunk):
        env, remaining = {}, dict(users)
        x = _chunk(inputs, i, chunk, device)
        for node in nodes:
            if node.op == "output":
                break
            if node.op == "placeholder":
                env[node.name] = {0: x}
                continue
            local = {s.name: env[s.name][0] for s in node.all_input_nodes}
            out = _node_values(fp_graph, fp_nodes, env, fp_nodes[node.name], 0)
            _guard_inplace(node, out, local, users)
            if node.name in sums:
                dims = [0] + list(range(2, out.ndim))
                sums[node.name].append(out.double().sum(dim=dims))
                counts[node.name] += out.numel() // out.shape[1]
            env[node.name] = {0: out}
            for source in node.all_input_nodes:
                remaining[source.name] -= 1
                if remaining[source.name] == 0:
                    env.pop(source.name, None)
        del env, x
    means = {name: (sum(sums[name]) / counts[name], counts[name]) for name in targets}
    _FP_CACHE.clear()
    _FP_CACHE[key] = {"graph": fp_graph, "inputs": inputs, "means": means}
    STATS["fp_passes"] += 1
    STATS["fp_seconds"] += time.monotonic() - tick
    return means


def bias_correct(fp_graph, q_graph, plan, quantizers, activation_scales, inputs, chunk=32):
    """Drop-in for ``tools.experiment_b2.engine.bias_correct`` (same arguments, same effect, same report)."""
    device = next(iter(quantizers.values())).levels.device
    scales = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in activation_scales.items()}
    nodes = list(q_graph.graph.nodes)
    if [n.name for n in nodes] != [n.name for n in fp_graph.graph.nodes]:
        raise ValueError("bias correction needs identical FP32 and quantized topologies")
    means = fp_channel_means(fp_graph, plan, inputs, chunk, device)
    corrected = [n.name for n in nodes if n.name in plan and plan[n.name]["kind"] in ("conv", "linear")]
    if sorted(corrected) != sorted(means):
        raise ValueError("conv/linear nodes of the plan differ from the cached FP32 means")
    tick = time.monotonic()
    users = {n.name: len(n.users) for n in nodes}
    remaining = dict(users)
    starts = list(range(0, len(inputs), chunk))
    env_q, report = {}, {}

    def quantize_one(name, tensor):
        row = plan[name]
        if not row["quantizes"]:
            return tensor
        return quantizers[row["signedness"]](tensor, scales[name])

    def run_chunk(node, i):
        local = {s.name: _load(env_q[s.name][i], device) for s in node.all_input_nodes}
        env = {name: {i: tensor} for name, tensor in local.items()}
        out = _node_values(q_graph, nodes, env, node, i)
        _guard_inplace(node, out, local, users)
        return out

    for node in nodes:
        if node.op == "output":
            break
        if node.op == "placeholder":
            env_q[node.name] = [_store(quantize_one(node.name, _chunk(inputs, s, chunk, device))) for s in starts]
            continue
        if plan[node.name]["kind"] in ("conv", "linear"):
            raw, partial = [], []
            for i in range(len(starts)):
                t = run_chunk(node, i)
                dims = [0] + list(range(2, t.ndim))
                partial.append(t.double().sum(dim=dims))
                raw.append(_store(t))
                ndim = t.ndim
                del t
            mean_fp, count = means[node.name]
            mean_q = sum(partial) / count
            delta = (mean_fp - mean_q).to(torch.float32)
            module = q_graph.get_submodule(node.target)
            if module.bias is None:
                raise ValueError("bias correction expects folded biases")
            module.bias.add_(delta)
            shape = (1, -1) + (1,) * (ndim - 2)
            outs = []
            for i in range(len(starts)):
                t = _load(raw[i], device) + delta.reshape(shape)
                raw[i] = None
                outs.append(_store(quantize_one(node.name, t)))
                del t
            report[node.name] = {"max_abs_correction": float(delta.abs().max()),
                                 "rms_correction": float(delta.double().pow(2).mean().sqrt()),
                                 "rms_fp32_channel_mean": float(mean_fp.pow(2).mean().sqrt())}
        else:
            outs = [_store(quantize_one(node.name, run_chunk(node, i))) for i in range(len(starts))]
        env_q[node.name] = outs
        for source in node.all_input_nodes:
            remaining[source.name] -= 1
            if remaining[source.name] == 0:
                env_q.pop(source.name, None)
    env_q.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    STATS["q_passes"] += 1
    STATS["q_seconds"] += time.monotonic() - tick
    return report


def install():
    """Patch ``bias_correct`` in this process (the frozen arms engine and the B2 engine used by ``runner.build``)."""
    import tools.experiment_b2.engine as b2_engine
    import tools.experiment_b2_attrib.engine as attrib_engine
    attrib_engine.bias_correct = bias_correct
    b2_engine.bias_correct = bias_correct
