"""Low-GPU-memory, bit-identical twin of ``tools.experiment_b2.blocks.bias_correct_blocks`` (lane Q3, r4).

``blocks.bias_correct_blocks`` (read-only) keeps the FP32 and the quantized activations of all
256 correction images of every live node on the GPU (about 5 GB for ResNet18, 7-8 GB for
MobileNetV2).  This twin performs the same GPU operations on the same chunks in the same order,
but parks every live activation in host memory between nodes:

* each chunk's node inputs are copied to the GPU (an exact copy that keeps the strides), the node
  is run there exactly as in the original, and the output is copied back;
* the per-chunk channel sums ``t.double().sum(dims)`` are taken on the GPU from the same tensors
  and added in the same chunk order as the original's ``sum(...)`` (Python ``sum`` starts at 0);
* after ``delta`` is known, each quantized chunk goes back to the GPU for ``t + delta`` and the
  storage quantizer (stateless while no audit runs), as in the original.

So the corrected biases, the report and everything after them are bit-identical; only the peak
GPU memory changes (one chunk of one node's operands instead of all live activations).  The
GPU proof is a redirected matrix cell that reproduces an existing record (configuration identity,
which contains the per-layer correction report, logits sha256 and readout arrays); the CPU proof
is ``tests/unit/test_experiment_b2_collapse_lowmem.py``.

``patched()`` swaps the function into ``tools.experiment_b2.blocks`` in this process only and adds
this file to the cell record's ``own_sources``; the configuration identity does not change
(it describes the method, which is unchanged).
"""
from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np
import torch

from tools.experiment_b.common import ROOT, file_hash

THIS = "tools/experiment_b2_collapse/lowmem.py"


def bias_correct_blocks_lowmem(fp_graph, q_graph, plan, quantizer, weights, inputs, chunk=None):
    from tools.experiment_b2 import blocks
    from tools.experiment_b_ext.shared import quantize_axis
    chunk = blocks.BIAS_CHUNK if chunk is None else chunk
    device = quantizer.levels.device
    host = torch.device("cpu")
    nodes = list(q_graph.graph.nodes)
    fp_nodes = {n.name: n for n in fp_graph.graph.nodes}
    if [n.name for n in nodes] != list(fp_nodes):
        raise ValueError("bias correction needs identical FP32 and quantized topologies")
    remaining = {n.name: len(n.users) for n in nodes}
    env_fp, env_q, report = {}, {}, {}
    # the original builds every chunk on the device; the device copy of a host chunk is the same tensor
    chunks = [torch.from_numpy(np.array(inputs[i:i + chunk], dtype=np.float32))
              for i in range(0, len(inputs), chunk)]

    def to_device(t):
        return t.to(device) if isinstance(t, torch.Tensor) else t

    def to_host(t):
        return t.to(host) if isinstance(t, torch.Tensor) else t

    def store_one(name, t_device):
        """``stored`` of the original for one chunk (on the device), returned on the host."""
        if not plan[name]["quantizes"]:
            return to_host(t_device)
        quantizer.context = None
        return to_host(quantize_axis(t_device, quantizer, 1))

    def run(graph, node, env, index, block):
        args = torch.fx.node.map_arg(node.args, lambda n: to_device(env[n.name][index]))
        kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: to_device(env[n.name][index]))
        quantizer.context = None
        result = blocks._evaluate(graph, node, args, kwargs, quantizer, weights if block else None)
        quantizer.context = None
        return result

    for node in nodes:
        if node.op == "output":
            break
        if node.op == "placeholder":
            env_fp[node.name] = chunks
            env_q[node.name] = [store_one(node.name, to_device(t)) for t in chunks]
            continue
        linear = plan[node.name]["kind"] in ("conv", "linear")
        fp_out, q_out, sums_fp, sums_q, count, ndim = [], [], [], [], 0, None
        for i in range(len(chunks)):
            t = run(fp_graph, fp_nodes[node.name], env_fp, i, False)
            if linear:
                ndim = t.ndim
                dims = [0] + list(range(2, t.ndim))
                count += t.numel() // t.shape[1]
                sums_fp.append(t.double().sum(dim=dims))
            fp_out.append(to_host(t))
            del t
        for i in range(len(chunks)):
            t = run(q_graph, node, env_q, i, True)
            if linear:
                sums_q.append(t.double().sum(dim=dims))
                q_out.append(to_host(t))
            else:
                q_out.append(store_one(node.name, t))
            del t
        if linear:
            mean_fp = sum(sums_fp) / count
            mean_q = sum(sums_q) / count
            delta = (mean_fp - mean_q).to(torch.float32)
            module = q_graph.get_submodule(node.target)
            if module.bias is None:
                raise ValueError("bias correction expects folded biases")
            module.bias.add_(delta)
            shape = (1, -1) + (1,) * (ndim - 2)
            q_out = [store_one(node.name, to_device(t) + delta.reshape(shape)) for t in q_out]
            report[node.name] = {"max_abs_correction": float(delta.abs().max()),
                                 "rms_correction": float(delta.double().pow(2).mean().sqrt()),
                                 "rms_fp32_channel_mean": float(mean_fp.pow(2).mean().sqrt())}
        env_fp[node.name], env_q[node.name] = fp_out, q_out
        for source in node.all_input_nodes:
            remaining[source.name] -= 1
            if remaining[source.name] == 0:
                env_fp.pop(source.name, None)
                env_q.pop(source.name, None)
    return report


@contextmanager
def patched():
    """``blocks.bias_correct_blocks`` -> the low-memory twin, in this process only."""
    from tools.experiment_b2 import blocks, matrix
    original_function, original_own = blocks.bias_correct_blocks, matrix.own_sources

    def own_sources():
        return {**original_own(), THIS: file_hash(ROOT / THIS)}

    blocks.bias_correct_blocks, matrix.own_sources = bias_correct_blocks_lowmem, own_sources
    try:
        yield
    finally:
        blocks.bias_correct_blocks, matrix.own_sources = original_function, original_own


def validate_identity(model, format_name, recipe_name, device, matrix, original_matrix):
    """Build one cell's configuration with the low-memory correction (no evaluation) and compare its identity,
    which contains the per-layer correction report and the quantized-weight hashes, with the original record."""
    import gc
    from types import SimpleNamespace
    import torch as _torch
    from tools.experiment_b.common import seal, unseal
    from .cells import BASE
    found = sorted((original_matrix / "cells").glob(f"{model}--{format_name}--{recipe_name}--1000--*.json"))
    if len(found) != 1:
        raise SystemExit(f"no unique original matrix cell for {model}/{format_name}/{recipe_name}")
    original = unseal(found[0])
    args = SimpleNamespace(model=model, format=format_name, recipe=recipe_name, device=device, images=matrix.IMAGES)
    build = matrix.build_shared(args)
    identity, layers = build[5], len(build[4]["scales"].get("bias_correction") or {})
    check = {"original_cell": str(found[0].relative_to(ROOT)), "identity_equal": original["configuration_sha256"] == identity,
             "recomputed_identity": identity, "corrected_layers": layers, "evaluated": False,
             "bias_correction": THIS}
    del build
    gc.collect()
    if device.startswith("cuda"):
        _torch.cuda.empty_cache()
    path = BASE / "matrix" / "checks" / f"{model}--{format_name}--{recipe_name}--lowmem-identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        seal(path, check)
    return check


def run_cells_lowmem(model, format_name, recipes, device, validate=None):
    """``cells.run_cells`` with the low-memory correction; prints the peak GPU memory of each cell.

    ``validate``: recipe whose configuration identity is first rebuilt and compared with the original matrix
    record (no evaluation).  Any failed reproduction (identity-only, or a full cell that has an original
    counterpart) stops the process before the next cell (exit 5)."""
    import torch as _torch
    from tools.experiment_b2 import frozen  # noqa: F401
    from .cells import BASE, compare, redirected
    from tools.experiment_b.common import seal
    with redirected() as (matrix, original_matrix), patched():
        if validate:
            check = validate_identity(model, format_name, validate, device, matrix, original_matrix)
            print(json.dumps({"model": model, "format": format_name, "validate": validate, "check": check}), flush=True)
            if not check["identity_equal"]:
                return 5
        for recipe in recipes:
            if device.startswith("cuda"):
                _torch.cuda.reset_peak_memory_stats()
            record = matrix.run_cell(model, format_name, recipe, device)
            check = compare(record, original_matrix)
            if check is not None:
                path = BASE / "matrix" / "checks" / f"{model}--{format_name}--{recipe}--lowmem.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    seal(path, {**check, "bias_correction": THIS})
            peak = _torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
            print(json.dumps({"model": model, "format": format_name, "recipe": recipe, "status": record["status"],
                              "top1_expected": round(record["readout"]["top1_expected_percent"], 2),
                              "peak_gpu_mib_allocated": None if peak is None else round(peak),
                              "check": check}), flush=True)
            if check is not None and not all(check.values()):
                return 5
    return 0
