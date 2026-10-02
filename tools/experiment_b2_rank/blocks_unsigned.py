"""Shared-exponent block engine with unsigned activation blocks at non-negative nodes (lane Q2).

``tools.experiment_b2.blocks`` (read-only) has one activation quantizer for every stored tensor and every
convolution patch.  This module is a copy of its ``prepare_blocks`` / ``bias_correct_blocks`` / interpreter with one
change: a second activation quantizer over the unsigned sibling codebook (``unsigned.sibling_table``) is used

* for the stored tensor of every quantizing node that ``boundaries.analyze`` proves non-negative, and
* for the convolution patches of every convolution whose input node is proven non-negative.

Weights keep the signed codebook.  With ``unsigned=False`` (no node uses the second quantizer) the engine is the
original one; a unit test proves bit-identity against ``blocks.prepare_blocks`` on a small graph, and a GPU check
reproduces a sealed L1 cell.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn

from tools.experiment_b2.blocks import (BIAS_CHUNK, BLOCK, BLOCK_PROTOCOL, B2BlockQuantizer, BlockInterpreter,
                                        _evaluate, audit_rows, block_plan)
from tools.experiment_b_ext.shared import quantize_axis
from .unsigned import RULE, describe, sibling_table


class UnsignedBlockQuantizer(B2BlockQuantizer):
    """``B2BlockQuantizer`` over the unsigned sibling element codebook (same block, scale and search rules)."""

    def __init__(self, name, rule, device):
        super().__init__(name, rule, device)
        levels, boundaries, ties = sibling_table(name)
        self.levels = torch.tensor(levels, device=device)
        self.boundaries = torch.tensor(boundaries, device=device)
        self.ties = torch.tensor(ties, device=device)
        self.signedness = "unsigned"


def unsigned_sets(graph, plan, enabled=True):
    """Names of stored tensors and of convolutions (patches) that take the unsigned codebook."""
    if not enabled:
        return frozenset(), frozenset()
    stored = {name for name, row in plan.items() if row["quantizes"] and row["nonnegative"]}
    patches = set()
    for node in graph.graph.nodes:
        if node.op == "call_module" and isinstance(graph.get_submodule(node.target), nn.Conv2d):
            source = node.args[0]
            if isinstance(source, torch.fx.Node) and plan[source.name]["nonnegative"]:
                patches.add(node.name)
    return frozenset(stored), frozenset(patches)


class UnsignedBlockInterpreter(BlockInterpreter):
    def __init__(self, graph, quantizer, unsigned_quantizer, weights, plan, stored, patches):
        super().__init__(graph, quantizer, weights, plan)
        self.unsigned_quantizer, self.unsigned_stored, self.unsigned_patches = unsigned_quantizer, stored, patches

    def run_node(self, node):
        if node.op == "call_module" and isinstance(self.module.get_submodule(node.target), (nn.Conv2d, nn.Linear)):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            q = self.unsigned_quantizer if node.name in self.unsigned_patches else self.quantizer
            result = _evaluate(self.module, node, args, kwargs, q, self.weights)
        else:
            result = torch.fx.Interpreter.run_node(self, node)
        row = self.plan.get(node.name)
        if row is not None and row["quantizes"]:
            q = self.unsigned_quantizer if node.name in self.unsigned_stored else self.quantizer
            q.context = f"stored:{node.name}"
            result = quantize_axis(result, q, 1)
        return result


def bias_correct_blocks(fp_graph, q_graph, plan, quantizer, unsigned_quantizer, stored_set, patch_set, weights,
                        inputs, chunk=BIAS_CHUNK):
    """``blocks.bias_correct_blocks`` with the per-node quantizer choice of this module (otherwise identical)."""
    device = quantizer.levels.device
    nodes = list(q_graph.graph.nodes)
    fp_nodes = {n.name: n for n in fp_graph.graph.nodes}
    if [n.name for n in nodes] != list(fp_nodes):
        raise ValueError("bias correction needs identical FP32 and quantized topologies")
    remaining = {n.name: len(n.users) for n in nodes}
    env_fp, env_q, report = {}, {}, {}
    chunks = [torch.from_numpy(np.array(inputs[i:i + chunk], dtype=np.float32)).to(device)
              for i in range(0, len(inputs), chunk)]

    def stored(name, tensors):
        if not plan[name]["quantizes"]:
            return tensors
        q = unsigned_quantizer if name in stored_set else quantizer
        q.context = None
        return [quantize_axis(t, q, 1) for t in tensors]

    def run(graph, node, env, index, block):
        args = torch.fx.node.map_arg(node.args, lambda n: env[n.name][index])
        kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: env[n.name][index])
        q = unsigned_quantizer if node.name in patch_set else quantizer
        q.context = None
        result = _evaluate(graph, node, args, kwargs, q, weights if block else None)
        q.context = None
        return result

    for node in nodes:
        if node.op == "output":
            break
        if node.op == "placeholder":
            env_fp[node.name], env_q[node.name] = chunks, stored(node.name, chunks)
            continue
        fp_out = [run(fp_graph, fp_nodes[node.name], env_fp, i, False) for i in range(len(chunks))]
        q_out = [run(q_graph, node, env_q, i, True) for i in range(len(chunks))]
        if plan[node.name]["kind"] in ("conv", "linear"):
            dims = [0] + list(range(2, fp_out[0].ndim))
            count = sum(t.numel() // t.shape[1] for t in fp_out)
            mean_fp = sum(t.double().sum(dim=dims) for t in fp_out) / count
            mean_q = sum(t.double().sum(dim=dims) for t in q_out) / count
            delta = (mean_fp - mean_q).to(torch.float32)
            module = q_graph.get_submodule(node.target)
            if module.bias is None:
                raise ValueError("bias correction expects folded biases")
            module.bias.add_(delta)
            shape = (1, -1) + (1,) * (fp_out[0].ndim - 2)
            q_out = [t + delta.reshape(shape) for t in q_out]
            report[node.name] = {"max_abs_correction": float(delta.abs().max()),
                                 "rms_correction": float(delta.double().pow(2).mean().sqrt()),
                                 "rms_fp32_channel_mean": float(mean_fp.pow(2).mean().sqrt())}
        env_fp[node.name], env_q[node.name] = fp_out, stored(node.name, q_out)
        for source in node.all_input_nodes:
            remaining[source.name] -= 1
            if remaining[source.name] == 0:
                env_fp.pop(source.name, None)
                env_q.pop(source.name, None)
    return report


def prepare_blocks(graph, name, recipe, device, *, bias_inputs=None, unsigned=True):
    """``blocks.prepare_blocks`` plus unsigned activation blocks at proven non-negative nodes."""
    plan = block_plan(graph, recipe, name)
    activation = B2BlockQuantizer(name, recipe.activation_range, device)
    unsigned_activation = UnsignedBlockQuantizer(name, recipe.activation_range, device)
    weight = B2BlockQuantizer(name, recipe.weight_range, device)
    result = copy.deepcopy(graph)
    stored_set, patch_set = unsigned_sets(result, plan, unsigned)
    weights, weight_info = {}, {}
    with torch.no_grad():
        for node in result.graph.nodes:
            if node.op != "call_module":
                continue
            module = result.get_submodule(node.target)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                flat = module.weight.detach().reshape(module.weight.shape[0], -1)
                quantized = weight(flat).reshape_as(module.weight)
                weights[node.name] = quantized
                error = (quantized - module.weight).double().pow(2).sum()
                weight_info[node.name] = {
                    "sha256_of_fp32_bytes": hashlib.sha256(quantized.cpu().numpy().tobytes()).hexdigest(),
                    "sqnr_db": float(10 * torch.log10(module.weight.double().pow(2).sum() / error)) if error > 0 else None,
                    "rows": int(flat.shape[0]), "k": int(flat.shape[1]),
                    "blocks": int(flat.shape[0]) * -(-int(flat.shape[1]) // BLOCK)}
        bias_report = None
        if recipe.bias_correction == "empirical":
            if bias_inputs is None:
                raise ValueError("bias correction needs calibration inputs")
            with torch.inference_mode():
                bias_report = bias_correct_blocks(graph, result, plan, activation, unsigned_activation, stored_set,
                                                  patch_set, weights, bias_inputs)
    metadata = {"block": BLOCK_PROTOCOL, "activation_rule": recipe.activation_range, "weight_rule": recipe.weight_range,
                "weights": weight_info,
                "boundaries": {k: {"quantizes": v["quantizes"], "reason": v["reason"], "kind": v["kind"]}
                               for k, v in plan.items()}}
    if bias_report is not None:
        metadata["bias_correction"] = bias_report
    if unsigned:
        metadata["rank_unsigned_blocks"] = {"rule": RULE, "codebook": describe(name),
                                            "unsigned_stored": sorted(stored_set), "unsigned_patches": sorted(patch_set)}
    engine = UnsignedBlockInterpreter(result, activation, unsigned_activation, weights, plan, stored_set, patch_set)
    return engine, metadata


def occupancy(engine, inputs, device, images):
    """``matrix.shared_occupancy`` over both activation quantizers (row keys carry a ``:unsigned`` suffix)."""
    from tools.experiment_b2.matrix import occupancy_summary
    if engine.quantizer.rule == "percentile_99_9":
        return None
    for q in (engine.quantizer, engine.unsigned_quantizer):
        q.audit = {}
    try:
        with torch.inference_mode():
            for start in range(0, images, 8):
                engine.run(torch.from_numpy(np.array(inputs[start:start + 8], dtype=np.float32)).to(device))
        rows = dict(audit_rows(engine.quantizer))
        rows.update({f"{k}:unsigned": v for k, v in audit_rows(engine.unsigned_quantizer).items()})
    finally:
        for q in (engine.quantizer, engine.unsigned_quantizer):
            q.audit = None
    stored = {k[7:]: v for k, v in rows.items() if k.startswith("stored:")}
    patches = {k[6:]: v for k, v in rows.items() if k.startswith("patch:")}
    return {"images": images, "summary": occupancy_summary(stored), "patch_summary": occupancy_summary(patches),
            "nodes": stored, "patches": patches}
