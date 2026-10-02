"""Shared-exponent formats (BFP6, MXFP4/6/8) under the B2 boundary and range rules.

Block semantics are those validated in ``tools/experiment_b_ext/shared.py`` (read-only,
imported): one E8M0 power-of-two scale per block of 32 elements, stored activations
blocked along the channel axis, convolution patches and weights blocked along the
reduction axis K, wide FP32 accumulation.  What B2 adds:

* boundaries: the B2 plan (``boundaries.analyze``) decides which tensors are blocked,
  with one change: a max-pool output is always re-blocked, because its elements come
  from source blocks with different exponents and cannot be forwarded as codes;
* range rules: ``maxabs`` and ``percentile_99_9`` are the extension's rules, executed
  by the extension's own code.  ``mse`` is the B2 scale search restricted to what an
  E8M0 scale can express: per block, the exponent ``e_maxabs - d`` with ``d`` in
  ``0..6`` that minimises the block's squared reconstruction error (ties: smallest
  ``d``).  Larger exponents are never better for these codebooks (doubling a level
  below half the top level gives a level), so they are not searched;
* bias correction: the B2 sequential empirical correction, executed on the block engine;
* unsigned codes: not defined for these formats (as for every non-integer format).

For weights the search is offline.  For activations it is a run-time operation
(seven trial encodings per block); the intrinsic alternative is the ``maxabs`` rule.
With ``boundaries=v1``, no bias correction and both rules ``maxabs`` (or both
``percentile_99_9``) this module reproduces ``prepare_shared`` bit for bit.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.experiment_b.common import ROOT, file_hash
from tools.experiment_b_ext.shared import BlockQuantizer, block_conv2d, quantize_axis
from .boundaries import analyze, quantizing_nodes

SHARED = ("bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3")
BLOCK = 32
CLIP_OCTAVES = 6
BIAS_CHUNK = 8
CHUNK_BLOCKS = 1 << 17   # blocks per slice of the vectorised path (4.2 million elements)
BLOCK_PROTOCOL = {
    "version": "b2-shared-block-1",
    "semantics": "tools.experiment_b_ext.shared: e8m0 scale per 32 elements; stored activations along the channel "
                 "axis; convolution patches and weights along reduction K; fp32 framework reduction",
    "rules": {"maxabs": "extension rule (smallest covering power of two of block max-abs)",
              "percentile_99_9": "extension rule",
              "mse": "per block: exponent(maxabs) - d, d = 0..6, minimising the block sum of squared errors "
                     "(float64); ties keep the smallest d; incomplete blocks zero-padded"},
    "clip_octaves": CLIP_OCTAVES,
    "maxpool": "requantized into new blocks (no code passthrough)",
    "unsigned": "not defined for shared-exponent formats",
    "bias_correction": "b2 sequential empirical correction on the block engine",
    "bias_correction_chunk": BIAS_CHUNK,
    "activation_search_is_runtime": True,
}


def block_sources():
    return {name: file_hash(ROOT / name) for name in ("tools/experiment_b2/blocks.py",
                                                      "tools/experiment_b_ext/shared.py")}


class B2BlockQuantizer(BlockQuantizer):
    """The extension's block quantizer plus the per-block exponent search and an occupancy audit."""

    def __init__(self, name, rule, device):
        if rule not in ("maxabs", "percentile_99_9", "mse"):
            raise ValueError(f"unknown block rule: {rule}")
        super().__init__(name, "maxabs" if rule == "mse" else rule, device)
        self.rule = rule
        self.audit = None      # dict when auditing
        self.context = None    # key of the tensor being quantized (set by the interpreter)

    def __call__(self, values):
        if self.rule != "mse" and self.audit is None:
            return super().__call__(values)
        if self.rule == "percentile_99_9":
            raise ValueError("the occupancy audit is not defined for the percentile block rule")
        return self.core(values, search=self.rule == "mse")

    def core(self, values, search):
        """Vectorised block QDQ with the max-abs exponent, optionally searching ``d`` octaves of clipping.

        Blocks are independent, so the tensor is processed in slices of ``CHUNK_BLOCKS`` blocks to bound memory.
        """
        if not torch.isfinite(values).all() or values.ndim < 1 or values.shape[-1] < 1:
            raise ValueError("invalid shared-block input")
        k = values.shape[-1]
        pad = (-k) % BLOCK
        x = (F.pad(values, (0, pad)) if pad else values).reshape(-1, BLOCK)
        pieces, counts, clips = [], None, None
        for begin in range(0, x.shape[0], CHUNK_BLOCKS):
            output, positions, clip = self._chunk(x[begin:begin + CHUNK_BLOCKS], search)
            pieces.append(output)
            if self.audit is not None and self.context is not None:
                count = torch.bincount(positions.flatten(), minlength=len(self.levels)).double()
                chosen = torch.bincount(clip.flatten(), minlength=CLIP_OCTAVES + 1).double()
                counts, clips = (count, chosen) if counts is None else (counts + count, clips + chosen)
        result = torch.cat(pieces).reshape(*values.shape[:-1], -1)[..., :k]
        if counts is not None:
            if pad:  # padded elements are exact zeros and were counted at the zero level
                counts[int((self.levels == 0).nonzero()[0])] -= pad * (values.numel() // k)
            entry = self.audit.setdefault(self.context, {"counts": torch.zeros_like(counts), "clip": torch.zeros_like(clips),
                                                         "signal": 0.0, "noise": 0.0})
            entry["counts"] += counts
            entry["clip"] += clips
            entry["signal"] += float(values.double().pow(2).sum())
            entry["noise"] += float((values - result).double().pow(2).sum())
        return result

    def _chunk(self, x, search):
        target = x.abs().amax(dim=-1, keepdim=True) / self.levels[-1]
        mantissa, exponent = torch.frexp(target)
        exponent = torch.where(target == 0, torch.full_like(exponent, -127),
                               torch.where(mantissa == .5, exponent - 1, exponent)).clamp(-127, 127)
        best = None
        for d in range(CLIP_OCTAVES + 1 if search else 1):
            scale = torch.ldexp(torch.ones_like(target), (exponent - d).clamp(-127, 127))
            y = x / scale
            positions = torch.bucketize(y.contiguous(), self.boundaries, right=False)
            adjacent = positions.clamp_max(len(self.boundaries) - 1)
            positions += ((y == self.boundaries[adjacent]) & self.ties[adjacent]).long()
            del adjacent
            reconstructed = self.levels[positions]
            output = torch.where(reconstructed == 0, torch.copysign(reconstructed, y), reconstructed) * scale
            del reconstructed, y
            positions = positions.to(torch.int16)
            if not search:
                return output, positions.long(), torch.zeros_like(exponent)
            error = (output - x).double().pow(2).sum(dim=-1, keepdim=True)
            if best is None:
                best, best_error = (output, positions, torch.zeros_like(exponent)), error
            else:
                better = error < best_error
                best = (torch.where(better, output, best[0]), torch.where(better, positions, best[1]),
                        torch.where(better, torch.full_like(exponent, d), best[2]))
                best_error = torch.where(better, error, best_error)
        return best[0], best[1].long(), best[2]


def block_plan(graph, recipe, name):
    """The B2 boundary plan with max-pool outputs re-blocked."""
    if name not in SHARED:
        raise ValueError(f"not a shared-exponent format: {name}")
    if recipe.equalization != "none" or not recipe.quantize_input or not recipe.quantize_logits:
        raise ValueError("shared-exponent cells support neither equalization nor boundary exemptions")
    plan = analyze(graph, recipe, name)
    for row in plan.values():
        if row["reason"] == "maxpool_passthrough":
            row.update(quantizes=True, reason=None, signedness="signed")
    if any(row["signedness"] == "unsigned" for row in plan.values()):
        raise ValueError("unsigned boundary in a shared-exponent plan")
    return plan


def _evaluate(graph, node, args, kwargs, quantizer, weights):
    """One FX node on the block engine (``weights is None``: plain FP32)."""
    if node.op == "call_module":
        module = graph.get_submodule(node.target)
        if weights is not None and isinstance(module, nn.Conv2d):
            quantizer.context = f"patch:{node.name}"
            return block_conv2d(args[0], module, quantizer, weights[node.name])
        if weights is not None and isinstance(module, nn.Linear):
            return F.linear(args[0], weights[node.name], module.bias)
        return module(*args, **kwargs)
    if node.op == "call_function":
        return node.target(*args, **kwargs)
    raise ValueError(f"unsupported node on the block engine: {node.op}")


class BlockInterpreter(torch.fx.Interpreter):
    def __init__(self, graph, quantizer, weights, plan):
        super().__init__(graph)
        self.quantizer, self.weights, self.plan = quantizer, weights, plan

    def run_node(self, node):
        if node.op == "call_module" and isinstance(self.module.get_submodule(node.target), (nn.Conv2d, nn.Linear)):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            result = _evaluate(self.module, node, args, kwargs, self.quantizer, self.weights)
        else:
            result = super().run_node(node)
        row = self.plan.get(node.name)
        if row is not None and row["quantizes"]:
            self.quantizer.context = f"stored:{node.name}"
            result = quantize_axis(result, self.quantizer, 1)
        return result


def bias_correct_blocks(fp_graph, q_graph, plan, quantizer, weights, inputs, chunk=BIAS_CHUNK):
    """``engine.bias_correct`` for the block engine: same sequential rule, block operators."""
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
        quantizer.context = None
        return [quantize_axis(t, quantizer, 1) for t in tensors]

    def run(graph, node, env, index, block):
        args = torch.fx.node.map_arg(node.args, lambda n: env[n.name][index])
        kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: env[n.name][index])
        quantizer.context = None
        result = _evaluate(graph, node, args, kwargs, quantizer, weights if block else None)
        quantizer.context = None
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


def prepare_blocks(graph, name, recipe, device, *, bias_inputs=None):
    """Block engine for one shared-exponent format and B2 recipe: ``(interpreter, metadata)``."""
    plan = block_plan(graph, recipe, name)
    activation = B2BlockQuantizer(name, recipe.activation_range, device)
    weight = B2BlockQuantizer(name, recipe.weight_range, device)
    result = copy.deepcopy(graph)
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
                bias_report = bias_correct_blocks(graph, result, plan, activation, weights, bias_inputs)
    metadata = {"block": BLOCK_PROTOCOL, "activation_rule": recipe.activation_range, "weight_rule": recipe.weight_range,
                "weights": weight_info,
                "boundaries": {k: {"quantizes": v["quantizes"], "reason": v["reason"], "kind": v["kind"]}
                               for k, v in plan.items()}}
    if bias_report is not None:
        metadata["bias_correction"] = bias_report
    return BlockInterpreter(result, activation, weights, plan), metadata


def audit_rows(quantizer):
    """Occupancy per audited tensor (``stored:<node>`` and ``patch:<conv>``) after an audited pass."""
    rows = {}
    for key, entry in quantizer.audit.items():
        p = (entry["counts"] / entry["counts"].sum()).cpu().numpy()
        clip = (entry["clip"] / entry["clip"].sum()).cpu().numpy()
        levels = quantizer.levels.cpu().numpy()
        nonzero = p[p > 0]
        rows[key] = {"elements": float(entry["counts"].sum()), "blocks": float(entry["clip"].sum()),
                     "entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
                     "levels_used": int((p > 0).sum()), "levels_total": int(len(p)),
                     "fraction_zero": float(p[levels == 0].sum()), "fraction_at_extremes": float(p[0] + p[-1]),
                     "fraction_blocks_clipped": float(1 - clip[0]),
                     "mean_clip_octaves": float((clip * np.arange(len(clip))).sum()),
                     "sqnr_db": float(10 * np.log10(entry["signal"] / entry["noise"])) if entry["noise"] > 0 else None}
    return rows


def metadata_bits(interpreter, sample):
    """Scale metadata per element: 8 bits per block of up to 32 elements (run on one image batch)."""
    shapes = {}

    class Recorder(BlockInterpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if isinstance(result, torch.Tensor):
                shapes[node.name] = tuple(result.shape)
            return result

    with torch.inference_mode():
        Recorder(interpreter.module, interpreter.quantizer, interpreter.weights, interpreter.plan).run(sample)
    images = sample.shape[0]
    stored_elements = stored_blocks = patch_elements = patch_blocks = 0
    for node in interpreter.module.graph.nodes:
        row = interpreter.plan.get(node.name)
        if row is not None and row["quantizes"]:
            shape = shapes[node.name]
            positions = int(np.prod(shape[2:])) if len(shape) > 2 else 1
            stored_elements += shape[1] * positions
            stored_blocks += -(-shape[1] // BLOCK) * positions
        if node.op == "call_module" and isinstance(interpreter.module.get_submodule(node.target), nn.Conv2d):
            module = interpreter.module.get_submodule(node.target)
            out = shapes[node.name]
            k = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
            locations = module.groups * out[2] * out[3]
            patch_elements += k * locations
            patch_blocks += -(-k // BLOCK) * locations
    weight_elements = sum(int(w.numel()) for w in interpreter.weights.values())
    weight_blocks = sum(int(w.shape[0]) * -(-int(w[0].numel()) // BLOCK) for w in interpreter.weights.values())
    return {"rule": "one 8-bit E8M0 scale per block of up to 32 elements; an incomplete block still carries one scale",
            "images_in_sample": images,
            "weights": {"elements": weight_elements, "blocks": weight_blocks,
                        "scale_bits_per_element": 8 * weight_blocks / weight_elements},
            "stored_activations_per_image": {"elements": stored_elements, "blocks": stored_blocks,
                                             "scale_bits_per_element": 8 * stored_blocks / stored_elements},
            "convolution_patches_per_image": {"elements": patch_elements, "blocks": patch_blocks,
                                              "scale_bits_per_element": 8 * patch_blocks / patch_elements,
                                              "note": "transient MAC-operand blocks; scales exist in the datapath, "
                                                      "they are not stored with the feature map"}}
