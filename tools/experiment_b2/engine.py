"""B2 quantize-dequantize engine: v1 arithmetic with recipe-controlled boundaries.

With the v1-equivalent recipes this module calls the very same v1 functions
(``threshold``, ``scale_for``, ``Quantizer.__call__``) on the same inputs in
the same order, so its output is bit-identical to ``prepare_qdq`` of v1.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from tools.experiment_b.quantizer import scale_for, threshold
from .boundaries import analyze, kind, quantizing_nodes
from .codebook import TableQuantizer, codebook_id
from .scales import SEARCH, mse_search, scale_for_top, simple_span


class B2Interpreter(torch.fx.Interpreter):
    def __init__(self, graph, quantizers, plan, activation_scales, audit=False):
        super().__init__(graph)
        device = next(iter(quantizers.values())).levels.device
        self.quantizers = quantizers
        self.plan = plan
        self.scales = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in activation_scales.items()}
        if set(quantizing_nodes(plan)) != set(self.scales):
            raise ValueError("activation scale/node coverage mismatch")
        self.audit = {} if audit else None

    def run_node(self, node):
        result = super().run_node(node)
        row = self.plan.get(node.name)
        if row is not None and row["quantizes"]:
            quantizer, scale = self.quantizers[row["signedness"]], self.scales[node.name]
            quantized = quantizer(result, scale)
            if self.audit is not None:
                self._record(node.name, quantizer, result, quantized, scale)
            result = quantized
        return result

    def _record(self, name, quantizer, raw, quantized, scale):
        counts = torch.bincount(quantizer.indices(raw, scale).flatten(), minlength=len(quantizer.levels)).double()
        entry = self.audit.setdefault(name, {"counts": torch.zeros_like(counts), "signal": 0.0, "noise": 0.0})
        entry["counts"] += counts
        entry["signal"] += float((raw.double() ** 2).sum())
        entry["noise"] += float(((raw - quantized).double() ** 2).sum())


def _node_values(graph, nodes, env, node, index):
    args = torch.fx.node.map_arg(node.args, lambda n: env[n.name][index])
    kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: env[n.name][index])
    if node.op == "call_module":
        return graph.get_submodule(node.target)(*args, **kwargs)
    if node.op == "call_function":
        return node.target(*args, **kwargs)
    raise ValueError(f"unsupported node in staged execution: {node.op}")


def bias_correct(fp_graph, q_graph, plan, quantizers, activation_scales, inputs, chunk=32):
    """Sequential empirical bias correction (in place on ``q_graph``).

    Both graphs are executed node by node over all calibration inputs.  At
    every conv/linear node the per-output-channel mean of the FP32 output
    minus the mean of the quantized-network output is added to the bias, and
    the already computed quantized outputs are shifted by the same amount, so
    every later layer sees its corrected inputs (one pass, exact ordering).
    """
    device = next(iter(quantizers.values())).levels.device
    scales = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in activation_scales.items()}
    nodes = list(q_graph.graph.nodes)
    if [n.name for n in nodes] != [n.name for n in fp_graph.graph.nodes]:
        raise ValueError("bias correction needs identical FP32 and quantized topologies")
    fp_nodes = {n.name: n for n in fp_graph.graph.nodes}
    remaining = {n.name: len(n.users) for n in nodes}
    env_fp, env_q, report = {}, {}, {}
    chunks = [torch.from_numpy(np.array(inputs[i:i + chunk], dtype=np.float32)).to(device) for i in range(0, len(inputs), chunk)]

    def quantize(name, tensors):
        row = plan[name]
        if not row["quantizes"]:
            return tensors
        return [quantizers[row["signedness"]](t, scales[name]) for t in tensors]

    for node in nodes:
        if node.op == "output":
            break
        if node.op == "placeholder":
            env_fp[node.name] = chunks
            env_q[node.name] = quantize(node.name, chunks)
            continue
        fp_out = [_node_values(fp_graph, fp_nodes, env_fp, fp_nodes[node.name], i) for i in range(len(chunks))]
        q_out = [_node_values(q_graph, nodes, env_q, node, i) for i in range(len(chunks))]
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
        env_fp[node.name] = fp_out
        env_q[node.name] = quantize(node.name, q_out)
        for source in node.all_input_nodes:
            remaining[source.name] -= 1
            if remaining[source.name] == 0:
                env_fp.pop(source.name, None)
                env_q.pop(source.name, None)
    return report


def prepare_b2(graph, name, recipe, arrays, maxima, device, *, bias_inputs=None, audit=False):
    """Build the B2 engine for one format and recipe.

    ``arrays``/``maxima`` are the v1 FP32 calibration observations.
    ``bias_inputs`` (preprocessed calibration images) is required when the
    recipe uses bias correction.  Returns ``(interpreter, metadata)``.
    """
    plan = analyze(graph, recipe, name)
    quantizers = {"signed": TableQuantizer(name, "signed", device)}
    if any(row["signedness"] == "unsigned" for row in plan.values()):
        quantizers["unsigned"] = TableQuantizer(name, "unsigned", device)
    result = copy.deepcopy(graph)
    equalization = None
    if recipe.equalization != "none":
        from .equalization import equalize
        equalization, arrays, maxima = equalize(result, plan, arrays, maxima)
    activation_scales, activation_info = {}, {}
    v1_like = recipe.v1_equivalent() is not None
    for node in quantizing_nodes(plan):
        quantizer = quantizers[plan[node]["signedness"]]
        if recipe.activation_range == "mse":
            values = torch.tensor(arrays[node], dtype=torch.float32, device=device).reshape(1, -1)
            scales, info = mse_search(values, quantizer, maxima=[maxima[node]])
            activation_scales[node] = float(scales[0])
            activation_info[node] = {k: v[0] for k, v in info.items()}
        elif v1_like:
            activation_scales[node] = scale_for(maxima[node] if recipe.activation_range == "maxabs"
                                                else threshold(arrays[node], recipe.activation_range), name)
        else:
            activation_scales[node] = scale_for_top(simple_span(arrays[node], maxima[node], recipe.activation_range),
                                                    quantizer.top)
    weight_scales, weight_info = {}, {}
    signed = quantizers["signed"]
    with torch.no_grad():
        for node in result.graph.nodes:
            if node.op != "call_module":
                continue
            module = result.get_submodule(node.target)
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if recipe.weight_range == "mse":
                found, info = mse_search(module.weight.detach().reshape(module.weight.shape[0], -1), signed)
                scales = [float(x) for x in found]
                weight_info[node.name] = {"median_ratio_to_maxabs_scale": float(np.median(info["ratio_to_maxabs_scale"])),
                                          "min_ratio_to_maxabs_scale": float(np.min(info["ratio_to_maxabs_scale"]))}
            else:
                source = module.weight.detach().cpu().numpy()
                scales = [scale_for(threshold(channel, recipe.weight_range), name) for channel in source]
            shaped = torch.tensor(scales, dtype=torch.float32, device=device).reshape((-1,) + (1,) * (module.weight.ndim - 1))
            module.weight.copy_(signed(module.weight, shaped))
            weight_scales[node.name] = scales
        bias_report = None
        if recipe.bias_correction == "empirical":
            if bias_inputs is None:
                raise ValueError("bias correction needs calibration inputs")
            with torch.inference_mode():
                bias_report = bias_correct(graph, result, plan, quantizers, activation_scales, bias_inputs)
    metadata = {"activation_scales": activation_scales, "weight_scales": weight_scales,
                "scale_storage": "fp32; include_scale_metadata_and_application_cost_in_hardware_analysis",
                "boundaries": {k: {"quantizes": v["quantizes"], "reason": v["reason"], "signedness": v["signedness"],
                                   "nonnegative": v["nonnegative"], "kind": v["kind"],
                                   "codebook": codebook_id(name, v["signedness"]) if v["quantizes"] else None}
                               for k, v in plan.items()}}
    if recipe.activation_range == "mse" or recipe.weight_range == "mse":
        metadata["mse_search"] = SEARCH
        metadata["activation_search"] = activation_info
        metadata["weight_search"] = weight_info
    if bias_report is not None:
        metadata["bias_correction"] = bias_report
    if equalization is not None:
        metadata["equalization"] = equalization
    return B2Interpreter(result, quantizers, plan, activation_scales, audit=audit), metadata


def audit_summary(interpreter):
    """Per-node code occupancy after ``audit=True`` runs: entropy, saturation, zeros, SQNR."""
    rows = {}
    for name, entry in interpreter.audit.items():
        counts = entry["counts"]
        total = float(counts.sum())
        p = (counts / total).cpu().numpy()
        quantizer = interpreter.quantizers[interpreter.plan[name]["signedness"]]
        levels = quantizer.levels.cpu().numpy()
        nonzero = p[p > 0]
        order = np.sort(p)[::-1]
        rows[name] = {
            "elements": total,
            "entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
            "levels_used": int((p > 0).sum()), "levels_total": int(len(p)),
            "levels_for_99_percent": int(np.searchsorted(np.cumsum(order), 0.99) + 1),
            "fraction_zero": float(p[levels == 0].sum()),
            "fraction_at_extremes": float(p[0] + p[-1]),
            "fraction_abs_level_ge_1": float(p[np.abs(levels) >= 1].sum()),
            "sqnr_db": float(10 * np.log10(entry["signal"] / entry["noise"])) if entry["noise"] > 0 else None,
            "kind": interpreter.plan[name]["kind"],
        }
    return rows
