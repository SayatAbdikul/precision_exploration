"""Sequential bias correction with labelled variants and per-layer instrumentation (lane Q3).

The reference is ``tools.experiment_b2.engine.bias_correct`` (read-only).  Policy
``global`` repeats its arithmetic operation by operation (same means, same order,
same shift of the stored quantized outputs), so its corrected biases are bit-identical
to the reference; the other policies change exactly one thing:

* ``executed``          the corrected layer is recomputed with its new bias instead of shifted;
* ``dfq_weights_only``  the correction pass runs with every activation boundary unquantized
                        (DFQ Appendix D: "run on a network with quantized weights only");
* ``cap_rms``           the correction vector is scaled so that its RMS is at most the RMS of
                        the FP32 channel means of that layer;
* ``linear_only``       only layers whose output does not feed a nonlinearity are corrected
                        (block outputs, projections, downsample, classifier);
* ``local_empirical``   the own-error part only: mean of (W - W_q) x_q on the quantized
                        network's (corrected) input;
* ``analytic_fp32``     the own-error part on the FP32 network's input: mean of (W - W_q) x_fp
                        (DFQ section 4.2 with the empirical FP32 input instead of batch-norm
                        statistics; evaluated as a convolution, so padding is exact);
* ``none``              nothing is applied (instrumentation of the uncorrected network).

Instrumentation (all on the correction images, nothing from the screen): per conv/linear
layer the applied and the global correction, the local / inherited split of the global
correction (global = local + inherited, inherited = mean of W (x_fp - x_q)), the RMS of the
FP32 channel means, the fraction of always-off channels in front of a rectifying consumer
(positive on fewer than 1 percent of positions) in FP32 and in the quantized network after
this layer's correction, and the image-to-image spread (std over images of the per-image
channel mean, averaged over channels) in both networks.

Storage: with ``store="cpu"`` the per-node activations of all correction images are kept in
host memory and every operation still runs on the device on the same values, so the
results do not depend on ``store`` (checked by the unit tests).
"""
from __future__ import annotations

import math

import numpy as np
import torch

POLICIES = ("global", "executed", "dfq_weights_only", "cap_rms", "linear_only", "local_empirical", "analytic_fp32",
            "none")
OFF_THRESHOLD = {"relu": 0.0, "relu6": 0.0, "hardswish": -3.0, "hardsigmoid": -3.0}
OFF_FRACTION = 0.01


def consumer_labels(graph):
    """``{conv/linear node name: label of its only consumer nonlinearity or 'identity'}``."""
    from tools.experiment_b2_recon.engine import consumer_activation, weight_layers
    return {node.name: consumer_activation(graph, node)[0] for node, _ in weight_layers(graph)}


def _values(graph, node, env, index, device):
    def fetch(n):
        value = env[n.name][index]
        return value.to(device, non_blocking=False) if value.device != device else value
    args = torch.fx.node.map_arg(node.args, fetch)
    kwargs = torch.fx.node.map_arg(node.kwargs, fetch)
    if node.op == "call_module":
        return graph.get_submodule(node.target)(*args, **kwargs)
    if node.op == "call_function":
        return node.target(*args, **kwargs)
    raise ValueError(f"unsupported node in staged execution: {node.op}")


def _channel_stats(tensors):
    """Per-image channel means ``[images, channels]`` (float64, CPU)."""
    dims = list(range(2, tensors[0].ndim))
    return torch.cat([(t.double().mean(dim=dims) if dims else t.double()).cpu() for t in tensors])


def _spread(per_image):
    return float(per_image.std(dim=0).mean()) if per_image.shape[0] > 1 else float("nan")


def _rms(vector):
    return float(vector.double().pow(2).mean().sqrt())


def staged_correct(fp_graph, q_graph, plan, quantizers, activation_scales, inputs, *, policy="global", chunk=32,
                   store="cpu", instrument=True):
    """Apply one correction policy in place on ``q_graph``; returns the per-layer report."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    device = next(iter(quantizers.values())).levels.device
    scales = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in activation_scales.items()}
    nodes = list(q_graph.graph.nodes)
    if [n.name for n in nodes] != [n.name for n in fp_graph.graph.nodes]:
        raise ValueError("bias correction needs identical FP32 and quantized topologies")
    targets = [n.target for n in nodes if n.op == "call_module" and plan.get(n.name, {}).get("kind") in ("conv", "linear")]
    if len(targets) != len(set(targets)):
        raise ValueError("a conv/linear module is called by more than one node: a correction would be applied twice")
    fp_nodes = {n.name: n for n in fp_graph.graph.nodes}
    consumers = consumer_labels(fp_graph)
    remaining = {n.name: len(n.users) for n in nodes}
    env_fp, env_q, report = {}, {}, {}
    keep = (lambda t: t.cpu()) if store == "cpu" else (lambda t: t)
    chunks = [torch.from_numpy(np.array(inputs[i:i + chunk], dtype=np.float32)) for i in range(0, len(inputs), chunk)]
    quantize_activations = policy != "dfq_weights_only"

    def quantize(name, tensors):
        row = plan[name]
        if not row["quantizes"] or not quantize_activations:
            return tensors
        return [quantizers[row["signedness"]](t, scales[name]) for t in tensors]

    for node in nodes:
        if node.op == "output":
            break
        if node.op == "placeholder":
            on_device = [c.to(device) for c in chunks]
            env_fp[node.name] = [keep(c) for c in on_device]
            env_q[node.name] = [keep(t) for t in quantize(node.name, on_device)]
            continue
        n = len(chunks)
        is_layer = plan[node.name]["kind"] in ("conv", "linear")
        if not is_layer:
            env_fp[node.name] = [keep(_values(fp_graph, fp_nodes[node.name], env_fp, i, device)) for i in range(n)]
            env_q[node.name] = [keep(quantize(node.name, [_values(q_graph, node, env_q, i, device)])[0])
                                for i in range(n)]
        else:
            # one chunk at a time on the device; sums in the same order as the reference
            module = q_graph.get_submodule(node.target)
            fp_module = fp_graph.get_submodule(node.target)
            if module.bias is None:
                raise ValueError("bias correction expects folded biases")
            want_parts = instrument or policy in ("local_empirical", "analytic_fp32")
            bias_offset = (fp_module.bias.double() - module.bias.double())
            fp_out, q_out, sum_fp, sum_q, sum_local, sum_analytic, count, ndim = [], [], 0, 0, 0, 0, 0, None
            for i in range(n):
                f = _values(fp_graph, fp_nodes[node.name], env_fp, i, device)
                q = _values(q_graph, node, env_q, i, device)
                ndim = f.ndim
                dims = [0] + list(range(2, ndim))
                count += f.numel() // f.shape[1]
                sum_fp = sum_fp + f.double().sum(dim=dims)
                sum_q = sum_q + q.double().sum(dim=dims)
                if want_parts:
                    # own error on the quantized input: f_fp(x_q) - f_q(x_q) = (W - W_q) x_q (+ bias difference)
                    sum_local = sum_local + (_values(fp_graph, fp_nodes[node.name], env_q, i, device).double()
                                             - q.double()).sum(dim=dims)
                    sum_analytic = sum_analytic + (f.double() - _values(q_graph, node, env_fp, i, device).double()
                                                   ).sum(dim=dims)
                fp_out.append(keep(f))
                q_out.append(keep(q))
                del f, q
            mean_fp = sum_fp / count
            mean_q = sum_q / count
            delta = (mean_fp - mean_q).to(torch.float32)
            row = {"consumer": consumers.get(node.name, "identity")}
            local = sum_local / count - bias_offset if want_parts else None
            analytic = sum_analytic / count - bias_offset if want_parts else None
            if policy in ("global", "executed", "dfq_weights_only"):
                applied = delta
            elif policy == "cap_rms":
                ratio = _rms(mean_fp) / max(_rms(delta), 1e-30)
                applied = (delta.double() * min(1.0, ratio)).to(torch.float32)
            elif policy == "linear_only":
                applied = delta if row["consumer"] == "identity" else torch.zeros_like(delta)
            elif policy == "local_empirical":
                applied = local.to(torch.float32)
            elif policy == "analytic_fp32":
                applied = analytic.to(torch.float32)
            else:
                applied = torch.zeros_like(delta)
            module.bias.add_(applied)
            shape = (1, -1) + (1,) * (ndim - 2)
            threshold = OFF_THRESHOLD.get(row["consumer"])
            stored, q_images, fp_images = [], [], []
            above_fp = above_q = 0
            for i in range(n):
                if policy == "executed":
                    q = _values(q_graph, node, env_q, i, device)
                else:
                    q = q_out[i].to(device) + applied.reshape(shape)
                if instrument:
                    f = fp_out[i].to(device)
                    fp_images.append(_channel_stats([f]))
                    q_images.append(_channel_stats([q]))
                    if threshold is not None:
                        above_fp = above_fp + (f > threshold).double().sum(dim=dims)
                        above_q = above_q + (q > threshold).double().sum(dim=dims)
                    del f
                stored.append(keep(quantize(node.name, [q])[0]))
                del q
            row.update({"rms_applied": _rms(applied), "rms_global": _rms(delta), "rms_fp32_channel_mean": _rms(mean_fp),
                        "max_abs_applied": float(applied.abs().max()), "mean_applied": float(applied.double().mean())})
            if instrument:
                inherited = delta.double() - local
                fp_images, q_images = torch.cat(fp_images), torch.cat(q_images)
                row.update({
                    "rms_local": _rms(local), "rms_inherited": _rms(inherited),
                    "rms_analytic_fp32": _rms(analytic),
                    "mean_inherited": float(inherited.mean()), "mean_local": float(local.mean()),
                    "spread_fp32": _spread(fp_images), "spread_q": _spread(q_images),
                    "correlation_per_image_means": float(np.corrcoef(fp_images.flatten().numpy(),
                                                                    q_images.flatten().numpy())[0, 1])
                    if q_images.std() > 0 else float("nan"),
                    "off_fraction_fp32": float(((above_fp / count) < OFF_FRACTION).double().mean())
                    if threshold is not None else None,
                    "off_fraction_q": float(((above_q / count) < OFF_FRACTION).double().mean())
                    if threshold is not None else None})
            report[node.name] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}
            env_fp[node.name] = fp_out
            env_q[node.name] = stored
            del q_out
        for source in node.all_input_nodes:
            remaining[source.name] -= 1
            if remaining[source.name] == 0:
                env_fp.pop(source.name, None)
                env_q.pop(source.name, None)
    return report
