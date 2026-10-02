"""Memory-lean B2 empirical bias correction: the same computation as ``tools.experiment_b2.engine.bias_correct``
(read-only, not edited), with the per-chunk activations of the 256 calibration images parked in host memory instead of
on the GPU.

Every tensor operation is the one B2 performs, on the same device, on the same 32-image chunks and in the same order
(node outputs per chunk, float64 per-channel sums added chunk by chunk, the shift and the activation quantizer per
chunk), so the corrected biases are bit-identical; only the storage between operations moves to the host.  GPU memory
then holds one chunk's working set (a few hundred MB for ResNet18) instead of every live activation of 256 images
for two networks (several GB).  Verified by re-running sealed arms (the B2 default must reproduce the sealed B2 top-5
lists, and sealed AXE/naive arms their per-image readouts).
"""
from __future__ import annotations

import numpy as np
import torch

SLICE = 8


def _values(graph, env, node, index, device):
    # copy=True: an in-place module (e.g. ReLU(inplace=True)) must not alter the parked host tensor; the device copy
    # it alters is the one B2 would alter, and B2 keeps no other reference to it (checked: rtn reproduces B2).
    args = torch.fx.node.map_arg(node.args, lambda n: env[n.name][index].to(device, copy=True))
    kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: env[n.name][index].to(device, copy=True))
    if node.op == "call_module":
        return graph.get_submodule(node.target)(*args, **kwargs)
    if node.op == "call_function":
        return node.target(*args, **kwargs)
    raise ValueError(f"unsupported node in staged execution: {node.op}")


def bias_correct_lean(fp_graph, q_graph, plan, quantizers, activation_scales, inputs, chunk=32):
    """Drop-in replacement of B2's ``bias_correct`` (same signature, same in-place effect and report)."""
    device = next(iter(quantizers.values())).levels.device
    host = torch.device("cpu")
    scales = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in activation_scales.items()}
    nodes = list(q_graph.graph.nodes)
    if [n.name for n in nodes] != [n.name for n in fp_graph.graph.nodes]:
        raise ValueError("bias correction needs identical FP32 and quantized topologies")
    fp_nodes = {n.name: n for n in fp_graph.graph.nodes}
    remaining = {n.name: len(n.users) for n in nodes}
    env_fp, env_q, report = {}, {}, {}
    chunks = [torch.from_numpy(np.array(inputs[i:i + chunk], dtype=np.float32)) for i in range(0, len(inputs), chunk)]

    def quantize(name, tensor):
        row = plan[name]
        if not row["quantizes"]:
            return tensor
        # the B quantizer is elementwise, so slices of SLICE images give the same values with a quarter of the
        # temporary memory (its int64 bucket indices are the largest allocation)
        quantizer = quantizers[row["signedness"]]
        return torch.cat([quantizer(part, scales[name]) for part in torch.split(tensor, SLICE)])

    for node in nodes:
        if node.op == "output":
            break
        if node.op == "placeholder":
            env_fp[node.name] = chunks
            env_q[node.name] = [quantize(node.name, c.to(device)).to(host) for c in chunks]
            continue
        mac = plan[node.name]["kind"] in ("conv", "linear")
        fp_out, q_raw = [], []
        sum_fp = sum_q = 0
        count = 0
        for i in range(len(chunks)):
            f = _values(fp_graph, env_fp, fp_nodes[node.name], i, device)
            q = _values(q_graph, env_q, node, i, device)
            if mac:
                dims = [0] + list(range(2, f.ndim))
                count += f.numel() // f.shape[1]
                sum_fp = sum_fp + f.double().sum(dim=dims)
                sum_q = sum_q + q.double().sum(dim=dims)
                fp_out.append(f.to(host))
                q_raw.append(q.to(host))
            else:
                fp_out.append(f.to(host))
                q_raw.append(quantize(node.name, q).to(host))
            del f, q
        if mac:
            mean_fp, mean_q = sum_fp / count, sum_q / count
            delta = (mean_fp - mean_q).to(torch.float32)
            module = q_graph.get_submodule(node.target)
            if module.bias is None:
                raise ValueError("bias correction expects folded biases")
            module.bias.add_(delta)
            shape = (1, -1) + (1,) * (fp_out[0].ndim - 2)
            q_raw = [quantize(node.name, t.to(device) + delta.reshape(shape)).to(host) for t in q_raw]
            report[node.name] = {"max_abs_correction": float(delta.abs().max()),
                                 "rms_correction": float(delta.double().pow(2).mean().sqrt()),
                                 "rms_fp32_channel_mean": float(mean_fp.pow(2).mean().sqrt())}
        env_fp[node.name] = fp_out
        env_q[node.name] = q_raw
        for source in node.all_input_nodes:
            remaining[source.name] -= 1
            if remaining[source.name] == 0:
                env_fp.pop(source.name, None)
                env_q.pop(source.name, None)
    return report
