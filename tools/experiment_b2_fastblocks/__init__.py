"""Block-format bias correction with less GPU memory (lane S1, protocol speed-protocol-v1 part 2).

`bias_correct_blocks` is a drop-in replacement for `tools.experiment_b2.blocks.bias_correct_blocks` (read-only,
imported, never edited). The original keeps every chunk (8 images) of every live activation of both the FP32 and
the quantized graph on the GPU (256 bias-correction images: several GB for MobileNets). This version keeps those
chunks in host memory (pageable by default, pinned with pin=True) and moves one chunk at a time to the GPU, where
it runs the same torch operations on the same tensors (same shapes, strides and dtypes; the same per-chunk channel
sums added in the same order; the same delta, bias update, `t + delta` and store quantization). Each GPU
operation is deterministic for a given input (the B2 runs set deterministic algorithms), so the deltas, the bias
report and the corrected biases are bit-identical; only the order in which independent chunks are computed
differs. A use inside one process:

    import tools.experiment_b2.blocks as blocks
    from tools.experiment_b2_fastblocks import bias_correct_blocks
    blocks.bias_correct_blocks = bias_correct_blocks       # prepare_blocks looks the name up at call time

The configuration identity of a cell is unchanged (it hashes tools/experiment_b2 sources, which are not
edited); a record made this way states its execution path in a sidecar (see tools/run/speed_fastblocks.py).
"""
from __future__ import annotations
import numpy as np
import torch
from tools.experiment_b2.blocks import BIAS_CHUNK, _evaluate, quantize_axis

STATS = {'peak_gpu_bytes': 0}


def _host(t, pin):
    if not torch.is_tensor(t):
        return t
    out = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device='cpu', pin_memory=pin)
    out.copy_(t)
    return out


def _device(t, device):
    if not torch.is_tensor(t):
        return t
    out = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=device)
    out.copy_(t)
    return out


def bias_correct_blocks(fp_graph, q_graph, plan, quantizer, weights, inputs, chunk=BIAS_CHUNK, pin=False):
    """`blocks.bias_correct_blocks` with per-chunk activations in host memory, one chunk on the GPU at a time."""
    device = quantizer.levels.device
    nodes = list(q_graph.graph.nodes)
    fp_nodes = {n.name: n for n in fp_graph.graph.nodes}
    if [n.name for n in nodes] != list(fp_nodes):
        raise ValueError("bias correction needs identical FP32 and quantized topologies")
    remaining = {n.name: len(n.users) for n in nodes}
    env_fp, env_q, report = {}, {}, {}
    chunks = [torch.from_numpy(np.array(inputs[i:i + chunk], dtype=np.float32)) for i in range(0, len(inputs), chunk)]

    def store_one(name, t):           # t on the device
        if not plan[name]["quantizes"]:
            return t
        quantizer.context = None
        return quantize_axis(t, quantizer, 1)

    def run(graph, node, env, index, block):
        args = torch.fx.node.map_arg(node.args, lambda n: _device(env[n.name][index], device))
        kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: _device(env[n.name][index], device))
        quantizer.context = None
        result = _evaluate(graph, node, args, kwargs, quantizer, weights if block else None)
        quantizer.context = None
        return result

    for node in nodes:
        if node.op == "output":
            break
        if node.op == "placeholder":
            env_fp[node.name] = [_host(c, pin) for c in chunks]
            env_q[node.name] = [_host(store_one(node.name, _device(c, device)), pin) for c in chunks]
            continue
        mac = plan[node.name]["kind"] in ("conv", "linear")
        fp_out, q_out, sums_fp, sums_q = [], [], [], []
        dims = None; count = 0; ndim = None
        for i in range(len(chunks)):
            t = run(fp_graph, fp_nodes[node.name], env_fp, i, False)
            if mac:
                ndim = t.ndim; dims = [0] + list(range(2, t.ndim)); count += t.numel() // t.shape[1]
                sums_fp.append(t.double().sum(dim=dims))
            fp_out.append(_host(t, pin)); del t
        for i in range(len(chunks)):
            t = run(q_graph, node, env_q, i, True)
            if mac:
                sums_q.append(t.double().sum(dim=dims))
                q_out.append(_host(t, pin))
            else:
                q_out.append(_host(store_one(node.name, t), pin))
            del t
        if mac:
            mean_fp = sum(sums_fp) / count
            mean_q = sum(sums_q) / count
            delta = (mean_fp - mean_q).to(torch.float32)
            module = q_graph.get_submodule(node.target)
            if module.bias is None:
                raise ValueError("bias correction expects folded biases")
            module.bias.add_(delta)
            shape = (1, -1) + (1,) * (ndim - 2)
            q_out = [_host(store_one(node.name, _device(t, device) + delta.reshape(shape)), pin) for t in q_out]
            report[node.name] = {"max_abs_correction": float(delta.abs().max()),
                                 "rms_correction": float(delta.double().pow(2).mean().sqrt()),
                                 "rms_fp32_channel_mean": float(mean_fp.pow(2).mean().sqrt())}
        env_fp[node.name], env_q[node.name] = fp_out, q_out
        for source in node.all_input_nodes:
            remaining[source.name] -= 1
            if remaining[source.name] == 0:
                env_fp.pop(source.name, None)
                env_q.pop(source.name, None)
        if device.type == 'cuda':
            STATS['peak_gpu_bytes'] = max(STATS['peak_gpu_bytes'], torch.cuda.max_memory_allocated(device))
    return report
