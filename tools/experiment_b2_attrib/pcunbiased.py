"""Per-channel activation scales from an unbiased sample of the calibration activations (lane Q4 r6, addendum 8).

Replaces the subsample of ``pcfull.py`` (``pcf`` arms, addendum 2) for the new ``pcu`` arms.  The review of
2026-10-02 (finding B1) showed that ``pcfull.collect`` keeps every k-th value in (image, row, column) order with
``k = ceil(256 * H * W / 4096) = H * W / 16``; on 112x112 maps k = 784 = 7 * 112, so all 4096 search values of a
channel come from image column 0, on 56x56 maps from columns {0, 28}, on 28x28 maps from four columns.  The MSE
search then fitted border pixels, not the activation distribution.  ``pcf`` records stay as measured.

``pcu`` keeps everything of ``pcf`` except the sample of search values:

* anchor: the exact per-channel max |x| over all 256 images and all positions (unchanged);
* search values: a simple random sample without replacement of ``min(n, 4096)`` positions (image, row, column)
  out of the ``n = images * H * W`` positions of the node's FP32 output, the same positions for every channel of
  the node, drawn by ``torch.randperm`` on a CPU ``torch.Generator`` seeded with ``SEED + crc32(node name)`` and
  sorted; every position has the same inclusion probability, so the sample is unbiased for the per-channel value
  distribution;
* quantizer: the boundary's own B2 table and the frozen ``mse_search`` (unchanged).

Folding into conv/linear consumers and the bias-correction refit are the v1 arm engine (unchanged).
"""
from __future__ import annotations

import hashlib
import zlib

import numpy as np
import torch

from tools.experiment_b2.scales import mse_search

MAX_VALUES = 4096
CHUNK = 32
SEED = 20261002
SPEC = {"version": "pcu_v1", "images": "bias-correction inputs (first 256 of imagenet_calibration_2k)",
        "anchor": "exact per-channel max |x| of the FP32 activation", "max_values_per_channel": MAX_VALUES,
        "subsample": "simple random sample without replacement of min(n, max_values_per_channel) positions "
                     "(image, row, column) of the node output, n = images * rows * columns; the same positions for "
                     "every channel; torch.randperm(n, generator=torch.Generator().manual_seed(seed + "
                     "zlib.crc32(node name)))[:max_values_per_channel], sorted",
        "seed": SEED,
        "search": "tools/experiment_b2/scales.py mse_search with maxima = per-channel anchor",
        "supersedes": "pcf_v1 subsample (every k-th value; border columns only on 112x112, 56x56 and 28x28 maps)"}


def positions(node, n, m=MAX_VALUES, seed=SEED):
    """Sorted int64 positions in [0, n): all of them when ``n <= m``, else a seeded simple random sample of ``m``."""
    if n <= m:
        return torch.arange(n, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed + zlib.crc32(node.encode()))
    return torch.randperm(n, generator=generator)[:m].sort().values


def coverage(pos, per_image, shape):
    """Rows, columns and images the sampled positions touch (evidence against a border-biased sample)."""
    within = pos % per_image
    out = {"images_touched": int(torch.unique(pos // per_image).numel())}
    if len(shape) == 4:
        h, w = int(shape[2]), int(shape[3])
        out.update(rows=h, columns=w, distinct_rows=int(torch.unique(within // w).numel()),
                   distinct_columns=int(torch.unique(within % w).numel()))
    return out


def collect(fp_graph, nodes, inputs, device, m=MAX_VALUES, seed=SEED):
    """Per node: (per-channel max |x| float32 [C], per-channel FP32 values float32 [C, k], sample info) on the CPU.

    The values of one channel are indexed in (image, row, column) order over all inputs; the kept indices are
    ``positions(node, n)`` (``n`` known from the first chunk's shape and the number of inputs).
    """
    wanted = set(nodes)
    maxima, kept, offsets, chosen, info = {}, {n: [] for n in nodes}, {n: 0 for n in nodes}, {}, {}
    total = len(inputs)

    class Probe(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if node.name in wanted:
                t = result.detach()
                c = t.shape[1]
                flat = t.transpose(0, 1).reshape(c, -1).float()
                per_image = flat.shape[1] // t.shape[0]
                if node.name not in chosen:
                    n = total * per_image
                    chosen[node.name] = positions(node.name, n, m, seed)
                    info[node.name] = {"positions_total": int(n), "positions_kept": int(chosen[node.name].numel()),
                                       "positions_sha256": hashlib.sha256(chosen[node.name].numpy().tobytes()).hexdigest(),
                                       **coverage(chosen[node.name], per_image, tuple(t.shape))}
                start = offsets[node.name]
                pos = chosen[node.name]
                lo, hi = torch.searchsorted(pos, torch.tensor([start, start + flat.shape[1]]))
                local = (pos[int(lo):int(hi)] - start).to(flat.device)
                kept[node.name].append(flat[:, local].cpu())
                offsets[node.name] = start + flat.shape[1]
                top = flat.abs().amax(dim=1)
                maxima[node.name] = top if node.name not in maxima else torch.maximum(maxima[node.name], top)
            return result

    probe = Probe(fp_graph)
    with torch.inference_mode():
        for start in range(0, total, CHUNK):
            probe.run(torch.from_numpy(np.array(inputs[start:start + CHUNK], dtype=np.float32)).to(device))
    out = {}
    for name in nodes:
        values = torch.cat(kept[name], dim=1).numpy().astype(np.float32)
        if values.shape[1] != info[name]["positions_kept"]:
            raise RuntimeError(f"{name}: kept {values.shape[1]} values, expected {info[name]['positions_kept']}")
        out[name] = (maxima[name].cpu().numpy().astype(np.float32), values, info[name])
    return out


def channel_cache(shared, nodes, inputs):
    """``{node: (scales float32 [C], info)}`` in the format of ``engine.channel_params``."""
    stats = collect(shared.fp_graph, sorted(nodes), inputs, shared.device)
    cache = {}
    for node in sorted(nodes):
        maxima, values, sample = stats[node]
        quantizer = shared.quantizers[shared.plan[node]["signedness"]]
        with torch.inference_mode():
            found, info = mse_search(torch.as_tensor(values, device=shared.device), quantizer, maxima=maxima)
        scales = np.asarray([float(x) for x in found], dtype=np.float32)
        cache[node] = (scales, {"pcu": SPEC["version"], "channels": int(len(scales)), "values_per_channel": int(values.shape[1]),
                                "sample": sample,
                                "median_ratio_to_maxabs_scale": float(np.median(info["ratio_to_maxabs_scale"])),
                                "min_ratio_to_maxabs_scale": float(np.min(info["ratio_to_maxabs_scale"])),
                                "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
                                "scale_spread_log2": float(np.log2(scales.max() / scales.min())),
                                "per_tensor_scale": shared.activation_scales[node]})
    return cache
