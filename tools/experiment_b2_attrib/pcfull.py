"""Per-channel activation scales from full calibration activations (lane Q4 r3, protocol addendum 2).

The v1 per-channel arms (``pc:``) fit one scale per channel on the v1 calibration samples regrouped by channel:
256 samples per image spread over all channels leave 400-2000 samples per channel, and the MSE search is
anchored at that sparse sample maximum.  On MobileNetV3 INT8 those scales clip (``features_16_2`` loses 9.4 dB of
SQNR on the screen audit) and every ``pc:`` arm is worse than the default.

``pcf:`` arms fit the same per-channel MSE search on the FP32 activations of the 256 bias-correction images
(first 256 images of the frozen calibration list, the cached inputs that the bias correction already uses):

* anchor: the exact per-channel max |x| over all 256 images and all positions (as the per-tensor default anchors
  its search at the observed tensor maximum);
* search values: per channel, the values in (image, row, column) order, every k-th with
  ``k = ceil(n / MAX_VALUES)`` so that at most ``MAX_VALUES`` values per channel enter the search;
* quantizer: the boundary's own B2 table (signed or unsigned) and the frozen ``mse_search``.

Everything else of the arm (folding into conv/linear consumers, bias correction refit) is the v1 arm engine.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from tools.experiment_b2.scales import mse_search

MAX_VALUES = 4096
CHUNK = 32
SPEC = {"version": "pcf_v1", "images": "bias-correction inputs (first 256 of imagenet_calibration_2k)",
        "anchor": "exact per-channel max |x| of the FP32 activation", "max_values_per_channel": MAX_VALUES,
        "subsample": "every k-th value in (image, row, column) order, k = ceil(n / max_values_per_channel)",
        "search": "tools/experiment_b2/scales.py mse_search with maxima = per-channel anchor"}


def collect(fp_graph, nodes, inputs, device):
    """Per node: (per-channel max |x| float32 [C], per-channel FP32 values float32 [C, m]) on the CPU.

    Values of one channel are indexed in (image, row, column) order over all inputs; index i is kept when
    ``i % k == 0`` with ``k = ceil(n / MAX_VALUES)`` (n = images * rows * columns).
    """
    wanted = set(nodes)
    maxima, kept, offsets = {}, {n: [] for n in nodes}, {n: 0 for n in nodes}
    total = len(inputs)

    class Probe(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if node.name in wanted:
                t = result.detach()
                c = t.shape[1]
                flat = t.transpose(0, 1).reshape(c, -1).float()
                per_image = flat.shape[1] // t.shape[0]
                k = max(1, math.ceil(total * per_image / MAX_VALUES))
                start = offsets[node.name]
                first = (-start) % k
                kept[node.name].append(flat[:, first::k].cpu())
                offsets[node.name] = start + flat.shape[1]
                top = flat.abs().amax(dim=1)
                maxima[node.name] = top if node.name not in maxima else torch.maximum(maxima[node.name], top)
            return result

    probe = Probe(fp_graph)
    with torch.inference_mode():
        for start in range(0, total, CHUNK):
            probe.run(torch.from_numpy(np.array(inputs[start:start + CHUNK], dtype=np.float32)).to(device))
    return {name: (maxima[name].cpu().numpy().astype(np.float32), torch.cat(kept[name], dim=1).numpy().astype(np.float32))
            for name in nodes}


def channel_cache(shared, nodes, inputs):
    """``{node: (scales float32 [C], info)}`` in the format of ``engine.channel_params``."""
    stats = collect(shared.fp_graph, sorted(nodes), inputs, shared.device)
    cache = {}
    for node in sorted(nodes):
        maxima, values = stats[node]
        quantizer = shared.quantizers[shared.plan[node]["signedness"]]
        with torch.inference_mode():
            found, info = mse_search(torch.as_tensor(values, device=shared.device), quantizer, maxima=maxima)
        scales = np.asarray([float(x) for x in found], dtype=np.float32)
        cache[node] = (scales, {"pcf": SPEC["version"], "channels": int(len(scales)), "values_per_channel": int(values.shape[1]),
                                "median_ratio_to_maxabs_scale": float(np.median(info["ratio_to_maxabs_scale"])),
                                "min_ratio_to_maxabs_scale": float(np.min(info["ratio_to_maxabs_scale"])),
                                "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
                                "scale_spread_log2": float(np.log2(scales.max() / scales.min())),
                                "per_tensor_scale": shared.activation_scales[node]})
    return cache
