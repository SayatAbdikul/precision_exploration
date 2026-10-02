"""Integer-domain overflow checks and the engine's closed-form certificate for given weight integers.

``prefix_stats``: on real images, the input codes of every MAC node are recovered from the B2 simulator's stored
activations, and for every output element (i) the order-free extremes sum max(0, a q) and sum min(0, a q) (exact,
float64 on integers; they bound every partial sum in every order) and (ii) every literal prefix of the sequential
sum in PyTorch tap order (input channel, kernel row, kernel column), in int32, are reduced to per-node extremes.
A P-bit register never overflows on these images iff every extreme lies in [-2^(P-1), 2^(P-1)-1].
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from tools.experiment_b2.engine import B2Interpreter

from .build import input_store


def signed_bits(lo, hi):
    """Smallest two's-complement width holding [lo, hi] (same convention as the engine's certificates)."""
    lo, hi = int(min(lo, 0)), int(max(hi, 0))
    return max(hi.bit_length(), (-lo - 1).bit_length() if lo < 0 else 0) + 1


class _Checker(B2Interpreter):
    def __init__(self, setup, graph, codes, chunk_elements=40_000_000):
        super().__init__(graph, setup.quantizers, setup.plan, setup.act_scales)
        self.setup, self.codes, self.chunk = setup, codes, chunk_elements
        self.stats = {}

    def run_node(self, node):
        if node.op == "call_module" and node.name in self.codes:
            args, _ = self.fetch_args_kwargs_from_env(node)
            self._check(node, self.module_of(node), args[0])
        return super().run_node(node)

    def module_of(self, node):
        return self.module.get_submodule(node.target)

    def _update(self, name, key, value, fn):
        entry = self.stats.setdefault(name, {})
        entry[key] = value if key not in entry else fn(entry[key], value)

    def _check(self, node, module, value):
        a = self.setup.input_codes(node, value)  # float32 integers
        q = self.codes[node.name].to(a.device)
        if isinstance(module, nn.Conv2d):
            w = q.reshape(module.weight.shape).double()
            kw = dict(stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
            ad = a.double()
            ap, an = ad.clamp(min=0), ad.clamp(max=0)
            wp, wn = w.clamp(min=0), w.clamp(max=0)
            pos = nn.functional.conv2d(ap, wp, **kw) + nn.functional.conv2d(an, wn, **kw)
            neg = nn.functional.conv2d(ap, wn, **kw) + nn.functional.conv2d(an, wp, **kw)
            cols = nn.functional.unfold(a, module.kernel_size, dilation=module.dilation, padding=module.padding,
                                        stride=module.stride).round().to(torch.int32)  # [B, K, L]
        else:
            ad = a.reshape(a.shape[0], -1).double()
            w = q.double()
            pos = ad.clamp(min=0) @ w.clamp(min=0).t() + ad.clamp(max=0) @ w.clamp(max=0).t()
            neg = ad.clamp(min=0) @ w.clamp(max=0).t() + ad.clamp(max=0) @ w.clamp(min=0).t()
            cols = a.reshape(a.shape[0], -1, 1).round().to(torch.int32)  # [B, K, 1]
        self._update(node.name, "orderfree_hi", int(pos.max()), max)
        self._update(node.name, "orderfree_lo", int(neg.min()), min)
        qi = q.reshape(q.shape[0], -1).to(torch.int32)
        C, K = qi.shape
        L = cols.shape[2]
        step = max(1, self.chunk // (K * L))
        hi, lo, total_hi, total_lo = -2**62, 2**62, -2**62, 2**62
        for b in range(cols.shape[0]):
            A = cols[b]  # [K, L]
            for c0 in range(0, C, step):
                prod = qi[c0:c0 + step, :, None] * A[None, :, :]  # [c, K, L]
                pre = torch.cumsum(prod, dim=1, dtype=torch.int32)
                hi, lo = max(hi, int(pre.max())), min(lo, int(pre.min()))
                total_hi, total_lo = max(total_hi, int(pre[:, -1].max())), min(total_lo, int(pre[:, -1].min()))
                del prod, pre
        self._update(node.name, "prefix_hi", hi, max)
        self._update(node.name, "prefix_lo", lo, min)
        self._update(node.name, "total_hi", total_hi, max)
        self._update(node.name, "total_lo", total_lo, min)


def prefix_stats(setup, result, codes, images, batch=8):
    """Per-node extremes on the first ``images`` screen images of the deployed network ``result``."""
    images = min(images, len(setup.inputs))
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)  # integer cumsum is exact; the flag only rejects the CUDA kernel
    try:
        checker = _Checker(setup, result, codes)
        with torch.inference_mode():
            for start in range(0, images, batch):
                tensor = torch.from_numpy(np.array(setup.inputs[start:min(start + batch, images)], dtype=np.float32))
                checker.run(tensor.to(setup.device))
    finally:
        torch.use_deterministic_algorithms(deterministic)
    stats = checker.stats
    for entry in stats.values():
        if entry["orderfree_hi"] > 2**31 - 1 or entry["orderfree_lo"] < -2**31:
            raise ValueError("int32 prefix arithmetic would not be exact")
        entry["prefix_bits"] = signed_bits(entry["prefix_lo"], entry["prefix_hi"])
        entry["orderfree_bits"] = signed_bits(entry["orderfree_lo"], entry["orderfree_hi"])
    return {"images": images, "nodes": stats,
            "prefix_bits": max(e["prefix_bits"] for e in stats.values()),
            "orderfree_bits": max(e["orderfree_bits"] for e in stats.values())}


def engine_certificate(case_l8, codes):
    """The exact engine's certificate (read-only import) on the adapted B2 export with the weights replaced."""
    from pathlib import Path
    from tools.experiment_b.common import ROOT
    from tools.scaled_bridge_v2.certificates import certify
    from tools.scaled_bridge_v2.export import load_export
    path = Path(ROOT) / "artifacts/scaled_bridge_v2/exports/9ecfd61292dc72b1c37e8955b29e054ea9b056c52538d562a3785d407d8e135f" \
        / case_l8 / "export.json"
    export, arrays, _ = load_export(case_l8, path)
    export, arrays = copy.deepcopy(export), dict(arrays)
    original = {}
    for node in export["nodes"]:
        if node["op"] in ("conv", "linear"):
            key = node["mac"]["weight_units"]
            original[node["name"]] = arrays[key]
            if codes is not None:
                q = codes[node["name"]].cpu().numpy().astype(np.int64).reshape(arrays[key].shape)
                arrays[key] = q
    cert = certify(export, arrays)
    layers = {k: {f: v[f] for f in ("K", "signed_bits_absolute", "signed_bits_range", "signed_bits_structural")}
              for k, v in cert.items()}
    return {"layers": layers,
            "signed_bits_absolute": max(v["signed_bits_absolute"] for v in layers.values()),
            "signed_bits_range": max(v["signed_bits_range"] for v in layers.values()),
            "signed_bits_structural": max(v["signed_bits_structural"] for v in layers.values())}, original
