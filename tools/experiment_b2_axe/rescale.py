"""Addendum-2 arms (exploratory, post hoc): an accumulator-aware per-channel weight scale.

``rescale-P<P>``: for every output channel, the smallest factor k >= 1 such that nearest rounding of the weights on
the coarser grid s_c * k meets the v1 overflow bound; codes round((w/s_c)/k).  ``axers-P<P>``: the v1 AXE-OPTQ
(unchanged) on that coarser grid.  Both meet exactly the v1 guarantee (the bound is on integer codes; the scale does
not enter it).  Protocol: public/experiments/configs/breadth-study/b2-axe-protocol-v1-addendum-2.json.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from . import axe
from .build import Setup

ITERATIONS = 60


def _codes(W, k, qmin, qmax):
    return torch.round(W / k[:, None]).clamp(qmin, qmax)


def _feasible(q, limit):
    return (torch.clamp(q, min=0).sum(1) <= limit) & ((-torch.clamp(q, max=0)).sum(1) <= limit)


def rescale_factors(W, *, qmin, qmax, P, N, iterations=ITERATIONS):
    """Per row of ``W`` [C, K] (code units of the base scale): the smallest k >= 1 (bisection) whose nearest codes
    meet the bound.  Feasibility is monotone in k (|round(x/k)| is non-increasing in k; clipping is monotone), so
    bisection on [1, 2 max|W| + 1] (all codes 0 at the upper end) finds it to ``2^-iterations`` of the interval.

    Returns ``(k float64 [C], Q int64 [C, K])``; rows that meet the bound at k = 1 keep k = 1 exactly.
    """
    W = W.detach().to(torch.float64).cpu()
    limit = axe.integer_limit(P, N)
    C = W.shape[0]
    one = torch.ones(C, dtype=torch.float64)
    at_one = _feasible(_codes(W, one, qmin, qmax), limit)
    lo = one.clone()
    hi = 2 * W.abs().amax(dim=1) + 2
    if not bool(_feasible(_codes(W, hi, qmin, qmax), limit).all()):
        raise AssertionError("upper bisection end is not feasible")
    for _ in range(iterations):
        mid = (lo + hi) / 2
        ok = _feasible(_codes(W, mid, qmin, qmax), limit)
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    k = torch.where(at_one, one, hi)
    Q = _codes(W, k, qmin, qmax)
    if not bool(_feasible(Q, limit).all()):
        raise AssertionError("rescaled codes violate the bound")
    return k, Q.to(torch.int64)


def factors_digest(factors):
    h = hashlib.sha256()
    for name in sorted(factors):
        h.update(name.encode())
        h.update(np.ascontiguousarray(factors[name].cpu().numpy().astype("<f8")).tobytes())
    return h.hexdigest()


def factor_summary(k):
    k = k.to(torch.float64)
    return {"max_k": float(k.max()), "fraction_rescaled": float((k > 1).double().mean()),
            "mean_log2_k": float(torch.log2(k).mean()), "channels": int(k.numel())}


class ScaledSetup(Setup):
    """``Setup`` whose per-channel weight scales can be multiplied by factors k (one arm at a time)."""

    def __init__(self, case, device):
        super().__init__(case, device)
        self.base_shaped = dict(self.shaped)
        self.base_real = dict(self.real)

    def reset(self):
        self.shaped, self.real = dict(self.base_shaped), dict(self.base_real)

    def use_factors(self, factors):
        """Scales s_c * k_c (float32) and weights (w/s_c)/k_c in the new code units, for every MAC node."""
        self.reset()
        for name, k in factors.items():
            base = self.base_shaped[name]
            shape = (-1,) + (1,) * (base.dim() - 1)
            self.shaped[name] = (base.double() * k.to(base.device).reshape(shape)).to(torch.float32)
            self.real[name] = self.base_real[name] / k.to(self.base_real[name].device)[:, None]

    def rescale(self, P):
        """Factors and codes of the ``rescale-P`` arm for every MAC node (base scales, nothing changed)."""
        N = self.spec["N"]
        factors, codes = {}, {}
        for node, _ in self.layers:
            k, q = rescale_factors(self.base_real[node.name], qmin=self.qmin, qmax=self.qmax, P=P, N=N)
            factors[node.name], codes[node.name] = k, q.to(self.base_real[node.name].device)
        return factors, codes
