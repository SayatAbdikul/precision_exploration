"""AdaRound (Nagel et al., ICML 2020, arXiv 2004.10568) for one layer, over any sorted codebook.

Equation numbers refer to the paper.  One continuous variable ``V`` per weight
decides whether the weight goes to the lower or the upper of its two
neighbouring codebook levels:

    W_soft = s * (lo + h(V) * (hi - lo))                          (eq. 22)
    h(V)   = clip(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)      (eq. 23)
    f_reg  = sum 1 - |2 h(V) - 1| ** beta                         (eq. 24)
    loss   = || f_a(W x + b) - f_a(W_soft x_hat + b) ||^2 + lambda * f_reg   (eq. 25)

For a uniform integer codebook ``lo = clip(floor(W/s), n, p)`` and
``hi = clip(floor(W/s) + 1, n, p)``, which is eq. 22 including its clip.  For
a non-uniform codebook (minifloat, posit) the same variable interpolates
between two neighbouring levels of unequal distance; that is an adaptation
and not part of the published method.

Convergence of ``h(V)`` to exactly 0 or 1 is not guaranteed by the method.
For one soft variable the reconstruction term is a convex quadratic with
curvature about ``s^2 E[x^2]`` and the regulariser at ``beta = 2`` a concave
one with curvature ``4 lambda``; the variable is forced to an end only when
the second is larger.  ``learn_rounding`` therefore reports how many soft
variables are not binary after the last iteration; the hard decision
(``V >= 0``) is taken regardless.

Nothing here reads data from disk or knows about models; ``fit.py`` does.
"""
from __future__ import annotations

import math

import torch

ZETA, GAMMA = 1.1, -0.1
SETTINGS = {
    "version": "b2-recon-adaround-1",
    "zeta": ZETA, "gamma": GAMMA, "lambda": 0.01, "beta_start": 20.0, "beta_end": 2.0, "warm_start": 0.2,
    "beta_schedule": "cosine", "iterations": 10000, "batch_size": 32, "optimizer": "Adam", "learning_rate": 1e-3,
    "loss": "squared error summed over output channels, mean over batch and spatial positions; f_reg summed",
    "hard_decision": "V >= 0 (h >= 0.5) rounds up", "fallback_to_nearest": False,
}


def rectified_sigmoid(v):
    """Eq. 23: reaches exactly 0 and 1 for finite ``v`` and keeps a gradient in between."""
    return torch.clamp(torch.sigmoid(v) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


def regulariser(h, beta):
    """Eq. 24: zero exactly when every ``h`` is 0 or 1, largest (one per weight) at ``h = 0.5``."""
    return (1.0 - (2.0 * h - 1.0).abs().pow(beta)).sum()


def beta_at(step, iterations, settings=SETTINGS):
    """Annealed exponent for 0-based ``step``; ``None`` during the warm start (no regulariser)."""
    warm = settings["warm_start"] * iterations
    if step < warm:
        return None
    fraction = (step - warm) / max(1.0, iterations - warm)
    start, end = settings["beta_start"], settings["beta_end"]
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * fraction))


def neighbours(weight, scale, levels):
    """The two neighbouring levels of every ``weight / scale`` and its position between them.

    Returns ``(lo, hi, fraction)``, FP32 tensors of the weight's shape, with
    ``lo <= weight/scale <= hi`` whenever the value is inside the codebook
    range; outside it ``lo == hi ==`` the end level (the weight is clipped and
    has no rounding choice) and ``fraction`` is 0.
    """
    if levels.ndim != 1 or not bool((levels[1:] > levels[:-1]).all()):
        raise ValueError("the codebook levels must be strictly ascending")
    y = weight / scale
    last = len(levels) - 1
    index = torch.bucketize(y.contiguous(), levels, right=True) - 1  # largest level <= y
    below = index < 0
    lower = index.clamp(0, last)
    upper = torch.where(below, lower, (lower + 1).clamp_max(last))
    lo, hi = levels[lower], levels[upper]
    gap = hi - lo
    fraction = torch.where(gap > 0, (y - lo) / torch.where(gap > 0, gap, torch.ones_like(gap)), torch.zeros_like(y))
    return lo, hi, fraction.clamp(0.0, 1.0)


def initial_v(fraction):
    """``V`` with ``h(V) == fraction``: the soft weight starts at the FP32 weight (paper, Figure 3)."""
    return torch.log((fraction - GAMMA) / (ZETA - fraction))


def soft_weight(lo, hi, v, scale):
    return scale * (lo + rectified_sigmoid(v) * (hi - lo))


def hard_up(v):
    """Final decision: up when ``h(V) >= 0.5``, i.e. ``V >= 0``."""
    return v >= 0


def hard_weight(lo, hi, up, scale):
    """The deployed weight: a codebook level times the scale, the same FP32 product B2 forms."""
    return torch.where(up, hi, lo) * scale


def nearest_up(lo, hi, weight, scale, quantizer):
    """Up/down decisions of the B2 round-to-nearest quantizer on the same grid (ties as B2 breaks them)."""
    nearest = quantizer(weight, scale)
    return (nearest != lo * scale) & (hi > lo), nearest


def reconstruction_loss(output, target):
    """Squared error summed over the channel dimension, mean over everything else."""
    return (output - target).pow(2).sum(dim=1).mean()


def learn_rounding(weight, scale, levels, forward, activation, x_hat, target_of, *, seed, settings=SETTINGS,
                   device=None, trace=None):
    """Learn the rounding of one layer.

    ``weight``: FP32 weight; ``scale``: broadcastable FP32 scale; ``levels``: codebook.
    ``forward(w, x)``: the layer with weight ``w`` (bias included).
    ``activation``: ``f_a``.  ``x_hat``: ``[N, ...]`` inputs of the layer in the
    partly quantized network.  ``target_of(index)``: FP32 targets
    ``f_a(W x + b)`` for a LongTensor of image indices.  Data may live on the
    CPU; every mini-batch is moved to ``device``.  Returns ``(up, statistics)``.
    """
    device = device or weight.device
    iterations, batch = int(settings["iterations"]), int(settings["batch_size"])
    lo, hi, fraction = neighbours(weight, scale, levels)
    v = initial_v(fraction).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v], lr=settings["learning_rate"])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    count = x_hat.shape[0]
    order, cursor = torch.randperm(count, generator=generator), 0
    history = []
    for step in range(iterations):
        if cursor + batch > count:
            order, cursor = torch.randperm(count, generator=generator), 0
        index = order[cursor:cursor + batch]
        cursor += batch
        where = index.to(x_hat.device)
        inputs = x_hat.index_select(0, where).to(device, non_blocking=True)
        target = target_of(index)
        h = rectified_sigmoid(v)
        output = activation(forward(scale * (lo + h * (hi - lo)), inputs))
        loss = reconstruction = reconstruction_loss(output, target)
        beta = beta_at(step, iterations, settings)
        if beta is not None:
            loss = reconstruction + settings["lambda"] * regulariser(h, beta)
        if not torch.isfinite(loss):
            raise ValueError("AdaRound loss is not finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if trace is not None and (step % trace == 0 or step == iterations - 1):
            history.append([step, float(reconstruction.detach()), float(loss.detach())])
    with torch.no_grad():
        h = rectified_sigmoid(v)
        up = hard_up(v) & (hi > lo)
        free = hi > lo
        unsettled = ((h > 0) & (h < 1) & free).sum()
        statistics = {"weights": int(weight.numel()), "free_weights": int(free.sum()),
                      "clipped_weights": int((~free).sum()), "unsettled_soft_variables": int(unsettled),
                      "far_from_binary_soft_variables": int(((h > 0.01) & (h < 0.99) & free).sum()),
                      "rounded_up": int(up.sum()), "trace_step_reconstruction_total": history}
    return up, statistics
