"""Activation codes beyond the B2 default: affine (zero point) and per-channel scales.

Both quantizers wrap a frozen B2 ``TableQuantizer`` (same levels, midpoints and tie rule) and are called with
the signature the B2 engine uses, ``quantizer(values, scale)``.  The engine passes the node's reference scale;
each quantizer checks it and then applies its own parameters.  This keeps ``B2Interpreter`` and
``bias_correct`` (frozen) usable unchanged.

Affine code of one tensor (zero point ``z``, a level of the codebook with ``z <= 0``):
    y = x / s + z,  l = Q(y) (round to nearest level, B2 tie rule),  x_hat = (l - z) * s.
``z = 0`` is the B2 symmetric code.  Because ``z`` is itself a level, ``x = 0`` is reproduced exactly (zero
padding and exact zeros stay exact).  For integer codebooks this is the usual integer zero point.

Per-channel code: ``x_hat[:, c] = Q(x[:, c] / s_c) * s_c`` with one scale per channel (dimension 1).
"""
from __future__ import annotations

import numpy as np
import torch

from tools.experiment_b.classifier import sample_indices
from tools.experiment_b2.scales import PER_OCTAVE, REFINE, coarse_ratios, mse_search, scale_for_top

AFFINE_SEARCH = {
    "version": "b2_attrib_affine_search_v1",
    "objective": "sum_squared_error_of_fp32_qdq_in_float64 over the v1 calibration samples of the node",
    "grid": "B2 ratio grid (scales.coarse_ratios + the same refinement) relative to the asymmetric span "
            "(max(sample max, 0) - min(sample min, 0)) / (top level - bottom level)",
    "zero_point": "for each scale, the smallest codebook level >= bottom level - min/scale, capped at 0",
    "selection": "affine (scale, zero point) replaces the B2 symmetric MSE scale only if its error is strictly "
                 "smaller than the error of the B2 symmetric scale on the same samples",
}
PER_CHANNEL_SEARCH = {
    "version": "b2_attrib_per_channel_search_v1",
    "samples": "v1 calibration samples regrouped by channel (rotating_channel_stratified_256_per_image_v1: "
               "channel of sample k of calibration image i is (k + i*min(256, C)) % C), truncated to the "
               "smallest per-channel count",
    "search": "tools.experiment_b2.scales.mse_search per channel row, anchored at the per-channel sample max-abs",
}


def _positions(base, y):
    """Level index of every element of ``y`` (already divided by the scale): the B2 tie rule."""
    positions = torch.bucketize(y.contiguous(), base.boundaries, right=False)
    adjacent = positions.clamp_max(len(base.boundaries) - 1)
    return positions + ((y == base.boundaries[adjacent]) & base.ties[adjacent]).long()


class AffineQuantizer:
    """Affine code over a B2 table: ``x_hat = (Q(x/s + z) - z) * s``."""

    def __init__(self, base, scale, zero):
        self.base, self.name, self.signedness = base, base.name, f"{base.signedness}+affine"
        self.levels, self.top, self.min_positive = base.levels, base.top, base.min_positive
        self.scale = float(np.float32(scale))
        self.zero = float(np.float32(zero))
        if self.zero > 0 or not bool((base.levels == self.zero).any()):
            raise ValueError("the zero point must be a non-positive level of the codebook")
        device = base.levels.device
        self.scale_t = torch.tensor(self.scale, dtype=torch.float32, device=device)
        self.zero_t = torch.tensor(self.zero, dtype=torch.float32, device=device)

    def _check(self, scale):
        if float(scale) != self.scale:
            raise ValueError("affine quantizer called with a scale it was not built for")

    def __call__(self, values, scale):
        self._check(scale)
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite input to the affine quantizer")
        y = values / self.scale_t + self.zero_t
        return (self.levels[_positions(self.base, y)] - self.zero_t) * self.scale_t

    def indices(self, values, scale):
        self._check(scale)
        return _positions(self.base, values / self.scale_t + self.zero_t)


class PerChannelQuantizer:
    """One scale per channel (dimension 1) over a B2 table; the node keeps a reference scale for the engine."""

    def __init__(self, base, scales, reference):
        self.base, self.name, self.signedness = base, base.name, f"{base.signedness}+per_channel"
        self.levels, self.top, self.min_positive = base.levels, base.top, base.min_positive
        self.reference = float(np.float32(reference))
        self.scales = torch.as_tensor(np.asarray(scales, dtype=np.float32), device=base.levels.device)
        if self.scales.ndim != 1 or not torch.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("invalid per-channel scales")

    def _shaped(self, values, scale):
        if float(scale) != self.reference:
            raise ValueError("per-channel quantizer called with a reference scale it was not built for")
        if values.ndim < 2 or values.shape[1] != len(self.scales):
            raise ValueError("per-channel scales do not match the channel dimension")
        return self.scales.reshape((1, -1) + (1,) * (values.ndim - 2))

    def __call__(self, values, scale):
        return self.base(values, self._shaped(values, scale))

    def indices(self, values, scale):
        return self.base.indices(values, self._shaped(values, scale))


def qdq_error(values, quantizer, scale):
    """Float64 sum of squared QDQ error of a B2 table quantizer at one scalar scale."""
    diff = (quantizer(values, torch.tensor(np.float32(scale), device=values.device)) - values).double()
    return float((diff * diff).sum())


def _affine_error(values, base, scale, zero):
    s = torch.tensor(np.float32(scale), device=values.device)
    z = torch.tensor(np.float32(zero), device=values.device)
    y = values / s + z
    diff = ((base.levels[_positions(base, y)] - z) * s - values).double()
    return float((diff * diff).sum())


def zero_point_for(levels, lowest, scale):
    """Smallest level >= bottom - lowest/scale, capped at 0 (``levels`` ascending float32 numpy)."""
    target = np.float64(levels[0]) - np.float64(lowest) / np.float64(scale)
    candidates = levels[(levels.astype(np.float64) >= target - 1e-9) & (levels <= 0)]
    return float(candidates[0]) if len(candidates) else 0.0


def affine_search(values, base, symmetric_scale):
    """Affine MSE search for one node (``values``: 1-D float32 tensor of calibration samples).

    Returns ``(params, info)``: ``params`` is ``{"scale", "zero"}`` or ``None`` when the B2 symmetric scale
    is at least as good; ``info`` holds both errors and the chosen ratio.
    """
    values = values.reshape(-1)
    levels = base.levels.cpu().numpy()
    symmetric = qdq_error(values, base, symmetric_scale)
    low, high = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    info = {"symmetric_sse": symmetric, "sample_min": low, "sample_max": high, "count": int(values.numel())}
    if high - low <= 0 or levels[0] >= 0:
        return None, {**info, "affine_sse": None, "reason": "zero span or no negative levels"}
    anchor = scale_for_top(high - low, float(levels[-1]) - float(levels[0]))

    def evaluate(ratio):
        scale = float(np.float32(anchor * ratio))
        zero = zero_point_for(levels, low, scale)
        return _affine_error(values, base, scale, zero), scale, zero

    best = (float("inf"), None, None, None)
    for ratio in coarse_ratios(base):
        error, scale, zero = evaluate(float(ratio))
        if error < best[0]:
            best = (error, scale, zero, float(ratio))
    centre = best[3]
    for step in range(-REFINE + 1, REFINE):
        if step == 0:
            continue
        ratio = float(np.float32(centre * 2.0 ** (step / (PER_OCTAVE * REFINE))))
        error, scale, zero = evaluate(ratio)
        if error < best[0]:
            best = (error, scale, zero, ratio)
    error, scale, zero, ratio = best
    info = {**info, "affine_sse": error, "affine_scale": scale, "affine_zero": zero, "ratio_to_anchor": ratio}
    if not error < symmetric or zero == 0.0:
        return None, {**info, "reason": "symmetric code at least as good" if not error < symmetric
                      else "best affine candidate has zero point 0"}
    return {"scale": scale, "zero": zero}, {**info, "reason": "affine selected"}


def channel_samples(flat, channels, images):
    """Regroup the v1 calibration samples of one node by channel: float32 ``[channels, m]``.

    ``flat`` is the node's concatenated sample vector (``images`` rows of ``count`` samples each, in
    calibration order).  The channel of every sample is recomputed with the v1 sampler itself; within a
    channel the samples keep calibration order (image, then sample index).
    """
    flat = np.asarray(flat, dtype=np.float32)
    if flat.size % images:
        raise ValueError("calibration samples are not a whole number of images")
    count = flat.size // images
    # sample_indices only needs a spatial size consistent with ``count``; channel ids do not depend on it.
    spatial = max(1, -(-count // channels))
    ids = np.empty((images, count), dtype=np.int64)
    for ordinal in range(images):
        ch, _ = sample_indices(channels, spatial, ordinal)
        if len(ch) < count:
            raise ValueError("sampler reproduces fewer samples than were stored")
        ids[ordinal] = ch[:count]
    ids = ids.ravel()
    order = np.argsort(ids, kind="stable")
    sizes = np.bincount(ids, minlength=channels)
    m = int(sizes.min())
    if m == 0:
        raise ValueError("a channel has no calibration sample")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    take = order[(starts[:, None] + np.arange(m)[None, :]).ravel()]
    return flat[take].reshape(channels, m), {"per_channel_samples": m, "max_per_channel_samples": int(sizes.max())}


def per_channel_search(samples, base, device):
    values = torch.as_tensor(samples, dtype=torch.float32, device=device)
    scales, info = mse_search(values, base)
    return scales, {"median_ratio_to_maxabs_scale": float(np.median(info["ratio_to_maxabs_scale"])),
                    "sse": float(np.sum(info["mse"]) * samples.shape[1])}
