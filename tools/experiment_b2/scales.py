"""Range rules for B2: v1 max-abs and percentile, plus a format-agnostic MSE scale search.

The MSE rule minimises the reconstruction error of the *actual* quantize-
dequantize function of the codebook over a grid of scales.  The grid is
expressed as a ratio to the max-abs scale ``maxabs / top_level``:

* ratio < 1 clips the tail (what helps uniform integer codebooks);
* ratio > 1 moves the bulk of the values towards smaller levels (what helps
  tapered codebooks such as posit, whose levels are dense near 1 and sparse
  near the top level).

There is no per-format tuning: the upward extent of the grid is the dynamic
range of the codebook itself.
"""
from __future__ import annotations

import math

import numpy as np

from tools.experiment_b.quantizer import threshold

CLIP_OCTAVES = 6
PER_OCTAVE = 8
REFINE = 8
MAX_UP_OCTAVES = 30
SEARCH = {"version": "b2_mse_scale_search_v1", "objective": "sum_squared_error_of_fp32_qdq_in_float64",
          "grid": "ratio_to_maxabs_scale; log2 spaced", "clip_octaves": CLIP_OCTAVES, "per_octave": PER_OCTAVE,
          "up_octaves": "ceil(log2(top_level/min_positive_level)) capped at 30",
          "refine": "2*REFINE-1 log-spaced points between the coarse neighbours of the coarse optimum",
          "tie": "first candidate in ascending ratio order", "refine_steps": REFINE,
          "zero_tensor": "unit scale, as v1"}


def scale_for_top(span, top):
    """The v1 ``scale_for`` formula for an arbitrary top level."""
    scale = np.float32(span / float(top)) if span > 0 else np.float32(1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid FP32 external scale")
    return float(scale)


def simple_span(values, maximum, rule):
    """Max-abs or percentile span exactly as v1 computes it."""
    if rule == "maxabs":
        return float(maximum) if maximum is not None else threshold(values, "maxabs")
    if rule == "percentile_99_9":
        return threshold(values, "percentile_99_9")
    raise ValueError(f"not a simple range rule: {rule}")


def coarse_ratios(quantizer):
    up = min(MAX_UP_OCTAVES, max(0, math.ceil(math.log2(quantizer.top / quantizer.min_positive))))
    steps = np.arange(-CLIP_OCTAVES * PER_OCTAVE, up * PER_OCTAVE + 1)
    return np.exp2(steps / PER_OCTAVE)


def mse_search(values, quantizer, maxima=None):
    """Per-row MSE-optimal scales.

    ``values``: float32 tensor ``[rows, elements]`` (one row per independently
    scaled group: one row for an activation node, one row per output channel
    for a weight).  ``maxima``: optional per-row max-abs to anchor the grid
    (defaults to the row maximum).  Returns ``(scales float32 [rows], info)``.
    """
    import torch
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("invalid input to the MSE scale search")
    rows = values.shape[0]
    device = values.device
    span = values.abs().amax(dim=1) if maxima is None else torch.as_tensor(maxima, dtype=torch.float32, device=device)
    if span.shape != (rows,):
        raise ValueError("per-row maxima shape mismatch")
    base = np.array([scale_for_top(float(x), quantizer.top) for x in span.cpu().numpy()], dtype=np.float32)
    base_t = torch.tensor(base, device=device).reshape(rows, 1)

    def errors(ratio):
        scale = (base_t * ratio.reshape(rows, 1)).to(torch.float32)
        diff = (quantizer(values, scale) - values).double()
        return (diff * diff).sum(dim=1)

    best_error = torch.full((rows,), float("inf"), dtype=torch.float64, device=device)
    best_ratio = torch.ones(rows, dtype=torch.float32, device=device)
    reference_error = None
    for value in coarse_ratios(quantizer):
        ratio = torch.full((rows,), float(value), dtype=torch.float32, device=device)
        error = errors(ratio)
        if value == 1.0:
            reference_error = error.clone()
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_ratio = torch.where(better, ratio, best_ratio)
    centre = best_ratio.clone()
    for step in range(-REFINE + 1, REFINE):
        if step == 0:
            continue
        ratio = (centre.double() * 2.0 ** (step / (PER_OCTAVE * REFINE))).to(torch.float32)
        error = errors(ratio)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_ratio = torch.where(better, ratio, best_ratio)
    scales = (base_t.reshape(rows) * best_ratio).to(torch.float32).cpu().numpy()
    zero = span.cpu().numpy() <= 0
    scales[zero] = np.float32(1)
    count = values.shape[1]
    info = {"ratio_to_maxabs_scale": [float(x) for x in best_ratio.cpu().numpy()],
            "mse": [float(x) / count for x in best_error.cpu().numpy()],
            "mse_at_maxabs_scale": [float(x) / count for x in reference_error.cpu().numpy()]}
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("MSE search produced an invalid scale")
    return scales, info
