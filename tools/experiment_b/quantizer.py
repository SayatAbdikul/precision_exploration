"""FP32 codebook QDQ: an explicit surrogate, not the exact arithmetic engine.

Finite levels and midpoint computations are rounded to FP32. External scales
are FP32, normalization/reconstruction are FP32, overflow clips to finite
endpoints. Log/NF levels and near-boundary behavior can differ from the oracle.
"""
from __future__ import annotations

from functools import lru_cache
import json
import numpy as np

from .common import ROOT, RECIPES


@lru_cache(maxsize=25)
def table(name):
    from public.formats.oracle.number_format import NumberFormat
    fmt = NumberFormat(json.loads((ROOT / f"public/formats/manifests/accepted/{name}.json").read_text()))
    if fmt.manifest["scaling"]["mode"] == "intrinsic_shared":
        raise ValueError("scalar QDQ must not pretend to implement shared-block formats")
    by_value = {}
    for code in range(1 << fmt.bits):
        value = fmt.decode(code)
        if value.is_finite():
            key = np.float32(float(value))
            by_value[key] = min(code, by_value.get(key, code))
    levels = np.array(sorted(by_value), dtype=np.float32)
    codes = np.array([by_value[x] for x in levels])
    boundaries = ((levels[:-1].astype(np.float64) + levels[1:].astype(np.float64)) / 2).astype(np.float32)
    if fmt.manifest["rounding"] != "rne" or not np.all(np.diff(levels) > 0):
        raise ValueError("unsupported codebook/rounding")
    choose_upper = np.array([(int(b) & 1, int(b)) < (int(a) & 1, int(a)) for a, b in zip(codes[:-1], codes[1:])])
    return levels, boundaries, choose_upper


def threshold(values, recipe):
    values = np.asarray(values, dtype=np.float32)
    if recipe not in RECIPES or not values.size or not np.isfinite(values).all():
        raise ValueError("invalid scale calibration input/recipe")
    values = np.abs(values)
    return float(values.max()) if recipe == "maxabs" else float(np.quantile(values, .999, method="linear"))


def scale_for(span, name):
    # Unit scale for a zero training distribution; no inference-data adaptation.
    scale = np.float32(span / float(table(name)[0][-1])) if span > 0 else np.float32(1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid FP32 external scale")
    return float(scale)


def numpy_qdq(values, name, scale):
    levels, boundaries, ties = table(name)
    y = np.asarray(values, dtype=np.float32) / np.asarray(scale, dtype=np.float32)
    positions = np.searchsorted(boundaries, y, side="left")
    adjacent = np.minimum(positions, len(boundaries)-1)
    positions = positions + ((y == boundaries[adjacent]) & ties[adjacent])
    result = levels[positions]
    result = np.where((result == 0) & np.signbit(y), np.float32(-0.0), result)
    return (result * np.asarray(scale, dtype=np.float32)).astype(np.float32)


class Quantizer:
    def __init__(self, name, device):
        import torch
        self.name = name
        levels, boundaries, ties = table(name)
        self.levels = torch.tensor(levels, device=device)
        self.boundaries = torch.tensor(boundaries, device=device)
        self.ties = torch.tensor(ties, device=device)

    def __call__(self, values, scale):
        import torch
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite input to B quantizer")
        y = values / scale
        positions = torch.bucketize(y.contiguous(), self.boundaries, right=False)
        adjacent = positions.clamp_max(len(self.boundaries)-1)
        positions += ((y == self.boundaries[adjacent]) & self.ties[adjacent]).long()
        result = self.levels[positions]
        result = torch.where(result == 0, torch.copysign(result, y), result)
        return result * scale
