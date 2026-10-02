"""Codebooks for B2: the accepted signed tables of v1 plus an unsigned integer variant.

The unsigned variant is not an accepted manifest.  It is a distinct activation
format identity (``<format>:unsigned``) defined here for integer formats only:
levels ``0 .. 2**bits - 1`` with round-to-nearest, ties to the even code.
"""
from __future__ import annotations

from functools import lru_cache
import json

import numpy as np

from tools.experiment_b.common import ROOT, digest
from tools.experiment_b.quantizer import Quantizer, table

UNSIGNED_VARIANT = "b2_unsigned_integer_v1"


def manifest(name):
    return json.loads((ROOT / f"public/formats/manifests/accepted/{name}.json").read_text())


@lru_cache(maxsize=32)
def family_bits(name):
    index = json.loads((ROOT / "public/formats/manifests/accepted/index.json").read_text())["manifests"]
    for entry in index:
        if entry["name"] == name:
            return entry["family"], int(entry["bits"])
    raise ValueError(f"unknown accepted format: {name}")


def supports_unsigned(name):
    return family_bits(name)[0] == "integer"


@lru_cache(maxsize=32)
def unsigned_table(name):
    family, bits = family_bits(name)
    if family != "integer":
        raise ValueError("the unsigned codebook variant is defined for integer formats only")
    levels = np.arange(1 << bits, dtype=np.float32)
    boundaries = ((levels[:-1].astype(np.float64) + levels[1:].astype(np.float64)) / 2).astype(np.float32)
    # Same preference as v1: the even code wins a tie.
    choose_upper = np.array([(code + 1) % 2 == 0 for code in range((1 << bits) - 1)])
    return levels, boundaries, choose_upper


def get_table(name, signedness):
    if signedness == "signed":
        return table(name)
    if signedness == "unsigned":
        return unsigned_table(name)
    raise ValueError(f"unknown signedness: {signedness}")


def codebook_id(name, signedness):
    return name if signedness == "signed" else f"{name}:unsigned"


@lru_cache(maxsize=32)
def level_codes(name, signedness):
    """Manifest code of every level of the table (smallest code among duplicates, as v1)."""
    levels = get_table(name, signedness)[0]
    if signedness == "unsigned":
        return np.arange(len(levels), dtype=np.int64)
    from public.formats.oracle.number_format import NumberFormat
    fmt = NumberFormat(manifest(name))
    by_value = {}
    for code in range(1 << fmt.bits):
        value = fmt.decode(code)
        if value.is_finite():
            key = np.float32(float(value))
            by_value[key] = min(code, by_value.get(key, code))
    codes = np.array([by_value[x] for x in levels], dtype=np.int64)
    return codes


def describe(name, signedness):
    levels, boundaries, ties = get_table(name, signedness)
    entry = {"base_format": name, "signedness": signedness, "levels": [float(x) for x in levels],
             "codes": [int(x) for x in level_codes(name, signedness)],
             "boundaries": [float(x) for x in boundaries], "choose_upper_tie": [bool(x) for x in ties],
             "top_level": float(levels[-1])}
    if signedness == "unsigned":
        entry["variant"] = UNSIGNED_VARIANT
    entry["identity_sha256"] = digest(entry)
    return entry


def numpy_qdq(values, name, signedness, scale):
    """Reference QDQ in NumPy, the same arithmetic as ``tools.experiment_b.quantizer.numpy_qdq``."""
    levels, boundaries, ties = get_table(name, signedness)
    y = np.asarray(values, dtype=np.float32) / np.asarray(scale, dtype=np.float32)
    positions = np.searchsorted(boundaries, y, side="left")
    adjacent = np.minimum(positions, len(boundaries) - 1)
    positions = positions + ((y == boundaries[adjacent]) & ties[adjacent])
    result = levels[positions]
    result = np.where((result == 0) & np.signbit(y), np.float32(-0.0), result)
    return (result * np.asarray(scale, dtype=np.float32)).astype(np.float32)


class TableQuantizer(Quantizer):
    """The v1 torch quantizer (its ``__call__`` is inherited unchanged) over any B2 table."""

    def __init__(self, name, signedness, device):
        import torch
        if signedness == "signed":
            super().__init__(name, device)
        else:
            self.name = name
            levels, boundaries, ties = unsigned_table(name)
            self.levels = torch.tensor(levels, device=device)
            self.boundaries = torch.tensor(boundaries, device=device)
            self.ties = torch.tensor(ties, device=device)
        self.signedness = signedness
        self.top = float(self.levels[-1])
        positive = self.levels[self.levels > 0]
        self.min_positive = float(positive[0])

    def indices(self, values, scale):
        """Level index of every element (used by the export and the occupancy audit)."""
        import torch
        y = values / scale
        positions = torch.bucketize(y.contiguous(), self.boundaries, right=False)
        adjacent = positions.clamp_max(len(self.boundaries) - 1)
        positions = positions + ((y == self.boundaries[adjacent]) & self.ties[adjacent]).long()
        return positions
