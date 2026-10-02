"""Unsigned n-bit codebooks for non-integer formats (lane Q2, protocol b2-rank-protocol-v1).

Rule ``b2_rank_unsigned_sibling_v1`` (one rule for every family, fixed before any measurement):

    U_n(F) = the non-negative finite values of F's (n+1)-bit sibling,

where the sibling is F's encoding with one more least-significant bit and every other field and convention
unchanged: float and MX element formats get one more mantissa bit (same exponent width, bias, subnormals,
infinity/NaN convention); logarithmic formats one more fractional bit of the exponent (same integer bits);
BFP one more fraction bit of the mantissa; posit(n, es) becomes posit(n+1, es).  The sign bit of the sibling is
always 0, so its remaining n bits are the unsigned code.  For integers the rule gives exactly B2's unsigned
integer codebook (non-negative half of INT(n+1) = 0 .. 2^n - 1), which is the reason for choosing it.

The sibling manifest exists only in memory: no manifest file is added (that would change the v1 source identity).
Values are decoded by the project's oracle (``public.formats.oracle.number_format.NumberFormat``).  The table has
the shape of ``tools.experiment_b.quantizer.table``: ascending levels, midpoint boundaries, and the same
round-half-to-even-code preference on ties.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
import re

import numpy as np

from tools.experiment_b.common import ROOT, digest

RULE = "b2_rank_unsigned_sibling_v1"
SCALAR = ("fp8_e4m3fn", "fp8_e5m2", "fp7_e3m3", "fp6_e2m3", "fp6_e3m2", "fp5_e2m2", "fp4_e2m1",
          "posit8_es1", "posit6_es1", "posit4_es0", "log8", "log6", "log4")
BLOCK = ("bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1")
EXCLUDED = {"nf4": "codebook format without a sign-magnitude encoding (an NF5 sibling has 17 non-negative levels)",
            "binary_pm1": "no (n+1)-bit sibling; at chance under every recipe",
            "ternary": "no (n+1)-bit sibling; at chance under every recipe",
            "q1_6": "output alias of signed INT8; its unsigned variant is INT8's own unsigned codebook"}


def manifest(name):
    return json.loads((ROOT / f"public/formats/manifests/accepted/{name}.json").read_text())


def sibling_manifest(name):
    """In-memory manifest of the (n+1)-bit sibling of an accepted format."""
    m = copy.deepcopy(manifest(name))
    family = m["family"]
    m["bits"] = m["bits"] + 1
    m["name"] = f"{name}__sibling{m['bits']}"
    if family in ("float", "mx_float"):
        m["float"]["mantissa_bits"] += 1
        m["encoding"] = re.sub(r"m(\d+)$", lambda x: f"m{int(x.group(1)) + 1}", m["encoding"])
    elif family == "logarithmic":
        m["logarithmic"]["fractional_bits"] += 1
    elif family == "bfp":
        m["numeric"]["fractional_bits"] += 1
    elif family == "integer":
        pass  # INT(n+1): the identity check of the rule against B2's unsigned integer codebook
    elif family == "posit":
        pass  # posit(n+1, es)
    else:
        raise ValueError(f"no unsigned sibling rule for family {family} ({name})")
    return m  # block formats keep their block fields; element values are decoded at scale 1 = 2^0


@lru_cache(maxsize=64)
def sibling_table(name):
    """``(levels, boundaries, choose_upper)`` of U_n(name) (float32 / float32 / bool), codes are the n-bit codes."""
    from public.formats.oracle.number_format import NumberFormat
    m = sibling_manifest(name)
    fmt = NumberFormat(m)
    n = m["bits"] - 1
    by_value = {}
    for code in range(1 << n):          # sign bit 0: the n-bit unsigned code is the sibling code itself
        value = fmt.decode(code)
        if value.is_finite() and value >= 0:
            key = np.float32(float(value))
            if key == 0:
                key = np.float32(0.0)
            by_value[key] = min(code, by_value.get(key, code))
    levels = np.array(sorted(by_value), dtype=np.float32)
    codes = np.array([by_value[x] for x in levels])
    if levels[0] != 0 or not np.all(np.diff(levels) > 0) or len(levels) > (1 << n):
        raise ValueError(f"invalid unsigned sibling codebook for {name}")
    boundaries = ((levels[:-1].astype(np.float64) + levels[1:].astype(np.float64)) / 2).astype(np.float32)
    choose_upper = np.array([(int(b) & 1, int(b)) < (int(a) & 1, int(a)) for a, b in zip(codes[:-1], codes[1:])])
    return levels, boundaries, choose_upper


def signed_levels(name):
    from public.formats.oracle.number_format import NumberFormat
    fmt = NumberFormat(manifest(name))
    values = {np.float32(float(fmt.decode(c))) for c in range(1 << fmt.bits) if fmt.decode(c).is_finite()}
    return np.array(sorted(abs(v) for v in values if v >= 0), dtype=np.float32)


def describe(name):
    """Identity and facts of U_n(name) (what enters the configuration of an unsigned cell)."""
    levels, boundaries, ties = sibling_table(name)
    signed = signed_levels(name)
    positive, spositive = levels[levels > 0], signed[signed > 0]
    entry = {"rule": RULE, "base_format": name, "sibling_manifest": sibling_manifest(name),
             "levels": int(len(levels)), "levels_sha256": digest([float(x) for x in levels]),
             "boundaries_sha256": digest([float(x) for x in boundaries]),
             "ties_sha256": digest([bool(x) for x in ties]),
             "signed_nonnegative_levels": int(len(signed)),
             "max_over_signed_max": float(positive[-1] / spositive[-1]),
             "min_positive_over_signed_min_positive": float(positive[0] / spositive[0]),
             "contains_signed_levels": bool(np.isin(signed, levels).all())}
    entry["identity_sha256"] = digest(entry)
    return entry
