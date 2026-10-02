"""The cells of lane Q2 (protocol b2-rank-protocol-v1), in run-priority order, and how each recipe maps per format.

``cells()`` yields dicts ``{part, model, format, recipe, variant, priority, source}``; ``source`` is ``l1`` for a cell
that L1 already sealed (reused read-only, never run) and ``new`` otherwise.
"""
from __future__ import annotations

from tools.experiment_b.common import formats

MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
INTEGERS = ("int8", "int6", "int5", "int4")
BLOCK = ("bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3")
# Formats at chance (< 1 percent) under both default and minimal on that model in L1's matrix; their factorial
# cells run last (priority 5) because they cannot take part in any reversal that is not between chance formats.
L1_CHANCE_BOTH = {
    "resnet18": ("ternary", "binary_pm1"),
    "mobilenet_v2": ("int4", "fp4_e2m1", "posit4_es0", "log4", "mxfp4_e2m1", "ternary", "binary_pm1"),
    "mobilenet_v3_large": ("int5", "int4", "fp4_e2m1", "posit4_es0", "log4", "mxfp4_e2m1", "ternary", "binary_pm1"),
}
# Factorial corners as (U, W, B): U unsigned (integers only), W weight range mse(1)/maxabs(0), B bias correction.
SCALAR_CORNERS = {
    (1, 1, 1): "default", (0, 1, 1): "default_signed", (1, 1, 0): "default_no_bias_correction",
    (1, 0, 1): "default_weight_maxabs", (1, 0, 0): "cum3_act_mse", (0, 1, 0): "rank_signed_no_bias_correction",
    (0, 0, 1): "rank_signed_weight_maxabs", (0, 0, 0): "minimal"}
INTRINSIC_CORNERS = {(1, 1): "cum5_act_maxabs", (1, 0): "rank_intrinsic_no_bias_correction",
                     (0, 1): "rank_intrinsic_weight_maxabs", (0, 0): "cum1_fused"}
SEARCHED_CORNERS = {(1, 1): "default", (1, 0): "default_no_bias_correction", (0, 1): "default_weight_maxabs",
                    (0, 0): "minimal"}
L1_RECIPES = ("default", "minimal", "cum5_act_maxabs", "cum1_fused")
UNSIGNED_SCALAR = ("fp8_e4m3fn", "fp8_e5m2", "fp7_e3m3", "fp6_e2m3", "fp6_e3m2", "fp5_e2m2", "fp4_e2m1",
                   "posit8_es1", "posit6_es1", "posit4_es0", "log8", "log6", "log4")


def recipe_for(format_name, u, w, b, arm="intrinsic"):
    """Cell recipe of a format at factorial corner (u, w, b).  For non-integers U is a no-op (u is ignored)."""
    if format_name in BLOCK:
        return (INTRINSIC_CORNERS if arm == "intrinsic" else SEARCHED_CORNERS)[(w, b)]
    if format_name not in INTEGERS:
        u = 1 if (w, b) != (0, 0) else 0  # the named arm that exists: default-side names, minimal for (0, 0)
        if (w, b) == (0, 0):
            return "minimal"
    return SCALAR_CORNERS[(u, w, b)]


def cells():
    names = [row["name"] for row in formats()]
    out = []

    def add(part, model, name, recipe, variant, priority):
        source = "l1" if variant == "plain" and part != "searched" and recipe in L1_RECIPES else "new"
        if part == "searched" and recipe in ("default", "minimal"):
            source = "l1"
        out.append({"part": part, "model": model, "format": name, "recipe": recipe, "variant": variant,
                    "priority": priority, "source": source})

    for model in MODELS:
        for name in names:
            chance = name in L1_CHANCE_BOTH[model]
            if name in INTEGERS:
                for corner, recipe in SCALAR_CORNERS.items():
                    add("factorial", model, name, recipe, "plain", 5 if chance else (1 if corner[0] == 0 else 2))
            elif name in BLOCK:
                for recipe in INTRINSIC_CORNERS.values():
                    add("factorial", model, name, recipe, "plain", 5 if chance else 3)
                for recipe in SEARCHED_CORNERS.values():
                    add("searched", model, name, recipe, "plain", 4 if model == "resnet18" else 6)
            else:
                for recipe in ("default", "default_no_bias_correction", "default_weight_maxabs", "minimal"):
                    add("factorial", model, name, recipe, "plain", 5 if chance else 2)
            if name in UNSIGNED_SCALAR:
                add("unsigned", model, name, "default", "unsigned", 1)
            if name in BLOCK:
                add("unsigned", model, name, "cum5_act_maxabs", "unsigned", 3)
                add("unsigned", model, name, "default", "unsigned", 6)
    return out
