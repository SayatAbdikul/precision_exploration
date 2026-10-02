"""Frozen B2 recipes, registered after the INT8 diagnosis on dev512 (2026-10-01).

Freeze rule (protocol b2-baseline-repair-protocol-v1, stage S2): the default is the full
ladder with deployable boundaries (fused_relu); no leave-one-out arm showed that removing a
step is better with a 95 percent paired interval excluding zero, so no step was dropped.
The same switches apply to every scalar format; ``unsigned`` only takes effect for integers.
"""
from __future__ import annotations

from dataclasses import replace

from .recipe import Recipe, register

DEFAULT = Recipe(boundaries="fused_relu", unsigned=True, activation_range="mse", weight_range="mse",
                 bias_correction="empirical")
FROZEN = {
    "default": DEFAULT,
    # Named ablation arms (one switch away from the default, plus the two-switch minimal recipe).
    "default_fused_all": replace(DEFAULT, boundaries="fused_all"),
    "default_signed": replace(DEFAULT, unsigned=False),
    "default_no_bias_correction": replace(DEFAULT, bias_correction="none"),
    "default_weight_maxabs": replace(DEFAULT, weight_range="maxabs"),
    "minimal": Recipe(boundaries="fused_relu", activation_range="mse"),
}
for _name, _recipe in FROZEN.items():
    register(_name, _recipe)
