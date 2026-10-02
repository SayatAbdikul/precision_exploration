"""Frozen detector recipe, registered after the dev128 diagnosis of 2026-10-01 (protocol
b2-detector-protocol-v1 with addendum 1; evidence results/summaries/b2-detector-v1/effects-128.json).

Classifier default where the same switch exists: unsigned, activation MSE, weight MSE, max-pool pass-through.
Differences: bias correction dropped (leave-one-out +11.2 [+7.5, +13.9] on dev128); SiLU fused (+1.42
[+0.26, +2.09]); slice and upsample pass-through (free); decoded boxes and the 84-channel joins are not
k-bit stores (-2.16 [-3.30, -1.50] when the boxes are put back).  Concatenation pass-through was not
adopted (+0.66 [-0.15, +1.45]).  Projection constant, box expectation and sigmoid scores stay quantized
(putting them back costs nothing at INT8).
"""
from __future__ import annotations

from dataclasses import replace

from .recipe import DetRecipe, HEAD_FP32, register

DEFAULT = DetRecipe(boundaries="fused_silu", passthrough="nonarith", unsigned=True, activation_range="mse",
                    weight_range="mse", bias_correction="none", q_projection=True, q_box_logits=True, q_dfl=True,
                    q_boxes=False, q_class_logits=True, q_scores=True, q_joins=False)
FROZEN = {
    "default": DEFAULT,
    # Ablations, one change from the default.
    "default_unfused_silu": replace(DEFAULT, boundaries="v1"),
    "default_head_logits": replace(DEFAULT, q_projection=False, q_dfl=False, q_scores=False),
    "default_q_boxes": replace(DEFAULT, q_boxes=True),
    "default_bias_correction": replace(DEFAULT, bias_correction="empirical"),
    "default_concat_passthrough": replace(DEFAULT, passthrough="nonarith_concat"),
    "default_signed": replace(DEFAULT, unsigned=False),
    "default_act_maxabs": replace(DEFAULT, activation_range="maxabs"),
    "default_weight_maxabs": replace(DEFAULT, weight_range="maxabs"),
    # Residual account (diagnostics, not hardware recipes).
    "default_fp32_head": replace(DEFAULT, **HEAD_FP32),
    "default_no_input": replace(DEFAULT, quantize_input=False),
    "default_weights_only": replace(DEFAULT, quantize_activations=False),
    "default_activations_only": replace(DEFAULT, quantize_weights=False),
}
for _name, _recipe in FROZEN.items():
    register(_name, _recipe)
