"""Arms on the ladder base without bias correction (protocol addendum 1).  A separate module so that
``recipe.py`` (a numeric source) and every sealed configuration identity stay unchanged."""
from __future__ import annotations

from dataclasses import replace

from .recipe import CUMULATIVE, HEAD_FP32, register

_B = CUMULATIVE["cum5_weight_mse"]
ARMS2 = {
    "cum5_concat_passthrough": replace(_B, passthrough="nonarith_concat"),
    "cum5_fused_silu": replace(_B, boundaries="fused_silu"),
    "cum5_fused_concat": replace(_B, boundaries="fused_silu", passthrough="nonarith_concat"),
    "cum5_signed": replace(_B, unsigned=False),
    "cum5_act_maxabs": replace(_B, activation_range="maxabs"),
    "cum5_no_passthrough": replace(_B, passthrough="none"),
    "cum5_q_projection": replace(_B, q_projection=True),
    "cum5_q_dfl": replace(_B, q_dfl=True),
    "cum5_q_boxes": replace(_B, q_boxes=True),
    "cum5_q_scores_also": replace(_B, q_scores=True),
    "cum5_q_scores_not_logits": replace(_B, q_scores=True, q_class_logits=False),
    "cum5_v1_head": replace(_B, q_projection=True, q_dfl=True, q_boxes=True, q_scores=True, q_joins=True),
    "cum5_fp32_head": replace(_B, **HEAD_FP32),
    "cum5_fp32_box_logits": replace(_B, q_box_logits=False),
    "cum5_fp32_class_logits": replace(_B, q_class_logits=False),
    "cum5_no_input": replace(_B, quantize_input=False),
    "cum5_weights_only": replace(_B, quantize_activations=False),
    "cum5_activations_only": replace(_B, quantize_weights=False),
}
for _name, _recipe in ARMS2.items():
    register(_name, _recipe)
