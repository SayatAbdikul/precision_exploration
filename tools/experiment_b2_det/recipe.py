"""Detector recipe switches (Experiment B2 detector), named arms and their hardware meaning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

BOUNDARIES = ("v1", "fused_silu")
PASSTHROUGH = ("none", "nonarith", "nonarith_concat")
RANGES = ("maxabs", "percentile_99_9", "mse")
BIAS = ("none", "empirical")
FLAGS = ("unsigned", "q_projection", "q_box_logits", "q_dfl", "q_boxes", "q_class_logits", "q_scores", "q_joins",
         "quantize_input", "quantize_weights", "quantize_activations")


@dataclass(frozen=True)
class DetRecipe:
    boundaries: str = "v1"
    passthrough: str = "none"
    unsigned: bool = False
    activation_range: str = "maxabs"
    weight_range: str = "maxabs"
    bias_correction: str = "none"
    # Head: which tensors quantize (v1: all of them).
    q_projection: bool = True      # constant 0..15 of the distribution-focal-loss expectation
    q_box_logits: bool = True      # 64-channel output of the last box convolution
    q_dfl: bool = True             # expectation (distance in bins)
    q_boxes: bool = True           # decoded, stride-scaled xywh in pixels (and the box group of the joins)
    q_class_logits: bool = True    # 80-channel output of the last class convolution
    q_scores: bool = True          # sigmoid output (and the score group of the joins)
    q_joins: bool = True           # 84-channel concatenations of boxes and scores
    # Diagnostics, never part of a frozen recipe.
    quantize_input: bool = True
    quantize_weights: bool = True
    quantize_activations: bool = True

    def __post_init__(self):
        if (self.boundaries not in BOUNDARIES or self.passthrough not in PASSTHROUGH
                or self.activation_range not in RANGES or self.weight_range not in RANGES
                or self.bias_correction not in BIAS):
            raise ValueError(f"invalid detector recipe: {self}")
        for key in FLAGS:
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"detector recipe switch {key} must be a bool")

    def as_dict(self):
        return asdict(self)

    def v1_equivalent(self):
        """Name of the v1 detector recipe this reproduces bit for bit, or None."""
        if replace(self, activation_range="maxabs", weight_range="maxabs") != DetRecipe():
            return None
        if self.activation_range == self.weight_range and self.activation_range != "mse":
            return self.activation_range
        return None


V1_MAXABS = DetRecipe()
V1_PERCENTILE = DetRecipe(activation_range="percentile_99_9", weight_range="percentile_99_9")
HEAD_LOGITS = dict(q_projection=False, q_dfl=False, q_boxes=False, q_scores=False, q_joins=False)
HEAD_FP32 = dict(HEAD_LOGITS, q_box_logits=False, q_class_logits=False)

# One change at a time from v1 max-abs.
ONE = {
    "a_fused_silu": replace(V1_MAXABS, boundaries="fused_silu"),
    "b_no_projection": replace(V1_MAXABS, q_projection=False),
    "b_no_box_logits": replace(V1_MAXABS, q_box_logits=False),
    "b_no_dfl": replace(V1_MAXABS, q_dfl=False),
    "b_no_boxes": replace(V1_MAXABS, q_boxes=False),
    "b_no_class_logits": replace(V1_MAXABS, q_class_logits=False),
    "b_no_scores": replace(V1_MAXABS, q_scores=False),
    "b_no_joins": replace(V1_MAXABS, q_joins=False),
    "b_head_logits": replace(V1_MAXABS, **HEAD_LOGITS),
    "b_head_fp32": replace(V1_MAXABS, **HEAD_FP32),
    "c_act_mse": replace(V1_MAXABS, activation_range="mse"),
    "c_act_percentile": replace(V1_MAXABS, activation_range="percentile_99_9"),
    "c_weight_mse": replace(V1_MAXABS, weight_range="mse"),
    "d_bias_correction": replace(V1_MAXABS, bias_correction="empirical"),
    "e_passthrough": replace(V1_MAXABS, passthrough="nonarith"),
    "e_passthrough_concat": replace(V1_MAXABS, passthrough="nonarith_concat"),
    "u_unsigned": replace(V1_MAXABS, unsigned=True),
    "g_no_input": replace(V1_MAXABS, quantize_input=False),
}

# Cumulative ladder: head first (the v1-specific tensors), then (e), unsigned, (c), (d), and (a) last because
# the classifier default does not fuse a non-clamp activation.
_C1 = replace(V1_MAXABS, **HEAD_LOGITS)
_C2 = replace(_C1, passthrough="nonarith")
_C3 = replace(_C2, unsigned=True)
_C4 = replace(_C3, activation_range="mse")
_C5 = replace(_C4, weight_range="mse")
_C6 = replace(_C5, bias_correction="empirical")
_C7 = replace(_C6, passthrough="nonarith_concat")
_C8 = replace(_C7, boundaries="fused_silu")
CUMULATIVE = {
    "cum1_head_logits": _C1, "cum2_passthrough": _C2, "cum3_unsigned": _C3, "cum4_act_mse": _C4,
    "cum5_weight_mse": _C5, "cum6_bias_correction": _C6, "cum7_concat_passthrough": _C7, "cum8_fused_silu": _C8,
    # cum6 with one more change (the two contract-changing graph options, separately).
    "cum6_fused_silu": replace(_C6, boundaries="fused_silu"),
    # Leave-one-out around the full ladder cum8.
    "full_signed": replace(_C8, unsigned=False),
    "full_act_maxabs": replace(_C8, activation_range="maxabs"),
    "full_weight_maxabs": replace(_C8, weight_range="maxabs"),
    "full_no_bias_correction": replace(_C8, bias_correction="none"),
    "full_no_passthrough": replace(_C8, passthrough="none"),
    "full_unfused_silu": _C7,
    "full_no_concat_passthrough": replace(_C8, passthrough="nonarith"),
    # Head variants on the conservative graph cum6 (one head tensor put back, or one more taken out).
    "cum6_q_projection": replace(_C6, q_projection=True),
    "cum6_q_dfl": replace(_C6, q_dfl=True),
    "cum6_q_boxes": replace(_C6, q_boxes=True),
    "cum6_q_scores_also": replace(_C6, q_scores=True),
    "cum6_q_scores_not_logits": replace(_C6, q_scores=True, q_class_logits=False),
    "cum6_fp32_head": replace(_C6, **HEAD_FP32),
    "cum6_fp32_box_logits": replace(_C6, q_box_logits=False),
    "cum6_fp32_class_logits": replace(_C6, q_class_logits=False),
    "cum6_v1_head": replace(_C6, q_projection=True, q_dfl=True, q_boxes=True, q_scores=True, q_joins=True),
    # Residual account (diagnostics).
    "cum6_no_input": replace(_C6, quantize_input=False),
    "cum6_weights_only": replace(_C6, quantize_activations=False),
    "cum6_activations_only": replace(_C6, quantize_weights=False),
    "cum6_signed": replace(_C6, unsigned=False),
    "cum6_act_maxabs": replace(_C6, activation_range="maxabs"),
    "cum6_weight_maxabs": replace(_C6, weight_range="maxabs"),
    "cum6_no_bias_correction": _C5,
    "cum6_no_passthrough": replace(_C6, passthrough="none"),
}

NAMED = {"v1_maxabs": V1_MAXABS, "v1_percentile_99_9": V1_PERCENTILE, **ONE, **CUMULATIVE}


def named(name):
    if name not in NAMED:
        raise ValueError(f"unknown detector recipe name: {name}")
    return NAMED[name]


def register(name, recipe):
    if name in NAMED and NAMED[name] != recipe:
        raise ValueError(f"recipe name already bound to different switches: {name}")
    NAMED[name] = recipe


def hardware_semantics(recipe):
    """What a hardware cost model must follow for each switch value (text)."""
    head = []
    for flag, text in (
            ("q_projection", "the constant 0..15 of the box expectation is stored as weight codes with one scale"),
            ("q_box_logits", "the 64-channel box-distribution logits are stored as k-bit codes (network output)"),
            ("q_dfl", "the softmax expectation (distance in bins) is requantized to k-bit codes"),
            ("q_boxes", "decoded stride-scaled xywh pixel coordinates are requantized to k-bit codes"),
            ("q_class_logits", "the 80-channel class logits are stored as k-bit codes (network output)"),
            ("q_scores", "the sigmoid output is requantized to k-bit codes"),
            ("q_joins", "the 84-channel joins requantize boxes and scores with one scale per group")):
        head.append(text if getattr(recipe, flag) else f"NOT quantized ({flag}=false): "
                    + {"q_projection": "exact integers 0..15 in the postprocessor",
                       "q_box_logits": "box logits leave the accelerator as wide values",
                       "q_dfl": "softmax and expectation run in the FP32 postprocessor",
                       "q_boxes": "anchor decode and stride scaling run in the FP32 postprocessor; no k-bit box store",
                       "q_class_logits": "class logits leave the accelerator as wide values",
                       "q_scores": "the sigmoid runs in the FP32 postprocessor on dequantized logits",
                       "q_joins": "no 84-channel k-bit store; the postprocessor reads boxes and scores"}[flag])
    return {
        "requantization_points": {
            "v1": "every convolution output is requantized and every SiLU is a k-bit code-to-code lookup followed "
                  "by its own scale (two requantizations per conv-SiLU pair)",
            "fused_silu": "a convolution feeding only SiLU keeps the wide accumulator; SiLU is evaluated on the wide "
                          "value and requantized once (a wide-precision nonlinearity or a large table is required)",
        }[recipe.boundaries],
        "passthrough": {
            "none": "channel slice, max-pool, upsample and concatenation outputs are requantized with their own scale",
            "nonarith": "channel slice, max-pool and nearest upsample forward codes and the scale of their source "
                        "(no requantizer); a concatenation requantizes its inputs to one output scale",
            "nonarith_concat": "as nonarith, and a concatenation forwards the codes of its inputs: the consuming "
                               "convolution receives operand groups with different activation scales, so the scale "
                               "must be applied per group (per-group partial sums or per-group requantization shift)",
        }[recipe.passthrough],
        "signedness": ("non-negative boundaries (input image, sigmoid output, box expectation) store unsigned integer "
                       "codes; the first convolution has an unsigned k-bit activation operand"
                       if recipe.unsigned else "all codes signed as in the accepted manifests"),
        "activation_scale_metadata": "one FP32 scale per quantizing node (" + recipe.activation_range + ")",
        "weight_scale_metadata": "one FP32 scale per output channel (" + recipe.weight_range + ")",
        "bias_constants": ("replaced offline by corrected FP32 values; no new storage or operator"
                           if recipe.bias_correction == "empirical" else "folded FP32 bias unchanged"),
        "head": head,
        "diagnostic": [k for k in ("quantize_input", "quantize_weights", "quantize_activations") if not getattr(recipe, k)],
    }
