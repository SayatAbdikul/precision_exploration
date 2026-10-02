"""B2 recipe switches, the named arms of the diagnosis and their hardware meaning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

BOUNDARIES = ("v1", "fused_relu", "fused_all")
RANGES = ("maxabs", "percentile_99_9", "mse")
BIAS = ("none", "empirical")
EQUALIZATION = ("none", "cle")


@dataclass(frozen=True)
class Recipe:
    boundaries: str = "v1"
    unsigned: bool = False
    activation_range: str = "maxabs"
    weight_range: str = "maxabs"
    bias_correction: str = "none"
    equalization: str = "none"
    quantize_input: bool = True
    quantize_logits: bool = True

    def __post_init__(self):
        if (self.boundaries not in BOUNDARIES or self.activation_range not in RANGES
                or self.weight_range not in RANGES or self.bias_correction not in BIAS
                or self.equalization not in EQUALIZATION):
            raise ValueError(f"invalid B2 recipe: {self}")
        for key in ("unsigned", "quantize_input", "quantize_logits"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"B2 recipe switch {key} must be a bool")

    def as_dict(self):
        return asdict(self)

    def v1_equivalent(self):
        """Name of the v1 recipe this reproduces bit for bit, or None."""
        if (self.boundaries != "v1" or self.unsigned or self.bias_correction != "none"
                or self.equalization != "none" or not self.quantize_input or not self.quantize_logits):
            return None
        if self.activation_range == self.weight_range and self.activation_range in ("maxabs", "percentile_99_9"):
            return self.activation_range
        return None


V1_MAXABS = Recipe()
V1_PERCENTILE = Recipe(activation_range="percentile_99_9", weight_range="percentile_99_9")

# One-at-a-time arms: each changes exactly one switch of v1_maxabs.
ONE_AT_A_TIME = {
    "a_fused_relu": replace(V1_MAXABS, boundaries="fused_relu"),
    "a_fused_all": replace(V1_MAXABS, boundaries="fused_all"),
    "b_unsigned": replace(V1_MAXABS, unsigned=True),
    "c_act_mse": replace(V1_MAXABS, activation_range="mse"),
    "c_act_percentile": replace(V1_MAXABS, activation_range="percentile_99_9"),
    "d_weight_mse": replace(V1_MAXABS, weight_range="mse"),
    "d_weight_percentile": replace(V1_MAXABS, weight_range="percentile_99_9"),
    "e_bias_correction": replace(V1_MAXABS, bias_correction="empirical"),
    "g_no_logits": replace(V1_MAXABS, quantize_logits=False),
    "g_no_input": replace(V1_MAXABS, quantize_input=False),
    "g_no_input_no_logits": replace(V1_MAXABS, quantize_input=False, quantize_logits=False),
    # Direct test of the logit-clipping hypothesis: the v1 percentile recipe with FP32 logits.
    "g_percentile_no_logits": replace(V1_PERCENTILE, quantize_logits=False),
}

# Cumulative ladder in the order (a) (b) (c) (d) (e).
_C1 = replace(V1_MAXABS, boundaries="fused_relu")
_C2 = replace(_C1, unsigned=True)
_C3 = replace(_C2, activation_range="mse")
_C4 = replace(_C3, weight_range="mse")
_C5 = replace(_C4, bias_correction="empirical")
CUMULATIVE = {
    "cum1_fused": _C1,
    "cum2_unsigned": _C2,
    "cum3_act_mse": _C3,
    "cum4_weight_mse": _C4,
    "cum5_bias_correction": _C5,
    # Same ladder with every nonlinearity fused (Hardswish/Hardsigmoid evaluated before requantization).
    "cum3_act_mse_fused_all": replace(_C3, boundaries="fused_all"),
    "cum5_fused_all": replace(_C5, boundaries="fused_all"),
    # Leave-one-out arms around the full ladder.
    "cum5_signed": replace(_C5, unsigned=False),
    "cum5_weight_maxabs": replace(_C5, weight_range="maxabs"),
    "cum5_act_maxabs": replace(_C5, activation_range="maxabs"),
    "cum5_v1_boundaries": replace(_C5, boundaries="v1"),
    "cum3_plus_bias_correction": replace(_C3, bias_correction="empirical"),
    # Diagnostic exemptions on top of the ladder.
    "cum5_no_logits": replace(_C5, quantize_logits=False),
    "cum5_no_input_no_logits": replace(_C5, quantize_input=False, quantize_logits=False),
    # Cross-layer equalization, only examined if the ladder leaves a gap.
    "cum5_cle": replace(_C5, equalization="cle"),
    "cum4_cle": replace(_C4, equalization="cle"),
}

NAMED = {"v1_maxabs": V1_MAXABS, "v1_percentile_99_9": V1_PERCENTILE, **ONE_AT_A_TIME, **CUMULATIVE}


def named(name):
    if name not in NAMED:
        raise ValueError(f"unknown B2 recipe name: {name}")
    return NAMED[name]


def register(name, recipe):
    """Frozen recipes (default and ablations) are registered by ``frozen.py``."""
    if name in NAMED and NAMED[name] != recipe:
        raise ValueError(f"recipe name already bound to different switches: {name}")
    NAMED[name] = recipe


def hardware_semantics(recipe):
    """What a hardware cost model must change to follow this recipe (text, per switch)."""
    return {
        "requantization_points": {
            "v1": "every conv/linear/add/mul/pool output is requantized with its own scale; each "
                  "activation output is requantized again",
            "fused_relu": "conv/linear/add/mul feeding only ReLU/ReLU6 keep the wide accumulator through the "
                          "clamp and are requantized once after it; max-pool forwards codes unchanged; "
                          "Hardswish/Hardsigmoid still take a requantized input and requantize their output",
            "fused_all": "as fused_relu, and Hardswish/Hardsigmoid are evaluated on the wide accumulator value "
                         "before the single requantization (a wide-precision nonlinearity or LUT is required)",
        }[recipe.boundaries],
        "signedness": ("non-negative boundaries store unsigned integer codes (integer formats only): the "
                       "activation operand of the next MAC is unsigned k-bit, weights stay signed"
                       if recipe.unsigned else "all codes signed as in the accepted manifests"),
        "activation_scale_metadata": {
            "maxabs": "one FP32 scale per quantizing node, from the calibration maximum",
            "percentile_99_9": "one FP32 scale per quantizing node, from the 99.9th percentile",
            "mse": "one FP32 scale per quantizing node, from an offline scale search; same storage as v1",
        }[recipe.activation_range],
        "weight_scale_metadata": {
            "maxabs": "one FP32 scale per output channel",
            "percentile_99_9": "one FP32 scale per output channel; weights beyond it are clipped",
            "mse": "one FP32 scale per output channel from an offline search; same storage as v1",
        }[recipe.weight_range],
        "bias_constants": ("bias constants are replaced offline by corrected values; no new storage or operator"
                           if recipe.bias_correction == "empirical" else "folded FP32 bias unchanged"),
        "equalization": ("weights and biases of adjacent layers are rescaled per channel offline; no run-time cost"
                         if recipe.equalization == "cle" else "none"),
        "input": "input image quantized" if recipe.quantize_input else "DIAGNOSTIC: FP32 input (not a hardware recipe)",
        "logits": "logits quantized" if recipe.quantize_logits else "DIAGNOSTIC: FP32 logits (wide accumulator read out)",
    }
