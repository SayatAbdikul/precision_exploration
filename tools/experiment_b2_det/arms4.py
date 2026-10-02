"""The recipe the written freeze rule prescribes (protocol addendum 3), registered after review 1.

Protocol v1 lists the box-distribution logits as a head flag and adopts a head exemption only if the recipe
without it is worse on dev128 by more than 0.5 point with an interval excluding zero; a tensor whose quantization
does not meet that test stays quantized.  On the base cum5 (addendum 1) the box-logit exemption met the test
(+0.56 [+0.21, +0.87]) and the join exemption was never tested before the freeze.  The frozen ``default`` keeps
the box logits quantized and the joins unquantized; ``conformant`` follows the rule.  It is not a re-freeze: the
owner decides which recipe the paper uses.

Block formats define a join store only for the whole 84-channel tensor (``engine.analyze`` raises otherwise), so
for bfp6 and mxfp8_e4m3 the conformant recipe is the registered arm ``default_fp32_box_logits``
(``CONFORMANT_ARM``).  A separate module so that ``recipe.py`` (a numeric source) and every sealed configuration
identity stay unchanged.
"""
from __future__ import annotations

from dataclasses import replace

from .engine import BLOCK_FORMATS
from .frozen import DEFAULT
from .recipe import register
from . import arms3  # noqa: F401  (registers default_fp32_box_logits)

ARMS4 = {
    "conformant": replace(DEFAULT, q_box_logits=False, q_joins=True),
}
for _name, _recipe in ARMS4.items():
    register(_name, _recipe)


def conformant_arm(format_name):
    """Registered arm name of the protocol-conformant recipe for one format."""
    return "default_fp32_box_logits" if format_name in BLOCK_FORMATS else "conformant"
