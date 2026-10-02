"""Check and residual arms around the frozen recipe (protocol addendum 2), registered after the freeze.

None of them is a candidate recipe: the frozen ``default`` is not changed by anything measured here.
A separate module so that ``recipe.py`` (a numeric source) and every sealed configuration identity stay unchanged.
"""
from __future__ import annotations

from dataclasses import replace

from .frozen import DEFAULT
from .recipe import register

ARMS3 = {
    # The join exemption on its own: the 84-channel joins store the (already quantized) scores again.
    "default_q_joins": replace(DEFAULT, q_joins=True),
    # Residual account: which network output costs what when it leaves the accelerator as k-bit codes.
    "default_fp32_box_logits": replace(DEFAULT, q_box_logits=False),
    "default_fp32_class_logits": replace(DEFAULT, q_class_logits=False),
}
for _name, _recipe in ARMS3.items():
    register(_name, _recipe)
