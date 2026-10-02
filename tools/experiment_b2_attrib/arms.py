"""Arm names and their meaning.

    ref_default            the frozen B2 default (reproduces the sealed matrix cell)
    ref_weights_only       every activation boundary wide (weights and bias correction as the default)
    <term>+<term>+...      a combination of terms; a term is OP:TARGET

OP:
    wide     keep the TARGET boundaries wide (FP32), quantize everything else as the default
    only     quantize only the TARGET boundaries, keep every other boundary wide
    affine   affine (zero-point) code at the signed TARGET boundaries (search falls back to the symmetric
             default where it is not better on the calibration samples)
    pc       per-channel activation scales at the TARGET boundaries (never at the input or at linear outputs);
             conv/linear consumers get the scales folded into their weights before weight quantization

TARGET:
    all | signed | kind.<B2 kind> | role.<group> | position.<group> | node.<FX node name>

``wide``/``only`` arms are diagnostic (mixed precision is used only to locate error, never as a recipe);
``affine``/``pc`` arms without any wide term are uniform-precision recipe arms.
"""
from __future__ import annotations

from .engine import Arm

OPS = ("wide", "only", "affine", "pc")


def resolve(target, plan, groups):
    quantizing = [name for name, row in plan.items() if row["quantizes"]]
    if target == "all":
        return quantizing
    if target == "signed":
        return [n for n in quantizing if plan[n]["signedness"] == "signed"]
    scheme, _, rest = target.partition(".")
    if scheme == "kind":
        found = [n for n in quantizing if plan[n]["kind"] == rest]
    elif scheme in ("role", "position"):
        if rest not in groups[scheme]:
            raise ValueError(f"unknown {scheme} group: {rest}")
        found = list(groups[scheme][rest])
    elif scheme == "node":
        if rest not in quantizing:
            raise ValueError(f"not a quantizing boundary: {rest}")
        found = [rest]
    else:
        raise ValueError(f"unknown arm target: {target}")
    if not found:
        raise ValueError(f"empty arm target: {target}")
    return found


def parse_arm(name, plan, groups):
    if name == "ref_default":
        return Arm(name, kind="reference")
    if name == "ref_weights_only":
        return Arm(name, wide=tuple(resolve("all", plan, groups)), kind="reference")
    quantizing = set(resolve("all", plan, groups))
    wide, affine, per_channel = set(), set(), set()
    for term in name.split("+"):
        op, _, target = term.partition(":")
        if op not in OPS or not target:
            raise ValueError(f"malformed arm term: {term}")
        members = set(resolve(target, plan, groups))
        if op == "wide":
            wide |= members
        elif op == "only":
            wide |= quantizing - members
        elif op == "affine":
            affine |= {n for n in members if plan[n]["signedness"] == "signed"}
        else:
            per_channel |= {n for n in members if plan[n]["kind"] not in ("input", "linear")}
    if (affine or per_channel) & wide:
        raise ValueError(f"arm {name} changes the code of a boundary it keeps wide")
    if "affine" in name and not affine:
        raise ValueError(f"arm {name}: no signed boundary in the affine target")
    if "pc:" in name and not per_channel:
        raise ValueError(f"arm {name}: no eligible boundary in the per-channel target")
    kind = "diagnostic" if wide else "recipe"
    return Arm(name, wide=tuple(sorted(wide)), affine=tuple(sorted(affine)), per_channel=tuple(sorted(per_channel)),
               kind=kind)
