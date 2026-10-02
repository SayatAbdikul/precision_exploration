"""Addendum-2 combination arms (CPU): per format, affine:signed + the best of the four pcf arms.

'best' = highest top-1 expected on the screen (ties by lower mean KL); prints queue lines for the low-memory runner.
The MobileNetV3 INT8 combination is repeated on MobileNetV2 and ResNet18 INT8 when its pcf target exists there.
"""
from __future__ import annotations

import sys

from . import analysis

PCF = ("pcf:role.dw_conv", "pcf:role.se_product", "pcf:kind.hardswish", "pcf:all")


def best_pcf(model, fmt):
    records = analysis.arm_records(model, fmt)
    found = [(records[a]["readout"]["top1_expected_percent"], -records[a]["kl_nats_mean"], a) for a in PCF if a in records]
    if len(found) < len(PCF):
        return None
    return max(found)[2]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    formats = argv[0].split(",") if argv else ["int8", "int6", "fp6_e2m3", "fp8_e4m3fn"]
    int8 = None
    for fmt in formats:
        best = best_pcf("mobilenet_v3_large", fmt)
        if best is None:
            print(f"# {fmt}: pcf arms incomplete", file=sys.stderr)
            continue
        if fmt == "int8":
            int8 = best
        print(f"arms --model mobilenet_v3_large --format {fmt} --stage B-r3 --audit --arms affine:signed+{best}")
    if int8 is not None and int8 in ("pcf:all",):
        for model in ("mobilenet_v2", "resnet18"):
            print(f"arms --model {model} --format int8 --stage B-reg --audit --arms affine:signed+{int8}")
    elif int8 is not None:
        print(f"# regression combination skipped: {int8} has no target in MobileNetV2/ResNet18", file=sys.stderr)


if __name__ == "__main__":
    main()
