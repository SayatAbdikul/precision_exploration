"""Finite unscaled dot-product bounds; evidence for later accumulator policies.

This is a conservative grid bound, not a claim about observed model error.
Bias and MX/BFP exponent composition need separate domain bounds.
"""
from __future__ import annotations

from fractions import Fraction
from functools import reduce
import json
from math import gcd, lcm
from pathlib import Path

from public.formats.oracle.manifest import manifest_sha256
from public.inference.reference.arithmetic import format_named, pow2

ROOT = Path(__file__).resolve().parents[2]


def bounds(name, k):
    if type(k) is not int or k < 1:
        raise ValueError("reduction length must be a positive integer")
    fmt = format_named(name)
    decoded = [fmt.decode(code) for code in range(1 << fmt.bits)]
    finite = [Fraction(value) for value in decoded if value.is_finite()]
    maximum = max(abs(value) for value in finite)
    denominator = lcm(*(value.denominator for value in finite))
    numerator = reduce(gcd, (int(abs(value)*denominator) for value in finite))
    quantum = Fraction(numerator, denominator)
    product_quantum = quantum**2
    magnitude = k*maximum**2
    coefficient_bound = magnitude/product_quantum
    integer_bound = (coefficient_bound.numerator + coefficient_bound.denominator - 1)//coefficient_bound.denominator
    dyadic = not denominator & (denominator-1)
    result = {"format": name, "manifest_sha256": manifest_sha256(fmt.manifest), "k": k,
              "finite_codes": len(finite), "special_codes_excluded": len(decoded)-len(finite),
              "maximum_operand_magnitude": str(maximum), "maximum_sum_magnitude": str(magnitude),
              "operand_grid_quantum": str(quantum), "product_grid_quantum": str(product_quantum),
              "dyadic": dyadic, "scaling_mode": fmt.manifest["scaling"]["mode"]}
    if dyadic:
        exponent = (product_quantum.numerator & -product_quantum.numerator).bit_length()-1-(product_quantum.denominator.bit_length()-1)
        binary_units = magnitude/pow2(exponent)
        precision = int(binary_units).bit_length()
        def covers(precision_bits, minimum, maximum_exp):
            return precision <= precision_bits and exponent >= minimum and magnitude <= (2-pow2(1-precision_bits))*pow2(maximum_exp)
        result.update(signed_grid_accumulator_bits=integer_bound.bit_length()+1,
                      sufficient_float_significand_bits=precision,
                      fp32_covers_finite_unscaled_product_grid=covers(24,-149,127),
                      fp64_covers_finite_unscaled_product_grid=covers(53,-1074,1023))
    else:
        result["native_binary_limitation"] = "oracle-decoded operand grid is not dyadic; no finite-width binary accumulator exactly preserves it"
    if fmt.manifest["scaling"]["mode"] == "intrinsic_shared":
        result["scope_limitation"] = "unit shared scale only; does not bound composition across E8M0 blocks"
    elif fmt.manifest["scaling"]["mode"] == "required_mapping":
        result["scope_limitation"] = "unit mapping scale only; numerical accumulator-domain conversion must be specified"
    return result


def main():
    names = sorted(path.stem for path in (ROOT/"public/formats/manifests/accepted").glob("*.json") if path.name != "index.json")
    records = [bounds(name,k) for name in names for k in (3,9,27,64,128,256,1024,4608)]
    report = {"schema_version": "2.0.0", "records": records,
              "scope": "conservative finite unscaled exact-dot grid bounds, homogeneous W=A, no bias; not an accepted Experiment A policy",
              "remaining": ["bias-domain bounds", "MX/BFP cross-block scaling", "family-specific wide accumulator resolution", "observed workload rounding/overflow error"]}
    (ROOT/"results/summaries/phase2-accumulator-bounds.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
