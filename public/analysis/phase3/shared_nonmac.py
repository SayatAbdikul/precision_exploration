"""Conditional shared-domain non-MAC coverage, including common-scale sums."""
from fractions import Fraction
from functools import lru_cache

from public.analysis.phase3.finite_fp64_nonmac import exact_grid
from public.analysis.phase3.fixed_mac_bounds import quantum
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import SharedEncoding, parse_encoding


@lru_cache(maxsize=4)
def aligned_sum_proof(name):
    fmt = format_named(name)
    if fmt.manifest["scaling"]["mode"] != "intrinsic_shared" or fmt.manifest["block"]["shared_scale_format"] != "e8m0":
        raise ValueError("common-scale proof requires intrinsic E8M0 operands")
    values = [Fraction(v) for c in range(1 << fmt.bits) if (v := fmt.decode(c)).is_finite()]
    q, maximum = quantum(values), max(abs(v) for v in values)
    checks = []
    for exponent in range(-127, 128):
        scale = Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**(-exponent))
        checks.append(exact_grid(q*scale, 2*maximum*scale))
    return {"status": "exact_output_codes" if all(checks) else "pending",
            "finite_operand_codes": len(values), "scales_checked": len(checks),
            "unscaled_quantum": str(q), "unscaled_maximum_absolute_sum": str(2*maximum),
            "common_scale_sums_exact_fp64": all(checks),
            "condition": "both branches have already been requantized to the same per-block E8M0 alignment scale"}


def prove_nonmac(graph):
    domains, records = dict(graph["inputs"]), []
    for node in graph["nodes"]:
        attrs, op = node["attrs"], node["op"]
        domain = domains[node["inputs"][0]]
        domains[node["name"]] = attrs.get("output", domain)
        if op in {"linear", "conv2d", "depthwise_conv2d", "block_conv2d"}:
            continue
        row = {"node": node["name"], "op": op, "status": "pending"}
        x, out = parse_encoding(domain), parse_encoding(domains[node["name"]])
        if not all(isinstance(e, SharedEncoding) for e in (x, out)):
            row["reason"] = "outside runtime shared input/output policy"
        elif (op in {"flatten", "broadcast_multiply", "channel_slice", "concatenate", "resize_nearest", "scaled_nonlinear_lut"}
                or op == "pool2d" and attrs["kind"] == "max"
                or op == "activation" and attrs["function"] in {"identity", "relu", "relu6"}):
            row.update(status="no_accumulator_rounding", reason="movement, exact arithmetic or prescribed LUT followed by intrinsic output storage")
        elif op == "elementwise" and attrs["operation"] == "add":
            alignment = parse_encoding(attrs["alignment"])
            if (attrs.get("accumulator") != "fp64_e11m52_accumulator" or not isinstance(alignment, SharedEncoding)
                    or x != alignment or out != alignment or any(domains[name] != domain for name in node["inputs"])):
                row["reason"] = "requires matching shared branch/alignment/output policies and FP64"
            else:
                row.update(aligned_sum_proof(alignment.format))
        else:
            row["reason"] = "variable-scale pooling, nonlinear arithmetic or box/DFL precision needs separate evidence"
        records.append(row)
    return {"status": "conditional_shared_nonmac_coverage_only", "records": records,
            "pending_nodes": [r["node"] for r in records if r["status"] == "pending"],
            "limits": ["finite stored operands and unchanged intrinsic scale selection/output stores",
                       "residual proof starts after both branches use the same common block scale; it does not compare unaligned sums",
                       "identical exact/FP64 sums feed identical output scale search, including its existing tie policy",
                       "does not prove convolution/MAC precision, independent-scale pooling, nonfinite reachability, native shapes or graph acceptance"]}
