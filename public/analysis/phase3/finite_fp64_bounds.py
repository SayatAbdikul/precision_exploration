"""Local FP64 error bounds over finite non-shared operand codebooks.

The reference is the oracle's declared finite values, including its retained
approximations for irrational formats. A spacing ratio is a scale comparison,
not proof that errors cannot cross an output rounding threshold.
"""
from collections import Counter
from fractions import Fraction
from math import prod

from public.analysis.phase3.fp64_bounds import error_bound
from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import Encoding, parse_encoding


def finite_codebook(encoding):
    fmt = format_named(encoding.format)
    return {code: Fraction(value) for code in range(1 << fmt.bits)
            if (value := fmt.decode(code)).is_finite()}


def finite_mac_bounds(graph):
    domains, records, pending = dict(graph["inputs"]), [], []
    for node in graph["nodes"]:
        attrs, domain = node["attrs"], domains[node["inputs"][0]]
        domains[node["name"]] = attrs.get("output", domain)
        if node["op"] == "block_conv2d":
            pending.append({"node": node["name"], "reason": "shared block MAC needs scale-aware bounds"})
            continue
        if node["op"] not in {"linear", "conv2d", "depthwise_conv2d"}:
            continue
        reason = None
        if attrs["accumulator"] != "fp64_e11m52_accumulator":
            reason = "MAC does not use FP64"
        weight = graph["constants"][node["inputs"][1]]
        x, w, out = (parse_encoding(e) for e in (domain, weight["encoding"], attrs["output"]))
        if (any(not isinstance(e, Encoding) or e.block_size is not None for e in (x, w, out))
                or x.axis is not None or w.axis not in (None, 0) or out.axis is not None):
            reason = "requires ordinary scalar activations/output and scalar or per-output-channel weights"
        if reason:
            pending.append({"node": node["name"], "reason": reason})
            continue
        x_values, w_values, out_values = (finite_codebook(e) for e in (x, w, out))
        if not set(weight["codes"]) <= w_values.keys():
            pending.append({"node": node["name"], "reason": "encoded weights contain nonfinite or invalid codes"})
            continue
        values = sorted(set(out_values.values()))
        if not x_values or len(values) < 2:
            pending.append({"node": node["name"], "reason": "finite input/output domain is empty or has no spacing"})
            continue
        minimum_spacing = min(b-a for a, b in zip(values, values[1:]))*out.scales[0]
        maximum_x = max(abs(v) for v in x_values.values())*x.scales[0]
        k, channels = prod(weight["shape"][1:]), weight["shape"][0]
        if len(weight["codes"]) != k*channels or ("bias" in attrs and len(attrs["bias"]) != channels):
            raise ValueError("finite FP64 bound weight/bias shape mismatch")
        w.validate_shape(tuple(weight["shape"]))
        bounds = []
        for channel in range(channels):
            counts = Counter(weight["codes"][channel*k:(channel+1)*k])
            magnitude = sum((abs(w_values[code])*count for code, count in counts.items()), Fraction(0))
            bias = abs(real(attrs["bias"][channel])) if "bias" in attrs else None
            bounds.append(error_bound(maximum_x*w.scales[channel if w.axis == 0 else 0]*magnitude,
                                      k, absolute_bias=bias))
        worst = max(range(channels), key=lambda c: bounds[c]["absolute_error_bound"])
        maximum = bounds[worst]["absolute_error_bound"]
        ratio = maximum/minimum_spacing
        records.append({"node": node["name"], "k": k, "channels": channels,
                        "roundings_bound": bounds[worst]["roundings_bound"], "worst_error_channel": worst,
                        "maximum_absolute_error_bound": str(maximum),
                        "minimum_finite_output_spacing": str(minimum_spacing),
                        "bound_in_minimum_output_spacings": str(ratio),
                        "bound_in_minimum_output_spacings_approx": float(ratio),
                        "overflow_excluded": all(b["overflow_excluded"] for b in bounds),
                        "below_half_minimum_output_spacing": ratio < Fraction(1, 2),
                        "finite_activation_codes": len(x_values),
                        "excluded_activation_codes": (1 << format_named(x.format).bits)-len(x_values)})
    return {"status": "finite_local_bounds_only_not_accepted", "macs": records, "pending": pending,
            "limits": ["same stored finite operands, exact multiply-add followed by one FP64 rounding per reduction term",
                       "actual weight magnitudes and worst finite activation cover all sequential prefixes including padding",
                       "prescribed bias store and final addition plus subnormal rounding are included",
                       "oracle finite values define the reference; irrational-format approximation error is separate",
                       "spacing comparison does not prove output-code stability or whole-graph accuracy",
                       "nonfinite activations, shared scales and non-MAC operators require separate evidence"]}
