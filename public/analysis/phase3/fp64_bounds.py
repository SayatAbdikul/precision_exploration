"""Conservative local FP64 Model C bounds for scalar-mapped integer MACs.

These bounds compare a layer fed the same stored operands with its exact-real
dot product and original real bias. They do not bound an entire network's
quality, prove output-code equality at rounding thresholds, or accept a graph.
"""
from fractions import Fraction
from math import prod

from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import parse_encoding, SharedEncoding

UNIT_ROUNDOFF = Fraction(1, 2**53)
HALF_MIN_SUBNORMAL = Fraction(1, 2**1075)
MAX_FINITE = Fraction((2**53-1)*2**971)


def error_bound(sum_absolute_products, k, *, absolute_bias=None):
    total = Fraction(sum_absolute_products)
    if type(k) is not int or k < 1 or total < 0:
        raise ValueError("FP64 bound requires a positive reduction length and nonnegative magnitude")
    n = k
    if absolute_bias is not None:
        bias = Fraction(absolute_bias)
        if bias < 0:
            raise ValueError("absolute bias must be nonnegative")
        total += bias
        n += 2  # Store bias, then add it to the final reduction state.
    denominator = 1-n*UNIT_ROUNDOFF
    if denominator <= 0:
        raise ValueError("FP64 rounding bound requires n*u < 1")
    error = (n*UNIT_ROUNDOFF*total+n*HALF_MIN_SUBNORMAL)/denominator
    return {"roundings_bound": n, "sum_magnitude_bound": total, "absolute_error_bound": error,
            "overflow_excluded": total+error <= MAX_FINITE}


def integer_mac_bounds(graph):
    domains, records, pending = dict(graph["inputs"]), [], []
    for node in graph["nodes"]:
        attrs = node["attrs"]
        domain = domains[node["inputs"][0]]
        domains[node["name"]] = attrs.get("output", domain)
        if node["op"] not in {"linear", "conv2d", "depthwise_conv2d"}:
            continue
        if attrs["accumulator"] != "fp64_e11m52_accumulator":
            pending.append({"node": node["name"], "reason": "MAC does not use FP64"})
            continue
        weight = graph["constants"][node["inputs"][1]]
        x, w, output = (parse_encoding(value) for value in (domain, weight["encoding"], attrs["output"]))
        if (any(isinstance(e, SharedEncoding) for e in (x, w, output)) or x.axis is not None
                or w.axis not in {None, 0} or output.axis is not None
                or any(format_named(e.format).family != "integer" for e in (x, w, output))):
            pending.append({"node": node["name"], "reason": "bound requires scalar-mapped integer activations and scalar/per-output-channel integer weights"})
            continue
        xf, wf = format_named(x.format), format_named(w.format)
        x_max = max(abs(Fraction(xf.decode(c))) for c in range(1 << xf.bits))*x.scales[0]
        raw_magnitudes = [abs(int(wf.decode(c))) for c in range(1 << wf.bits)]
        k, channels = prod(weight["shape"][1:]), weight["shape"][0]
        if len(weight["codes"]) != k*channels:
            raise ValueError("MAC bound weight shape/payload mismatch")
        if "bias" in attrs and len(attrs["bias"]) != channels:
            raise ValueError("MAC bound bias channel count mismatch")
        bounds = []
        for channel in range(channels):
            magnitude = sum(raw_magnitudes[code] for code in weight["codes"][channel*k:(channel+1)*k])
            bias = abs(real(attrs["bias"][channel])) if "bias" in attrs else None
            bounds.append(error_bound(x_max*w.scales[channel if w.axis == 0 else 0]*magnitude, k, absolute_bias=bias))
        maximum = max(b["absolute_error_bound"] for b in bounds)
        ratio = maximum/output.scales[0]
        records.append({"node": node["name"], "k": k, "channels": channels,
                        "roundings_bound": bounds[0]["roundings_bound"],
                        "maximum_absolute_error_bound": str(maximum), "bound_in_output_steps": str(ratio),
                        "bound_in_output_steps_approx": float(ratio),
                        "overflow_excluded": all(b["overflow_excluded"] for b in bounds),
                        "below_half_output_step": ratio < Fraction(1, 2)})
    return {"status": "local_bounds_only_not_accepted", "macs": records, "pending": pending,
            "formula": "(n*u*(sum_abs_products+abs_bias) + n*2^-1075)/(1-n*u), u=2^-53; n=K plus two bias rounds when stored",
            "limits": ["one rounding after each exact multiply-add; no separate product rounding is assumed",
                       "magnitude bound includes every stored weight and worst finite activation code, including padded positions",
                       "subnormal absolute rounding error is included; overflow exclusion is checked separately",
                       "tiny absolute errors may still cross output-store thresholds; native sensitivity checks remain required",
                       "non-MAC operators and network-level error propagation are outside this local bound"]}
