"""Exact finite-product grid and prefix headroom checks for fixed accumulators."""
from collections import Counter
from fractions import Fraction
from math import gcd, lcm, prod

from public.inference.reference.arithmetic import format_named, real, round_integer
from public.inference.tensor import Encoding, parse_encoding


def quantum(values):
    values = [Fraction(v) for v in values]
    denominator = lcm(*(v.denominator for v in values))
    numerator = gcd(*(abs(v.numerator)*(denominator//v.denominator) for v in values))
    return Fraction(numerator, denominator)


def fixed_mac_bounds(graph):
    domains, records = dict(graph["inputs"]), []
    for node in graph["nodes"]:
        attrs = node["attrs"]
        domain = domains[node["inputs"][0]]
        domains[node["name"]] = attrs.get("output", domain)
        if node["op"] not in {"linear", "conv2d", "depthwise_conv2d"}:
            continue
        row = {"node": node["name"], "status": "pending"}
        acc = format_named(attrs["accumulator"])
        weight = graph["constants"][node["inputs"][1]]
        x, w = parse_encoding(domain), parse_encoding(weight["encoding"])
        if (acc.family != "fixed_point" or acc.manifest["rounding"] != "rne"
                or any(not isinstance(e, Encoding) or e.block_size is not None for e in (x, w))
                or x.axis is not None or w.axis not in (None, 0)):
            row["reason"] = "requires RNE fixed accumulator and ordinary scalar/per-channel operands"
            records.append(row)
            continue
        xf, wf = format_named(x.format), format_named(w.format)
        finite_x = [Fraction(value)*x.scales[0] for c in range(1 << xf.bits) if (value := xf.decode(c)).is_finite()]
        decoded_w = {c: Fraction(value) for c in set(weight["codes"]) if (value := wf.decode(c)).is_finite()}
        if len(decoded_w) != len(set(weight["codes"])):
            row["reason"] = "encoded weights contain nonfinite values"
            records.append(row)
            continue
        step = Fraction(1, 1 << acc.manifest["numeric"]["fractional_bits"])
        x_quantum, maximum_x = quantum(finite_x), max(abs(v) for v in finite_x)
        k, channels = prod(weight["shape"][1:]), weight["shape"][0]
        if len(weight["codes"]) != channels*k or ("bias" in attrs and len(attrs["bias"]) != channels):
            raise ValueError("fixed MAC shape/payload mismatch")
        limit = (1 << (acc.bits-1))-1
        checks = []
        for channel in range(channels):
            counts = Counter(weight["codes"][channel*k:(channel+1)*k])
            scale = w.scales[channel if w.axis == 0 else 0]
            exact_products = all((x_quantum*decoded_w[c]*scale/step).denominator == 1 for c in counts)
            dot = maximum_x*sum((abs(decoded_w[c])*n for c,n in counts.items()), Fraction(0))*scale/step
            bias = real(attrs["bias"][channel]) if "bias" in attrs else Fraction(0)
            raw_bias = round_integer(Fraction(bias)/step)
            headroom = Fraction(limit)-dot-abs(raw_bias)
            checks.append({"channel": channel, "products_on_grid": exact_products,
                           "maximum_absolute_dot_codes": str(dot), "stored_bias_codes_before_saturation": raw_bias,
                           "remaining_headroom_codes": str(headroom), "overflow_excluded": headroom >= 0,
                           "bias_storage_error": str(abs(Fraction(raw_bias)*step-bias))})
        safe = all(c["products_on_grid"] and c["overflow_excluded"] for c in checks)
        row.update(status="exact_products_and_sums_after_bias_store" if safe else "pending",
                   k=k, accumulator=attrs["accumulator"], accumulator_step=str(step),
                   activation_quantum=str(x_quantum), channels=checks)
        records.append(row)
    return {"status": "local_fixed_mac_bounds_only", "records": records,
            "pending_nodes": [r["node"] for r in records if r["status"] == "pending"],
            "limits": ["all finite activation codes and actual retained weights; NaR/nonfinite activations excluded",
                       "absolute product-sum bound covers every sequential prefix including padding",
                       "bias is rounded once to the prescribed fixed grid; its error is reported separately",
                       "non-MAC rounding, native image conformance and whole-graph acceptance are outside this proof"]}
