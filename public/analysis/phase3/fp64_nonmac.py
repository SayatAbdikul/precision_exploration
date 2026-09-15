"""Local output-boundary proofs for FP64 integer-domain residuals and pooling."""
from fractions import Fraction
from math import prod

from public.analysis.phase3.accumulator_probes import hard_activation_probe
from public.analysis.phase3.box_bounds import prove_boxes
from public.analysis.phase3.fp64_bounds import error_bound, UNIT_ROUNDOFF, HALF_MIN_SUBNORMAL, MAX_FINITE
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, parse_encoding


def mean_boundary_margin(input_scale, count, output_scale, input_bits, output_bits):
    """Smallest distance between any possible integer sum's mean and a store tie.

    The possible integer-sum interval is conservative: it includes every sum
    from `count` input codes. For each output threshold, the closest possible
    sum is its integer floor or ceiling, clipped to that interval.
    """
    if type(count) is not int or count < 1 or input_scale <= 0 or output_scale <= 0:
        raise ValueError("invalid integer mean domain")
    lo, hi = -count*(1 << (input_bits-1)), count*((1 << (input_bits-1))-1)
    step = Fraction(input_scale, count)
    minimum = None
    for code in range(-(1 << (output_bits-1)), (1 << (output_bits-1))-1):
        threshold = (Fraction(code)+Fraction(1, 2))*output_scale
        at = threshold/step
        floor = at.numerator//at.denominator
        for total in (max(lo, min(hi, floor)), max(lo, min(hi, floor+1))):
            distance = abs(total*step-threshold)
            minimum = distance if minimum is None else min(minimum, distance)
    return minimum


def prove_nonmac(graph, shapes):
    domains, records = dict(graph["inputs"]), []
    for node in graph["nodes"]:
        attrs, op = node["attrs"], node["op"]
        source = node["inputs"][0]
        domain = domains[source]
        domains[node["name"]] = attrs.get("output", domain)
        if op in {"linear", "conv2d", "depthwise_conv2d", "block_conv2d"}:
            continue
        row = {"node": node["name"], "op": op, "status": "pending"}
        output = parse_encoding(domains[node["name"]])
        x = parse_encoding(domain)
        ordinary = all(isinstance(e, Encoding) and e.axis is None and format_named(e.format).family == "integer" for e in (x, output))
        if not ordinary:
            row["reason"] = "proof only covers scalar integer operand domains"
        elif (op in {"flatten", "reshape", "broadcast_multiply", "channel_slice", "concatenate", "resize_nearest", "lut"}
                or op == "pool2d" and attrs["kind"] == "max"
                or op == "activation" and attrs["function"] in {"identity", "relu", "relu6"}):
            row.update(status="no_accumulator_rounding", reason="this operator uses exact movement/arithmetic or the prescribed output/LUT store")
        elif attrs.get("accumulator") != "fp64_e11m52_accumulator":
            row["reason"] = "operator does not use the FP64 policy"
        elif op == "activation" and attrs["function"] in {"hard_swish", "hard_sigmoid"}:
            probe = hard_activation_probe(domain, attrs["output"], attrs["function"], attrs["accumulator"])
            matches = not probe["requires_precision_revision"]
            row.update(status="exact_output_codes" if matches else "pending", evidence=probe)
        elif op == "elementwise" and attrs["operation"] == "add":
            alignment = parse_encoding(attrs["alignment"])
            if alignment != output:
                row["reason"] = "residual alignment and output are different grids"
            else:
                maximum = (1 << format_named(output.format).bits)*output.scales[0]
                bound = error_bound(maximum, 1)
                margin = output.scales[0]/2  # Exact aligned sum is itself an integer code value.
                safe = bound["overflow_excluded"] and bound["absolute_error_bound"] < margin
                row.update(status="exact_output_codes" if safe else "pending",
                           absolute_error_bound=str(bound["absolute_error_bound"]), boundary_margin=str(margin),
                           overflow_excluded=bound["overflow_excluded"])
        elif op == "decode_boxes":
            if source not in shapes or len(shapes[source]) != 4 or shapes[source][1] != 4:
                row["reason"] = "box decoding needs its fixed observed N x 4 x H x W shape"
            else:
                row.update(prove_boxes(x.scales[0], output.scales[0], format_named(x.format).bits,
                                       format_named(output.format).bits, *shapes[source][-2:], attrs["stride"]))
        elif op == "adaptive_average_pool2d":
            size = attrs.get("output_size", 1)
            if size not in (1, [1, 1], (1, 1)) or source not in shapes or len(shapes[source]) != 4:
                row["reason"] = "global pooling needs its fixed observed NCHW shape"
            else:
                count = prod(shapes[source][-2:])
                bits, output_bits = format_named(x.format).bits, format_named(output.format).bits
                maximum = (1 << (bits-1))*x.scales[0]
                summed = error_bound(count*maximum, count)
                divided_error = summed["absolute_error_bound"]/count
                error = divided_error+UNIT_ROUNDOFF*(maximum+divided_error)+HALF_MIN_SUBNORMAL
                overflow = summed["overflow_excluded"] and maximum+error <= MAX_FINITE
                margin = mean_boundary_margin(x.scales[0], count, output.scales[0], bits, output_bits)
                safe = overflow and error < margin
                row.update(status="exact_output_codes" if safe else "pending", reduction_length=count,
                           absolute_error_bound=str(error), boundary_margin=str(margin), overflow_excluded=overflow,
                           reason="all possible integer-sum means stay farther from output thresholds than the FP64 error bound" if safe
                           else "an output threshold may be crossed; ordered native sensitivity evidence is still required")
        else:
            row["reason"] = "operator needs a separate precision proof"
        records.append(row)
    return {"status": "local_nonmac_checks_only", "records": records,
            "pending_nodes": [r["node"] for r in records if r["status"] == "pending"],
            "limits": ["conditional on unchanged stored inputs and the declared fixed shapes",
                       "global mean proof includes ordered FP64 summation and the final rounded division",
                       "output-code equality includes saturation; exact ties are conservatively left pending",
                       "does not accept MAC rounding, native conformance, or a whole workload graph"]}
