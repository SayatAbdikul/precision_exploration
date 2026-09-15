"""Local finite-posit proofs for non-MAC operations under a fixed accumulator."""
from fractions import Fraction
from functools import lru_cache
from math import prod

from public.analysis.phase3.fixed_mac_bounds import quantum
from public.inference.reference.arithmetic import format_named
from public.inference.reference.operators import activation
from public.inference.tensor import Encoding, Tensor, parse_encoding


@lru_cache(maxsize=32)
def finite_values(encoding):
    fmt = format_named(encoding.format)
    return tuple(sorted({Fraction(v)*encoding.scales[0] for c in range(1 << fmt.bits)
                         if (v := fmt.decode(c)).is_finite()}))


def mean_thresholds(values, count, output_values, accumulator_step):
    """Test a conservative sum lattice, explicitly distinguishing exact ties."""
    if type(count) is not int or count < 1 or accumulator_step <= 0:
        raise ValueError("invalid fixed mean domain")
    q = quantum(values)
    lo, hi = count*min(values)/q, count*max(values)/q
    if lo.denominator != 1 or hi.denominator != 1:
        raise ValueError("sum interval is not on its declared lattice")
    lo, hi = int(lo), int(hi)
    step = q/count
    margin, ties, preserved = None, 0, True
    for a, b in zip(output_values, output_values[1:]):
        threshold = (a+b)/2
        position = threshold/step
        floor = position.numerator//position.denominator
        # Neighbors on both sides of a possible exact tie must be considered.
        for integer in {max(lo, min(hi, i)) for i in (floor-1, floor, floor+1, floor+2)}:
            distance = abs(integer*step-threshold)
            if distance == 0:
                ties += 1
                preserved &= (threshold/accumulator_step).denominator == 1
            else:
                margin = distance if margin is None else min(margin, distance)
    return {"minimum_nonzero_margin": margin, "exact_ties": ties, "ties_on_accumulator_grid": preserved}


@lru_cache(maxsize=64)
def hard_codes(source, output, function, accumulator):
    fmt = format_named(source.format)
    codes = tuple(c for c in range(1 << fmt.bits) if fmt.decode(c).is_finite())
    inputs = Tensor((len(codes),), codes, source)
    exact = []
    for value in inputs.values():
        clipped = min(max(Fraction(value)+3, Fraction(0)), Fraction(6))
        exact.append((Fraction(value)*clipped if function == "hard_swish" else clipped)/6)
    target = Tensor.quantize(exact, inputs.shape, output)
    actual = activation(inputs, function=function, output=output, accumulator=accumulator)
    mismatches = [i for i, (a,b) in enumerate(zip(actual.codes, target.codes)) if a != b]
    return {"finite_codes_tested": len(codes), "mismatches": len(mismatches),
            "examples": [{"input_code": codes[i], "actual_output_code": actual.codes[i], "exact_output_code": target.codes[i]}
                         for i in mismatches[:8]]}


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
        x, output = parse_encoding(domain), parse_encoding(domains[node["name"]])
        if any(not isinstance(e, Encoding) or e.axis is not None or format_named(e.format).family != "posit" for e in (x, output)):
            row["reason"] = "proof requires scalar finite posit operand domains"
        elif (op in {"flatten", "reshape", "broadcast_multiply", "channel_slice", "concatenate", "resize_nearest", "lut"}
                or op == "pool2d" and attrs["kind"] == "max"
                or op == "activation" and attrs["function"] in {"identity", "relu", "relu6"}):
            row.update(status="no_accumulator_rounding", reason="movement, prescribed LUT/output store, or exact arithmetic")
        elif "accumulator" not in attrs or format_named(attrs["accumulator"]).family != "fixed_point":
            row["reason"] = "operator does not declare a fixed accumulator"
        else:
            acc = format_named(attrs["accumulator"])
            step = Fraction(1, 1 << acc.manifest["numeric"]["fractional_bits"])
            limit = ((1 << (acc.bits-1))-1)*step
            values = finite_values(x)
            maximum, q = max(abs(v) for v in values), quantum(values)
            if op == "elementwise" and attrs["operation"] == "add":
                alignment = parse_encoding(attrs["alignment"])
                if not isinstance(alignment, Encoding) or alignment.axis is not None or format_named(alignment.format).family != "posit":
                    row["reason"] = "alignment is outside scalar posit proof"
                else:
                    aligned = finite_values(alignment)
                    on_grid = (quantum(aligned)/step).denominator == 1
                    headroom = limit-2*max(abs(v) for v in aligned)
                    row.update(status="exact_output_codes" if on_grid and headroom >= 0 else "pending",
                               sums_on_grid=on_grid, headroom=str(headroom))
            elif op == "adaptive_average_pool2d":
                if attrs.get("output_size", 1) not in (1, [1, 1], (1, 1)) or source not in shapes or len(shapes[source]) != 4:
                    row["reason"] = "global mean needs a fixed observed NCHW shape"
                else:
                    count = prod(shapes[source][-2:])
                    sums_exact = (q/step).denominator == 1 and count*maximum <= limit
                    margins = mean_thresholds(values, count, finite_values(output), step)
                    margin = margins["minimum_nonzero_margin"]
                    safe = sums_exact and margins["ties_on_accumulator_grid"] and margin is not None and margin > step/2
                    row.update(status="exact_output_codes" if safe else "pending", reduction_length=count,
                               sum_exact_and_overflow_excluded=sums_exact, final_rounding_bound=str(step/2),
                               minimum_nonzero_margin=str(margin), exact_ties=margins["exact_ties"],
                               ties_on_accumulator_grid=margins["ties_on_accumulator_grid"])
            elif op == "activation" and attrs["function"] in {"hard_swish", "hard_sigmoid"}:
                result = hard_codes(x, output, attrs["function"], attrs["accumulator"])
                row.update(status="exact_output_codes" if result["mismatches"] == 0 else "pending", evidence=result)
            elif op == "decode_boxes":
                if source not in shapes or len(shapes[source]) != 4 or shapes[source][1] != 4:
                    row["reason"] = "box decode needs fixed N x 4 x H x W shape"
                else:
                    stride = attrs["stride"]
                    if type(stride) is not int or stride < 1:
                        raise ValueError("box stride must be a positive integer")
                    anchor = Fraction(2*max(shapes[source][-2:])-1, 2)
                    # Includes both endpoints, their sum/difference, center half,
                    # and stride multiplication; all are dyadic on this grid.
                    on_grid = (q/(2*step)).denominator == 1 and (Fraction(1, 2)/step).denominator == 1
                    bound = 2*(anchor+maximum)*stride
                    row.update(status="exact_output_codes" if on_grid and bound <= limit else "pending",
                               intermediate_values_on_grid=on_grid, maximum_intermediate_bound=str(bound), headroom=str(limit-bound))
            else:
                row["reason"] = "operator needs a separate precision proof"
        records.append(row)
    return {"status": "local_fixed_nonmac_checks_only", "records": records,
            "pending_nodes": [r["node"] for r in records if r["status"] == "pending"],
            "limits": ["conditional on identical finite stored inputs, prescribed alignment/LUTs and fixed observed shapes",
                       "pool proof uses a superset of possible sums; exact threshold ties must be on the accumulator grid",
                       "hard activations enumerate every finite input code and compare exact rational output stores",
                       "does not accept nonfinite/NaR behavior, stored-bias error, native conformance or a whole graph"]}
