"""Local finite-codebook non-MAC checks for ordinary FP64 operand domains."""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement
from math import prod

from public.analysis.phase3.fixed_mac_bounds import quantum
from public.analysis.phase3.fixed_nonmac import finite_values, mean_thresholds
from public.analysis.phase3.fp64_bounds import UNIT_ROUNDOFF, HALF_MIN_SUBNORMAL, MAX_FINITE
from public.inference.reference.arithmetic import binary, format_named
from public.inference.reference.operators import activation, clamp
from public.inference.tensor import Encoding, Tensor, parse_encoding

ACCUMULATOR = "fp64_e11m52_accumulator"


def exact_grid(q, maximum):
    """Sufficient condition for every bounded lattice value to be exact FP64."""
    q, maximum = Fraction(q), Fraction(maximum)
    if q <= 0 or maximum < 0:
        raise ValueError("invalid finite FP64 lattice")
    if q.denominator & (q.denominator-1):
        return False
    step = Fraction(q.numerator & -q.numerator, q.denominator)
    return step >= Fraction(1, 2**1074) and maximum <= MAX_FINITE and maximum/step <= 2**53


@lru_cache(maxsize=128)
def residual_check(alignment, output):
    values = finite_values(alignment)
    if exact_grid(quantum(values), 2*max(abs(v) for v in values)):
        return {"status": "exact_output_codes", "method": "every aligned sum lies on an exactly representable FP64 lattice"}
    # A common aligned store makes order irrelevant for a single binary sum.
    pairs = list(combinations_with_replacement(values, 2))
    exact = [a+b for a, b in pairs]
    acc = format_named(ACCUMULATOR)
    rounded = [acc.rounded(v) for v in exact]
    reference = Tensor.quantize(exact, (len(exact),), output)
    actual = Tensor.quantize(rounded, (len(exact),), output)
    mismatches = [i for i, (a, b) in enumerate(zip(reference.codes, actual.codes)) if a != b]
    return {"status": "exact_output_codes" if not mismatches else "pending",
            "method": "all unordered finite aligned-value pairs, prescribed output store",
            "pairs": len(exact), "mismatches": len(mismatches),
            "examples": [{"left": str(pairs[i][0]), "right": str(pairs[i][1]),
                          "exact_sum": str(exact[i]), "fp64_sum": str(rounded[i]),
                          "exact_output_code": reference.codes[i], "fp64_output_code": actual.codes[i]}
                         for i in mismatches[:8]]}


@lru_cache(maxsize=128)
def hard_check(source, output, function):
    fmt = format_named(source.format)
    codes = tuple(c for c in range(1 << fmt.bits) if fmt.decode(c).is_finite())
    inputs = Tensor((len(codes),), codes, source)
    exact = []
    for value in inputs.values():
        clipped = clamp(binary(value, Fraction(3), "add"), Fraction(0), Fraction(6))
        multiplied = binary(value, clipped, "mul") if function == "hard_swish" else clipped
        # Preserve the format's negative-zero semantics in the exact path.
        exact.append(binary(multiplied, Fraction(1, 6), "mul"))
    reference = Tensor.quantize(exact, inputs.shape, output)
    actual = activation(inputs, function=function, output=output, accumulator=ACCUMULATOR)
    mismatches = [i for i, (a, b) in enumerate(zip(reference.codes, actual.codes)) if a != b]
    return {"finite_codes_tested": len(codes), "mismatches": len(mismatches),
            "examples": [{"input_code": codes[i], "exact_output_code": reference.codes[i], "fp64_output_code": actual.codes[i]}
                         for i in mismatches[:8]]}


@lru_cache(maxsize=128)
def mean_check(source, output, count):
    values, outputs = finite_values(source), finite_values(output)
    q, maximum = quantum(values), max(abs(v) for v in values)
    exact_sum = exact_grid(q, count*maximum)
    if not exact_sum:
        return {"status": "pending", "reason": "finite sum lattice lacks an exact FP64 prefix proof", "reduction_length": count}
    margins = mean_thresholds(values, count, outputs, Fraction(1, 2**1074))
    acc = format_named(ACCUMULATOR)
    ties_exact = all(Fraction(acc.rounded((a+b)/2)) == (a+b)/2 for a, b in zip(outputs, outputs[1:]))
    bound = UNIT_ROUNDOFF*maximum+HALF_MIN_SUBNORMAL
    margin = margins["minimum_nonzero_margin"]
    safe = ties_exact and margin is not None and bound < margin
    return {"status": "exact_output_codes" if safe else "pending", "reduction_length": count,
            "sum_prefixes_exact": exact_sum, "division_error_bound": str(bound),
            "minimum_nonzero_threshold_margin": str(margin), "exact_ties": margins["exact_ties"],
            "all_output_thresholds_exact_fp64": ties_exact}


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
        if any(not isinstance(e, Encoding) or e.axis is not None or e.block_size is not None for e in (x, output)):
            row["reason"] = "requires ordinary scalar finite operand domains"
        elif (op in {"flatten", "reshape", "broadcast_multiply", "channel_slice", "concatenate", "resize_nearest", "lut"}
                or op == "pool2d" and attrs["kind"] == "max"
                or op == "activation" and attrs["function"] in {"identity", "relu", "relu6"}):
            row.update(status="no_accumulator_rounding", reason="movement, prescribed LUT/output store or exact arithmetic")
        elif attrs.get("accumulator") != ACCUMULATOR:
            row["reason"] = "operator does not use the FP64 policy"
        elif op == "activation" and attrs["function"] in {"hard_swish", "hard_sigmoid"}:
            result = hard_check(x, output, attrs["function"])
            row.update(status="exact_output_codes" if result["mismatches"] == 0 else "pending", evidence=result)
        elif op == "elementwise" and attrs["operation"] == "add":
            alignment = parse_encoding(attrs["alignment"])
            if not isinstance(alignment, Encoding) or alignment.axis is not None or alignment.block_size is not None:
                row["reason"] = "alignment is outside ordinary scalar finite domains"
            else:
                row.update(residual_check(alignment, output))
        elif op == "adaptive_average_pool2d":
            if attrs.get("output_size", 1) not in (1, [1, 1], (1, 1)) or source not in shapes or len(shapes[source]) != 4:
                row["reason"] = "global mean needs a fixed observed NCHW shape"
            else:
                row.update(mean_check(x, output, prod(shapes[source][-2:])))
        elif op == "decode_boxes":
            if source not in shapes or len(shapes[source]) != 4 or shapes[source][1] != 4:
                row["reason"] = "box decode needs fixed N x 4 x H x W shape"
            else:
                values = finite_values(x)
                stride = attrs["stride"]
                if type(stride) is not int or stride < 1:
                    raise ValueError("box stride must be a positive integer")
                anchor = Fraction(2*max(shapes[source][-2:])-1, 2)
                step = quantum([*values, Fraction(1, 2)])/2
                maximum = 2*(anchor+max(abs(v) for v in values))*stride
                exact = exact_grid(step, maximum)
                row.update(status="exact_output_codes" if exact else "pending", intermediate_lattice=str(step),
                           maximum_intermediate_bound=str(maximum), intermediate_arithmetic_exact=exact)
        else:
            row["reason"] = "operator requires separate precision evidence"
        records.append(row)
    return {"status": "local_finite_fp64_nonmac_only", "records": records,
            "pending_nodes": [r["node"] for r in records if r["status"] == "pending"],
            "limits": ["identical stored finite inputs, prescribed alignment/LUT/output stores and fixed observed shapes",
                       "dyadic lattice proof covers every prefix; global-mean division compares non-tie margins and exact ties separately",
                       "non-dyadic residuals enumerate aligned value pairs using oracle-defined finite values",
                       "nonfinite behavior, shared scales, MAC/bias sensitivity and whole-graph native acceptance remain separate"]}
