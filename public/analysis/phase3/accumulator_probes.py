"""Bounded numerical probes; evidence of sensitivity, never graph acceptance."""
from fractions import Fraction

from public.inference.reference.arithmetic import format_named
from public.inference.reference.detector import softmax
from public.inference.reference.operators import activation
from public.inference.tensor import Tensor, parse_encoding


def probe_graph(graph):
    domains, probes = dict(graph["inputs"]), []
    for node in graph["nodes"]:
        attrs = node["attrs"]
        domain = domains[node["inputs"][0]]
        domains[node["name"]] = attrs.get("output", domain)
        if node["op"] == "activation" and attrs["function"] in {"hard_swish", "hard_sigmoid"}:
            result = hard_activation_probe(domain, attrs["output"], attrs["function"], attrs["accumulator"])
        elif node["op"] == "dfl":
            result = uniform_softmax_probe(domain, attrs["output"], attrs["bins"], attrs["accumulator"])
        else:
            continue
        probes.append({"node": node["name"], **result})
    return probes


def hard_activation_probe(input_domain, output_domain, function, accumulator):
    source, output = parse_encoding(input_domain), parse_encoding(output_domain)
    fmt = format_named(source.format)
    if fmt.family != "integer" or source.axis is not None or output.axis is not None:
        raise ValueError("hard activation probe requires scalar-mapped integer operands")
    if function not in {"hard_swish", "hard_sigmoid"}:
        raise ValueError("unsupported hard activation probe")
    inputs = Tensor((1 << fmt.bits,), tuple(range(1 << fmt.bits)), source)
    exact = []
    for value in inputs.values():
        clipped = min(max(value+3, Fraction(0)), Fraction(6))
        exact.append((value*clipped if function == "hard_swish" else clipped)/6)
    target = Tensor.quantize(exact, inputs.shape, output)
    variants = {}
    for policy in dict.fromkeys((accumulator, "int64_accumulator", "fp64_e11m52_accumulator")):
        actual = activation(inputs, function=function, output=output, accumulator=policy)
        mismatches = [i for i, (a, b) in enumerate(zip(actual.codes, target.codes)) if a != b]
        variants[policy] = {"mismatching_codes": len(mismatches), "tested_codes": len(inputs.codes),
                            "maximum_stored_value_error": str(max(abs(a-b) for a, b in zip(actual.values(), target.values()))),
                            "examples": [{"input_code": i, "input_value": str(inputs.value((i,))),
                                          "actual_output_code": actual.codes[i], "exact_output_code": target.codes[i]}
                                         for i in mismatches[:8]]}
    return {"scope": "exhaustive finite input codes for this scalar activation domain",
            "reference": "exact rational activation with only the prescribed output store",
            "function": function, "declared_accumulator": accumulator, "variants": variants,
            "requires_precision_revision": variants[accumulator]["mismatching_codes"] > 0,
            "limits": ["input-code coverage does not measure code frequency on workload images",
                       "FP64 equality for this node does not accept other operators or a whole-graph policy"]}


def uniform_softmax_probe(input_domain, probability_domain, bins, accumulator):
    source, output = parse_encoding(input_domain), parse_encoding(probability_domain)
    if format_named(source.format).family != "integer" or source.axis is not None or output.axis is not None:
        raise ValueError("softmax probe requires scalar-mapped integer operands")
    if type(bins) is not int or bins < 2:
        raise ValueError("softmax probe needs at least two bins")
    inputs = Tensor.quantize([Fraction(0)]*bins, (1, bins), source)
    expected = Tensor.quantize([Fraction(1, bins)]*bins, inputs.shape, output)
    variants = {}
    for policy in dict.fromkeys((accumulator, "int64_accumulator", "fp64_e11m52_accumulator")):
        actual = softmax(inputs, axis=1, accumulator=policy, output=output)
        variants[policy] = {"mismatching_codes": sum(a != b for a, b in zip(actual.codes, expected.codes)),
                            "output_codes": list(actual.codes), "stored_probability_sum": str(sum(actual.values()))}
    return {"scope": "one equal-logit softmax vector; exact pre-store probabilities are 1/bins",
            "bins": bins, "declared_accumulator": accumulator, "variants": variants,
            "exact_output_codes": list(expected.codes), "exact_stored_probability_sum": str(sum(expected.values())),
            "output_representation_already_zeros_uniform_probabilities": all(value == 0 for value in expected.values()),
            "requires_precision_revision": variants[accumulator]["mismatching_codes"] > 0,
            "limits": ["equal logits are a counterexample probe, not exhaustive softmax or detector acceptance",
                       "output representation may also erase uniform probabilities independently of accumulator precision"]}
