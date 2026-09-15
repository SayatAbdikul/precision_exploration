"""Analytical graph liveness and arithmetic inventories, without PPA claims."""
from collections import Counter
from math import prod

from public.analysis.phase3.storage import shared_scale_count, tensor_storage
from public.inference.reference.arithmetic import format_named


def format_support_prior(name):
    fmt = format_named(name)
    support = {
        "integer": ["signed integer product", "mapped product-scale addressing", "output requantization"],
        "fixed_point": ["signed fixed-point product", "binary-point alignment", "output rounding and saturation"],
        "float": ["exponent comparison and alignment", "significand product", "normalization and rounding", "special-code handling"],
        "bfp": ["signed mantissa product", "shared exponent addressing and alignment", "block scale selection and output conversion"],
        "mx_float": ["local floating-point decode and product", "shared exponent addressing and alignment", "block scale selection and output conversion"],
        "posit": ["regime length detection and decode", "decoded product", "quire alignment", "posit output encode and rounding"],
        "logarithmic": ["sign and log-magnitude decode", "log-product addition", "linear conversion before Model C sum", "log output conversion"],
        "codebook": ["codebook lookup", "decoded product", "nearest-code output selection"],
        "binary": ["sign-controlled add or subtract", "binary output selection"],
        "ternary": ["zero gating and sign-controlled add or subtract", "three-level output selection"],
    }
    result = {"format": name, "family": fmt.family, "supporting_functions": support[fmt.family],
              "generic_decode_rom_example": {"entries": 1 << fmt.bits, "decoded_word_bits_assumption": 64,
                                             "bits": (1 << fmt.bits)*64,
                                             "scope": "alternative unscaled decode table per port; not a required circuit or exact real-value representation"}}
    if fmt.manifest["scaling"]["mode"] == "intrinsic_shared":
        result["frozen_scale_search"] = {"legal_e8m0_candidates_per_block": 255,
                                        "maximum_values_per_block": fmt.manifest["block"]["block_size"],
                                        "worst_case_scalar_reconstructions_per_block": 255*fmt.manifest["block"]["block_size"],
                                        "scope": "exhaustive MSE search work; hardware may use a proved equivalent optimization"}
    result["limits"] = ["functional inventory and illustrative ROM size, not synthesized datapath area or energy",
                        "table replication, scale range, intermediate precision and support arithmetic require an implementation choice"]
    return result


def encoded_storage(shape, encoding, mapped_scale_bits):
    fmt = format_named(encoding["format"])
    mode = fmt.manifest["scaling"]["mode"]
    if mode == "intrinsic_shared":
        count = shared_scale_count(shape, encoding["axis"], encoding["block_size"])
        width = fmt.manifest["scaling"]["scale_bits"]
    else:
        count = len(encoding["scales"]) if mode == "required_mapping" else 0
        width = mapped_scale_bits if count else 0
    return tensor_storage(shape, fmt.bits, scale_count=count, scale_bits=width)


def graph_resources(graph, shapes, *, mapped_scale_bits=64):
    """Treat every graph result as a fresh packed buffer, freed at last use.

    Views are deliberately counted as copies. This is a reproducible buffer
    schedule assumption, not the Python executor's allocation behavior.
    Accumulator storage alternatives are reported separately from activations.
    """
    names = set(graph["inputs"]) | {node["name"] for node in graph["nodes"]}
    if not names <= shapes.keys():
        raise ValueError(f"missing observed graph shapes: {sorted(names-shapes.keys())}")
    for name in names:
        tensor_storage(shapes[name], 1)  # Validate all observed dimensions.
    domains = dict(graph["inputs"])
    allocations = {name: encoded_storage(shapes[name], domain, mapped_scale_bits)
                   for name, domain in domains.items()}
    last_use = {name: -1 for name in names}
    for index, node in enumerate(graph["nodes"]):
        for source in node["inputs"]:
            if source in names:
                last_use[source] = index
    for name in graph["outputs"]:
        if name in names:
            last_use[name] = len(graph["nodes"])
    live = {name: row["byte_aligned_bits"] for name, row in allocations.items()}
    peak = sum(live.values())
    peak_node, peak_names = "graph_inputs", sorted(live)
    arithmetic, operations = Counter(), Counter()
    records = []
    for index, node in enumerate(graph["nodes"]):
        name, op, attrs = node["name"], node["op"], node["attrs"]
        operations[op] += 1
        output = dict(attrs.get("output", domains[node["inputs"][0]]))
        if op == "flatten" and output.get("block_size"):
            output["axis"] = min(output["axis"], attrs.get("start_dim", 1))
        domains[name] = output
        row = {"name": name, "op": op, "shape": shapes[name], "format": output["format"],
               "activation": encoded_storage(shapes[name], output, mapped_scale_bits), "arithmetic": {}}
        allocations[name] = row["activation"]
        elements = prod(shapes[name])
        uses_accumulator = (op in {"linear", "conv2d", "depthwise_conv2d", "block_conv2d", "dfl", "softmax",
                                  "decode_boxes", "elementwise", "adaptive_average_pool2d"}
                            or op == "activation" and attrs["function"] in {"hard_swish", "hard_sigmoid"}
                            or op == "pool2d" and attrs["kind"] == "average")
        acc_bits = format_named(attrs["accumulator"]).bits if uses_accumulator and "accumulator" in attrs else 0
        row["one_sequential_accumulator_bits"] = acc_bits
        row["one_accumulator_per_output_bits"] = elements * acc_bits
        if op in {"linear", "conv2d", "depthwise_conv2d", "block_conv2d", "dfl"}:
            weight = graph["constants"][node["inputs"][1]]
            k = attrs["bins"] if op == "dfl" else prod(weight["shape"][1:])
            row["reduction_length"] = k
            row["arithmetic"].update(model_c_products=elements*k, model_c_accumulator_adds=elements*k,
                                     stored_bias_adds=elements if attrs.get("bias") is not None else 0)
            if op == "block_conv2d":
                source_shape = shapes[node["inputs"][0]]
                groups = attrs.get("groups", 1)
                patches = [source_shape[0]*shapes[name][-2]*shapes[name][-1]*groups, k]
                fmt = format_named(attrs["patch_format"])
                row["fully_materialized_patch_buffer"] = encoded_storage(patches,
                    {"format": fmt.name, "axis": 1, "block_size": fmt.manifest["block"]["block_size"]}, mapped_scale_bits)
            if op == "dfl":
                row["arithmetic"].update(exponentials=elements*k, normalization_divisions=elements*k,
                                         softmax_sum_adds=elements*k, softmax_subtractions=elements*k,
                                         softmax_max_comparisons=elements*(k-1))
        elif op in {"elementwise", "broadcast_multiply"}:
            row["arithmetic"]["elementwise_"+attrs.get("operation", "mul")] = elements
        elif op == "lut":
            row["explicit_lookup_table_bits"] = len(attrs["table"])*format_named(output["format"]).bits
            row["arithmetic"]["lookup_evaluations"] = elements
        elif op == "scaled_nonlinear_lut":
            row["arithmetic"]["scaled_nonlinear_evaluations"] = elements
        elif op == "activation":
            row["arithmetic"][attrs["function"]+"_evaluations"] = elements
        elif op == "softmax":
            k = shapes[name][attrs["axis"]]
            row["arithmetic"].update(exponentials=elements, normalization_divisions=elements, softmax_sum_adds=elements,
                                     softmax_subtractions=elements, softmax_max_comparisons=elements//k*(k-1))
        elif op == "decode_boxes":
            # Reference decoder: four corners, two centers, width and height.
            row["arithmetic"].update(box_decode_adds=elements*2, box_decode_multiplies=elements*3//2)
        elif op == "adaptive_average_pool2d":
            row["arithmetic"].update(pool_sum_adds=prod(shapes[node["inputs"][0]]), pool_divisions=elements)
        elif op == "pool2d":
            kernel = attrs["kernel_size"]
            k = kernel*kernel if isinstance(kernel, int) else prod(kernel)
            if attrs["kind"] == "max":
                row["arithmetic"]["pool_comparisons_upper_bound"] = elements*(k-1)
            else:
                row["arithmetic"].update(pool_sum_adds_upper_bound=elements*k, pool_divisions=elements)
        arithmetic.update(row["arithmetic"])
        live[name] = row["activation"]["byte_aligned_bits"]
        row["live_activation_bits_before_release"] = sum(live.values())
        if row["live_activation_bits_before_release"] > peak:
            peak, peak_node, peak_names = row["live_activation_bits_before_release"], name, sorted(live)
        for source in list(live):
            if last_use[source] <= index:
                del live[source]
        records.append(row)
    return {"layers": records, "operator_counts": dict(operations), "arithmetic_counts": dict(arithmetic),
            "peak_live_activation_bits": peak, "peak_node": peak_node, "peak_live_buffers": peak_names,
            "sum_materialized_activation_bits": sum(row["byte_aligned_bits"] for row in allocations.values()),
            "maximum_sequential_accumulator_bits": max((r["one_sequential_accumulator_bits"] for r in records), default=0),
            "maximum_output_accumulator_bank_bits": max((r["one_accumulator_per_output_bits"] for r in records), default=0),
            "explicit_lookup_table_bits": sum(r.get("explicit_lookup_table_bits", 0) for r in records),
            "mapped_scale_bits_assumption": mapped_scale_bits,
            "limits": ["fixed input shape, sequential graph order, fresh packed output buffer including views, free after last use",
                       "constants and stored bias are separate from activation liveness; no buffer or LUT deduplication assumed",
                       "accumulator alternatives are one serial state versus one state per output; neither specifies hardware parallelism",
                       "MAC counts include padded zero products; DFL counts include its projection and normalization",
                       "patch buffers are optional full-materialization estimates, excluded from activation liveness",
                       "internal softmax buffers, line buffers, caches, control, scale-search cost and allocator overhead are excluded",
                       "operation counts do not estimate area, latency or energy; nonlinear evaluation and scale conversion costs remain architecture-dependent"]}
