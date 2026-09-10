"""Small explicit graph IR; unsupported nodes fail before execution."""
from __future__ import annotations

import copy
import hashlib
from math import prod

from public.experiments.registry.identity import canonical_json_bytes
from public.formats.oracle.manifest import manifest_sha256
from public.inference.reference.arithmetic import format_named
from public.inference.reference import operators as ref
from public.inference.reference import detector
from public.inference.operators.dispatch import Operators
from public.inference.tensor import Encoding, SharedEncoding, Tensor, parse_encoding

OPS = {
    "conv2d": (2, {"accumulator", "output", "bias", "stride", "padding", "dilation", "groups", "activation"}),
    "depthwise_conv2d": (2, {"accumulator", "output", "bias", "stride", "padding", "dilation", "activation"}),
    "linear": (2, {"accumulator", "output", "bias", "activation"}),
    "activation": (1, {"function", "output", "accumulator"}),
    "elementwise": (2, {"operation", "output", "accumulator", "alignment"}),
    "pool2d": (1, {"kind", "kernel_size", "output", "stride", "padding", "accumulator", "count_include_pad"}),
    "lut": (1, {"table", "output", "input_encoding", "function"}),
    "scaled_nonlinear_lut": (1,{"function","output"}),
    "reshape": (1, {"shape"}),
    "flatten": (1, {"start_dim"}),
    "adaptive_average_pool2d": (1, {"output", "accumulator", "output_size"}),
    "broadcast_multiply": (2, {"output"}),
    "concatenate": (None, {"axis", "output"}),
    "channel_slice": (1, {"start", "stop", "output"}),
    "resize_nearest": (1, {"factor", "output"}),
    "softmax": (1, {"axis", "accumulator", "output"}),
    "dfl": (2, {"bins", "accumulator", "output"}),
    "decode_boxes": (1, {"stride", "accumulator", "output"}),
    "block_conv2d": (2, {"kernel_size", "patch_format", "accumulator", "output", "bias", "stride", "padding", "dilation", "groups", "activation"}),
}


def required_formats(document):
    names = set()
    for tensor in document["constants"].values():
        names.add(tensor["encoding"]["format"])
    for encoding in document["inputs"].values():
        names.add(encoding["format"])
    for node in document["nodes"]:
        attrs = node["attrs"]
        for key in ("output", "alignment", "input_encoding"):
            if key in attrs:
                names.add(attrs[key]["format"])
        if "accumulator" in attrs:
            names.add(attrs["accumulator"])
        if "patch_format" in attrs:
            names.add(attrs["patch_format"])
    return names


def freeze_graph(*, inputs, constants, nodes, outputs, provenance):
    graph = {"schema_version": "2.0.0", "inputs": copy.deepcopy(inputs),
             "constants": {name: value.document() for name, value in constants.items()},
             "nodes": copy.deepcopy(nodes), "outputs": list(outputs), "provenance": copy.deepcopy(provenance)}
    graph["manifest_hashes"] = {name: manifest_sha256(format_named(name).manifest) for name in sorted(required_formats(graph))}
    validate_graph(graph)
    return graph


def validate_graph(document):
    expected = {"schema_version", "inputs", "constants", "nodes", "outputs", "manifest_hashes", "provenance"}
    if set(document) != expected or document["schema_version"] != "2.0.0":
        raise ValueError("unsupported executable graph schema")
    canonical_json_bytes(document)
    if not document["inputs"] or not document["outputs"]:
        raise ValueError("graph needs inputs and outputs")
    names = set(document["inputs"])
    if names & set(document["constants"]):
        raise ValueError("duplicate graph input/constant name")
    names.update(document["constants"])
    for encoding in document["inputs"].values():
        parse_encoding(encoding)
    for tensor in document["constants"].values():
        Tensor.from_document(tensor)
    for node in document["nodes"]:
        if set(node) != {"name", "op", "inputs", "attrs"} or node["name"] in names:
            raise ValueError("invalid or duplicate graph node")
        if node["op"] not in OPS:
            raise ValueError(f"unsupported exact operator: {node['op']}")
        arity, allowed = OPS[node["op"]]
        if (arity is not None and len(node["inputs"]) != arity) or not node["inputs"] or any(name not in names for name in node["inputs"]):
            raise ValueError("invalid input edges or node order")
        if set(node["attrs"]) - allowed:
            raise ValueError("unsupported operator attributes")
        if node["op"] not in {"reshape", "flatten"} and "output" not in node["attrs"]:
            raise ValueError("operator output encoding is required")
        for key in ("output", "alignment", "input_encoding"):
            if key in node["attrs"]:
                parse_encoding(node["attrs"][key])
        names.add(node["name"])
    if any(name not in names for name in document["outputs"]) or len(set(document["outputs"])) != len(document["outputs"]):
        raise ValueError("invalid graph outputs")
    expected_hashes = {name: manifest_sha256(format_named(name).manifest) for name in sorted(required_formats(document))}
    if document["manifest_hashes"] != expected_hashes:
        raise ValueError("graph manifest identity mismatch")
    return document


def graph_sha256(document):
    validate_graph(document)
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def execute(document, inputs, *, backend="reference", library=None):
    validate_graph(document)
    if set(inputs) != set(document["inputs"]):
        raise ValueError("graph input names mismatch")
    for name, encoding in document["inputs"].items():
        expected = parse_encoding(encoding)
        compatible = (inputs[name].encoding.format == expected.format and inputs[name].encoding.axis == expected.axis
                      and inputs[name].encoding.block_size == expected.block_size) if isinstance(expected,SharedEncoding) else inputs[name].encoding == expected
        if not compatible:
            raise ValueError("graph input encoding mismatch")
    values = {**inputs, **{name: Tensor.from_document(value) for name, value in document["constants"].items()}}
    operators = Operators(backend, library)
    traces, execution_modes = {}, {}
    for node in document["nodes"]:
        args = [values[name] for name in node["inputs"]]
        attrs = copy.deepcopy(node["attrs"])
        for key in ("output", "alignment", "input_encoding"):
            if key in attrs:
                attrs[key] = parse_encoding(attrs[key])
        if node["op"] == "reshape":
            source = args[0]
            if source.encoding.axis is not None:
                raise ValueError("reshape of scaled axes requires an explicit domain conversion")
            if prod(attrs["shape"]) != prod(source.shape):
                raise ValueError("reshape changes tensor element count")
            value = Tensor(tuple(attrs["shape"]), source.codes, source.encoding)
        elif node["op"] == "lut":
            if args[0].encoding != attrs.pop("input_encoding"):
                raise ValueError("LUT input domain mismatch")
            function = attrs.pop("function")
            if tuple(attrs["table"]) != ref.nonlinear_lut(args[0].encoding, attrs["output"], function):
                raise ValueError("LUT contents do not match declared function/domain")
            value = ref.apply_lut(*args, **attrs)
        elif node["op"] in {"conv2d", "depthwise_conv2d", "linear", "block_conv2d"}:
            value = getattr(operators, node["op"])(*args, **attrs)
            execution_modes[node["name"]] = "reference_rational" if operators.native is None else operators.native.last_strategy
        elif hasattr(detector,node["op"]):
            value = getattr(detector,node["op"])(*args, **attrs)
        else:
            value = getattr(ref, node["op"])(*args, **attrs)
        values[node["name"]] = value
        traces[node["name"]] = {"shape": list(value.shape), "encoding": value.encoding.document(),
                               "sha256": hashlib.sha256(canonical_json_bytes(value.document())).hexdigest()}
    return {"outputs": {name: values[name] for name in document["outputs"]}, "layers": traces,
            "backend": backend, "execution_modes": execution_modes, "graph_sha256": graph_sha256(document)}
