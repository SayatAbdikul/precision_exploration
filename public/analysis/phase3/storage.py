"""Nominal storage priors, with explicit metadata and representation assumptions."""
from math import prod

from public.inference.reference.arithmetic import format_named


def tensor_storage(shape, bits, *, scale_count=0, scale_bits=0):
    if not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("storage shape must contain positive integer dimensions")
    if any(type(value) is not int or value < 0 for value in (bits, scale_count, scale_bits)) or bits == 0:
        raise ValueError("invalid storage width/metadata count")
    values = prod(shape)
    payload = values * bits
    metadata = scale_count * scale_bits
    padded = 8*((payload+7)//8) + 8*((metadata+7)//8)
    return {"values": values, "nominal_payload_bits": payload, "metadata_bits": metadata,
            "byte_aligned_bits": padded, "padding_bits": padded-payload-metadata,
            "effective_bits_per_value": padded/values}


def shared_scale_count(shape, axis, block_size):
    if type(axis) is not int or not 0 <= axis < len(shape) or type(block_size) is not int or block_size <= 0:
        raise ValueError("invalid shared-scale layout")
    return prod(shape[:axis]+shape[axis+1:]) * ((shape[axis]+block_size-1)//block_size)


def graph_storage(graph, *, mapped_scale_bits=64):
    records = []
    for name, tensor in graph["constants"].items():
        encoding = tensor["encoding"]
        fmt = format_named(encoding["format"])
        mode = fmt.manifest["scaling"]["mode"]
        count = 0 if mode == "none" else len(encoding["scales"])
        width = fmt.manifest["scaling"]["scale_bits"] if mode == "intrinsic_shared" else mapped_scale_bits if count else 0
        records.append({"name": name, "format": fmt.name, **tensor_storage(tensor["shape"], fmt.bits, scale_count=count, scale_bits=width)})
    bias_bits = sum(len(node["attrs"].get("bias", [])) * format_named(node["attrs"]["accumulator"]).bits
                    for node in graph["nodes"] if "accumulator" in node["attrs"])
    return {"constants": records, "nominal_bias_bits": bias_bits,
            "total_constant_and_bias_bits": sum(row["byte_aligned_bits"] for row in records)+bias_bits,
            "mapped_scale_bits_assumption": mapped_scale_bits,
            "limits": ["nominal bit packing, not current host byte storage or measured memory use",
                       "mapped scale width is an analytical assumption; exact rational scale implementation remains separate",
                       "does not include peak activation liveness, local memory, control, or accumulator datapath area",
                       "not final area, timing, power, or energy evidence"]}
