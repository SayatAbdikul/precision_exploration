"""FX lowering for the folded mandatory torchvision classifiers.

Activation scales are supplied explicitly by node name. This importer does
not calibrate on evaluation inputs or silently run unsupported PyTorch ops.
"""
from __future__ import annotations

import operator
from fractions import Fraction

from public.inference.tensor import Encoding, SharedEncoding, Tensor
from public.inference.reference.arithmetic import format_named
from public.quantization.calibration.mse import mse_scale
from public.quantization.graph.executable import freeze_graph
from public.workloads.models.deployment import fold_batchnorm


def lower(model, *, input_encoding: Encoding, weight_format: str, accumulator: str,
          provenance, activation_encodings=None, weight_calibration_policy="rational_mse_v1"):
    import torch
    from torch import nn
    from torch.fx import symbolic_trace
    graph_module = symbolic_trace(fold_batchnorm(model))
    activation_encodings = activation_encodings or {}
    names, encodings, inputs, constants, nodes = {}, {}, {}, {}, []
    output_names = []
    weight_fmt = format_named(weight_format)
    from public.quantization.calibration.numpy_mse import mse_scale_numpy,POLICY
    if weight_calibration_policy not in {"rational_mse_v1",POLICY}:
        raise ValueError("unsupported weight calibration policy")
    scale_search = mse_scale_numpy if weight_calibration_policy == POLICY else mse_scale

    def out_encoding(node):
        if node.name in activation_encodings:
            return activation_encodings[node.name]
        if isinstance(input_encoding,SharedEncoding):
            return SharedEncoding(input_encoding.format,axis=1)
        if input_encoding.format and format_named(input_encoding.format).manifest["scaling"]["mode"] == "none":
            return input_encoding
        raise ValueError(f"missing frozen calibration encoding for node {node.name}")

    def emit(node, op, arguments, attrs, encoding):
        names[node], encodings[node] = node.name, encoding
        nodes.append({"name": node.name, "op": op, "inputs": [names[a] for a in arguments], "attrs": attrs})

    def weight(module, name):
        values = module.weight.detach().cpu().contiguous()
        if weight_fmt.manifest["scaling"]["mode"] == "intrinsic_shared":
            values = values.reshape(values.shape[0],-1)
            constants[name] = Tensor.quantize(values.flatten().tolist(),tuple(values.shape),SharedEncoding(weight_format,axis=1))
            return name
        if weight_fmt.family == "float" and weight_fmt.manifest["scaling"]["mode"] == "none":
            from public.quantization.ptq.encoding import direct_float_tensor
            constants[name] = direct_float_tensor(values.numpy(), Encoding(weight_format))
            return name
        if weight_fmt.manifest["scaling"]["mode"] == "required_mapping":
            scales = tuple(scale_search(channel.flatten().numpy(), weight_format)["scale"] if weight_calibration_policy == POLICY else scale_search(channel.flatten().tolist(), weight_format)["scale"] for channel in values)
            encoding = Encoding(weight_format, scales=scales, axis=0)
        else:
            encoding = Encoding(weight_format)
        from public.quantization.ptq.encoding import float_tensor
        constants[name] = float_tensor(values.numpy(), encoding)
        return name

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            names[node], encodings[node] = node.name, input_encoding
            inputs[node.name] = input_encoding.document()
        elif node.op == "output":
            result = node.args[0]
            output_names = [names[result]] if result in names else [names[item] for item in result]
        elif node.op == "call_module":
            module = graph_module.get_submodule(node.target)
            source = node.args[0]
            if isinstance(module, (nn.Identity, nn.Dropout)):
                names[node], encodings[node] = names[source], encodings[source]
                continue
            if isinstance(module, nn.Flatten):
                if module.end_dim != -1:
                    raise ValueError("only flatten through the last dimension is supported")
                emit(node, "flatten", [source], {"start_dim": module.start_dim}, encodings[source])
                continue
            encoding = out_encoding(node)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                key = weight(module, node.name + "_weight")
                attrs = {"accumulator": accumulator, "output": encoding.document()}
                if module.bias is not None:
                    attrs["bias"] = [str(v) for v in module.bias.detach().cpu().tolist()]
                if isinstance(module, nn.Conv2d):
                    if module.padding_mode != "zeros" or isinstance(module.padding, str):
                        raise ValueError("only explicit zero-padded Conv2D is supported")
                    attrs.update(stride=list(module.stride), padding=list(module.padding), dilation=list(module.dilation), groups=module.groups)
                    if weight_fmt.manifest["scaling"]["mode"] == "intrinsic_shared" or isinstance(encodings[source],SharedEncoding):
                        tensor = constants[key]
                        constants[key] = Tensor((tensor.shape[0],module.weight[0].numel()),tensor.codes,tensor.encoding)
                        attrs.update(kernel_size=list(module.kernel_size),patch_format=encodings[source].format)
                        op = "block_conv2d"
                    else:
                        op = "conv2d"
                else:
                    op = "linear"
                names[node], encodings[node] = node.name, encoding
                nodes.append({"name": node.name, "op": op, "inputs": [names[source], key], "attrs": attrs})
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid)):
                function = {nn.ReLU: "relu", nn.ReLU6: "relu6", nn.Hardswish: "hard_swish", nn.Hardsigmoid: "hard_sigmoid"}[type(module)]
                emit(node, "activation", [source], {"function": function, "output": encoding.document(), "accumulator": accumulator}, encoding)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                emit(node, "adaptive_average_pool2d", [source], {"output_size": module.output_size, "accumulator": accumulator, "output": encoding.document()}, encoding)
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                if module.ceil_mode or isinstance(module, nn.MaxPool2d) and (module.dilation != 1 or module.return_indices):
                    raise ValueError("unsupported pooling mode")
                attrs = {"kind": "max" if isinstance(module, nn.MaxPool2d) else "average", "kernel_size": module.kernel_size,
                         "stride": module.stride, "padding": module.padding, "output": encoding.document()}
                if isinstance(module, nn.AvgPool2d):
                    if module.divisor_override is not None:
                        raise ValueError("pool divisor override unsupported")
                    attrs.update(accumulator=accumulator, count_include_pad=module.count_include_pad)
                emit(node, "pool2d", [source], attrs, encoding)
            else:
                raise ValueError(f"unsupported exact module: {node.target} ({type(module).__name__})")
        elif node.op == "call_function" and node.target is torch.flatten:
            end_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("end_dim", -1)
            if end_dim != -1:
                raise ValueError("only flatten through last dimension is supported")
            emit(node, "flatten", [node.args[0]], {"start_dim": node.args[1] if len(node.args)>1 else node.kwargs.get("start_dim", 0)}, encodings[node.args[0]])
        elif node.op == "call_function" and node.target in {operator.add, operator.mul}:
            encoding = out_encoding(node)
            if node.target is operator.add:
                emit(node, "elementwise", node.args, {"operation": "add", "accumulator": accumulator,
                     "alignment": encoding.document(), "output": encoding.document()}, encoding)
            else:
                emit(node, "broadcast_multiply", node.args, {"output": encoding.document()}, encoding)
        elif node.op == "call_function" and node.target is torch.nn.functional.adaptive_avg_pool2d:
            encoding = out_encoding(node)
            emit(node, "adaptive_average_pool2d", [node.args[0]], {"output_size": node.args[1], "accumulator": accumulator,
                 "output": encoding.document()}, encoding)
        else:
            raise ValueError(f"unsupported exact graph operation: {node.op} {node.target}")
    return freeze_graph(inputs=inputs, constants=constants, nodes=nodes, outputs=output_names,
                        provenance={**provenance,"weight_calibration_policy":weight_calibration_policy})
