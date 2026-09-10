"""Lower the frozen YOLOv8 detection modules to explicit candidate arithmetic."""
from public.inference.tensor import Encoding, SharedEncoding, Tensor
from public.inference.reference.arithmetic import format_named
from public.inference.reference.operators import nonlinear_lut
from public.quantization.calibration.mse import mse_scale
from public.quantization.ptq.encoding import direct_float_tensor
from public.quantization.graph.executable import freeze_graph
from public.workloads.models.deployment import fold_batchnorm


def lower(model, *, input_encoding, weight_format, accumulator, provenance, activation_encodings=None,
          weight_calibration_policy="rational_mse_v1", constant_observer=None):
    from torch import nn
    from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF, Concat, Detect
    model = fold_batchnorm(model)
    activation_encodings = activation_encodings or {}
    constants, nodes, encodings = {}, [], {"images": input_encoding}
    wf = format_named(weight_format)
    from public.quantization.calibration.numpy_mse import mse_scale_numpy,POLICY
    if weight_calibration_policy not in {"rational_mse_v1",POLICY}:
        raise ValueError("unsupported weight calibration policy")
    scale_search = mse_scale_numpy if weight_calibration_policy == POLICY else mse_scale

    def encoding(name):
        if name in activation_encodings:
            return activation_encodings[name]
        if isinstance(input_encoding,SharedEncoding):
            return SharedEncoding(input_encoding.format,axis=1)
        if format_named(input_encoding.format).manifest["scaling"]["mode"] == "none":
            return input_encoding
        raise ValueError(f"missing frozen activation encoding: {name}")

    def emit(name, op, sources, attrs):
        enc = encoding(name)
        nodes.append({"name":name,"op":op,"inputs":list(sources),"attrs":{**attrs,"output":enc.document()}})
        encodings[name] = enc
        return name

    def weight(module, name):
        tensor = module.weight.detach().cpu().contiguous()
        if constant_observer is not None:
            constant_observer(name,tensor)
        if wf.family == "float" and wf.manifest["scaling"]["mode"] == "none":
            constants[name] = direct_float_tensor(tensor.numpy(),Encoding(weight_format))
        else:
            if wf.manifest["scaling"]["mode"] == "intrinsic_shared":
                tensor = tensor.reshape(tensor.shape[0],-1)
                constants[name] = Tensor.quantize(tensor.flatten().tolist(),tuple(tensor.shape),SharedEncoding(weight_format,axis=1))
                return name
            scales = tuple(scale_search(c.flatten().numpy() if weight_calibration_policy == POLICY else c.flatten().tolist(),weight_format)["scale"] for c in tensor) if wf.manifest["scaling"]["mode"] == "required_mapping" else ("1",)
            domain = Encoding(weight_format,scales=scales,axis=0 if len(scales)>1 else None)
            from public.quantization.ptq.encoding import float_tensor
            constants[name] = float_tensor(tensor.numpy(),domain)
        return name

    def visit(module, source, name):
        if isinstance(module,nn.Sequential):
            for index,child in enumerate(module):
                source = visit(child,source,f"{name}_{index}")
            return source
        if isinstance(module,Conv):
            source = visit(module.conv,source,name+"_conv")
            return visit(module.act,source,name+"_act")
        if isinstance(module,nn.Conv2d):
            if module.padding_mode != "zeros" or isinstance(module.padding,str):
                raise ValueError("unsupported detector convolution padding")
            attrs = dict(accumulator=accumulator,stride=list(module.stride),padding=list(module.padding),
                         dilation=list(module.dilation),groups=module.groups)
            if module.bias is not None:
                attrs["bias"] = [str(v) for v in module.bias.detach().cpu().tolist()]
            key = weight(module,name+"_weight")
            op = "conv2d"
            if constants[key].encoding.block_size or isinstance(encodings[source],SharedEncoding):
                value = constants[key]
                constants[key] = Tensor((value.shape[0],module.weight[0].numel()),value.codes,value.encoding)
                attrs.update(kernel_size=list(module.kernel_size),patch_format=encodings[source].format)
                op = "block_conv2d"
            return emit(name,op,[source,key],attrs)
        if isinstance(module,nn.Identity):
            return source
        if isinstance(module,(nn.SiLU,nn.Sigmoid,nn.Tanh)):
            function = {nn.SiLU:"silu",nn.Sigmoid:"sigmoid",nn.Tanh:"tanh"}[type(module)]
            inp,out = encodings[source],encoding(name)
            if inp.axis is not None or out.axis is not None:
                return emit(name,"scaled_nonlinear_lut",[source],dict(function=function))
            return emit(name,"lut",[source],dict(function=function,input_encoding=inp.document(),table=list(nonlinear_lut(inp,out,function))))
        if isinstance(module,(nn.ReLU,nn.ReLU6)):
            return emit(name,"activation",[source],dict(function="relu6" if isinstance(module,nn.ReLU6) else "relu"))
        if isinstance(module,Bottleneck):
            result = visit(module.cv2,visit(module.cv1,source,name+"_cv1"),name+"_cv2")
            return emit(name,"elementwise",[source,result],dict(operation="add",accumulator=accumulator,alignment=encoding(name).document())) if module.add else result
        if isinstance(module,C2f):
            combined = visit(module.cv1,source,name+"_cv1")
            branches = [emit(name+f"_split_{i}","channel_slice",[combined],dict(start=i*module.c,stop=(i+1)*module.c)) for i in range(2)]
            for index,child in enumerate(module.m):
                branches.append(visit(child,branches[-1],name+f"_m_{index}"))
            return visit(module.cv2,emit(name+"_cat","concatenate",branches,dict(axis=1)),name+"_cv2")
        if isinstance(module,SPPF):
            branches = [visit(module.cv1,source,name+"_cv1")]
            for index in range(3):
                branches.append(visit(module.m,branches[-1],name+f"_pool_{index}"))
            return visit(module.cv2,emit(name+"_cat","concatenate",branches,dict(axis=1)),name+"_cv2")
        if isinstance(module,nn.MaxPool2d):
            if module.ceil_mode or module.dilation != 1 or module.return_indices:
                raise ValueError("unsupported detector pooling")
            return emit(name,"pool2d",[source],dict(kind="max",kernel_size=module.kernel_size,stride=module.stride,padding=module.padding))
        if isinstance(module,nn.Upsample):
            if module.mode != "nearest" or module.size is not None or int(module.scale_factor) != module.scale_factor:
                raise ValueError("detector only supports integer nearest upsampling")
            return emit(name,"resize_nearest",[source],dict(factor=int(module.scale_factor)))
        if isinstance(module,Concat):
            return emit(name,"concatenate",source,dict(axis=module.d))
        if isinstance(module,Detect):
            if module.end2end or module.reg_max < 2:
                raise ValueError("only the frozen YOLOv8 DFL detection head is supported")
            levels = []
            for level,feature in enumerate(source):
                prefix = name+f"_level_{level}"
                boxes = visit(module.cv2[level],feature,prefix+"_box_logits")
                classes = visit(module.cv3[level],feature,prefix+"_class_logits")
                distances = emit(prefix+"_dfl","dfl",[boxes,weight(module.dfl.conv,prefix+"_projection")],dict(bins=module.reg_max,accumulator=accumulator))
                boxes = emit(prefix+"_boxes","decode_boxes",[distances],dict(stride=int(module.stride[level]),accumulator=accumulator))
                classes = visit(nn.Sigmoid(),classes,prefix+"_scores")
                joined = emit(prefix+"_cat","concatenate",[boxes,classes],dict(axis=1))
                flat = prefix+"_flat"
                nodes.append({"name":flat,"op":"flatten","inputs":[joined],"attrs":{"start_dim":2}})
                encodings[flat] = encodings[joined]
                levels.append(flat)
            return emit(name,"concatenate",levels,dict(axis=2))
        raise ValueError(f"unsupported exact YOLO module: {type(module).__name__}")

    previous, outputs = "images", []
    for index,module in enumerate(model.model):
        source = previous if module.f == -1 else ([previous if f == -1 else outputs[f] for f in module.f] if isinstance(module.f,list) else outputs[module.f])
        previous = visit(module,source,f"model_{index}")
        outputs.append(previous)
    return freeze_graph(inputs={"images":input_encoding.document()},constants=constants,nodes=nodes,outputs=[previous],
                        provenance={**provenance,"detector_semantics":"candidate-dfl-and-boxes-1",
                                    "weight_calibration_policy":weight_calibration_policy})
