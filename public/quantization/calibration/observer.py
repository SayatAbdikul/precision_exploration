"""FP32-only calibration observers, separate from exact inference execution."""
import hashlib
import numpy as np

SAMPLING = "16_evenly_spaced_flat_positions_per_node_per_image_v1"


class Observer:
    def __init__(self):
        self.samples = {}
        self.counts = {}
        self.image_nodes = set()
        self.expected_nodes = None
        self.images = 0

    def record(self,name,tensor):
        values = tensor.detach().cpu().contiguous().numpy().ravel()
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"nonfinite/empty calibration observation: {name}")
        if name in self.image_nodes:
            raise ValueError(f"duplicate observation in one image: {name}")
        self.image_nodes.add(name)
        selected = values[np.linspace(0,len(values)-1,min(16,len(values)),dtype=np.int64)].astype("<f4")
        self.samples.setdefault(name,[]).append(selected)
        self.counts[name] = self.counts.get(name,0)+len(values)

    def finish_image(self):
        if not self.image_nodes or self.expected_nodes is not None and self.image_nodes != self.expected_nodes:
            raise ValueError("calibration node coverage changed between images")
        self.expected_nodes = set(self.image_nodes)
        self.image_nodes.clear()
        self.images += 1

    def arrays(self):
        if self.image_nodes or self.images == 0:
            raise ValueError("incomplete calibration observations")
        return {name:np.concatenate(chunks) for name,chunks in sorted(self.samples.items())}

    def summary(self):
        return {"sampling":SAMPLING,"image_count":self.images,"nodes":{
            name:{"observed_elements":self.counts[name],"sample_count":len(array),
                  "sample_sha256":hashlib.sha256(array.tobytes()).hexdigest()}
            for name,array in self.arrays().items()}}


def classifier_interpreter(model,observer):
    import torch
    from public.workloads.models.deployment import fold_batchnorm
    graph = torch.fx.symbolic_trace(fold_batchnorm(model))
    class Interpreter(torch.fx.Interpreter):
        def run_node(self,node):
            result = super().run_node(node)
            if node.op != "output" and isinstance(result,torch.Tensor):
                observer.record(node.name,result)
            return result
    return Interpreter(graph)


def detector_observe(graph,constants,inputs,observer):
    """Evaluate the lowering's FP32 observation plan, never an inference fallback."""
    import torch
    from torch.nn import functional as F
    values = {**constants,"images":inputs}
    observer.record("images",inputs)
    for node in graph["nodes"]:
        args = [values[name] for name in node["inputs"]]
        attrs,op = node["attrs"],node["op"]
        x = args[0]
        if op == "conv2d":
            bias = torch.tensor([float(v) for v in attrs["bias"]],dtype=torch.float32) if "bias" in attrs else None
            result = F.conv2d(x,args[1],bias,stride=attrs.get("stride",1),padding=attrs.get("padding",0),dilation=attrs.get("dilation",1),groups=attrs.get("groups",1))
        elif op == "lut":
            result = {"silu":F.silu,"sigmoid":torch.sigmoid,"tanh":torch.tanh}[attrs["function"]](x)
        elif op == "activation":
            result = {"relu":F.relu,"relu6":F.relu6}[attrs["function"]](x)
        elif op == "elementwise":
            result = x+args[1] if attrs["operation"] == "add" else x*args[1]
        elif op == "concatenate":
            result = torch.cat(args,dim=attrs["axis"])
        elif op == "channel_slice":
            result = x[:,attrs["start"]:attrs["stop"]]
        elif op == "resize_nearest":
            result = F.interpolate(x,scale_factor=attrs["factor"],mode="nearest")
        elif op == "pool2d" and attrs["kind"] == "max":
            result = F.max_pool2d(x,attrs["kernel_size"],attrs.get("stride"),attrs.get("padding",0))
        elif op == "flatten":
            result = x.flatten(attrs["start_dim"])
        elif op == "dfl":
            b,_,h,w = x.shape
            probabilities = x.reshape(b,4,attrs["bins"],h*w).transpose(1,2).softmax(1)
            result = F.conv2d(probabilities,args[1]).reshape(b,4,h,w)
        elif op == "decode_boxes":
            h,w = x.shape[-2:]
            yy,xx = torch.meshgrid(torch.arange(h,dtype=torch.float32)+.5,torch.arange(w,dtype=torch.float32)+.5,indexing="ij")
            anchors = torch.stack((xx,yy))[None]
            lo,hi = anchors-x[:,:2],anchors+x[:,2:]
            result = torch.cat(((lo+hi)/2,hi-lo),1)*attrs["stride"]
        else:
            raise ValueError(f"unsupported FP32 calibration observation: {op}")
        values[node["name"]] = result
        observer.record(node["name"],result)
    return values[graph["outputs"][0]]
