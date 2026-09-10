"""Compare exact backends on real classifier weights and synthetic inputs.

This is architecture integration evidence, not fixed-image network conformance
or an ImageNet quality result. Input shape and provenance are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from public.experiments.registry.identity import canonical_json_bytes
from public.inference.tensor import Encoding, Tensor
from public.inference.conformance_job import source_identity
from public.quantization.graph.torchvision import lower
from public.quantization.graph.executable import execute, graph_sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n"), required=True)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--backends", nargs="+", choices=("reference", "cpp", "cuda"), default=["cpp", "cuda"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    execution_source = source_identity()
    if args.input_size < 8:
        parser.error("input-size must be at least 8")
    import torch
    from torchvision import models
    import numpy as np
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / f"public/workloads/models/manifests/{args.model}.json").read_text())
    checkpoint = ROOT / manifest["checkpoint_path"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint identity mismatch")
    lower_model = lower
    if args.model == "yolov8n":
        os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/"cache/ultralytics"))
        os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"cache/matplotlib"))
        from ultralytics import YOLO
        from public.quantization.graph.yolo import lower as lower_model
        model = YOLO(str(checkpoint)).model
    else:
        model = getattr(models, args.model)(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoding = Encoding("fp6_e3m2")
    graph = lower_model(model.eval(), input_encoding=encoding, weight_format="fp6_e3m2", accumulator="fp32_e8m23_accumulator",
                  provenance={"kind": "synthetic_input_real_checkpoint", "model": args.model,
                              "checkpoint_sha256": manifest["checkpoint_sha256"], "source_graph_sha256": manifest["deployment_graph_sha256"]})
    identity = graph_sha256(graph)
    path = ROOT / "artifacts/folded_graphs" / f"{args.model}-phase2-{identity}.json"
    path.write_bytes(canonical_json_bytes(graph))
    shape = (1,3,args.input_size,args.input_size)
    from public.quantization.ptq.encoding import direct_float_tensor
    array = np.linspace(-1,1,num=int(np.prod(shape)),dtype=np.float32).reshape(shape)
    inputs = {next(iter(graph["inputs"])): direct_float_tensor(array,encoding)}
    print(f"{args.model}: lowered {len(graph['nodes'])} nodes; input {shape}", flush=True)
    records, expected = [], None
    for backend in args.backends:
        started = time.perf_counter()
        result = execute(graph,inputs,backend=backend)
        elapsed = time.perf_counter()-started
        if expected is not None:
            for name, layer in expected["layers"].items():
                if result["layers"][name] != layer:
                    raise ValueError(f"first backend mismatch: {name}")
            if expected["outputs"] != result["outputs"]:
                raise ValueError("network output mismatch")
        expected = result
        records.append({"backend":backend,"seconds":elapsed,"layers":result["layers"],
                        "outputs":{name:value.document() for name,value in result["outputs"].items()}})
        print(f"{backend}: {elapsed:.3f}s, {len(result['layers'])} layer hashes recorded",flush=True)
    output = args.output or ROOT / "results/summaries" / f"phase2-{args.model}-synthetic-smoke.json"
    if source_identity() != execution_source:
        raise RuntimeError("engine sources changed during execution; rerun before publishing evidence")
    report = {"schema_version":"2.0.0","source_sha256":execution_source,"model":args.model,"input_shape":list(shape),
              "input":"deterministic float32 linear ramp [-1,1], quantized to FP6",
              "scope":"synthetic-input architecture integration; not ImageNet quality or frozen-image validation",
              "graph_sha256":identity,"graph_artifact":str(path.relative_to(ROOT)),
              "backend_equality_checked":len(records)>1,"records":records}
    output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
