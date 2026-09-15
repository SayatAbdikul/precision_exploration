"""Frozen FP32 loading and reference samples, separate from exact candidate execution."""
import os
from pathlib import Path

from tools.phase3.common import ROOT, read, checked, file_hash


def load_runtime(model_name, plan, root=ROOT):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    import numpy as np
    import torch
    from public.quantization.calibration.observer import classifier_interpreter, detector_observe
    manifest = read(checked(plan["inputs"][model_name], root))
    checkpoint = root / manifest["checkpoint_path"]
    if file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash mismatch")
    if torch.__version__.split("+")[0] != "2.3.0":
        raise ValueError("Phase 3 requires frozen torch 2.3.0")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if model_name != "yolov8n":
        import torchvision
        from torchvision import models
        from PIL import Image
        from public.workloads.models.torchvision_eval import MODEL_SPECS
        if torchvision.__version__.split("+")[0] != manifest["framework_version"]:
            raise ValueError("frozen torchvision version mismatch")
        _, weights_class, weights_name = MODEL_SPECS[model_name]
        transform = getattr(getattr(models, weights_class), weights_name).transforms()
        model = getattr(models, model_name)(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        class Proxy:
            target = None
            def record(self, name, value):
                self.target.record(name, value)
        proxy = Proxy()
        interpreter = classifier_interpreter(model, proxy)
        def sample(path):
            with Image.open(path) as image:
                return transform(image.convert("RGB")).unsqueeze(0), None
        def observe(inputs, references):
            proxy.target = references
            with torch.inference_mode():
                return interpreter.run(inputs)
    else:
        import cv2
        import ultralytics
        from ultralytics.data.augment import LetterBox
        from public.quantization.graph.yolo import lower
        from public.inference.tensor import Encoding
        if ultralytics.__version__ != manifest["framework_version"]:
            raise ValueError("frozen detector version mismatch")
        model = ultralytics.YOLO(checkpoint).model.eval()
        constants = {}
        graph = lower(model, input_encoding=Encoding("fp6_e3m2"), weight_format="fp6_e3m2",
                      accumulator="fp32_e8m23_accumulator", provenance={"kind": "fp32_diagnostic_observation"},
                      constant_observer=lambda name, value: constants.__setitem__(name, value))
        letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
        def sample(path):
            value = cv2.imread(str(path))
            if value is None:
                raise ValueError(f"cannot decode {path}")
            shape = value.shape[:2]
            value = letterbox(image=value)[:, :, ::-1].transpose(2, 0, 1)
            return torch.from_numpy(np.ascontiguousarray(value)).float().unsqueeze(0)/255, shape
        def observe(inputs, references):
            with torch.inference_mode():
                return detector_observe(graph, constants, inputs, references)
    return sample, observe, manifest
