"""Diagnostic (lane L6 r2): why B2's empirical bias correction collapses ResNet18 INT4 to one class.

Builds the faithfulness setup check arm ``N-A32-bc`` (INT4 weights, one MSE scale
per layer, nearest rounding, FP32 activations, ``tools.experiment_b2.engine.bias_correct``
on the 256 bias-correction images) and writes, per conv/linear layer, the size of
the correction against the RMS of the FP32 channel means, plus how much the
corrected and the uncorrected quantized networks vary across images at each
layer (standard deviation over 64 screen images of the per-image channel means).
Reads only the calibration list and 64 screen images; writes one JSON file.

    artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.experiment_b2_recon.diagnose_bias
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

from tools.experiment_b.common import ROOT  # noqa: E402

OUT = ROOT / "artifacts/experiment_b2_recon/diagnostics/bias-correction-resnet18-int4-per-layer-r2.json"


def main(model="resnet18", wformat="int4", rule="mse_per_layer", device="cuda", screen_images=64):
    import torch
    from tools.experiment_b2 import frozen  # noqa: F401
    from tools.experiment_b2.common import BIAS_CORRECTION_IMAGES
    from tools.experiment_b2.data import cached_inputs, v1_calibration
    from tools.experiment_b2.runner import model_setup
    from .engine import build_engine, weight_layers

    setup = model_setup(model, device)
    graph = setup["graph"]
    arrays, maxima, _ = v1_calibration(model)
    bias_inputs, _ = cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"], build=False)
    runs = {}
    for correction in ("none", "empirical"):
        runs[correction] = build_engine(graph, wformat, None, None, arrays, maxima, device, weight_rule=rule,
                                        bias_inputs=bias_inputs, bias_correction=correction)
    names = [node.name for node, _ in weight_layers(graph)]
    captured = {}

    def hook_for(tag, name):
        def hook(_module, _inputs, output):
            # per-image channel means of the conv/linear output (before the nonlinearity)
            dims = list(range(2, output.ndim))
            captured.setdefault(tag, {})[name] = (output.mean(dim=dims) if dims else output).float().cpu()
        return hook

    handles = []
    for tag, module_graph in (("fp32", graph), ("none", runs["none"][2]), ("empirical", runs["empirical"][2])):
        for node, module in weight_layers(module_graph):
            handles.append(module.register_forward_hook(hook_for(tag, node.name)))
    screen = torch.from_numpy(np.array(setup["inputs"][:screen_images], dtype=np.float32)).to(device)
    predictions = {}
    with torch.inference_mode():
        for tag, run in (("fp32", graph), ("none", runs["none"][0]), ("empirical", runs["empirical"][0])):
            predictions[tag] = run(screen).argmax(dim=1).cpu().tolist()
    for handle in handles:
        handle.remove()
    report = runs["empirical"][1]["bias_correction"]
    layers = []
    for name in names:
        row = {"node": name, **{k: round(v, 5) for k, v in report.get(name, {}).items()}}
        for tag in ("fp32", "none", "empirical"):
            values = captured[tag][name]
            row[f"across_image_std_{tag}"] = round(float(values.std(dim=0).mean()), 5)
            row[f"channel_mean_{tag}"] = round(float(values.mean()), 5)
        row["correlation_with_fp32_none"] = round(float(np.corrcoef(captured["fp32"][name].flatten(),
                                                                    captured["none"][name].flatten())[0, 1]), 4)
        row["correlation_with_fp32_empirical"] = round(float(np.corrcoef(captured["fp32"][name].flatten(),
                                                                         captured["empirical"][name].flatten())[0, 1]), 4)
        layers.append(row)
    result = {"what": __doc__.splitlines()[0], "model": model, "weight_format": wformat, "weight_rule": rule,
              "activations": "fp32", "bias_correction_images": len(bias_inputs), "screen_images_used": screen_images,
              "evidence": "development_evidence_screen1k (first 64 screen images, diagnostic only)",
              "distinct_predictions": {tag: len(set(v)) for tag, v in predictions.items()},
              "layers": layers}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"wrote": str(OUT.relative_to(ROOT)), "distinct_predictions": result["distinct_predictions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
