"""Input-mean drift at every conv/linear input (lane Q3, protocol addendum 3; diagnostic only).

For each listed arm the engine is rebuilt exactly as ``evaluate.run_group`` builds it (same spec from
``evaluate.mode1_arms`` / ``mode3_arms``, same L6 fit, same 256 correction images, same calibration
observations), the rebuilt deployed state is compared with the sealed record's digest, and the 256
correction images are run through the FP32 graph and through the rebuilt engine.  At the input of every
conv/linear module the per-channel mean over images and positions is taken in both networks; the record
keeps, per layer, drift = E[x_q] - E[x_fp] averaged over channels, the share of channels with a positive
drift and the FP32 input mean.  No screen image is used and no accuracy is computed.  One JSON file per
network, written once.
"""
from __future__ import annotations

import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, seal, unseal

from .common import BASE, PROTOCOL_FILE, own_sources

ADDENDUM = ROOT / "public/experiments/configs/breadth-study/b2-collapse-protocol-v1-addendum-3.json"
ARMS = {
    "resnet18": ["faith-w4-a32-perlayer-N--none", "faith-w4-a32-perlayer-N--global",
                 "w4-a4-default-N--none", "w4-a4-default-N--global",
                 "w4-a4-default-L--none", "w4-a4-default-L--global",
                 "w5-a5-default-N--none", "w5-a5-default-N--global",
                 "fp6_e3m2-default-N--none", "fp6_e3m2-default-N--global"],
    "mobilenet_v2": ["w4-a4-default-L--none", "w4-a4-default-L--global",
                     "w4-a4-default-N--none", "w4-a4-default-N--global",
                     "fp8_e5m2-default-N--none", "fp8_e5m2-default-N--global"],
}
LATE = {"resnet18": ("layer4_0_conv2", "layer4_1_conv1", "layer4_1_conv2", "fc"),
        "mobilenet_v2": ("features_17_conv_1_0", "features_17_conv_2", "classifier_1")}
# Mode 3 has no lane 'none' arm: the uncorrected default engine is compared with the sealed
# default_no_bias_correction record (same recipe except the correction), for information.
UNSEALED_COUNTERPART = {"fp6_e3m2-default-N--none": "fp6_e3m2-default_no_bias_correction-N--b2",
                        "fp8_e5m2-default-N--none": "fp8_e5m2-default_no_bias_correction-N--b2"}
SOURCES = ("drift.py",)


def drift_sources():
    folder = ROOT / "tools/experiment_b2_collapse"
    return {**own_sources(), **{f"tools/experiment_b2_collapse/{n}": file_hash(folder / n) for n in SOURCES}}


def spec_for(model, name):
    """The arm spec of ``name`` as the sealed arms define it (mode-3 'none' is built from the 'global' spec)."""
    from .evaluate import label, mode1_arms, mode3_arms
    for s in mode1_arms(model) + mode3_arms(model):
        if label(s) == name:
            return s
    cell, policy = name.rsplit("--", 1)
    for s in mode3_arms(model):
        if label(s) == f"{cell}--global" and policy == "none":
            return {**s, "correction": "none"}
    raise KeyError(name)


def input_means(run, graph, targets, data, device, chunk=32):
    """Per-channel mean (float64) of the input of each target module over images and positions."""
    import torch
    sums = {t: None for t in targets}
    counts = {t: 0 for t in targets}
    hooks = []
    for t in targets:
        def hook(module, inputs, output, t=t):
            x = inputs[0]
            s = x.double().sum(dim=[0] + list(range(2, x.ndim))).cpu()
            sums[t] = s if sums[t] is None else sums[t] + s
            counts[t] += x.numel() // x.shape[1]
        hooks.append(graph.get_submodule(t).register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for i in range(0, len(data), chunk):
                run(torch.from_numpy(np.asarray(data[i:i + chunk], dtype=np.float32)).to(device))
    finally:
        for h in hooks:
            h.remove()
    return {t: sums[t] / counts[t] for t in targets}


def layer_rows(layers, fp, q):
    rows = {}
    for name, target in layers:
        d = (q[target] - fp[target]).numpy()
        rows[name] = {"drift_mean": float(d.mean()), "share_positive": float((d > 0).mean()),
                      "fp_mean": float(fp[target].numpy().mean()), "drift_rms": float(np.sqrt((d ** 2).mean()))}
    return rows


def run_model(model, device="cuda", tag="r1", only=None, log=print):
    import torch
    from torch import nn
    from tools.experiment_b2 import frozen  # noqa: F401
    from tools.experiment_b2.common import BIAS_CORRECTION_IMAGES
    from tools.experiment_b2.data import cached_inputs, v1_calibration
    from tools.experiment_b2.recipe import named
    from tools.experiment_b2.runner import model_setup
    from tools.experiment_b2_recon.engine import state_digest
    from tools.experiment_b2_recon.fit import find_fit, load_fit
    from .build import build
    out_path = BASE / "drift" / f"input-drift--{model}--{tag}.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists (written once; use a new tag)")
    started = time.monotonic()
    setup = model_setup(model, device)
    graph = setup["graph"].eval()
    arrays, maxima, calibration = v1_calibration(model)
    bias_inputs, bias_rows = cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"],
                                           build=False)
    layers = [(n.name, n.target) for n in graph.graph.nodes
              if n.op == "call_module" and isinstance(graph.get_submodule(n.target), (nn.Conv2d, nn.Linear))]
    targets = [t for _, t in layers]
    fp = input_means(graph, graph, targets, bias_inputs, device)
    arms, fits = {}, {}
    folder = BASE / "evals" / model
    for name in ARMS[model] if only is None else [a for a in ARMS[model] if a in only]:
        tick = time.monotonic()
        s = spec_for(model, name)
        learned = None
        if s["fit"]:
            rule, mode, seed = s["fit"]
            fit_folder = find_fit(model, s["weight_format"], rule, seed, mode)
            if fit_folder not in fits:
                fits[fit_folder] = load_fit(fit_folder, graph, device)
            learned = fits[fit_folder][0]
        recipe = named(s["recipe"]) if s["recipe"] else None
        run, metadata, deployed = build(graph, {**s, "recipe": recipe}, arrays, maxima, device, bias_inputs,
                                        learned=learned)
        rebuilt = state_digest(deployed)
        sealed_name = UNSEALED_COUNTERPART.get(name, name)
        sealed_path = folder / f"{sealed_name}.json"
        sealed = unseal(sealed_path)["deployed_state_sha256"] if sealed_path.exists() else None
        q = input_means(run, deployed, targets, bias_inputs, device)
        rows = layer_rows(layers[1:], fp, q)
        late = {n: rows[n] for n in LATE[model]}
        arms[name] = {"record": str(sealed_path.relative_to(ROOT)) if sealed_path.exists() else None,
                      "record_is_counterpart": name in UNSEALED_COUNTERPART,
                      "deployed_state_sha256": rebuilt, "deployed_state_equal_to_record": (
                          None if sealed is None else sealed == rebuilt),
                      "late_inputs": late, "layers": rows,
                      "layers_negative_drift": int(sum(r["drift_mean"] < 0 for r in rows.values())),
                      "median_share_positive": float(np.median([r["share_positive"] for r in rows.values()])),
                      "seconds": round(time.monotonic() - tick, 1)}
        log(json.dumps({"model": model, "arm": name, "equal_to_record": arms[name]["deployed_state_equal_to_record"],
                        "late": {n: [round(r["drift_mean"], 4), round(r["share_positive"], 3)]
                                 for n, r in late.items()}, "seconds": arms[name]["seconds"]}), flush=True)
        del run, deployed
        torch.cuda.empty_cache() if device == "cuda" else None
    payload = {
        "what": "input-mean drift E[x_q] - E[x_fp] at every conv/linear input, correction images (diagnostic)",
        "protocol": {"path": str(PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(PROTOCOL_FILE)},
        "addendum": {"path": str(ADDENDUM.relative_to(ROOT)), "sha256": file_hash(ADDENDUM)},
        "model": model, "model_context": setup["context"], "runtime": setup["env"],
        "correction_images": {"images": len(bias_rows), "rows_sha256": digest(bias_rows)},
        "calibration_observations": calibration["identity"], "own_sources": drift_sources(),
        "late_inputs": list(LATE[model]), "fp32_input_mean": {n: float(fp[t].numpy().mean()) for n, t in layers[1:]},
        "arms": arms, "evidence": "diagnostic_calibration_images_no_evaluation",
        "cost": {"seconds": round(time.monotonic() - started, 1), "device": device,
                 "peak_gpu_mib": (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seal(out_path, payload)
    log(json.dumps({"written": str(out_path.relative_to(ROOT)), "seconds": payload["cost"]["seconds"]}), flush=True)
    return 0 if all(a["deployed_state_equal_to_record"] is not False or a["record_is_counterpart"]
                    for a in arms.values()) else 5
