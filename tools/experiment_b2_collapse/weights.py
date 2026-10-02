"""Mode 2/3 weight analysis on the CPU (lane Q3): where per-channel weight scales put the weights on a codebook.

For every conv/linear layer of the three folded classifiers and every format in ``FORMATS`` the
weights are quantized per output channel with the scale rules ``maxabs`` (B2 ``minimal``: the
channel max-abs at the top level), ``anchor_one`` (max-abs at the level 1.0) and ``mse`` (B2
``default`` search), and the layer reports the weight SQNR, the share of weights whose level
is at least 1 in magnitude, the share of weights that land on zero, the mean relative error
of the largest 1 percent of weights (by magnitude, per layer), the median search ratio and the
share of channels whose MSE scale clips (ratio below 1).  Writes one JSON file, once.

    .venv/bin/python -m tools.run.experiment_b2_collapse weights --tag r1
"""
from __future__ import annotations

import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, file_hash

from .common import BASE, PROTOCOL_FILE, own_sources

FORMATS = ("posit8_es1", "posit6_es1", "fp8_e5m2", "fp6_e3m2", "log6", "fp6_e2m3", "int6", "fp8_e4m3fn")
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")


def layer_rows(graph, fmt):
    import torch
    from tools.experiment_b2.codebook import TableQuantizer
    from tools.experiment_b2.scales import mse_search, scale_for_top
    from tools.experiment_b2_recon.engine import weight_layers
    q = TableQuantizer(fmt, "signed", torch.device("cpu"))
    rows = {}
    for node, module in weight_layers(graph):
        w = module.weight.detach().reshape(module.weight.shape[0], -1).float()
        span = w.abs().amax(dim=1)
        base = torch.tensor([scale_for_top(float(s), q.top) for s in span]).reshape(-1, 1)
        one = torch.tensor([scale_for_top(float(s), 1.0) for s in span]).reshape(-1, 1)
        mse, info = mse_search(w, q)
        rules = {"maxabs": base, "anchor_one": one, "mse": torch.tensor(mse).reshape(-1, 1)}
        flat = w.abs().flatten()
        top = flat >= torch.quantile(flat, 0.99) if flat.numel() > 100 else flat >= flat.max()
        row = {"weights": int(w.numel()), "channels": int(w.shape[0]),
               "mse_median_ratio_to_maxabs": float(np.median(info["ratio_to_maxabs_scale"])),
               "mse_share_channels_clipping": float(np.mean(np.array(info["ratio_to_maxabs_scale"]) < 1))}
        for name, scale in rules.items():
            out = q(w, scale)
            err = (out - w).double()
            signal = w.double().pow(2).sum()
            row[name] = {
                "sqnr_db": float(10 * torch.log10(signal / err.pow(2).sum())) if err.abs().sum() > 0 else None,
                "share_level_abs_ge_1": float(((w / scale).abs() >= 1).double().mean()),
                "share_zero": float(((out == 0) & (w != 0)).double().mean()),
                "top1pct_mean_relative_error": float((err.abs().flatten()[top] / flat[top].double()).mean())}
        rows[node.name] = row
    return rows


def main(tag="r1"):
    import torch
    from tools.experiment_b.classifier import configure, load_model
    out = BASE / "weights" / f"weight-placement-{tag}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    configure("cpu")
    tick = time.monotonic()
    result = {"what": __doc__.splitlines()[0], "protocol": {"path": str(PROTOCOL_FILE.relative_to(ROOT)),
                                                             "sha256": file_hash(PROTOCOL_FILE)},
              "own_sources": {**own_sources(), "tools/experiment_b2_collapse/weights.py":
                              file_hash(ROOT / "tools/experiment_b2_collapse/weights.py")},
              "evidence": "weights only (no images)", "models": {}}
    with torch.no_grad():
        for model in MODELS:
            graph, _, _ = load_model(model, "cpu")
            result["models"][model] = {fmt: layer_rows(graph, fmt) for fmt in FORMATS}
    summary = {}
    for model, by_format in result["models"].items():
        for fmt, rows in by_format.items():
            entry = {}
            for rule in ("maxabs", "anchor_one", "mse"):
                values = [r[rule]["sqnr_db"] for r in rows.values() if r[rule]["sqnr_db"] is not None]
                entry[rule] = {"median_layer_sqnr_db": float(np.median(values)), "min_layer_sqnr_db": float(np.min(values)),
                               "median_share_level_abs_ge_1": float(np.median([r[rule]["share_level_abs_ge_1"] for r in rows.values()])),
                               "median_top1pct_relative_error": float(np.median([r[rule]["top1pct_mean_relative_error"] for r in rows.values()]))}
            entry["mse_median_ratio_to_maxabs"] = float(np.median([r["mse_median_ratio_to_maxabs"] for r in rows.values()]))
            entry["mse_median_share_channels_clipping"] = float(np.median([r["mse_share_channels_clipping"] for r in rows.values()]))
            summary[f"{model}/{fmt}"] = entry
    result["summary"] = summary
    result["seconds"] = time.monotonic() - tick
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"wrote": str(out.relative_to(ROOT)), "seconds": round(result["seconds"])}))
    return 0
