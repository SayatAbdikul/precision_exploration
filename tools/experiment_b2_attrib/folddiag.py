"""Fold-mechanism diagnostic of the ``pcf:all`` arm (lane Q4 r5, protocol addendum 6; CPU only).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. nice -n 10 .venv-b/bin/python -m tools.experiment_b2_attrib.folddiag [--models m1,m2]

For every dense (conv/linear) consumer that ``pcf:all`` folds per-input-channel activation scale ratios ``r_c`` into,
compare the default per-output-channel weight codes ``Qw(W)`` with the folded codes ``Qw(W r)`` (the arm's own
``engine._fold``): zero-code fraction, input channels silenced (all codes zero), and the range-weighted relative
weight error ``||(W_eff - W) r|| / ||W r||``.  No accuracy is measured and no arm record is written.
Output: artifacts/experiment_b2_attrib/folddiag/fold-diagnostic-v1.json (written once; ``--out`` for a dry run).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from tools.experiment_b.common import ROOT, file_hash
from . import pcfull, runner
from .engine import _fold
from .groups import consumers, partition
from .lmrunner import parse_lm

OUT = ROOT / "artifacts/experiment_b2_attrib/folddiag/fold-diagnostic-v1.json"
PROTOCOL = ROOT / "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-6.json"
MODELS = ("mobilenet_v2", "mobilenet_v3_large", "resnet18")


def _codes(quantized, scales):
    shaped = torch.tensor([float(x) for x in scales], dtype=torch.float64).reshape((-1,) + (1,) * (quantized.ndim - 1))
    return torch.round(quantized.double() / shaped)


def _per_input_channel(t):
    """Move the input-channel axis first and flatten the rest: [C_in, ...]."""
    return t.transpose(0, 1).reshape(t.shape[1], -1)


def consumer_row(shared, q_graph, user, ratio):
    node = {n.name: n for n in q_graph.graph.nodes}[user]
    module = q_graph.get_submodule(node.target)
    original = shared.fp_graph.get_submodule(node.target).weight.detach().float()
    default_q = shared.weight_graph.get_submodule(node.target).weight.detach().float()
    fold = _fold(shared, q_graph, user, ratio)  # the arm's function: module.weight = Qw(W r) / r
    folded_eff = module.weight.detach().float()
    if isinstance(module, torch.nn.Linear):
        shape = (1, -1)
    else:
        shape = (1, -1, 1, 1)
    r = torch.as_tensor(ratio, dtype=torch.float32).reshape(shape)
    folded_q = folded_eff * r  # = Qw(W r) up to one float rounding
    signed = shared.quantizers["signed"]
    default_codes = _codes(default_q, shared.weight_scales[user])
    from tools.experiment_b2.scales import mse_search
    found, _ = mse_search((original * r).reshape(original.shape[0], -1), signed)
    folded_codes = _codes(folded_q, found)
    weighted_den = float(((original * r).double() ** 2).sum().sqrt())
    err_default = float((((default_q - original) * r).double() ** 2).sum().sqrt()) / weighted_den
    err_folded = float((((folded_eff - original) * r).double() ** 2).sum().sqrt()) / weighted_den
    silenced = lambda codes: int(((_per_input_channel(codes) != 0).sum(dim=1) == 0).sum())
    return {"consumer": user, "kind": type(module).__name__, "in_channels": int(original.shape[1]),
            "zero_code_fraction_default": round(float((default_codes == 0).double().mean()), 4),
            "zero_code_fraction_folded": round(float((folded_codes == 0).double().mean()), 4),
            "silenced_input_channels_default": silenced(default_codes),
            "silenced_input_channels_folded": silenced(folded_codes),
            "weighted_rel_error_default": round(err_default, 5), "weighted_rel_error_folded": round(err_folded, 5),
            "codes_integral": bool(float((folded_q.double() / torch.tensor([float(x) for x in found], dtype=torch.float64)
                                          .reshape((-1,) + (1,) * (original.ndim - 1)) - folded_codes).abs().max()) < 1e-3),
            "fold_weight_scales_sha256": fold["weight_scales_sha256"]}


def diagnose(model, format_name="int8"):
    tick = time.monotonic()
    setup, shared, bias_inputs, info = runner.shared_state(model, format_name, "cpu")
    groups = partition(model, shared.fp_graph, shared.plan)
    arm, pcf_nodes = parse_lm("pcf:all", shared.plan, groups)
    cache = pcfull.channel_cache(shared, pcf_nodes, bias_inputs)
    q_graph = copy.deepcopy(shared.weight_graph)
    rows = []
    for node in sorted(pcf_nodes):
        scales, meta = cache[node]
        ratio = np.asarray(scales, np.float32) / np.float32(shared.activation_scales[node])
        for user, label in consumers(shared.fp_graph, shared.plan, node):
            if label not in ("conv", "linear"):
                continue
            module = q_graph.get_submodule({n.name: n for n in q_graph.graph.nodes}[user].target)
            if isinstance(module, torch.nn.Conv2d) and module.groups != 1:
                continue  # depthwise consumers: one input channel per output channel, nothing moves
            row = consumer_row(shared, q_graph, user, ratio)
            row.update({"boundary": node, "scale_spread_log2": round(float(meta["scale_spread_log2"]), 3)})
            rows.append(row)
    def med(key):
        return round(float(np.median([r[key] for r in rows])), 5) if rows else None
    summary = {"dense_folded_consumers": len(rows),
               "median_zero_code_fraction_default": med("zero_code_fraction_default"),
               "median_zero_code_fraction_folded": med("zero_code_fraction_folded"),
               "silenced_input_channels_default": sum(r["silenced_input_channels_default"] for r in rows),
               "silenced_input_channels_folded": sum(r["silenced_input_channels_folded"] for r in rows),
               "input_channels_total": sum(r["in_channels"] for r in rows),
               "median_weighted_rel_error_default": med("weighted_rel_error_default"),
               "median_weighted_rel_error_folded": med("weighted_rel_error_folded"),
               "max_weighted_rel_error_folded": max((r["weighted_rel_error_folded"] for r in rows), default=None),
               "consumers_with_folded_error_above_2x_default": sum(
                   r["weighted_rel_error_folded"] > 2 * r["weighted_rel_error_default"] for r in rows),
               "spearman_spread_vs_error_ratio": None}
    if len(rows) > 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr([r["scale_spread_log2"] for r in rows],
                           [r["weighted_rel_error_folded"] / r["weighted_rel_error_default"] for r in rows])
        summary["spearman_spread_vs_error_ratio"] = {"rho": round(float(rho), 3), "p": float(p), "n": len(rows)}
    worst = sorted(rows, key=lambda r: r["weighted_rel_error_folded"] - r["weighted_rel_error_default"], reverse=True)
    return {"model": model, "format": format_name, "summary": summary, "worst5": [r["consumer"] for r in worst[:5]],
            "consumers": rows, "seconds": round(time.monotonic() - tick, 1), "calibration": info["bias_inputs"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    out = ROOT / args.out if not args.out.startswith("/") else __import__("pathlib").Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: written once")
    from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
    torch.set_grad_enabled(False)
    result = {"protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": file_hash(PROTOCOL)},
              "source_sha256": file_hash(ROOT / "tools/experiment_b2_attrib/folddiag.py"),
              "evidence": "development: calibration images only (first 256 of the frozen calibration list); no accuracy",
              "models": {m: diagnose(m) for m in args.models.split(",")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({m: v["summary"] for m, v in result["models"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
