"""B2 networks with given weight integers: setup, sequential AXE/OPTQ fits, deployment, evaluation, checks.

Everything numeric is imported from the B2 and L6 reconstruction modules (read-only): activation plan and scales
(``activation_setup``), weight scales (``weight_scale``, B2's per-channel MSE search), bias correction
(``bias_correct``), the interpreter, the screen runner and the tie readout.  Only the weight integers differ.
"""
from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2.engine import B2Interpreter, bias_correct
from tools.experiment_b2_recon.engine import activation_setup, weight_layers, weight_scale

from . import axe

CASES = {
    "resnet18-int8": {"model": "resnet18", "weight_format": "int8", "activation_format": "int8", "recipe": "default",
                      "N": 8, "l8_case": "resnet18-int8-default-b2",
                      "P": [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14]},
    "resnet18-int6": {"model": "resnet18", "weight_format": "int6", "activation_format": "int6", "recipe": "default",
                      "N": 6, "l8_case": "resnet18-int6-default-b2",
                      "P": [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]},
    "resnet18-w4a8": {"model": "resnet18", "weight_format": "int4", "activation_format": "int8",
                      "recipe": "default_no_bias_correction", "N": 8, "l8_case": None, "P": []},
}
CALIBRATION_IMAGES = 256
CHECK_IMAGES = 32


class _Stop(Exception):
    pass


class _Capture(B2Interpreter):
    """Runs the B2 graph up to ``target`` and keeps the tensor that feeds it."""

    def __init__(self, graph, quantizers, plan, scales, target):
        super().__init__(graph, quantizers, plan, scales)
        self.target, self.value = target, None

    def run_node(self, node):
        if node.name == self.target:
            args, _ = self.fetch_args_kwargs_from_env(node)
            self.value = args[0]
            raise _Stop
        return super().run_node(node)


def input_store(plan, node):
    """The quantizing node whose stored codes reach a MAC node (through flatten and code-forwarding max-pool)."""
    source = node.args[0]
    while not plan[source.name]["quantizes"]:
        if len(source.all_input_nodes) != 1:
            raise ValueError(f"{node.name}: input is not a stored code tensor")
        source = source.all_input_nodes[0]
    return source.name


def codes_digest(codes):
    h = hashlib.sha256()
    for name in sorted(codes):
        h.update(name.encode())
        h.update(np.ascontiguousarray(codes[name].cpu().numpy().astype("<i2")).tobytes())
    return h.hexdigest()


class Setup:
    """Frozen model, B2 activation recipe and weight scales of one case (nothing written)."""

    def __init__(self, case, device):
        from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
        from tools.experiment_b2.common import BIAS_CORRECTION_IMAGES
        from tools.experiment_b2.data import cached_inputs, v1_calibration
        from tools.experiment_b2.recipe import named
        from tools.experiment_b2.runner import model_setup
        self.case, self.spec, self.device = case, CASES[case], device
        spec = self.spec
        setup = model_setup(spec["model"], device)
        self.setup = setup
        self.graph, self.rows, self.inputs = setup["graph"], setup["rows"], setup["inputs"]
        self.recipe = named(spec["recipe"])
        self.arrays, self.maxima, self.calibration = v1_calibration(spec["model"])
        self.plan, self.quantizers, self.act_scales = activation_setup(
            self.graph, spec["activation_format"], self.recipe, self.arrays, self.maxima, device)
        if CALIBRATION_IMAGES != BIAS_CORRECTION_IMAGES:
            raise ValueError("calibration set must be the B2 bias-correction set")
        self.calib, self.calib_rows = cached_inputs("imagenet_calibration_2k", CALIBRATION_IMAGES, setup["transform"],
                                                    build=False)
        self.wq = TableQuantizer(spec["weight_format"], "signed", device)
        self.qmin, self.qmax = float(self.wq.levels.min()), float(self.wq.levels.max())
        self.layers = [(node, module) for node, module in weight_layers(self.graph)]
        self.scales, self.shaped, self.real = {}, {}, {}
        with torch.no_grad():
            for node, module in self.layers:
                values, shaped = weight_scale(module, spec["weight_format"], "mse_per_channel", self.wq)
                self.scales[node.name], self.shaped[node.name] = values, shaped
                # the float32 quotient that B2's TableQuantizer rounds, so that rounding it reproduces B2's nearest codes
                quotient = (module.weight.detach() / shaped).double()
                self.real[node.name] = quotient.reshape(module.weight.shape[0], -1)
        rtn = self.rtn_codes()
        for name, q in rtn.items():
            if not torch.equal(torch.round(self.real[name]).clamp(self.qmin, self.qmax).to(torch.int64), q):
                raise AssertionError(f"{name}: rounding the weight quotient does not reproduce B2's nearest codes")

    # -- weights -------------------------------------------------------------------------------------------------
    def rtn_codes(self):
        """B2's own nearest rounding (TableQuantizer) expressed as integers."""
        codes = {}
        with torch.no_grad():
            for node, module in self.layers:
                q = self.wq(module.weight.detach(), self.shaped[node.name]) / self.shaped[node.name]
                r = torch.round(q)
                if (q - r).abs().max() > 1e-3:
                    raise ValueError(f"{node.name}: B2 weights are not integers on the scale grid")
                codes[node.name] = r.to(torch.int64).reshape(module.weight.shape[0], -1)
        return codes

    def set_weights(self, graph, codes, only=None):
        with torch.no_grad():
            for node, module in weight_layers(graph):
                if only is not None and node.name not in only:
                    continue
                q = codes[node.name].reshape(module.weight.shape).to(torch.float32)
                module.weight.copy_(q * self.shaped[node.name])

    # -- calibration statistics ----------------------------------------------------------------------------------
    def input_codes(self, node, value):
        scale = self.act_scales[input_store(self.plan, node)]
        x = value / scale
        r = torch.round(x)
        if (x - r).abs().max() > 1e-3:
            raise ValueError(f"{node.name}: stored input is not on its code grid")
        return r

    def hessian(self, work, node, module, batch=16):
        """X X^T (float64, exact on the integer input codes) of ``node`` over the calibration images."""
        K = module.weight[0].numel()
        H = torch.zeros(K, K, dtype=torch.float64, device=self.device)
        rows = 0
        for start in range(0, len(self.calib), batch):
            tensor = torch.from_numpy(np.array(self.calib[start:start + batch], dtype=np.float32)).to(self.device)
            cap = _Capture(work, self.quantizers, self.plan, self.act_scales, node.name)
            try:
                with torch.inference_mode():
                    cap.run(tensor)
            except _Stop:
                pass
            x = self.input_codes(node, cap.value)
            if isinstance(module, nn.Conv2d):
                if module.groups != 1:
                    raise ValueError("grouped convolutions are not supported in v1")
                cols = nn.functional.unfold(x, module.kernel_size, dilation=module.dilation, padding=module.padding,
                                            stride=module.stride)  # [B, K, L]
                X = cols.transpose(0, 1).reshape(K, -1).double()
            else:
                X = x.reshape(x.shape[0], -1).t().double()
            H += X @ X.t()
            rows += X.shape[1]
        return 2 * H / rows

    def fit(self, method, P=None, log=print):
        """Sequential fit over all MAC nodes in topological order. ``method``: optq | axe | naive."""
        N = self.spec["N"]
        codes, infos = {}, {}
        work = copy.deepcopy(self.graph)
        for node, module in self.layers:
            tick = time.monotonic()
            W = self.real[node.name]
            if method == "naive":
                q, info = axe.naive(W, qmin=self.qmin, qmax=self.qmax, P=P, N=N)
            else:
                H = self.hessian(work, node, module)
                q, info = axe.optq(W, H, qmin=self.qmin, qmax=self.qmax, P=P if method == "axe" else None, N=N)
                del H
            if method != "optq" and not axe.bound_holds(q, P, N):
                raise AssertionError(f"{node.name}: bound violated by the {method} integers at P={P}")
            codes[node.name] = q
            self.set_weights(work, codes, only={node.name})
            beta, neg = axe.sign_sums(q)
            info.update({"seconds": round(time.monotonic() - tick, 2), "max_beta": int(beta.max()),
                         "max_neg": int(neg.max()), "zero_fraction": float((q == 0).double().mean())})
            infos[node.name] = info
        del work
        return codes, infos

    # -- deployment and evaluation -------------------------------------------------------------------------------
    def deploy(self, codes):
        result = copy.deepcopy(self.graph)
        self.set_weights(result, codes)
        report = None
        if self.recipe.bias_correction == "empirical":
            with torch.inference_mode():
                report = bias_correct(self.graph, result, self.plan, self.quantizers, self.act_scales, self.calib)
        elif self.recipe.bias_correction != "none":
            raise ValueError(self.recipe.bias_correction)
        return result, report

    def interpreter(self, result):
        return B2Interpreter(result, self.quantizers, self.plan, self.act_scales)

    def screen(self, result, batch):
        from tools.experiment_b2_recon.evaluate import readout, run_screen
        logits, top5 = run_screen(self.interpreter(result).run, self.inputs, self.rows, self.device, batch)
        ties, per_image = readout(logits, top5, self.rows)
        return ties, per_image, top5
