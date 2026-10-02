"""Lane Q3 r2: the driver's arm patches and the lean replacement of the heavy reference arms."""
from __future__ import annotations

import copy
import importlib

import pytest
import torch

from tools.experiment_b2 import frozen  # noqa: F401
from tools.experiment_b2.engine import bias_correct
from tools.experiment_b2_collapse.correct import staged_correct

from test_experiment_b2_collapse import biases, setup  # same folder (pytest prepend import mode)


def fresh_evaluate():
    from tools.experiment_b2_collapse import evaluate
    return importlib.reload(evaluate)


@pytest.mark.parametrize("model", ["resnet18", "mobilenet_v2", "mobilenet_v3_large"])
def test_patch_groups_skip_heavy_and_lean(model):
    from tools.run.experiment_b2_collapse_drive import patch_groups
    evaluate = fresh_evaluate()
    original = [evaluate.label(s) for s in evaluate.mode3_arms(model)]
    patch_groups(evaluate, reverse=True, skip_heavy=True, lean_maxabs=True)
    patched = [evaluate.label(s) for s in evaluate.GROUPS["mode3"](model)]
    heavy = {f"{fmt}-default_weight_maxabs-N--b2" for fmt in evaluate.WIDE}
    lean = [f"{fmt}-default_weight_maxabs-N--global" for fmt in evaluate.WIDE]
    assert heavy <= set(original) and not heavy & set(patched)
    assert set(patched) == (set(original) - heavy) | set(lean)
    assert patched == list(reversed([x for x in original if x not in heavy] + lean))
    # mode 1 has no heavy arm: same arms, reversed
    patched1 = [evaluate.label(s) for s in evaluate.GROUPS["mode1"](model)]
    assert patched1 == list(reversed([evaluate.label(s) for s in evaluate.mode1_arms(model)]))
    fresh_evaluate()


@pytest.mark.parametrize("fmt,recipe_name", [("fp8_e5m2", "default_weight_maxabs"), ("log6", "default_weight_maxabs"),
                                             ("fp6_e3m2", "default"), ("posit6_es1", "default_weight_maxabs")])
def test_lean_global_equals_b2_on_wide_exponent_formats(fmt, recipe_name):
    graph, q_graph, plan, quantizers, scales, inputs = setup(fmt, recipe_name)
    reference, mine = copy.deepcopy(q_graph), copy.deepcopy(q_graph)
    with torch.inference_mode():
        bias_correct(graph, reference, plan, quantizers, scales, inputs, chunk=16)
        staged_correct(graph, mine, plan, quantizers, scales, inputs, policy="global", chunk=16, store="cpu")
    for name, value in biases(reference).items():
        assert torch.equal(value, biases(mine)[name]), name
