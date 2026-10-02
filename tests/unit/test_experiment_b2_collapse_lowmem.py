"""Lane Q3 r4: the low-memory block bias correction is bit-identical to blocks.bias_correct_blocks (CPU proof).

The GPU proof is a redirected matrix cell reproducing an existing record (artifacts/experiment_b2_collapse/
matrix/checks/*--lowmem.json).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from tools.experiment_b2 import blocks, frozen  # noqa: F401
from tools.experiment_b2.recipe import named
from tools.experiment_b2_collapse.lowmem import bias_correct_blocks_lowmem, patched

from test_experiment_b2_collapse import Tiny  # noqa: E402  (tests/unit is on sys.path under pytest)


def _prepare(fmt, recipe_name, seed=0):
    torch.manual_seed(seed)
    graph = torch.fx.symbolic_trace(Tiny().eval())
    inputs = np.random.default_rng(seed).normal(size=(40, 3, 6, 6)).astype(np.float32)
    with torch.inference_mode():
        engine, metadata = blocks.prepare_blocks(copy.deepcopy(graph), fmt, named(recipe_name), torch.device("cpu"),
                                                 bias_inputs=inputs)
    return engine, metadata, inputs


@pytest.mark.parametrize("fmt,recipe_name", [("mxfp6_e3m2", "default"), ("mxfp6_e3m2", "default_weight_maxabs"),
                                             ("mxfp4_e2m1", "default"), ("bfp6", "default")])
def test_lowmem_block_correction_is_bit_identical(fmt, recipe_name):
    reference, ref_meta, inputs = _prepare(fmt, recipe_name)
    with patched():
        assert blocks.bias_correct_blocks is bias_correct_blocks_lowmem
        mine, my_meta, _ = _prepare(fmt, recipe_name)
    assert blocks.bias_correct_blocks is not bias_correct_blocks_lowmem      # restored
    assert "bias_correction" in ref_meta and ref_meta == my_meta              # report floats equal exactly
    for name in ("conv1", "conv2", "fc"):
        assert torch.equal(reference.module.get_submodule(name).bias, mine.module.get_submodule(name).bias), name
    x = torch.from_numpy(inputs[:8])
    with torch.inference_mode():
        assert torch.equal(reference.run(x), mine.run(x))


def test_lowmem_chunking_matches_original_for_other_chunk_sizes():
    """Same result as the original when both use a chunk size that does not divide the image count."""
    torch.manual_seed(1)
    graph = torch.fx.symbolic_trace(Tiny().eval())
    inputs = np.random.default_rng(1).normal(size=(21, 3, 6, 6)).astype(np.float32)
    recipe = named("default")
    with torch.inference_mode():
        _, meta = blocks.prepare_blocks(copy.deepcopy(graph), "mxfp6_e3m2", recipe, torch.device("cpu"),
                                        bias_inputs=inputs)
    plan = blocks.block_plan(graph, recipe, "mxfp6_e3m2")
    activation = blocks.B2BlockQuantizer("mxfp6_e3m2", recipe.activation_range, torch.device("cpu"))
    weight = blocks.B2BlockQuantizer("mxfp6_e3m2", recipe.weight_range, torch.device("cpu"))
    q_ref, q_mine = copy.deepcopy(graph), copy.deepcopy(graph)
    weights = {}
    with torch.inference_mode():
        for name in ("conv1", "conv2", "fc"):
            w = q_ref.get_submodule(name).weight
            weights[name] = weight(w.reshape(w.shape[0], -1)).reshape_as(w)
        a = blocks.bias_correct_blocks(graph, q_ref, plan, activation, weights, inputs, chunk=5)
        b = bias_correct_blocks_lowmem(graph, q_mine, plan, activation, weights, inputs, chunk=5)
    assert a == b
    for name in ("conv1", "conv2", "fc"):
        assert torch.equal(q_ref.get_submodule(name).bias, q_mine.get_submodule(name).bias)
