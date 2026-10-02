"""Lane Q2 (tools/experiment_b2_rank): codebook rule, recipes, block engine regression (CPU)."""
import numpy as np
import pytest
import torch
from torch import nn

from tools.experiment_b2 import blocks, codebook
from tools.experiment_b2.recipe import NAMED
from tools.experiment_b2_rank import blocks_unsigned, cells, unsigned


@pytest.mark.parametrize("name", ["int8", "int6", "int5", "int4"])
def test_sibling_rule_gives_b2_unsigned_integer_codebook(name):
    mine, b2 = unsigned.sibling_table(name), codebook.unsigned_table(name)
    assert all(np.array_equal(a, b) for a, b in zip(mine, b2))


@pytest.mark.parametrize("name", unsigned.SCALAR + unsigned.BLOCK)
def test_sibling_codebooks_fit_and_refine(name):
    levels, boundaries, ties = unsigned.sibling_table(name)
    bits = unsigned.manifest(name)["bits"]
    assert levels[0] == 0 and np.all(np.diff(levels) > 0) and len(levels) <= 1 << bits
    assert len(boundaries) == len(ties) == len(levels) - 1
    facts = unsigned.describe(name)
    assert facts["contains_signed_levels"] and facts["levels"] > facts["signed_nonnegative_levels"]


def test_new_recipes_complete_the_factorial():
    d = NAMED["default"]
    corners = {(r.unsigned, r.weight_range, r.bias_correction) for r in
               (NAMED[n] for n in ("default", "default_signed", "default_no_bias_correction", "default_weight_maxabs",
                                   "cum3_act_mse", "minimal", "rank_signed_no_bias_correction",
                                   "rank_signed_weight_maxabs"))}
    assert len(corners) == 8
    for name in cells.NEW_RECIPES:
        r = NAMED[name]
        assert r.boundaries == d.boundaries and r.equalization == "none" and r.quantize_input and r.quantize_logits
    i = NAMED["cum5_act_maxabs"]
    assert NAMED["rank_intrinsic_no_bias_correction"] == i.__class__(**{**i.as_dict(), "bias_correction": "none"})
    assert NAMED["rank_intrinsic_weight_maxabs"] == i.__class__(**{**i.as_dict(), "weight_range": "maxabs"})


def toy():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 40, 3, padding=1),
                          nn.ReLU(), nn.Conv2d(40, 8, 1), nn.Flatten(), nn.Linear(8 * 4 * 4, 10))
    return torch.fx.symbolic_trace(model.eval())


@pytest.mark.parametrize("name,recipe", [("mxfp6_e3m2", "default"), ("bfp6", "cum5_act_maxabs"),
                                         ("mxfp4_e2m1", "cum1_fused")])
def test_block_engine_without_unsigned_is_bit_identical(name, recipe):
    graph, r = toy(), NAMED[recipe]
    x = torch.randn(4, 3, 8, 8)
    bias = x.numpy() if r.bias_correction != "none" else None
    with torch.inference_mode():
        a, ma = blocks.prepare_blocks(graph, name, r, "cpu", bias_inputs=bias)
        b, mb = blocks_unsigned.prepare_blocks(graph, name, r, "cpu", bias_inputs=bias, unsigned=False)
        assert torch.equal(a.run(x), b.run(x))
    assert ma == mb


def test_block_engine_uses_unsigned_codebook_at_nonnegative_nodes():
    graph, r = toy(), NAMED["cum5_act_maxabs"]
    x = torch.randn(4, 3, 8, 8)
    with torch.inference_mode():
        a, _ = blocks.prepare_blocks(graph, "mxfp4_e2m1", r, "cpu", bias_inputs=x.numpy())
        b, meta = blocks_unsigned.prepare_blocks(graph, "mxfp4_e2m1", r, "cpu", bias_inputs=x.numpy())
        assert not torch.equal(a.run(x), b.run(x))
    stored, patches = meta["rank_unsigned_blocks"]["unsigned_stored"], meta["rank_unsigned_blocks"]["unsigned_patches"]
    assert stored and patches and all(b.plan[s]["nonnegative"] for s in stored)
    assert len(b.unsigned_quantizer.levels) == 16 and float(b.unsigned_quantizer.levels.min()) == 0.0


def test_unsigned_variant_patch_is_scoped():
    before = (codebook.supports_unsigned("fp6_e2m3"), codebook.unsigned_table)
    with cells.unsigned_variant("fp6_e2m3"):
        assert codebook.supports_unsigned("fp6_e2m3") and codebook.supports_unsigned("int6")
        assert not codebook.supports_unsigned("fp6_e3m2")
        assert np.array_equal(codebook.unsigned_table("fp6_e2m3")[0], unsigned.sibling_table("fp6_e2m3")[0])
    assert (codebook.supports_unsigned("fp6_e2m3"), codebook.unsigned_table) == before


def test_unsigned_variant_keeps_block_plan_signed():
    """Regression (queue r1, 2026-10-02): the block unsigned variant must not mark plan rows unsigned."""
    from tools.experiment_b2 import codebook
    from tools.experiment_b2_rank.cells import unsigned_variant
    graph, r = toy(), NAMED["cum5_act_maxabs"]
    x = torch.randn(4, 3, 8, 8)
    with unsigned_variant("mxfp4_e2m1"):
        assert not codebook.supports_unsigned("mxfp4_e2m1")
        with torch.inference_mode():
            b, meta = blocks.prepare_blocks(graph, "mxfp4_e2m1", r, "cpu", bias_inputs=x.numpy())
        assert meta["rank_unsigned_blocks"]["unsigned_stored"]
    with unsigned_variant("fp4_e2m1"):
        assert codebook.supports_unsigned("fp4_e2m1")
