"""Lane Q4 r6 (protocol addendum 8): the unbiased per-channel sample of the pcu arms (CPU, synthetic data)."""
import math

import numpy as np
import pytest
import torch

from tools.experiment_b2_attrib import pcfull, pcunbiased
from tools.experiment_b2_attrib.pcurunner import parse_pcu


def _fake_plan():
    plan = {"x": {"kind": "input", "quantizes": True, "signedness": "signed"},
            "c1": {"kind": "conv", "quantizes": True, "signedness": "signed"},
            "h1": {"kind": "hardswish", "quantizes": True, "signedness": "signed"},
            "r1": {"kind": "relu", "quantizes": True, "signedness": "unsigned"},
            "fc": {"kind": "linear", "quantizes": True, "signedness": "signed"}}
    groups = {"role": {"stem": ["x", "c1"], "b": ["h1", "r1", "fc"]}, "position": {"p": ["x", "c1", "h1", "r1", "fc"]}}
    return plan, groups


class Identity(torch.nn.Module):
    def forward(self, x):
        y = x * 1.0
        return y


def _probe_inputs(images, channels, h, w):
    # value of (image i, channel c, row r, column k) encodes its position exactly (float32 holds these integers)
    i = torch.arange(images).view(-1, 1, 1, 1)
    c = torch.arange(channels).view(1, -1, 1, 1)
    r = torch.arange(h).view(1, 1, -1, 1)
    k = torch.arange(w).view(1, 1, 1, -1)
    return (((i * h + r) * w + k) + 0.25 * c).float().numpy()


def test_positions_are_a_seeded_sorted_sample_without_replacement():
    a = pcunbiased.positions("features_0_0", 256 * 112 * 112)
    b = pcunbiased.positions("features_0_0", 256 * 112 * 112)
    c = pcunbiased.positions("features_0_2", 256 * 112 * 112)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.numel() == 4096 and torch.unique(a).numel() == 4096 and bool((a[1:] > a[:-1]).all())
    assert torch.equal(pcunbiased.positions("se", 256), torch.arange(256))


@pytest.mark.parametrize("hw", [112, 56, 28])
def test_pcu_covers_the_map_where_the_pcf_stride_hit_border_columns(hw):
    per_image, n = hw * hw, 256 * hw * hw
    k = max(1, math.ceil(n / pcfull.MAX_VALUES))       # the pcf stride (review finding B1)
    pcf_columns = {int(p % per_image % hw) for p in range(0, n, k)}
    assert len(pcf_columns) <= 4                          # 112: {0}; 56: {0, 28}; 28: four columns
    pos = pcunbiased.positions("node", n)
    cover = pcunbiased.coverage(pos, per_image, (256, 8, hw, hw))
    assert cover["distinct_columns"] == hw and cover["distinct_rows"] == hw and cover["images_touched"] >= 250
    # inclusion is uniform: the mean sampled column is near the middle of the map (sd of the mean ~ hw / sqrt(12*4096))
    columns = (pos % per_image % hw).double()
    assert abs(float(columns.mean()) - (hw - 1) / 2) < 6 * hw / math.sqrt(12 * 4096)


def test_collect_keeps_exactly_the_sampled_positions_across_chunks():
    graph = torch.fx.symbolic_trace(Identity())
    node = [n.name for n in graph.graph.nodes if n.op == "call_function"][0]
    images, channels, h, w = 70, 3, 9, 11                 # 70 images: three chunks of 32, 32, 6
    inputs = list(_probe_inputs(images, channels, h, w))
    maxima, values, info = pcunbiased.collect(graph, [node], inputs, "cpu", m=500)[node]
    pos = pcunbiased.positions(node, images * h * w, 500)
    assert values.shape == (channels, 500) and info["positions_kept"] == 500
    for ch in range(channels):
        np.testing.assert_array_equal(values[ch], pos.numpy().astype(np.float32) + 0.25 * ch)
    np.testing.assert_allclose(maxima, [images * h * w - 1 + 0.25 * ch for ch in range(channels)])


def test_parse_pcu_is_the_pcf_twin_renamed():
    plan, groups = _fake_plan()
    arm, nodes = parse_pcu("pcu:all", plan, groups)
    assert arm.name == "pcu:all" and arm.note == "pcu" and nodes == ("c1", "h1", "r1") and arm.kind == "recipe"
    arm, nodes = parse_pcu("affine:signed+pcu:role.stem", plan, groups)
    assert nodes == ("c1",) and arm.affine == ("fc", "h1", "x") and arm.per_channel == ("c1",)
    assert parse_pcu("ref_default", plan, groups)[1] == ()
    for bad in ("pcu:all+pcf:node.h1", "pcu:node.x", "wide:node.c1+pcu:node.c1"):
        with pytest.raises(ValueError):
            parse_pcu(bad, plan, groups)


def test_holm_and_bootstrap_helpers_of_the_v2_summary():
    from tools.experiment_b2_attrib import summarize2
    adjusted = summarize2.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    x = list(range(13))
    stat = summarize2.spearman_ci(x, [v * 2.0 for v in x])
    assert stat["rho"] == 1.0 and stat["ci95"] == [1.0, 1.0]
    within = summarize2.within_cell([[{"x": i, "y": i} for i in range(6)], [{"x": i, "y": -i} for i in range(6)]], "x", "y")
    assert within["mean_within_cell_rho"] == 0.0 and within["per_cell_rho"] == [1.0, -1.0]
