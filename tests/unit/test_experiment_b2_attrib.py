"""Lane Q4 (activation attribution): CPU tests of the arm engine, the codes and the group definitions."""
import numpy as np
import pytest
import torch

from tools.experiment_b.classifier import sample_indices
from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2.recipe import named
from tools.experiment_b2_attrib import quant
from tools.experiment_b2_attrib.arms import parse_arm


def test_affine_with_zero_point_zero_is_the_b2_code():
    base = TableQuantizer("int8", "signed", "cpu")
    x = torch.linspace(-3, 9, 4001)
    affine = quant.AffineQuantizer(base, 0.05, 0.0)
    assert torch.equal(affine(x, torch.tensor(np.float32(0.05))), base(x, torch.tensor(np.float32(0.05))))


def test_affine_code_keeps_zero_exact_and_beats_symmetric_on_hardswish_range():
    base = TableQuantizer("int8", "signed", "cpu")
    rng = np.random.default_rng(0)
    pre = torch.tensor(rng.normal(0, 2.5, 200000), dtype=torch.float32)
    x = torch.nn.functional.hardswish(pre)
    from tools.experiment_b2.scales import mse_search
    sym, _ = mse_search(x.reshape(1, -1), base)
    params, info = quant.affine_search(x, base, float(sym[0]))
    assert params is not None and params["zero"] < 0 and info["affine_sse"] < info["symmetric_sse"]
    q = quant.AffineQuantizer(base, params["scale"], params["zero"])
    assert float(q(torch.zeros(3), torch.tensor(q.scale)).abs().max()) == 0.0
    # Lowest calibration value is representable (no clipping at the bottom of the range).
    assert float(q(x.min().reshape(1), torch.tensor(q.scale))[0]) <= float(x.min()) + q.scale / 2


def test_affine_quantizer_rejects_foreign_scale_and_non_level_zero():
    base = TableQuantizer("int8", "signed", "cpu")
    q = quant.AffineQuantizer(base, 0.1, -100.0)
    with pytest.raises(ValueError):
        q(torch.zeros(2), torch.tensor(0.2))
    with pytest.raises(ValueError):
        quant.AffineQuantizer(base, 0.1, -0.5)
    with pytest.raises(ValueError):
        quant.AffineQuantizer(base, 0.1, 3.0)


def test_zero_point_rule():
    levels = np.arange(-128, 128, dtype=np.float32)
    assert quant.zero_point_for(levels, -0.375, 0.0328) == -116.0  # ceil(-128 + 11.43)
    assert quant.zero_point_for(levels, 0.0, 0.1) == -128.0
    assert quant.zero_point_for(levels, -100.0, 0.1) == 0.0       # range needs more than the negative half


def test_affine_search_falls_back_for_symmetric_data():
    base = TableQuantizer("int8", "signed", "cpu")
    x = torch.tensor(np.random.default_rng(1).normal(0, 1, 100000), dtype=torch.float32)
    from tools.experiment_b2.scales import mse_search
    sym, _ = mse_search(x.reshape(1, -1), base)
    params, info = quant.affine_search(x, base, float(sym[0]))
    assert params is None and info["reason"]


def test_channel_samples_regroup_the_v1_sampler():
    channels, spatial, images = 300, 49, 40
    rows = []
    for ordinal in range(images):
        tensor = np.add.outer(np.arange(channels) * 1.0, np.zeros(spatial)) + 1000.0 * ordinal
        ch, location = sample_indices(channels, spatial, ordinal)
        rows.append(tensor[ch, location])
    samples, info = quant.channel_samples(np.concatenate(rows), channels, images)
    assert samples.shape[0] == channels and info["per_channel_samples"] >= 1
    assert np.all(np.mod(samples, 1000.0) == np.arange(channels)[:, None])
    small, _ = quant.channel_samples(np.concatenate([np.add.outer(np.arange(16.0), np.zeros(4))[sample_indices(16, 4, i)]
                                                     for i in range(5)]), 16, 5)
    assert small.shape == (16, 20) and np.all(small == np.arange(16.0)[:, None])


def test_per_channel_quantizer_broadcasts_over_channels():
    base = TableQuantizer("int8", "signed", "cpu")
    scales = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    q = quant.PerChannelQuantizer(base, scales, 0.1)
    x = torch.randn(2, 3, 4, 4)
    out = q(x, torch.tensor(np.float32(0.1)))
    for c, s in enumerate(scales):
        assert torch.equal(out[:, c], base(x[:, c], torch.tensor(s)))
    with pytest.raises(ValueError):
        q(x, torch.tensor(0.5))


def _fake_plan():
    plan = {"x": {"kind": "input", "quantizes": True, "signedness": "signed"},
            "c1": {"kind": "conv", "quantizes": True, "signedness": "signed"},
            "h1": {"kind": "hardswish", "quantizes": True, "signedness": "signed"},
            "r1": {"kind": "relu", "quantizes": True, "signedness": "unsigned"},
            "id": {"kind": "identity", "quantizes": False, "signedness": None},
            "fc": {"kind": "linear", "quantizes": True, "signedness": "signed"}}
    groups = {"role": {"a": ["x", "c1"], "b": ["h1", "r1", "fc"]}, "position": {"p": ["x", "c1", "h1", "r1", "fc"]}}
    return plan, groups


def test_arm_grammar():
    plan, groups = _fake_plan()
    assert parse_arm("ref_default", plan, groups).wide == ()
    assert set(parse_arm("ref_weights_only", plan, groups).wide) == {"x", "c1", "h1", "r1", "fc"}
    only = parse_arm("only:role.a", plan, groups)
    assert set(only.wide) == {"h1", "r1", "fc"} and only.kind == "diagnostic"
    assert parse_arm("wide:node.h1", plan, groups).wide == ("h1",)
    affine = parse_arm("affine:role.b", plan, groups)
    assert affine.affine == ("fc", "h1") and affine.kind == "recipe"          # unsigned r1 is skipped
    pc = parse_arm("pc:all", plan, groups)
    assert pc.per_channel == ("c1", "h1", "r1")                                 # never the input or a linear output
    assert parse_arm("affine:kind.hardswish+pc:kind.conv", plan, groups).per_channel == ("c1",)
    for bad in ("wide:node.id", "nope:all", "wide:role.zzz", "affine:kind.relu", "wide:node.h1+affine:node.h1"):
        with pytest.raises(ValueError):
            parse_arm(bad, plan, groups)


@pytest.fixture(scope="module")
def resnet():
    from tools.experiment_b.classifier import load_model
    graph, _, _ = load_model("resnet18", "cpu")
    return graph


def test_groups_partition_all_models(resnet):
    from tools.experiment_b.classifier import load_model
    from tools.experiment_b2.boundaries import analyze
    from tools.experiment_b2_attrib.groups import partition
    sizes = {"mobilenet_v3_large": (121, {"hswish": 19, "se_product": 8, "dw_conv": 9, "residual_add": 10}),
             "mobilenet_v2": (65, {"project": 17, "residual_add": 10}),
             "resnet18": (31, {"block_out": 8, "downsample": 3})}
    for model, (total, some) in sizes.items():
        graph = resnet if model == "resnet18" else load_model(model, "cpu")[0]
        plan = analyze(graph, named("default"), "int8")
        parts = partition(model, graph, plan)
        for scheme in ("role", "position"):
            assert sum(len(v) for v in parts[scheme].values()) == total
        for group, size in some.items():
            assert len(parts["role"][group]) == size


def test_shared_engine_with_empty_arm_equals_prepare_b2(resnet):
    """The arm engine (shared preparation + empty arm) is prepare_b2 bit for bit (synthetic calibration)."""
    from tools.experiment_b2.boundaries import analyze, quantizing_nodes
    from tools.experiment_b2.engine import prepare_b2
    from tools.experiment_b2_attrib.engine import Arm, build_arm, prepare_shared
    torch.manual_seed(0)
    recipe = named("default")
    plan = analyze(resnet, recipe, "int6")
    rng = np.random.default_rng(0)
    arrays, maxima = {}, {}
    for node in quantizing_nodes(plan):
        values = rng.normal(0, 1, 512).astype(np.float32)
        if plan[node]["signedness"] == "unsigned":
            values = np.abs(values)
        arrays[node], maxima[node] = values, float(np.abs(values).max())
    bias_inputs = rng.normal(0, 1, (4, 3, 64, 64)).astype(np.float32)
    probe = torch.from_numpy(bias_inputs[:1])
    with torch.inference_mode():
        reference, meta = prepare_b2(resnet, "int6", recipe, arrays, maxima, "cpu", bias_inputs=bias_inputs)
        shared = prepare_shared("resnet18", resnet, "int6", recipe, arrays, maxima, "cpu", probe)
        engine, _ = build_arm(shared, Arm("ref_default"), bias_inputs)
    assert shared.activation_scales == meta["activation_scales"] and shared.weight_scales == meta["weight_scales"]
    for a, b in zip(reference.module.state_dict().values(), engine.module.state_dict().values()):
        assert torch.equal(a, b)
    x = torch.from_numpy(bias_inputs[1:3])
    with torch.inference_mode():
        assert torch.equal(reference.run(x), engine.run(x))
        # A wide arm differs, and keeping every boundary wide leaves only weight quantization.
        wide, meta_wide = build_arm(shared, Arm("w", wide=tuple(quantizing_nodes(plan))), bias_inputs)
        assert meta_wide["quantizing_boundaries"] == 0
        assert not torch.equal(wide.run(x), engine.run(x))
        # Per-channel scales with all ratios 1 and folds reproduce the default weights of the consumer.
        node = "layer1_0_relu"
        shared.channel_cache[node] = (np.full(64, shared.activation_scales[node], np.float32), {})
        pc, meta_pc = build_arm(shared, Arm("pc", per_channel=(node,)), bias_inputs)
        assert [f["consumer"] for f in meta_pc["folds"]] == ["layer1_0_conv2"]
        assert torch.equal(pc.module.get_submodule("layer1.0.conv2").weight,
                           engine.module.get_submodule("layer1.0.conv2").weight)
        assert torch.equal(pc.run(x), engine.run(x))


def test_code_sums_and_affine_width_rule():
    from tools.experiment_b2_attrib.engine import code_sums
    base = TableQuantizer("int8", "signed", "cpu")
    w = torch.tensor([[0.5, -0.25, 0.1], [1.0, 1.0, -1.0]])
    scales = torch.tensor([[0.01], [0.02]])
    abs_sum, total = code_sums(base(w, scales), scales)
    assert abs_sum == [85, 150] and total == [35, 50]
    # zero point term: 128 * 85 + 117 * 35 needs one bit more than 128 * 85 alone? (closed form only)
    assert (128 * 85).bit_length() + 1 == 15 and (128 * 85 + 117 * 35).bit_length() + 1 == 15
