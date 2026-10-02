"""Stage-2 additions to B2: tie-aware readout, shared-exponent block engine, matrix helpers."""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tools.experiment_b2 import blocks, readout  # noqa: E402
from tools.experiment_b2 import frozen  # noqa: E402,F401
from tools.experiment_b2.recipe import V1_MAXABS, V1_PERCENTILE, named  # noqa: E402
from tools.experiment_b_ext.shared import BlockQuantizer, codebook, prepare_shared  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.i1, self.r1 = nn.Conv2d(3, 40, 3, padding=1), nn.Identity(), nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.c2, self.hs = nn.Conv2d(40, 40, 3, padding=1, groups=40), nn.Hardswish()
        self.se_pool, self.fc, self.hsig = nn.AdaptiveAvgPool2d(1), nn.Conv2d(40, 40, 1), nn.Hardsigmoid()
        self.c3, self.r2 = nn.Conv2d(40, 40, 1), nn.ReLU6()
        self.avg, self.drop, self.lin = nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(), nn.Linear(40, 7)

    def forward(self, x):
        a = self.pool(self.r1(self.i1(self.c1(x))))
        b = self.hs(self.c2(a))
        b = self.hsig(self.fc(self.se_pool(b))) * b
        d = self.r2(self.c3(b) + a)
        return self.lin(self.drop(torch.flatten(self.avg(d), 1)))


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(11)
    return torch.fx.symbolic_trace(Tiny().eval()), torch.randn(16, 3, 8, 8)


def _logits(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_readout_hand_checkable_cases(tmp_path):
    base = [0.0] * 8
    rows, labels = [], []
    rows.append([5, 1, 1, 1, 1, 1, 1, 1]); labels.append(0)       # unique maximum
    rows.append([5, 5, 1, 1, 1, 1, 1, 1]); labels.append(1)       # two maxima, label has the higher index
    rows.append([5, 5, 1, 1, 1, 1, 1, 1]); labels.append(0)       # two maxima, label has the lower index
    rows.append([9, 8, 7, 3, 3, 3, 3, 0]); labels.append(5)       # three above, label tied 4-way for two places
    rows.append([9, 8, 7, 6, 5, 4, 3, 2]); labels.append(5)       # five strictly above
    rows.append([-0.0, 0.0, -1, -1, -1, -1, -1, -1]); labels.append(1)  # signed zeros are equal
    rows.append(base); labels.append(7)                          # everything tied
    arrays = readout.batch_readout(_logits(rows), torch.tensor(labels))
    assert readout.check(arrays) == 7
    assert arrays["greater"].tolist() == [0, 0, 0, 3, 5, 0, 0]
    assert arrays["equal"].tolist() == [1, 2, 2, 4, 1, 2, 8]
    assert arrays["equal_lower"].tolist() == [0, 1, 0, 2, 0, 1, 7]
    assert arrays["tie_size"].tolist() == [1, 2, 2, 1, 1, 2, 8]
    assert arrays["argmax_lowest"].tolist() == [0, 0, 0, 0, 0, 0, 0]
    assert arrays["top5_lowest"][3].tolist() == [0, 1, 2, 3, 4]
    c = readout.credits(arrays)
    assert c["top1_expected"].tolist() == [1, .5, .5, 0, 0, .5, 1 / 8]
    assert c["top1_lowest_index"].tolist() == [1, 0, 1, 0, 0, 0, 0]
    assert c["top1_strict"].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert c["top1_optimistic"].tolist() == [1, 1, 1, 0, 0, 1, 1]
    assert c["top5_expected"].tolist() == [1, 1, 1, .5, 0, 1, 5 / 8]
    assert c["top5_lowest_index"].tolist() == [1, 1, 1, 0, 0, 1, 0]
    s = readout.summary(arrays)
    assert s["images_with_tied_top1"] == 4 and s["images_with_label_tied_across_the_top5_boundary"] == 2
    assert s["top1_expected_percent"] == pytest.approx(100 * (1 + .5 + .5 + .5 + 1 / 8) / 7)
    path = tmp_path / "r.npz"
    readout.save(path, arrays)
    loaded = readout.load(path)
    assert all(np.array_equal(loaded[k], arrays[k]) and loaded[k].dtype == readout.DTYPES[k] for k in readout.DTYPES)
    with pytest.raises(FileExistsError):
        readout.save(path, arrays)
    broken = dict(arrays, equal_lower=arrays["equal"].copy())
    with pytest.raises(ValueError):
        readout.check(broken)


def test_expected_credit_is_the_mean_over_random_tie_breaking():
    rng = np.random.default_rng(3)
    logits = rng.integers(0, 4, size=(6, 12)).astype(np.float32)   # many ties
    labels = np.array([0, 3, 5, 7, 9, 11])
    arrays = readout.batch_readout(torch.tensor(logits), torch.tensor(labels))
    c = readout.credits(arrays)
    draws, top1, top5 = 20000, np.zeros(6), np.zeros(6)
    for _ in range(draws):
        order = np.argsort(-(logits + rng.uniform(0, .5, size=logits.shape)), axis=1)   # random order inside ties
        top1 += order[:, 0] == labels
        top5 += (order[:, :5] == labels[:, None]).any(axis=1)
    assert np.abs(top1 / draws - c["top1_expected"]).max() < .015
    assert np.abs(top5 / draws - c["top5_expected"]).max() < .015


@pytest.mark.parametrize("name", blocks.SHARED)
def test_block_core_equals_the_extension_and_search_only_lowers_the_error(name):
    torch.manual_seed(5)
    values = torch.randn(3, 4, 70) * 3            # 70 = two full blocks and one of six elements
    values[0, 0, :] = 0
    top = float(codebook(name)[0][-1])
    values[1, 1, :32] = (torch.rand(32) * 1.8 - .9) * top
    values[1, 1, 31] = 1.03 * top                 # just above a power-of-two step: clipping it pays
    values[2, 2, 40] = -90.0
    reference = BlockQuantizer(name, "maxabs", "cpu")(values)
    plain = blocks.B2BlockQuantizer(name, "maxabs", "cpu")
    assert torch.equal(plain(values), reference)                       # the extension's own code
    assert torch.equal(plain.core(values, search=False), reference)    # the vectorised path
    assert torch.equal(blocks.B2BlockQuantizer(name, "percentile_99_9", "cpu")(values),
                       BlockQuantizer(name, "percentile_99_9", "cpu")(values))
    searched = blocks.B2BlockQuantizer(name, "mse", "cpu")
    searched.audit, searched.context = {}, "t"
    output = searched(values)

    def block_error(result):
        padded = torch.nn.functional.pad((result - values).double().pow(2), (0, 26))
        return padded.reshape(3, 4, 3, 32).sum(-1)
    assert (block_error(output) <= block_error(reference)).all()
    coarse = name in ("bfp6", "mxfp4_e2m1")   # wide-exponent minifloats gain nothing from clipping one element
    if coarse:
        assert block_error(output)[1, 1, 0] < block_error(reference)[1, 1, 0]   # that block is clipped by one octave
    assert torch.equal(output[0, 0], torch.zeros(70))
    rows = blocks.audit_rows(searched)["t"]
    assert rows["elements"] == values.numel() and rows["blocks"] == 3 * 4 * 3
    assert (0 < rows["fraction_blocks_clipped"] if coarse else 0 <= rows["fraction_blocks_clipped"]) and rows["sqnr_db"] > 0
    # Values that are levels at the max-abs exponent are reproduced exactly: no clipping is chosen.
    levels = torch.tensor(codebook(name)[0])
    exact = levels[torch.randint(0, len(levels), (5, 32))]
    exact[:, 0] = levels[-1]
    assert torch.equal(blocks.B2BlockQuantizer(name, "mse", "cpu")(exact), exact)


@pytest.mark.parametrize("name", blocks.SHARED)
def test_larger_exponents_are_never_better(name):
    levels = codebook(name)[0].astype(np.float64)
    low = levels[np.abs(levels) <= levels[-1] / 2]
    assert np.isin(2 * low, levels).all()


@pytest.mark.parametrize("name,recipe", [("bfp6", V1_MAXABS), ("mxfp4_e2m1", V1_PERCENTILE), ("mxfp8_e4m3", V1_MAXABS)])
def test_block_engine_with_switches_off_is_the_extension(tiny, name, recipe):
    graph, inputs = tiny
    with torch.inference_mode():
        old, _ = prepare_shared(graph, name, recipe.v1_equivalent(), "cpu")
        new, meta = blocks.prepare_blocks(graph, name, recipe, "cpu")
        assert all(torch.equal(old.weights[k], new.weights[k]) for k in old.weights)
        assert torch.equal(old.run(inputs), new.run(inputs))
    assert all(row["quantizes"] or row["reason"] == "passthrough" for row in meta["boundaries"].values())


def test_block_plan_fuses_relu_and_reblocks_maxpool(tiny):
    graph, _ = tiny
    plan = blocks.block_plan(graph, named("default"), "mxfp6_e3m2")
    assert not plan["c1"]["quantizes"] and plan["c1"]["reason"] == "fused_into:r1"
    assert plan["r1"]["quantizes"] and plan["pool"]["quantizes"] and plan["pool"]["reason"] is None
    assert not plan["add"]["quantizes"] and plan["r2"]["quantizes"]
    assert plan["c2"]["quantizes"] and plan["hs"]["quantizes"]           # Hardswish is not fused under fused_relu
    assert all(row["signedness"] in (None, "signed") for row in plan.values())
    with pytest.raises(ValueError):
        blocks.block_plan(graph, named("default"), "int8")
    with pytest.raises(ValueError):
        blocks.block_plan(graph, dataclasses.replace(named("default"), quantize_logits=False), "bfp6")


def _trace(interpreter, inputs, names):
    seen = {}

    class Probe(type(interpreter)):
        def run_node(self, node):
            result = super().run_node(node)
            if node.name in names:
                seen[node.name] = result.clone()
            return result
    interpreter.__class__ = Probe
    with torch.inference_mode():
        interpreter.run(inputs)
    return seen


def test_block_bias_correction_and_search_recipes(tiny):
    graph, inputs = tiny
    layers = ("c1", "c2", "fc", "c3", "lin")
    plain_recipe = named("minimal")
    fixed_recipe = dataclasses.replace(plain_recipe, bias_correction="empirical")
    with torch.inference_mode():
        plain, _ = blocks.prepare_blocks(graph, "mxfp4_e2m1", plain_recipe, "cpu")
        fixed, meta = blocks.prepare_blocks(graph, "mxfp4_e2m1", fixed_recipe, "cpu", bias_inputs=inputs.numpy())
    reference = _trace(torch.fx.Interpreter(graph), inputs, layers)
    before, after = _trace(plain, inputs, layers), _trace(fixed, inputs, layers)

    def gap(seen, name):
        # Conv outputs that are stored are compared after their own quantization, so use the staged definition:
        dims = [0] + list(range(2, seen[name].ndim))
        return float((seen[name].mean(dims) - reference[name].mean(dims)).abs().max())
    assert set(meta["bias_correction"]) == set(layers)
    assert gap(after, "c1") < 2e-5      # a fused convolution: its wide output is what the correction fits
    assert gap(before, "c1") > 1e-3
    for name in layers:
        assert torch.equal(plain.weights[name], fixed.weights[name])
    with pytest.raises(ValueError, match="needs calibration inputs"):
        blocks.prepare_blocks(graph, "mxfp4_e2m1", fixed_recipe, "cpu")
    # The weight search lowers the weight error of every layer or leaves it unchanged.
    with torch.inference_mode():
        searched, smeta = blocks.prepare_blocks(graph, "mxfp4_e2m1", named("default"), "cpu", bias_inputs=inputs.numpy())
        _, pmeta = blocks.prepare_blocks(graph, "mxfp4_e2m1", plain_recipe, "cpu")
    for name in layers:
        assert smeta["weights"][name]["sqnr_db"] >= pmeta["weights"][name]["sqnr_db"] - 1e-9
    bits = blocks.metadata_bits(searched, inputs[:1])
    # c1: 40 rows of K=27 -> 40 blocks; c2 depthwise: 40 rows of K=9; fc, c3: 40 rows of K=40 -> 2 blocks each; lin: 7 x 2.
    assert bits["weights"]["blocks"] == 40 + 40 + 80 + 80 + 14
    assert bits["weights"]["elements"] == 40 * 27 + 40 * 9 + 1600 + 1600 + 280
    assert bits["stored_activations_per_image"]["scale_bits_per_element"] > 8 / 32
