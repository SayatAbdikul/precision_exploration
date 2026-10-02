"""Hand-checkable tests of the AdaRound module of the B2 reconstruction baseline (lane L6)."""
import copy
import dataclasses
import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tools.experiment_b.classifier import ObservingInterpreter  # noqa: E402
from tools.experiment_b.quantizer import table  # noqa: E402
from tools.experiment_b2 import frozen  # noqa: E402,F401
from tools.experiment_b2.codebook import TableQuantizer  # noqa: E402
from tools.experiment_b2.engine import prepare_b2  # noqa: E402
from tools.experiment_b2.recipe import named  # noqa: E402
from tools.experiment_b2_recon import adaround, fit  # noqa: E402
from tools.experiment_b2_recon.engine import (build_engine, consumer_activation, quantize_weights,  # noqa: E402
                                              weight_layers, weight_scale)

# Fewer iterations with a proportionally larger step: Adam moves V by about lr per step, and a soft variable
# needs to travel log(11) = 2.4 to reach exactly 0 or 1.
FAST = {**adaround.SETTINGS, "iterations": 2000, "learning_rate": 1e-2}


class Tiny(nn.Module):
    """Every operator kind the classifiers use (the graph of the B2 unit tests)."""

    def __init__(self):
        super().__init__()
        self.c1, self.i1, self.r1 = nn.Conv2d(3, 8, 3, padding=1), nn.Identity(), nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.c2, self.i2, self.hs = nn.Conv2d(8, 8, 3, padding=1), nn.Identity(), nn.Hardswish()
        self.se_pool, self.fc, self.hsig = nn.AdaptiveAvgPool2d(1), nn.Conv2d(8, 8, 1), nn.Hardsigmoid()
        self.c3, self.r2 = nn.Conv2d(8, 8, 1), nn.ReLU6()
        self.avg, self.drop, self.lin = nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(), nn.Linear(8, 5)

    def forward(self, x):
        a = self.pool(self.r1(self.i1(self.c1(x))))
        b = self.hs(self.i2(self.c2(a)))
        b = self.hsig(self.fc(self.se_pool(b))) * b
        d = self.r2(self.c3(b) + a)
        return self.lin(self.drop(torch.flatten(self.avg(d), 1)))


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(7)
    graph = torch.fx.symbolic_trace(Tiny().eval())
    inputs = torch.randn(64, 3, 8, 8)
    observer = ObservingInterpreter(graph)
    with torch.inference_mode():
        observer.run(inputs)
    return graph, inputs, dict(observer.samples), dict(observer.maxima)


def test_rectified_sigmoid_is_eq_23():
    v = torch.tensor([-50.0, -3.0, 0.0, 1.0, 3.0, 50.0], dtype=torch.float64)
    h = adaround.rectified_sigmoid(v)
    assert h[0] == 0 and h[1] == 0 and h[4] == 1 and h[5] == 1  # reaches the ends at finite V
    assert float(h[2]) == pytest.approx(0.5, abs=1e-15)
    assert h[3] == pytest.approx(1.2 / (1 + math.exp(-1.0)) - 0.1)
    grid = torch.linspace(-6, 6, 241, dtype=torch.float64, requires_grad=True)
    values = adaround.rectified_sigmoid(grid)
    assert bool((values[1:] >= values[:-1]).all())
    values.sum().backward()
    inside = (values > 0) & (values < 1)
    assert bool((grid.grad[inside] > 0).all()) and bool((grid.grad[~inside] == 0).all())
    # The interior ends where sigmoid(V) * 1.2 - 0.1 leaves [0, 1]: |V| = log(11).
    assert float(grid[inside].abs().max()) < math.log(11) < float(grid[~inside].abs().min()) + 0.05


def test_initial_v_reproduces_the_fp32_weight():
    fraction = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.9, 1.0], dtype=torch.float64)
    assert torch.allclose(adaround.rectified_sigmoid(adaround.initial_v(fraction)), fraction, atol=1e-12)
    levels = torch.tensor(table("int4")[0])
    weight = torch.tensor([[0.33, -0.71, 0.05, 0.6]])
    scale = torch.tensor([[0.1]])
    lo, hi, fraction = adaround.neighbours(weight, scale, levels)
    soft = adaround.soft_weight(lo, hi, adaround.initial_v(fraction), scale)
    assert torch.allclose(soft, weight, atol=1e-6)


def test_regulariser_is_eq_24():
    assert float(adaround.regulariser(torch.tensor([0.0, 1.0, 1.0, 0.0]), 2.0)) == 0.0
    assert float(adaround.regulariser(torch.tensor([0.5, 0.5, 0.5]), 7.0)) == 3.0
    assert float(adaround.regulariser(torch.tensor([0.75]), 2.0)) == pytest.approx(1 - 0.5 ** 2)
    assert float(adaround.regulariser(torch.tensor([0.25, 0.9]), 4.0)) == pytest.approx((1 - 0.5 ** 4) + (1 - 0.8 ** 4))
    # A larger beta leaves the soft variable freer (the penalty is flat near 1 over a wider range).
    h = torch.tensor([0.3])
    assert float(adaround.regulariser(h, 20.0)) > float(adaround.regulariser(h, 2.0))
    h = torch.tensor([0.3, 0.8], requires_grad=True)
    adaround.regulariser(h, 2.0).backward()
    assert h.grad[0] > 0 and h.grad[1] < 0  # descending the penalty moves 0.3 down and 0.8 up


def test_beta_schedule():
    n = 1000
    assert adaround.beta_at(0, n) is None and adaround.beta_at(199, n) is None
    assert adaround.beta_at(200, n) == pytest.approx(20.0)
    assert adaround.beta_at(600, n) == pytest.approx(11.0)
    assert adaround.beta_at(n, n) == pytest.approx(2.0)
    values = [adaround.beta_at(step, n) for step in range(200, n)]
    assert all(a >= b for a, b in zip(values, values[1:])) and values[-1] < 2.001


def test_neighbours_equal_eq_22_on_an_integer_grid():
    levels = torch.tensor(table("int4")[0])
    n, p = -8.0, 7.0
    scale = torch.tensor(0.25)
    weight = torch.tensor([-3.0, -2.01, -2.0, -1.99, -0.13, 0.0, 0.13, 0.5, 1.7, 1.75, 1.76, 9.0])
    lo, hi, fraction = adaround.neighbours(weight, scale, levels)
    floor = torch.floor(weight / scale)
    for h in (0.0, 0.3, 1.0):
        paper = scale * torch.clamp(floor + h, n, p)
        ours = scale * (lo + h * (hi - lo))
        assert torch.equal(paper, ours)
    assert torch.equal(lo[[0, -1]], torch.tensor([n, p])) and torch.equal(hi[[0, -1]], torch.tensor([n, p]))
    assert fraction[2] == 0 and lo[2] == -8 and hi[2] == -7  # exactly on a level: lower neighbour is the level
    assert bool(((fraction >= 0) & (fraction <= 1)).all())


@pytest.mark.parametrize("name", ["fp6_e2m3", "posit8_es1", "int6"])
def test_neighbours_bracket_the_value_and_nearest_is_one_of_them(name):
    quantizer = TableQuantizer(name, "signed", "cpu")
    torch.manual_seed(3)
    weight = torch.randn(4, 50) * torch.tensor([[0.01], [0.3], [2.0], [40.0]])
    scale = torch.tensor([[0.02], [0.05], [0.5], [3.0]])
    lo, hi, fraction = adaround.neighbours(weight, scale, quantizer.levels)
    y = weight / scale
    inside = (y >= quantizer.levels[0]) & (y <= quantizer.levels[-1])
    assert bool((lo[inside] <= y[inside]).all()) and bool((y[inside] <= hi[inside]).all())
    position = torch.searchsorted(quantizer.levels, lo.contiguous())
    assert bool((quantizer.levels[(position + 1).clamp_max(len(quantizer.levels) - 1)][inside] >= hi[inside]).all())
    assert bool((hi[~inside] == lo[~inside]).all())
    up, nearest = adaround.nearest_up(lo, hi, weight, scale, quantizer)
    assert torch.equal(adaround.hard_weight(lo, hi, up, scale), nearest)  # round-to-nearest is one of the two choices


def identity(value):
    return value


def problem(weight, inputs, scale=1.0, name="int8"):
    quantizer = TableQuantizer(name, "signed", "cpu")
    weight = torch.tensor(weight, dtype=torch.float32)
    scale = torch.tensor(scale, dtype=torch.float32)
    forward = lambda w, x: nn.functional.linear(x, w)  # noqa: E731
    target = forward(weight, inputs)
    return quantizer, weight, scale, forward, (lambda index: target.index_select(0, index))


def test_known_optimum_three_equal_inputs():
    """W = [0.3, 0.45, 0.25] on the integer grid, the three inputs always equal.

    The layer computes ``t * (w1 + w2 + w3) = t``.  Nearest rounds every weight
    to 0 and the output is 0.  Exactly one weight rounded up gives sum 1 and
    zero error; that is the optimum (any other choice has error ``(k-1)^2 t^2``).
    """
    torch.manual_seed(0)
    t = torch.randn(256, 1)
    inputs = t.repeat(1, 3)
    quantizer, weight, scale, forward, target_of = problem([[0.3, 0.45, 0.25]], inputs)
    up, statistics = adaround.learn_rounding(weight, scale, quantizer.levels, forward, identity, inputs, target_of,
                                             seed=1, settings=FAST)
    lo, hi, _ = adaround.neighbours(weight, scale, quantizer.levels)
    learned = adaround.hard_weight(lo, hi, up, scale)
    assert int(up.sum()) == 1 and float(learned.sum()) == 1.0
    assert float((forward(learned, inputs) - forward(weight, inputs)).abs().max()) < 1e-6
    nearest = quantizer(weight, scale)
    assert float(nearest.abs().sum()) == 0.0
    assert statistics["unsettled_soft_variables"] == 0  # every h(V) is exactly 0 or 1 at the end


def test_against_exhaustive_search_on_a_small_correlated_layer():
    """2 x 6 weights, strongly correlated inputs: every one of the 64 choices per output row is enumerated.

    AdaRound is a relaxation and does not promise the optimum.  On this fixed
    problem it finds the best choice for one row and the second best of 64 for
    the other, where round-to-nearest is the second and the seventh best.
    """
    torch.manual_seed(5)
    mixing = torch.randn(6, 6) * 0.3 + torch.ones(6, 6) * 0.5
    inputs = torch.randn(512, 6) @ mixing
    # Step 0.05: the curvature of the reconstruction term per soft variable, s^2 E[x^2], is below 4 lambda,
    # the regime of real layers, in which the regulariser can force a binary solution (see adaround.py).
    raw = (torch.rand(2, 6) * 3 - 1.5) * 0.05
    quantizer, weight, scale, forward, target_of = problem(raw.tolist(), inputs, scale=0.05)
    up, statistics = adaround.learn_rounding(weight, scale, quantizer.levels, forward, identity, inputs, target_of,
                                             seed=2, settings=FAST)
    lo, hi, _ = adaround.neighbours(weight, scale, quantizer.levels)
    target = forward(weight, inputs)

    def error(candidate):
        return (forward(candidate, inputs) - target).pow(2).mean(dim=0)  # per output row

    every = [[], []]
    for bits in itertools.product([False, True], repeat=6):
        for row in range(2):
            mask = torch.zeros(2, 6, dtype=torch.bool)
            mask[row] = torch.tensor(bits)
            every[row].append(float(error(adaround.hard_weight(lo, hi, mask, scale))[row]))
    learned = error(adaround.hard_weight(lo, hi, up, scale))
    nearest = error(quantizer(weight, scale))

    def rank(value, row):
        return sum(other < float(value) - 1e-12 for other in every[row])

    assert [rank(learned[row], row) for row in range(2)] == [0, 1]
    assert [rank(nearest[row], row) for row in range(2)] == [1, 6]
    assert bool((learned < nearest).all())
    assert statistics["unsettled_soft_variables"] == 0


def test_converges_to_hard_decisions_and_is_deterministic():
    torch.manual_seed(11)
    inputs = torch.randn(256, 12)
    raw = torch.randn(4, 12) * 0.15
    quantizer, weight, scale, forward, target_of = problem(raw.tolist(), inputs, scale=0.05, name="int4")
    runs = [adaround.learn_rounding(weight, scale, quantizer.levels, forward, torch.relu, inputs,
                                    lambda index: torch.relu(forward(weight, inputs.index_select(0, index))),
                                    seed=9, settings=FAST) for _ in range(2)]
    assert torch.equal(runs[0][0], runs[1][0])
    statistics = runs[0][1]
    assert statistics["unsettled_soft_variables"] == 0 and statistics["far_from_binary_soft_variables"] == 0
    assert statistics["free_weights"] + statistics["clipped_weights"] == 48
    lo, hi, _ = adaround.neighbours(weight, scale, quantizer.levels)
    learned = adaround.hard_weight(lo, hi, runs[0][0], scale)
    codes = learned / scale
    assert float((codes - codes.round()).abs().max()) < 1e-4 and codes.round().min() >= -8 and codes.round().max() <= 7  # INT4 grid
    assert statistics["clipped_weights"] > 0  # some weights lie outside the grid and have no choice
    assert bool(((learned == lo * scale) | (learned == hi * scale)).all())  # only the two neighbours are reachable


def test_builder_is_the_b2_engine_for_nearest_rounding(tiny):
    graph, inputs, arrays, maxima = tiny
    for name, recipe_name in (("int8", "default"), ("int6", "default_no_bias_correction"), ("posit8_es1", "default"),
                              ("fp6_e2m3", "minimal"), ("int4", "minimal")):
        recipe = named(recipe_name)
        bias_inputs = inputs.numpy()[:32]
        with torch.inference_mode():
            reference, meta = prepare_b2(graph, name, recipe, arrays, maxima, "cpu", bias_inputs=bias_inputs)
            run, metadata, deployed = build_engine(graph, name, name, recipe, arrays, maxima, "cpu",
                                                   bias_inputs=bias_inputs)
            assert torch.equal(reference.run(inputs), run(inputs))
        assert metadata["activation_scales"] == meta["activation_scales"]
        assert metadata["weight_scales"] == meta["weight_scales"]


def test_mixed_precision_and_learned_weights_stay_on_their_grids(tiny):
    graph, inputs, arrays, maxima = tiny
    recipe = named("default_no_bias_correction")
    quantizer = TableQuantizer("int4", "signed", "cpu")
    learned = {}
    for node, module in weight_layers(graph):
        _, shaped = weight_scale(module, "int4", "mse_per_channel", quantizer)
        lo, hi, _ = adaround.neighbours(module.weight.detach(), shaped, quantizer.levels)
        up, _ = adaround.nearest_up(lo, hi, module.weight.detach(), shaped, quantizer)
        learned[node.name] = up
    with torch.inference_mode():
        nearest_run, _, nearest_graph = build_engine(graph, "int4", "int8", recipe, arrays, maxima, "cpu")
        learned_run, metadata, learned_graph = build_engine(graph, "int4", "int8", recipe, arrays, maxima, "cpu",
                                                            learned=learned)
        assert torch.equal(nearest_run(inputs), learned_run(inputs))  # nearest decisions reproduce nearest weights
        same_format, _, _ = build_engine(graph, "int8", "int8", recipe, arrays, maxima, "cpu")
        assert not torch.equal(same_format(inputs), nearest_run(inputs))
    assert all(row["differ_from_nearest"] == 0 for row in metadata["rounding"].values())
    for node, module in weight_layers(nearest_graph):
        _, shaped = weight_scale(graph.get_submodule(node.target), "int4", "mse_per_channel", quantizer)
        codes = module.weight / shaped
        assert float((codes - codes.round()).abs().max()) < 1e-4 and codes.round().min() >= -8 and codes.round().max() <= 7
    # Flipping one decision changes exactly that weight by one step of its channel.
    flipped = {key: value.clone() for key, value in learned.items()}
    first = weight_layers(graph)[0][0].name
    flipped[first][0, 0, 0, 0] = ~flipped[first][0, 0, 0, 0]
    scratch = copy.deepcopy(graph)
    _, report = quantize_weights(scratch, "int4", "mse_per_channel", "cpu", flipped)
    assert sum(row["differ_from_nearest"] for row in report.values()) == 1
    with torch.inference_mode():
        weights_only, _, _ = build_engine(graph, "int4", None, None, arrays, maxima, "cpu",
                                          weight_rule="mse_per_layer", bias_correction="none")
        corrected, _, _ = build_engine(graph, "int4", None, None, arrays, maxima, "cpu", weight_rule="mse_per_layer",
                                       bias_correction="empirical", bias_inputs=inputs.numpy())
        a, b, reference = weights_only(inputs), corrected(inputs), graph(inputs)
    # FP32 activations: empirical bias correction removes the mean error of the logits on its own images.
    assert float((b - reference).mean(dim=0).abs().max()) < 1e-4 < float((a - reference).mean(dim=0).abs().max())


def test_per_layer_scale_is_one_value_and_the_three_rules_differ(tiny):
    graph = tiny[0]
    quantizer = TableQuantizer("int4", "signed", "cpu")
    module = graph.get_submodule("c2")
    per_layer, _ = weight_scale(module, "int4", "mse_per_layer", quantizer)
    per_channel, _ = weight_scale(module, "int4", "mse_per_channel", quantizer)
    maxabs, _ = weight_scale(module, "int4", "maxabs_per_channel", quantizer)
    assert len(set(per_layer)) == 1 and len(per_layer) == 8 and len(set(per_channel)) > 1
    expected = [float(np.float32(float(channel.abs().max()) / 7.0)) for channel in module.weight.detach()]
    assert maxabs == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError):
        weight_scale(module, "int4", "something", quantizer)


def test_activation_function_of_each_layer(tiny):
    graph = tiny[0]
    found = {node.name: consumer_activation(graph, node)[0] for node, _ in weight_layers(graph)}
    assert found == {"c1": "relu", "c2": "hardswish", "fc": "hardsigmoid", "c3": "identity", "lin": "identity"}
    assert [node.name for node, _ in weight_layers(graph)] == ["c1", "c2", "fc", "c3", "lin"]


def test_layer_records_round_trip_and_are_written_once(tmp_path):
    torch.manual_seed(2)
    up = torch.rand(5, 3, 3, 3) > 0.5
    path = tmp_path / "000--layer.npz"
    fit.write_layer(path, up, [0.5, 0.25, 1.0, 2.0, 4.0], {"node": "layer", "weights": 135})
    loaded, scales, statistics = fit.read_layer(path)
    assert torch.equal(loaded, up) and scales.dtype == np.float32 and statistics == {"node": "layer", "weights": 135}
    assert path.stat().st_size < 1000  # packed bits, not floats
    with pytest.raises(FileExistsError):
        fit.write_layer(path, up, [1.0] * 5, {})
    assert torch.equal(fit.read_layer(path)[0], up) and not list(tmp_path.glob("*.partial"))


def test_sequential_asymmetric_fit_on_a_small_graph(tiny):
    """Two layers of the tiny graph: inputs are captured before the layer, the second layer sees the first one's
    learned weight, the learned error is not above nearest, and a repeated layer gives the same bits."""
    graph, inputs, _, _ = tiny
    settings = FAST
    quantizer = TableQuantizer("int4", "signed", "cpu")
    data = inputs.numpy()
    q_graph = copy.deepcopy(graph)
    fp_run, q_run = fit._CaptureFP(graph), fit._CaptureFP(q_graph)
    layers = weight_layers(graph)
    with torch.no_grad():
        captured = fit.layer_input(fp_run, "c2", inputs)
        expected = graph.pool(graph.r1(graph.i1(graph.c1(inputs))))
    assert torch.equal(captured, expected)
    results = []
    for index, (node, module) in enumerate(layers[:2]):
        up, scales, statistics = fit.learn_layer(index, node, graph, q_graph, fp_run, q_run, data, "int4",
                                                 "mse_per_channel", quantizer, "cpu", 100 + index, settings, 3.0)
        results.append((up, statistics))
        assert statistics["calibration_error_learned"] <= statistics["calibration_error_nearest"]
        assert statistics["far_from_binary_soft_variables"] == 0
        assert fit.tensor_sha(q_graph.get_submodule(node.target).weight) == statistics["learned_weight_sha256"]
    assert results[0][1]["activation"] == "relu" and results[1][1]["activation"] == "hardswish"
    assert results[0][1]["calibration_error_fp32_weight"] == 0.0   # first layer: same input on both sides
    assert results[1][1]["calibration_error_fp32_weight"] > 0.0    # second layer: its input comes from quantized c1
    assert results[1][1]["differ_from_nearest"] > 0
    with torch.no_grad():
        shifted = fit.layer_input(q_run, "c2", inputs)
    assert not torch.equal(shifted, captured)
    # Repeating the second layer from the same state reproduces its bits (what a resumed job relies on).
    again, _, _ = fit.learn_layer(1, layers[1][0], graph, q_graph, fp_run, q_run, data, "int4", "mse_per_channel",
                                  quantizer, "cpu", 101, settings, 3.0)
    assert torch.equal(again, results[1][0])
    assert dataclasses.is_dataclass(named("default"))
