"""Guards and read-only verification helpers added after review 1 of Experiment B2."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tools.experiment_b.classifier import ObservingInterpreter  # noqa: E402
from tools.experiment_b2 import verify  # noqa: E402
from tools.experiment_b2.boundaries import analyze  # noqa: E402
from tools.experiment_b2.engine import prepare_b2  # noqa: E402
from tools.experiment_b2.guards import check_graph, structural_problems  # noqa: E402
from tools.experiment_b2.recipe import Recipe  # noqa: E402


class Residual(nn.Module):
    def __init__(self, shift=None, factor=None):
        super().__init__()
        self.conv, self.relu, self.head = nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 1)
        self.shift, self.factor = shift, factor

    def forward(self, x):
        a = self.relu(self.conv(x))
        b = a + a
        if self.shift is not None:
            b = b + self.shift
        if self.factor is not None:
            b = b * self.factor
        return self.head(b)


def test_guard_accepts_node_only_add_and_mul():
    graph = torch.fx.symbolic_trace(Residual().eval())
    assert structural_problems(graph) == [] and check_graph(graph)
    torchvision = pytest.importorskip("torchvision")
    for name in ("resnet18", "mobilenet_v2", "mobilenet_v3_large"):
        assert check_graph(torch.fx.symbolic_trace(getattr(torchvision.models, name)(weights=None).eval()))


@pytest.mark.parametrize("kwargs, node", [({"shift": -3.0}, "add_1"), ({"factor": -1.0}, "mul")])
def test_guard_rejects_literal_operands_that_the_analysis_cannot_see(kwargs, node):
    graph = torch.fx.symbolic_trace(Residual(**kwargs).eval())
    # The hazard: the structural analysis calls this tensor non-negative although it is not.
    plan = analyze(graph, Recipe(boundaries="fused_relu", unsigned=True), "int8")
    assert plan[node]["nonnegative"] and plan[node]["signedness"] == "unsigned"
    with torch.inference_mode():
        seen = {}

        class Trace(torch.fx.Interpreter):
            def run_node(self, n):
                seen[n.name] = super().run_node(n)
                return seen[n.name]

        Trace(graph).run(torch.randn(2, 3, 6, 6))
    assert float(seen[node].min()) < 0
    assert len(structural_problems(graph)) == 1 and node in structural_problems(graph)[0]
    with pytest.raises(ValueError, match="non-negativity analysis is unsound"):
        check_graph(graph)


def test_probe_reproduces_the_engine_and_measures_the_bias_residual():
    torch.manual_seed(3)
    graph = torch.fx.symbolic_trace(Residual().eval())
    inputs = torch.randn(16, 3, 8, 8)
    observer = ObservingInterpreter(graph)
    with torch.inference_mode():
        observer.run(inputs)
    recipe = Recipe(boundaries="fused_relu", unsigned=True, activation_range="mse", weight_range="mse",
                    bias_correction="empirical")
    with torch.inference_mode():
        engine, meta = prepare_b2(graph, "int4", recipe, dict(observer.samples), dict(observer.maxima), "cpu",
                                  bias_inputs=inputs.numpy())
        Probe = verify._probe_class()
        run = Probe(engine.module, engine.plan, engine)
        assert torch.equal(run.run(inputs), engine.run(inputs))
    fp = verify._channel_means(Probe(graph, engine.plan), inputs.numpy(), 16, "cpu")
    q = verify._channel_means(Probe(engine.module, engine.plan, engine), inputs.numpy(), 16, "cpu")
    assert set(fp) == {"conv", "head"} == set(meta["bias_correction"])
    with torch.inference_mode():
        direct = graph.conv(inputs).double().mean(dim=(0, 2, 3))
    torch.testing.assert_close(fp["conv"], direct, rtol=0, atol=1e-12)
    # On the images it was fitted on, in one batch, the corrected engine reproduces the FP32 channel means.
    for name in fp:
        assert float((fp[name] - q[name]).abs().max()) < 2e-5, name
    # No negative value reaches an unsigned quantizer in a graph the guard accepts.
    assert run.negative and sum(run.negative.values()) == 0


def test_vendor_census_tells_float_from_quantized_and_finds_standalone_relu6():
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.ra = nn.Conv2d(3, 4, 3, padding=1), nn.ReLU()
            self.b, self.rb = nn.Conv2d(4, 4, 3, padding=1), nn.ReLU6()

        def forward(self, x):
            return self.rb(self.b(self.ra(self.a(x))))

    torch.manual_seed(5)
    model, x = Net().eval(), torch.randn(4, 3, 8, 8)
    float_census = verify.detailed_census(torch.fx.symbolic_trace(model))
    assert float_census["float_conv_or_linear_nodes"] == ["a", "b"]
    torch.backends.quantized.engine = "x86"
    prepared = prepare_fx(model, get_default_qconfig_mapping("x86"), example_inputs=(x,))
    with torch.inference_mode():
        prepared(x)
    converted = convert_fx(prepared)
    census = verify.detailed_census(converted)
    assert census["float_conv_or_linear_nodes"] == []
    # FX fuses conv+ReLU into one quantized module but leaves ReLU6 as a module of its own.
    assert census["standalone_activation_modules"] == {"ReLU6": 1}
    assert any(path.endswith("ConvReLU2d") and "quantized" in path for path in census["modules"])
    codes = verify.code_census(converted, x)
    assert codes and all(row["distinct_codes"] >= 1 and 0 <= row["min_code"] <= row["max_code"] <= 255
                         for row in codes.values())
    assert np.isfinite([row["scale"] for row in codes.values()]).all()
    # The x86 kernels of the shipped default are arithmetically right on their own quantized operands:
    # conv outputs within one code of the FP32 evaluation, ReLU6 exactly a clamp of the codes.
    audit = verify.kernel_audit(converted, x)
    assert [row["kind"] for row in audit] == ["conv", "conv", "relu6"]
    assert all(row["max_code_error"] <= 1 for row in audit) and audit[-1]["fraction_codes_not_equal"] == 0.0


def test_kernel_audit_flags_arithmetic_that_disagrees_with_its_operands():
    class Wrong(nn.Module):
        """A ReLU6 stand-in that returns codes shifted by 3, as a broken clamp kernel would."""

        def forward(self, x):
            codes = (x.int_repr().int().clamp(40, 100) + 3).to(torch.uint8)  # zero point 40, 6.0 is code 100
            return torch._make_per_tensor_quantized_tensor(codes, x.q_scale(), x.q_zero_point())

    Wrong.__name__ = "ReLU6"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = Wrong()

        def forward(self, x):
            return self.act(x)

    class LeafTracer(torch.fx.Tracer):
        def is_leaf_module(self, module, qualified_name):
            return True

    model = Net()
    graph = torch.fx.GraphModule(model, LeafTracer().trace(model))
    x = torch.quantize_per_tensor(torch.linspace(-3, 9, 64).reshape(1, 1, 8, 8), 0.1, 40, torch.quint8)
    rows = verify.kernel_audit(graph, x)
    assert len(rows) == 1 and rows[0]["kind"] == "relu6" and rows[0]["max_code_error"] == 3
    assert rows[0]["fraction_code_error_above_1"] > 0.9
