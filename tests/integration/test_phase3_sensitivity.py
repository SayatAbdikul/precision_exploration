import pytest
import torch

from public.inference.tensor import Encoding, Tensor, QUANTIZATION_OBSERVER
from public.quantization.calibration.observer import classifier_interpreter
from public.quantization.graph.executable import freeze_graph
from tools.phase3.sensitivity import OneLayer, NoIntervention
from tools.phase3 import sensitivity


class TwoLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(2, 2, bias=False)
        self.last = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(2))
            self.last.weight.copy_(torch.tensor([[1., 2.]]))

    def forward(self, x):
        return self.last(self.first(x))


def graph():
    domain = Encoding("int4")
    return freeze_graph(inputs={"x": domain.document()},
        constants={"w": Tensor.quantize([1, 0, 0, 1], (2, 2), domain)},
        nodes=[{"name": "first", "op": "linear", "inputs": ["x", "w"],
                "attrs": {"accumulator": "int32_accumulator", "output": domain.document()}},
               {"name": "view", "op": "flatten", "inputs": ["first"], "attrs": {"start_dim": 1}}],
        outputs=["view"], provenance={"kind": "test"})


def test_only_selected_mac_changes_and_fp32_propagates_its_result():
    model = TwoLayers().eval()
    inputs = torch.tensor([[.6, .6]])
    intervention = OneLayer(graph(), "first", backend="reference")
    with torch.inference_mode():
        baseline = classifier_interpreter(model, NoIntervention()).run(inputs)
        result = classifier_interpreter(model, intervention).run(inputs)
    assert baseline.item() == pytest.approx(1.8)
    assert result.item() == 3.0  # selected layer stores [1,1], FP32 last layer gives 3
    assert torch.equal(inputs, torch.tensor([[.6, .6]]))
    assert intervention.result["diagnostics"]["mse"] == pytest.approx(.16)
    assert set(intervention.graph["constants"]) == {"w"}
    assert len(intervention.graph["nodes"]) == 1
    assert QUANTIZATION_OBSERVER.get() is None


def test_failure_restores_quantization_observer(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected sensitivity failure")
    monkeypatch.setattr(sensitivity, "execute", fail)
    intervention = OneLayer(graph(), "first", backend="reference")
    with torch.inference_mode(), pytest.raises(RuntimeError, match="injected"):
        classifier_interpreter(TwoLayers().eval(), intervention).run(torch.tensor([[.6, .6]]))
    assert QUANTIZATION_OBSERVER.get() is None


def test_view_aliases_and_missing_layers_are_not_mac_interventions():
    for target in ("view", "missing"):
        with pytest.raises(ValueError, match="MAC layer"):
            OneLayer(graph(), target)
