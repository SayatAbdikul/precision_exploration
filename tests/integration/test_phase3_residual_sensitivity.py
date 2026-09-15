import pytest
import torch

from public.inference.tensor import Encoding, QUANTIZATION_OBSERVER
from public.quantization.calibration.observer import classifier_interpreter
from public.quantization.graph.executable import freeze_graph
from tools.phase3.sensitivity import NoIntervention
from tools.phase3.residual_sensitivity import OneResidual
from tools.phase3 import sensitivity


class Branches(torch.nn.Module):
    def forward(self, x):
        left = x * .5
        right = x * .5
        return (left + right) * 3


def graph():
    a, b = Encoding("int4").document(), Encoding("int4", ("0.5",)).document()
    return freeze_graph(inputs={"x": a}, constants={}, nodes=[
        {"name": "mul", "op": "activation", "inputs": ["x"], "attrs": {"function": "identity", "output": a}},
        {"name": "mul_1", "op": "activation", "inputs": ["x"], "attrs": {"function": "identity", "output": b}},
        {"name": "add", "op": "elementwise", "inputs": ["mul", "mul_1"],
         "attrs": {"operation": "add", "alignment": a, "output": a, "accumulator": "fp64_e11m52_accumulator"}},
    ], outputs=["add"], provenance={"kind": "test"})


def test_two_branch_stores_and_alignment_feed_the_fp32_tail():
    inputs = torch.tensor([[1.2]])
    intervention = OneResidual(graph(), "add", backend="reference")
    with torch.inference_mode():
        baseline = classifier_interpreter(Branches(), NoIntervention()).run(inputs)
        result = classifier_interpreter(Branches(), intervention).run(inputs)
    assert baseline.item() == pytest.approx(3.6)
    # Left .6 -> 1; right .6 -> .5 -> alignment 0 (RNE); sum 1, FP32 tail *3.
    assert result.item() == 3
    assert torch.equal(inputs, torch.tensor([[1.2]]))
    assert set(intervention.values) == {"mul", "mul_1"}
    assert len(intervention.graph["nodes"]) == 1
    assert not intervention.graph["constants"]
    assert QUANTIZATION_OBSERVER.get() is None


def test_branch_capture_survives_later_alias_mutation_and_missing_input_fails():
    intervention = OneResidual(graph(), "add", backend="reference")
    left = torch.tensor([[.6]])
    intervention.record("mul", left)
    left.fill_(100)
    assert intervention.values["mul"].item() == pytest.approx(.6)
    with pytest.raises(ValueError, match="coverage"):
        intervention.record("add", torch.zeros(1, 1))
    assert QUANTIZATION_OBSERVER.get() is None


def test_residual_failure_restores_observer_and_other_operations_are_rejected(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected residual failure")
    monkeypatch.setattr(sensitivity, "execute", fail)
    intervention = OneResidual(graph(), "add", backend="reference")
    with torch.inference_mode(), pytest.raises(RuntimeError, match="injected"):
        classifier_interpreter(Branches(), intervention).run(torch.tensor([[1.2]]))
    assert QUANTIZATION_OBSERVER.get() is None
    for name in ("mul", "missing"):
        with pytest.raises(ValueError, match="two-input tensor addition"):
            OneResidual(graph(), name)
