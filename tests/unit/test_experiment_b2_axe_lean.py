"""Lane Q8: the lean bias correction is a drop-in replacement of B2's (the bit-identity check on ResNet18 is a GPU/CPU
run recorded in docs/analysis/accumulator-aware-ptq-2026-10-02.md; here only the interface and the in-place guard)."""
import inspect

import torch

from tools.experiment_b2.engine import bias_correct
from tools.experiment_b2_axe import lean


def test_same_signature_as_b2():
    assert inspect.signature(lean.bias_correct_lean) == inspect.signature(bias_correct)


def test_parked_tensors_survive_inplace_modules():
    graph = torch.fx.symbolic_trace(torch.nn.Sequential(torch.nn.ReLU(inplace=True)))
    node = [n for n in graph.graph.nodes if n.op == "call_module"][0]
    parked = torch.tensor([[-1.0, 2.0]])
    env = {node.args[0].name: [parked]}
    out = lean._values(graph, env, node, 0, torch.device("cpu"))
    assert torch.equal(out, torch.tensor([[0.0, 2.0]])) and torch.equal(parked, torch.tensor([[-1.0, 2.0]]))
