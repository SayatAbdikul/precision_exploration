"""Lane S1 part 2: the host-memory bias correction equals blocks.bias_correct_blocks (CPU, tiny graph)."""
import copy
import numpy as np
import torch
import torch.nn as nn
from tools.experiment_b2 import blocks
from tools.experiment_b2_fastblocks import bias_correct_blocks as fast


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1); self.c2 = nn.Conv2d(8, 8, 3, padding=1, groups=2)
        self.fc = nn.Linear(8 * 6 * 6, 5)

    def forward(self, x):
        y = torch.relu(self.c1(x))
        y = self.c2(y) + y
        return self.fc(torch.flatten(y, 1))


def test_fast_bias_correction_is_bit_identical_on_cpu():
    torch.manual_seed(0)
    graph = torch.fx.symbolic_trace(Tiny().eval())
    q = blocks.B2BlockQuantizer('mxfp8_e4m3', 'mse', 'cpu')
    kinds = {'c1': 'conv', 'c2': 'conv', 'fc': 'linear'}
    plan = {n.name: {'quantizes': n.op != 'output', 'kind': kinds.get(n.name, 'other')} for n in graph.graph.nodes}
    weights = {}
    for name in ('c1', 'c2', 'fc'):
        w = graph.get_submodule(name).weight.detach()
        weights[name] = blocks.B2BlockQuantizer('mxfp8_e4m3', 'maxabs', 'cpu')(w.reshape(w.shape[0], -1)).reshape_as(w)
    inputs = np.random.default_rng(1).standard_normal((20, 3, 6, 6)).astype(np.float32)
    a, b = copy.deepcopy(graph), copy.deepcopy(graph)
    with torch.inference_mode():
        ra = blocks.bias_correct_blocks(graph, a, plan, q, weights, inputs, chunk=8)
        rb = fast(graph, b, plan, q, weights, inputs, chunk=8)
    assert ra == rb and set(ra) == {'c1', 'c2', 'fc'}
    for name in ('c1', 'c2', 'fc'):
        assert torch.equal(a.get_submodule(name).bias, b.get_submodule(name).bias)
        assert not torch.equal(a.get_submodule(name).bias, graph.get_submodule(name).bias)
