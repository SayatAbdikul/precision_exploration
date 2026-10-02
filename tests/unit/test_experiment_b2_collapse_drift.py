"""Lane Q3, protocol addendum 3: the input-drift diagnostic resolves the sealed arms and measures what it says."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tools.experiment_b2_collapse.drift import ARMS, LATE, UNSEALED_COUNTERPART, input_means, layer_rows, spec_for
from tools.experiment_b2_collapse.evaluate import label


def test_every_listed_arm_resolves_to_the_sealed_spec():
    for model, names in ARMS.items():
        for name in names:
            s = spec_for(model, name)
            if name in UNSEALED_COUNTERPART:
                assert s["correction"] == "none" and s["recipe"] == "default"
                assert label(s) == name
            else:
                assert label(s) == name
        assert LATE[model]


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.relu(self.conv(x)).mean(dim=(2, 3)))


def test_input_means_equal_direct_means_and_zero_drift_for_the_same_network():
    torch.manual_seed(0)
    net = _Tiny().eval()
    data = np.random.default_rng(0).standard_normal((40, 3, 5, 5)).astype(np.float32)
    means = input_means(net, net, ["conv", "fc"], data, "cpu", chunk=16)
    x = torch.from_numpy(data).double()
    assert torch.allclose(means["conv"], x.mean(dim=(0, 2, 3)))
    with torch.no_grad():
        hidden = torch.relu(net.conv(torch.from_numpy(data))).mean(dim=(2, 3)).double()
    assert torch.allclose(means["fc"], hidden.mean(dim=0), atol=1e-6)
    rows = layer_rows([("conv", "conv"), ("fc", "fc")], means, means)
    assert all(r["drift_mean"] == 0 and r["share_positive"] == 0 for r in rows.values())


def test_drift_sign_follows_the_quantized_network():
    torch.manual_seed(0)
    fp = _Tiny().eval()
    q = _Tiny().eval()
    q.load_state_dict(fp.state_dict())
    with torch.no_grad():
        q.conv.bias.sub_(5.0)  # every ReLU output smaller: the fc input is attenuated
    data = np.random.default_rng(1).standard_normal((16, 3, 5, 5)).astype(np.float32)
    rows = layer_rows([("fc", "fc")], input_means(fp, fp, ["fc"], data, "cpu"), input_means(q, q, ["fc"], data, "cpu"))
    assert rows["fc"]["drift_mean"] < 0 and rows["fc"]["share_positive"] == 0
