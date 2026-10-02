"""Lane Q3, mode 2 (H2.1): the posit tables and the max-abs weight scale have no slip.

An independent posit decoder (regime, exponent, fraction from the bit pattern) must give exactly the
level tables B2 uses; B2 rounds to the nearest level by value; the max-abs per-channel weight scale of the
minimal recipe maps every channel maximum exactly to maxpos (the review of 2026-10-02 asked for this check
to be saved as a test).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tools.experiment_b2 import frozen  # noqa: F401
from tools.experiment_b2.codebook import TableQuantizer
from tools.experiment_b2_recon.engine import weight_scale


def _posit(bits, n, es):
    if bits == 0:
        return 0.0
    if bits == 1 << (n - 1):
        return None  # NaR
    sign = bits >> (n - 1)
    if sign:
        bits = (-bits) & ((1 << n) - 1)
    s = format(bits, f"0{n}b")[1:]
    first = s[0]
    run = len(s) - len(s.lstrip(first))
    k = run - 1 if first == "1" else -run
    rest = s[run + 1:]
    e = int(rest[:es].ljust(es, "0"), 2) if es else 0
    frac = rest[es:]
    f = 1 + (int(frac, 2) / (1 << len(frac)) if frac else 0)
    v = (2 ** (2 ** es)) ** k * 2 ** e * f
    return -v if sign else v


@pytest.mark.parametrize("name,n,levels,maxpos", [("posit8_es1", 8, 255, 4096.0), ("posit6_es1", 6, 63, 256.0)])
def test_tables_equal_an_independent_decoder(name, n, levels, maxpos):
    mine = np.array(sorted(v for v in (_posit(b, n, 1) for b in range(1 << n)) if v is not None))
    table = TableQuantizer(name, "signed", torch.device("cpu")).levels.double().numpy()
    assert len(mine) == levels == len(table)
    assert np.array_equal(mine, table)
    assert table.max() == maxpos


def test_rounding_is_value_nearest_in_the_regime_only_top():
    q = TableQuantizer("posit8_es1", "signed", torch.device("cpu"))
    w = torch.tensor([0.19 * 4096 + 1, 2100.0, 0.62 * 4096, 0.64 * 4096, 4096.0])
    out = q(w, torch.tensor(1.0)).tolist()
    # top positive levels ... 512, 1024, 4096: 0.19..0.625 of maxpos -> 1024 (= 0.25 of maxpos), above -> 4096
    assert out == [1024.0, 1024.0, 1024.0, 4096.0, 4096.0]


@pytest.mark.parametrize("name,maxpos", [("posit8_es1", 4096.0), ("posit6_es1", 256.0)])
def test_maxabs_per_channel_maps_every_channel_maximum_to_maxpos(name, maxpos):
    torch.manual_seed(0)
    module = nn.Conv2d(3, 5, 3)
    with torch.no_grad():
        module.weight.mul_(torch.linspace(0.1, 3.0, 5).reshape(5, 1, 1, 1))
    q = TableQuantizer(name, "signed", torch.device("cpu"))
    scales, shaped = weight_scale(module, name, "maxabs_per_channel", q)
    maxima = module.weight.detach().reshape(5, -1).abs().amax(dim=1)
    assert torch.allclose(maxima / torch.tensor(scales), torch.full((5,), maxpos), rtol=1e-6)
    out = q(module.weight.detach(), shaped)
    assert torch.allclose(out.reshape(5, -1).abs().amax(dim=1), maxima, rtol=1e-6)
