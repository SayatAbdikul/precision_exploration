"""Lane Q8 addendum-2: the accumulator-aware per-channel scale (CPU, synthetic layers)."""
import pytest
import torch

from tools.experiment_b2_axe import axe
from tools.experiment_b2_axe.rescale import _codes, _feasible, rescale_factors


def _weights(C=8, K=300, seed=0, spread=40.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(C, K, generator=g, dtype=torch.float64) * spread * torch.rand(C, 1, generator=g)


@pytest.mark.parametrize("P,N", [(14, 8), (16, 8), (18, 8), (12, 6), (14, 6)])
def test_rescaled_codes_meet_the_bound_with_the_smallest_factor(P, N):
    W = _weights(seed=P * N)
    qmin, qmax = -127.0, 127.0
    limit = axe.integer_limit(P, N)
    k, Q = rescale_factors(W, qmin=qmin, qmax=qmax, P=P, N=N)
    assert axe.bound_holds(Q, P, N)
    assert torch.equal(Q, _codes(W, k, qmin, qmax).to(torch.int64))
    for c in range(W.shape[0]):
        if k[c] == 1:
            assert torch.equal(Q[c], torch.round(W[c]).clamp(qmin, qmax).to(torch.int64))
        else:  # a slightly smaller factor must violate the bound (minimality up to bisection tolerance)
            smaller = k[c:c + 1] * (1 - 1e-9)
            assert not bool(_feasible(_codes(W[c:c + 1], smaller, qmin, qmax), limit).all())


def test_wide_P_keeps_every_channel_and_narrow_P_rescales():
    W = _weights(seed=3)
    k_wide, Q_wide = rescale_factors(W, qmin=-127, qmax=127, P=30, N=8)
    assert torch.all(k_wide == 1) and torch.equal(Q_wide, torch.round(W).clamp(-127, 127).to(torch.int64))
    k_narrow, _ = rescale_factors(W, qmin=-127, qmax=127, P=13, N=8)
    assert torch.all(k_narrow >= 1) and bool((k_narrow > 1).any())


def test_feasibility_is_monotone_in_the_factor():
    W = _weights(C=4, K=200, seed=7)
    limit = axe.integer_limit(15, 8)
    grid = torch.linspace(1, 60, 400, dtype=torch.float64)
    for c in range(W.shape[0]):
        flags = [bool(_feasible(_codes(W[c:c + 1], g.reshape(1), -127, 127), limit)) for g in grid]
        first = flags.index(True) if True in flags else len(flags)
        assert all(flags[first:])
