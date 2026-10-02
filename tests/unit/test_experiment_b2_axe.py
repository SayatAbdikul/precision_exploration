"""Lane Q8: AXE-OPTQ, the naive baseline and the overflow bound (CPU, small synthetic layers)."""
import itertools

import pytest
import torch

from tools.experiment_b2_axe import axe
from tools.experiment_b2_axe.check import signed_bits


def _layer(C=6, K=40, seed=0, spread=60.0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(C, K, generator=g, dtype=torch.float64) * spread
    X = torch.rand(K, 500, generator=g, dtype=torch.float64) * 255 * (torch.rand(K, 1, generator=g) > 0.1)
    H = 2 * X @ X.t() / X.shape[1]
    return W, H


def test_l1_threshold_is_the_euclidean_projection():
    W, _ = _layer()
    for z in (5.0, 50.0, 400.0, 1e9):
        lam = axe.l1_threshold(W, z)
        V = axe.soft(W, lam[:, None])
        for c in range(W.shape[0]):
            norm = float(W[c].abs().sum())
            if norm <= z:
                assert lam[c] == 0
            else:
                assert abs(float(V[c].abs().sum()) - z) < 1e-6 * z
                lo, hi = 0.0, float(W[c].abs().max())  # bisection on lambda
                for _ in range(200):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if float(axe.soft(W[c], mid).abs().sum()) > z else (lo, mid)
                assert abs(lo - float(lam[c])) < 1e-6 * max(1.0, lo)


@pytest.mark.parametrize("P,N", [(10, 8), (12, 8), (14, 8), (16, 8), (9, 6), (12, 6), (11, 4)])
def test_axe_and_naive_respect_the_bound(P, N):
    W, H = _layer(seed=P + N)
    qmax = 2 ** 7 - 1
    for Q, _ in (axe.optq(W, H, qmin=-qmax - 1, qmax=qmax, P=P, N=N), axe.naive(W, qmin=-qmax - 1, qmax=qmax, P=P, N=N)):
        assert axe.bound_holds(Q, P, N)
        beta, neg = axe.sign_sums(Q)
        # every partial sum for unsigned and for signed N-bit inputs stays in the signed P-bit range
        top = 2 ** (P - 1) - 1
        assert int((beta * (2 ** N - 1)).max()) <= top and int((neg * (2 ** N - 1)).max()) <= top
        signed_hi = beta * (2 ** (N - 1) - 1) + neg * 2 ** (N - 1)
        signed_lo = -(beta * 2 ** (N - 1) + neg * (2 ** (N - 1) - 1))
        assert int(signed_hi.max()) <= top and int(signed_lo.min()) >= -top - 1


def test_bound_is_tight_enough_to_bind_and_loose_at_large_P():
    W, H = _layer(spread=100.0)
    q_free, _ = axe.optq(W, H, qmin=-128, qmax=127)
    q_wide, info = axe.optq(W, H, qmin=-128, qmax=127, P=40, N=8)
    assert torch.equal(q_free, q_wide) and info["clipped_steps"] == 0
    q_tight, info = axe.optq(W, H, qmin=-128, qmax=127, P=14, N=8)
    assert info["clipped_steps"] > 0 and not torch.equal(q_free, q_tight)
    rtn = torch.round(W).clamp(-128, 127).to(torch.int64)
    q_naive, _ = axe.naive(W, qmin=-128, qmax=127, P=40, N=8)
    assert torch.equal(q_naive, rtn)


def _reference_optq(W, H, qmin, qmax, P=None, N=None, percdamp=0.01):
    """Algorithm 2 of the paper column by column (no lazy blocks), act order, same damping."""
    W, H = W.clone(), H.clone()
    C, K = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = torch.argsort(torch.diag(H), descending=True, stable=True)
    W, H = W[:, perm], H[perm][:, perm]
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    lam = axe.l1_threshold(W[:, torch.argsort(perm)], axe.l1_target(P, N)) if P else torch.zeros(C, dtype=W.dtype)
    b = torch.full((C,), axe.budget(P, N) if P else float("inf"), dtype=W.dtype)
    a = -b.clone()
    Q = torch.zeros_like(W)
    for i in range(K):
        V = torch.minimum(torch.maximum(axe.soft(W[:, i], lam), a), b)
        Q[:, i] = torch.round(V).clamp(qmin, qmax)
        E = (W[:, i] - Q[:, i]) / Hinv[i, i]
        W[:, i:] -= E[:, None] * Hinv[i, i:][None, :]
        b -= Q[:, i].clamp(min=0)
        a -= Q[:, i].clamp(max=0)
    return Q[:, torch.argsort(perm)].to(torch.int64)


@pytest.mark.parametrize("P", [None, 15, 13])
def test_blocked_optq_equals_the_column_by_column_algorithm(P):
    W, H = _layer(C=5, K=300, seed=3)
    Q, _ = axe.optq(W, H, qmin=-128, qmax=127, P=P, N=8 if P else None, block=64)
    R = _reference_optq(W, H, -128, 127, P=P, N=8 if P else None)
    assert (Q != R).double().mean() < 0.01  # float reassociation of the lazy updates may flip rare near-ties


def test_optq_beats_rtn_on_its_objective():
    W, H = _layer(C=8, K=64, seed=5, spread=3.0)
    Q, _ = axe.optq(W, H, qmin=-8, qmax=7)
    R = torch.round(W).clamp(-8, 7)
    err = lambda q: float(torch.einsum("ck,kl,cl->", W - q, H, W - q))
    assert err(Q.double()) < err(R)


def test_signed_bits_matches_the_engine_convention():
    from tools.scaled_bridge_v2.certificates import signed_bits as engine
    for lo, hi in itertools.product([0, -1, -2, -127, -128, -129, -2**25], [0, 1, 127, 128, 2**25 - 1, 2**25]):
        assert signed_bits(lo, hi) == engine(lo, hi)
