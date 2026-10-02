"""AXE on OPTQ (Colbert et al., arXiv:2409.17092v1, Algorithm 2) and the data-free naive baseline.

All weights are in units of the fixed per-output-channel scale (integer codes of the weight codebook).
Equation numbers refer to the v1 paper; protocol: public/experiments/configs/breadth-study/b2-axe-protocol-v1.json.

Guarantee (protocol "overflow_bound_enforced"): with B = (2^(P-1)-1)/(2^N-1) - 0.5 (Eq. 21, round-to-nearest), the
greedy clip keeps the integer sums beta = sum(q > 0) and -alpha = -sum(q < 0) of every channel at most B + 0.5, so for
any input codes in [mu, nu] with nu - mu = 2^N - 1 and mu <= 0 <= nu every partial sum lies in
[-(2^(P-1)-1), 2^(P-1)-1].
"""
from __future__ import annotations

import torch


def budget(P, N):
    """Eq. 21 with max(Delta) = 0.5: -A = B = (2^(P-1)-1)/(2^N-1) - 0.5 (code units)."""
    return (2 ** (P - 1) - 1) / (2 ** N - 1) - 0.5


def l1_target(P, N):
    """Eq. 4 radius in code units: Z = (2^P - 2)/(2^N - 1)."""
    return (2 ** P - 2) / (2 ** N - 1)


def integer_limit(P, N):
    """Largest integer value of beta (and of -alpha) that the bound admits: floor((2^(P-1)-1)/(2^N-1))."""
    return (2 ** (P - 1) - 1) // (2 ** N - 1)


def l1_threshold(w, z):
    """Eq. 16, per row of ``w`` [C, K]: lambda of the Euclidean projection onto the l1 ball of radius ``z``.

    lambda = (sum_{i<=rho} mu_i - z)/rho with mu the magnitudes sorted descending and rho the number of non-zero
    elements of the projection (Duchi et al. 2008); 0 for rows already inside the ball.
    """
    device = torch.as_tensor(w).device
    w = torch.as_tensor(w, dtype=torch.float64).cpu()  # CPU: deterministic cumsum (CUDA cumsum is not)
    mags = w.abs()
    mu, _ = torch.sort(mags, dim=1, descending=True)
    cs = mu.cumsum(dim=1)
    j = torch.arange(1, w.shape[1] + 1, dtype=torch.float64, device=w.device)
    z = torch.as_tensor(z, dtype=torch.float64).cpu().reshape(-1, 1).expand(w.shape[0], 1)
    ok = (mu - (cs - z) / j) > 0
    rho = (ok * j).amax(dim=1).long().clamp_min(1)
    lam = (cs.gather(1, (rho - 1)[:, None]).squeeze(1) - z[:, 0]) / rho.double()
    inside = mags.sum(dim=1) <= z[:, 0]
    return torch.where(inside, torch.zeros_like(lam), lam.clamp_min(0)).to(device)


def soft(x, lam):
    """Pi_lambda(x) = sign(x) (|x| - lambda)_+ (Eq. 14)."""
    return torch.sign(x) * torch.relu(x.abs() - lam)


def _clip(v, a, b):
    """Psi_{a,b}(v) = clip(v; a, b); when the two budgets are exhausted (a > b, both within half a step of 0) the
    result is b, which rounds to 0."""
    return torch.minimum(torch.maximum(v, a), b)


def _round(v, qmin, qmax):
    """Round to nearest, ties to even (the B2 integer codebooks: level parity = code parity), clipped to the book."""
    return torch.round(v).clamp(qmin, qmax)


def optq(W, H, *, qmin, qmax, P=None, N=None, order="act", block=128, percdamp=0.01):
    """AXE-OPTQ (``P`` given) or plain OPTQ (``P=None``) of one layer.

    W: [C, K] weights in code units; H: [K, K] Hessian proxy (any positive multiple of X X^T).
    Returns ``(Q int64 [C, K], info)``.
    """
    W = W.detach().to(torch.float64).clone()
    H = H.detach().to(torch.float64).clone()
    C, K = W.shape
    device = W.device
    constrained = P is not None
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0  # OPTQ reference code: inputs never active in calibration get weight 0
    lam = l1_threshold(W, l1_target(P, N)) if constrained else torch.zeros(C, dtype=torch.float64, device=device)
    perm = torch.argsort(torch.diag(H), descending=True, stable=True) if order == "act" else torch.arange(K, device=device)
    W, H = W[:, perm], H[perm][:, perm]
    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(K, dtype=torch.float64, device=device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    B = budget(P, N) if constrained else 0.0
    b = torch.full((C,), B, dtype=torch.float64, device=device)
    a = -b.clone()
    clipped = 0
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]
            v = soft(w, lam)
            if constrained:
                c = _clip(v, a, b)
                clipped += int((c != v).sum())
                v = c
            q = _round(v, qmin, qmax)
            Q[:, i1 + j] = q
            err = (w - q) / Hinv1[j, j]
            W1[:, j:] -= err[:, None] * Hinv1[j, j:][None, :]
            E1[:, j] = err
            if constrained:
                b -= torch.clamp(q, min=0)
                a -= torch.clamp(q, max=0)
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    inverse = torch.argsort(perm)
    Q = Q[:, inverse]
    info = {"dead_inputs": int(dead.sum()), "damp": float(damp), "clipped_steps": clipped,
            "lambda_max": float(lam.max()), "lambda_channels_positive": int((lam > 0).sum())}
    return Q.round().to(torch.int64), info


def naive(W, *, qmin, qmax, P, N):
    """Data-free baseline: the same Pi_lambda, greedy clip and budgets in natural index order; no error feedback."""
    W = W.detach().to(torch.float64)
    C, K = W.shape
    lam = l1_threshold(W, l1_target(P, N))
    V = soft(W, lam[:, None])
    b = torch.full((C,), budget(P, N), dtype=torch.float64, device=W.device)
    a = -b.clone()
    Q = torch.zeros_like(W)
    clipped = 0
    for i in range(K):
        c = _clip(V[:, i], a, b)
        clipped += int((c != V[:, i]).sum())
        q = _round(c, qmin, qmax)
        Q[:, i] = q
        b -= torch.clamp(q, min=0)
        a -= torch.clamp(q, max=0)
    info = {"clipped_steps": clipped, "lambda_max": float(lam.max()), "lambda_channels_positive": int((lam > 0).sum())}
    return Q.round().to(torch.int64), info


def sign_sums(Q):
    """Per channel (beta, -alpha): sums of the positive integers and of the magnitudes of the negative ones."""
    Q = Q.reshape(Q.shape[0], -1).to(torch.int64)
    return torch.clamp(Q, min=0).sum(1), (-torch.clamp(Q, max=0)).sum(1)


def bound_holds(Q, P, N):
    beta, neg = sign_sums(Q)
    limit = integer_limit(P, N)
    return bool((beta <= limit).all() and (neg <= limit).all())
