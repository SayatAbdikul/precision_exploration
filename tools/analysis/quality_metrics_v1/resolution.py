"""Part B: paired resolution against n, interval coverage (percentile, BCa, Wald), power and projections."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.stats import binomtest, norm

from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
from . import SEED

NS = (128, 256, 512, 1000)
PERMS = 50
EXPECTED_RESAMPLES = 2000
SIM_DATASETS = 2000
SIM_RESAMPLES = 2000
POWER_DATASETS = 4000
Z975, Z95, Z90, Z80 = norm.ppf(0.975), norm.ppf(0.95), norm.ppf(0.90), norm.ppf(0.80)


# ------------------------------------------------------------------------------------------- subsample widths
@lru_cache(maxsize=None)
def binary_width(n, left_only, right_only):
    """Width (pp) of the project's paired binary-outcome interval for one block with these discordant counts."""
    a = np.zeros(n, dtype=np.int8)
    b = np.zeros(n, dtype=np.int8)
    a[:left_only] = 1
    b[left_only:left_only + right_only] = 1
    low, high = paired_outcomes(a, b)["pointwise_95_interval_pp"]
    return high - low


_W = {}


def expected_weights(n):
    if n not in _W:
        index = np.random.default_rng([SEED, 11, n]).integers(0, n, size=(EXPECTED_RESAMPLES, n))
        weights = np.zeros((EXPECTED_RESAMPLES, n))
        np.add.at(weights, (np.arange(EXPECTED_RESAMPLES)[:, None], index), 1.0)
        _W[n] = weights
    return _W[n]


def subsample_contrast(diff_expected, left_binary, right_binary, perms=PERMS):
    """Per n: median widths (pp), between-block SD of the difference and mean analytic SE (pp)."""
    out = {}
    for n in NS:
        widths_b, widths_e, diffs, se_e, se_b = [], [], [], [], []
        for r in range(perms if n < 1000 else 1):
            perm = np.random.default_rng([SEED, 10, r]).permutation(1000) if n < 1000 else np.arange(1000)
            for k in range(1000 // n):
                block = perm[k * n:(k + 1) * n]
                d = diff_expected[block]
                lb, rb = left_binary[block], right_binary[block]
                lo, ro = int(((lb == 1) & (rb == 0)).sum()), int(((lb == 0) & (rb == 1)).sum())
                widths_b.append(binary_width(n, lo, ro))
                values = expected_weights(n) @ d / n * 100
                low, high = np.percentile(values, [2.5, 97.5])
                widths_e.append(high - low)
                diffs.append(100 * d.mean())
                se_e.append(100 * d.std(ddof=1) / math.sqrt(n))
                pd, dd = (lo + ro) / n, (ro - lo) / n
                se_b.append(100 * math.sqrt(max(pd - dd * dd, 0.0) / n))
        out[n] = {"blocks": len(diffs), "width_binary_median": float(np.median(widths_b)),
                  "width_expected_median": float(np.median(widths_e)),
                  "between_block_sd_expected": float(np.std(diffs, ddof=1)) if n <= 256 else None,
                  "se_expected_mean": float(np.mean(se_e)), "se_binary_mean": float(np.mean(se_b))}
    return out


def slope(widths):
    """Least-squares slope of log width against log n (sqrt scaling gives -0.5)."""
    ns = np.array([n for n in NS if widths.get(n, 0) > 0], dtype=float)
    ws = np.array([widths[int(n)] for n in ns], dtype=float)
    if len(ns) < 3:
        return None
    return float(np.polyfit(np.log(ns), np.log(ws), 1)[0])


# ------------------------------------------------------------------------------------------- interval coverage
def bca_bounds(theta_hat, draws, n, left_only, right_only, alpha=0.05):
    """BCa for the paired difference of binary outcomes: mid-rank bias correction and closed-form jackknife acceleration."""
    below = np.count_nonzero(draws < theta_hat - 1e-15)
    equal = np.count_nonzero(np.abs(draws - theta_hat) <= 1e-15)
    share = (below + 0.5 * equal) / len(draws)
    if share <= 0 or share >= 1:
        return float(draws.min()), float(draws.max())
    z0 = norm.ppf(share)
    same = n - left_only - right_only
    values, counts = [], []
    if left_only:
        values.append((right_only - (left_only - 1)) / (n - 1)); counts.append(left_only)
    if right_only:
        values.append(((right_only - 1) - left_only) / (n - 1)); counts.append(right_only)
    if same:
        values.append((right_only - left_only) / (n - 1)); counts.append(same)
    values, counts = np.array(values), np.array(counts, dtype=float)
    mean = (values * counts).sum() / counts.sum()
    u = mean - values
    denominator = (counts * u ** 2).sum()
    a = (counts * u ** 3).sum() / (6 * denominator ** 1.5) if denominator > 0 else 0.0
    bounds = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        level = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        bounds.append(float(np.quantile(draws, level)))
    return bounds[0], bounds[1]


@lru_cache(maxsize=None)
def intervals(n, left_only, right_only):
    """Percentile, BCa and Wald 95 percent intervals (proportions) for one paired data set."""
    same = n - left_only - right_only
    rng = np.random.default_rng([SEED, 20, n, left_only, right_only])
    draws = rng.multinomial(n, np.array([left_only, same, right_only]) / n, size=SIM_RESAMPLES)
    theta = (draws[:, 2] - draws[:, 0]) / n
    theta_hat = (right_only - left_only) / n
    percentile = tuple(np.quantile(theta, [0.025, 0.975]))
    bca = bca_bounds(theta_hat, theta, n, left_only, right_only)
    pd = (left_only + right_only) / n
    half = Z975 * math.sqrt(max(pd - theta_hat ** 2, 0.0) / n)
    return percentile, bca, (theta_hat - half, theta_hat + half)


def coverage(p_left_only, p_right_only, n, datasets=SIM_DATASETS, scenario=0):
    truth = p_right_only - p_left_only
    rng = np.random.default_rng([SEED, 21, scenario, n])
    counts = rng.multinomial(n, [p_left_only, 1 - p_left_only - p_right_only, p_right_only], size=datasets)
    hits = np.zeros(3)
    widths = np.zeros(3)
    for lo, _, ro in counts:
        for j, (low, high) in enumerate(intervals(n, int(lo), int(ro))):
            hits[j] += low <= truth + 1e-12 and truth - 1e-12 <= high
            widths[j] += high - low
    share = hits / datasets
    return {"percentile": share[0], "bca": share[1], "wald": share[2],
            "mc_se": float(math.sqrt(0.95 * 0.05 / datasets)),
            "width_percentile_pp": 100 * widths[0] / datasets, "width_bca_pp": 100 * widths[1] / datasets,
            "width_wald_pp": 100 * widths[2] / datasets}


# ------------------------------------------------------------------------------------------- power
def mdd(n, pd, alpha_z=Z975, power_z=Z80):
    """Minimum detectable paired difference (proportion) for discordance pd."""
    return (alpha_z + power_z) * math.sqrt(pd / n)


@lru_cache(maxsize=None)
def mcnemar_p(discordant, right_only):
    return binomtest(right_only, discordant).pvalue if discordant else 1.0


def simulated_power(n, pd, delta, datasets=POWER_DATASETS, seed_tag=0):
    """Share of exact McNemar tests with p < 0.05 when the true difference is delta (proportion)."""
    p_right, p_left = (pd + delta) / 2, (pd - delta) / 2
    if p_left < 0:
        return None
    rng = np.random.default_rng([SEED, 30, seed_tag, n])
    counts = rng.multinomial(n, [p_left, 1 - pd, p_right], size=datasets)
    return float(np.mean([mcnemar_p(int(lo + ro), int(ro)) < 0.05 for lo, _, ro in counts]))


def images_needed(pd, margin, kind):
    """Images for power 0.8 at alpha 0.05 (normal approximation) for one hypothesis type."""
    if kind == "difference":  # detect a true difference equal to the margin, two-sided
        z = Z975 + Z80
        effect = margin
    elif kind == "equivalence":  # TOST, true difference 0, margins +-margin
        z = Z95 + Z90
        effect = margin
    elif kind == "noninferiority":  # one-sided, true difference 0
        z = Z95 + Z80
        effect = margin
    elif kind == "beyond_margin":  # interval bound beyond the margin when the true difference is twice the margin
        z = Z975 + Z80
        effect = margin
    else:
        raise ValueError(kind)
    return math.ceil(z * z * pd / effect ** 2)
