"""Proper scoring rules and calibration statistics for discrete and continuous predictive distributions.

All functions are vectorized over rows. Lower is better for every score.

Discrete (counts) — the predictive is a PMF on {0..K} with tail mass lumped into K:
  crps_discrete     exact CRPS = sum_k (F(k) - 1{y <= k})^2 (+ (y - K - 1) when y > K)
  log_score         -log p(y) (used with the Rao-Blackwellized PMF of the NB primitives)
  pit_randomized    F(y-1) + V (F(y) - F(y-1)), uniform under calibration even for counts
  central_interval  smallest/largest integers with F >= a/2 and F >= 1 - a/2; reports the
                    interval's actual predictive mass (>= nominal for counts)
  brier             (P(Y >= t) - 1{y >= t})^2
Continuous (minutes) — the predictive is a quantile grid:
  crps_quantiles    2 * mean over levels of the pinball loss (quantile-score decomposition)
Multivariate:
  variogram_score   order-0.5 variogram score over teammates (sensitive to dependence)
"""

from __future__ import annotations

import numpy as np

EPS_PROB = 1e-12


def normalize_pmf(pmf: np.ndarray) -> np.ndarray:
    pmf = np.asarray(pmf, dtype=np.float64)
    tot = pmf.sum(axis=1, keepdims=True)
    if np.any(tot <= 0):
        raise ValueError("PMF rows must have positive mass")
    return pmf / tot


def cdf_from_pmf(pmf: np.ndarray) -> np.ndarray:
    F = np.cumsum(normalize_pmf(pmf), axis=1)
    F[:, -1] = 1.0
    return np.minimum(F, 1.0)


def crps_discrete(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    F = cdf_from_pmf(pmf)
    K = F.shape[1] - 1
    y = np.asarray(y, dtype=np.int64)
    k = np.arange(K + 1)[None, :]
    ind = (y[:, None] <= k).astype(np.float64)
    # Beyond the support F = 1 and the indicator is 0 for k = K+1 .. y-1 (k = K is already in the sum).
    return ((F - ind) ** 2).sum(axis=1) + np.maximum(y - K - 1, 0)


def log_score(pmf: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (-log p(y), floored) where ``floored`` flags p(y) below EPS_PROB or y beyond support."""
    p = normalize_pmf(pmf)
    K = p.shape[1] - 1
    y = np.asarray(y, dtype=np.int64)
    py = p[np.arange(len(y)), np.clip(y, 0, K)]
    floored = (py < EPS_PROB) | (y > K)
    return -np.log(np.maximum(py, EPS_PROB)), floored


def pit_randomized(pmf: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    F = cdf_from_pmf(pmf)
    K = F.shape[1] - 1
    y = np.clip(np.asarray(y, dtype=np.int64), 0, K)
    idx = np.arange(len(y))
    Fy = F[idx, y]
    Fprev = np.where(y > 0, F[idx, np.maximum(y - 1, 0)], 0.0)
    return Fprev + rng.random(len(y)) * (Fy - Fprev)


def quantile_discrete(pmf: np.ndarray, level: float) -> np.ndarray:
    F = cdf_from_pmf(pmf)
    return np.argmax(F >= level - 1e-12, axis=1)


def central_interval(pmf: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central (1-alpha) interval [lo, hi] and its predictive mass P(lo <= Y <= hi)."""
    F = cdf_from_pmf(pmf)
    lo = np.argmax(F >= alpha / 2 - 1e-12, axis=1)
    hi = np.argmax(F >= 1 - alpha / 2 - 1e-12, axis=1)
    idx = np.arange(F.shape[0])
    mass = F[idx, hi] - np.where(lo > 0, F[idx, np.maximum(lo - 1, 0)], 0.0)
    return lo, hi, mass


def prob_at_least(pmf: np.ndarray, t: int) -> np.ndarray:
    F = cdf_from_pmf(pmf)
    if t <= 0:
        return np.ones(F.shape[0])
    if t - 1 >= F.shape[1]:
        return np.zeros(F.shape[0])
    return 1.0 - F[:, t - 1]


def brier(p: np.ndarray, event: np.ndarray) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(event, float)) ** 2


def pmf_mean(pmf: np.ndarray) -> np.ndarray:
    p = normalize_pmf(pmf)
    return p @ np.arange(p.shape[1])


# ----------------------------------------------------------------------------- continuous

def crps_quantiles(q: np.ndarray, levels: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS approximated by 2 * mean pinball loss over an evenly spaced quantile grid."""
    y = np.asarray(y, float)[:, None]
    tau = np.asarray(levels)[None, :]
    diff = y - q
    pin = np.maximum(tau * diff, (tau - 1) * diff)
    return 2.0 * pin.mean(axis=1)


def crps_samples(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS: E|X - y| - 0.5 E|X - X'| via the sorted-sample identity. x: (n, S)."""
    x = np.sort(np.asarray(x, dtype=np.float64), axis=1)
    n, S = x.shape
    y = np.asarray(y, float)
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    i = np.arange(1, S + 1)
    term2 = (2 * (x * (2 * i - S - 1)[None, :]).sum(axis=1)) / (S * S)
    return term1 - 0.5 * term2


def pit_from_quantiles(q: np.ndarray, levels: np.ndarray, y: np.ndarray, p0: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """PIT for minutes: interpolated CDF; randomized over the point mass at 0 minutes."""
    y = np.asarray(y, float)
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = np.interp(y[i], q[i], levels, left=0.0, right=1.0)
    zero = y <= 0
    out[zero] = rng.random(zero.sum()) * p0[zero]
    return out


def pit_samples(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized PIT from samples (handles ties for integer-valued totals)."""
    y = np.asarray(y, float)[:, None]
    below = (x < y).mean(axis=1)
    equal = (x == y).mean(axis=1)
    return below + rng.random(len(below)) * equal


# ----------------------------------------------------------------------------- multivariate

def variogram_score(joint: np.ndarray, y: np.ndarray, groups: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Variogram score of order p for each group (team-game) of rows.

    joint: (n, J) joint draws, y: (n,) outcomes, groups: (n,) group labels (contiguous).
    VS = sum_{i<j} (|y_i - y_j|^p - E|X_i - X_j|^p)^2, unit weights.
    """
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(groups)]
    out = np.empty(len(starts))
    joint = np.asarray(joint, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    for g, (a, b) in enumerate(zip(starts, ends)):
        X, Y = joint[a:b], y[a:b]
        iu = np.triu_indices(b - a, k=1)
        ey = np.abs(Y[:, None] - Y[None, :])[iu] ** p
        ex = (np.abs(X[:, None, :] - X[None, :, :]) ** p).mean(axis=2)[iu]
        out[g] = float(np.sum((ey - ex) ** 2))
    return out
