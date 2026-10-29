"""Proper scoring rules and calibration statistics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from nba_sim.evaluation import metrics as M
from nba_sim.simulation.predictive import MINUTE_LEVELS


def _poisson_pmf(lam: float, K: int) -> np.ndarray:
    p = stats.poisson.pmf(np.arange(K + 1), lam)
    p[K] += stats.poisson.sf(K, lam)
    return p


def _brute_crps(pmf_row: np.ndarray, y: int, extra: int = 200) -> float:
    """CRPS = sum over all integers k >= 0 of (F(k) - 1{y <= k})^2, F = 1 beyond the support."""
    F = np.cumsum(pmf_row / pmf_row.sum())
    total = 0.0
    for k in range(len(F) + extra):
        Fk = F[k] if k < len(F) else 1.0
        total += (Fk - float(y <= k)) ** 2
    return total


def test_crps_discrete_equals_brute_force():
    # hand-written values
    pmf = np.array([[0.2, 0.5, 0.3], [0.0, 0.0, 1.0], [0.6, 0.1, 0.3], [0.25, 0.25, 0.5]])
    y = np.array([1, 0, 2, 7])
    got = M.crps_discrete(pmf, y)
    expected = [_brute_crps(pmf[i], y[i]) for i in range(len(y))]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_crps_point_masses():
    K = 10
    for x in (0, 3, 10):
        pmf = np.zeros((1, K + 1))
        pmf[0, x] = 1.0
        assert M.crps_discrete(pmf, np.array([x]))[0] == pytest.approx(0.0)
        for y in (0, 2, 5, 9, 11, 14):   # 11, 14 lie beyond the support K = 10
            assert M.crps_discrete(pmf, np.array([y]))[0] == pytest.approx(abs(y - x))
        assert M.crps_samples(np.full((1, 7), float(x)), np.array([4.0]))[0] == pytest.approx(abs(4 - x))


def test_crps_discrete_matches_crps_samples():
    rng = np.random.default_rng(0)
    K = 40
    pmf = np.vstack([_poisson_pmf(lam, K) for lam in (2.0, 7.5, 15.0)])
    y = np.array([1, 9, 22])
    draws = np.vstack([rng.choice(K + 1, size=40_000, p=pmf[i]) for i in range(3)]).astype(float)
    np.testing.assert_allclose(M.crps_samples(draws, y), M.crps_discrete(pmf, y), rtol=0.02, atol=0.01)


def test_crps_samples_brute_force():
    # hand-written values
    x = np.array([[1.0, 4.0, 2.0, 8.0]])
    y = np.array([3.0])
    e1 = np.abs(x - y[:, None]).mean()
    e2 = np.abs(x[0][:, None] - x[0][None, :]).mean()
    assert M.crps_samples(x, y)[0] == pytest.approx(e1 - 0.5 * e2)


def test_log_score():
    # hand-written values
    pmf = np.array([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3], [1.0, 0.0, 0.0], [0.2, 0.5, 0.3]])
    y = np.array([1, 2, 2, 5])
    ls, floored = M.log_score(pmf, y)
    assert ls[0] == pytest.approx(-np.log(0.5))
    assert ls[1] == pytest.approx(-np.log(0.3))
    assert floored.tolist() == [False, False, True, True]
    assert ls[2] == pytest.approx(-np.log(M.EPS_PROB))
    # Unnormalized input is normalized first.
    ls2, _ = M.log_score(pmf[:1] * 4, y[:1])
    assert ls2[0] == pytest.approx(ls[0])


def test_normalize_pmf_rejects_zero_rows():
    with pytest.raises(ValueError):
        M.normalize_pmf(np.zeros((1, 3)))


def test_randomized_pit_uniform_under_calibration():
    rng = np.random.default_rng(42)
    K, lam, n = 30, 4.0, 5000
    pmf = np.tile(_poisson_pmf(lam, K), (n, 1))
    y = rng.poisson(lam, n)      # outcomes drawn from the predictive itself
    pit = M.pit_randomized(pmf, y, rng)
    assert np.all((pit >= 0) & (pit <= 1))
    assert stats.kstest(pit, "uniform").pvalue > 0.01
    # A miscalibrated (too low) predictive is detected.
    pit_bad = M.pit_randomized(np.tile(_poisson_pmf(2.0, K), (n, 1)), y, rng)
    assert stats.kstest(pit_bad, "uniform").pvalue < 1e-6


def test_central_interval_mass_at_least_nominal():
    K = 60
    pmf = np.vstack([_poisson_pmf(lam, K) for lam in (0.3, 1.0, 4.0, 11.0, 25.0)])
    for alpha in (0.5, 0.2, 0.05):
        lo, hi, mass = M.central_interval(pmf, alpha)
        assert np.all(lo <= hi)
        assert np.all(mass >= 1 - alpha - 1e-9)
        F = np.cumsum(pmf, axis=1)
        np.testing.assert_allclose(mass, F[np.arange(5), hi] - np.where(lo > 0, F[np.arange(5), lo - 1], 0))


def test_quantile_discrete_median():
    # hand-written values
    pmf = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    assert M.quantile_discrete(pmf, 0.5).tolist() == [1, 0]


def test_prob_at_least():
    # hand-written values
    pmf = np.array([[0.2, 0.5, 0.3]])
    assert M.prob_at_least(pmf, 0)[0] == 1.0
    assert M.prob_at_least(pmf, 1)[0] == pytest.approx(0.8)
    assert M.prob_at_least(pmf, 2)[0] == pytest.approx(0.3)
    assert M.prob_at_least(pmf, 5)[0] == 0.0


def test_brier():
    # hand-written values
    np.testing.assert_allclose(M.brier(np.array([0.8, 0.3, 1.0]), np.array([True, False, False])),
                               [0.04, 0.09, 1.0])


def test_pmf_mean():
    # hand-written values
    assert M.pmf_mean(np.array([[0.2, 0.5, 0.3]]))[0] == pytest.approx(1.1)


def test_crps_quantiles_close_to_exact_normal():
    mu, s = 20.0, 6.0
    y = np.array([5.0, 15.0, 20.0, 27.0, 40.0])
    q = np.tile(stats.norm.ppf(MINUTE_LEVELS, mu, s), (len(y), 1))
    z = (y - mu) / s
    exact = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    np.testing.assert_allclose(M.crps_quantiles(q, MINUTE_LEVELS, y), exact, rtol=0.01)


def test_pit_from_quantiles_point_mass_at_zero():
    rng = np.random.default_rng(3)
    q = np.tile(np.linspace(0.0, 40.0, len(MINUTE_LEVELS)), (3, 1))
    y = np.array([0.0, 20.0, 50.0])
    pit = M.pit_from_quantiles(q, MINUTE_LEVELS, y, np.array([0.3, 0.3, 0.3]), rng)
    assert 0 <= pit[0] <= 0.3
    assert pit[1] == pytest.approx(0.5, abs=0.01)
    assert pit[2] == 1.0


def test_pit_samples_bounds():
    rng = np.random.default_rng(5)
    x = np.tile(np.arange(10.0), (3, 1))
    pit = M.pit_samples(x, np.array([-1.0, 4.0, 20.0]), rng)
    assert pit[0] == 0.0 and pit[2] == 1.0
    assert 0.4 <= pit[1] <= 0.5


def test_variogram_score_rewards_correct_dependence():
    """Two players with strongly correlated outcomes: the correctly-correlated predictive must
    beat a predictive with identical marginals but independent players."""
    rng = np.random.default_rng(11)
    cov = 25.0 * np.array([[1.0, 0.9], [0.9, 1.0]])
    mean = np.array([20.0, 15.0])
    J, G = 2000, 400
    right = rng.multivariate_normal(mean, cov, size=J).T                    # (2, J)
    wrong = np.vstack([rng.permutation(right[0]), rng.permutation(right[1])])   # same marginals
    outcomes = rng.multivariate_normal(mean, cov, size=G)                   # (G, 2)
    y = outcomes.ravel()
    groups = np.repeat(np.arange(G), 2)
    vs_right = M.variogram_score(np.tile(right, (G, 1)), y, groups)
    vs_wrong = M.variogram_score(np.tile(wrong, (G, 1)), y, groups)
    assert np.all(vs_right >= 0) and np.all(vs_wrong >= 0)
    assert vs_right.shape == (G,)
    assert vs_right.mean() < vs_wrong.mean()


def test_variogram_score_zero_for_matching_point_forecast():
    # hand-written values: degenerate predictive equal to the outcome -> score 0
    y = np.array([3.0, 7.0, 1.0, 4.0, 4.0])
    joint = np.tile(y[:, None], (1, 5))
    vs = M.variogram_score(joint, y, np.array([0, 0, 0, 1, 1]))
    np.testing.assert_allclose(vs, 0.0)
