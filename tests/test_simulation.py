"""Simulation engine, playing-time constraints and frozen predictive summaries."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from nba_sim.models.common import PRIMS
from nba_sim.simulation.constraints import (
    allocate_minutes,
    ensure_min_players,
    group_starts,
    team_minute_residual,
)
from nba_sim.simulation.predictive import (
    SUPPORT,
    empirical_counts,
    rao_blackwell_binomial,
    rao_blackwell_pmf,
    summarize,
)

from conftest import MODEL_NAMES, N_SIMS

ALL_COUNTS = PRIMS + ["fg3a", "fg2m", "fg3m", "ftm", "oreb"]


# ----------------------------------------------------------------------------- reproducibility

@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_same_seed_same_draws_different_seed_different(fitted_models, target_rows, name):
    m = fitted_models[name]
    a = m.simulate(target_rows, 50, np.random.default_rng(123))
    b = m.simulate(target_rows, 50, np.random.default_rng(123))
    c = m.simulate(target_rows, 50, np.random.default_rng(124))
    np.testing.assert_array_equal(a.minutes, b.minutes)
    for s in ALL_COUNTS:
        np.testing.assert_array_equal(a.counts[s], b.counts[s])
    assert not np.array_equal(a.stat("pts"), c.stat("pts"))
    assert not np.array_equal(a.minutes, c.minutes)


# ----------------------------------------------------------------------------- box-score coherence

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_counts_nonnegative_integers(sim_results, target_rows, name):
    res = sim_results[name]
    assert res.n_sims == N_SIMS
    assert res.minutes.shape == (len(target_rows), N_SIMS)
    for s in ALL_COUNTS:
        x = res.counts[s]
        assert x.shape == res.minutes.shape
        assert np.issubdtype(x.dtype, np.integer), s
        assert x.min() >= 0, s
    assert np.all(res.minutes >= 0)
    assert np.isfinite(res.minutes).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_makes_do_not_exceed_attempts(sim_results, name):
    c = sim_results[name].counts
    assert np.all(c["fg3a"] <= c["fga"])
    assert np.all(c["fg2m"] <= c["fga"] - c["fg3a"])
    assert np.all(c["fg3m"] <= c["fg3a"])
    assert np.all(c["ftm"] <= c["fta"])
    assert np.all(c["oreb"] <= c["reb"])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_points_identity(sim_results, name):
    res = sim_results[name]
    c = res.counts
    np.testing.assert_array_equal(res.stat("pts"), 2 * c["fg2m"] + 3 * c["fg3m"] + c["ftm"])
    np.testing.assert_array_equal(res.stat("fgm"), c["fg2m"] + c["fg3m"])
    np.testing.assert_array_equal(res.stat("dreb"), c["reb"] - c["oreb"])
    np.testing.assert_array_equal(res.stat("pra"), res.stat("pts") + c["reb"] + c["ast"])
    with pytest.raises(KeyError):
        res.stat("not_a_stat")


@pytest.mark.parametrize("name", ["B", "C", "C0"])
def test_zero_minutes_zero_stats(sim_results, name):
    res = sim_results[name]
    off = res.minutes <= 0
    assert off.any()
    for s in PRIMS:
        assert np.all(res.counts[s][off] == 0), s


@pytest.mark.parametrize("name", ["B", "C", "C0"])
def test_team_minutes_constraint(sim_results, cfg, name):
    res = sim_results[name]
    lay = res.layout
    mc = cfg["models"]
    cap = mc["max_regulation_minutes"] + mc["overtime_period_minutes"] * res.n_ot     # (n_games, S)
    total = (5.0 * cap)[lay.team_game]                                                   # (n_teams, S)
    resid = team_minute_residual(res.minutes.astype(np.float64), lay.team_idx, total)
    # float32 storage of ~13 players x ~20 minutes: allow a tiny rounding residual.
    assert np.abs(resid).max() < 1e-3
    assert np.all(res.minutes <= cap[lay.team_game][lay.team_idx] + 1e-4)
    assert res.n_ot.shape == (lay.n_games, N_SIMS)
    assert res.n_ot.min() >= 0


@pytest.mark.parametrize("name", ["B", "C", "C0"])
def test_at_least_five_players_per_team_sim(sim_results, name):
    res = sim_results[name]
    lay = res.layout
    n_playing = np.add.reduceat((res.minutes > 0).astype(int), group_starts(lay.team_idx), axis=0)
    assert n_playing.min() >= 5


def test_model_a_has_no_overtime_and_bounded_minutes(sim_results, cfg):
    res = sim_results["A"]
    assert np.all(res.n_ot == 0)
    assert res.minutes.max() <= cfg["models"]["max_regulation_minutes"] + 1e-4


def test_layout_pairs_opponents(sim_results):
    lay = sim_results["C"].layout
    assert lay.n_teams == 2 * lay.n_games
    np.testing.assert_array_equal(lay.opp_team[lay.opp_team], np.arange(lay.n_teams))
    np.testing.assert_array_equal(lay.team_game[lay.opp_team], lay.team_game)
    assert np.all(lay.opp_team != np.arange(lay.n_teams))


def test_simulated_team_points_plausible(sim_results):
    """Sanity on real-data fits: mean team points close to the 2022-23 scoring level."""
    for name in ["B", "C"]:
        tp = sim_results[name].team_totals("pts")
        assert 95 < tp.mean() < 130, name


def test_model_c_rao_blackwell_binomial_consistent(sim_results):
    res = sim_results["C"]
    assert set(res.rb_binom) == set(PRIMS)
    for s, (T, share) in res.rb_binom.items():
        assert T.shape == share.shape == res.minutes.shape
        assert np.all(res.counts[s] <= T), s
        assert np.all((share >= 0) & (share <= 1 + 1e-6))


# ----------------------------------------------------------------------------- constraints

def test_allocate_minutes_respects_caps_and_totals():
    # hand-written values: one sim column, team 0 has 7 listed players, team 1 has 5
    m = np.array([[40.0], [30.0], [30.0], [20.0], [20.0], [10.0], [0.0],
                  [10.0], [10.0], [10.0], [10.0], [10.0]])
    team_idx = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cap = np.array([[48.0], [53.0]])
    total = 5 * cap
    out = allocate_minutes(m, team_idx, total, cap)
    np.testing.assert_allclose(team_minute_residual(out, team_idx, total), 0.0, atol=1e-6)
    assert np.all(out <= cap[team_idx] + 1e-9)
    assert out[6, 0] == 0.0, "a player who did not play stays at zero"
    # Five players for 5*53 minutes: everyone at the cap.
    np.testing.assert_allclose(out[7:, 0], 53.0)
    # Team 0 needs 240 from 150 raw minutes: x1.6 puts the top three at the 48 cap; the remaining
    # 96 minutes go to the free players in proportion 20:20:10 -> 38.4, 38.4, 19.2.
    np.testing.assert_allclose(out[:6, 0], [48.0, 48.0, 48.0, 38.4, 38.4, 19.2], atol=1e-6)


def test_allocate_minutes_scales_down():
    # hand-written values
    m = np.array([[48.0, 30.0], [48.0, 30.0], [48.0, 30.0], [48.0, 30.0], [48.0, 30.0], [48.0, 30.0]])
    team_idx = np.zeros(6, dtype=int)
    cap = np.array([[48.0, 48.0]])
    out = allocate_minutes(m, team_idx, 5 * cap, cap)
    np.testing.assert_allclose(out.sum(axis=0), [240.0, 240.0])
    np.testing.assert_allclose(out[:, 0], 40.0)


def test_group_starts_requires_sorted():
    # hand-written values
    np.testing.assert_array_equal(group_starts(np.array([0, 0, 1, 1, 1, 2])), [0, 2, 5])
    with pytest.raises(ValueError):
        group_starts(np.array([0, 1, 0]))


def test_ensure_min_players():
    # hand-written values: team 0 (6 players) has only 3 playing in sim 0 and 5 in sim 1
    team_idx = np.zeros(6, dtype=int)
    rank = np.array([0, 1, 2, 3, 4, 5])
    play = np.array([[True, True], [False, True], [False, True], [True, True], [False, True], [True, False]])
    out = ensure_min_players(play, team_idx, rank, 5)
    assert out[:, 0].tolist() == [True, True, True, True, True, True]   # top-5 forced, #5 already played
    np.testing.assert_array_equal(out[:, 1], play[:, 1])                 # untouched when already 5


# ----------------------------------------------------------------------------- predictive summaries

def test_empirical_counts_rows_sum_to_n_sims(sim_results):
    x = sim_results["C"].stat("pts")
    K = SUPPORT["pts"]
    cnt = empirical_counts(x, K)
    assert cnt.shape == (x.shape[0], K + 1)
    assert cnt.dtype == np.uint16
    assert np.all(cnt.sum(axis=1) == x.shape[1])
    # Tail lumping, hand-written values.
    small = empirical_counts(np.array([[0, 1, 5, 9]]), 3)
    assert small.tolist() == [[1, 1, 0, 2]]


@pytest.mark.parametrize("mu,size", [(3.5, 2.0), (12.0, 7.5), (0.4, 50.0)])
def test_rao_blackwell_pmf_matches_scipy_nbinom(mu, size):
    K = 30
    mu_arr = np.full((3, 17), mu)          # constant mu across draws and rows
    pmf = rao_blackwell_pmf(mu_arr, size, K)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)
    p = size / (size + mu)
    expected = stats.nbinom.pmf(np.arange(K), size, p)
    np.testing.assert_allclose(pmf[:, :K], np.broadcast_to(expected, (3, K)), atol=1e-6)
    np.testing.assert_allclose(pmf[:, K], stats.nbinom.sf(K - 1, size, p), atol=1e-6)


def test_rao_blackwell_pmf_zero_mean_is_point_mass():
    pmf = rao_blackwell_pmf(np.zeros((2, 5)), 3.0, 10)
    np.testing.assert_allclose(pmf[:, 0], 1.0)
    np.testing.assert_allclose(pmf[:, 1:], 0.0)


@pytest.mark.parametrize("T,p", [(10, 0.3), (25, 0.08), (4, 0.9)])
def test_rao_blackwell_binomial_matches_scipy(T, p):
    K = 15
    pmf = rao_blackwell_binomial(np.full((2, 9), T), np.full((2, 9), p), K)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)
    expected = stats.binom.pmf(np.arange(K), T, p)
    np.testing.assert_allclose(pmf[:, :K], np.broadcast_to(expected, (2, K)), atol=1e-6)
    np.testing.assert_allclose(pmf[:, K], stats.binom.sf(K - 1, T, p), atol=1e-6)


@pytest.mark.parametrize("name", ["A", "C"])
def test_summarize_shapes_and_normalization(sim_results, name):
    res = sim_results[name]
    s = summarize(res, joint_samples=50)
    n = res.minutes.shape[0]
    for stat, K in SUPPORT.items():
        assert s[f"pmf_{stat}"].shape == (n, K + 1)
        assert np.all(s[f"pmf_{stat}"].sum(axis=1) == N_SIMS)
    for p in PRIMS:
        np.testing.assert_allclose(s[f"rb_{p}"].sum(axis=1), 1.0, atol=1e-4)
    assert s["min_q"].shape == (n, 199)
    assert np.all(np.diff(s["min_q"], axis=1) >= -1e-4)
    assert np.all((s["min_p0"] >= 0) & (s["min_p0"] <= 1))
    assert s["joint_pts"].shape == (n, 50)
    assert int(s["n_sims"][0]) == N_SIMS
