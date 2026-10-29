"""Pregame feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sim.features.build import decay_factor, lagged_decayed_sums


def _brute_force(values: np.ndarray, group: np.ndarray, d: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for t in range(len(values)):
        acc = np.zeros(values.shape[1])
        # Only rows strictly before t in the same group, decayed by distance.
        for u in range(t):
            if group[u] == group[t]:
                lag = sum(1 for v in range(u + 1, t) if group[v] == group[t])
                acc += d ** lag * values[u]
        out[t] = acc
    return out


def test_lagged_decayed_sums_matches_brute_force():
    # hand-written values: two contiguous groups, two columns
    values = np.array([[1.0, 10.0], [2.0, 0.0], [3.0, 5.0], [4.0, 1.0],
                       [100.0, 7.0], [200.0, 0.0], [300.0, 2.0]])
    group = np.array([7, 7, 7, 7, 9, 9, 9])
    d = 0.8
    got = lagged_decayed_sums(values, group, d)
    np.testing.assert_allclose(got, _brute_force(values, group, d), rtol=1e-12, atol=1e-12)
    # First row of each group is exactly zero.
    assert np.all(got[0] == 0) and np.all(got[4] == 0)
    # Row 1 = x0, row 2 = x1 + d*x0 (explicit arithmetic).
    np.testing.assert_allclose(got[1], values[0])
    np.testing.assert_allclose(got[2], values[1] + d * values[0])
    # Group 9 never sees group 7's large-magnitude history.
    np.testing.assert_allclose(got[5], values[4])


def test_lagged_decayed_sums_value_at_t_ignores_rows_at_or_after_t():
    # hand-written values
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group = np.zeros(5, dtype=int)
    base = lagged_decayed_sums(values, group, 0.5)
    for t in range(5):
        changed = values.copy()
        changed[t:] = [99.0] * (5 - t)
        got = lagged_decayed_sums(changed, group, 0.5)
        np.testing.assert_array_equal(got[: t + 1], base[: t + 1])


def test_lagged_decayed_sums_groups_independent():
    # hand-written values: scrambling group 1 leaves group 0 bit-identical
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group = np.array([0, 0, 0, 1, 1, 1])
    a = lagged_decayed_sums(values, group, 0.9)
    v2 = values.copy()
    v2[3:] = [1e6, -1e6, 3e5]
    b = lagged_decayed_sums(v2, group, 0.9)
    np.testing.assert_array_equal(a[:3], b[:3])
    assert b[3] == 0.0


def test_decay_factor_halflife():
    assert decay_factor(10) ** 10 == pytest.approx(0.5)


def test_first_game_has_no_history(features):
    first = features[features["n_prior"] == 0]
    assert len(first) > 0
    assert (first["s_one"] == 0).all()
    assert (first["l_minutes"] == 0).all()
    assert first["last_min"].isna().all()
    # Each player's chronologically first active row has n_prior == 0.
    f = features.sort_values(["player_id", "game_date", "game_id"])
    assert (f.groupby("player_id")["n_prior"].first() == 0).all()


def test_n_prior_counts_earlier_active_games(features):
    f = features.sort_values(["player_id", "game_date", "game_id"])
    expected = f.groupby("player_id").cumcount()
    assert (f["n_prior"].to_numpy() == expected.to_numpy()).all()


def test_s_one_saturates_below_geometric_limit(features, cfg):
    d = decay_factor(cfg["features"]["halflife_games"])
    limit = 1.0 / (1.0 - d)
    assert features["s_one"].max() < limit
    assert features["s_one"].min() >= 0
    # s_one is monotone in n_prior within a player (sum of d^k, k < n_prior).
    n = features["n_prior"].to_numpy()
    np.testing.assert_allclose(features["s_one"].to_numpy(), (1 - d ** n) / (1 - d), rtol=1e-10)


def test_last3_min_matches_manual_mean(features):
    pid = features.groupby("player_id").size().idxmax()   # real player with the most games
    p = features[features["player_id"] == pid].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    assert len(p) >= 30
    mins = p["minutes"].to_numpy()
    for i in range(1, len(p)):
        manual = mins[max(0, i - 3): i].mean()
        assert p.loc[i, "last3_min"] == pytest.approx(manual), i
        assert p.loc[i, "last_min"] == pytest.approx(mins[i - 1])
    assert np.isnan(p.loc[0, "last3_min"])


def test_days_rest_and_b2b(features):
    f = features.sort_values(["player_id", "game_date", "game_id"])
    gap = f.groupby("player_id")["game_date"].diff().dt.days
    np.testing.assert_array_equal(f["days_rest"].to_numpy(), gap.clip(upper=10).to_numpy())
    assert (f["b2b"].to_numpy() == (gap == 1).to_numpy().astype(float)).all()
    assert f["days_rest"].max() <= 10
    assert (f["b2b"] == 1).any()


def test_features_chronological_and_active_only(features, ds):
    assert features["game_date"].is_monotonic_increasing
    assert set(features["status"].unique()) <= {"played", "dnp"}
    n_active = (ds.player_games["status"] != "inactive").sum()
    assert len(features) == n_active
    assert not features.duplicated(["game_id", "player_id"]).any()


def test_team_features_are_lagged(features):
    # A team's first game has no team history: pace/opponent context are undefined (NaN), not leaked.
    f = features.sort_values(["team_id", "game_date"])
    first_dates = f.groupby("team_id")["game_date"].min()
    first = f.merge(first_dates.rename("d0"), left_on="team_id", right_index=True)
    first = first[first["game_date"] == first["d0"]]
    assert (first["t_games"] == 0).all()
    assert first["t_pace"].isna().all()
