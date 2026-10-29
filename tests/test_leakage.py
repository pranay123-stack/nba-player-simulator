"""Look-ahead leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sim.data.schema import COUNT_STATS, OUTCOME_COLUMNS
from nba_sim.features.build import build_features
from nba_sim.features.leakage_checks import (
    LeakageError,
    assert_disjoint_games,
    assert_training_before,
    perturbation_check,
    pregame_view,
)
from nba_sim.simulation.pregame import historical_spec, prepare_target

from conftest import ORIGIN

PERTURB_DATE = "2022-12-15"


def test_perturbation_check_finds_no_leak(ds, cfg):
    assert (ds.player_games["game_date"] == pd.Timestamp(PERTURB_DATE)).any()
    diff = perturbation_check(ds.player_games, ds.team_games, cfg, PERTURB_DATE)
    assert diff.empty, f"features changed when future outcomes were scrambled:\n{diff}"


def test_perturbation_check_detects_a_leaky_feature(ds, cfg, monkeypatch):
    """Sanity check of the checker itself: a feature that reads same-day minutes must be caught."""
    import nba_sim.features.leakage_checks as lc

    real = lc.build_features

    def leaky(pg, tg, c):
        f = real(pg, tg, c)
        f["leak"] = f["minutes"] * 2.0
        return f

    monkeypatch.setattr(lc, "build_features", leaky)
    diff = lc.perturbation_check(ds.player_games, ds.team_games, cfg, PERTURB_DATE)
    assert "leak" in diff.index


def test_pregame_view_strips_outcomes(features):
    view = pregame_view(features.head(50))
    assert not set(OUTCOME_COLUMNS) & set(view.columns)
    assert "status" not in view.columns and "comment" not in view.columns
    assert {"game_id", "player_id", "s_minutes", "is_home"} <= set(view.columns)
    # Original frame untouched.
    assert "minutes" in features.columns


def test_assert_training_before(features):
    train = features[features["game_date"] < ORIGIN]
    assert_training_before(train, ORIGIN)
    with pytest.raises(LeakageError):
        assert_training_before(features[features["game_date"] <= ORIGIN], ORIGIN)
    assert_training_before(features.iloc[:0], ORIGIN)


def test_assert_disjoint_games(features):
    train = features[features["game_date"] < ORIGIN]
    later = features[features["game_date"] >= ORIGIN]["game_id"].unique()
    assert_disjoint_games(train, later)
    with pytest.raises(LeakageError):
        assert_disjoint_games(train, [train["game_id"].iloc[-1]])


def test_model_refuses_rows_before_origin(fitted_models, features):
    early = features[(features["game_date"] < ORIGIN) & (features["game_date"] >= ORIGIN - pd.Timedelta(days=2))]
    for name, model in fitted_models.items():
        with pytest.raises(ValueError, match="precede"):
            model.simulate(early, 10, np.random.default_rng(0))


def test_model_fit_refuses_training_on_or_after_origin(cfg, features):
    from nba_sim.models import build_model

    with pytest.raises(LeakageError):
        build_model("A", cfg).fit(features[features["game_date"] <= ORIGIN], ORIGIN)


def test_simulate_does_not_read_outcomes(fitted_models, target_rows):
    """Same seed, outcomes scrambled in the target rows -> identical draws."""
    scrambled = target_rows.copy()
    rng = np.random.default_rng(1)
    for c in ["minutes", *COUNT_STATS, "team_margin", "team_minutes"]:
        scrambled[c] = rng.permutation(scrambled[c].to_numpy())
    m = fitted_models["C"]
    a = m.simulate(target_rows, 50, np.random.default_rng(9))
    b = m.simulate(scrambled, 50, np.random.default_rng(9))
    np.testing.assert_array_equal(a.minutes, b.minutes)
    np.testing.assert_array_equal(a.stat("pts"), b.stat("pts"))


@pytest.fixture(scope="module")
def prepared(ds, cfg):
    games = ds.games[ds.games["game_date"] == pd.Timestamp("2023-01-10")]
    gid = int(games["game_id"].iloc[0])
    spec = historical_spec(ds, gid)
    history, target = prepare_target(ds, spec, cfg)
    return gid, spec, history, target


def test_prepare_target_is_outcome_free(prepared, ds):
    gid, spec, history, target = prepared
    assert len(target) > 10
    assert (target["game_id"] == gid).all()
    assert target[["minutes", *COUNT_STATS, "team_margin", "team_minutes"]].isna().all().all()
    assert history["game_date"].max() < spec.date
    assert gid not in set(history["game_id"])
    n_active = ((ds.player_games["game_id"] == gid) & (ds.player_games["status"] != "inactive")).sum()
    assert len(target) == n_active
    assert target["team_id"].nunique() == 2


def test_prepare_target_features_match_full_build(prepared, features):
    """The demo path (cut + append outcome-free rows) yields the same features as the backtest path."""
    gid, _, _, target = prepared
    full = features[features["game_id"] == gid].set_index("player_id").sort_index()
    tgt = target.set_index("player_id").sort_index()
    assert list(full.index) == list(tgt.index)
    cols = ["s_minutes", "l_pts", "s_one", "v_played", "n_prior", "last3_min", "days_rest", "missed_games",
            "pace_dev", "oppdef_pts", "vac_pts", "team_mpg_sum", "n_active", "t_pace", "lg_pace"]
    for c in cols:
        assert np.allclose(full[c].to_numpy(float), tgt[c].to_numpy(float), equal_nan=True), c
    assert np.isfinite(tgt["pace_dev"]).all()


def test_prepare_target_history_matches_full_build(prepared, features):
    """History features from the cut equal the full build's features for the same rows."""
    _, spec, history, _ = prepared
    full = features[features["game_date"] < spec.date].set_index(["game_id", "player_id"]).sort_index()
    hist = history.set_index(["game_id", "player_id"]).sort_index()
    assert full.index.equals(hist.index)
    for c in ["s_minutes", "l_pts", "pace_dev", "vac_reb"]:
        assert np.allclose(full[c].to_numpy(float), hist[c].to_numpy(float), equal_nan=True), c
