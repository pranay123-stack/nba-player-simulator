"""Backtest chronology, the full predict-freeze-score pipeline, and the CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nba_sim.config import load_config
from nba_sim.evaluation import backtest as bt
from nba_sim.features.leakage_checks import LeakageError, assert_disjoint_games

EVAL_FROM = pd.Timestamp("2022-12-01")   # first scored origin gets ~7 weeks of same-season history
E2E_MODELS = ["A", "B", "C"]


# ----------------------------------------------------------------------------- chronology

def test_make_windows_ordered_contiguous_and_covering(features):
    wins = bt.make_windows(features, ["2022-23"], 14)
    assert len(wins) >= 6
    assert [w.idx for w in wins] == list(range(len(wins)))
    for a, b in zip(wins[:-1], wins[1:]):
        assert a.origin < a.end <= b.origin, "windows must be ordered and non-overlapping"
    for w in wins:
        assert w.end - w.origin == pd.Timedelta(days=14)
    dates = pd.Series(features["game_date"].unique())
    hits = sum(((dates >= w.origin) & (dates < w.end)).astype(int) for w in wins)
    assert (hits == 1).all(), "every game date belongs to exactly one window"
    assert wins[0].origin == features["game_date"].min()


def test_make_windows_unknown_season(features):
    with pytest.raises(ValueError):
        bt.make_windows(features, ["1999-00"], 14)


def test_training_frame_strictly_before_origin(features):
    for w in bt.make_windows(features, ["2022-23"], 14)[1:]:
        train = bt.training_frame(features, w.origin, w.season, lookback=3)
        assert len(train) > 0
        assert train["game_date"].max() < w.origin
        target = features[(features["game_date"] >= w.origin) & (features["game_date"] < w.end)]
        assert_disjoint_games(train, target["game_id"].unique())
        with pytest.raises(LeakageError):
            assert_disjoint_games(features[features["game_date"] < w.end], target["game_id"].unique())


# ----------------------------------------------------------------------------- end to end

def _e2e_cfg() -> dict:
    return load_config(overrides={"backtest": {
        "n_simulations": 100, "refit_every_days": 7, "seed": 20251029, "eval_seasons": ["2022-23"],
        "models": E2E_MODELS, "joint_samples": 100,
    }})


def _run(features, ds, out_dir):
    """Run the real backtest, skipping origins before EVAL_FROM so the scored window has history."""
    real = bt.make_windows

    def later_windows(feats, seasons, refit_days):
        return [w for w in real(feats, seasons, refit_days) if w.origin >= EVAL_FROM]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bt, "make_windows", later_windows)
        return bt.run_backtest(_e2e_cfg(), features, out_dir, ds.manifest, max_windows=1)


@pytest.fixture(scope="module")
def backtest_runs(features, ds, tmp_path_factory):
    d1 = tmp_path_factory.mktemp("bt_run1")
    d2 = tmp_path_factory.mktemp("bt_run2")
    return (d1, _run(features, ds, d1)), (d2, _run(features, ds, d2))


def test_backtest_writes_frozen_predictions(backtest_runs):
    (out, manifest), _ = backtest_runs
    windows = manifest["prediction_stage"]["windows"]
    assert len(windows) == 1
    w = windows[0]
    assert pd.Timestamp(w["origin"]) >= EVAL_FROM
    for name in E2E_MODELS:
        files = sorted((out / "predictions" / name).glob("w*.npz"))
        assert len(files) == 1, name
        base = files[0].with_suffix("")
        assert base.with_suffix(".rows.parquet").exists()
        assert base.with_suffix(".meta.json").exists()
        assert (out / "models" / name / f"{base.name}.json").exists()
        meta = json.loads(base.with_suffix(".meta.json").read_text())
        assert meta["model"] == name
        assert meta["n_sims"] == 100
        assert meta["seed_sequence"] == [20251029, w["idx"], E2E_MODELS.index(name)]
        assert pd.Timestamp(meta["train_date_max"]) < pd.Timestamp(meta["origin"])
        assert meta["origin"] == w["origin"]
        rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
        assert len(rows) == meta["n_rows"] > 0
        # Frozen rows carry identities only, never outcomes.
        assert not {"minutes", "pts", "reb", "status"} & set(rows.columns)
        dates = pd.to_datetime(rows["game_date"])
        assert dates.min() >= pd.Timestamp(meta["origin"])
        assert dates.max() < pd.Timestamp(meta["end_exclusive"])
        with np.load(files[0]) as z:
            assert z["pmf_pts"].shape[0] == len(rows)
            assert int(z["n_sims"][0]) == 100
    assert (out / "manifest.json").exists()
    assert manifest["n_simulations"] == 100


def test_backtest_scores_valid(backtest_runs):
    (out, _), _ = backtest_runs
    for k in ("player", "threshold", "team", "joint"):
        assert (out / "scores" / f"{k}_scores.parquet").exists(), k
    player = pd.read_parquet(out / "scores" / "player_scores.parquet")
    assert set(player["model"]) == set(E2E_MODELS)
    assert set(player["stat"]) == set(bt.SCORED_STATS)
    assert np.isfinite(player["crps"]).all()
    assert (player["crps"] >= 0).all()
    assert player["pit"].between(0, 1).all()
    prims = player[player["stat"].isin(["reb", "ast", "stl", "blk", "tov", "fga", "fta"])]
    assert np.isfinite(prims["logs"]).all()
    thr = pd.read_parquet(out / "scores" / "threshold_scores.parquet")
    assert thr["p"].between(0, 1).all()
    assert thr["brier"].between(0, 1).all()
    team = pd.read_parquet(out / "scores" / "team_scores.parquet")
    assert np.isfinite(team["crps"]).all() and np.isfinite(team["margin_crps"]).all()
    joint = pd.read_parquet(out / "scores" / "joint_scores.parquet")
    assert (joint["variogram"] >= 0).all()


def test_backtest_reproducible(backtest_runs):
    (out1, _), (out2, _) = backtest_runs
    for k in ("player", "threshold", "team", "joint"):
        a = pd.read_parquet(out1 / "scores" / f"{k}_scores.parquet")
        b = pd.read_parquet(out2 / "scores" / f"{k}_scores.parquet")
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    for name in E2E_MODELS:
        f1 = next((out1 / "predictions" / name).glob("w*.npz"))
        f2 = out2 / "predictions" / name / f1.name
        with np.load(f1) as z1, np.load(f2) as z2:
            assert z1.files == z2.files
            for key in z1.files:
                np.testing.assert_array_equal(z1[key], z2[key], err_msg=f"{name}:{key}")


def test_backtest_skips_existing_predictions(backtest_runs, features, ds):
    """Re-running into the same directory must not overwrite frozen predictions."""
    (out, _), _ = backtest_runs
    f = next((out / "predictions" / "A").glob("w*.npz"))
    mtime = f.stat().st_mtime_ns
    _run(features, ds, out)
    assert f.stat().st_mtime_ns == mtime


# ----------------------------------------------------------------------------- CLI

def _out(result) -> str:
    text = result.output or ""
    try:
        text += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return text


def test_cli_help():
    from nba_sim.cli import app

    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0, _out(res)
    for cmd in ("inspect-data", "build-dataset", "backtest", "simulate"):
        assert cmd in res.output


def test_cli_inspect_data_missing_raw(tmp_path):
    from nba_sim.cli import app

    res = CliRunner().invoke(app, ["inspect-data", "--data-path", str(tmp_path)])
    assert res.exit_code != 0
    assert "no raw box-score CSVs" in _out(res)
    assert not isinstance(res.exception, Exception) or isinstance(res.exception, SystemExit)
