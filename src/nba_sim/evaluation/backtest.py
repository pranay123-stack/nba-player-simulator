"""Rolling-origin backtest.

Stage 1 (predict): for each origin, fit every model on rows dated strictly before the origin, then
simulate the games in [origin, next origin) from their pregame view and write frozen prediction
files. Player/team features inside the window still update game by game (they only use earlier
games); model parameters are refit at each origin.

Stage 2 (evaluate): load the frozen files, join the actual outcomes, compute scores. Outcomes are
read for the first time here.
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nba_sim.evaluation import metrics as M
from nba_sim.features.leakage_checks import assert_disjoint_games, assert_training_before
from nba_sim.models import build_model
from nba_sim.models.common import PRIMS
from nba_sim.reporting.manifest import code_fingerprint, write_manifest
from nba_sim.simulation.predictive import (
    JOINT_STATS,
    MINUTE_LEVELS,
    SUPPORT,
    concat_summaries,
    load_predictions,
    save_predictions,
    summarize,
)

log = logging.getLogger(__name__)

SCORED_STATS = ["minutes", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a",
                "ftm", "fta", "pra"]


@dataclass
class Window:
    idx: int
    season: str
    origin: pd.Timestamp
    end: pd.Timestamp


def season_list(features: pd.DataFrame) -> list[str]:
    return sorted(features["season"].unique())


def make_windows(features: pd.DataFrame, seasons: list[str], refit_days: int) -> list[Window]:
    out = []
    for season in seasons:
        dates = features.loc[features["season"] == season, "game_date"]
        if dates.empty:
            raise ValueError(f"season {season} not in dataset")
        start, last = dates.min(), dates.max()
        origin = start
        while origin <= last:
            end = origin + pd.Timedelta(days=refit_days)
            if ((dates >= origin) & (dates < end)).any():
                out.append(Window(len(out), season, origin, end))
            origin = end
    return out


def training_frame(features: pd.DataFrame, origin: pd.Timestamp, season: str, lookback: int) -> pd.DataFrame:
    seasons = season_list(features)
    i = seasons.index(season)
    allowed = seasons[max(0, i - lookback): i + 1]
    train = features[(features["game_date"] < origin) & features["season"].isin(allowed)]
    assert_training_before(train, origin)
    return train


def chunk_games(rows: pd.DataFrame, games_per_chunk: int):
    gids = rows["game_id"].unique()
    for a in range(0, len(gids), games_per_chunk):
        yield rows[rows["game_id"].isin(gids[a:a + games_per_chunk])]


def simulate_rows(model, rows: pd.DataFrame, n_sims: int, seed_seq: list[int], joint: int,
                  games_per_chunk: int = 30) -> tuple[dict, pd.DataFrame]:
    parts, idents = [], []
    for c, chunk in enumerate(chunk_games(rows, games_per_chunk)):
        rng = np.random.default_rng([*seed_seq, c])
        res = model.simulate(chunk, n_sims, rng)
        parts.append(summarize(res, joint))
        idents.append(res.rows)
    return concat_summaries(parts), pd.concat(idents, ignore_index=True)


def run_predictions(cfg: dict, features: pd.DataFrame, out_dir: Path, seasons: list[str] | None = None,
                    models: list[str] | None = None, n_sims: int | None = None,
                    max_windows: int | None = None) -> dict:
    bc = cfg["backtest"]
    seasons = seasons or bc["eval_seasons"]
    models = models or bc["models"]
    n_sims = n_sims or bc["n_simulations"]
    windows = make_windows(features, seasons, bc["refit_every_days"])
    if max_windows:
        windows = windows[:max_windows]
    log.info("backtest: %d windows over %s, models %s, %d sims", len(windows), seasons, models, n_sims)
    t0 = time.time()
    code = code_fingerprint()   # recorded in every prediction file: the code that produced it
    for w in windows:
        rows = features[(features["game_date"] >= w.origin) & (features["game_date"] < w.end)]
        train = training_frame(features, w.origin, w.season, bc["train_lookback_seasons"])
        assert_disjoint_games(train, rows["game_id"].unique())
        for mi, name in enumerate(models):
            path = out_dir / "predictions" / name / f"w{w.idx:03d}"
            if path.with_suffix(".npz").exists():
                continue
            t = time.time()
            model = build_model(name, cfg).fit(train, w.origin)
            summary, ident = simulate_rows(model, rows, n_sims, [bc["seed"], w.idx, mi], bc["joint_samples"])
            meta = {"window": w.idx, "season": w.season, "origin": str(w.origin.date()),
                    "end_exclusive": str(w.end.date()), "model": name, "n_sims": n_sims,
                    "seed_sequence": [bc["seed"], w.idx, mi], "n_train_rows": int(len(train)),
                    "train_date_max": str(train["game_date"].max().date()),
                    "n_rows": int(len(ident)), "n_games": int(ident["game_id"].nunique()), "code": code}
            save_predictions(path, summary, ident, meta)
            (out_dir / "models" / name).mkdir(parents=True, exist_ok=True)
            (out_dir / "models" / name / f"w{w.idx:03d}.json").write_text(json.dumps(model.describe(), indent=1))
            log.info("w%03d %s %s model %s: %d rows, %.1fs (elapsed %.0fs)", w.idx, w.season, w.origin.date(),
                     name, len(ident), time.time() - t, time.time() - t0)
    return {"windows": [w.__dict__ | {"origin": str(w.origin.date()), "end": str(w.end.date())} for w in windows],
            "models": models, "n_sims": n_sims, "seasons": seasons}


# ----------------------------------------------------------------------------- evaluation

def _lvl(level: float) -> int:
    """Index of a quantile level in MINUTE_LEVELS."""
    i = int(np.argmin(np.abs(MINUTE_LEVELS - level)))
    if abs(MINUTE_LEVELS[i] - level) > 1e-9:
        raise ValueError(f"level {level} not on the minutes quantile grid")
    return i


def _pregame_segments(f: pd.DataFrame) -> pd.DataFrame:
    """Segments defined from pregame information only (model independent)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        exp_min = np.where(f["s_one"] > 0, f["s_minutes"] / f["s_one"], np.nan)
    role = pd.cut(pd.Series(exp_min), [-np.inf, 12, 24, 30, np.inf],
                  labels=["deep bench (<12)", "bench (12-24)", "rotation (24-30)", "starter (30+)"]).astype(object)
    role[np.isnan(exp_min)] = "no history"
    hist = pd.cut(f["n_prior"], [-1, 19, 99, np.inf], labels=["<20 games", "20-99 games", "100+ games"]).astype(str)
    return pd.DataFrame({"role": role.to_numpy(), "history": hist.to_numpy(), "exp_min": exp_min}, index=f.index)


def evaluate_block(arrays: dict, ident: pd.DataFrame, actual: pd.DataFrame, model: str, window: int,
                   thresholds: dict, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    a = ident[["game_id", "team_id", "player_id"]].merge(actual, on=["game_id", "player_id"], how="left",
                                                         suffixes=("", "_a"))
    if a["minutes"].isna().any():
        raise ValueError("missing actual outcomes for predicted rows")
    base = pd.DataFrame({"model": model, "window": window, "game_id": a["game_id"], "team_id": a["team_id"],
                         "player_id": a["player_id"], "season": a["season"], "game_date": a["game_date"],
                         "role": a["role"], "history": a["history"], "exp_min": a["exp_min"]})
    frames, thr = [], []
    for stat in SCORED_STATS:
        y = a[stat].to_numpy()
        rec = base.copy()
        rec["stat"] = stat
        rec["y"] = y
        if stat == "minutes":
            q = arrays["min_q"].astype(np.float64)
            rec["mean"] = arrays["mean_minutes"]
            rec["median"] = q[:, _lvl(0.5)]
            rec["crps"] = M.crps_quantiles(q, MINUTE_LEVELS, y)
            rec["logs"] = np.nan
            rec["logs_floored"] = False
            rec["pit"] = M.pit_from_quantiles(q, MINUTE_LEVELS, y, arrays["min_p0"], rng)
            for lvl in (50, 80, 95):
                a2 = (1 - lvl / 100.0) / 2
                lo, hi = q[:, _lvl(a2)], q[:, _lvl(1 - a2)]
                rec[f"cov{lvl}"] = (y >= lo) & (y <= hi)
                rec[f"mass{lvl}"] = lvl / 100.0
        else:
            pmf = arrays[f"pmf_{stat}"].astype(np.float64)
            rec["mean"] = arrays[f"mean_{stat}"]
            rec["median"] = M.quantile_discrete(pmf, 0.5)
            rec["crps"] = M.crps_discrete(pmf, y)
            if stat in PRIMS:
                rec["logs"], rec["logs_floored"] = M.log_score(arrays[f"rb_{stat}"], y)
            else:
                rec["logs"], rec["logs_floored"] = np.nan, False
            rec["pit"] = M.pit_randomized(pmf, y, rng)
            for lvl in (50, 80, 95):
                lo, hi, mass = M.central_interval(pmf, 1 - lvl / 100.0)
                rec[f"cov{lvl}"] = (y >= lo) & (y <= hi)
                rec[f"mass{lvl}"] = mass
        rec["abs_err"] = np.abs(y - rec["median"])
        rec["sq_err"] = (y - rec["mean"]) ** 2
        frames.append(rec)
        for t in thresholds.get(stat, []):
            if stat == "minutes":
                q = arrays["min_q"].astype(np.float64)
                p = np.array([1 - np.interp(t, q[i], MINUTE_LEVELS, left=0, right=1) for i in range(len(y))])
            else:
                p = M.prob_at_least(arrays[f"pmf_{stat}"].astype(np.float64), t)
            tr = base[["model", "window", "game_id", "player_id", "season", "role"]].copy()
            tr["stat"], tr["threshold"], tr["p"], tr["event"] = stat, t, p, (y >= t)
            tr["brier"] = M.brier(p, y >= t)
            thr.append(tr)

    # Team level: totals and margins from aligned draws.
    team_idx = arrays["team_idx"]
    team_pts = arrays["team_pts"].astype(np.float64)
    act_team = np.zeros(team_pts.shape[0])
    np.add.at(act_team, team_idx, a["pts"].to_numpy())
    first_row = np.unique(team_idx, return_index=True)[1]
    teams = pd.DataFrame({"model": model, "window": window, "game_id": a["game_id"].to_numpy()[first_row],
                          "team_id": a["team_id"].to_numpy()[first_row], "season": a["season"].to_numpy()[first_row],
                          "y": act_team})
    teams["mean"] = team_pts.mean(axis=1)
    teams["sd"] = team_pts.std(axis=1)
    teams["crps"] = M.crps_samples(team_pts, act_team)
    teams["pit"] = M.pit_samples(team_pts, act_team, rng)
    lo, hi = np.quantile(team_pts, [0.1, 0.9], axis=1)
    teams["cov80"] = (act_team >= lo) & (act_team <= hi)
    opp = arrays["opp_team"]
    margin = team_pts - team_pts[opp]
    act_margin = act_team - act_team[opp]
    teams["margin_y"] = act_margin
    teams["margin_sd"] = margin.std(axis=1)
    teams["margin_crps"] = M.crps_samples(margin, act_margin)
    teams["margin_pit"] = M.pit_samples(margin, act_margin, rng)

    # Dependence: variogram score over teammates + team-sum dispersion of joint draws.
    groups = team_idx
    joint_rows = []
    for stat in JOINT_STATS:
        J = arrays[f"joint_{stat}"].astype(np.float64)
        y = a[stat].to_numpy(float)
        vs = M.variogram_score(J, y, groups)
        tot_draw = np.zeros((team_pts.shape[0], J.shape[1]))
        np.add.at(tot_draw, groups, J)
        tot_y = np.zeros(team_pts.shape[0])
        np.add.at(tot_y, groups, y)
        # Same-draw independence benchmark: team-total variance if teammates were independent.
        var_indep = np.zeros(team_pts.shape[0])
        np.add.at(var_indep, groups, J.var(axis=1))
        joint_rows.append(pd.DataFrame({
            "model": model, "window": window, "game_id": teams["game_id"], "team_id": teams["team_id"],
            "season": teams["season"], "stat": stat, "variogram": vs,
            "sq_err_total": (tot_y - tot_draw.mean(axis=1)) ** 2, "var_total": tot_draw.var(axis=1),
            "var_total_if_indep": var_indep,
        }))
    return {"player": pd.concat(frames, ignore_index=True), "threshold": pd.concat(thr, ignore_index=True),
            "team": teams, "joint": pd.concat(joint_rows, ignore_index=True)}


def run_evaluation(cfg: dict, features: pd.DataFrame, out_dir: Path, models: list[str] | None = None) -> dict:
    models = models or cfg["backtest"]["models"]
    seg = _pregame_segments(features)
    outcome_cols = ["game_id", "player_id", "season", "game_date", "minutes", "pts", "reb", "ast", "stl",
                    "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]
    actual = features[outcome_cols].join(seg)
    actual["pra"] = actual["pts"] + actual["reb"] + actual["ast"]
    for c in ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "pra"]:
        actual[c] = actual[c].astype(np.int64)
    out = {k: [] for k in ("player", "threshold", "team", "joint")}
    thresholds = cfg["evaluation"]["thresholds"]
    for name in models:
        files = sorted((out_dir / "predictions" / name).glob("w*.npz"))
        if not files:
            raise FileNotFoundError(f"no frozen predictions for model {name} in {out_dir}")
        for f in files:
            arrays, ident, meta = load_predictions(f.with_suffix(""))
            rng = np.random.default_rng([cfg["backtest"]["seed"], 7, meta["window"], zlib.crc32(name.encode())])
            res = evaluate_block(arrays, ident, actual, name, meta["window"], thresholds, rng)
            for k in out:
                out[k].append(res[k])
        log.info("evaluated model %s: %d blocks", name, len(files))
    score_dir = out_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, parts in out.items():
        df = pd.concat(parts, ignore_index=True)
        p = score_dir / f"{k}_scores.parquet"
        df.to_parquet(p, index=False)
        paths[k] = str(p)
    return paths


def run_backtest(cfg: dict, features: pd.DataFrame, out_dir: Path, dataset_manifest: dict, **kw) -> dict:
    started = pd.Timestamp.now(tz="UTC")
    pred = run_predictions(cfg, features, out_dir, **kw)
    scores = run_evaluation(cfg, features, out_dir, pred["models"])
    manifest = write_manifest(out_dir / "manifest.json", cfg, dataset_manifest, extra={
        "kind": "backtest", "started_utc": str(started), "prediction_stage": pred, "score_files": scores,
        "n_simulations": pred["n_sims"],
    })
    return manifest
