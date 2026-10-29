"""Construct outcome-free pregame rows for a target game and fit-ready history.

Both the historical demo and the `simulate` command go through ``prepare_target``: the raw data is
cut to games dated strictly before the target date, outcome-free rows for the target game are
appended, and features are rebuilt from that cut. The target game's box score therefore never
enters feature construction or model fitting, by construction rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_sim.data.ingest import Dataset
from nba_sim.data.schema import COUNT_STATS, STATUS_DNP, STATUS_INACTIVE
from nba_sim.features.build import build_features
from nba_sim.features.leakage_checks import LeakageError

TARGET_GAME_ID = -1  # placeholder id for hypothetical games


@dataclass
class TargetSpec:
    date: pd.Timestamp
    home_team_id: int
    away_team_id: int
    home: str
    away: str
    game_id: int = TARGET_GAME_ID
    # player_id -> team_id; None means "derive from recent appearances"
    active: dict[int, int] | None = None
    names: dict[int, str] | None = None


def team_lookup(ds: Dataset, code: str, before: pd.Timestamp) -> int:
    tg = ds.team_games[(ds.team_games["team"] == code.upper()) & (ds.team_games["game_date"] < before)]
    if tg.empty:
        known = sorted(ds.team_games["team"].unique())
        raise ValueError(f"unknown team code {code!r} before {before.date()}; known codes: {known}")
    return int(tg["team_id"].iloc[-1])


def recent_roster(ds: Dataset, team_id: int, before: pd.Timestamp, last_n_games: int = 10) -> dict[int, str]:
    """Players whose latest active appearance before ``before`` was for ``team_id`` and who were
    active in at least one of the team's last ``last_n_games`` games."""
    pg = ds.player_games
    hist = pg[(pg["game_date"] < before) & (pg["status"] != STATUS_INACTIVE)]
    team_games = hist.loc[hist["team_id"] == team_id, ["game_id", "game_date"]].drop_duplicates()
    recent_ids = team_games.sort_values("game_date")["game_id"].tail(last_n_games)
    cand = hist[hist["game_id"].isin(recent_ids) & (hist["team_id"] == team_id)]["player_id"].unique()
    latest = hist[hist["player_id"].isin(cand)].sort_values("game_date").groupby("player_id").tail(1)
    latest = latest[latest["team_id"] == team_id]
    return dict(zip(latest["player_id"], latest["player_name"]))


def historical_spec(ds: Dataset, game_id: int) -> TargetSpec:
    g = ds.games[ds.games["game_id"] == game_id]
    if g.empty:
        raise ValueError(f"game_id {game_id} not found in dataset")
    g = g.iloc[0]
    rows = ds.player_games[(ds.player_games["game_id"] == game_id) & (ds.player_games["status"] != STATUS_INACTIVE)]
    return TargetSpec(date=pd.Timestamp(g["game_date"]), home_team_id=int(g["home_team_id"]),
                      away_team_id=int(g["away_team_id"]), home=g["home"], away=g["away"], game_id=int(game_id),
                      active=dict(zip(rows["player_id"], rows["team_id"])),
                      names=dict(zip(rows["player_id"], rows["player_name"])))


def _target_rows(ds: Dataset, spec: TargetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.active is None:
        active, names = {}, {}
        for tid in (spec.home_team_id, spec.away_team_id):
            roster = recent_roster(ds, tid, spec.date)
            if len(roster) < 5:
                raise ValueError(f"team {tid}: only {len(roster)} recent players before {spec.date.date()}")
            active.update({pid: tid for pid in roster})
            names.update(roster)
    else:
        active, names = spec.active, spec.names or {}
    teams = {spec.home_team_id: spec.home, spec.away_team_id: spec.away}
    opp = {spec.home_team_id: spec.away_team_id, spec.away_team_id: spec.home_team_id}
    season = _season_of(ds, spec.date)
    rows = pd.DataFrame({
        "game_id": spec.game_id, "season": season, "game_date": spec.date,
        "team_id": [active[p] for p in active], "player_id": list(active),
    })
    rows["team"] = rows["team_id"].map(teams)
    rows["opp_team_id"] = rows["team_id"].map(opp)
    rows["opp"] = rows["opp_team_id"].map(teams)
    rows["is_home"] = rows["team_id"] == spec.home_team_id
    rows["player_name"] = rows["player_id"].map(lambda p: names.get(p, str(p)))
    rows["status"] = STATUS_DNP          # active list; the outcome itself is unknown
    rows["starter"] = False              # starting lineups are NOT used (no pregame timestamp)
    rows["minutes"] = np.nan
    for c in COUNT_STATS + ["plus_minus"]:
        rows[c] = np.nan
    rows["season_type"] = "regular"
    rows["comment"] = None
    team_rows = pd.DataFrame({
        "game_id": spec.game_id, "team_id": [spec.home_team_id, spec.away_team_id], "season": season,
        "season_type": "regular", "game_date": spec.date, "team": [spec.home, spec.away],
        "opp_team_id": [spec.away_team_id, spec.home_team_id], "opp": [spec.away, spec.home],
        "is_home": [True, False],
    })
    return rows, team_rows


def _season_of(ds: Dataset, date: pd.Timestamp) -> str:
    seasons = ds.games.groupby("season")["game_date"].agg(["min", "max"])
    inside = seasons[(seasons["min"] <= date) & (seasons["max"] >= date)]
    if len(inside):
        return inside.index[0]
    y = date.year if date.month >= 8 else date.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def prepare_target(ds: Dataset, spec: TargetSpec, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (history_features, target_rows): history strictly before the target date, target rows
    with pregame features and NO outcomes."""
    pg = ds.player_games[ds.player_games["game_date"] < spec.date]
    tg = ds.team_games[ds.team_games["game_date"] < spec.date]
    if spec.game_id in set(pg["game_id"]) or spec.game_id in set(tg["game_id"]):
        raise LeakageError("target game present in the history cut")
    rows, team_rows = _target_rows(ds, spec)
    pg_all = pd.concat([pg, rows[pg.columns.intersection(rows.columns)]], ignore_index=True)
    tg_all = pd.concat([tg, team_rows], ignore_index=True)
    feats = build_features(pg_all, tg_all, cfg)
    target = feats[feats["game_id"] == spec.game_id].copy()
    history = feats[feats["game_id"] != spec.game_id].copy()
    outcome_cols = ["minutes", *COUNT_STATS, "team_margin", "team_minutes"]
    if target[outcome_cols].notna().any().any():
        raise LeakageError("target rows carry outcome values")
    return history, target
