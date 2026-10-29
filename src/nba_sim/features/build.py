"""Pregame feature construction.

Every feature for a player-game row is a function of (a) outcomes from strictly earlier games and
(b) pregame facts about the target game itself (date, home/away, opponent, the active list).
Player histories are exponentially decayed sums shifted by one game; team and league features are
shifted by one team-game / one calendar day. No feature is normalized with full-sample statistics;
shrinkage priors are applied later, at model-fit time, from training rows only.

FEATURE_DOCS records, for each feature, what it is built from and when it becomes available.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from nba_sim.data.schema import STATUS_INACTIVE, STATUS_PLAYED

log = logging.getLogger(__name__)

# Counting stats tracked in player histories (fg2a/fg2m derived).
HIST_STATS = ["fga", "fg3a", "fg3m", "fgm", "fta", "ftm", "reb", "oreb", "ast", "stl", "blk", "tov", "pts"]
# Stats whose opponent-allowed environment is tracked at team level.
TEAM_STATS = ["fga", "fg3a", "fta", "reb", "oreb", "ast", "stl", "blk", "tov", "pts"]

FEATURE_DOCS: dict[str, str] = {
    "n_prior": "count of the player's earlier active games (any team) | before tip-off",
    "s_<x>, l_<x>": "decayed sums (short/long half-life, in player games) of x over earlier active games; "
                    "x in one, played, starter, minutes, min2 and counting stats | before tip-off",
    "v_<x>": "3-game half-life decayed sums of one/played/starter/minutes (rotation status) | before tip-off",
    "last_min": "minutes in the player's previous active game | before tip-off",
    "last3_min": "mean minutes over the previous 3 active games | before tip-off",
    "days_rest": "days since the player's previous active game, capped at 10 | before tip-off",
    "b2b": "previous active game was the day before | schedule, before tip-off",
    "missed_games": "team games missed since the player's previous active game for this team | before tip-off",
    "new_team": "player's previous active game was for a different team (or none) | before tip-off",
    "is_home": "home indicator | schedule",
    "n_active": "players on the team's active list | ASSUMED known pregame (official inactive list); "
                "this dataset only records it in the postgame box score",
    "team_mpg_sum": "sum over the active list of each player's decayed minutes-per-active-game | "
                    "active list + earlier games",
    "pace_dev": "log(expected game pace / league pace) from both teams' decayed pace through the "
                "previous game, league mean as of the previous day | before tip-off",
    "oppdef_<s>": "log(opponent's decayed allowed-per-possession rate of s / league mean as of the "
                  "previous day) | before tip-off",
    "vac_<s>": "log(team's decayed per-game s / production expected from the active list at usual "
               "per-minute rates and minutes rescaled to 240) | active list + earlier games",
}


def decay_factor(halflife: float) -> float:
    return 0.5 ** (1.0 / halflife)


def lagged_decayed_sums(values: np.ndarray, group: np.ndarray, d: float) -> np.ndarray:
    """Per-group decayed sums S_t = x_t + d*S_{t-1}, returned *lagged* (value before row t).

    ``values`` is (n, k) sorted so each group is contiguous and chronological; ``group`` holds the
    group label per row. The result at row t only depends on rows t' < t within the same group.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    starts = np.r_[True, group[1:] != group[:-1]] if n else np.zeros(0, bool)
    bounds = np.r_[np.flatnonzero(starts), n]
    within = np.empty_like(values)
    # Filter each group separately so a row never depends, even numerically, on another group.
    for a, b in zip(bounds[:-1], bounds[1:]):
        within[a:b] = lfilter([1.0], [1.0, -d], values[a:b], axis=0)
    lagged = np.empty_like(within)
    lagged[1:] = within[:-1]
    lagged[starts] = 0.0
    return lagged


def _player_features(act: pd.DataFrame, short_h: float, long_h: float, recent_h: float = 3.0) -> pd.DataFrame:
    act = act.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    grp = act["player_id"].to_numpy()
    played = (act["status"] == STATUS_PLAYED).to_numpy(float)
    minutes = act["minutes"].to_numpy(float)
    base = {
        "one": np.ones(len(act)),
        "played": played,
        "starter": act["starter"].to_numpy(float),
        "minutes": minutes,
        "min2": minutes ** 2,
    }
    for s in HIST_STATS:
        base[s] = act[s].to_numpy(float)
    names = list(base)
    mat = np.column_stack([base[k] for k in names])
    short = lagged_decayed_sums(mat, grp, decay_factor(short_h))
    long_ = lagged_decayed_sums(mat, grp, decay_factor(long_h))
    out = {"s_" + k: short[:, i] for i, k in enumerate(names)}
    out.update({"l_" + k: long_[:, i] for i, k in enumerate(names)})
    # Very recent rotation status (minutes/play/start only): rotations change within a few games.
    rot = ["one", "played", "starter", "minutes"]
    recent = lagged_decayed_sums(np.column_stack([base[k] for k in rot]), grp, decay_factor(recent_h))
    out.update({"v_" + k: recent[:, i] for i, k in enumerate(rot)})
    feats = pd.DataFrame(out)

    g = act.groupby("player_id", sort=False)
    feats["n_prior"] = g.cumcount().to_numpy()
    feats["last_min"] = g["minutes"].shift(1).to_numpy()
    feats["last3_min"] = g["minutes"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).to_numpy()
    prev_date = g["game_date"].shift(1)
    rest = (act["game_date"] - prev_date).dt.days
    feats["days_rest"] = rest.clip(upper=10).to_numpy()
    feats["b2b"] = (rest == 1).astype(float).to_numpy()
    prev_team = g["team_id"].shift(1)
    prev_tidx = g["team_game_idx"].shift(1)
    same_team = (prev_team == act["team_id"]).to_numpy()
    feats["new_team"] = (~same_team).astype(float)
    feats["missed_games"] = np.where(same_team, act["team_game_idx"].to_numpy() - prev_tidx.to_numpy() - 1, 0.0)
    feats.index = act.index
    return pd.concat([act, feats], axis=1)


def _team_features(tg: pd.DataFrame, team_h: float) -> pd.DataFrame:
    tg = tg.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    opp_tot = tg[["game_id", "team_id", *TEAM_STATS]].rename(
        columns={"team_id": "opp_team_id", **{s: "allowed_" + s for s in TEAM_STATS}})
    tg = tg.merge(opp_tot, on=["game_id", "opp_team_id"], how="left")
    scale = 240.0 / tg["team_minutes"]
    cols = {"one": np.ones(len(tg)), "pace48": tg["pace48"], "opp_poss": tg["opp_poss"]}
    for s in TEAM_STATS:
        cols["pg_" + s] = tg[s] * scale                # own production per regulation game
        cols["al_" + s] = tg["allowed_" + s]           # allowed, per-possession later
    names = list(cols)
    mat = np.column_stack([np.asarray(cols[k], float) for k in names])
    sums = lagged_decayed_sums(mat, tg["team_id"].to_numpy(), decay_factor(team_h))
    S = pd.DataFrame(sums, columns=names)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({"game_id": tg["game_id"], "team_id": tg["team_id"], "game_date": tg["game_date"]})
        out["t_pace"] = S["pace48"] / S["one"]
        for s in TEAM_STATS:
            out["t_pg_" + s] = S["pg_" + s] / S["one"]
            out["t_al_" + s] = S["al_" + s] / S["opp_poss"]
    out["t_games"] = S["one"]
    # League means as of the previous calendar day: cross-sectional mean of each team's latest
    # lagged value, so the league reference never includes the target date.
    league_cols = ["t_pace"] + ["t_al_" + s for s in TEAM_STATS]
    daily = out.pivot_table(index="game_date", columns="team_id", values=league_cols, aggfunc="last")
    league = {}
    for c in league_cols:
        latest = daily[c].ffill()
        league[c] = latest.mean(axis=1)
    league = pd.DataFrame(league)
    # The pivot holds pregame values on each team's game dates; forward-filling them gives each
    # team's latest pregame value, which only uses games before that date. Shift one day so the
    # league mean on date D uses values set on dates < D.
    league = league.shift(1)
    league.columns = ["lg_" + c[2:] for c in league.columns]
    out = out.merge(league, left_on="game_date", right_index=True, how="left")
    return out


def build_features(pg: pd.DataFrame, tg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return one row per active (played or DNP) player-game with pregame features and outcomes."""
    fc = cfg["features"]
    tg = tg.sort_values(["team_id", "game_date", "game_id"]).copy()
    tg["team_game_idx"] = tg.groupby("team_id").cumcount()
    act = pg[pg["status"] != STATUS_INACTIVE].merge(
        tg[["game_id", "team_id", "team_game_idx"]], on=["game_id", "team_id"], how="left")
    feats = _player_features(act, fc["halflife_games"], fc["long_halflife_games"],
                             fc.get("recent_halflife_games", 3.0))

    team = _team_features(tg, fc["team_halflife_games"])
    own = team.drop(columns=["game_date"]).set_index(["game_id", "team_id"])
    feats = feats.join(own[["t_pace", "lg_pace", "t_games"] + [f"t_pg_{s}" for s in TEAM_STATS]],
                       on=["game_id", "team_id"])
    opp = own[["t_pace"] + [f"t_al_{s}" for s in TEAM_STATS] + [f"lg_al_{s}" for s in TEAM_STATS]]
    opp = opp.rename(columns={"t_pace": "o_pace", **{f"t_al_{s}": f"o_al_{s}" for s in TEAM_STATS}})
    opp.index = opp.index.set_names(["game_id", "opp_team_id"])
    feats = feats.join(opp, on=["game_id", "opp_team_id"])
    # Game outcomes used only as training targets (stripped from target rows by pregame_view).
    res = tg.set_index(["game_id", "team_id"])[["margin", "team_minutes"]].rename(columns={"margin": "team_margin"})
    feats = feats.join(res, on=["game_id", "team_id"])

    with np.errstate(divide="ignore", invalid="ignore"):
        feats["pace_dev"] = np.log(0.5 * (feats["t_pace"] + feats["o_pace"]) / feats["lg_pace"])
        for s in TEAM_STATS:
            feats[f"oppdef_{s}"] = np.log(feats[f"o_al_{s}"] / feats[f"lg_al_{s}"])
    feats = feats.drop(columns=[f"o_al_{s}" for s in TEAM_STATS] + [f"lg_al_{s}" for s in TEAM_STATS])

    # Active-list context (pregame by assumption, see FEATURE_DOCS["n_active"]).
    with np.errstate(divide="ignore", invalid="ignore"):
        mpg = np.where(feats["s_one"] > 0, feats["s_minutes"] / feats["s_one"], np.nan)
    feats["mpg_raw"] = mpg
    key = ["game_id", "team_id"]
    feats["n_active"] = feats.groupby(key)["player_id"].transform("size")
    feats["team_mpg_sum"] = feats.groupby(key)["mpg_raw"].transform(lambda x: x.fillna(0).sum())
    scale = (240.0 / feats["team_mpg_sum"].where(feats["team_mpg_sum"] > 0)).clip(0.6, 2.0)
    for s in TEAM_STATS:
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(feats["l_minutes"] > 0, feats[f"l_{s}"] / feats["l_minutes"], np.nan)
        expected = feats.assign(_e=np.nan_to_num(rate) * np.nan_to_num(mpg)).groupby(key)["_e"].transform("sum")
        with np.errstate(divide="ignore", invalid="ignore"):
            feats[f"vac_{s}"] = np.log(feats[f"t_pg_{s}"] / (scale * expected))
    feats = feats.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)
    log.info("built features: %d active player-games, %d columns", len(feats), feats.shape[1])
    return feats
