"""Data validation: structural integrity, plausibility and coverage checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_sim.data.schema import STATUS_INACTIVE, STATUS_PLAYED


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    coverage: pd.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
            "coverage": None if self.coverage is None else self.coverage.reset_index().to_dict("records"),
        }


def validate(pg: pd.DataFrame, tg: pd.DataFrame, minute_tol: float = 0.25) -> ValidationReport:
    rep = ValidationReport()
    s = rep.summary

    dups = pg.duplicated(["game_id", "player_id"]).sum()
    s["duplicate_player_game_rows"] = int(dups)
    if dups:
        rep.errors.append(f"{dups} duplicate (game_id, player_id) rows")

    teams_per_game = pg.groupby("game_id")["team_id"].nunique()
    bad_games = int((teams_per_game != 2).sum())
    s["games_without_exactly_two_teams"] = bad_games
    if bad_games:
        rep.errors.append(f"{bad_games} games do not have exactly two teams")

    s["rows_missing_opponent"] = int(pg["opp_team_id"].isna().sum())
    if s["rows_missing_opponent"]:
        rep.errors.append(f"{s['rows_missing_opponent']} rows without an opponent")

    home_counts = pg.drop_duplicates(["game_id", "team_id"]).groupby("game_id")["is_home"].sum()
    s["games_without_one_home_team"] = int((home_counts != 1).sum())
    if s["games_without_one_home_team"]:
        rep.errors.append(f"{s['games_without_one_home_team']} games without exactly one home team")

    active = pg[pg["status"] != STATUS_INACTIVE]
    played = pg[pg["status"] == STATUS_PLAYED]
    s["status_counts"] = pg["status"].value_counts().to_dict()

    # Minutes plausibility.
    max_possible = 48 + 5 * 6
    s["minutes_negative"] = int((active["minutes"] < 0).sum())
    s["minutes_over_78"] = int((active["minutes"] > max_possible).sum())
    s["active_rows_missing_minutes"] = int(active["minutes"].isna().sum())
    for k in ("minutes_negative", "minutes_over_78", "active_rows_missing_minutes"):
        if s[k]:
            rep.errors.append(f"{k}: {s[k]}")

    # Team minute accounting: 240 + 25 per overtime period.
    resid = tg["team_minutes"] - (240 + 25 * tg["n_ot"])
    s["team_minute_residual_max_abs"] = float(resid.abs().max())
    n_bad_tm = int((resid.abs() > minute_tol).sum())
    s["team_games_minutes_inconsistent"] = n_bad_tm
    if n_bad_tm:
        rep.warnings.append(f"{n_bad_tm} team-games whose minutes differ from 240+25*OT by >{minute_tol}")
    s["overtime_rate"] = float((tg["n_ot"] > 0).mean())

    # Internal box-score consistency.
    s["pts_identity_violations"] = int((played["pts"] != 2 * played["fgm"] + played["fg3m"] + played["ftm"]).sum())
    s["fgm_gt_fga"] = int((played["fgm"] > played["fga"]).sum())
    s["fg3m_gt_fgm"] = int((played["fg3m"] > played["fgm"]).sum())
    s["fg3a_gt_fga"] = int((played["fg3a"] > played["fga"]).sum())
    s["ftm_gt_fta"] = int((played["ftm"] > played["fta"]).sum())
    s["reb_identity_violations"] = int((played["oreb"] + played["dreb"] != played["reb"]).sum())
    for k in ("pts_identity_violations", "fgm_gt_fga", "fg3m_gt_fgm", "fg3a_gt_fga", "ftm_gt_fta",
              "reb_identity_violations"):
        if s[k]:
            rep.errors.append(f"{k}: {s[k]}")

    stats = ["pts", "reb", "ast", "stl", "blk", "tov", "fga", "fta"]
    s["negative_counts"] = int((active[stats] < 0).sum().sum())
    if s["negative_counts"]:
        rep.errors.append(f"{s['negative_counts']} negative counting stats")
    dnp = pg[pg["status"] == "dnp"]
    s["dnp_rows_with_nonzero_stats"] = int((dnp[stats].fillna(0).sum(axis=1) > 0).sum())
    if s["dnp_rows_with_nonzero_stats"]:
        rep.errors.append(f"{s['dnp_rows_with_nonzero_stats']} DNP rows with non-zero stats")
    zero_min = played[(played["minutes"] <= 0) & (played[stats].sum(axis=1) > 0)]
    s["zero_minute_rows_with_stats"] = int(len(zero_min))
    if len(zero_min):
        rep.warnings.append(f"{len(zero_min)} sub-second appearances (0:00 logged) credited with a stat")
    s["active_rows_missing_stats"] = int(active[stats].isna().any(axis=1).sum())
    if s["active_rows_missing_stats"]:
        rep.errors.append(f"{s['active_rows_missing_stats']} active rows with missing stats")

    # Suspicious-but-possible extremes are reported, not removed.
    s["extreme_rows"] = {
        "pts>=60": int((played["pts"] >= 60).sum()),
        "reb>=25": int((played["reb"] >= 25).sum()),
        "ast>=20": int((played["ast"] >= 20).sum()),
        "minutes>=55": int((played["minutes"] >= 55).sum()),
    }

    cov = pg.groupby("season").agg(
        games=("game_id", "nunique"),
        rows=("player_id", "size"),
        players=("player_id", "nunique"),
        played=("status", lambda x: int((x == "played").sum())),
        dnp=("status", lambda x: int((x == "dnp").sum())),
        inactive=("status", lambda x: int((x == STATUS_INACTIVE).sum())),
        first_date=("game_date", "min"),
        last_date=("game_date", "max"),
    )
    cov["active_per_team_game"] = (cov["played"] + cov["dnp"]) / (2 * cov["games"])
    rep.coverage = cov
    # 2011-12 lockout, 2019-20 COVID suspension, 2020-21 shortened, 2012-13: BOS-IND cancelled (Apr 2013).
    expected = {"2011-12": 990, "2012-13": 1229, "2019-20": 1059, "2020-21": 1080}
    for season, n in cov["games"].items():
        exp = expected.get(season, 1230)
        if n != exp:
            rep.warnings.append(f"season {season}: {n} games (expected {exp})")
    s["record_type"] = "player-game box scores" if len(pg) > 10 * pg["game_id"].nunique() else "UNKNOWN"
    s["n_rows"] = int(len(pg))
    s["n_games"] = int(pg["game_id"].nunique())
    s["n_players"] = int(pg["player_id"].nunique())
    s["share_rows_inactive"] = float(np.mean(pg["status"] == STATUS_INACTIVE))
    return rep
