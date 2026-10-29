"""Normalized internal schema.

Raw source: NBA.com ``boxscoretraditionalv3`` rows as published (MIT) in
github.com/NocturneBear/NBA-Data-2010-2024. One raw row = one player listed in one team's box score.

Normalized tables
-----------------
player_games : one row per (game_id, player_id) listed in a box score
games        : one row per game (home/away, final score, overtime periods)
team_games   : one row per (game_id, team_id) with team totals and possessions
"""

from __future__ import annotations

# Raw column -> normalized column for counting statistics.
RAW_STAT_COLUMNS = {
    "fieldGoalsMade": "fgm",
    "fieldGoalsAttempted": "fga",
    "threePointersMade": "fg3m",
    "threePointersAttempted": "fg3a",
    "freeThrowsMade": "ftm",
    "freeThrowsAttempted": "fta",
    "reboundsOffensive": "oreb",
    "reboundsDefensive": "dreb",
    "reboundsTotal": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
    "foulsPersonal": "pf",
    "points": "pts",
    "plusMinusPoints": "plus_minus",
}

RAW_REQUIRED = [
    "season_year", "game_date", "gameId", "matchup", "teamId", "teamTricode",
    "personId", "personName", "position", "comment", "minutes", *RAW_STAT_COLUMNS,
]

COUNT_STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
               "ast", "stl", "blk", "tov", "pf", "pts"]

# Player status derived from the box-score comment.
#   played   : logged minutes
#   dnp      : active (dressed) but did not play ("DNP - ...")  -> outcome is 0 minutes
#   inactive : "DND"/"NWT"/... -> on the official inactive list, excluded from forecasts
STATUS_PLAYED, STATUS_DNP, STATUS_INACTIVE = "played", "dnp", "inactive"

# Columns that describe a game's *outcome*. They must never be visible to the prediction
# path for a target game (see features.leakage_checks).
OUTCOME_COLUMNS = ["minutes", "status_played", *COUNT_STATS, "plus_minus", "starter",
                   "team_margin", "team_minutes"]

PLAYER_GAME_COLUMNS = [
    "game_id", "season", "game_date", "team_id", "team", "opp_team_id", "opp", "is_home",
    "player_id", "player_name", "status", "starter", "minutes", *COUNT_STATS, "plus_minus",
]
