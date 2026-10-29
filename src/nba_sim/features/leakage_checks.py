"""Guards against look-ahead leakage.

Two mechanisms:

* ``pregame_view`` strips every outcome column from the target rows before they reach a model's
  ``simulate`` method, so a model physically cannot read the result it is forecasting.
* ``perturbation_check`` recomputes features after scrambling the outcomes of a target date and
  every later date; features on the target date must be bit-for-bit unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sim.data.schema import COUNT_STATS, OUTCOME_COLUMNS
from nba_sim.features.build import build_features


class LeakageError(AssertionError):
    pass


def pregame_view(rows: pd.DataFrame) -> pd.DataFrame:
    """Copy of target rows with all outcome columns removed."""
    drop = [c for c in OUTCOME_COLUMNS + ["status", "comment"] if c in rows.columns]
    return rows.drop(columns=drop).copy()


def assert_training_before(train: pd.DataFrame, origin: pd.Timestamp) -> None:
    if len(train) and train["game_date"].max() >= origin:
        raise LeakageError(f"training rows reach {train['game_date'].max().date()} >= origin {origin.date()}")


def assert_disjoint_games(train: pd.DataFrame, target_game_ids) -> None:
    overlap = set(train["game_id"].unique()) & set(np.atleast_1d(target_game_ids))
    if overlap:
        raise LeakageError(f"target games present in training data: {sorted(overlap)[:5]}")


def perturbation_check(pg: pd.DataFrame, tg: pd.DataFrame, cfg: dict, target_date,
                       seed: int = 0) -> pd.DataFrame:
    """Scramble outcomes on/after ``target_date``; return the max abs feature change on that date.

    The active list (who dressed) is pregame information by assumption and is left untouched;
    minutes, counting stats, starter flags and team totals are all scrambled.
    """
    target_date = pd.Timestamp(target_date)
    base = build_features(pg, tg, cfg)
    rng = np.random.default_rng(seed)
    pg2, tg2 = pg.copy(), tg.copy()
    later = pg2["game_date"] >= target_date
    active = later & pg2["minutes"].notna()
    pg2.loc[active, "minutes"] = rng.uniform(0, 48, active.sum())
    pg2.loc[later, "starter"] = rng.random(later.sum()) < 0.5
    for c in COUNT_STATS:
        pg2.loc[active, c] = rng.integers(0, 40, active.sum()).astype(float)
    tl = tg2["game_date"] >= target_date
    for c in ["pace48", "poss", "opp_poss", "pts", "team_minutes", *COUNT_STATS]:
        if c in tg2:
            tg2.loc[tl, c] = tg2.loc[tl, c].to_numpy() * rng.uniform(0.5, 1.5, tl.sum())
    pert = build_features(pg2, tg2, cfg)

    key = ["game_id", "player_id"]
    a = base[base["game_date"] == target_date].set_index(key).sort_index()
    b = pert[pert["game_date"] == target_date].set_index(key).sort_index()
    outcome_like = set(OUTCOME_COLUMNS) | {"status", "comment", "min2"}
    cols = [c for c in a.columns if c not in outcome_like and pd.api.types.is_numeric_dtype(a[c])]
    diff = (a[cols].astype(float) - b[cols].astype(float)).abs().max().fillna(0.0)
    return diff[diff > 0].sort_values(ascending=False).to_frame("max_abs_change")
