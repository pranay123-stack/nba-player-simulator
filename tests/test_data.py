"""Ingestion, normalization and validation on the real fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sim.data.ingest import (
    DataNotFoundError,
    derive_status,
    ingest,
    load_dataset,
    parse_minutes,
    raw_files,
)
from nba_sim.data.schema import STATUS_DNP, STATUS_INACTIVE, STATUS_PLAYED
from nba_sim.data.validate import validate


# ----------------------------------------------------------------------------- parsing

def test_parse_minutes_mm_ss():
    # hand-written values
    s = pd.Series(["12:30", "0:00", None, " 5:07 ", "48:00"])
    out = parse_minutes(s)
    assert out.iloc[0] == pytest.approx(12.5)
    assert out.iloc[1] == 0.0
    assert np.isnan(out.iloc[2]), "missing minutes must stay NaN, never silently zero"
    assert out.iloc[3] == pytest.approx(5 + 7 / 60)
    assert out.iloc[4] == pytest.approx(48.0)


@pytest.mark.parametrize("garbage", ["abc", "12.5", "12:", ":30", "1:2:3"])
def test_parse_minutes_rejects_garbage(garbage):
    # hand-written values
    with pytest.raises(ValueError, match="unparseable minutes"):
        parse_minutes(pd.Series(["10:00", garbage]))


def test_derive_status_rules():
    # hand-written values
    comment = pd.Series(["DNP - Coach's Decision", "DND - Injury/Illness", "NWT - Personal", None, None, "  "])
    minutes = pd.Series([np.nan, np.nan, np.nan, 0.0, 31.5, 12.0])
    st = derive_status(comment, minutes)
    assert st.tolist() == [STATUS_DNP, STATUS_INACTIVE, STATUS_INACTIVE, STATUS_PLAYED, STATUS_PLAYED,
                           STATUS_PLAYED]


def test_derive_status_no_comment_no_minutes_is_inactive():
    # hand-written values
    st = derive_status(pd.Series([None]), pd.Series([np.nan]))
    assert st.iloc[0] == STATUS_INACTIVE


# ----------------------------------------------------------------------------- normalized fixture

def test_fixture_manifest(ds):
    m = ds.manifest
    assert m["seasons"] == ["2022-23"]
    assert m["date_min"] == "2022-10-18"
    assert m["date_max"] == "2023-01-15"
    assert m["n_games"] == ds.games["game_id"].nunique() > 600
    assert len(m["dataset_id"]) == 16


def test_status_minutes_mapping_on_fixture(ds):
    pg = ds.player_games
    c = pg["comment"].fillna("").str.strip().str.upper()
    dnp = pg[pg["status"] == STATUS_DNP]
    assert len(dnp) > 0
    assert (dnp["minutes"] == 0.0).all(), "DNP rows are active with a 0-minute outcome"
    assert dnp["comment"].str.upper().str.startswith("DNP").all()
    inactive = pg[pg["status"] == STATUS_INACTIVE]
    assert len(inactive) > 0
    assert inactive["minutes"].isna().all()
    assert inactive["pts"].isna().all()
    assert (c[c.str.startswith(("DND", "NWT"))].index.isin(inactive.index)).all()
    played = pg[pg["status"] == STATUS_PLAYED]
    assert played["minutes"].notna().all()
    assert (played["comment"].fillna("").str.strip() == "").all()


def test_no_duplicate_player_games(ds):
    assert not ds.player_games.duplicated(["game_id", "player_id"]).any()


def test_every_game_two_teams_one_home(ds):
    pg = ds.player_games
    assert (pg.groupby("game_id")["team_id"].nunique() == 2).all()
    homes = pg.drop_duplicates(["game_id", "team_id"]).groupby("game_id")["is_home"].sum()
    assert (homes == 1).all()
    assert pg["opp_team_id"].notna().all()
    assert (pg["opp_team_id"] != pg["team_id"]).all()
    g = ds.games
    assert g["game_id"].is_unique
    assert (g["home_team_id"] != g["away_team_id"]).all()


def test_team_minutes_match_regulation_plus_overtime(ds):
    tg = ds.team_games
    resid = tg["team_minutes"] - (240 + 25 * tg["n_ot"])
    assert resid.abs().max() < 0.25
    assert (tg["n_ot"] > 0).any(), "fixture should contain overtime games"
    # Both teams of a game agree on the overtime count.
    assert (tg.groupby("game_id")["n_ot"].nunique() == 1).all()


def test_validate_passes_on_fixture(ds):
    rep = validate(ds.player_games, ds.team_games)
    assert rep.ok, rep.errors
    assert rep.summary["duplicate_player_game_rows"] == 0
    assert rep.summary["pts_identity_violations"] == 0
    assert rep.summary["record_type"] == "player-game box scores"
    # Partial season: the only expected warning class is the game count.
    assert all("games (expected" in w for w in rep.warnings)


def test_validate_flags_duplicates(ds):
    pg = pd.concat([ds.player_games, ds.player_games.iloc[:3]], ignore_index=True)
    rep = validate(pg, ds.team_games)
    assert not rep.ok
    assert rep.summary["duplicate_player_game_rows"] == 3


# ----------------------------------------------------------------------------- missing data

def test_missing_raw_dir_raises_no_fallback(tmp_path):
    with pytest.raises(DataNotFoundError):
        raw_files(tmp_path)
    with pytest.raises(DataNotFoundError):
        ingest(tmp_path)
    with pytest.raises(DataNotFoundError):
        ingest(tmp_path / "does_not_exist")
    assert issubclass(DataNotFoundError, FileNotFoundError)


def test_missing_processed_dir_raises(tmp_path):
    with pytest.raises(DataNotFoundError):
        load_dataset(tmp_path)


# ----------------------------------------------------------------------------- full dataset

@pytest.mark.slow
def test_full_raw_dataset_validates(raw_dir):
    ds = ingest(raw_dir)
    rep = validate(ds.player_games, ds.team_games)
    assert rep.ok, rep.errors
    assert not ds.player_games.duplicated(["game_id", "player_id"]).any()
    assert len(ds.manifest["seasons"]) >= 10
