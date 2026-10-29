"""Raw box-score ingestion into the normalized schema."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nba_sim.data.schema import (
    COUNT_STATS,
    PLAYER_GAME_COLUMNS,
    RAW_REQUIRED,
    RAW_STAT_COLUMNS,
    STATUS_DNP,
    STATUS_INACTIVE,
    STATUS_PLAYED,
)

log = logging.getLogger(__name__)

REGULAR_SEASON_GLOB = "regular_season_box_scores_*.csv"
PLAYOFF_GLOB = "play_off_box_scores_*.csv"


class DataNotFoundError(FileNotFoundError):
    """Raised when the raw dataset is missing. There is deliberately no synthetic fallback."""


@dataclass
class Dataset:
    player_games: pd.DataFrame
    games: pd.DataFrame
    team_games: pd.DataFrame
    manifest: dict


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def raw_files(raw_dir: Path, include_playoffs: bool = False) -> list[Path]:
    files = sorted(raw_dir.glob(REGULAR_SEASON_GLOB))
    if include_playoffs:
        files += sorted(raw_dir.glob(PLAYOFF_GLOB))
    if not files:
        raise DataNotFoundError(
            f"no raw box-score CSVs matching '{REGULAR_SEASON_GLOB}' in {raw_dir}. "
            "See data/README.md for download instructions."
        )
    return files


def read_raw(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"minutes": str, "comment": str, "position": str, "jerseyNum": str})
        missing = [c for c in RAW_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name}: missing required columns {missing}")
        df["source_file"] = f.name
        df["season_type"] = "playoffs" if "play_off" in f.name else "regular"
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    log.info("read %d raw rows from %d files", len(raw), len(files))
    return raw


def parse_minutes(s: pd.Series) -> pd.Series:
    """'MM:SS' -> float minutes. Missing stays NaN (never silently zero)."""
    parts = s.str.extract(r"^\s*(\d+):(\d{1,2})\s*$")
    out = parts[0].astype(float) + parts[1].astype(float) / 60.0
    bad = s.notna() & out.isna()
    if bad.any():
        raise ValueError(f"unparseable minutes values: {s[bad].unique()[:5].tolist()}")
    return out


def derive_status(comment: pd.Series, minutes: pd.Series) -> pd.Series:
    """Map box-score comments to played / dnp / inactive.

    'DNP ...' means the player was active (dressed) but did not enter the game; this is a genuine
    zero-minute outcome the minutes model has to forecast. 'DND' (did not dress), 'NWT' (not with
    team), suspensions etc. are inactive designations and are excluded from forecasting.
    """
    c = comment.fillna("").str.strip().str.upper()
    status = pd.Series(STATUS_INACTIVE, index=comment.index, dtype=object)
    status[c.str.startswith("DNP")] = STATUS_DNP
    played = minutes.notna() & (c == "")
    # Rows logging 0:00 with no comment are sub-second appearances (a few carry a stat), so they
    # stay 'played' with 0.0 minutes rather than being relabelled as DNP.
    status[played] = STATUS_PLAYED
    return status


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "game_id": raw["gameId"].astype(np.int64),
            "season": raw["season_year"].astype(str),
            "season_type": raw["season_type"].astype(str),
            "game_date": pd.to_datetime(raw["game_date"]),
            "team_id": raw["teamId"].astype(np.int64),
            "team": raw["teamTricode"].astype(str),
            "player_id": raw["personId"].astype(np.int64),
            "player_name": raw["personName"].astype(str),
            "matchup": raw["matchup"].astype(str),
            "comment": raw["comment"],
        }
    )
    minutes = parse_minutes(raw["minutes"])
    df["status"] = derive_status(raw["comment"], minutes)
    df["starter"] = raw["position"].notna() & (raw["position"].astype(str).str.strip() != "")
    df["minutes"] = np.where(df["status"] == STATUS_PLAYED, minutes,
                             np.where(df["status"] == STATUS_DNP, 0.0, np.nan))
    for raw_col, col in RAW_STAT_COLUMNS.items():
        vals = raw[raw_col].astype(float)
        df[col] = np.where(df["status"] == STATUS_INACTIVE, np.nan, vals)
    df["is_home"] = df["matchup"].str.contains(" vs. ", regex=False)
    # Opponent = the other team in the same game.
    teams = df[["game_id", "team_id", "team"]].drop_duplicates()
    opp = teams.merge(teams, on="game_id", suffixes=("", "_o"))
    opp = opp[opp["team_id"] != opp["team_id_o"]].rename(columns={"team_id_o": "opp_team_id", "team_o": "opp"})
    df = df.merge(opp[["game_id", "team_id", "opp_team_id", "opp"]], on=["game_id", "team_id"], how="left")
    df = df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)
    return df


def build_team_games(pg: pd.DataFrame, max_reg: float = 48.0, ot_len: float = 5.0) -> pd.DataFrame:
    active = pg[pg["status"] != STATUS_INACTIVE]
    agg = active.groupby(["game_id", "team_id"], as_index=False).agg(
        season=("season", "first"), season_type=("season_type", "first"), game_date=("game_date", "first"),
        team=("team", "first"), opp_team_id=("opp_team_id", "first"), opp=("opp", "first"),
        is_home=("is_home", "first"), team_minutes=("minutes", "sum"), n_active=("player_id", "size"),
        n_played=("status", lambda s: int((s == STATUS_PLAYED).sum())),
        **{c: (c, "sum") for c in COUNT_STATS},
    )
    reg_total = 5 * max_reg
    agg["n_ot"] = np.round((agg["team_minutes"] - reg_total) / (5 * ot_len)).clip(lower=0).astype(int)
    # Possessions (standard box-score estimator).
    agg["poss"] = agg["fga"] + 0.44 * agg["fta"] - agg["oreb"] + agg["tov"]
    opp = agg[["game_id", "team_id", "pts", "poss"]].rename(
        columns={"team_id": "opp_team_id", "pts": "opp_pts", "poss": "opp_poss"})
    agg = agg.merge(opp, on=["game_id", "opp_team_id"], how="left")
    game_minutes = max_reg + ot_len * agg["n_ot"]
    agg["pace48"] = 0.5 * (agg["poss"] + agg["opp_poss"]) * max_reg / game_minutes
    agg["margin"] = agg["pts"] - agg["opp_pts"]
    return agg.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)


def build_games(tg: pd.DataFrame) -> pd.DataFrame:
    home = tg[tg["is_home"]].set_index("game_id")
    away = tg[~tg["is_home"]].set_index("game_id")
    g = pd.DataFrame(
        {
            "season": home["season"], "season_type": home["season_type"], "game_date": home["game_date"],
            "home_team_id": home["team_id"], "home": home["team"],
            "away_team_id": away["team_id"], "away": away["team"],
            "home_pts": home["pts"], "away_pts": away["pts"], "n_ot": home["n_ot"],
        }
    ).reset_index()
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def ingest(raw_dir: Path | None, include_playoffs: bool = False, files: list[Path] | None = None) -> Dataset:
    """Ingest every raw CSV in ``raw_dir`` (or an explicit ``files`` list, e.g. a gzipped fixture)."""
    files = [Path(f) for f in files] if files else raw_files(raw_dir, include_playoffs)
    raw = read_raw(files)
    pg = normalize(raw)
    tg = build_team_games(pg)
    games = build_games(tg)
    manifest = {
        "source": "github.com/NocturneBear/NBA-Data-2010-2024 (MIT); underlying data NBA.com boxscoretraditionalv3",
        "files": {f.name: sha256_file(f) for f in files},
        "n_player_rows": int(len(pg)),
        "n_games": int(games["game_id"].nunique()),
        "seasons": sorted(pg["season"].unique().tolist()),
        "date_min": str(pg["game_date"].min().date()),
        "date_max": str(pg["game_date"].max().date()),
    }
    manifest["dataset_id"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()[:16]
    return Dataset(pg[PLAYER_GAME_COLUMNS + ["season_type", "comment"]], games, tg, manifest)


def save_dataset(ds: Dataset, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "player_games": out_dir / "player_games.parquet",
        "games": out_dir / "games.parquet",
        "team_games": out_dir / "team_games.parquet",
        "manifest": out_dir / "dataset_manifest.json",
    }
    ds.player_games.to_parquet(paths["player_games"], index=False)
    ds.games.to_parquet(paths["games"], index=False)
    ds.team_games.to_parquet(paths["team_games"], index=False)
    paths["manifest"].write_text(json.dumps(ds.manifest, indent=2))
    return paths


def load_dataset(processed_dir: Path) -> Dataset:
    pgp = processed_dir / "player_games.parquet"
    if not pgp.exists():
        raise DataNotFoundError(
            f"processed dataset not found in {processed_dir}; run `nba-sim build-dataset` first"
        )
    return Dataset(
        pd.read_parquet(pgp),
        pd.read_parquet(processed_dir / "games.parquet"),
        pd.read_parquet(processed_dir / "team_games.parquet"),
        json.loads((processed_dir / "dataset_manifest.json").read_text()),
    )
