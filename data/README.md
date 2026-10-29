# Data

## Source

Player-game box scores for every NBA regular-season game from 2010-11 through 2023-24
(16,658 games, 424,478 player rows), as published in
[github.com/NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024)
under the MIT licence. The rows are NBA.com `boxscoretraditionalv3` box scores (one row per player
listed in a team's box score, including DNP / DND / NWT entries with their comment).

The pipeline is pinned to upstream commit `a5f108b5b1f08074d78b9e8e901926a9ce4c06c5` (2024-07-22).
The source has no 2024-25 season, so the default backtest covers 2021-22 to 2023-24.

### Why this source

| candidate | status on 2026-10-29 |
|---|---|
| `stats.nba.com` via `nba_api` | timed out from this machine even with browser headers (NBA blocks many IPs); not usable here |
| Kaggle `wyattowalsh/basketball` | needs a Kaggle API token (none configured) |
| NocturneBear/NBA-Data-2010-2024 | public, MIT, real player-game rows with minutes, full shooting lines, starter flag and DNP reasons — **used** |

### Licensing caveat

The repository is MIT-licensed, but the underlying statistics come from NBA.com, whose terms
restrict commercial use of its data. This project is research/demonstration use only. Check
NBA.com's terms before any commercial use.

## Download

```bash
mkdir -p data/raw && cd data/raw
B=https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/a5f108b5b1f08074d78b9e8e901926a9ce4c06c5
for f in regular_season_box_scores_2010_2024_part_1.csv regular_season_box_scores_2010_2024_part_2.csv \
         regular_season_box_scores_2010_2024_part_3.csv play_off_box_scores_2010_2024.csv LICENSE README.md; do
  curl -sSfL -o "$f" "$B/$f"
done
sha256sum *.csv
```

Expected SHA-256 values (also recorded in `data/processed/dataset_manifest.json` and in every run manifest):

```
ca9636d08c577e3d752f69e32f90b173f4c3374c15f4dc8494b655049fdfc557  regular_season_box_scores_2010_2024_part_1.csv
5643af8a9f15ce3540dced759375ca79e77c71644e23270136f23b7a57fc182d  regular_season_box_scores_2010_2024_part_2.csv
65a01fee3df865a272e578cf4b365dce7dbc62eef2ce6f15f0844cf6694ef375  regular_season_box_scores_2010_2024_part_3.csv
d60906641c67026c167c1f95b8ccc1e61a8c3e4d0525eaa1f4be4329cd1ea498  play_off_box_scores_2010_2024.csv
```

Then build the normalized tables and features:

```bash
nba-sim inspect-data --data-path data/raw     # validation only -> outputs/data_validation.json
nba-sim build-dataset --data-path data/raw    # -> data/processed/*.parquet
```

If the files are missing, every command fails with an explicit error. There is no synthetic
fallback.

## Normalized schema (`data/processed/`)

| table | grain | key columns |
|---|---|---|
| `player_games.parquet` | one row per player listed in a box score | `game_id, player_id, team_id, opp_team_id, is_home, status, starter, minutes`, counting stats |
| `team_games.parquet` | one row per team per game | team totals, `team_minutes`, `n_ot`, `poss`, `pace48`, `margin` |
| `games.parquet` | one row per game | home/away ids and points, `n_ot` |
| `features.parquet` | one row per *active* player-game | pregame features (see `src/nba_sim/features/build.py::FEATURE_DOCS`) + outcomes for training/evaluation |
| `dataset_manifest.json` | — | source, file checksums, `dataset_id`, coverage |

Status derivation (`data/ingest.py::derive_status`):

* `played`: minutes logged and no comment. This includes 6 sub-second "0:00" appearances credited with a stat.
* `dnp`: comment starts with `DNP`. The player was active but never entered, so this is a genuine 0-minute outcome.
* `inactive`: `DND`, `NWT`, suspensions and the like. These players are excluded from forecasting. Missing statistics are never filled with zero; inactive rows keep `NaN`.

## Validation results (full dataset)

`nba-sim inspect-data` checks duplicates, team/game identifier consistency, minutes plausibility,
team-minute accounting (240 + 25 per overtime), box-score identities (PTS = 2·FGM + 3PM + FTM,
FGM ≤ FGA, OREB + DREB = REB), negative counts, stats on DNP rows, missingness and season coverage.
On the pinned files there are **0 errors** and 2 warnings:

* 6 sub-second appearances logged as 0:00 and credited with a stat.
* 2012-13 has 1,229 games. This is correct: BOS–IND was cancelled in April 2013.

Seasons with fewer than 1,230 games are real: the 2011-12 lockout (990), the 2019-20 COVID
suspension (1,059) and 2020-21 (1,080). Inactive designations are only partly listed in the box
scores (about 2.6% of rows), which is why the active list rather than the roster defines the forecast universe.
