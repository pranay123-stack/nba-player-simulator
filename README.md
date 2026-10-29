# NBA Player Outcome Simulation & Quantitative Validation

This engine forecasts **full probability distributions** for NBA player box scores (minutes, points,
rebounds, assists, steals, blocks, turnovers, FGM/FGA, 3PM/3PA, FTM/FTA and combos such as PRA)
by simulating whole games jointly. It then tests, chronologically and out of sample, whether
modelling uncertain minutes and shared game factors makes those distributions better.

| deliverable | file |
|---|---|
| 1. Simulation engine | `src/nba_sim/` (CLI `nba-sim`, Streamlit app `app/streamlit_app.py`) |
| 2. Validation report | [`reports/validation_report.md`](reports/validation_report.md) |
| 3. Research note (ablation and verdict) | [`reports/research_note.md`](reports/research_note.md) |
| 4. Historical matchup demo (10,000 sims per model) | [`reports/historical_matchup_demo.md`](reports/historical_matchup_demo.md) |

## Headline results (rolling-origin backtest, 2021-22 to 2023-24)

The backtest covers 3,690 games and 93,848 active player-games per model, with 39 rolling origins and
2,000 joint draws per player-game. Every number below is copied from the generated
[`reports/research_note.md`](reports/research_note.md). CRPS change uses a 95% game-level cluster bootstrap; negative means better.

| comparison | minutes | PTS | REB | AST | PRA | team points | margin |
|---|---|---|---|---|---|---|---|
| **B vs A** (simulated minutes) | −15.4% | −8.5% | −9.1% | −5.6% | −11.7% | −36.9% | −35.6% |
| **C0 vs B** (pregame game conditions) | tie | −0.2% | −0.3% | −0.9% | −0.4% | −5.1% | −4.0% |
| **C vs C0** (modelled dependence) | tie | tie | tie | tie | +0.04% (worse) | −2.9% | −5.2% |
| **C vs B** (all shared factors) | tie | −0.2% | −0.4% | −0.9% | −0.3% | −7.9% | −9.0% |

* **Uncertain playing time: evidence of improvement.** Every key metric improves, and points improve in every season. The gain is largest for deep-bench players (minutes CRPS −21.5%) and smallest for starters (−6.6%). Model A's PRA intervals are badly too narrow: 0.72 coverage against 0.85 predictive mass, because it treats a player's stats as independent.
* **Shared game factors: mixed, segment-specific improvement.** Player marginals barely move (by design, the dependence layer re-allocates variance). The gains are at the team level: Model C's team-points predicted SD is 14.0 against a realized RMSE of 12.2, compared with 17.2 for B, and margin CRPS is 9% better. The points variogram score ties B, and is slightly worse than C0.
* **Overall verdict: mixed or segment-specific improvement.** The verdict comes from rules coded before the final run. All models still produce team-total distributions somewhat wider than realized.
* **Historical demo (BOS at DEN, 2024-03-07):** exactly 10,000 draws per model. The predictions were frozen and hashed before the box score was loaded. See [`reports/historical_matchup_demo.md`](reports/historical_matchup_demo.md).


## Data

Real NBA.com box scores for every regular-season game from 2010-11 to 2023-24 (16,658 games), from
the MIT-licensed mirror [NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024),
pinned to a commit and verified with SHA-256 checksums. `stats.nba.com` was unreachable from the
build machine and Kaggle needs credentials. See [`data/README.md`](data/README.md) for download
steps, the schema, validation results and the licensing caveat. There is **no synthetic data
anywhere**: tests run on a committed slice of the real data.

## Models

| model | what it adds |
|---|---|
| **A** independent baseline | Per-game NB2 means from shrunk, decayed player averages. Stats, minutes and players are all independent. |
| **B** uncertain playing time | Minutes are simulated first. A logistic model gives P(enters game), a linear model plus empirical residuals gives minutes if he plays, and overtime is drawn. Minutes are water-filled so each team sums to exactly 240 + 25·OT, with no player above 48 + 5·OT. Stats are NB given minutes. |
| **C0** (ablation) | B plus observable pregame game conditions in the means: expected pace, opponent allowed rates, vacated usage given the active list, and home court. |
| **C** shared game factors | C0 plus modelled dependence. Team totals of FGA/FTA/REB/STL/BLK/TOV are drawn jointly, with a within-team correlation matrix and per-stat cross-team correlation. AST is conditioned on team makes. Team shot-mix and shooting shocks act on the logit scale. Totals are allocated to players by multinomial with share shocks, and minutes are reallocated in blowouts. All parameters are estimated from training residuals at each refit. |

All four models share the same coherent box-score structure:

* Primitive counts are drawn first.
* Then 3PA ~ Bin(FGA, π₃), OREB ~ Bin(REB, π_o), and makes ~ Bin(attempts, %).
* Then PTS = 2·FG2M + 3·FG3M + FTM.
* PRA, PR, PA and RA are summed from the same draw, never sampled separately.

## Leakage controls

* Features are decayed sums **shifted one game**. Team and league features are shifted one game or one day. Shrinkage priors and all model parameters are fit at each origin from training rows only.
* `perturbation_check` scrambles every outcome on or after a date and asserts the features on that date are bit-identical. A planted leaky feature must be caught (`tests/test_leakage.py`).
* `pregame_view` strips all outcome columns before rows reach `simulate`. `fit` asserts that training dates are earlier than the origin.
* The demo and `simulate` rebuild the target game's rows with **empty outcomes** from data cut strictly before the game date, and assert the game id is not in training. The demo also SHA-256-hashes the frozen predictions before loading the box score.
* Backtest predictions are written to disk per rolling origin (stage 1) before outcomes are joined (stage 2).
* **Information assumptions** (disclosed in every report):
  * The game's *active list* is treated as known pregame. It is published before tip-off but has no timestamp in this data.
  * Starting lineups, injury reports and betting lines are **not** used.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,app]"
# download the data (see data/README.md), then:
.venv/bin/nba-sim inspect-data --data-path data/raw
.venv/bin/nba-sim build-dataset --data-path data/raw
```

Python 3.11+ is required (tested on 3.12.3, pandas 3.0, numpy 2.5, scipy 1.18). Configuration is
in `configs/default.yaml`; every command accepts `--config`.

## Commands (all run as shown on the build machine)

```bash
nba-sim inspect-data --data-path data/raw                     # validation report -> outputs/data_validation.json
nba-sim build-dataset --data-path data/raw                    # data/processed/*.parquet
nba-sim backtest --start-season 2021-22 --end-season 2023-24  # outputs/backtest/main/ (~45 min, 4 models x 2,000 draws)
nba-sim report --output reports/                              # validation_report.md, research_note.md, figures/
nba-sim demo-historical --n-simulations 10000 --seed 42       # reports/historical_matchup_demo.md, outputs/demo/<game_id>/
nba-sim simulate --home BOS --away MIA --date 2024-04-14 --n-simulations 10000 --seed 42
nba-sim evaluate --run-name main                              # re-score frozen predictions
.venv/bin/python -m pytest                                    # test suite (~20 s)
.venv/bin/streamlit run app/streamlit_app.py                  # interactive viewer
```

`demo-historical` without `--game-id` selects the game with a **pregame-only rule**: two strong
teams late in the final season, and every core player has 100 or more prior games. It never looks at the outcome.
`simulate` works for any date. It uses only games before that date, and the roster is every player
active for the team in its last 10 games (exclude players with `--exclude <ids>`).

## Outputs and reproducibility

```
outputs/backtest/main/
  manifest.json            config, dataset checksums, code fingerprint, windows, seeds, score files
  predictions/<model>/wNNN.npz|.rows.parquet|.meta.json   frozen predictive distributions per origin
  models/<model>/wNNN.json fitted parameters per origin
  scores/*.parquet         player, threshold, team and joint scores
  comparisons.csv, summary_*.csv
outputs/demo/<game_id>/    frozen predictions (+ hashes), comparison.csv, figures, manifest.json
outputs/simulations/<date>_<away>_at_<home>_<model>/   player_distributions.csv, summary.json
```

For the reported run, the prediction files carry code hash `018abcd5…`. `manifest.json` carries
`99f449b3…` because `reporting/validation_report.py` was edited after the run started; model,
simulation and evaluation code were unchanged.

Each prediction file records its seed sequence, its training window and the SHA-256
fingerprint of the source code that produced it. Re-running a command with the same config, data
and code reproduces the files bit-for-bit (`tests/test_end_to_end.py` checks this).

## Repository layout

```
configs/default.yaml       all tunables (seeds, simulation counts, refit cadence, thresholds)
data/README.md             acquisition, schema, validation
src/nba_sim/data           ingest, schema, validate
src/nba_sim/features       build (pregame features + FEATURE_DOCS), leakage_checks
src/nba_sim/models         common (priors, GLMs, NB), baseline (A), minutes (B), shared_factors (C)
src/nba_sim/simulation     engine, constraints (team minutes), predictive (frozen PMFs), pregame (target rows)
src/nba_sim/evaluation     backtest (rolling origin), metrics (CRPS, log score, PIT, Brier, variogram)
src/nba_sim/reporting      validation_report, matchup_demo, demo_report, manifest
tests/                     test suite on a real-data fixture
app/streamlit_app.py       viewer
```

## Limitations

* The active list is assumed known pregame, and late scratches are not modelled. Starting lineups are not used, so a production system with lineup feeds would do better on minutes.
* League-wide scoring drifts within seasons faster than the recency weighting can follow. Bias by season is reported.
* Team-total distributions remain somewhat wider than realized, even for Model C. Possessions are not modelled explicitly.
* Validation is regular season only, 2021-22 to 2023-24, with refits every 14 days.
* This is a forecasting-quality study. Nothing here measures or implies betting profitability.
