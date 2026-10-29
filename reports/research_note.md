# Research note: do uncertain minutes and shared game factors improve NBA player outcome distributions?

*Generated from saved backtest outputs by `nba-sim report`. Numbers are not typed by hand.*

## 1. Hypothesis

H1: simulating playing time explicitly (Model B) improves out-of-sample player outcome distributions relative to an independent-player model (A). H2: adding shared game-level factors (Model C) improves them further, at least for quantities that depend on dependence between players (team totals, margins, joint teammate outcomes). Null for both: no difference in proper scores.

## 2. Modelling approach

- **A**: independent players and stats; per-game NB means; minutes drawn independently
- **B**: A + simulated minutes (play/no-play, rotation features, team-minute constraint, OT); stats NB given minutes
- **C0**: B + pregame game conditions (pace, opponent, vacated usage, home) in the means only; players independent given minutes (ablation)
- **C**: B + pregame game conditions + shared dependence: team totals drawn jointly (within-team correlation, cross-team correlation, team shot-mix and shooting shocks) and allocated to players; blowout minutes

All models share the same features, shrinkage priors, shooting structure (attempts → binomial makes → derived points, so box scores are internally coherent) and the same recency weighting. C0 isolates C's observable game conditions from its modelled dependence.

## 3. Data and evaluation design

- **Data:** github.com/NocturneBear/NBA-Data-2010-2024 (MIT); underlying data NBA.com boxscoretraditionalv3; dataset id `eedb6060747f242f`; raw files SHA-256 in `outputs/backtest/main/manifest.json`. Regular season only.
- **Evaluation period:** seasons 2021-22, 2022-23, 2023-24; 39 rolling origins (2021-10-19 … 2024-04-09), refit every 14 days; training at each origin = all rows dated before the origin within the current and previous 3 seasons, recency-weighted (half-life 365 days) with intercepts re-estimated on the last 30 days.
- **Sample:** 3,690 games, 93,848 active player-games per model (played or DNP). Players on the inactive list (DND/NWT) are excluded by design — the forecast universe is the active list, assumed known pregame (published before tip-off; no timestamp in the data).
- **Simulation:** 2,000 joint draws per player-game and model; seed 20251029 with per-(window, model, chunk) seed sequences recorded in each prediction file.
- **Freeze protocol:** each window's predictions are written to `predictions/<model>/wNNN.npz` (full discrete predictive PMFs, minutes quantiles, team and joint draws) before any outcome is read; `evaluate` joins outcomes afterwards.

Primary score: CRPS (proper, works for every stat). Log score for primitive counts, Brier for predeclared thresholds, variogram score and team-total CRPS for dependence. Uncertainty: 95% game-level cluster bootstrap.

## 4. Main results

**Key findings (computed):**

- Simulating minutes (B vs A) changes CRPS by minutes -15.4%, pts -8.5%, reb -9.1%, ast -5.6%, pra -11.7%.
  The minutes-CRPS gain is largest for **deep bench (<12)** players (-21.5%) and smallest for **starter (30+)** (-6.6%).
- For observable game conditions (C0 vs B), player CRPS improves for pts, reb, ast, pra; ties for minutes; worsens for none. Team-points CRPS -5.1% (improves), margin CRPS -4.0% (improves).
- For modelled dependence (C vs C0), player CRPS improves for none; ties for minutes, pts, reb, ast; worsens for pra. Team-points CRPS -2.9% (improves), margin CRPS -5.2% (improves).
- For the full shared-factor model (C vs B), player CRPS improves for pts, reb, ast, pra; ties for minutes; worsens for none. Team-points CRPS -7.9% (improves), margin CRPS -9.0% (improves).

| stat | comparison | CRPS diff [95% CI] | rel | verdict |
|---|---|---|---|---|
| minutes | B − A | -0.644 [-0.665, -0.623] | -15.4% | improves |
| minutes | C0 − B | +0.000 [-0.001, +0.001] | +0.0% | tie |
| minutes | C − C0 | +0.001 [-0.001, +0.002] | +0.0% | tie |
| minutes | C − B | +0.001 [-0.001, +0.003] | +0.0% | tie |
| pts | B − A | -0.259 [-0.267, -0.251] | -8.5% | improves |
| pts | C0 − B | -0.006 [-0.008, -0.004] | -0.2% | improves |
| pts | C − C0 | +0.001 [-0.000, +0.002] | +0.0% | tie |
| pts | C − B | -0.005 [-0.007, -0.004] | -0.2% | improves |
| reb | B − A | -0.117 [-0.120, -0.113] | -9.1% | improves |
| reb | C0 − B | -0.004 [-0.005, -0.003] | -0.3% | improves |
| reb | C − C0 | -0.000 [-0.001, +0.000] | -0.0% | tie |
| reb | C − B | -0.004 [-0.005, -0.003] | -0.4% | improves |
| ast | B − A | -0.045 [-0.047, -0.043] | -5.6% | improves |
| ast | C0 − B | -0.007 [-0.008, -0.006] | -0.9% | improves |
| ast | C − C0 | +0.000 [-0.000, +0.000] | +0.0% | tie |
| ast | C − B | -0.007 [-0.008, -0.006] | -0.9% | improves |
| pra | B − A | -0.504 [-0.520, -0.490] | -11.7% | improves |
| pra | C0 − B | -0.014 [-0.017, -0.011] | -0.4% | improves |
| pra | C − C0 | +0.002 [+0.000, +0.003] | +0.0% | worsens |
| pra | C − B | -0.012 [-0.015, -0.010] | -0.3% | improves |

Team level:

| quantity | comparison | diff [95% CI] | rel | verdict |
|---|---|---|---|---|
| team points CRPS | B − A | -4.381 [-4.568, -4.201] | -36.9% | improves |
| margin CRPS | B − A | -5.137 [-5.416, -4.860] | -35.6% | improves |
| team points CRPS | C0 − B | -0.384 [-0.444, -0.323] | -5.1% | improves |
| margin CRPS | C0 − B | -0.373 [-0.478, -0.264] | -4.0% | improves |
| team points CRPS | C − C0 | -0.207 [-0.223, -0.191] | -2.9% | improves |
| margin CRPS | C − C0 | -0.467 [-0.497, -0.438] | -5.2% | improves |
| team points CRPS | C − B | -0.591 [-0.653, -0.525] | -7.9% | improves |
| margin CRPS | C − B | -0.840 [-0.948, -0.726] | -9.0% | improves |
| team points CRPS | C − A | -4.972 [-5.177, -4.776] | -41.9% | improves |
| margin CRPS | C − A | -5.976 [-6.289, -5.673] | -41.4% | improves |
## 5. Calibration and distributional findings

- **minutes** — A: cov80 0.834 vs mass 0.800, PIT-in-80 0.790; B: cov80 0.844 vs mass 0.800, PIT-in-80 0.806; C0: cov80 0.845 vs mass 0.800, PIT-in-80 0.806; C: cov80 0.845 vs mass 0.800, PIT-in-80 0.806
- **pts** — A: cov80 0.858 vs mass 0.866, PIT-in-80 0.766; B: cov80 0.871 vs mass 0.869, PIT-in-80 0.797; C0: cov80 0.871 vs mass 0.869, PIT-in-80 0.798; C: cov80 0.866 vs mass 0.869, PIT-in-80 0.793
- **pra** — A: cov80 0.716 vs mass 0.846, PIT-in-80 0.649; B: cov80 0.851 vs mass 0.855, PIT-in-80 0.791; C0: cov80 0.852 vs mass 0.855, PIT-in-80 0.792; C: cov80 0.850 vs mass 0.855, PIT-in-80 0.791
- **Team points** — A: realized RMSE 20.8 vs predicted SD 23.9, 80% coverage 0.86; B: realized RMSE 13.0 vs predicted SD 17.2, 80% coverage 0.91; C0: realized RMSE 12.2 vs predicted SD 17.2, 80% coverage 0.94; C: realized RMSE 12.2 vs predicted SD 14.0, 80% coverage 0.87
- **Margins** — A: predicted SD 33.8, PIT in [.1,.9] 0.91; B: predicted SD 23.9, PIT in [.1,.9] 0.94; C0: predicted SD 23.9, PIT in [.1,.9] 0.95; C: predicted SD 18.9, PIT in [.1,.9] 0.90

![PIT](figures/pit_histograms.png)

## 6. Ablation results

![Ablation by season](figures/ablation_by_season.png)

Points CRPS by season — B vs A: 2021-22 improves, 2022-23 improves, 2023-24 improves; C vs B: 2021-22 tie, 2022-23 improves, 2023-24 improves.

![Ablation by role](figures/ablation_by_role.png)

## 7. Limitations and possible explanations

- The active list is treated as pregame information (published before tip-off, but untimestamped here). Without it, every model would face extra availability uncertainty, and B's minutes model, which redistributes minutes over the active list, would likely lose some of its advantage.
- Starting lineups are not used. They are a strong pregame minutes signal, available ~30 minutes before tip, so B/C understate what a production system could do.
- C's dependence layer acts on team totals and two latent shooting shocks; it does not model possessions explicitly, and team-points distributions remain wider than realized for every model (§5).
- Most of C's dependence leaves player *marginals* nearly unchanged by design (it re-allocates variance), so marginal scores such as player CRPS are not expected to move much; its effect shows up in team totals, margins and joint scores.
- Three seasons, one league, regular season only; 14-day refits.

## 8. Final verdict

Rules (written into the report code before the final backtest was run, applied mechanically): a component shows *evidence of improvement* if every key player metric (CRPS of minutes, PTS, REB, AST, PRA) — plus, for minutes, points CRPS in every season, and for shared factors, team-total, margin and variogram scores — improves with a 95% CI excluding zero; *mixed* if some improve and not all worsen; *no meaningful improvement* otherwise.

- **Uncertain playing time (B vs A): Evidence of improvement.**
- **Shared game-level factors (C vs B): Mixed or segment-specific improvement.**

**Overall verdict on the research question: Mixed or segment-specific improvement.**

Statistical accuracy on historical data is not evidence of betting profitability; no market prices were used.
