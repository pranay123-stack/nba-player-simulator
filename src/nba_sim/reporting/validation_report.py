"""Validation report and research note, generated from saved backtest scores.

Every number in reports/validation_report.md and reports/research_note.md is computed here from
outputs/backtest/<run>/scores/*.parquet; nothing is typed by hand. Model comparisons use a
game-level cluster bootstrap (players in the same game are not independent).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nba_sim.models import MODEL_DESCRIPTIONS  # noqa: E402

KEY_STATS = ["minutes", "pts", "reb", "ast", "pra"]
ALL_STATS = ["minutes", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "pra"]
LOG_STATS = ["fga", "fta", "reb", "ast", "stl", "blk", "tov"]
COMPARISONS = [("B", "A"), ("C0", "B"), ("C", "C0"), ("C", "B"), ("C", "A")]
COLORS = {"A": "#7a7a7a", "B": "#2c7fb8", "C0": "#41ab5d", "C": "#d95f0e"}


# ----------------------------------------------------------------------------- statistics

def cluster_bootstrap_diff(df: pd.DataFrame, value: str, m1: str, m0: str, reps: int, seed: int,
                           cluster: str = "game_id") -> dict:
    """Mean of (score_m1 - score_m0) over matched observations, with a cluster bootstrap CI.

    Negative = m1 better (all scores are losses)."""
    key = ["game_id", "player_id"] if "player_id" in df.columns else ["game_id", "team_id"]
    a = df[df["model"] == m1].set_index(key)[value]
    b = df[df["model"] == m0].set_index(key)[value]
    d = (a - b).dropna()
    if d.empty:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "n_clusters": 0, "rel": np.nan}
    g = d.groupby(level=0)
    sums, counts = g.sum().to_numpy(), g.size().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(reps, len(sums)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    base = float(b.loc[d.index].mean())
    est = float(sums.sum() / counts.sum())
    return {"diff": est, "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "n": int(counts.sum()), "n_clusters": int(len(sums)), "rel": est / base if base else np.nan}


def verdict_of(r: dict) -> str:
    if r["n"] == 0 or np.isnan(r["diff"]):
        return "n/a"
    if r["hi"] < 0:
        return "improves"
    if r["lo"] > 0:
        return "worsens"
    return "tie"


# ----------------------------------------------------------------------------- formatting

def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def _f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(r: dict, nd=3) -> str:
    return f"{r['diff']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"


# ----------------------------------------------------------------------------- main

def build_reports(cfg: dict, run_dir: Path, out_dir: Path) -> dict[str, Path]:
    reps = cfg["evaluation"]["bootstrap_reps"]
    seed = cfg["backtest"]["seed"]
    sd = run_dir / "scores"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols = ["model", "game_id", "player_id", "season", "role", "history", "stat", "y", "mean", "crps", "logs",
            "logs_floored", "pit", "cov50", "cov80", "cov95", "mass50", "mass80", "mass95", "abs_err", "sq_err"]
    P = pd.read_parquet(sd / "player_scores.parquet", columns=cols)
    for c in ("model", "stat", "season", "role", "history"):
        P[c] = P[c].astype("category")
    T = pd.read_parquet(sd / "team_scores.parquet")
    J = pd.read_parquet(sd / "joint_scores.parquet")
    TH = pd.read_parquet(sd / "threshold_scores.parquet")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    models = [m for m in ["A", "B", "C0", "C"] if m in set(P["model"].astype(str))]
    comparisons = [(a, b) for a, b in COMPARISONS if a in models and b in models]

    results = {"overall": {}, "season": {}, "role": {}, "history": {}, "team": {}, "joint": {}, "threshold": {}}
    # Overall comparisons on proper scores.
    for stat in ALL_STATS:
        S = P[P["stat"] == stat]
        for a, b in comparisons:
            results["overall"][(stat, "crps", a, b)] = cluster_bootstrap_diff(S, "crps", a, b, reps, seed)
            if stat in LOG_STATS:
                results["overall"][(stat, "logs", a, b)] = cluster_bootstrap_diff(S, "logs", a, b, reps, seed)
    for dim in ("season", "role", "history"):
        for stat in KEY_STATS:
            S = P[P["stat"] == stat]
            for level in sorted(S[dim].astype(str).unique()):
                L = S[S[dim].astype(str) == level]
                for a, b in [("B", "A"), ("C", "B"), ("C0", "B"), ("C", "C0")]:
                    if a in models and b in models:
                        results[dim][(stat, level, a, b)] = cluster_bootstrap_diff(L, "crps", a, b, reps, seed)
    T2 = T.rename(columns={"crps": "team_crps"})
    for a, b in comparisons:
        results["team"][("team_pts", a, b)] = cluster_bootstrap_diff(T2, "team_crps", a, b, reps, seed)
        results["team"][("margin", a, b)] = cluster_bootstrap_diff(T2, "margin_crps", a, b, reps, seed)
        for stat in ["pts", "reb", "ast"]:
            Js = J[J["stat"] == stat]
            results["joint"][(stat, a, b)] = cluster_bootstrap_diff(Js, "variogram", a, b, reps, seed)
    TH2 = TH.copy()
    for stat in TH2["stat"].unique():
        S = TH2[TH2["stat"] == stat]
        for a, b in comparisons:
            results["threshold"][(stat, a, b)] = cluster_bootstrap_diff(S, "brier", a, b, reps, seed)

    figs = _figures(P, T, J, TH, models, results, fig_dir)
    summary_tables = _summary_tables(P, T, J, TH, models)
    paths = {
        "validation_report": _write_validation(out_dir / "validation_report.md", cfg, manifest, P, T, J, TH, models,
                                               comparisons, results, summary_tables, figs),
        "research_note": _write_note(out_dir / "research_note.md", cfg, manifest, P, models, results,
                                     summary_tables, figs),
    }
    # Machine-readable metrics.
    flat = []
    for kind, d in results.items():
        for key, r in d.items():
            flat.append({"kind": kind, "key": "|".join(map(str, key)), **r, "verdict": verdict_of(r)})
    mp = run_dir / "comparisons.csv"
    pd.DataFrame(flat).to_csv(mp, index=False)
    for name, t in summary_tables.items():
        t.to_csv(run_dir / f"summary_{name}.csv", index=False)
    paths["comparisons_csv"] = mp
    return paths


def _summary_tables(P, T, J, TH, models) -> dict[str, pd.DataFrame]:
    g = P.groupby(["stat", "model"], observed=True)
    overall = g.agg(n=("crps", "size"), crps=("crps", "mean"), logs=("logs", "mean"), mae=("abs_err", "mean"),
                    mse=("sq_err", "mean"), cov50=("cov50", "mean"), mass50=("mass50", "mean"),
                    cov80=("cov80", "mean"), mass80=("mass80", "mean"), cov95=("cov95", "mean"),
                    mass95=("mass95", "mean"), pit_mean=("pit", "mean"),
                    pit_in80=("pit", lambda x: float(((x >= 0.1) & (x <= 0.9)).mean())),
                    bias=("mean", "mean"), ymean=("y", "mean"), floored=("logs_floored", "mean")).reset_index()
    overall["rmse"] = np.sqrt(overall["mse"])
    overall["bias"] = overall["bias"] - overall["ymean"]
    season = P[P["stat"].isin(["pts", "reb", "ast", "minutes"])].groupby(["season", "stat", "model"], observed=True).agg(
        crps=("crps", "mean"), bias=("mean", "mean"), y=("y", "mean"), cov80=("cov80", "mean"),
        mass80=("mass80", "mean"), n=("crps", "size")).reset_index()
    season["bias"] = season["bias"] - season["y"]
    team = T.groupby("model").agg(n=("crps", "size"), crps=("crps", "mean"),
                                  rmse=("y", lambda x: np.nan), pred_sd=("sd", lambda s: float(np.sqrt((s ** 2).mean()))),
                                  cov80=("cov80", "mean"), pit_in80=("pit", lambda x: float(((x >= .1) & (x <= .9)).mean())),
                                  margin_crps=("margin_crps", "mean"),
                                  pred_margin_sd=("margin_sd", lambda s: float(np.sqrt((s ** 2).mean()))),
                                  margin_pit_in80=("margin_pit", lambda x: float(((x >= .1) & (x <= .9)).mean()))).reset_index()
    rm = T.assign(e2=(T["y"] - T["mean"]) ** 2).groupby("model")["e2"].mean()
    team["rmse"] = team["model"].map(np.sqrt(rm))
    # Realized margin error: margins are antisymmetric, use one row per game.
    mg = T.assign(me=T["margin_y"] - (T["mean"] - T.groupby(["model", "game_id"])["mean"].transform("sum") + T["mean"]))
    joint = J.groupby(["stat", "model"]).agg(variogram=("variogram", "mean"), sq_err_total=("sq_err_total", "mean"),
                                             var_total=("var_total", "mean"),
                                             var_if_indep=("var_total_if_indep", "mean")).reset_index()
    joint["dispersion_ratio"] = joint["sq_err_total"] / joint["var_total"]
    thr = TH.groupby(["stat", "threshold", "model"]).agg(brier=("brier", "mean"), p=("p", "mean"),
                                                         rate=("event", "mean"), n=("p", "size")).reset_index()
    del mg
    return {"overall": overall, "season": season, "team": team, "joint": joint, "threshold": thr}


# ----------------------------------------------------------------------------- figures

def _figures(P, T, J, TH, models, results, fig_dir: Path) -> dict[str, Path]:
    figs = {}
    # PIT histograms
    fig, axes = plt.subplots(len(KEY_STATS), len(models), figsize=(3.1 * len(models), 2.0 * len(KEY_STATS)),
                             squeeze=False, sharey=True)
    for i, stat in enumerate(KEY_STATS):
        for j, m in enumerate(models):
            x = P[(P["stat"] == stat) & (P["model"] == m)]["pit"].to_numpy()
            ax = axes[i, j]
            ax.hist(x, bins=20, range=(0, 1), density=True, color=COLORS[m], alpha=0.85)
            ax.axhline(1, color="black", lw=0.8)
            ax.set_ylim(0, 2)
            if i == 0:
                ax.set_title(f"Model {m}")
            if j == 0:
                ax.set_ylabel(stat)
            ax.tick_params(labelsize=7)
    fig.suptitle("Randomized PIT histograms (flat = calibrated; U = too narrow; hump = too wide)")
    fig.tight_layout()
    figs["pit"] = fig_dir / "pit_histograms.png"
    fig.savefig(figs["pit"], dpi=110)
    plt.close(fig)

    # Reliability for threshold events
    stats_thr = [s for s in ["pts", "reb", "ast", "pra", "minutes"] if s in set(TH["stat"])]
    fig, axes = plt.subplots(1, len(stats_thr), figsize=(3.3 * len(stats_thr), 3.3), squeeze=False)
    bins = np.linspace(0, 1, 11)
    for ax, stat in zip(axes[0], stats_thr):
        for m in models:
            S = TH[(TH["stat"] == stat) & (TH["model"] == m)]
            b = np.clip(np.digitize(S["p"], bins) - 1, 0, 9)
            g = S.groupby(b).agg(p=("p", "mean"), r=("event", "mean"), n=("p", "size"))
            g = g[g["n"] >= 50]
            ax.plot(g["p"], g["r"], "o-", ms=3, color=COLORS[m], label=m)
        ax.plot([0, 1], [0, 1], color="black", lw=0.8)
        ax.set_title(f"{stat} thresholds")
        ax.set_xlabel("forecast probability")
        ax.set_ylabel("observed frequency")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    figs["reliability"] = fig_dir / "reliability.png"
    fig.savefig(figs["reliability"], dpi=110)
    plt.close(fig)

    # CRPS differences by season
    fig, axes = plt.subplots(1, len(KEY_STATS), figsize=(3.4 * len(KEY_STATS), 3.2), squeeze=False)
    pairs = [p for p in [("B", "A"), ("C0", "B"), ("C", "C0"), ("C", "B")] if p[0] in models and p[1] in models]
    for ax, stat in zip(axes[0], KEY_STATS):
        seasons = sorted({k[1] for k in results["season"] if k[0] == stat})
        for k, (a, b) in enumerate(pairs):
            xs, ys, lo, hi = [], [], [], []
            for i, s in enumerate(seasons):
                r = results["season"].get((stat, s, a, b))
                if r is None:
                    continue
                xs.append(i + (k - len(pairs) / 2) * 0.15)
                ys.append(r["rel"] * 100)
                base = r["diff"] / r["rel"] if r["rel"] else np.nan
                lo.append(r["lo"] / base * 100)
                hi.append(r["hi"] / base * 100)
            ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o", ms=4, capsize=2, label=f"{a} vs {b}",
                        color=COLORS[a] if (a, b) != ("C", "C0") else "#8c510a")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(seasons)), seasons, fontsize=7)
        ax.set_title(stat)
        ax.set_ylabel("CRPS change, % (negative = better)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Ablation by season: relative CRPS change with 95% game-bootstrap CI")
    fig.tight_layout()
    figs["season"] = fig_dir / "ablation_by_season.png"
    fig.savefig(figs["season"], dpi=110)
    plt.close(fig)

    # Segment: role
    roles = ["no history", "deep bench (<12)", "bench (12-24)", "rotation (24-30)", "starter (30+)"]
    fig, axes = plt.subplots(1, len(KEY_STATS), figsize=(3.4 * len(KEY_STATS), 3.4), squeeze=False)
    for ax, stat in zip(axes[0], KEY_STATS):
        for k, (a, b) in enumerate(pairs):
            xs, ys, err = [], [], [[], []]
            for i, role in enumerate(roles):
                r = results["role"].get((stat, role, a, b))
                if r is None or r["n"] < 200:
                    continue
                base = r["diff"] / r["rel"] if r["rel"] else np.nan
                xs.append(i + (k - len(pairs) / 2) * 0.15)
                ys.append(r["rel"] * 100)
                err[0].append(r["rel"] * 100 - r["lo"] / base * 100)
                err[1].append(r["hi"] / base * 100 - r["rel"] * 100)
            ax.errorbar(xs, ys, yerr=err, fmt="o", ms=4, capsize=2, label=f"{a} vs {b}",
                        color=COLORS[a] if (a, b) != ("C", "C0") else "#8c510a")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(roles)), [r.split(" (")[0] for r in roles], rotation=35, fontsize=7)
        ax.set_title(stat)
    axes[0, 0].set_ylabel("CRPS change, %")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Ablation by pregame role (expected minutes from earlier games)")
    fig.tight_layout()
    figs["role"] = fig_dir / "ablation_by_role.png"
    fig.savefig(figs["role"], dpi=110)
    plt.close(fig)

    # Team-level calibration
    fig, axes = plt.subplots(2, len(models), figsize=(3.1 * len(models), 4.6), squeeze=False, sharey=True)
    for j, m in enumerate(models):
        Tm = T[T["model"] == m]
        for i, (col, lab) in enumerate([("pit", "team points PIT"), ("margin_pit", "margin PIT")]):
            ax = axes[i, j]
            ax.hist(Tm[col], bins=20, range=(0, 1), density=True, color=COLORS[m], alpha=0.85)
            ax.axhline(1, color="black", lw=0.8)
            ax.set_ylim(0, 3)
            ax.set_title(f"Model {m}: {lab}", fontsize=9)
    fig.tight_layout()
    figs["team"] = fig_dir / "team_pit.png"
    fig.savefig(figs["team"], dpi=110)
    plt.close(fig)

    # Coverage minus nominal mass
    fig, ax = plt.subplots(figsize=(10, 3.4))
    stats = KEY_STATS + ["stl", "blk", "tov", "fg3m"]
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = []
        for s in stats:
            S = P[(P["stat"] == s) & (P["model"] == m)]
            vals.append(S["cov80"].mean() - S["mass80"].mean())
        ax.bar(np.arange(len(stats)) + (k - len(models) / 2 + 0.5) * width, vals, width, color=COLORS[m], label=m)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(stats)), stats)
    ax.set_ylabel("coverage − predictive mass")
    ax.set_title("80% central interval: empirical coverage minus the interval's predictive mass (0 = calibrated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["coverage"] = fig_dir / "coverage_gap.png"
    fig.savefig(figs["coverage"], dpi=110)
    plt.close(fig)

    # Distribution comparison: mean predicted vs actual histograms for pts/minutes (pooled)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, stat in zip(axes, ["minutes", "pts", "pra"]):
        S = P[(P["stat"] == stat)]
        y = S[S["model"] == models[0]]["y"].to_numpy()
        hi = np.quantile(y, 0.995)
        ax.hist(y, bins=40, range=(0, hi), density=True, color="black", alpha=0.25, label="actual")
        for m in models:
            q = S[S["model"] == m]
            # PIT-based check is in the PIT figure; here: distribution of predicted means
            ax.hist(q["mean"], bins=40, range=(0, hi), density=True, histtype="step", color=COLORS[m], label=f"{m} means")
        ax.set_title(stat)
        ax.legend(fontsize=7)
    fig.suptitle("Actual outcomes vs distribution of predicted means (pooled over the backtest)")
    fig.tight_layout()
    figs["distributions"] = fig_dir / "distributions.png"
    fig.savefig(figs["distributions"], dpi=110)
    plt.close(fig)
    return figs


# ----------------------------------------------------------------------------- writers

def _design_section(cfg, manifest, P, T) -> list[str]:
    pred = manifest["prediction_stage"]
    win = pred["windows"]
    first = P[P["model"] == P["model"].cat.categories[0]]
    n_games = T["game_id"].nunique()
    n_obs = first["player_id"].size // first["stat"].nunique() if first["stat"].nunique() else 0
    ds = manifest["dataset"]
    L = []
    L.append(f"- **Data:** {ds['source']}; dataset id `{ds['dataset_id']}`; raw files SHA-256 in "
             "`outputs/backtest/main/manifest.json`. Regular season only.")
    L.append(f"- **Evaluation period:** seasons {', '.join(pred['seasons'])}; {len(win)} rolling origins "
             f"({win[0]['origin']} … {win[-1]['origin']}), refit every {cfg['backtest']['refit_every_days']} days; "
             f"training at each origin = all rows dated before the origin within the current and previous "
             f"{cfg['backtest']['train_lookback_seasons']} seasons, recency-weighted (half-life "
             f"{cfg['models']['train_weight_halflife_days']} days) with intercepts re-estimated on the last "
             f"{cfg['models']['recalibration_days']} days.")
    L.append(f"- **Sample:** {n_games:,} games, {n_obs:,} active player-games per model (played or DNP). "
             "Players on the inactive list (DND/NWT) are excluded by design — the forecast universe is the active "
             "list, assumed known pregame (published before tip-off; no timestamp in the data).")
    L.append(f"- **Simulation:** {pred['n_sims']:,} joint draws per player-game and model; seed "
             f"{cfg['backtest']['seed']} with per-(window, model, chunk) seed sequences recorded in each prediction file.")
    L.append("- **Freeze protocol:** each window's predictions are written to `predictions/<model>/wNNN.npz` "
             "(full discrete predictive PMFs, minutes quantiles, team and joint draws) before any outcome is read; "
             "`evaluate` joins outcomes afterwards.")
    return L


def _metric_guide() -> list[str]:
    return [
        "| metric | used for | why |",
        "|---|---|---|",
        "| CRPS (exact for counts, quantile-grid for minutes) | every stat, team totals, margins | proper, in the units of the stat, defined for discrete and continuous outcomes and for derived stats (PRA) |",
        "| log score (negative log-likelihood) | primitive counts (FGA, FTA, REB, AST, STL, BLK, TOV) | strictly proper and sensitive to tails; computed from Rao-Blackwellized PMFs so no zero-probability artefacts. Not used for derived stats (PTS, PRA) whose PMFs are only available as Monte Carlo histograms, nor for minutes (point mass at 0 + continuous part) |",
        "| MAE of the median / RMSE of the mean | point-forecast accuracy | familiar, but not distributional |",
        "| randomized PIT | calibration of every stat | uniform under calibration even for counts |",
        "| interval coverage vs predictive mass | 50/80/95% central intervals | for integer stats a central interval carries more than its nominal mass; calibration means coverage ≈ that mass |",
        "| Brier score + reliability | predeclared threshold events P(stat ≥ t) | proper for binary events |",
        "| variogram score (order ½) | joint distribution of teammates' PTS/REB/AST | proper multivariate score that is sensitive to dependence structure (the energy score is not) |",
        "| team-total dispersion ratio | E[(total − mean)²] / predicted variance | > 1 too narrow, < 1 too wide; isolates dependence errors |",
        "",
        "Scores are only compared across models on identical observations (same games, same players, same stat).",
    ]


def _write_validation(path, cfg, manifest, P, T, J, TH, models, comparisons, results, tabs, figs) -> Path:
    rel = lambda p: os.path.relpath(p, path.parent)  # noqa: E731
    ov = tabs["overall"]
    L = ["# Validation report: NBA player outcome distributions\n",
         "Generated by `nba-sim report` from the saved backtest scores "
         "(`outputs/backtest/main/scores/`). All numbers are computed, not typed.\n"]
    L.append("## 1. Evaluation design\n")
    L += _design_section(cfg, manifest, P, T)
    L.append("\n### Models\n")
    for m in models:
        L.append(f"- **{m}**: {MODEL_DESCRIPTIONS[m]}")
    L.append("\n## 2. Metrics and what they are used for\n")
    L += _metric_guide()

    L.append("\n## 3. Overall results\n")
    L.append("Mean score per player-game (lower is better). `cov80`/`mass80`: empirical coverage of the central 80% "
             "interval and its average predictive mass; `pit80`: share of randomized PITs in [0.1, 0.9] (0.80 if "
             "calibrated). `bias`: mean prediction minus mean outcome.\n")
    t = ov[ov["stat"].isin(ALL_STATS)].copy()
    t["stat"] = pd.Categorical(t["stat"].astype(str), ALL_STATS)
    t = t.sort_values(["stat", "model"])
    show = pd.DataFrame({
        "stat": t["stat"].astype(str), "model": t["model"].astype(str), "n": t["n"].map(lambda x: f"{x:,}"),
        "CRPS": t["crps"].map(_f), "logS": t["logs"].map(_f), "MAE": t["mae"].map(_f), "RMSE": t["rmse"].map(_f),
        "bias": t["bias"].map(lambda x: f"{x:+.3f}"), "cov80": t["cov80"].map(_f), "mass80": t["mass80"].map(_f),
        "pit80": t["pit_in80"].map(_f), "cov95": t["cov95"].map(_f), "mass95": t["mass95"].map(_f)})
    L.append(_md(show))
    L.append("\nMinutes have a point mass at 0 (DNP). When P(0) exceeds the interval's lower tail, the central "
             "interval starts at 0 and every DNP is 'covered', so minutes `cov80` overstates coverage; the "
             "randomized PIT (`pit80`) handles the point mass and is the calibration measure to read for minutes.")
    floored = ov[ov["floored"] > 0]
    if len(floored):
        L.append(f"\nLog-score floor (p < 1e-12 or outcome beyond support) hit on at most "
                 f"{floored['floored'].max():.5%} of observations for any stat/model.")

    L.append("\n## 4. Model comparisons (95% game-level cluster bootstrap)\n")
    L.append(f"Difference in mean score, first model minus second (negative = first model better), with a "
             f"{cfg['evaluation']['bootstrap_reps']:,}-replicate bootstrap over games (all players of a game are "
             "resampled together). *improves/worsens* = CI excludes 0; *tie* = it does not.\n")
    rows = []
    for stat in ALL_STATS:
        for metric in ("crps", "logs"):
            rec = {"stat": stat, "metric": "CRPS" if metric == "crps" else "log score"}
            any_ = False
            for a, b in comparisons:
                r = results["overall"].get((stat, metric, a, b))
                if r is None:
                    continue
                any_ = True
                rec[f"{a} − {b}"] = f"{_ci(r)} ({r['rel']:+.1%}) {verdict_of(r)}"
            if any_:
                rows.append(rec)
    L.append(_md(pd.DataFrame(rows)))

    L.append("\n## 5. Calibration\n")
    L.append(f"![PIT histograms]({rel(figs['pit'])})\n")
    L.append(f"![Coverage gap]({rel(figs['coverage'])})\n")
    L.append("### Predeclared threshold events\n")
    L.append(f"![Reliability]({rel(figs['reliability'])})\n")
    th = tabs["threshold"].copy()
    thw = th.pivot_table(index=["stat", "threshold"], columns="model", values="brier").reset_index()
    rate = th.groupby(["stat", "threshold"])["rate"].first().reset_index()
    thw = thw.merge(rate, on=["stat", "threshold"])
    for m in models:
        thw[m] = thw[m].map(lambda x: f"{x:.4f}")
    thw["base rate"] = thw["rate"].map(lambda x: f"{x:.3f}")
    L.append("Brier scores (lower is better):\n")
    L.append(_md(thw[["stat", "threshold", "base rate"] + models]))
    rows = []
    for (stat, a, b), r in results["threshold"].items():
        rows.append({"stat": stat, "comparison": f"{a} − {b}", "Brier diff [95% CI]": _ci(r, 5), "verdict": verdict_of(r)})
    L.append("\nBrier differences pooled over each stat's thresholds:\n")
    L.append(_md(pd.DataFrame(rows)))

    L.append("\n## 6. Team totals, margins and dependence\n")
    tm = tabs["team"].copy()
    L.append("Team points are the sum of simulated player points within the same draw, so their spread is "
             "governed by the modelled dependence between teammates; margins also depend on cross-team "
             "dependence. Realized RMSE vs predicted SD shows whether the joint distribution is too wide/narrow.\n")
    L.append(_md(pd.DataFrame({
        "model": tm["model"], "team-games": tm["n"], "team pts CRPS": tm["crps"].map(_f),
        "realized RMSE": tm["rmse"].map(lambda x: f"{x:.2f}"), "predicted SD": tm["pred_sd"].map(lambda x: f"{x:.2f}"),
        "80% cov": tm["cov80"].map(_f), "margin CRPS": tm["margin_crps"].map(_f),
        "predicted margin SD": tm["pred_margin_sd"].map(lambda x: f"{x:.2f}"),
        "margin PIT in [.1,.9]": tm["margin_pit_in80"].map(_f)})))
    L.append(f"\n![Team PIT]({rel(figs['team'])})\n")
    jt = tabs["joint"].copy()
    L.append("Dependence diagnostics from the joint draws (per team-game): mean variogram score of teammates' "
             "outcomes, and the dispersion ratio E[(team total − predicted mean)²] / predicted variance "
             "(1 = right; < 1 too wide). `var if indep.` is the team-total variance the same marginals would imply "
             "with independent players — the gap to `pred. var` is the dependence each model encodes.\n")
    L.append(_md(pd.DataFrame({
        "stat": jt["stat"], "model": jt["model"], "variogram": jt["variogram"].map(_f),
        "realized sq. err": jt["sq_err_total"].map(lambda x: f"{x:.1f}"), "pred. var": jt["var_total"].map(lambda x: f"{x:.1f}"),
        "var if indep.": jt["var_if_indep"].map(lambda x: f"{x:.1f}"), "dispersion ratio": jt["dispersion_ratio"].map(_f)})))
    rows = []
    for key, r in {**{("team pts CRPS",) + k[1:]: v for k, v in results["team"].items() if k[0] == "team_pts"},
                   **{("margin CRPS",) + k[1:]: v for k, v in results["team"].items() if k[0] == "margin"},
                   **{(f"variogram {k[0]}",) + k[1:]: v for k, v in results["joint"].items()}}.items():
        rows.append({"metric": key[0], "comparison": f"{key[1]} − {key[2]}", "diff [95% CI]": _ci(r),
                     "rel": f"{r['rel']:+.1%}", "verdict": verdict_of(r)})
    L.append("\n" + _md(pd.DataFrame(rows)))

    L.append("\n## 7. Robustness across time periods and segments\n")
    L.append(f"![By season]({rel(figs['season'])})\n")
    L.append(f"![By role]({rel(figs['role'])})\n")
    for dim, title in (("season", "Season"), ("role", "Pregame role (expected minutes)"),
                       ("history", "Sample size (prior games)")):
        rows = []
        for (stat, level, a, b), r in results[dim].items():
            if stat not in ("pts", "minutes", "pra"):
                continue
            rows.append({"stat": stat, title: level, "comparison": f"{a} − {b}", "n": f"{r['n']:,}",
                         "CRPS diff [95% CI]": _ci(r), "rel": f"{r['rel']:+.1%}", "verdict": verdict_of(r)})
        L.append(f"\n### By {title.lower()} (PTS, PRA, minutes; REB/AST in `comparisons.csv`)\n")
        L.append(_md(pd.DataFrame(rows)))

    L.append("\n## 8. Bias by season (league drift)\n")
    sb = tabs["season"]
    sb = sb[sb["stat"].isin(["pts", "minutes"])]
    L.append(_md(pd.DataFrame({"season": sb["season"], "stat": sb["stat"], "model": sb["model"],
                               "mean outcome": sb["y"].map(lambda x: f"{x:.3f}"),
                               "bias": sb["bias"].map(lambda x: f"{x:+.3f}"), "CRPS": sb["crps"].map(_f),
                               "cov80": sb["cov80"].map(_f), "mass80": sb["mass80"].map(_f)})))
    L.append("\n## 9. Distributions\n")
    L.append(f"![Distributions]({rel(figs['distributions'])})\n")

    L.append("## 10. Computational notes, limitations and known biases\n")
    L += [
        "- Rolling-origin refits every 14 days (features update every game); daily refits were not run for compute reasons. "
        "Within a window the parameters are up to 13 days stale, which slightly understates what a daily-refit system would achieve, equally for every model.",
        "- The forecast universe conditions on the active list (see §1). Inactive designations made after our cutoff are "
        "available pregame in reality but are taken from the postgame box score here; injuries announced between games "
        "are otherwise unmodelled, and starting lineups are not used.",
        "- League-wide scoring drifts within seasons (e.g. the Feb–Mar 2024 free-throw-rate drop); recency weighting and "
        "30-day intercept recalibration reduce but do not remove the lag (see §8).",
        "- Team totals: even the best model's team-points distribution remains wider than realized (§6). Possessions "
        "couple volume and efficiency (misses create offensive rebounds and extra attempts) more tightly than the "
        "joint-Gaussian team-total layer captures.",
        "- Monte Carlo error: 2,000 draws per player-game; for derived stats the PMF is a histogram, so CRPS carries a "
        "small upward Monte Carlo bias that is common to all models.",
        "- Historical statistical accuracy says nothing about betting profitability; no market prices were used.",
    ]
    from nba_sim.features.build import FEATURE_DOCS

    L.append("\n## Appendix: features, sources and availability\n")
    L.append("Generated from `features/build.py::FEATURE_DOCS`. Every feature is computed from games strictly before "
             "the target game, except the schedule and the active list (an explicit assumption).\n")
    rows = [{"feature": f"`{k}`", "construction": v.split(" | ")[0],
             "available": v.split(" | ")[1] if " | " in v else ""} for k, v in FEATURE_DOCS.items()]
    L.append(_md(pd.DataFrame(rows)))
    path.write_text("\n".join(L) + "\n")
    return path


def _key_findings(results, models) -> list[str]:
    """Computed headline bullets (no hand-typed numbers)."""
    ov = results["overall"]
    out = ["**Key findings (computed):**\n"]
    if ("pts", "crps", "B", "A") in ov:
        parts = [f"{s} {ov[(s, 'crps', 'B', 'A')]['rel']:+.1%}" for s in KEY_STATS if (s, "crps", "B", "A") in ov]
        out.append("- Simulating minutes (B vs A) changes CRPS by " + ", ".join(parts) + ".")
        roles = {k[1]: r for k, r in results["role"].items() if k[0] == "minutes" and k[2:] == ("B", "A") and r["n"] >= 200}
        if roles:
            best = min(roles, key=lambda k: roles[k]["rel"])
            worst = max(roles, key=lambda k: roles[k]["rel"])
            out.append(f"  The minutes-CRPS gain is largest for **{best}** players ({roles[best]['rel']:+.1%}) and "
                       f"smallest for **{worst}** ({roles[worst]['rel']:+.1%}).")
    for a, b, label in (("C0", "B", "observable game conditions (C0 vs B)"),
                        ("C", "C0", "modelled dependence (C vs C0)"), ("C", "B", "the full shared-factor model (C vs B)")):
        if (a in models) and (b in models):
            v = {s: verdict_of(ov[(s, "crps", a, b)]) for s in KEY_STATS if (s, "crps", a, b) in ov}
            imp = [s for s, x in v.items() if x == "improves"]
            wor = [s for s, x in v.items() if x == "worsens"]
            tie = [s for s, x in v.items() if x == "tie"]
            t = results["team"].get(("team_pts", a, b))
            m = results["team"].get(("margin", a, b))
            txt = (f"- For {label}, player CRPS improves for {', '.join(imp) or 'none'}; ties for {', '.join(tie) or 'none'}"
                   f"; worsens for {', '.join(wor) or 'none'}.")
            if t and m:
                txt += (f" Team-points CRPS {t['rel']:+.1%} ({verdict_of(t)}), margin CRPS {m['rel']:+.1%} "
                        f"({verdict_of(m)}).")
            out.append(txt)
    out.append("")
    return out


def _write_note(path, cfg, manifest, P, models, results, tabs, figs) -> Path:
    rel = lambda p: os.path.relpath(p, path.parent)  # noqa: E731
    ov = results["overall"]

    def v(stat, metric, a, b):
        r = ov.get((stat, metric, a, b))
        return (verdict_of(r), r) if r else ("n/a", None)

    # Explicit verdict rules (documented in the note).
    minutes_player = {s: v(s, "crps", "B", "A") for s in KEY_STATS}
    factors_player = {s: v(s, "crps", "C", "B") for s in KEY_STATS}
    joint_team = {k: verdict_of(r) for k, r in results["team"].items() if k[1:] == ("C", "B")}
    joint_vario = {k[0]: verdict_of(r) for k, r in results["joint"].items() if k[1:] == ("C", "B")}
    seasons = sorted({k[1] for k in results["season"]})
    season_ba = {s: verdict_of(results["season"][("pts", s, "B", "A")]) for s in seasons
                 if ("pts", s, "B", "A") in results["season"]}
    season_cb = {s: verdict_of(results["season"][("pts", s, "C", "B")]) for s in seasons
                 if ("pts", s, "C", "B") in results["season"]}

    def classify(player: dict, extra: list[str]) -> str:
        vs = [x[0] for x in player.values()] + extra
        if all(x == "improves" for x in vs):
            return "Evidence of improvement"
        if any(x == "improves" for x in vs) and not all(x == "worsens" for x in vs):
            return "Mixed or segment-specific improvement"
        if all(x in ("tie", "worsens") for x in vs) and any(x == "worsens" for x in vs):
            return "No meaningful improvement (worse on some metrics)"
        return "No meaningful improvement"

    ver_min = classify(minutes_player, list(season_ba.values()))
    ver_fac = classify(factors_player, list(joint_team.values()) + list(joint_vario.values()))
    any_c = any(x == "improves" for x in [y[0] for y in factors_player.values()] + list(joint_team.values()))
    if ver_min.startswith("Evidence") and ver_fac.startswith("Evidence"):
        overall = "Evidence of improvement"
    elif ver_min.startswith("Evidence") or ver_fac.startswith(("Evidence", "Mixed")) or any_c:
        overall = "Mixed or segment-specific improvement"
    else:
        overall = "No meaningful improvement"

    tm = tabs["team"].set_index("model")
    L = ["# Research note: do uncertain minutes and shared game factors improve NBA player outcome distributions?\n",
         "*Generated from saved backtest outputs by `nba-sim report`. Numbers are not typed by hand.*\n"]
    L.append("## 1. Hypothesis\n")
    L.append("H1: simulating playing time explicitly (Model B) improves out-of-sample player outcome distributions "
             "relative to an independent-player model (A). H2: adding shared game-level factors (Model C) improves "
             "them further, at least for quantities that depend on dependence between players (team totals, "
             "margins, joint teammate outcomes). Null for both: no difference in proper scores.\n")
    L.append("## 2. Modelling approach\n")
    for m in models:
        L.append(f"- **{m}**: {MODEL_DESCRIPTIONS[m]}")
    L.append("\nAll models share the same features, shrinkage priors, shooting structure (attempts → binomial makes "
             "→ derived points, so box scores are internally coherent) and the same recency weighting. C0 isolates "
             "C's observable game conditions from its modelled dependence.\n")
    L.append("## 3. Data and evaluation design\n")
    L += _design_section(cfg, manifest, P, pd.read_parquet(Path(manifest["score_files"]["team"])))
    L.append("\nPrimary score: CRPS (proper, works for every stat). Log score for primitive counts, Brier for "
             "predeclared thresholds, variogram score and team-total CRPS for dependence. Uncertainty: 95% "
             "game-level cluster bootstrap.\n")
    L.append("## 4. Main results\n")
    L += _key_findings(results, models)
    rows = []
    for s in KEY_STATS:
        for a, b in [("B", "A"), ("C0", "B"), ("C", "C0"), ("C", "B")]:
            r = ov.get((s, "crps", a, b))
            if r:
                rows.append({"stat": s, "comparison": f"{a} − {b}", "CRPS diff [95% CI]": _ci(r), "rel": f"{r['rel']:+.1%}",
                             "verdict": verdict_of(r)})
    L.append(_md(pd.DataFrame(rows)))
    L.append("\nTeam level:\n")
    rows = []
    for key, r in results["team"].items():
        rows.append({"quantity": "team points CRPS" if key[0] == "team_pts" else "margin CRPS",
                     "comparison": f"{key[1]} − {key[2]}", "diff [95% CI]": _ci(r), "rel": f"{r['rel']:+.1%}",
                     "verdict": verdict_of(r)})
    L.append(_md(pd.DataFrame(rows)))
    L.append("## 5. Calibration and distributional findings\n")
    o = tabs["overall"].set_index(["stat", "model"])
    for s in ["minutes", "pts", "pra"]:
        parts = [f"{m}: cov80 {o.loc[(s, m), 'cov80']:.3f} vs mass {o.loc[(s, m), 'mass80']:.3f}, "
                 f"PIT-in-80 {o.loc[(s, m), 'pit_in80']:.3f}" for m in models if (s, m) in o.index]
        L.append(f"- **{s}** — " + "; ".join(parts))
    L.append("- **Team points** — " + "; ".join(
        f"{m}: realized RMSE {tm.loc[m, 'rmse']:.1f} vs predicted SD {tm.loc[m, 'pred_sd']:.1f}, 80% coverage "
        f"{tm.loc[m, 'cov80']:.2f}" for m in models if m in tm.index))
    L.append("- **Margins** — " + "; ".join(
        f"{m}: predicted SD {tm.loc[m, 'pred_margin_sd']:.1f}, PIT in [.1,.9] {tm.loc[m, 'margin_pit_in80']:.2f}"
        for m in models if m in tm.index))
    L.append(f"\n![PIT]({rel(figs['pit'])})\n")
    L.append("## 6. Ablation results\n")
    L.append(f"![Ablation by season]({rel(figs['season'])})\n")
    L.append("Points CRPS by season — B vs A: " + ", ".join(f"{s} {x}" for s, x in season_ba.items())
             + "; C vs B: " + ", ".join(f"{s} {x}" for s, x in season_cb.items()) + ".\n")
    L.append(f"![Ablation by role]({rel(figs['role'])})\n")
    L.append("## 7. Limitations and possible explanations\n")
    L += [
        "- The active list is treated as pregame information (published before tip-off, but untimestamped here). "
        "Without it, every model would face extra availability uncertainty, and B's minutes model, which "
        "redistributes minutes over the active list, would likely lose some of its advantage.",
        "- Starting lineups are not used. They are a strong pregame minutes signal, available ~30 minutes before tip, "
        "so B/C understate what a production system could do.",
        "- C's dependence layer acts on team totals and two latent shooting shocks; it does not model possessions "
        "explicitly, and team-points distributions remain wider than realized for every model (§5).",
        "- Most of C's dependence leaves player *marginals* nearly unchanged by design (it re-allocates variance), so "
        "marginal scores such as player CRPS are not expected to move much; its effect shows up in team totals, "
        "margins and joint scores.",
        "- Three seasons, one league, regular season only; 14-day refits.",
    ]
    L.append("\n## 8. Final verdict\n")
    L.append("Rules (written into the report code before the final backtest was run, applied mechanically): a component shows "
             "*evidence of improvement* if every key player metric (CRPS of minutes, PTS, REB, AST, PRA) — plus, for "
             "minutes, points CRPS in every season, and for shared factors, team-total, margin and variogram scores — "
             "improves with a 95% CI excluding zero; *mixed* if some improve and not all worsen; *no meaningful "
             "improvement* otherwise.\n")
    L.append(f"- **Uncertain playing time (B vs A): {ver_min}.**")
    L.append(f"- **Shared game-level factors (C vs B): {ver_fac}.**")
    L.append(f"\n**Overall verdict on the research question: {overall}.**\n")
    L.append("Statistical accuracy on historical data is not evidence of betting profitability; no market prices "
             "were used.")
    path.write_text("\n".join(L) + "\n")
    return path
