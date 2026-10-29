"""Historical matchup demonstration and ad-hoc matchup simulation.

Protocol for the historical demo (enforced in code):
  1. Pick the game by a PREGAME rule (or take --game-id). The rule never looks at the outcome.
  2. Cut the data to games dated strictly before the game date; append outcome-free rows for the
     target game's active list (simulation.pregame.prepare_target).
  3. Fit each model on that history only; assert the game id is absent from training.
  4. Draw exactly N (default 10,000) joint simulations per model with a recorded seed.
  5. Write the frozen predictions and their SHA-256 hashes to disk.
  6. Only then load the actual box score, verify the hashes are unchanged, and compare.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nba_sim.config import resolve  # noqa: E402
from nba_sim.data.ingest import Dataset  # noqa: E402
from nba_sim.data.schema import STATUS_INACTIVE  # noqa: E402
from nba_sim.evaluation import metrics as M  # noqa: E402
from nba_sim.evaluation.backtest import training_frame  # noqa: E402
from nba_sim.features.leakage_checks import assert_disjoint_games, assert_training_before  # noqa: E402
from nba_sim.models import MODEL_DESCRIPTIONS, build_model  # noqa: E402
from nba_sim.reporting.manifest import code_fingerprint, write_manifest  # noqa: E402
from nba_sim.simulation.predictive import MINUTE_LEVELS, load_predictions, save_predictions, summarize  # noqa: E402
from nba_sim.simulation.pregame import TargetSpec, historical_spec, prepare_target, recent_roster, team_lookup  # noqa: E402

log = logging.getLogger(__name__)

SHOW_STATS = ["minutes", "pts", "reb", "ast", "pra", "fg3m", "stl", "blk", "tov", "fga", "fta"]
DEMO_MODELS = ["A", "B", "C"]
COLORS = {"A": "#7a7a7a", "B": "#2c7fb8", "C": "#d95f0e", "C0": "#41ab5d"}


# ----------------------------------------------------------------------------- selection

def select_demo_game(ds: Dataset, season: str | None = None, start: str = "03-01") -> tuple[int, dict]:
    """Pregame-only rule: in the last ~6 weeks of the final season, among games where every player on
    both active lists with >=15 expected minutes has >=100 prior NBA games, pick the game with the
    highest combined pregame win percentage of the two teams (ties: earliest date)."""
    games = ds.games.copy()
    season = season or sorted(games["season"].unique())[-1]
    yr = int(season[:4]) + 1
    cand = games[(games["season"] == season) & (games["game_date"] >= pd.Timestamp(f"{yr}-{start}"))]
    tg = ds.team_games[ds.team_games["season"] == season].sort_values("game_date")
    pg = ds.player_games[ds.player_games["status"] != STATUS_INACTIVE]
    rows = []
    for g in cand.itertuples():
        prior = tg[tg["game_date"] < g.game_date]
        wp = {}
        for tid in (g.home_team_id, g.away_team_id):
            t = prior[prior["team_id"] == tid]
            if len(t) < 40:
                break
            wp[tid] = float((t["margin"] > 0).mean())
        if len(wp) < 2:
            continue
        active = pg[pg["game_id"] == g.game_id]["player_id"].unique()
        hist = pg[(pg["game_date"] < g.game_date) & pg["player_id"].isin(active)]
        n_prior = hist.groupby("player_id").size()
        recent = hist.sort_values("game_date").groupby("player_id").tail(10)
        exp_min = recent.groupby("player_id")["minutes"].mean()
        core = exp_min[exp_min >= 15].index
        if len(core) < 14 or (n_prior.reindex(core).fillna(0) < 100).any():
            continue
        rows.append({"game_id": int(g.game_id), "date": g.game_date, "home": g.home, "away": g.away,
                     "home_pregame_wpct": wp[g.home_team_id], "away_pregame_wpct": wp[g.away_team_id],
                     "combined_wpct": wp[g.home_team_id] + wp[g.away_team_id], "core_players": len(core)})
    if not rows:
        raise ValueError(f"no game in {season} satisfies the pregame selection rule")
    table = pd.DataFrame(rows).sort_values(["combined_wpct", "date"], ascending=[False, True])
    best = table.iloc[0].to_dict()
    best["n_candidates"] = int(len(table))
    best["rule"] = select_demo_game.__doc__.strip().replace("\n    ", " ")
    return int(best["game_id"]), best


# ----------------------------------------------------------------------------- summaries

def player_summary(arrays: dict, rows: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    base = rows[["player_id", "player_name", "team", "opp", "is_home"]].copy().reset_index(drop=True)
    out: dict[str, np.ndarray] = {}
    q = arrays["min_q"].astype(float)
    lv = lambda x: int(np.argmin(np.abs(MINUTE_LEVELS - x)))  # noqa: E731
    out["minutes_mean"] = arrays["mean_minutes"]
    out["minutes_median"] = q[:, lv(0.5)]
    for lvl in (50, 80, 95):
        a = (1 - lvl / 100) / 2
        out[f"minutes_lo{lvl}"], out[f"minutes_hi{lvl}"] = q[:, lv(a)], q[:, lv(1 - a)]
    out["p_dnp"] = arrays["min_p0"]
    for s in [x for x in SHOW_STATS if x != "minutes"]:
        pmf = arrays[f"pmf_{s}"].astype(float)
        out[f"{s}_mean"] = arrays[f"mean_{s}"]
        out[f"{s}_median"] = M.quantile_discrete(pmf, 0.5)
        for lvl in (50, 80, 95):
            lo, hi, _ = M.central_interval(pmf, 1 - lvl / 100)
            out[f"{s}_lo{lvl}"], out[f"{s}_hi{lvl}"] = lo, hi
        for t in thresholds.get(s, []):
            out[f"P({s}>={t})"] = M.prob_at_least(pmf, t)
    return pd.concat([base, pd.DataFrame(out)], axis=1)


def _percentile_discrete(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mid-distribution percentile F(y-1) + 0.5 p(y)."""
    F = M.cdf_from_pmf(pmf)
    K = F.shape[1] - 1
    yc = np.clip(y, 0, K).astype(int)
    idx = np.arange(len(y))
    Fy = F[idx, yc]
    Fp = np.where(yc > 0, F[idx, np.maximum(yc - 1, 0)], 0.0)
    return Fp + 0.5 * (Fy - Fp)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- simulate command

def simulate_matchup(ds: Dataset, cfg: dict, home: str, away: str, date: pd.Timestamp, n_sims: int, seed: int,
                     model_name: str = "C", exclude: list[int] | None = None) -> dict[str, Path]:
    home_id, away_id = team_lookup(ds, home, date), team_lookup(ds, away, date)
    active, names = {}, {}
    for tid in (home_id, away_id):
        roster = recent_roster(ds, tid, date)
        for pid, nm in roster.items():
            if exclude and pid in exclude:
                continue
            active[pid], names[pid] = tid, nm
    spec = TargetSpec(date=date, home_team_id=home_id, away_team_id=away_id, home=home.upper(), away=away.upper(),
                      active=active, names=names)
    history, target = prepare_target(ds, spec, cfg)
    season = target["season"].iloc[0]
    if season not in set(history["season"]):
        season = sorted(history["season"].unique())[-1]   # date beyond the data: latest seasons available
    train = training_frame(history, date, season, cfg["backtest"]["train_lookback_seasons"])
    model = build_model(model_name, cfg).fit(train, date)
    res = model.simulate(target, n_sims, np.random.default_rng(seed))
    arrays = summarize(res, cfg["backtest"]["joint_samples"])
    table = player_summary(arrays, res.rows, cfg["evaluation"]["thresholds"])
    out = resolve(f"outputs/simulations/{date.date()}_{away.upper()}_at_{home.upper()}_{model_name}")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "player_distributions.csv", index=False)
    meta = {"home": home.upper(), "away": away.upper(), "date": str(date.date()), "model": model_name,
            "n_simulations": n_sims, "seed": seed, "train_date_max": str(train["game_date"].max().date()),
            "n_train_rows": int(len(train)), "roster_rule": "players active for the team in its last 10 games "
            "before the date whose latest appearance was for this team", "excluded": exclude or []}
    team_pts = arrays["team_pts"].astype(float)
    first_rows = np.unique(arrays["team_idx"], return_index=True)[1]
    meta["team_points"] = {res.rows["team"].iloc[i]: {"mean": float(team_pts[k].mean()),
                                                      "p10": float(np.quantile(team_pts[k], 0.1)),
                                                      "p90": float(np.quantile(team_pts[k], 0.9))}
                           for k, i in enumerate(first_rows)}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "players": table.to_dict("records")}, indent=1,
                                                 default=float))
    return {"csv": out / "player_distributions.csv", "json": out / "summary.json"}


# ----------------------------------------------------------------------------- historical demo

def run_demo(ds: Dataset, cfg: dict, game_id: int | None, n_sims: int, seed: int,
             backtest_run: str = "main") -> dict[str, Path]:
    if n_sims != cfg["demo"]["n_simulations"]:
        log.warning("demo n_simulations=%d differs from the configured %d", n_sims, cfg["demo"]["n_simulations"])
    selection = None
    if game_id is None:
        game_id, selection = select_demo_game(ds)
    spec = historical_spec(ds, game_id)
    out = resolve(f"outputs/demo/{game_id}")
    out.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: forecast from data strictly before the game date ---------------------------
    history, target = prepare_target(ds, spec, cfg)
    season = target["season"].iloc[0]
    train = training_frame(history, spec.date, season, cfg["backtest"]["train_lookback_seasons"])
    assert_training_before(train, spec.date)
    assert_disjoint_games(train, [game_id])
    pred_meta, hashes = {}, {}
    for mi, name in enumerate(DEMO_MODELS):
        model = build_model(name, cfg).fit(train, spec.date)
        rng = np.random.default_rng([seed, mi])
        res = model.simulate(target, n_sims, rng)
        if res.n_sims != n_sims:
            raise AssertionError("simulation count mismatch")
        arrays = summarize(res, cfg["backtest"]["joint_samples"])
        path = out / f"predictions_{name}"
        meta = {"game_id": game_id, "model": name, "n_simulations": n_sims, "seed_sequence": [seed, mi],
                "information_cutoff": f"games dated before {spec.date.date()} (cutoff = 00:00 local on game day)",
                "train_date_max": str(train["game_date"].max().date()), "n_train_rows": int(len(train)),
                "train_seasons": sorted(train["season"].unique()), "code": code_fingerprint(),
                "model_description": MODEL_DESCRIPTIONS[name], "fitted": model.describe()}
        save_predictions(path, arrays, res.rows, meta)
        hashes[name] = _sha(path.with_suffix(".npz"))
        pred_meta[name] = meta
    freeze = {"frozen_utc": str(pd.Timestamp.now(tz="UTC")), "sha256": hashes, "selection": selection}
    (out / "frozen_predictions.json").write_text(json.dumps(freeze, indent=1, default=str))
    log.info("predictions frozen for game %s: %s", game_id, hashes)

    # ---- stage 2: only now read the actual box score -----------------------------------------
    for name, h in hashes.items():
        if _sha((out / f"predictions_{name}").with_suffix(".npz")) != h:
            raise AssertionError(f"frozen predictions for {name} changed before evaluation")
    actual = ds.player_games[(ds.player_games["game_id"] == game_id) & (ds.player_games["status"] != STATUS_INACTIVE)]
    actual = actual.assign(pra=actual["pts"] + actual["reb"] + actual["ast"])
    thresholds = cfg["evaluation"]["thresholds"]
    tables, scores = {}, []
    for name in DEMO_MODELS:
        arrays, rows, _ = load_predictions(out / f"predictions_{name}")
        tab = player_summary(arrays, rows, thresholds)
        act = rows[["player_id"]].merge(actual, on="player_id", how="left")
        extra: dict[str, np.ndarray] = {}
        for s in SHOW_STATS:
            extra[f"{s}_actual"] = act[s].to_numpy()
            if s == "minutes":
                q = arrays["min_q"].astype(float)
                extra["minutes_pctile"] = np.array([np.interp(y, q[i], MINUTE_LEVELS, left=0, right=1)
                                                    for i, y in enumerate(act["minutes"])])
                extra["minutes_crps"] = M.crps_quantiles(q, MINUTE_LEVELS, act["minutes"].to_numpy())
            else:
                pmf = arrays[f"pmf_{s}"].astype(float)
                y = act[s].to_numpy().astype(int)
                extra[f"{s}_pctile"] = _percentile_discrete(pmf, y)
                extra[f"{s}_crps"] = M.crps_discrete(pmf, y)
        tab = pd.concat([pd.Series(name, index=tab.index, name="model"), tab, pd.DataFrame(extra)], axis=1)
        tables[name] = tab
        team_pts = arrays["team_pts"].astype(float)
        tidx = arrays["team_idx"]
        act_team = np.zeros(team_pts.shape[0])
        np.add.at(act_team, tidx, act["pts"].to_numpy())
        for k in range(team_pts.shape[0]):
            first = int(np.flatnonzero(tidx == k)[0])
            scores.append({"model": name, "team": rows["team"].iloc[first], "pred_mean": team_pts[k].mean(),
                           "pred_lo80": np.quantile(team_pts[k], 0.1), "pred_hi80": np.quantile(team_pts[k], 0.9),
                           "actual": act_team[k],
                           "pctile": float((team_pts[k] < act_team[k]).mean() + 0.5 * (team_pts[k] == act_team[k]).mean()),
                           "crps": float(M.crps_samples(team_pts[k][None, :], act_team[k:k + 1])[0])})
    comp = pd.concat(tables.values(), ignore_index=True)
    comp.to_csv(out / "comparison.csv", index=False)
    team_tab = pd.DataFrame(scores)
    team_tab.to_csv(out / "team_points.csv", index=False)
    figs = _demo_figures(out, tables, team_tab, ds, game_id)
    context = _backtest_context(backtest_run)
    info = {"game_id": game_id, "date": str(spec.date.date()), "home": spec.home, "away": spec.away,
            "n_simulations": n_sims, "seed": seed, "models": DEMO_MODELS, "selection": selection,
            "frozen": freeze, "prediction_meta": {k: {kk: v for kk, v in m.items() if kk != "fitted"}
                                                  for k, m in pred_meta.items()},
            "figures": {k: str(v) for k, v in figs.items()}, "backtest_context": context}
    write_manifest(out / "manifest.json", cfg, ds.manifest, extra={"kind": "historical_demo", **info})
    from nba_sim.reporting.demo_report import write_demo_report

    report = write_demo_report(info, comp, team_tab, ds, resolve("reports/historical_matchup_demo.md"))
    return {"report": report, "comparison": out / "comparison.csv", "team_points": out / "team_points.csv",
            "manifest": out / "manifest.json", "frozen": out / "frozen_predictions.json"}


def _backtest_context(run: str) -> dict | None:
    p = resolve(f"outputs/backtest/{run}/scores/player_scores.parquet")
    if not p.exists():
        return None
    s = pd.read_parquet(p, columns=["model", "stat", "cov50", "cov80", "cov95", "mass80", "crps"])
    s = s[s["stat"].isin(["minutes", "pts", "reb", "ast", "pra"])]
    g = s.groupby(["model", "stat"]).agg(cov50=("cov50", "mean"), cov80=("cov80", "mean"), cov95=("cov95", "mean"),
                                         mass80=("mass80", "mean"), crps=("crps", "mean"), n=("crps", "size"))
    return {"run": run, "table": g.reset_index().to_dict("records")}


def _demo_figures(out: Path, tables: dict, team_tab: pd.DataFrame, ds: Dataset, game_id: int) -> dict[str, Path]:
    figs = {}
    # Figures live with the reports (committed); predictions stay under outputs/.
    fig_dir = resolve(f"reports/figures/demo_{game_id}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    c = tables["C"]
    show = c[c["minutes_mean"] >= 12].sort_values(["team", "minutes_mean"], ascending=[True, False])
    # 1. points distributions per player (all three models) with the actual result marked
    n = len(show)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 2.6 * rows_n), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
        for name in DEMO_MODELS:
            arrays, rows, _ = load_predictions(out / f"predictions_{name}")
            i = int(np.flatnonzero(rows["player_id"].to_numpy() == r["player_id"])[0])
            pmf = arrays["pmf_pts"][i].astype(float)
            pmf = pmf / pmf.sum()
            ax.plot(np.arange(len(pmf)), pmf, color=COLORS[name], lw=1.6 if name == "C" else 1.1, label=f"Model {name}")
        ax.axvline(r["pts_actual"], color="black", ls="--", lw=1.2, label="actual")
        ax.set_xlim(0, max(45, r["pts_actual"] + 5))
        ax.set_title(f"{r['player_name']} ({r['team']})", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Predicted points distributions (10,000 draws each) and the actual result", fontsize=11)
    fig.tight_layout()
    figs["points_distributions"] = fig_dir / "points_distributions.png"
    fig.savefig(figs["points_distributions"], dpi=110)
    plt.close(fig)

    # 2. predicted vs actual for pts/reb/ast/minutes with 80% intervals (Model C)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, s in zip(axes, ["minutes", "pts", "reb", "ast"]):
        lo, hi = show[f"{s}_lo80"], show[f"{s}_hi80"]
        ax.errorbar(show[f"{s}_mean"], show[f"{s}_actual"], xerr=[show[f"{s}_mean"] - lo, hi - show[f"{s}_mean"]],
                    fmt="o", color=COLORS["C"], ecolor="#f4b183", ms=4, lw=1)
        lim = [0, max(show[f"{s}_hi80"].max(), show[f"{s}_actual"].max()) + 2]
        ax.plot(lim, lim, color="grey", lw=0.8)
        ax.set_xlabel(f"predicted mean (bar = 80% interval)")
        ax.set_ylabel("actual")
        ax.set_title(s)
    fig.suptitle("Model C: predicted vs actual (players with >=12 expected minutes)")
    fig.tight_layout()
    figs["pred_vs_actual"] = fig_dir / "pred_vs_actual.png"
    fig.savefig(figs["pred_vs_actual"], dpi=110)
    plt.close(fig)

    # 3. coverage in this game by model
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.25
    for k, name in enumerate(DEMO_MODELS):
        t = tables[name]
        t = t[t["minutes_mean"] >= 12]
        cov = [np.mean((t[f"{s}_actual"] >= t[f"{s}_lo80"]) & (t[f"{s}_actual"] <= t[f"{s}_hi80"]))
               for s in ["minutes", "pts", "reb", "ast", "pra"]]
        ax.bar(np.arange(5) + (k - 1) * width, cov, width, color=COLORS[name], label=f"Model {name}")
    ax.axhline(0.8, color="black", ls=":", lw=1)
    ax.set_xticks(np.arange(5), ["minutes", "pts", "reb", "ast", "pra"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share inside 80% interval")
    ax.set_title("80% interval coverage in this single game (small sample)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["coverage"] = fig_dir / "coverage.png"
    fig.savefig(figs["coverage"], dpi=110)
    plt.close(fig)

    # 4. team points by model
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, name in enumerate(DEMO_MODELS):
        t = team_tab[team_tab["model"] == name].reset_index(drop=True)
        for j, r in t.iterrows():
            y = j + (k - 1) * 0.22
            ax.plot([r["pred_lo80"], r["pred_hi80"]], [y, y], color=COLORS[name], lw=4, alpha=0.7,
                    label=f"Model {name} 80%" if j == 0 else None)
            ax.plot(r["pred_mean"], y, "o", color=COLORS[name])
    t = team_tab[team_tab["model"] == "C"].reset_index(drop=True)
    for j, r in t.iterrows():
        ax.plot(r["actual"], j, "k*", ms=12, label="actual" if j == 0 else None)
    ax.set_yticks(range(len(t)), t["team"])
    ax.set_xlabel("team points")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Team points: predicted 80% intervals vs actual")
    fig.tight_layout()
    figs["team_points"] = fig_dir / "team_points.png"
    fig.savefig(figs["team_points"], dpi=110)
    plt.close(fig)
    return figs
