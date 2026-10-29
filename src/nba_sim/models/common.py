"""Shared estimation machinery: shrinkage priors, GLM fitters, negative-binomial dispersion.

All priors are computed from the training rows passed to ``fit`` (dates strictly before the
forecast origin), never from the full sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln, logit

# Primitive counts sampled from (conditional) negative binomials. Everything else is derived:
# fg3a ~ Bin(fga, pi3), oreb ~ Bin(reb, pi_oreb), makes ~ Bin(attempts, pct), pts = 2*fg2m + 3*fg3m + ftm.
PRIMS = ["fga", "fta", "reb", "ast", "stl", "blk", "tov"]
# Stats every model reports (all integer-valued box-score quantities).
REPORTED = ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
            "oreb", "dreb", "pra", "pr", "pa", "ra"]

EPS = 1e-9


@dataclass
class Priors:
    lg_rate: dict[str, float]          # league per-minute rates, played rows
    p_play0: float                     # play rate of players with little history
    mwp0: float                        # minutes-when-played of players with little history
    start0: float
    lg_p2: float
    lg_p3: float
    lg_pft: float
    lg_pi3: float                      # share of FGA that are threes
    lg_pioreb: float                   # share of REB that are offensive
    n_ot_probs: np.ndarray             # P(0, 1, 2, 3+ overtime periods)
    mean_abs_margin: float
    extra: dict = field(default_factory=dict)


def fit_priors(train: pd.DataFrame, weights: np.ndarray | None = None) -> Priors:
    w_all = pd.Series(np.ones(len(train)) if weights is None else weights, index=train.index)
    played = train[train["minutes"] > 0]
    w = w_all[played.index]
    tot = lambda c: float((played[c] * w).sum())  # noqa: E731
    mins = tot("minutes")
    lg_rate = {s: tot(s) / mins for s in ["fga", "fg3a", "fgm", "fg3m", "fta", "ftm",
                                          "reb", "oreb", "ast", "stl", "blk", "tov", "pts"]}
    new = train[train["n_prior"] < 10]
    new_played = new[new["minutes"] > 0]
    fg2a = tot("fga") - tot("fg3a")
    fg2m = tot("fgm") - tot("fg3m")
    return Priors(
        lg_rate=lg_rate,
        p_play0=float((new["minutes"] > 0).mean()) if len(new) else 0.6,
        mwp0=float(new_played["minutes"].mean()) if len(new_played) else 12.0,
        start0=float(new["starter"].mean()) if len(new) else 0.1,
        lg_p2=float(fg2m / fg2a),
        lg_p3=tot("fg3m") / tot("fg3a"),
        lg_pft=tot("ftm") / tot("fta"),
        lg_pi3=tot("fg3a") / tot("fga"),
        lg_pioreb=tot("oreb") / tot("reb"),
        n_ot_probs=_ot_probs(train),
        mean_abs_margin=float(train.drop_duplicates(["game_id"])["team_margin"].abs().mean()),
    )


def _ot_probs(train: pd.DataFrame) -> np.ndarray:
    tm = train.drop_duplicates(["game_id", "team_id"])["team_minutes"]
    n_ot = np.clip(np.round((tm - 240) / 25), 0, 3).astype(int)
    counts = np.bincount(n_ot, minlength=4).astype(float)
    return counts / counts.sum()


def derive(df: pd.DataFrame, pr: Priors, cfg: dict) -> pd.DataFrame:
    """Shrunk player quantities from raw decayed sums. Uses only pregame columns."""
    fc = cfg["features"]
    k_rate = fc["rate_prior_minutes"]
    k_g = fc["count_prior_games"]
    sp = fc["shooting_prior_attempts"]
    out = pd.DataFrame(index=df.index)
    out["play_s"] = (df["s_played"] + 3 * pr.p_play0) / (df["s_one"] + 3)
    out["play_l"] = (df["l_played"] + 3 * pr.p_play0) / (df["l_one"] + 3)
    out["mwp_s"] = (df["s_minutes"] + 2 * pr.mwp0) / (df["s_played"] + 2)
    out["mwp_l"] = (df["l_minutes"] + 2 * pr.mwp0) / (df["l_played"] + 2)
    out["mpg_s"] = out["play_s"] * out["mwp_s"]
    out["mpg_l"] = out["play_l"] * out["mwp_l"]
    out["start_s"] = (df["s_starter"] + pr.start0) / (df["s_one"] + 1)
    out["play_v"] = (df["v_played"] + 1 * pr.p_play0) / (df["v_one"] + 1)
    out["mwp_v"] = (df["v_minutes"] + 1 * pr.mwp0) / (df["v_played"] + 1)
    out["start_v"] = (df["v_starter"] + 0.5 * pr.start0) / (df["v_one"] + 0.5)
    out["played_last"] = (df["last_min"].fillna(0) > 0).astype(float)
    for s in PRIMS + ["pts"]:
        lg = pr.lg_rate[s]
        out[f"rate_s_{s}"] = (df[f"s_{s}"] + k_rate * lg) / (df["s_minutes"] + k_rate)
        out[f"rate_l_{s}"] = (df[f"l_{s}"] + k_rate * lg) / (df["l_minutes"] + k_rate)
        prior_pg = lg * out["mpg_l"]
        out[f"pg_s_{s}"] = (df[f"s_{s}"] + k_g * prior_pg) / (df["s_one"] + k_g)
        out[f"pg_l_{s}"] = (df[f"l_{s}"] + k_g * prior_pg) / (df["l_one"] + k_g)
    out["p2"] = (df["l_fgm"] - df["l_fg3m"] + sp["fg2"] * pr.lg_p2) / (df["l_fga"] - df["l_fg3a"] + sp["fg2"])
    out["p3"] = (df["l_fg3m"] + sp["fg3"] * pr.lg_p3) / (df["l_fg3a"] + sp["fg3"])
    out["pft"] = (df["l_ftm"] + sp["ft"] * pr.lg_pft) / (df["l_fta"] + sp["ft"])
    out["pi3"] = (df["l_fg3a"] + 40 * pr.lg_pi3) / (df["l_fga"] + 40)
    out["pioreb"] = (df["l_oreb"] + 30 * pr.lg_pioreb) / (df["l_reb"] + 30)
    key = [df["game_id"], df["team_id"]]
    team_mpg = out["mpg_s"].groupby(key).transform("sum")
    out["log_team_ratio"] = np.log(np.clip(240.0 / team_mpg, 0.5, 3.0))
    n_act = out["mpg_s"].groupby(key).transform("size")
    out["rank_frac"] = out["mpg_s"].groupby(key).rank(ascending=False, method="first") / n_act
    out["last_min_f"] = df["last_min"].fillna(out["mwp_s"] * out["play_s"])
    out["last3_min_f"] = df["last3_min"].fillna(out["mpg_s"])
    out["missed_any"] = (df["missed_games"] > 0).astype(float)
    out["log_missed"] = np.log1p(df["missed_games"].clip(lower=0).fillna(0))
    out["new_team"] = df["new_team"].astype(float)
    out["b2b"] = df["b2b"].fillna(0).astype(float)
    out["low_hist"] = (df["n_prior"] < 5).astype(float)
    # Pregame game-condition columns used by Model C, passed through unchanged.
    passthrough = [c for c in df.columns if c == "pace_dev" or c.startswith(("oppdef_", "vac_"))]
    for c in passthrough:
        out[c] = df[c]
    out["is_home"] = df["is_home"].astype(float)
    return out


# ----------------------------------------------------------------------------- GLM fitters

def fit_poisson(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, ridge: float = 1e-6,
                iters: int = 50, weights: np.ndarray | None = None) -> np.ndarray:
    """Poisson GLM with log link via IRLS (quasi-likelihood mean model for NB2 data)."""
    n, k = X.shape
    off = np.zeros(n) if offset is None else offset
    sw = np.ones(n) if weights is None else weights
    beta = np.zeros(k)
    beta[0] = np.log(max(y.mean(), EPS)) - off.mean()
    for _ in range(iters):
        eta = np.clip(X @ beta + off, -20, 8)
        mu = np.exp(eta)
        z = eta - off + (y - mu) / mu
        XtW = X.T * (mu * sw)
        new = np.linalg.solve(XtW @ X + ridge * np.eye(k), XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4, iters: int = 50,
                 weights: np.ndarray | None = None) -> np.ndarray:
    n, k = X.shape
    sw = np.ones(n) if weights is None else weights
    beta = np.zeros(k)
    for _ in range(iters):
        p = np.clip(expit(X @ beta), 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        z = X @ beta + (y - p) / w
        XtW = X.T * (w * sw)
        new = np.linalg.solve(XtW @ X + ridge * np.eye(k), XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def fit_ols(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    sq = np.sqrt(weights)
    return np.linalg.lstsq(X * sq[:, None], y * sq, rcond=None)[0]


def nb_loglik(y: np.ndarray, mu: np.ndarray, size: float, weights: np.ndarray | None = None) -> float:
    r = size
    mu = np.maximum(mu, EPS)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
    return float(np.sum(ll if weights is None else ll * weights))


def fit_nb_size(y: np.ndarray, mu: np.ndarray, lo: float = 0.2, hi: float = 5000.0,
                weights: np.ndarray | None = None) -> float:
    """MLE of the NB2 size parameter given fixed means (larger size = closer to Poisson)."""
    res = minimize_scalar(lambda lr: -nb_loglik(y, mu, np.exp(lr), weights), bounds=(np.log(lo), np.log(hi)),
                          method="bounded", options={"xatol": 1e-3})
    return float(np.exp(res.x))


def recalibrate_intercept(kind: str, X: np.ndarray, y: np.ndarray, beta: np.ndarray, mask: np.ndarray,
                          weights: np.ndarray | None = None) -> np.ndarray:
    """Re-estimate only the intercept on recent rows (slopes fixed) to track the current league level."""
    beta = beta.copy()
    if mask.sum() < 50:
        return beta
    Xr, yr = X[mask], y[mask]
    wr = np.ones(len(yr)) if weights is None else weights[mask]
    eta = Xr @ beta
    if kind == "poisson":
        beta[0] += np.log(max(np.sum(wr * yr), EPS) / max(np.sum(wr * np.exp(eta)), EPS))
    elif kind == "logistic":
        delta = 0.0
        for _ in range(30):
            pr = expit(eta + delta)
            g = np.sum(wr * (yr - pr))
            h = np.sum(wr * pr * (1 - pr))
            step = g / max(h, EPS)
            delta += step
            if abs(step) < 1e-10:
                break
        beta[0] += delta
    elif kind == "ols":
        beta[0] += np.sum(wr * (yr - eta)) / np.sum(wr)
    else:
        raise ValueError(kind)
    return beta


def recency_weights(dates: pd.Series, origin: pd.Timestamp, halflife_days: float) -> np.ndarray:
    """Exponential down-weighting of older training rows (handles league-wide drift)."""
    age = (origin - pd.to_datetime(dates)).dt.days.to_numpy(float)
    return 0.5 ** (age / halflife_days)


def safe_log(x) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), EPS))


def safe_logit(p) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4))


def design(cols: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([np.ones(len(cols[0]))] + [np.asarray(c, dtype=float) for c in cols])
