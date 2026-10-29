"""Model A: independent-player, independent-stat baseline.

Each primitive count is an NB2 draw around a per-game mean built from shrunk decayed per-game
averages (short and long half-life, combined by a Poisson GLM fit on training rows). Minutes are
drawn independently from a play/no-play mixture with a truncated-normal minutes-when-played
distribution. Stats do NOT depend on simulated minutes, and players do not depend on each other.
Box-score coherence (makes <= attempts, points = 2*FG2M + 3*FG3M + FTM) is preserved by the
shared binomial split structure.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import truncnorm

from nba_sim.features.leakage_checks import assert_training_before, pregame_view
from nba_sim.models.common import (
    PRIMS,
    Priors,
    derive,
    design,
    fit_logistic,
    fit_nb_size,
    fit_poisson,
    fit_priors,
    recalibrate_intercept,
    recency_weights,
    safe_log,
    safe_logit,
)
from nba_sim.simulation.engine import ID_COLUMNS, Layout, SimResult, sample_box, sample_overtime

log = logging.getLogger(__name__)


class BaseModel:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mc = cfg["models"]
        self.priors: Priors | None = None
        self.params: dict = {}
        self.origin: pd.Timestamp | None = None

    # -- interface -------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame, origin) -> "BaseModel":
        self.origin = pd.Timestamp(origin)
        assert_training_before(train, self.origin)
        self.w = recency_weights(train["game_date"], self.origin, self.mc["train_weight_halflife_days"])
        self.recent = (train["game_date"] >= self.origin - pd.Timedelta(days=self.mc["recalibration_days"])).to_numpy()
        self.priors = fit_priors(train, self.w)
        d = derive(train, self.priors, self.cfg)
        self._fit_shooting_calibration(train, d)
        self._fit(train, self._calibrated(d))
        return self

    def simulate(self, rows: pd.DataFrame, n_sims: int, rng: np.random.Generator) -> SimResult:
        if self.priors is None:
            raise RuntimeError(f"model {self.name} must be fit before simulate")
        if len(rows) and rows["game_date"].min() < self.origin:
            raise ValueError("target rows precede the model's training origin")
        view = pregame_view(rows).sort_values(["game_id", "team_id", "player_id"]).reset_index(drop=True)
        d = self._calibrated(derive(view, self.priors, self.cfg))
        layout = Layout.build(view, d["mpg_s"].to_numpy())
        res = self._simulate(view, d, layout, n_sims, rng)
        res.rows = view[[c for c in ID_COLUMNS if c in view.columns]].copy()
        return res

    # Shrunk percentages are recalibrated on training rows with a recency-weighted binomial GLM on
    # the logit scale. Shrinkage toward a lagging league mean otherwise under-rates high-volume
    # shooters and misses league-wide efficiency drift. Shared by every model.
    SHOOTING = {"p2": ("fg2m", "fg2a"), "p3": ("fg3m", "fg3a"), "pft": ("ftm", "fta"),
                "pi3": ("fg3a", "fga"), "pioreb": ("oreb", "reb")}

    def _fit_shooting_calibration(self, train: pd.DataFrame, d: pd.DataFrame) -> None:
        cal = {}
        cols = {"fg2m": train["fgm"] - train["fg3m"], "fg2a": train["fga"] - train["fg3a"]}
        for key, (num, den) in self.SHOOTING.items():
            made = (cols[num] if num in cols else train[num]).to_numpy(float)
            att = (cols[den] if den in cols else train[den]).to_numpy(float)
            ok = att > 0
            X = design([safe_logit(d[key].to_numpy()[ok])])
            beta = fit_logistic(X, made[ok] / att[ok], weights=att[ok] * self.w[ok])
            cal[key] = recalibrate_intercept("logistic", X, made[ok] / att[ok], beta, self.recent[ok], att[ok])
        self.shooting_cal = cal

    def _calibrated(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for key, beta in self.shooting_cal.items():
            d[key] = expit(beta[0] + beta[1] * safe_logit(d[key].to_numpy()))
        return d

    def describe(self) -> dict:
        out = {"model": self.name, "origin": str(self.origin.date()) if self.origin is not None else None,
               "shooting_calibration": _jsonable(getattr(self, "shooting_cal", {})),
               "params": _jsonable({k: v for k, v in self.params.items() if not k.startswith("resid_pool")})}
        return out

    # -- hooks -----------------------------------------------------------------------------
    def _fit(self, train: pd.DataFrame, d: pd.DataFrame) -> None:
        raise NotImplementedError

    def _simulate(self, view, d, layout, n_sims, rng) -> SimResult:
        raise NotImplementedError

    def _shooting(self, d: pd.DataFrame):
        col = lambda c: d[c].to_numpy()[:, None]  # noqa: E731
        return col("pi3"), col("pioreb"), col("p2"), col("p3"), col("pft")


class ModelA(BaseModel):
    name = "A"

    @staticmethod
    def _X(d: pd.DataFrame, s: str) -> np.ndarray:
        return design([safe_log(d[f"pg_s_{s}"]), safe_log(d[f"pg_l_{s}"])])

    def _fit(self, train, d):
        p = {"beta": {}, "size": {}}
        for s in PRIMS:
            X, y = self._X(d, s), train[s].to_numpy(float)
            beta = fit_poisson(X, y, weights=self.w)
            beta = recalibrate_intercept("poisson", X, y, beta, self.recent)
            mu = np.exp(X @ beta)
            p["beta"][s] = beta
            p["size"][s] = fit_nb_size(y, mu, weights=self.w)
        # Minutes: play probability from the shrunk play rate, minutes-when-played ~ truncated
        # normal around the shrunk decayed mean with a residual sd that depends on that mean.
        y_play = (train["minutes"] > 0).to_numpy(float)
        Xp = design([safe_logit(d["play_s"])])
        p["play_beta"] = recalibrate_intercept("logistic", Xp, y_play, fit_logistic(Xp, y_play, weights=self.w),
                                               self.recent)
        played = train["minutes"].to_numpy() > 0
        mwp = d["mwp_s"].to_numpy()[played]
        resid = train["minutes"].to_numpy()[played] - mwp
        edges = np.quantile(mwp, np.linspace(0, 1, 7))
        edges[0], edges[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(edges, mwp, side="right") - 1, 0, 5)
        p["sd_edges"] = edges
        p["sd_by_bin"] = np.array([resid[b == k].std() for k in range(6)])
        self.params = p

    def minutes_draws(self, d, n_sims, rng):
        n = len(d)
        p_play = expit(design([safe_logit(d["play_s"])]) @ self.params["play_beta"])
        mwp = d["mwp_s"].to_numpy()
        b = np.clip(np.searchsorted(self.params["sd_edges"], mwp, side="right") - 1, 0, 5)
        sd = self.params["sd_by_bin"][b]
        cap = self.mc["max_regulation_minutes"]
        lo, hi = (0.0 - mwp) / sd, (cap - mwp) / sd
        u = rng.random((n, n_sims))
        m = truncnorm.ppf(u, lo[:, None], hi[:, None], loc=mwp[:, None], scale=sd[:, None])
        play = rng.random((n, n_sims)) < p_play[:, None]
        return np.where(play, m, 0.0), p_play

    def _simulate(self, view, d, layout, n_sims, rng):
        n = len(view)
        shape = (n, n_sims)
        mu = {s: np.exp(self._X(d, s) @ self.params["beta"][s])[:, None] for s in PRIMS}
        counts = sample_box(rng, mu, self.params["size"], *self._shooting(d), shape)
        minutes, _ = self.minutes_draws(d, n_sims, rng)
        n_ot = np.zeros((layout.n_games, n_sims), dtype=np.int32)  # A does not model overtime
        return SimResult(self.name, view, minutes.astype(np.float32), counts, mu, dict(self.params["size"]),
                         layout, n_ot)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x
