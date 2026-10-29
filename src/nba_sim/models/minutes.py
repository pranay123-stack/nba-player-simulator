"""Model B: Model A plus explicit, uncertain playing time.

Minutes are simulated first:
  1. P(enters game | active) from a logistic model on rotation features (decayed play and start
     rates, minutes level, rank on the active list, games missed, new team, back-to-back, how much
     of the team's usual minutes are available on today's active list).
  2. Minutes | plays = linear mean on the same features + a residual drawn from the empirical
     training residuals in the same predicted-minutes bin (heteroscedastic, non-Gaussian).
  3. Overtime periods are drawn from their training frequency; minutes are then water-filled so
     each team sums to 240 + 25*OT with no player above 48 + 5*OT (see simulation.constraints).
Then each primitive count is NB2 with mean exp(b0 + b1*log rate_short + b2*log rate_long +
gamma*log minutes), fit on played training rows using actual minutes as exposure. Simulated zero
minutes give exactly zero stats. Players remain conditionally independent given minutes, except for
the negative dependence induced by the shared team-minutes budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from nba_sim.models.baseline import BaseModel
from nba_sim.models.common import (
    PRIMS,
    design,
    fit_logistic,
    fit_nb_size,
    fit_ols,
    fit_poisson,
    recalibrate_intercept,
    safe_log,
    safe_logit,
)
from nba_sim.simulation.constraints import allocate_minutes, ensure_min_players
from nba_sim.simulation.engine import SimResult, sample_box, sample_overtime

N_RESID_BINS = 8


class ModelB(BaseModel):
    name = "B"

    # -- design matrices ---------------------------------------------------------------------
    @staticmethod
    def _Xplay(d: pd.DataFrame) -> np.ndarray:
        return design([
            safe_logit(d["play_s"]), safe_logit(d["play_l"]), safe_logit(d["play_v"]), d["played_last"],
            d["mwp_s"] / 48, d["mwp_v"] / 48, d["start_s"], d["start_v"],
            d["rank_frac"], d["log_team_ratio"], d["missed_any"], d["log_missed"], d["new_team"],
            d["b2b"], d["low_hist"],
        ])

    @staticmethod
    def _Xmin(d: pd.DataFrame) -> np.ndarray:
        return design([
            d["mwp_s"], d["mwp_l"], d["mwp_v"], d["last_min_f"], d["last3_min_f"], d["start_s"], d["start_v"],
            d["mwp_s"] * d["log_team_ratio"], d["rank_frac"], d["missed_any"], d["log_missed"],
            d["new_team"], d["b2b"], d["low_hist"],
        ])

    def _Xstat(self, d: pd.DataFrame, s: str, minutes: np.ndarray) -> np.ndarray:
        return design([safe_log(d[f"rate_s_{s}"]), safe_log(d[f"rate_l_{s}"]), safe_log(minutes)])

    # -- fitting -----------------------------------------------------------------------------
    def _fit(self, train, d):
        p = {}
        y_play = (train["minutes"].to_numpy() > 0).astype(float)
        Xp = self._Xplay(d)
        p["play_beta"] = recalibrate_intercept("logistic", Xp, y_play, fit_logistic(Xp, y_play, weights=self.w),
                                               self.recent)
        played = y_play > 0
        Xm = self._Xmin(d)[played]
        y_min = train["minutes"].to_numpy()[played]
        p["min_beta"] = recalibrate_intercept("ols", Xm, y_min, fit_ols(Xm, y_min, weights=self.w[played]),
                                              self.recent[played])
        pred = Xm @ p["min_beta"]
        self._fit_residual_pool(p, pred, y_min - pred)
        self._store_minutes_diag(p, pred, y_min)
        self._fit_stats(p, train[played], d[played], train["minutes"].to_numpy()[played], self.w[played],
                        self.recent[played])
        self.params = p

    @staticmethod
    def _fit_residual_pool(p, pred, resid):
        edges = np.quantile(pred, np.linspace(0, 1, N_RESID_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, N_RESID_BINS - 1)
        p["resid_edges"] = edges
        p["resid_pool"] = [np.sort(resid[b == k]).astype(np.float32) for k in range(N_RESID_BINS)]

    @staticmethod
    def _store_minutes_diag(p, pred, y):
        p["minutes_train_rmse"] = float(np.sqrt(np.mean((y - pred) ** 2)))

    def _fit_stats(self, p, tr, d, minutes, w, recent):
        p["beta"], p["size"] = {}, {}
        for s in PRIMS:
            X, y = self._Xstat(d, s, minutes), tr[s].to_numpy(float)
            beta = recalibrate_intercept("poisson", X, y, fit_poisson(X, y, weights=w), recent)
            p["beta"][s] = beta
            p["size"][s] = fit_nb_size(y, np.exp(X @ beta), weights=w)

    # -- simulation --------------------------------------------------------------------------
    def draw_minutes(self, d, layout, n_sims, rng, resid_pool=None, resid_edges=None):
        """Play/no-play, minutes-when-playing, overtime, then the team-minute constraint."""
        p = self.params
        n = len(d)
        shape = (n, n_sims)
        p_play = expit(self._Xplay(d) @ p["play_beta"])
        play = rng.random(shape) < p_play[:, None]
        play = ensure_min_players(play, layout.team_idx, layout.rank_in_team, self.mc["min_players_on_court"])
        mean = self._Xmin(d) @ p["min_beta"]
        pool = resid_pool if resid_pool is not None else p["resid_pool"]
        edges = resid_edges if resid_edges is not None else p["resid_edges"]
        b = np.clip(np.searchsorted(edges, mean, side="right") - 1, 0, N_RESID_BINS - 1)
        resid = np.empty(shape, dtype=np.float64)
        for k in range(N_RESID_BINS):
            rows = np.flatnonzero(b == k)
            if len(rows):
                resid[rows] = rng.choice(pool[k], size=(len(rows), n_sims), replace=True)
        n_ot = sample_overtime(rng, self.priors.n_ot_probs, layout.n_games, n_sims)
        cap = (self.mc["max_regulation_minutes"] + self.mc["overtime_period_minutes"] * n_ot)[layout.team_game]
        total = 5.0 * cap
        raw = np.clip(mean[:, None] + resid, self.mc["min_minutes_if_played"], cap[layout.team_idx])
        m = allocate_minutes(np.where(play, raw, 0.0), layout.team_idx, total, cap)
        return m, n_ot, {"p_play": p_play, "mean_if_play": mean, "cap": cap, "total": total}

    def stat_means(self, d, minutes, extra_log=None):
        mu = {}
        for s in PRIMS:
            beta = self.params["beta"][s]
            base = beta[0] + beta[1] * safe_log(d[f"rate_s_{s}"]) + beta[2] * safe_log(d[f"rate_l_{s}"])
            if extra_log is not None:
                base = base + extra_log[s]
            with np.errstate(divide="ignore"):
                lm = np.where(minutes > 0, np.log(np.maximum(minutes, 1e-9)), -np.inf)
            mu[s] = np.where(minutes > 0, np.exp(base[:, None] + beta[3] * lm), 0.0)
        return mu

    def _simulate(self, view, d, layout, n_sims, rng):
        m, n_ot, _ = self.draw_minutes(d, layout, n_sims, rng)
        mu = self.stat_means(d, m)
        counts = sample_box(rng, mu, self.params["size"], *self._shooting(d), m.shape)
        return SimResult(self.name, view, m.astype(np.float32), counts,
                         {s: v.astype(np.float32) for s, v in mu.items()}, dict(self.params["size"]),
                         layout, n_ot)
