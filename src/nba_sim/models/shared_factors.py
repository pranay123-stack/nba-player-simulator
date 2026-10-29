"""Model C: Model B plus shared game-level factors.

Observable (pregame) game conditions enter each primitive's mean through a Poisson GLM:
expected pace, the opponent's allowed-per-possession rate for that stat, the team's "vacated"
production given today's active list, and home court.

Unobservable shared conditions are modelled through TEAM TOTALS, re-estimated at every refit from
residuals of the mean model evaluated at ACTUAL minutes (so they describe dependence beyond
playing time):

* For FGA, FTA, REB, STL, BLK, TOV the team total is drawn first,
      T_ts = round(M_ts + sqrt(phi_s * M_ts) * eps_ts),   M_ts = sum of players' conditional means,
  with eps jointly Gaussian: within a team the stats are correlated (estimated correlation matrix
  of standardized team residuals — possession structure: e.g. more FGA with more REB, fewer with
  more TOV), and across the two teams of a game each stat has its own correlation rho_s (pace,
  officiating). phi_s < (sum of player variances / M) means teammates compete for the stat
  (negative dependence); phi_s above it means a shared positive shock. One mechanism covers both.
* AST is drawn after makes: its team total is centred on M_ast + b * (team FGM residual), because
  assists require made baskets (estimated slope b; residual dispersion phi_ast).
* Each team total is allocated to players by a multinomial with shares proportional to
  mu_is * exp(sigma_s * xi_is - sigma_s^2 / 2); the share shocks restore each player's marginal
  overdispersion (sigma_s^2 = log(1 + 1/size_s)) without changing the team total.
* Two latent team shocks on the logit scale, shared by teammates and part of the same joint
  Gaussian as the totals: a shot-mix shock on each player's 3PA share (game plan / opponent
  scheme: more threes means fewer twos and fewer free throws) and a shooting shock on 2P/3P make
  probabilities. Their correlations with the totals are corrected for binomial dilution.
* Blowouts: minutes move from high-minute to low-minute players as the simulated margin grows
  (slope fit on training residuals). The margin comes from a first pass of the same draw; stats are
  then redrawn with the adjusted minutes and the same shocks.

Pairwise teammate correlations are therefore pair-specific: they depend on each player's share of
the team's expected production and minutes, and differ by stat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sim.models.common import PRIMS, design, fit_ols, safe_log
from nba_sim.models.minutes import N_RESID_BINS, ModelB
from nba_sim.simulation.constraints import allocate_minutes, group_starts
from nba_sim.simulation.engine import SimResult, sample_nb


def _team_moments(df: pd.DataFrame, a: np.ndarray, mu: np.ndarray, var_i: np.ndarray) -> dict[str, float]:
    """Team- and game-level residual moments of one stat (rows: played player-games)."""
    tmp = pd.DataFrame({"g": df["game_id"].to_numpy(), "t": df["team_id"].to_numpy(),
                        "a": a, "a2": a * a, "m": mu, "m2": mu * mu, "v": var_i})
    tg = tmp.groupby(["g", "t"], sort=True).agg(A=("a", "sum"), Q=("a2", "sum"), M=("m", "sum"),
                                                M2=("m2", "sum"), V=("v", "sum")).reset_index()
    first = tg.groupby("g").nth(0).set_index("g")
    second = tg.groupby("g").nth(1).set_index("g")
    j = first.add_suffix("1").join(second.add_suffix("2"), how="inner")
    z1 = j["A1"] / np.sqrt(j["M1"].clip(lower=1e-9))
    z2 = j["A2"] / np.sqrt(j["M2"].clip(lower=1e-9))
    return {
        "within_num": float((tg["A"] ** 2 - tg["Q"]).sum()),
        "within_den": float((tg["M"] ** 2 - tg["M2"]).sum()),
        "cross_num": float((j["A1"] * j["A2"]).sum()),
        "cross_den": float((j["M1"] * j["M2"]).sum()),
        "team_var_ratio": float((tg["A"] ** 2).sum() / tg["V"].sum()),   # var(T - M) / sum player var
        "phi": float((tg["A"] ** 2).sum() / tg["M"].sum()),              # var(T - M) / M
        "rho": float(np.corrcoef(z1, z2)[0, 1]) if len(j) > 10 else 0.0,
    }


def _binomial_shift_logit(p: np.ndarray, delta: np.ndarray) -> np.ndarray:
    lp = np.log(p) - np.log1p(-p)
    return 1.0 / (1.0 + np.exp(-(lp + delta)))


class ModelC(ModelB):
    name = "C"

    def _game_cols(self, d: pd.DataFrame, s: str) -> list[np.ndarray]:
        return [np.nan_to_num(d["pace_dev"].to_numpy()), np.nan_to_num(d[f"oppdef_{s}"].to_numpy()),
                np.clip(np.nan_to_num(d[f"vac_{s}"].to_numpy()), -1, 1), d["is_home"].to_numpy(float)]

    def _Xstat(self, d, s, minutes):
        return design([safe_log(d[f"rate_s_{s}"]), safe_log(d[f"rate_l_{s}"]), safe_log(minutes),
                       *self._game_cols(d, s)])

    # -- fitting -----------------------------------------------------------------------------
    def _fit(self, train, d):
        super()._fit(train, d)
        p = self.params
        played = train["minutes"].to_numpy() > 0
        tr = train[played]
        dp = d[played]
        mins = tr["minutes"].to_numpy()
        mu = {s: np.exp(self._Xstat(dp, s, mins) @ p["beta"][s]) for s in PRIMS}
        resid = {s: tr[s].to_numpy(float) - mu[s] for s in PRIMS}
        p["size_marginal"] = dict(p["size"])
        mom = {s: _team_moments(tr, resid[s], mu[s], mu[s] + mu[s] ** 2 / p["size"][s]) for s in PRIMS}
        p["moments"] = mom
        self._fit_team_totals(p, tr, dp, mu)
        if self.mc.get("blowout_adjustment", True):
            self._fit_blowout(p, train, d)

    def _fit_team_totals(self, p, tr, d, mu):
        """Team-total dispersions, cross-team correlations and the joint within-team correlation of
        total residuals plus two latent logit-scale shocks (3PA share 'mix', make probability 'shoot')."""
        g = tr["game_id"].to_numpy()
        t = tr["team_id"].to_numpy()
        keys = pd.MultiIndex.from_arrays([g, t])
        tsum = lambda x: pd.Series(np.asarray(x, dtype=float), index=keys).groupby(level=[0, 1]).sum()  # noqa: E731
        T = {s: tsum(tr[s]) for s in PRIMS}
        M = {s: tsum(mu[s]) for s in PRIMS}
        fga = tr["fga"].to_numpy(float)
        fg3a = tr["fg3a"].to_numpy(float)
        fg2a = fga - fg3a
        p2, p3, pi3 = d["p2"].to_numpy(), d["p3"].to_numpy(), d["pi3"].to_numpy()
        fgm_res = tr["fgm"].to_numpy(float) - fg2a * p2 - fg3a * p3
        X = tsum(fgm_res)
        resid = {s: T[s] - M[s] for s in PRIMS}
        b_ast = float((resid["ast"] * X).sum() / (X * X).sum())
        resid["ast"] = resid["ast"] - b_ast * X
        phi, z = {}, {}
        for s in PRIMS:
            phi[s] = float((resid[s] ** 2).sum() / M[s].sum())
            z[s] = resid[s] / np.sqrt(phi[s] * M[s].clip(lower=1e-9))
        # Latent logit shocks shared by teammates: variance from teammate covariance moments,
        # correlation with the totals corrected for binomial noise (dilution).
        lat = {}
        for name, e, w in (("mix", fg3a - fga * pi3, fga * pi3 * (1 - pi3)),
                           ("shoot", fgm_res, fg2a * p2 * (1 - p2) + fg3a * p3 * (1 - p3))):
            m = _team_moments(tr, e, w, w)
            tau2 = max(m["within_num"] / m["within_den"], 0.0)
            E, W = tsum(e), tsum(w)
            zz = E / np.sqrt(W + tau2 * W ** 2)
            share = float(np.mean(tau2 * W / (1 + tau2 * W)))
            lat[name] = {"tau": float(np.sqrt(tau2)), "z": zz, "share": share}
        names = list(PRIMS) + ["mix", "shoot"]
        Z = pd.DataFrame({**z, "mix": lat["mix"]["z"], "shoot": lat["shoot"]["z"]})[names]
        R = Z.corr().to_numpy().copy()
        for i, nm in enumerate(names):
            if nm in lat:
                sc = 1.0 / np.sqrt(max(lat[nm]["share"], 1e-3))
                R[i, :] *= sc
                R[:, i] *= sc
        R = np.clip(R, -0.95, 0.95)
        np.fill_diagonal(R, 1.0)
        w_, V = np.linalg.eigh(R)                      # nearest PSD correlation matrix
        R = V @ np.diag(np.clip(w_, 1e-4, None)) @ V.T
        dd = np.sqrt(np.diag(R))
        R = R / np.outer(dd, dd)
        Zr = pd.DataFrame(z).reset_index(names=["g", "t"])
        first = Zr.groupby("g").nth(0).set_index("g")
        second = Zr.groupby("g").nth(1).set_index("g")
        j = first.add_suffix("_1").join(second.add_suffix("_2"), how="inner")
        rho = {s: float(np.clip(np.corrcoef(j[f"{s}_1"], j[f"{s}_2"])[0, 1], -0.95, 0.95)) for s in PRIMS}
        p["team_totals"] = {
            "stats": names, "phi": phi, "rho": rho, "corr": R, "chol": np.linalg.cholesky(R),
            "ast_fgm_slope": b_ast,
            "share_sigma": {s: float(np.sqrt(np.log1p(1.0 / p["size"][s]))) for s in PRIMS},
            "team_var_ratio": {s: p["moments"][s]["team_var_ratio"] for s in PRIMS},
            "tau_mix": lat["mix"]["tau"], "tau_shoot": lat["shoot"]["tau"],
        }

    def _fit_blowout(self, p, train, d):
        """Slope of minutes residuals on (|margin| - E|margin|) x role, on played training rows."""
        played = train["minutes"].to_numpy() > 0
        mean = (self._Xmin(d) @ p["min_beta"])[played]
        y = train["minutes"].to_numpy()[played]
        excess = train["team_margin"].abs().to_numpy()[played] - self.priors.mean_abs_margin
        role = (mean - 24.0) / 10.0
        x = excess * role
        gamma = float(fit_ols(x[:, None], y - mean, weights=self.w[played])[0])
        p["blowout_gamma"] = gamma
        # Residual pool net of the blowout effect, so C does not double count that variance.
        resid = y - mean - gamma * x
        edges = p["resid_edges"]
        b = np.clip(np.searchsorted(edges, mean, side="right") - 1, 0, N_RESID_BINS - 1)
        p["resid_pool_c"] = [np.sort(resid[b == k]).astype(np.float32) for k in range(N_RESID_BINS)]

    # -- simulation --------------------------------------------------------------------------
    def _simulate(self, view, d, layout, n_sims, rng):
        p = self.params
        shape = (len(view), n_sims)
        use_blow = self.mc.get("blowout_adjustment", True) and "blowout_gamma" in p
        use_latent = self.mc.get("latent_factors", True)
        m, n_ot, info = self.draw_minutes(d, layout, n_sims, rng,
                                          resid_pool=p["resid_pool_c"] if use_blow else None)
        tt = p["team_totals"]
        k = len(tt["stats"])
        e_team = np.einsum("ij,tsj->tsi", tt["chol"], rng.standard_normal((layout.n_teams, n_sims, k)))
        shocks = {
            "team": {s: e_team[:, :, i] for i, s in enumerate(tt["stats"])},
            "game": {s: rng.standard_normal((layout.n_games, n_sims)) for s in tt["stats"]},
            "share": {s: rng.standard_normal(shape) for s in PRIMS},
        }
        extra = self._extra_log(d)
        out = self._draw_box(d, layout, m, shocks, extra, use_latent, rng)
        if use_blow:
            c = out["counts"]
            pts = 2 * c["fg2m"] + 3 * c["fg3m"] + c["ftm"]
            team_pts = np.zeros((layout.n_teams, n_sims))
            np.add.at(team_pts, layout.team_idx, pts)
            margin = team_pts - team_pts[layout.opp_team]
            excess = np.abs(margin) - self.priors.mean_abs_margin
            role = (info["mean_if_play"] - 24.0) / 10.0
            adj = m + p["blowout_gamma"] * excess[layout.team_idx] * role[:, None]
            cap = info["cap"]
            adj = np.where(m > 0, np.clip(adj, self.mc["min_minutes_if_played"], cap[layout.team_idx]), 0.0)
            m = allocate_minutes(adj, layout.team_idx, info["total"], cap)
            out = self._draw_box(d, layout, m, shocks, extra, use_latent, rng)
        res = SimResult(self.name, view, m.astype(np.float32), out["counts"],
                        {s: v.astype(np.float32) for s, v in out["mu"].items()}, out["size"], layout, n_ot)
        res.rb_binom = out["rb_binom"]
        return res

    def _draw_box(self, d, layout, minutes, shocks, extra, use_latent, rng):
        p = self.params
        shape = minutes.shape
        mu = self.stat_means(d, minutes, extra_log=extra)
        counts, rb_binom, nb_mu, size = {}, {}, {}, {}
        if not use_latent:   # ablation switch: independent NB draws around the C means
            for s in PRIMS:
                size[s], nb_mu[s] = p["size_marginal"][s], mu[s]
                counts[s] = sample_nb(rng, mu[s], size[s], shape)
        else:
            for s in PRIMS:
                if s != "ast":
                    counts[s], rb_binom[s] = self._allocate(mu[s], s, layout, shocks, rng, 0.0)
        pi3, pior, p2, p3, pft = self._shooting(d)
        p2s, p3s = p2, p3
        if use_latent:
            tt = p["team_totals"]
            sh = tt["tau_shoot"] * shocks["team"]["shoot"][layout.team_idx]
            p2s = _binomial_shift_logit(p2, sh)
            p3s = _binomial_shift_logit(p3, sh)
            pi3 = _binomial_shift_logit(pi3, tt["tau_mix"] * shocks["team"]["mix"][layout.team_idx])
        fga = counts["fga"]
        counts["fg3a"] = rng.binomial(fga, np.broadcast_to(pi3, shape)).astype(np.int32)
        fg2a = fga - counts["fg3a"]
        counts["fg2m"] = rng.binomial(fg2a, np.broadcast_to(p2s, shape)).astype(np.int32)
        counts["fg3m"] = rng.binomial(counts["fg3a"], np.broadcast_to(p3s, shape)).astype(np.int32)
        counts["ftm"] = rng.binomial(counts["fta"], np.broadcast_to(pft, shape)).astype(np.int32)
        counts["oreb"] = rng.binomial(counts["reb"], np.broadcast_to(pior, shape)).astype(np.int32)
        if use_latent:
            # Assists follow made baskets: centre the team total on the realised FGM residual.
            res = counts["fg2m"] + counts["fg3m"] - fg2a * p2 - counts["fg3a"] * p3
            starts = group_starts(layout.team_idx)
            fgm_res = np.add.reduceat(res, starts, axis=0)
            shift = p["team_totals"]["ast_fgm_slope"] * fgm_res
            counts["ast"], rb_binom["ast"] = self._allocate(mu["ast"], "ast", layout, shocks, rng, shift)
        return {"counts": counts, "rb_binom": rb_binom, "mu": nb_mu, "size": size}

    def _allocate(self, mu_s, s, layout, shocks, rng, shift):
        """Team total first, then multinomial allocation with lognormal share shocks."""
        tt = self.params["team_totals"]
        starts = group_starts(layout.team_idx)
        M = np.add.reduceat(mu_s, starts, axis=0)                               # (n_teams, S)
        rho = tt["rho"][s]
        g = shocks["game"][s][layout.team_game]
        second = (np.arange(layout.n_teams) % 2 == 1)[:, None]
        eps = np.sqrt(abs(rho)) * np.where(second, np.sign(rho), 1.0) * g + np.sqrt(1 - abs(rho)) * shocks["team"][s]
        T = np.maximum(np.rint(M + shift + np.sqrt(tt["phi"][s] * M) * eps), 0).astype(np.int64)
        sig = tt["share_sigma"][s]
        w = mu_s * np.exp(sig * shocks["share"][s] - 0.5 * sig * sig)
        W = np.add.reduceat(w, starts, axis=0)
        share = np.divide(w, W[layout.team_idx], out=np.zeros_like(w), where=W[layout.team_idx] > 0)
        pos = np.arange(len(layout.team_idx)) - starts[layout.team_idx]
        pv = np.zeros((layout.n_teams, share.shape[1], int(pos.max()) + 1))
        pv[layout.team_idx, :, pos] = share
        tot = pv.sum(axis=2, keepdims=True)
        pv = np.divide(pv, tot, out=np.zeros_like(pv), where=tot > 0)   # padding columns stay zero
        T = np.where(W > 0, T, 0)
        draws = rng.multinomial(T, pv)                                           # (n_teams, S, k_max)
        counts = draws[layout.team_idx, :, pos].astype(np.int32)
        # Rao-Blackwellization: Y_i | T, shares ~ Binomial(T, share_i).
        return counts, (T[layout.team_idx].astype(np.int32), share.astype(np.float32))

    def _extra_log(self, d):
        """Game-condition contribution to log means (columns 4.. of the stat design)."""
        out = {}
        for s in PRIMS:
            beta = self.params["beta"][s]
            out[s] = sum(b * c for b, c in zip(beta[4:], self._game_cols(d, s)))
        return out
