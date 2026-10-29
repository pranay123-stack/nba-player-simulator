"""Frozen predictive distributions.

A ``SimResult`` is reduced to a compact, outcome-free summary that is written to disk BEFORE any
actual result is read. The evaluator only ever sees these files plus the outcomes.

Stored per player-game row
  pmf_<stat>    empirical PMF on {0..K} as uint16 sample counts (tail >= K lumped into K)
  rb_<prim>     Rao-Blackwellized PMF of each primitive: mean over draws of the conditional pmf
                (NB(y | mu_s, size), or Binomial(y | T_s, share_s) for team-allocated stats), a
                smooth density used for log scores (no zero-probability artefacts)
  min_q         199 minutes quantiles (0.5%..99.5%) and min_p0 = P(minutes == 0)
  mean_<stat>, sd_<stat>
Stored per team-game
  team_pts      all simulated team points totals (aligned with the opponent's draws -> margins)
Stored per row, first J draws
  joint_<stat>  joint draws of pts/reb/ast/minutes for dependence scoring (variogram score)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_sim.models.common import PRIMS
from nba_sim.simulation.engine import SimResult

SUPPORT = {
    "pts": 80, "reb": 35, "ast": 30, "stl": 12, "blk": 15, "tov": 15, "fgm": 30, "fga": 50,
    "fg3m": 15, "fg3a": 30, "ftm": 30, "fta": 35, "oreb": 20, "dreb": 30,
    "pra": 110, "pr": 100, "pa": 95, "ra": 50,
}
JOINT_STATS = ["pts", "reb", "ast"]
MINUTE_LEVELS = np.arange(1, 200) / 200.0   # 0.5% .. 99.5%


def empirical_counts(x: np.ndarray, K: int) -> np.ndarray:
    """(n, S) integer draws -> (n, K+1) uint16 counts, tail lumped into K."""
    n, S = x.shape
    if S > np.iinfo(np.uint16).max:
        raise ValueError("too many draws for uint16 counts")
    xc = np.clip(x, 0, K).astype(np.int64)
    flat = xc + (np.arange(n)[:, None] * (K + 1))
    return np.bincount(flat.ravel(), minlength=n * (K + 1)).reshape(n, K + 1).astype(np.uint16)


def rao_blackwell_pmf(mu: np.ndarray, size: float, K: int, chunk: int = 512) -> np.ndarray:
    """Mean over draws of the NB2 pmf on {0..K}, tail mass lumped into K. mu: (n, S) or (n, 1)."""
    n = mu.shape[0]
    out = np.empty((n, K + 1), dtype=np.float32)
    r = float(size)
    for a in range(0, n, chunk):
        m = np.asarray(mu[a:a + chunk], dtype=np.float64)
        q = m / (r + m)                       # success-ish probability per draw
        logp0 = r * np.log1p(-np.minimum(q, 1 - 1e-12))
        pk = np.exp(logp0)                    # P(Y = 0 | mu_s)
        acc = np.empty((m.shape[0], K + 1))
        cum = np.zeros_like(pk)
        for k in range(K):
            acc[:, k] = pk.mean(axis=1)
            cum += pk
            pk = pk * (k + r) / (k + 1) * q
        acc[:, K] = np.clip(1.0 - cum, 0.0, 1.0).mean(axis=1)
        out[a:a + chunk] = acc
    return out


def rao_blackwell_binomial(n_trials: np.ndarray, prob: np.ndarray, K: int, chunk: int = 512) -> np.ndarray:
    """Mean over draws of the Binomial(T_s, p_s) pmf on {0..K}, tail lumped into K."""
    n = n_trials.shape[0]
    out = np.empty((n, K + 1), dtype=np.float32)
    for a in range(0, n, chunk):
        T = np.asarray(n_trials[a:a + chunk], dtype=np.float64)
        pr = np.clip(np.asarray(prob[a:a + chunk], dtype=np.float64), 0.0, 1 - 1e-12)
        pk = np.exp(T * np.log1p(-pr))
        ratio = pr / (1 - pr)
        acc = np.empty((T.shape[0], K + 1))
        cum = np.zeros_like(pk)
        for k in range(K):
            acc[:, k] = pk.mean(axis=1)
            cum += pk
            pk = pk * np.maximum(T - k, 0) / (k + 1) * ratio
        acc[:, K] = np.clip(1.0 - cum, 0.0, 1.0).mean(axis=1)
        out[a:a + chunk] = acc
    return out


def summarize(res: SimResult, joint_samples: int = 400) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    S = res.n_sims
    for stat, K in SUPPORT.items():
        x = res.stat(stat)
        out[f"pmf_{stat}"] = empirical_counts(x, K)
        out[f"mean_{stat}"] = x.mean(axis=1).astype(np.float32)
        out[f"sd_{stat}"] = x.std(axis=1).astype(np.float32)
    for s in PRIMS:
        if s in res.rb_binom:
            T, share = res.rb_binom[s]
            out[f"rb_{s}"] = rao_blackwell_binomial(T, share, SUPPORT[s])
        else:
            out[f"rb_{s}"] = rao_blackwell_pmf(res.mu[s], res.size[s], SUPPORT[s])
    m = res.minutes.astype(np.float64)
    out["min_q"] = np.quantile(m, MINUTE_LEVELS, axis=1).T.astype(np.float32)
    out["min_p0"] = (m <= 0).mean(axis=1).astype(np.float32)
    out["mean_minutes"] = m.mean(axis=1).astype(np.float32)
    out["sd_minutes"] = m.std(axis=1).astype(np.float32)
    out["team_pts"] = res.team_totals("pts").astype(np.int16)
    out["team_idx"] = res.layout.team_idx.astype(np.int32)
    out["opp_team"] = res.layout.opp_team.astype(np.int32)
    J = min(joint_samples, S)
    for stat in JOINT_STATS:
        out[f"joint_{stat}"] = res.stat(stat)[:, :J].astype(np.int16)
    out["joint_minutes"] = res.minutes[:, :J].astype(np.float16)
    out["n_sims"] = np.array([S])
    return out


def save_predictions(path: Path, summary: dict[str, np.ndarray], rows: pd.DataFrame, meta: dict) -> None:
    """Write a frozen prediction block: arrays (.npz), row identities (.parquet), metadata (.json)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **summary)
    rows.reset_index(drop=True).to_parquet(path.with_suffix(".rows.parquet"), index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))


def load_predictions(path: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    with np.load(path.with_suffix(".npz")) as z:
        arrays = {k: z[k] for k in z.files}
    rows = pd.read_parquet(path.with_suffix(".rows.parquet"))
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return arrays, rows, meta


def concat_summaries(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate chunk summaries, re-basing team indices."""
    out: dict[str, list] = {}
    offset = 0
    for p in parts:
        for k, v in p.items():
            if k in ("team_idx", "opp_team"):
                v = v + offset
            if k == "n_sims":
                out[k] = [v]
                continue
            out.setdefault(k, []).append(v)
        offset += p["team_pts"].shape[0]
    return {k: (v[0] if k == "n_sims" else np.concatenate(v, axis=0)) for k, v in out.items()}
