"""Playing-time coherence constraints.

A simulated team-game must satisfy:
  * every player's minutes lie in [0, 48 + 5 * n_ot]
  * the team's minutes sum to exactly 5 * (48 + 5 * n_ot) = 240 + 25 * n_ot
  * at least five players record minutes

Box-score counting stats are NOT forced to fixed totals: nothing in the rules fixes a team's
rebounds or assists, so imposing a sum would be an invented constraint.
"""

from __future__ import annotations

import numpy as np


def group_starts(team_idx: np.ndarray) -> np.ndarray:
    """Start offsets of contiguous groups (rows must be sorted by team_idx)."""
    if len(team_idx) == 0:
        return np.zeros(0, dtype=int)
    if np.any(np.diff(team_idx) < 0):
        raise ValueError("rows must be sorted by team-game index")
    return np.flatnonzero(np.r_[True, team_idx[1:] != team_idx[:-1]])


def ensure_min_players(play: np.ndarray, team_idx: np.ndarray, rank_in_team: np.ndarray,
                       min_players: int = 5) -> np.ndarray:
    """Force the ``min_players`` highest-ranked players to play in team-sims with too few players."""
    starts = group_starts(team_idx)
    count = np.add.reduceat(play.astype(np.int32), starts, axis=0)        # (n_teams, S)
    short = (count < min_players)[team_idx]                                 # (n, S)
    forced = (rank_in_team < min_players)[:, None] & short
    return play | forced


def allocate_minutes(m: np.ndarray, team_idx: np.ndarray, total: np.ndarray, cap: np.ndarray,
                     iters: int = 25, tol: float = 1e-6) -> np.ndarray:
    """Rescale positive minutes within each team-sim to hit ``total`` while respecting ``cap``.

    Water-filling: players at the cap are frozen, the rest are scaled proportionally, repeated
    until the team total matches. Zero-minute players (did not play) stay at zero.

    m      : (n, S) nonnegative minutes, rows sorted by team_idx
    total  : (n_teams, S) required team minutes
    cap    : (n_teams, S) per-player maximum
    """
    m = np.asarray(m, dtype=np.float64).copy()
    starts = group_starts(team_idx)
    cap_r = cap[team_idx]
    active = m > 0
    m = np.minimum(m, cap_r)
    for _ in range(iters):
        at_cap = active & (m >= cap_r - 1e-9)
        # If the capped players alone exceed the total (e.g. six players at the cap), freezing them
        # makes the total unreachable: release them so the whole team-sim scales down.
        over = np.add.reduceat(np.where(at_cap, m, 0.0), starts, axis=0) > total + tol
        at_cap &= ~over[team_idx]
        free = active & ~at_cap
        fixed = np.add.reduceat(np.where(at_cap, m, 0.0), starts, axis=0)
        flex = np.add.reduceat(np.where(free, m, 0.0), starts, axis=0)
        need = total - fixed
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(flex > 0, need / flex, 1.0)
        m = np.where(free, m * scale[team_idx], m)
        m = np.minimum(m, cap_r)
        err = np.abs(np.add.reduceat(m, starts, axis=0) - total)
        if err.max() < tol:
            break
    return m


def team_minute_residual(m: np.ndarray, team_idx: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.add.reduceat(m, group_starts(team_idx), axis=0) - total
