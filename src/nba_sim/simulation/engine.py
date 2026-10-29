"""Simulation primitives and the SimResult container shared by all models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_sim.models.common import PRIMS

ID_COLUMNS = ["game_id", "game_date", "season", "team_id", "team", "opp_team_id", "opp", "is_home",
              "player_id", "player_name"]


@dataclass
class Layout:
    """Row/team/game indexing for a block of target rows sorted by (game_id, team_id)."""

    team_idx: np.ndarray      # (n,) team-game index per row
    game_idx: np.ndarray      # (n,) game index per row
    team_game: np.ndarray     # (n_teams,) game index per team-game
    opp_team: np.ndarray      # (n_teams,) index of the opposing team-game
    rank_in_team: np.ndarray  # (n,) 0 = highest expected minutes
    n_teams: int
    n_games: int

    @classmethod
    def build(cls, view: pd.DataFrame, expected_minutes: np.ndarray) -> "Layout":
        keys = view[["game_id", "team_id"]].to_numpy()
        if len(view) and np.any(np.lexsort((keys[:, 1], keys[:, 0])) != np.arange(len(view))):
            raise ValueError("view must be sorted by (game_id, team_id)")
        team_codes, team_idx = np.unique(keys[:, 0] * 10**10 + keys[:, 1], return_inverse=True)
        games, game_idx = np.unique(keys[:, 0], return_inverse=True)
        team_game = np.zeros(len(team_codes), dtype=int)
        team_game[team_idx] = game_idx
        # Each game must contain exactly two team-games: the opponent is the other one.
        per_game = np.bincount(team_game, minlength=len(games))
        if np.any(per_game != 2):
            raise ValueError("every game in a simulation block needs both teams' active lists")
        first = np.searchsorted(team_game, np.arange(len(games)))
        opp_team = np.where(np.arange(len(team_codes)) == first[team_game], first[team_game] + 1, first[team_game])
        order = pd.DataFrame({"t": team_idx, "e": -np.asarray(expected_minutes)})
        rank = order.groupby("t")["e"].rank(method="first").to_numpy().astype(int) - 1
        return cls(team_idx, game_idx, team_game, opp_team, rank, len(team_codes), len(games))


@dataclass
class SimResult:
    """Joint draws for a block of player-games. All arrays are (n_rows, n_sims)."""

    model: str
    rows: pd.DataFrame
    minutes: np.ndarray
    counts: dict[str, np.ndarray]
    mu: dict[str, np.ndarray]                   # conditional NB means of the primitives
    size: dict[str, float]                      # NB size of the primitives
    layout: Layout
    n_ot: np.ndarray                            # (n_games, S)
    meta: dict = field(default_factory=dict)
    # Primitives drawn by team-total allocation: stat -> (T per row, share per row), both (n, S);
    # the conditional distribution is Binomial(T, share) instead of NB(mu, size).
    rb_binom: dict = field(default_factory=dict)

    @property
    def n_sims(self) -> int:
        return self.minutes.shape[1]

    def stat(self, name: str) -> np.ndarray:
        c = self.counts
        if name in c:
            return c[name]
        if name == "minutes":
            return self.minutes
        if name == "fgm":
            return c["fg2m"] + c["fg3m"]
        if name == "dreb":
            return c["reb"] - c["oreb"]
        if name == "pts":
            return 2 * c["fg2m"] + 3 * c["fg3m"] + c["ftm"]
        if name == "pra":
            return self.stat("pts") + c["reb"] + c["ast"]
        if name == "pr":
            return self.stat("pts") + c["reb"]
        if name == "pa":
            return self.stat("pts") + c["ast"]
        if name == "ra":
            return c["reb"] + c["ast"]
        raise KeyError(name)

    def team_totals(self, name: str) -> np.ndarray:
        x = self.stat(name)
        out = np.zeros((self.layout.n_teams, x.shape[1]), dtype=np.float64)
        np.add.at(out, self.layout.team_idx, x)
        return out


def sample_nb(rng: np.random.Generator, mu: np.ndarray, size: float | np.ndarray, shape) -> np.ndarray:
    """Negative binomial (NB2) draws via the gamma-Poisson mixture. mu = 0 gives exactly 0."""
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), shape)
    lam = rng.gamma(shape=size, scale=1.0, size=shape) * (mu / size)
    return rng.poisson(lam).astype(np.int32)


def sample_box(rng: np.random.Generator, mu: dict[str, np.ndarray], size: dict[str, float],
               pi3, pioreb, p2, p3, pft, shape) -> dict[str, np.ndarray]:
    """Draw a coherent box score: primitives, then binomial splits and makes."""
    counts = {s: sample_nb(rng, mu[s], size[s], shape) for s in PRIMS}
    fga = counts["fga"]
    counts["fg3a"] = rng.binomial(fga, np.broadcast_to(pi3, shape)).astype(np.int32)
    fg2a = fga - counts["fg3a"]
    counts["fg2m"] = rng.binomial(fg2a, np.broadcast_to(p2, shape)).astype(np.int32)
    counts["fg3m"] = rng.binomial(counts["fg3a"], np.broadcast_to(p3, shape)).astype(np.int32)
    counts["ftm"] = rng.binomial(counts["fta"], np.broadcast_to(pft, shape)).astype(np.int32)
    counts["oreb"] = rng.binomial(counts["reb"], np.broadcast_to(pioreb, shape)).astype(np.int32)
    return counts


def sample_overtime(rng: np.random.Generator, probs: np.ndarray, n_games: int, n_sims: int) -> np.ndarray:
    return rng.choice(len(probs), size=(n_games, n_sims), p=probs).astype(np.int32)
