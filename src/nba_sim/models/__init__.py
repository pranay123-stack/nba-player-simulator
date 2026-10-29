"""Model registry."""

from __future__ import annotations


def build_model(name: str, cfg: dict):
    from nba_sim.models.baseline import ModelA
    from nba_sim.models.minutes import ModelB
    from nba_sim.models.shared_factors import ModelC

    registry = {"A": ModelA, "B": ModelB, "C": ModelC, "C0": ModelC}
    if name not in registry:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(registry)}")
    if name == "C0":
        # Ablation: C's observable game-condition means, but no modelled dependence or blowouts.
        cfg = {**cfg, "models": {**cfg["models"], "latent_factors": False, "blowout_adjustment": False}}
        model = ModelC(cfg)
        model.name = "C0"
        return model
    return registry[name](cfg)


MODEL_DESCRIPTIONS = {
    "A": "independent players and stats; per-game NB means; minutes drawn independently",
    "B": "A + simulated minutes (play/no-play, rotation features, team-minute constraint, OT); stats NB given minutes",
    "C0": "B + pregame game conditions (pace, opponent, vacated usage, home) in the means only; "
          "players independent given minutes (ablation)",
    "C": "B + pregame game conditions + shared dependence: team totals drawn jointly (within-team correlation, "
         "cross-team correlation, team shot-mix and shooting shocks) and allocated to players; blowout minutes",
}
