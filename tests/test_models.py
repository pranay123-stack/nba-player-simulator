"""Model fitting on real fixture rows before an origin."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from nba_sim.models import MODEL_DESCRIPTIONS, build_model
from nba_sim.models.common import PRIMS, derive, fit_priors

from conftest import MODEL_NAMES, ORIGIN


def test_build_model_unknown_raises(cfg):
    with pytest.raises(ValueError, match="unknown model"):
        build_model("X", cfg)


def test_registry_names(cfg, fitted_models):
    assert set(MODEL_DESCRIPTIONS) == set(MODEL_NAMES)
    for name, m in fitted_models.items():
        assert m.name == name
        assert m.origin == ORIGIN
    c0 = fitted_models["C0"]
    assert c0.mc["latent_factors"] is False and c0.mc["blowout_adjustment"] is False
    # The C0 ablation must not mutate the shared config.
    assert cfg["models"]["latent_factors"] is True


def test_simulate_before_fit_raises(cfg, target_rows):
    with pytest.raises(RuntimeError):
        build_model("B", cfg).simulate(target_rows, 5, np.random.default_rng(0))


def test_priors_from_training_rows(train_rows):
    pr = fit_priors(train_rows)
    assert 0.4 < pr.lg_p2 < 0.65
    assert 0.25 < pr.lg_p3 < 0.45
    assert 0.65 < pr.lg_pft < 0.9
    assert 0 < pr.lg_pi3 < 1 and 0 < pr.lg_pioreb < 0.5
    assert pr.n_ot_probs.shape == (4,)
    assert pr.n_ot_probs.sum() == pytest.approx(1.0)
    assert pr.n_ot_probs[0] > 0.8
    assert all(v > 0 for v in pr.lg_rate.values())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_nb_sizes_positive(fitted_models, name):
    size = fitted_models[name].params["size"]
    assert set(size) == set(PRIMS)
    assert all(np.isfinite(v) and v > 0 for v in size.values())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_play_probabilities_in_unit_interval(fitted_models, target_rows, name):
    from nba_sim.features.leakage_checks import pregame_view

    m = fitted_models[name]
    view = pregame_view(target_rows).sort_values(["game_id", "team_id", "player_id"]).reset_index(drop=True)
    d = m._calibrated(derive(view, m.priors, m.cfg))
    if name == "A":
        from nba_sim.models.common import design, safe_logit

        p = expit(design([safe_logit(d["play_s"])]) @ m.params["play_beta"])
    else:
        p = expit(m._Xplay(d) @ m.params["play_beta"])
    assert np.all((p > 0) & (p < 1))
    # Real rotation structure: some near-certain players and some deep-bench players.
    assert p.max() > 0.9 and p.min() < 0.5
    for key in ("p2", "p3", "pft", "pi3", "pioreb"):
        assert np.all((d[key] > 0) & (d[key] < 1)), key


@pytest.mark.parametrize("name", ["C", "C0"])
def test_team_totals_valid(fitted_models, name):
    tt = fitted_models[name].params["team_totals"]
    assert all(np.isfinite(v) and v > 0 for v in tt["phi"].values())
    R = np.asarray(tt["corr"])
    k = len(tt["stats"])
    assert R.shape == (k, k)
    np.testing.assert_allclose(R, R.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(R).min() > -1e-10
    assert np.all(np.abs(R) <= 1 + 1e-12)
    L = np.asarray(tt["chol"])
    np.testing.assert_allclose(L @ L.T, R, atol=1e-10)
    assert all(-1 < r < 1 for r in tt["rho"].values())
    assert tt["tau_mix"] >= 0 and tt["tau_shoot"] >= 0


def test_model_b_minutes_parameters(fitted_models):
    p = fitted_models["B"].params
    assert 0 < p["minutes_train_rmse"] < 15
    assert len(p["resid_pool"]) == len(p["resid_edges"]) - 1
    assert all(len(pool) > 0 for pool in p["resid_pool"])


def test_model_c_blowout_fitted_only_when_enabled(fitted_models):
    assert np.isfinite(fitted_models["C"].params["blowout_gamma"])
    assert "blowout_gamma" not in fitted_models["C0"].params


def test_describe_is_json_serializable(fitted_models):
    import json

    for m in fitted_models.values():
        json.dumps(m.describe())


def test_fit_is_deterministic(cfg, train_rows, fitted_models):
    again = build_model("B", cfg).fit(train_rows, ORIGIN)
    for s in PRIMS:
        np.testing.assert_array_equal(again.params["beta"][s], fitted_models["B"].params["beta"][s])
