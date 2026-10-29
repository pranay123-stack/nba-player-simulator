"""Shared session fixtures built from the committed REAL box-score fixture (no synthetic data).

The fixture is an unmodified slice of the raw source (every regular-season game 2022-10-18 ..
2023-01-15). Everything expensive (ingest, features, model fits, simulations) is built once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_sim.config import PROJECT_ROOT, load_config
from nba_sim.data.ingest import ingest
from nba_sim.features.build import build_features

FIXTURE = Path(__file__).parent / "fixtures" / "real_boxscores_2022-23_subset.csv.gz"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Model fit origin and short target window used across model / simulation tests.
ORIGIN = pd.Timestamp("2023-01-01")
TARGET_DAYS = 5
N_SIMS = 200
MODEL_NAMES = ["A", "B", "C", "C0"]


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    assert FIXTURE.exists(), f"committed real fixture missing: {FIXTURE}"
    return FIXTURE


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def ds(fixture_path):
    return ingest(None, files=[fixture_path])


@pytest.fixture(scope="session")
def features(ds, cfg) -> pd.DataFrame:
    return build_features(ds.player_games, ds.team_games, cfg)


@pytest.fixture(scope="session")
def train_rows(features) -> pd.DataFrame:
    return features[features["game_date"] < ORIGIN]


@pytest.fixture(scope="session")
def target_rows(features) -> pd.DataFrame:
    end = ORIGIN + pd.Timedelta(days=TARGET_DAYS)
    return features[(features["game_date"] >= ORIGIN) & (features["game_date"] < end)]


@pytest.fixture(scope="session")
def fitted_models(cfg, train_rows) -> dict:
    from nba_sim.models import build_model

    return {name: build_model(name, cfg).fit(train_rows, ORIGIN) for name in MODEL_NAMES}


@pytest.fixture(scope="session")
def sim_results(fitted_models, target_rows) -> dict:
    return {name: m.simulate(target_rows, N_SIMS, np.random.default_rng([2025, i]))
            for i, (name, m) in enumerate(fitted_models.items())}


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    if not sorted(RAW_DIR.glob("regular_season_box_scores_*.csv")):
        pytest.skip(f"full raw dataset not present in {RAW_DIR}")
    return RAW_DIR
