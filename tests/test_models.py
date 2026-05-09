"""Model tests — Dixon-Coles plus calibrated XGBoost."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_match_features, split_features_target
from src.models import DixonColesModel, XGBMatchPredictor, EnsemblePredictor


@pytest.fixture(scope="module")
def synthetic_match_frame() -> pd.DataFrame:
    teams = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    rng = np.random.default_rng(7)
    rows = []
    base = pd.Timestamp("2015-01-01")
    for i in range(800):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": base + pd.Timedelta(days=i),
            "home_team": h,
            "away_team": a,
            "home_score": int(rng.poisson(1.4)),
            "away_score": int(rng.poisson(1.1)),
            "tournament": rng.choice(["Friendly", "FIFA World Cup qualification"]),
            "city": "Synthetic",
            "country": h,
            "neutral": bool(rng.random() < 0.3),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_elo(synthetic_match_frame):
    teams = sorted(set(synthetic_match_frame["home_team"]).union(synthetic_match_frame["away_team"]))
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"team": t, "date": pd.Timestamp("2014-01-01"), "elo": float(1700 + rng.integers(-150, 150))}
        for t in teams
    ])


@pytest.fixture(scope="module")
def feat_df(synthetic_match_frame, synthetic_elo):
    return build_match_features(synthetic_match_frame, synthetic_elo, squad_values=None)


@pytest.fixture(scope="module")
def fitted_dc(feat_df):
    return DixonColesModel().fit(feat_df, time_decay=False)


def test_dixon_coles_lambdas_positive(fitted_dc):
    l1, l2 = fitted_dc.predict_lambda("Alpha", "Bravo", neutral=True)
    assert l1 > 0 and l2 > 0


def test_outcome_probs_sum_to_one(fitted_dc):
    probs = fitted_dc.predict_outcome_probs("Alpha", "Bravo", neutral=True)
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_dixon_coles_simulate_returns_ints(fitted_dc):
    rng = np.random.default_rng(0)
    a, b = fitted_dc.simulate_match("Alpha", "Bravo", neutral=True, rng=rng)
    assert isinstance(a, int) and isinstance(b, int)
    assert a >= 0 and b >= 0


def test_xgb_calibrated_probs_sum_to_one(feat_df):
    X, y, w = split_features_target(feat_df)
    # tiny model for tests
    m = XGBMatchPredictor(n_trials=2, cv_folds=2)
    m.fit(X, y, sample_weight=w)
    proba = m.predict_proba(X.head(5))
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert proba.shape == (5, 3)


def test_ensemble_predict(feat_df, fitted_dc):
    X, y, w = split_features_target(feat_df)
    xgb = XGBMatchPredictor(n_trials=2, cv_folds=2)
    xgb.fit(X, y, sample_weight=w)
    ens = EnsemblePredictor(fitted_dc, xgb, dc_weight=0.5)
    ens.set_context(
        team_elo={t: 1700.0 for t in fitted_dc.teams_},
        team_value_eur_m={t: 100.0 for t in fitted_dc.teams_},
    )
    pr = ens.predict("Alpha", "Bravo", neutral=True)
    s = pr["home_win"] + pr["draw"] + pr["away_win"]
    assert abs(s - 1.0) < 1e-6
    assert pr["lambda_home"] > 0 and pr["lambda_away"] > 0
