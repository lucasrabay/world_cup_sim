"""Feature-engineering tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_match_features, split_features_target


@pytest.fixture
def synthetic_results() -> pd.DataFrame:
    teams = ["Brazil", "Argentina", "Spain", "Germany", "USA"]
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2018-01-01")
    for i in range(400):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append(
            {
                "date": base + pd.Timedelta(days=i),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(1.4)),
                "away_score": int(rng.poisson(1.1)),
                "tournament": rng.choice(["Friendly", "FIFA World Cup"]),
                "city": "Anywhere",
                "country": h,
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_elo(synthetic_results) -> pd.DataFrame:
    teams = sorted(set(synthetic_results["home_team"]).union(synthetic_results["away_team"]))
    rng = np.random.default_rng(1)
    rows = []
    for t in teams:
        rows.append({"team": t, "date": pd.Timestamp("2017-01-01"), "elo": float(1700 + rng.integers(-200, 200))})
    return pd.DataFrame(rows)


def test_features_no_nan(synthetic_results, synthetic_elo):
    df = build_match_features(synthetic_results, synthetic_elo, squad_values=None)
    for col in FEATURE_COLUMNS:
        assert df[col].notna().all(), f"NaN found in column {col}"
    assert len(df) == len(synthetic_results)


def test_outcome_encoding(synthetic_results, synthetic_elo):
    df = build_match_features(synthetic_results, synthetic_elo, squad_values=None)
    assert set(df["outcome"].unique()).issubset({0, 1, 2})


def test_time_decay_monotonic(synthetic_results, synthetic_elo):
    """Older matches should receive smaller sample weights."""
    df = build_match_features(synthetic_results, synthetic_elo, squad_values=None)
    df_sorted = df.sort_values("date")
    weights = df_sorted["sample_weight"].to_numpy()
    # Weights should be non-decreasing as we move forward in time.
    diffs = np.diff(weights)
    assert (diffs >= -1e-9).all(), "sample_weight should be (weakly) increasing in date"


def test_split_features_target(synthetic_results, synthetic_elo):
    df = build_match_features(synthetic_results, synthetic_elo, squad_values=None)
    X, y, w = split_features_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(w)
