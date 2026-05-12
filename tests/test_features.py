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


# ---------------------------------------------------------------------------
# Task 3 — confederation calibration + odds features
# ---------------------------------------------------------------------------
def test_confederation_scalars_ordering(synthetic_results):
    """Empirical fit must respect the canonical confederation ordering.

    With this small synthetic dataset the function falls through to the
    hardcoded fallback, which is by construction ordered. Once main.py
    feeds the real ~3 000-match WC corpus the fitted magnitudes are also
    sorted into the canonical slots, so the same invariant holds.
    """
    from src.features import fit_confederation_difficulty

    scalars = fit_confederation_difficulty(synthetic_results)
    order = ["UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"]
    values = [scalars[c] for c in order]
    for a, b in zip(values, values[1:]):
        assert a >= b, f"Confederation ordering violated: {values}"


def test_confederation_scalars_range(synthetic_results):
    from src.features import fit_confederation_difficulty

    scalars = fit_confederation_difficulty(synthetic_results)
    for c, v in scalars.items():
        assert 0.30 <= v <= 1.00, f"{c} scalar out of range: {v}"


def test_odds_implied_prob_sums_to_one():
    from src.data_loader import WC2026_GROUPS
    from src.odds_loader import build_odds_feature, fetch_tournament_odds

    odds_df = fetch_tournament_odds(api_key=None)  # forces fallback
    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    feature = build_odds_feature(odds_df, teams)
    assert abs(sum(feature.values()) - 1.0) < 1e-6
    assert set(feature.keys()) == set(teams)


def test_odds_baseline_probs_sum_to_one():
    from src.data_loader import WC2026_GROUPS
    from src.models import OddsBaselineModel
    from src.odds_loader import build_odds_feature, fetch_tournament_odds

    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    odds_df = fetch_tournament_odds(api_key=None)
    odds_lookup = build_odds_feature(odds_df, teams)
    model = OddsBaselineModel(odds_lookup)

    rng = np.random.default_rng(0)
    for _ in range(10):
        a, b = rng.choice(teams, size=2, replace=False)
        probs = model.predict(a, b)
        total = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert abs(total - 1.0) < 1e-6, f"{a} vs {b} probs sum to {total}"


def test_ensemble_four_components(synthetic_results, synthetic_elo):
    """A fully-equipped 4-component ensemble must still return calibrated probs."""
    from src.data_loader import WC2026_GROUPS, FALLBACK_ELO, SQUAD_VALUES
    from src.models import (
        DixonColesModel,
        ELOLogisticModel,
        EnsemblePredictor,
        OddsBaselineModel,
        XGBMatchPredictor,
    )
    from src.odds_loader import build_odds_feature, fetch_tournament_odds
    from src.features import split_features_target

    feat = build_match_features(synthetic_results, synthetic_elo, squad_values=None)
    dc = DixonColesModel().fit(feat, time_decay=False)
    xgb = XGBMatchPredictor(n_trials=2, cv_folds=2)
    X, y, w = split_features_target(feat)
    xgb.fit(X, y, sample_weight=w)
    elo_logit = ELOLogisticModel().fit(feat)

    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    odds_df = fetch_tournament_odds(api_key=None)
    odds_lookup = build_odds_feature(odds_df, teams)
    odds_model = OddsBaselineModel(odds_lookup)

    ens = EnsemblePredictor(
        dc, xgb,
        elo_logistic=elo_logit,
        odds_baseline=odds_model,
        weights=(0.25, 0.45, 0.10, 0.20),
    )
    ens.set_context(
        team_elo={t: FALLBACK_ELO.get(t, 1500.0) for t in teams},
        team_value_eur_m={t: float(SQUAD_VALUES.get(t, 80)) for t in teams},
        team_odds=odds_lookup,
    )
    probs = ens.predict("Spain", "Jordan", neutral=True)
    s = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(s - 1.0) < 1e-6
    assert probs["lambda_home"] > 0 and probs["lambda_away"] > 0
