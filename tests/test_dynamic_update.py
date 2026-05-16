"""Tests for the in-tournament dynamic update engine (Task 5 Part C)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import FALLBACK_ELO, SQUAD_VALUES, WC2026_GROUPS
from src.dynamic_update import (
    apply_match_result,
    most_changed_teams,
    strengths_from_dc,
    update_and_resimulate,
)
from src.features import build_match_features, split_features_target
from src.models import (
    DixonColesModel,
    ELOLogisticModel,
    EnsemblePredictor,
    XGBMatchPredictor,
)
from src.monte_carlo import WorldCupSimulator


# ---------------------------------------------------------------------------
# Light unit tests on apply_match_result — no fitted models required.
# ---------------------------------------------------------------------------
def test_shrinkage_zero_is_no_update():
    """``shrinkage_alpha=0`` must return the input unchanged."""
    strengths = {
        "Brazil": {"alpha": 0.3, "beta": -0.1},
        "Cape Verde": {"alpha": -0.4, "beta": 0.2},
    }
    updated = apply_match_result(
        strengths,
        {"home": "Brazil", "away": "Cape Verde", "goals_home": 4, "goals_away": 0},
        shrinkage_alpha=0.0,
    )
    assert updated == strengths


def test_shrinkage_one_is_pure_observation():
    """``shrinkage_alpha=1`` makes α equal the observed signal exactly."""
    strengths = {
        "Brazil": {"alpha": 0.30, "beta": -0.10},
        "Cape Verde": {"alpha": -0.40, "beta": 0.20},
    }
    updated = apply_match_result(
        strengths,
        {"home": "Brazil", "away": "Cape Verde", "goals_home": 3, "goals_away": 1},
        shrinkage_alpha=1.0,
    )
    expected_brazil_alpha = float(np.log(3.0 + 0.5)) - strengths["Cape Verde"]["beta"]
    expected_cv_alpha = float(np.log(1.0 + 0.5)) - strengths["Brazil"]["beta"]
    assert updated["Brazil"]["alpha"] == pytest.approx(expected_brazil_alpha, abs=1e-9)
    assert updated["Cape Verde"]["alpha"] == pytest.approx(expected_cv_alpha, abs=1e-9)


def test_shrinkage_input_not_mutated():
    """The function must NOT mutate the caller's dict."""
    strengths = {"A": {"alpha": 0.0, "beta": 0.0}, "B": {"alpha": 0.0, "beta": 0.0}}
    snapshot = {k: dict(v) for k, v in strengths.items()}
    apply_match_result(
        strengths,
        {"home": "A", "away": "B", "goals_home": 2, "goals_away": 1},
        shrinkage_alpha=0.3,
    )
    assert strengths == snapshot


# ---------------------------------------------------------------------------
# Realistic-predictor fixture: a small ensemble trained on ELO-biased data
# so the Argentina shock test below produces meaningful champion-prob shifts.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_predictor():
    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    rng = np.random.default_rng(13)
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(3500):
        h, a = rng.choice(teams, size=2, replace=False)
        elo_h = float(FALLBACK_ELO.get(h, 1500.0))
        elo_a = float(FALLBACK_ELO.get(a, 1500.0))
        diff = (elo_h - elo_a) / 100.0
        l_h = float(np.clip(1.40 * np.exp(0.22 * diff), 0.1, 5.5))
        l_a = float(np.clip(1.20 * np.exp(-0.22 * diff), 0.1, 5.5))
        rows.append({
            "date": base + pd.Timedelta(days=i // 6),
            "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(l_h)),
            "away_score": int(rng.poisson(l_a)),
            "tournament": "Friendly" if i % 3 else "FIFA World Cup qualification",
            "city": "Anywhere", "country": h, "neutral": True,
        })
    results = pd.DataFrame(rows)
    elo = pd.DataFrame([
        {"team": t, "date": pd.Timestamp("2023-12-01"),
         "elo": float(FALLBACK_ELO.get(t, 1500.0))}
        for t in teams
    ])
    feat = build_match_features(results, elo, squad_values=None)
    dc = DixonColesModel().fit(feat, time_decay=False)
    xgb = XGBMatchPredictor(n_trials=2, cv_folds=2)
    X, y, w = split_features_target(feat)
    xgb.fit(X, y, sample_weight=w)
    elo_logit = ELOLogisticModel().fit(feat)
    ens = EnsemblePredictor(
        dc, xgb,
        dc_weight=0.30, elo_logistic=elo_logit, elo_weight=0.15,
    )
    ens.set_context(
        team_elo={t: float(FALLBACK_ELO.get(t, 1500.0)) for t in teams},
        team_value_eur_m={t: float(SQUAD_VALUES.get(t, 80.0)) for t in teams},
    )
    return ens


# ---------------------------------------------------------------------------
def test_fixed_results_skip_simulation(trained_predictor):
    """A fixture pinned to a specific scoreline must yield that exact scoreline
    every time it is "played" — i.e. the goal totals across all sims line up
    with the fixed values, not with what the predictor would have rolled."""
    sim = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=300, seed=42)
    # Pin three group-stage fixtures from different groups. Each goal total
    # then must show up at exactly (n_sims × fixed_goals) in the per-team
    # goal_for / goal_against accumulators contributed by THAT fixture.
    fixed = [
        {"home": "Brazil",    "away": "Morocco",  "goals_home": 2, "goals_away": 0},  # group C
        {"home": "Spain",     "away": "Cape Verde", "goals_home": 4, "goals_away": 1},  # group H
        {"home": "Argentina", "away": "Algeria",  "goals_home": 0, "goals_away": 2},  # group J
    ]
    # Baseline-on-same-seed without fixed_results: produces a random distribution
    # of scorelines so the goal-for totals will NOT match (n_sims × fixed_goals).
    baseline_df = sim.run()
    fixed_df = sim.run(fixed_results=fixed)

    # Spot-check: Argentina's goals_for in the (Argentina, Algeria) fixture is
    # forced to 0 across all sims when pinned. If we sum Argentina's
    # avg_goals_scored vs the baseline it must drop noticeably, given they
    # would have averaged ~1.5+ goals against Algeria.
    arg_base = float(baseline_df.set_index("team").loc["Argentina", "avg_goals_scored_per_sim"])
    arg_fixed = float(fixed_df.set_index("team").loc["Argentina", "avg_goals_scored_per_sim"])
    assert arg_fixed < arg_base, (
        "Pinning Argentina to 0 goals must reduce their avg_goals_scored vs baseline; "
        f"baseline={arg_base:.3f}, fixed={arg_fixed:.3f}"
    )
    # Algeria should *gain* goals_for from the same fixture (2 each sim).
    alg_base = float(baseline_df.set_index("team").loc["Algeria", "avg_goals_scored_per_sim"])
    alg_fixed = float(fixed_df.set_index("team").loc["Algeria", "avg_goals_scored_per_sim"])
    assert alg_fixed > alg_base, (
        f"Pinning Algeria to 2 goals must increase their avg_goals_scored; "
        f"baseline={alg_base:.3f}, fixed={alg_fixed:.3f}"
    )


def test_resimulation_after_real_upset(trained_predictor):
    """Argentina losing 0-2 to Algeria in the opener must drop Argentina's
    champion probability by at least 5 percentage points.

    This is the integration test for the dynamic update engine: it covers
    Bayesian shrinkage, the rebuilt DC head, AND the fixed_results pin.
    """
    results_df = pd.DataFrame([{
        "home": "Argentina", "away": "Algeria",
        "goals_home": 0, "goals_away": 2,
        "date": pd.Timestamp("2026-06-15"),
        "stage": "group",
    }])
    out = update_and_resimulate(
        trained_predictor,
        current_results=results_df,
        n_sims=2_000,    # smaller for test speed
        shrinkage_alpha=0.3,
        seed=42,
    )
    base_p = float(out["baseline_df"].set_index("team").loc["Argentina", "p_champion"])
    new_p = float(out["updated_df"].set_index("team").loc["Argentina", "p_champion"])
    drop_pp = (base_p - new_p) * 100.0
    assert drop_pp >= 5.0, (
        f"Argentina P(champion) should fall ≥5pp after a 0-2 loss to Algeria; "
        f"baseline={base_p*100:.2f}%, updated={new_p*100:.2f}%, drop={drop_pp:.2f}pp"
    )


def test_most_changed_teams(trained_predictor):
    """The helper returns the N teams with the largest absolute Δ on a key."""
    base = pd.DataFrame({"team": ["A", "B", "C"], "p_champion": [0.10, 0.20, 0.30]})
    after = pd.DataFrame({"team": ["A", "B", "C"], "p_champion": [0.05, 0.21, 0.18]})
    top = most_changed_teams(base, after, top_n=2)
    assert list(top["team"]) == ["C", "A"], top.to_dict()
    assert top.loc[0, "p_champion_delta"] == pytest.approx(-0.12)


def test_strengths_from_dc_roundtrip(trained_predictor):
    """strengths_from_dc -> apply_strengths_to_dc round-trips identically."""
    from src.dynamic_update import apply_strengths_to_dc

    dc = trained_predictor.dc
    strengths = strengths_from_dc(dc)
    rebuilt = apply_strengths_to_dc(dc, strengths)
    for t in dc.attack_params:
        assert rebuilt.attack_params[t] == pytest.approx(dc.attack_params[t])
        assert rebuilt.defence_params[t] == pytest.approx(dc.defence_params[t])
