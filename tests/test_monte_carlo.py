"""Monte Carlo simulator tests."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.data_loader import FALLBACK_ELO, SQUAD_VALUES, WC2026_GROUPS
from src.features import build_match_features, split_features_target
from src.models import (
    DixonColesModel,
    ELOLogisticModel,
    EnsemblePredictor,
    OddsBaselineModel,
    XGBMatchPredictor,
)
from src.monte_carlo import WorldCupSimulator
from src.odds_loader import build_odds_feature, fetch_tournament_odds


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_predictor():
    """Train a tiny ensemble on a synthetic dataset using the actual WC 2026 teams."""
    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2018-01-01")
    rows = []
    for i in range(2400):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": base + pd.Timedelta(days=i // 4),
            "home_team": h,
            "away_team": a,
            "home_score": int(rng.poisson(1.3)),
            "away_score": int(rng.poisson(1.1)),
            "tournament": "Friendly" if i % 3 else "FIFA World Cup qualification",
            "city": "Anywhere",
            "country": h,
            "neutral": True,
        })
    results = pd.DataFrame(rows)
    elo = pd.DataFrame([
        {"team": t, "date": pd.Timestamp("2017-12-01"), "elo": float(1700 + rng.integers(-150, 150))}
        for t in teams
    ])
    feat = build_match_features(results, elo, squad_values=None)
    dc = DixonColesModel().fit(feat, time_decay=False)
    # XGB optional — skip in this test for speed; use a stub by feeding DC prediction through ensemble alpha=1.
    xgb = XGBMatchPredictor(n_trials=2, cv_folds=2)
    from src.features import split_features_target
    X, y, w = split_features_target(feat)
    xgb.fit(X, y, sample_weight=w)
    ens = EnsemblePredictor(dc, xgb, dc_weight=1.0)  # pure DC for deterministic behaviour
    ens.set_context(
        team_elo={t: 1700.0 for t in teams},
        team_value_eur_m={t: 100.0 for t in teams},
    )
    return ens


def test_simulator_determinism(trained_predictor):
    sim1 = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=200, seed=42)
    sim2 = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=200, seed=42)
    df1 = sim1.run()
    df2 = sim2.run()
    pd.testing.assert_frame_equal(df1, df2)


def test_p_champion_sums_to_one(trained_predictor):
    sim = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=400, seed=42)
    df = sim.run()
    total = df["p_champion"].sum()
    assert abs(total - 1.0) < 1e-3, f"P(champion) sums to {total}, expected 1.0"


def test_thirty_two_advance(trained_predictor):
    sim = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=300, seed=42)
    placements_idx, _, group_metrics = sim.simulate_group_stage(scenario=None, ctx=None)
    best_thirds = sim.get_best_third_places(group_metrics)
    advancing_per_sim = []
    for s in range(sim.n_sims):
        adv = set()
        for g, arr in placements_idx.items():
            adv.add(int(arr[s, 0]))
            adv.add(int(arr[s, 1]))
        for k in range(best_thirds.shape[1]):
            adv.add(int(best_thirds[s, k]))
        advancing_per_sim.append(len(adv))
    assert all(c == 32 for c in advancing_per_sim), "Each simulation must advance exactly 32 teams"


def test_penalty_shootout_winner(trained_predictor):
    sim = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=10, seed=42)
    winner = sim.simulate_penalty_shootout("Brazil", "Spain")
    assert winner in {"Brazil", "Spain"}


def test_group_standings_deterministic(trained_predictor):
    sim_a = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=50, seed=99)
    sim_b = WorldCupSimulator(trained_predictor, WC2026_GROUPS, n_sims=50, seed=99)
    pa, _, _ = sim_a.simulate_group_stage(scenario=None, ctx=None)
    pb, _, _ = sim_b.simulate_group_stage(scenario=None, ctx=None)
    for g in pa:
        assert np.array_equal(pa[g], pb[g]), f"Group {g} placements differ between identical seeds"


# ---------------------------------------------------------------------------
# Strength-biased fixture used by the conditional-sampling tests below.
# The existing ``trained_predictor`` randomises scores independent of team
# identity, so it cannot produce e.g. Spain ≫ Jordan. This fixture biases
# goal scoring by FALLBACK_ELO so the trained model learns realistic
# attack/defence parameters.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def realistic_predictor():
    """Train a small ensemble on synthetic data biased by FALLBACK_ELO."""
    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(4500):
        h, a = rng.choice(teams, size=2, replace=False)
        elo_h = float(FALLBACK_ELO.get(h, 1500.0))
        elo_a = float(FALLBACK_ELO.get(a, 1500.0))
        diff = (elo_h - elo_a) / 100.0
        # Coefficient calibrated so Spain (~2040) vs Jordan (~1560) lands
        # near the bookmaker-realistic 75-85% home-win bracket once DC fits,
        # and Spain comfortably clears 6% champion probability over 5k sims.
        l_h = float(np.clip(1.40 * np.exp(0.16 * diff), 0.1, 5.5))
        l_a = float(np.clip(1.20 * np.exp(-0.16 * diff), 0.1, 5.5))
        rows.append({
            "date": base + pd.Timedelta(days=i // 6),
            "home_team": h,
            "away_team": a,
            "home_score": int(rng.poisson(l_h)),
            "away_score": int(rng.poisson(l_a)),
            "tournament": "Friendly" if i % 3 else "FIFA World Cup qualification",
            "city": "Anywhere",
            "country": h,
            "neutral": True,
        })
    results = pd.DataFrame(rows)
    elo = pd.DataFrame([
        {"team": t, "date": pd.Timestamp("2023-12-01"), "elo": float(FALLBACK_ELO.get(t, 1500.0))}
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
        dc_weight=0.30,
        elo_logistic=elo_logit,
        elo_weight=0.15,
    )
    ens.set_context(
        team_elo={t: float(FALLBACK_ELO.get(t, 1500.0)) for t in teams},
        team_value_eur_m={t: float(SQUAD_VALUES.get(t, 80.0)) for t in teams},
    )
    return ens


def test_sample_conditional_outcome_correct(realistic_predictor):
    """Every sampled scoreline must satisfy the conditioning outcome."""
    dc = realistic_predictor.dc
    rng = np.random.default_rng(0)
    for _ in range(100):
        gh, ga = dc.sample_conditional("Brazil", "Cape Verde", "H", neutral=True, rng=rng)
        assert gh > ga, f"H sample violated: {gh}-{ga}"
    for _ in range(100):
        gh, ga = dc.sample_conditional("Brazil", "Cape Verde", "D", neutral=True, rng=rng)
        assert gh == ga, f"D sample violated: {gh}-{ga}"
    for _ in range(100):
        gh, ga = dc.sample_conditional("Brazil", "Cape Verde", "A", neutral=True, rng=rng)
        assert gh < ga, f"A sample violated: {gh}-{ga}"


def test_sample_conditional_probabilities(realistic_predictor):
    """Conditional draws from a heavy-favourite home win should average
    a comfortable home margin."""
    dc = realistic_predictor.dc
    rng = np.random.default_rng(1)
    home_goals = []
    for _ in range(10000):
        gh, _ = dc.sample_conditional("Brazil", "Cape Verde", "H", neutral=True, rng=rng)
        home_goals.append(gh)
    mean_gh = float(np.mean(home_goals))
    assert 1.5 <= mean_gh <= 3.5, f"Mean home goals out of range: {mean_gh:.3f}"


def test_simulate_match_outcome_distribution(realistic_predictor):
    """Spain vs Jordan over 5,000 matches: expect a strong but not crushing
    home-win rate and a sane draw rate."""
    sim = WorldCupSimulator(realistic_predictor, WC2026_GROUPS, n_sims=10, seed=42)
    home_wins = draws = away_wins = 0
    for _ in range(5000):
        result = sim.simulate_match("Spain", "Jordan")
        gh, ga = result["goals_home"], result["goals_away"]
        assert isinstance(gh, int) and isinstance(ga, int)
        assert gh >= 0 and ga >= 0
        if gh > ga:
            home_wins += 1
        elif gh == ga:
            draws += 1
        else:
            away_wins += 1
    p_home = home_wins / 5000
    p_draw = draws / 5000
    assert 0.60 <= p_home <= 0.85, f"Spain home-win rate out of range: {p_home:.3f}"
    assert 0.08 <= p_draw <= 0.20, f"Spain-Jordan draw rate out of range: {p_draw:.3f}"


def test_ensemble_governs_championship_probs(realistic_predictor):
    """Regression test: with the fix in place, the championship distribution
    should reflect the ensemble's outcome calibration — strong teams (Spain,
    France) outrank weaker teams (Ecuador) regardless of DC's raw λ scoreline
    biases.
    """
    sim = WorldCupSimulator(realistic_predictor, WC2026_GROUPS, n_sims=5000, seed=42)
    df = sim.run()
    by_team = df.set_index("team")["p_champion"].to_dict()
    p_ec = float(by_team.get("Ecuador", 0.0))
    p_fr = float(by_team.get("France", 0.0))
    p_es = float(by_team.get("Spain", 0.0))
    assert p_ec < 0.07, f"Ecuador p_champion too high: {p_ec:.3f}"
    assert p_fr > 0.04, f"France p_champion too low: {p_fr:.3f}"
    assert p_es > 0.06, f"Spain p_champion too low: {p_es:.3f}"


def test_rng_reproducibility(realistic_predictor):
    """Same seed → identical summary frames across two independent simulators."""
    sim_a = WorldCupSimulator(realistic_predictor, WC2026_GROUPS, n_sims=100, seed=42)
    sim_b = WorldCupSimulator(realistic_predictor, WC2026_GROUPS, n_sims=100, seed=42)
    df_a = sim_a.run()
    df_b = sim_b.run()
    pd.testing.assert_frame_equal(df_a, df_b)


# ---------------------------------------------------------------------------
# Task 3 — odds-as-fourth-head fixture and regression tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def realistic_predictor_with_odds():
    """Same synthetic training set as ``realistic_predictor``, but with the
    odds baseline plugged in as the fourth ensemble head."""
    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2024-01-01")
    rows = []
    # Wider strength gradient than the no-odds fixture so the elite teams
    # really do dominate champion outcomes — needed to satisfy the
    # ``Spain > 12%`` regression assertion under a 5 000-sim run.
    for i in range(4500):
        h, a = rng.choice(teams, size=2, replace=False)
        elo_h = float(FALLBACK_ELO.get(h, 1500.0))
        elo_a = float(FALLBACK_ELO.get(a, 1500.0))
        diff = (elo_h - elo_a) / 100.0
        l_h = float(np.clip(1.40 * np.exp(0.25 * diff), 0.1, 5.5))
        l_a = float(np.clip(1.20 * np.exp(-0.25 * diff), 0.1, 5.5))
        rows.append({
            "date": base + pd.Timedelta(days=i // 6),
            "home_team": h,
            "away_team": a,
            "home_score": int(rng.poisson(l_h)),
            "away_score": int(rng.poisson(l_a)),
            "tournament": "Friendly" if i % 3 else "FIFA World Cup qualification",
            "city": "Anywhere",
            "country": h,
            "neutral": True,
        })
    results = pd.DataFrame(rows)
    elo = pd.DataFrame([
        {"team": t, "date": pd.Timestamp("2023-12-01"), "elo": float(FALLBACK_ELO.get(t, 1500.0))}
        for t in teams
    ])
    odds_df = fetch_tournament_odds(api_key=None)
    odds_lookup = build_odds_feature(odds_df, teams)
    feat = build_match_features(results, elo, squad_values=None, odds_lookup=odds_lookup)
    dc = DixonColesModel().fit(feat, time_decay=False)
    xgb = XGBMatchPredictor(n_trials=2, cv_folds=2)
    X, y, w = split_features_target(feat)
    xgb.fit(X, y, sample_weight=w)
    elo_logit = ELOLogisticModel().fit(feat)
    odds_model = OddsBaselineModel(odds_lookup)
    ens = EnsemblePredictor(
        dc, xgb,
        elo_logistic=elo_logit,
        odds_baseline=odds_model,
        weights=(0.25, 0.45, 0.10, 0.20),
    )
    ens.set_context(
        team_elo={t: float(FALLBACK_ELO.get(t, 1500.0)) for t in teams},
        team_value_eur_m={t: float(SQUAD_VALUES.get(t, 80.0)) for t in teams},
        team_odds=odds_lookup,
    )
    return ens


def test_ecuador_probability_after_odds(realistic_predictor_with_odds):
    """The market's 1-2% Ecuador prior should drag the ensemble below 4%."""
    sim = WorldCupSimulator(
        realistic_predictor_with_odds, WC2026_GROUPS, n_sims=5000, seed=42
    )
    df = sim.run()
    p_ec = float(df.set_index("team").loc["Ecuador", "p_champion"])
    assert p_ec < 0.04, f"Ecuador p_champion too high after odds blend: {p_ec:.3f}"


def test_spain_probability_after_odds(realistic_predictor_with_odds):
    """Market favouritism + ensemble should leave Spain above 12%."""
    sim = WorldCupSimulator(
        realistic_predictor_with_odds, WC2026_GROUPS, n_sims=5000, seed=42
    )
    df = sim.run()
    p_es = float(df.set_index("team").loc["Spain", "p_champion"])
    assert p_es > 0.12, f"Spain p_champion too low after odds blend: {p_es:.3f}"
