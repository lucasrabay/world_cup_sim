"""Monte Carlo simulator tests."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.data_loader import WC2026_GROUPS
from src.features import build_match_features
from src.models import DixonColesModel, EnsemblePredictor, XGBMatchPredictor
from src.monte_carlo import WorldCupSimulator


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
