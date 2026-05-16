"""Record one live match result and re-run the WC predictor.

Usage::

    python scripts/record_match.py --home Brazil --away Morocco \
        --goals-home 2 --goals-away 0 --date 2026-06-15

The script appends the result to ``data/live_results.parquet``, applies the
Bayesian shrinkage update to the saved Dixon-Coles parameters, then runs
``WorldCupSimulator.run`` with the played fixtures pinned and writes the
fresh predictions to ``simulation/results/live_predictions.parquet``.

The top 10 most-changed teams are printed so a glance at the terminal tells
you how the result moved the field.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    FALLBACK_ELO, SQUAD_VALUES, WC2026_GROUPS,
)
from src.dynamic_update import (
    append_live_result, most_changed_teams, update_and_resimulate,
)
from src.models import (
    DixonColesModel, ELOLogisticModel, EnsemblePredictor, OddsBaselineModel,
    XGBMatchPredictor,
)
from src.odds_loader import build_odds_feature, fetch_tournament_odds
from src.utils import MODELS_SAVED, SIM_RESULTS


def _load_predictor() -> EnsemblePredictor:
    dc = DixonColesModel.load(MODELS_SAVED / "dixon_coles.json")
    xgb = XGBMatchPredictor.load(MODELS_SAVED / "xgb_calibrated.joblib")
    elo_logit = ELOLogisticModel.load(MODELS_SAVED / "elo_logistic.joblib")

    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    odds_df = fetch_tournament_odds(api_key=None)
    odds_lookup = build_odds_feature(odds_df, teams)
    odds_baseline = OddsBaselineModel(odds_lookup)
    elo_snap = {t: float(FALLBACK_ELO.get(t, 1500.0)) for t in teams}
    try:
        from src.features import (
            WC2026_ADJACENT_GROUP, WC2026_BRACKET_HALF, compute_path_features,
        )
        path_features = {
            t: compute_path_features(t, WC2026_GROUPS, elo_snap, WC2026_BRACKET_HALF, WC2026_ADJACENT_GROUP)
            for t in teams
        }
    except Exception:
        path_features = None

    ensemble = EnsemblePredictor(
        dc, xgb, elo_logistic=elo_logit, odds_baseline=odds_baseline,
        weights=(0.25, 0.45, 0.10, 0.20),
    )
    ensemble.set_context(
        team_elo=elo_snap,
        team_value_eur_m={t: float(SQUAD_VALUES.get(t, 80.0)) for t in teams},
        team_odds=odds_lookup,
        path_features=path_features,
    )
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one WC 2026 match result and re-simulate.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--goals-home", type=int, required=True, dest="goals_home")
    parser.add_argument("--goals-away", type=int, required=True, dest="goals_away")
    parser.add_argument("--date", default=None, help="ISO date (default: today)")
    parser.add_argument("--stage", choices=["group", "knockout"], default="group")
    parser.add_argument("--n-sims", type=int, default=50_000)
    parser.add_argument("--shrinkage", type=float, default=0.3)
    args = parser.parse_args()

    date = pd.Timestamp(args.date) if args.date else pd.Timestamp.utcnow()
    append_live_result({
        "home": args.home, "away": args.away,
        "goals_home": args.goals_home, "goals_away": args.goals_away,
        "date": date, "stage": args.stage,
    })

    print(f"Recorded {args.home} {args.goals_home}-{args.goals_away} {args.away} ({args.stage}, {date.date()})")
    print("Loading predictor and re-simulating …")
    predictor = _load_predictor()
    out = update_and_resimulate(
        predictor, n_sims=args.n_sims, shrinkage_alpha=args.shrinkage,
    )

    out_path = SIM_RESULTS / "live_predictions.parquet"
    out["updated_df"].to_parquet(out_path, index=False)
    print(f"Wrote updated predictions → {out_path}")
    print(f"({out['n_results_applied']} live results applied so far)\n")

    # Compare against the saved pre-tournament baseline if available — the
    # update_and_resimulate baseline is just the "without-updates" run, which
    # ignores fixed_results, so it lines up cleanly with the pre-tournament
    # snapshot stored on disk.
    baseline_path = SIM_RESULTS / "baseline.parquet"
    if baseline_path.exists():
        baseline = pd.read_parquet(baseline_path)
        changed = most_changed_teams(baseline, out["updated_df"], top_n=10)
    else:
        changed = most_changed_teams(out["baseline_df"], out["updated_df"], top_n=10)

    print("Top 10 most-changed teams (since pre-tournament baseline):")
    print(f"  {'Team':<28} {'before':>9} {'after':>9} {'Δ pp':>8}")
    for _, row in changed.iterrows():
        before = row["p_champion_before"] * 100
        after = row["p_champion_after"] * 100
        delta_pp = (after - before)
        arrow = "↑" if delta_pp > 0 else "↓" if delta_pp < 0 else "—"
        print(
            f"  {row['team']:<28} {before:8.2f}% {after:8.2f}% "
            f"{arrow} {delta_pp:+6.2f}"
        )


if __name__ == "__main__":
    main()
