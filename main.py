"""End-to-end pipeline runner.

Usage:
    python main.py [--skip-training] [--n-sims 50000]

Steps:
    1. Load / download historical match data
    2. Build the per-match feature matrix
    3. Fit Dixon-Coles
    4. Train + calibrate XGBoost (with optuna)
    5. Build the ensemble predictor
    6. Run baseline Monte Carlo
    7. Run scenarios
    8. Save artifacts to ``simulation/results/`` and ``models/saved/``
    9. Print top-10 win probabilities and top-5 group-stage upsets
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    HOST_NATIONS,
    SQUAD_VALUES,
    WC2026_GROUPS,
    download_results,
    load_elo_ratings,
    load_squad_values,
)
from src.features import build_match_features, split_features_target
from src.models import (
    DixonColesModel,
    EnsemblePredictor,
    XGBMatchPredictor,
    evaluate_model,
)
from src.monte_carlo import WorldCupSimulator
from src.utils import (
    DATA_PROCESSED,
    MODELS_SAVED,
    SIM_RESULTS,
    load_config,
    set_global_seed,
    setup_logging,
)


def _team_elo_snapshot(elo_df: pd.DataFrame, teams: list[str]) -> dict[str, float]:
    """Return latest ELO per team — used to seed predictor context."""
    if elo_df is None or elo_df.empty:
        from src.data_loader import FALLBACK_ELO

        return {t: FALLBACK_ELO.get(t, 1500.0) for t in teams}
    snapshot: dict[str, float] = {}
    grouped = elo_df.sort_values("date").groupby("team")
    for t, sub in grouped:
        snapshot[t] = float(sub["elo"].iloc[-1])
    from src.data_loader import FALLBACK_ELO

    for t in teams:
        snapshot.setdefault(t, FALLBACK_ELO.get(t, 1500.0))
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Reuse models in models/saved instead of re-training")
    parser.add_argument("--n-sims", type=int, default=None,
                        help="Override config.simulation.n_simulations")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    set_global_seed(int(cfg["simulation"]["random_seed"]))

    n_sims = int(args.n_sims if args.n_sims is not None else cfg["simulation"]["n_simulations"])

    print("=" * 70)
    print("FIFA WC 2026 — Monte Carlo Predictor")
    print("=" * 70)

    # 1. Data ----------------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[1/9] Loading match results …")
    results = download_results()
    print(f"     {len(results):,} historical matches loaded.")

    # ELO — feed the results frame so we get a meaningful fallback if clubelo
    # is unreachable (which it usually is for international teams).
    print("\n[2/9] Building ELO ratings …")
    teams_pool = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    elo = load_elo_ratings(teams_pool, results=results)
    print(f"     ELO history rows: {len(elo):,}")

    squad_values = load_squad_values()

    # 2. Features ------------------------------------------------------------
    print("\n[3/9] Engineering features …")
    feat_path = DATA_PROCESSED / "features.parquet"
    if feat_path.exists() and args.skip_training:
        feat_df = pd.read_parquet(feat_path)
        print(f"     Loaded cached feature matrix ({len(feat_df):,} rows).")
    else:
        feat_df = build_match_features(results, elo, squad_values)
        feat_df.to_parquet(feat_path, index=False)
        print(f"     Built and cached {len(feat_df):,} rows.")

    # 3. Dixon-Coles ---------------------------------------------------------
    print("\n[4/9] Fitting Dixon-Coles …")
    dc_path = MODELS_SAVED / "dixon_coles.json"
    if args.skip_training and dc_path.exists():
        dc = DixonColesModel.load(dc_path)
        print("     Loaded saved Dixon-Coles parameters.")
    else:
        dc = DixonColesModel().fit(feat_df, time_decay=True)
        dc.save(dc_path)
        # Show a few representative attack/defence params
        sample_teams = ["Brazil", "Spain", "Germany", "USA", "Cape Verde"]
        params_preview = {t: (round(dc.attack_params.get(t, 0.0), 3),
                              round(dc.defence_params.get(t, 0.0), 3)) for t in sample_teams if t in dc.attack_params}
        print(f"     γ={dc.home_advantage:.3f}  ρ={dc.rho:.3f}")
        print(f"     Sample (α, β): {params_preview}")

    # 4. XGBoost -------------------------------------------------------------
    print("\n[5/9] Training XGBoost (with Optuna + isotonic calibration) …")
    xgb_path = MODELS_SAVED / "xgb_calibrated.joblib"
    if args.skip_training and xgb_path.exists():
        xgb = XGBMatchPredictor.load(xgb_path)
        print("     Loaded saved XGBoost predictor.")
    else:
        # Train only on matches strictly before WC 2018 to keep WC 18/22 as held-out.
        train_mask = feat_df["date"] < "2018-01-01"
        X, y, w = split_features_target(feat_df.loc[train_mask])
        # Trim optuna trial budget for snappier full pipelines (still configurable via config.yaml).
        xgb = XGBMatchPredictor()
        xgb.fit(X, y, sample_weight=w)
        xgb.save(xgb_path)

    # Evaluate on held-out WC matches
    print("\n[6/9] Evaluating on WC 2018 + WC 2022 …")
    metrics = evaluate_model(xgb, feat_df)
    if metrics:
        print(
            f"     Brier={metrics.get('brier', float('nan')):.4f}  "
            f"LogLoss={metrics.get('logloss', float('nan')):.4f}  "
            f"Acc={metrics.get('accuracy', float('nan')):.3f}  "
            f"(n={metrics.get('n_test', 0)})"
        )
    (MODELS_SAVED / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    # SHAP plot
    try:
        train_mask = feat_df["date"] < "2018-01-01"
        X_train, _, _ = split_features_target(feat_df.loc[train_mask])
        xgb.explain(X_train.head(800))
    except Exception as exc:
        print(f"     [warn] SHAP plot failed: {exc}")

    # 5. Ensemble ------------------------------------------------------------
    print("\n[7/9] Building ensemble predictor …")
    ensemble = EnsemblePredictor(dc, xgb, dc_weight=0.5)
    ensemble.set_context(
        team_elo=_team_elo_snapshot(elo, teams_pool),
        team_value_eur_m=dict(zip(squad_values["team"], squad_values["value_eur_m"])),
    )

    # 6. Baseline Monte Carlo -----------------------------------------------
    print(f"\n[8/9] Running baseline Monte Carlo ({n_sims:,} sims) …")
    sim = WorldCupSimulator(ensemble, WC2026_GROUPS, n_sims=n_sims, seed=int(cfg["simulation"]["random_seed"]))
    t1 = time.perf_counter()
    baseline = sim.run()
    sim_secs = time.perf_counter() - t1
    print(f"     Done in {sim_secs:.1f}s")
    baseline.to_parquet(SIM_RESULTS / "baseline.parquet", index=False)
    baseline.to_csv(SIM_RESULTS / "baseline.csv", index=False)

    # 7. Scenarios -----------------------------------------------------------
    print("\n[9/9] Running scenarios …")
    scenarios_out = sim.run_scenarios()
    for name, df in scenarios_out.items():
        df.to_parquet(SIM_RESULTS / f"scenario_{name}.parquet", index=False)
        df.to_csv(SIM_RESULTS / f"scenario_{name}.csv", index=False)

    # Save run metadata
    (SIM_RESULTS / "run_meta.json").write_text(
        json.dumps(
            {
                "n_sims": n_sims,
                "seed": int(cfg["simulation"]["random_seed"]),
                "scenarios": list(scenarios_out.keys()),
                "elapsed_total_seconds": round(time.perf_counter() - t0, 2),
                "elapsed_sim_seconds": round(sim_secs, 2),
            },
            indent=2,
        )
    )

    # ---- Top results -------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOP 10 CHAMPION PROBABILITIES (BASELINE)")
    print("=" * 70)
    top10 = baseline.head(10)[["team", "group", "p_champion", "p_final", "p_semi"]]
    for _, row in top10.iterrows():
        print(
            f"  {row['team']:<28} ({row['group']})  "
            f"P(champion)={row['p_champion']*100:5.2f}%  "
            f"P(final)={row['p_final']*100:5.2f}%  "
            f"P(semi)={row['p_semi']*100:5.2f}%"
        )

    # Group-stage upset risks
    upsets: list[dict] = []
    for group, members in WC2026_GROUPS.items():
        for a, b in itertools.combinations(members, 2):
            up = sim.upset_probability(a, b)
            elo_a = ensemble._team_elo.get(a, 1500.0)
            elo_b = ensemble._team_elo.get(b, 1500.0)
            fav, dog = (a, b) if elo_a >= elo_b else (b, a)
            upsets.append(
                {"group": group, "favourite": fav, "underdog": dog, "p_upset": up}
            )
    up_df = pd.DataFrame(upsets).sort_values("p_upset", ascending=False).reset_index(drop=True)
    up_df.to_parquet(SIM_RESULTS / "upset_risks.parquet", index=False)

    print("\n" + "=" * 70)
    print("TOP 5 GROUP-STAGE UPSET RISKS")
    print("=" * 70)
    for _, row in up_df.head(5).iterrows():
        print(
            f"  Group {row['group']}: {row['underdog']:<28} over {row['favourite']:<28}  "
            f"P={row['p_upset']*100:5.2f}%"
        )

    print("\nResults written to:", SIM_RESULTS)
    print("Models written to: ", MODELS_SAVED)
    print(f"\nTotal elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
