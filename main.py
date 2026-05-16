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
from src.features import (
    build_match_features,
    compute_path_features,
    fit_confederation_difficulty,
    split_features_target,
    CONFEDERATION_DIFFICULTY,
    PATH_FEATURE_KEYS,
    WC2026_ADJACENT_GROUP,
    WC2026_BRACKET_HALF,
)
from src.evaluation import (
    ModelEvaluator,
    _DCAdapter,
    _ELOLogisticAdapter,
    _HomeWinBaseline,
    _UniformBaseline,
    _XGBFeatureAdapter,
)
from src.models import (
    DixonColesModel,
    ELOLogisticModel,
    EnsemblePredictor,
    OddsBaselineModel,
    XGBMatchPredictor,
    evaluate_model,
)
from src.monte_carlo import WorldCupSimulator
from src.odds_loader import build_odds_feature, fetch_tournament_odds
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

    # ---- Empirical confederation difficulty ------------------------------
    print("\n[2b] Fitting confederation difficulty scalars …")
    conf_scalars = fit_confederation_difficulty(results)
    print("     Fitted confederation difficulty scalars:")
    for conf in ("UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"):
        fitted = conf_scalars.get(conf, float("nan"))
        prior = CONFEDERATION_DIFFICULTY.get(conf, float("nan"))
        anchor = "  (anchor)" if conf == "UEFA" else ""
        print(f"       {conf:<10} {fitted:.3f}  (hardcoded was {prior:.3f}){anchor}")
    (MODELS_SAVED / "confederation_scalars.json").write_text(
        json.dumps(conf_scalars, indent=2)
    )

    # ---- Outright odds ---------------------------------------------------
    print("\n[2c] Loading pre-tournament outright odds …")
    odds_df = fetch_tournament_odds()  # API key from env, fallback otherwise
    odds_lookup = build_odds_feature(odds_df, teams_pool)
    market_top = sorted(odds_lookup.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"     Top-5 market implied probs: " +
          ", ".join(f"{t}={p*100:.1f}%" for t, p in market_top))

    # 2. Features ------------------------------------------------------------
    print("\n[3/9] Engineering features …")
    feat_path = DATA_PROCESSED / "features.parquet"
    if feat_path.exists() and args.skip_training:
        feat_df = pd.read_parquet(feat_path)
        print(f"     Loaded cached feature matrix ({len(feat_df):,} rows).")
    else:
        feat_df = build_match_features(
            results, elo, squad_values,
            odds_lookup=odds_lookup,
            confederation_scalars=conf_scalars,
        )
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
        # Hold out only WC 2018 + WC 2022 actual tournament matches; train on
        # everything else — including post-2022 matches like Euro 2024 and the
        # 2026 qualifiers. With the 180-day half-life pre-2018 data has tiny
        # weight anyway, so the practical training window is mostly recent.
        wc_holdout = (
            (feat_df["date"] >= "2018-01-01")
            & (feat_df["date"] <= "2023-01-01")
            & (feat_df.get("is_wc", 0) == 1)
        )
        train_mask = ~wc_holdout
        X, y, w = split_features_target(feat_df.loc[train_mask])
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
    print("\n[7/9] Building 4-way ensemble (DC=0.25, XGB=0.45, ELO=0.10, Odds=0.20) …")
    elo_logit_path = MODELS_SAVED / "elo_logistic.joblib"
    if args.skip_training and elo_logit_path.exists():
        elo_logit = ELOLogisticModel.load(elo_logit_path)
    else:
        wc_holdout = (
            (feat_df["date"] >= "2018-01-01")
            & (feat_df["date"] <= "2023-01-01")
            & (feat_df.get("is_wc", 0) == 1)
        )
        elo_logit = ELOLogisticModel().fit(feat_df.loc[~wc_holdout])
        elo_logit.save(elo_logit_path)

    odds_baseline = OddsBaselineModel(odds_lookup)

    # Path-difficulty features for every WC 2026 team — pre-computed once
    # and handed to the ensemble so it doesn't re-derive them per call.
    elo_snapshot = _team_elo_snapshot(elo, teams_pool)
    path_features = {
        t: compute_path_features(
            t, WC2026_GROUPS, elo_snapshot, WC2026_BRACKET_HALF, WC2026_ADJACENT_GROUP
        )
        for t in teams_pool
    }

    ensemble = EnsemblePredictor(
        dc,
        xgb,
        elo_logistic=elo_logit,
        odds_baseline=odds_baseline,
        weights=(0.25, 0.45, 0.10, 0.20),
    )
    ensemble.set_context(
        team_elo=elo_snapshot,
        team_value_eur_m=dict(zip(squad_values["team"], squad_values["value_eur_m"])),
        team_odds=odds_lookup,
        path_features=path_features,
    )

    # ---- Path difficulty table -------------------------------------------
    print("\n" + "=" * 70)
    print("GROUP PATH DIFFICULTY (hardest → easiest expected path to final)")
    print("=" * 70)
    team_group = {t: g for g, ts in WC2026_GROUPS.items() for t in ts}
    path_rows = sorted(
        ((t, path_features[t]) for t in teams_pool),
        key=lambda kv: -kv[1]["path_to_final_avg_elo"],
    )
    for team, pf in path_rows:
        print(
            f"  {team:<28} ({team_group[team]})  "
            f"group_avg_elo={pf['group_avg_elo_opponents']:6.1f}  "
            f"path_to_final_avg_elo={pf['path_to_final_avg_elo']:6.1f}"
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

    # ---- Model vs. Market benchmark --------------------------------------
    print("\n" + "=" * 70)
    print("MODEL vs. MARKET: CHAMPION PROBABILITY COMPARISON")
    print("=" * 70)
    bench = baseline[["team", "p_champion"]].copy()
    bench["market_implied_p"] = bench["team"].map(odds_lookup).fillna(0.0)
    bench["delta_pp"] = (bench["p_champion"] - bench["market_implied_p"]) * 100.0
    def _tag(d: float) -> str:
        if d > 3.0:
            return "OVER"
        if d < -3.0:
            return "UNDER"
        return "~"
    bench["over_under"] = bench["delta_pp"].map(_tag)
    bench = bench.sort_values(bench["delta_pp"].abs().name if False else "delta_pp", key=lambda s: s.abs(), ascending=False)
    bench.to_csv(SIM_RESULTS / "model_vs_market.csv", index=False)
    top_n = 12
    print(f"  {'Team':<28} {'Model':>8} {'Market':>8} {'Δ pp':>8}  Tag")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}  {'-'*5}")
    for _, row in bench.head(top_n).iterrows():
        print(
            f"  {row['team']:<28} {row['p_champion']*100:7.2f}% {row['market_implied_p']*100:7.2f}% "
            f"{row['delta_pp']:+7.2f}  {row['over_under']}"
        )

    # Spearman rank correlation + mean absolute error vs the market.
    from scipy.stats import spearmanr  # local import keeps top imports tidy
    bench_corr = bench.dropna(subset=["p_champion", "market_implied_p"])
    rho, _ = spearmanr(bench_corr["p_champion"], bench_corr["market_implied_p"])
    mae_pp = float((bench_corr["p_champion"] - bench_corr["market_implied_p"]).abs().mean() * 100.0)
    print(f"\n  Spearman rank correlation (model vs market): {rho:.3f}")
    print(f"  Mean absolute error vs market:               {mae_pp:.2f} pp")

    # ---- Model comparison table (Task 4 Part B) --------------------------
    # Build a 3-component ensemble for the "no-odds" baseline row so the
    # comparison includes both Task 2 and Task 3 ensemble variants.
    ensemble_3comp = EnsemblePredictor(
        dc, xgb,
        elo_logistic=elo_logit,
        weights=(0.30, 0.55, 0.15),
    )
    ensemble_3comp.set_context(
        team_elo=elo_snapshot,
        team_value_eur_m=dict(zip(squad_values["team"], squad_values["value_eur_m"])),
        team_odds=odds_lookup,
        path_features=path_features,
    )

    eval_models: dict[str, object] = {
        "Random Baseline": _UniformBaseline(),
        "Home Win Baseline": _HomeWinBaseline(),
        "ELO Logistic": _ELOLogisticAdapter(elo_logit, elo_snapshot),
        "Dixon-Coles": _DCAdapter(dc),
        "XGBoost (uncalibrated)": _XGBFeatureAdapter(
            xgb.raw_model_,
            lambda h, a, n: ensemble._xgb_features_for(h, a, n),
            feature_columns=xgb.feature_names_,
            feat_df=feat_df,
        ),
        "XGBoost (calibrated)": _XGBFeatureAdapter(
            xgb.calibrated_model,
            lambda h, a, n: ensemble._xgb_features_for(h, a, n),
            feature_columns=xgb.feature_names_,
            feat_df=feat_df,
        ),
        "Ensemble (3-component)": ensemble_3comp,
        "Ensemble (4-component+odds)": ensemble,
        "Betting Market": odds_baseline,
    }
    evaluator = ModelEvaluator(results, feat_df)
    comp_df = evaluator.evaluate_all(eval_models)
    comp_df.to_csv(SIM_RESULTS / "model_comparison.csv")
    evaluator.plot_calibration_grid()

    print("\n" + "=" * 70)
    n_test = int((feat_df["is_wc"].astype(int) == 1).loc[
        (feat_df["date"] >= "2018-01-01") & (feat_df["date"] <= "2023-01-01")
    ].sum())
    print(f"MODEL COMPARISON — WC 2018 + WC 2022 (n={n_test} matches)")
    print("=" * 70)
    print(f"  {'Model':<30} {'Brier↓':>8} {'LogLoss↓':>10} {'Acc↑':>8} {'RPS↓':>8} {'CalErr↓':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    for model_name, row in comp_df.iterrows():
        print(
            f"  {model_name:<30} {row['brier_score']:8.4f} {row['log_loss']:10.4f} "
            f"{row['accuracy']*100:7.1f}% {row['rps']:8.4f} {row['calibration_error']:9.4f}"
        )
    best_brier = comp_df["brier_score"].idxmin()
    best_rps = comp_df["rps"].idxmin()
    market_brier = comp_df.loc["Betting Market", "brier_score"] if "Betting Market" in comp_df.index else float("nan")
    ens_brier = comp_df.loc["Ensemble (4-component+odds)", "brier_score"] if "Ensemble (4-component+odds)" in comp_df.index else float("nan")
    elo_base_brier = comp_df.loc["ELO Logistic", "brier_score"] if "ELO Logistic" in comp_df.index else float("nan")
    print()
    print(f"  Best by Brier: {best_brier}")
    print(f"  Best by RPS:   {best_rps}")
    diff_mkt = market_brier - ens_brier
    if diff_mkt >= 0:
        print(f"  Ensemble beats market by:       {diff_mkt:.4f} Brier points")
    else:
        print(f"  Market beats ensemble by:       {-diff_mkt:.4f} Brier points")
    diff_elo = elo_base_brier - ens_brier
    if diff_elo >= 0:
        print(f"  Ensemble beats ELO baseline by: {diff_elo:.4f} Brier points")
    else:
        print(f"  ELO baseline beats ensemble by: {-diff_elo:.4f} Brier points")

    # ---- Bootstrap CIs (Part A) ------------------------------------------
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS — n_bootstrap=5000")
    print("=" * 70)
    ci_df = evaluator.bootstrap_all_metrics(n_bootstrap=5000)
    ci_df.to_csv(SIM_RESULTS / "model_comparison_with_ci.csv", index=False)
    pivot_pt = ci_df.pivot(index="model", columns="metric", values="point_estimate")
    pivot_lo = ci_df.pivot(index="model", columns="metric", values="ci_low")
    pivot_hi = ci_df.pivot(index="model", columns="metric", values="ci_high")
    # Reorder rows by Brier ascending.
    ordered_models = pivot_pt["brier"].sort_values().index
    print(f"  {'Model':<30}  {'Brier ± 95% CI':>26}    {'RPS ± 95% CI':>26}")
    print(f"  {'-'*30}  {'-'*26}    {'-'*26}")
    for m in ordered_models:
        brier_cell = (
            f"{pivot_pt.loc[m,'brier']:.4f} "
            f"[{pivot_lo.loc[m,'brier']:.3f}, {pivot_hi.loc[m,'brier']:.3f}]"
        )
        rps_cell = (
            f"{pivot_pt.loc[m,'rps']:.4f} "
            f"[{pivot_lo.loc[m,'rps']:.3f}, {pivot_hi.loc[m,'rps']:.3f}]"
        )
        print(f"  {m:<30}  {brier_cell:>26}    {rps_cell:>26}")

    # Pairwise vs the 3-component ensemble (the actual Brier winner).
    ref_name = "Ensemble (3-component)" if "Ensemble (3-component)" in comp_df.index else best_brier
    print(f"\n  Pairwise Brier comparisons vs {ref_name}:")
    pw = evaluator.pairwise_brier_vs(ref_name, n_bootstrap=5000)
    pw.to_csv(SIM_RESULTS / "pairwise_brier_vs_ensemble.csv", index=False)
    for _, row in pw.sort_values("mean_diff").iterrows():
        sig = "✓ significant" if row["significant_at_05"] else "   not sig "
        print(
            f"    vs {row['model']:<28} diff = {row['mean_diff']:+.4f}, "
            f"95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}], "
            f"p = {row['p_value']:.3f}  {sig}"
        )

    # Honesty check: does the ensemble actually beat the market with a CI
    # that excludes zero? Sign convention: mean_diff = Brier(other) − Brier(ref),
    # so a POSITIVE diff means the other model (market) has a worse Brier,
    # i.e. the reference (ensemble) beats it.
    if "Betting Market" in pw["model"].values:
        m_row = pw.loc[pw["model"] == "Betting Market"].iloc[0]
        if m_row["ci_low"] > 0:
            verdict = (
                f"Ensemble BEATS the market on Brier by {m_row['mean_diff']:.4f} "
                f"(95% CI [{m_row['ci_low']:+.3f}, {m_row['ci_high']:+.3f}], "
                f"p = {m_row['p_value']:.3f})."
            )
        elif m_row["ci_high"] < 0:
            verdict = (
                f"Market BEATS the ensemble by {abs(m_row['mean_diff']):.4f} "
                f"(95% CI [{m_row['ci_low']:+.3f}, {m_row['ci_high']:+.3f}])."
            )
        else:
            verdict = (
                f"Difference vs market is not statistically significant — "
                f"point diff {m_row['mean_diff']:+.4f}, 95% CI includes zero "
                f"[{m_row['ci_low']:+.3f}, {m_row['ci_high']:+.3f}], "
                f"p = {m_row['p_value']:.3f}. Performs COMPARABLY to the market."
            )
        print("\n  Honest verdict:")
        print(f"    {verdict}")

    # ---- WC 2018 vs WC 2022 split ----------------------------------------
    split_df = evaluator.evaluate_split(eval_models)
    split_df.to_csv(SIM_RESULTS / "model_comparison_split.csv")
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — WC 2018 vs WC 2022 SPLIT")
    print("=" * 70)
    print(f"  {'Model':<30} {'Brier 2018':>12} {'Brier 2022':>12} {'Δ':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
    for model_name, row in split_df.iterrows():
        flag = " ⚠️" if row["delta"] > 0.025 else ""
        print(
            f"  {model_name:<30} {row['brier_2018']:12.4f} {row['brier_2022']:12.4f} "
            f"{row['delta']:+8.4f}{flag}"
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
