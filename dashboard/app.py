"""Live, interactive Streamlit dashboard for the WC 2026 predictor.

Six pages:

1. Tournament Overview     — current 50k baseline + sortable champion table.
2. Scenario Builder        — sliders that re-run a 5 000-sim mini Monte Carlo
                              in roughly half a second.
3. Bracket Visualisation   — most-likely knockout bracket with hover probs.
4. Model vs Market         — bootstrap-CI table + scatter plot of disagreements.
5. Live Tournament Mode    — record real results, trigger the dynamic update
                              pipeline, see how predictions evolve.
6. Methodology & Caveats   — what the model does well, what it doesn't, and
                              the honest bootstrap-CI verdict.

All charts use Plotly's dark template; configuration lives in
``.streamlit/config.toml``.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the project importable regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    FALLBACK_ELO, HOST_NATIONS, SQUAD_VALUES, WC2026_GROUPS,
)
from src.dynamic_update import (  # noqa: E402
    LIVE_RESULTS_PATH, append_live_result, most_changed_teams,
    update_and_resimulate,
)
from src.models import (  # noqa: E402
    DixonColesModel, ELOLogisticModel, EnsemblePredictor, OddsBaselineModel,
    XGBMatchPredictor,
)
from src.monte_carlo import WorldCupSimulator  # noqa: E402
from src.odds_loader import build_odds_feature, fetch_tournament_odds  # noqa: E402
from src.utils import MODELS_SAVED, SIM_RESULTS  # noqa: E402

PLOTLY_TEMPLATE = "plotly_dark"


CONFEDERATION = {
    **{t: "UEFA" for t in [
        "England", "France", "Spain", "Germany", "Portugal", "Netherlands",
        "Belgium", "Norway", "Croatia", "Switzerland", "Austria", "Sweden",
        "Turkey", "Bosnia and Herzegovina", "Scotland", "Czechia",
    ]},
    **{t: "CONMEBOL" for t in [
        "Brazil", "Argentina", "Colombia", "Uruguay", "Ecuador", "Paraguay",
    ]},
    **{t: "CONCACAF" for t in ["USA", "Mexico", "Canada", "Panama", "Haiti", "Curacao"]},
    **{t: "CAF" for t in [
        "Senegal", "Morocco", "Ivory Coast", "Egypt", "Algeria", "Ghana",
        "Tunisia", "South Africa", "Cape Verde", "DR Congo",
    ]},
    **{t: "AFC" for t in [
        "Japan", "South Korea", "Australia", "Saudi Arabia", "Qatar", "Iran",
        "Jordan", "Iraq", "Uzbekistan",
    ]},
    **{t: "OFC" for t in ["New Zealand"]},
}


# ---------------------------------------------------------------------------
# Data loaders — cached so the app stays snappy across reruns.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_baseline() -> pd.DataFrame:
    p = SIM_RESULTS / "baseline.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_live_predictions() -> pd.DataFrame:
    p = SIM_RESULTS / "live_predictions.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_live_results() -> pd.DataFrame:
    if LIVE_RESULTS_PATH.exists():
        return pd.read_parquet(LIVE_RESULTS_PATH)
    return pd.DataFrame(columns=["home", "away", "goals_home", "goals_away", "date", "stage"])


@st.cache_data(show_spinner=False)
def _load_run_meta() -> dict:
    p = SIM_RESULTS / "run_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_ci_table() -> pd.DataFrame:
    p = SIM_RESULTS / "model_comparison_with_ci.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_pairwise() -> pd.DataFrame:
    p = SIM_RESULTS / "pairwise_brier_vs_ensemble.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_model_vs_market() -> pd.DataFrame:
    p = SIM_RESULTS / "model_vs_market.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_resource(show_spinner="Loading ensemble predictor …")
def _load_predictor() -> EnsemblePredictor | None:
    """Re-build the ensemble from saved artifacts. Used by the scenario
    builder and the live-update pipeline."""
    try:
        dc = DixonColesModel.load(MODELS_SAVED / "dixon_coles.json")
        xgb = XGBMatchPredictor.load(MODELS_SAVED / "xgb_calibrated.joblib")
        elo_logit = ELOLogisticModel.load(MODELS_SAVED / "elo_logistic.joblib")
    except Exception:
        return None
    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})
    odds_df = fetch_tournament_odds(api_key=None)
    odds_lookup = build_odds_feature(odds_df, teams)
    odds_baseline = OddsBaselineModel(odds_lookup)
    elo_snap = {t: float(FALLBACK_ELO.get(t, 1500.0)) for t in teams}
    # Reuse the path features module — fall back gracefully if anything is
    # missing (the dashboard is read-only, we never refit anything here).
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
        dc, xgb,
        elo_logistic=elo_logit,
        odds_baseline=odds_baseline,
        weights=(0.25, 0.45, 0.10, 0.20),
    )
    ensemble.set_context(
        team_elo=elo_snap,
        team_value_eur_m={t: float(SQUAD_VALUES.get(t, 80.0)) for t in teams},
        team_odds=odds_lookup,
        path_features=path_features,
    )
    return ensemble


@st.cache_resource(show_spinner=False)
def _get_simulator() -> WorldCupSimulator | None:
    ens = _load_predictor()
    if ens is None:
        return None
    return WorldCupSimulator(ens, WC2026_GROUPS, n_sims=5000, seed=42)


def _empty_state(msg: str) -> None:
    st.warning(msg)
    st.info("Run `python main.py` first to populate the simulation results.")


# ---------------------------------------------------------------------------
# Page 1 — Tournament Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    st.header("Tournament Overview")

    df = _load_baseline()
    if df.empty:
        _empty_state("No baseline simulation found.")
        return

    meta = _load_run_meta()
    n_sims = meta.get("n_sims", "?")
    elapsed = meta.get("elapsed_sim_seconds", "?")
    cols = st.columns(4)
    cols[0].metric("Simulations", f"{n_sims:,}" if isinstance(n_sims, int) else n_sims)
    cols[1].metric("Last sim runtime", f"{elapsed}s" if elapsed != "?" else "?")
    cols[2].metric("Teams", str(len(df)))
    favourite = df.iloc[0]
    cols[3].metric("Favourite", f"{favourite['team']}", f"{favourite['p_champion']*100:.1f}%")

    # Top 16 with Monte Carlo standard error (Wilson-ish: sqrt(p(1-p)/n)).
    top16 = df.head(16).copy()
    n = int(meta.get("n_sims", 50_000)) or 50_000
    top16["se"] = (top16["p_champion"] * (1 - top16["p_champion"]) / n) ** 0.5

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top16["p_champion"], y=top16["team"], orientation="h",
        error_x=dict(type="data", array=1.96 * top16["se"], color="#9ca3af"),
        marker=dict(color=top16["p_champion"], colorscale="Viridis", showscale=False),
        hovertemplate="%{y}: %{x:.2%}  ±%{error_x.array:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title="Top 16 — Champion Probability (95% Monte Carlo error bars)",
        template=PLOTLY_TEMPLATE, yaxis={"categoryorder": "total ascending"},
        height=520, margin=dict(l=120, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    show_cols = [
        "team", "group", "p_champion", "p_final", "p_semi", "p_qf", "p_r16",
    ]
    st.dataframe(
        df[show_cols]
            .assign(**{c: (df[c] * 100).round(2) for c in show_cols[2:]}),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# Page 2 — Interactive Scenario Builder
# ---------------------------------------------------------------------------
def _format_chip(mod: dict) -> str:
    parts = [mod["team"]]
    if mod.get("elo_delta", 0):
        parts.append(f"ELO {mod['elo_delta']:+d}")
    if mod.get("attack_mult", 1.0) != 1.0:
        parts.append(f"att×{mod['attack_mult']:.2f}")
    if mod.get("def_mult", 1.0) != 1.0:
        parts.append(f"def×{mod['def_mult']:.2f}")
    return "  ·  ".join(parts)


def page_scenario_builder() -> None:
    st.header("Scenario Builder")
    st.caption(
        "Tweak any team's ELO, attack λ, or defensive strength, then re-run the "
        "Monte Carlo. The 5 000-sim run completes in under a second."
    )

    baseline = _load_baseline()
    if baseline.empty:
        _empty_state("No baseline simulation found.")
        return

    if "scenario_mods" not in st.session_state:
        st.session_state.scenario_mods = []
    if "scenario_result" not in st.session_state:
        st.session_state.scenario_result = None

    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Modifications")
        team = st.selectbox("Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
        elo_delta = st.slider("ELO adjustment", -200, 200, 0, step=10)
        attack_mult = st.slider("Attack λ multiplier", 0.30, 1.50, 1.00, step=0.05)
        def_mult = st.slider("Defence strength multiplier", 0.30, 1.50, 1.00, step=0.05)

        b_add, b_clear = st.columns(2)
        if b_add.button("Add to scenario", use_container_width=True):
            st.session_state.scenario_mods.append({
                "team": team,
                "elo_delta": int(elo_delta),
                "attack_mult": float(attack_mult),
                "def_mult": float(def_mult),
            })
        if b_clear.button("Clear scenario", use_container_width=True):
            st.session_state.scenario_mods = []
            st.session_state.scenario_result = None

        if st.session_state.scenario_mods:
            st.markdown("**Active modifications:**")
            for i, mod in enumerate(st.session_state.scenario_mods):
                cols = st.columns([5, 1])
                cols[0].markdown(f"• {_format_chip(mod)}")
                if cols[1].button("×", key=f"rm_{i}"):
                    st.session_state.scenario_mods.pop(i)
                    st.rerun()
        else:
            st.info("No modifications yet — add one above.")

        run_clicked = st.button(
            "▶ Run simulation", type="primary", use_container_width=True,
            disabled=not st.session_state.scenario_mods,
        )

    with right:
        st.subheader("Results")
        if run_clicked and st.session_state.scenario_mods:
            sim = _get_simulator()
            if sim is None:
                st.error("Predictor unavailable — run `python main.py` to train it.")
            else:
                t0 = time.perf_counter()
                with st.spinner("Simulating 5 000 tournaments …"):
                    res = sim.run_scenario_live(st.session_state.scenario_mods, n_sims=5000)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                st.session_state.scenario_result = res
                st.success(f"Done in {elapsed_ms:.0f} ms")

        if st.session_state.scenario_result is None:
            st.info("Add at least one modification and click *Run simulation*.")
            return

        scenario_df = st.session_state.scenario_result
        merged = baseline[["team", "p_champion"]].rename(columns={"p_champion": "baseline"}) \
            .merge(
                scenario_df[["team", "p_champion"]].rename(columns={"p_champion": "scenario"}),
                on="team", how="left",
            )
        merged["delta_pp"] = (merged["scenario"] - merged["baseline"]) * 100.0

        top = merged.sort_values("baseline", ascending=False).head(16)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline", x=top["team"], y=top["baseline"] * 100,
            marker_color="#6b7280",
        ))
        fig.add_trace(go.Bar(
            name="Scenario", x=top["team"], y=top["scenario"] * 100,
            marker_color="#3b82f6",
        ))
        fig.update_layout(
            barmode="group", template=PLOTLY_TEMPLATE,
            title="P(champion) — top 16 baseline vs scenario",
            yaxis_title="P(champion) %", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        most_changed = merged.assign(abs_delta=lambda d: d["delta_pp"].abs()) \
            .sort_values("abs_delta", ascending=False).head(5)
        st.markdown("**Most affected teams:**")
        for _, row in most_changed.iterrows():
            sign = "↑" if row["delta_pp"] > 0 else "↓" if row["delta_pp"] < 0 else "—"
            colour = "#22c55e" if row["delta_pp"] > 0 else "#ef4444" if row["delta_pp"] < 0 else "#9ca3af"
            st.markdown(
                f"<div style='font-family:monospace'>"
                f"{sign} <b>{row['team']}</b> &nbsp; "
                f"<span style='color:{colour}'>{row['delta_pp']:+.2f}pp</span> "
                f"({row['baseline']*100:.2f}% → {row['scenario']*100:.2f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Full table"):
            st.dataframe(
                merged.assign(
                    baseline_pct=(merged["baseline"] * 100).round(2),
                    scenario_pct=(merged["scenario"] * 100).round(2),
                    delta_pp=merged["delta_pp"].round(2),
                )[["team", "baseline_pct", "scenario_pct", "delta_pp"]] \
                    .sort_values("delta_pp", key=lambda s: s.abs(), ascending=False),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Page 3 — Bracket visualisation
# ---------------------------------------------------------------------------
def page_bracket() -> None:
    st.header("Bracket Visualisation")
    df = _load_baseline()
    if df.empty:
        _empty_state("No baseline simulation found.")
        return

    teams_ranked = df.sort_values("p_champion", ascending=False).head(8)["team"].tolist()
    choice = st.selectbox("Show path for:", ["Expected (top 8)"] + teams_ranked)

    rounds = [
        ("Group exit", "p_group_exit"),
        ("R32", "p_r32"), ("R16", "p_r16"), ("QF", "p_qf"),
        ("SF", "p_semi"), ("Final", "p_final"), ("Champion", "p_champion"),
    ]
    fig = go.Figure()
    palette = px.colors.qualitative.Set2

    if choice == "Expected (top 8)":
        for i, team in enumerate(teams_ranked):
            row = df[df["team"] == team].iloc[0]
            xs = list(range(len(rounds)))
            ys = [row[c] * 100 for _, c in rounds]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(width=2 + ys[-1] / 5, color=palette[i % len(palette)]),
                name=team,
                hovertemplate=f"<b>{team}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            ))
    else:
        row = df[df["team"] == choice].iloc[0]
        xs = list(range(len(rounds)))
        ys = [row[c] * 100 for _, c in rounds]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(width=3, color="#3b82f6"),
            name=choice,
            hovertemplate=f"<b>{choice}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis=dict(tickmode="array", tickvals=list(range(len(rounds))),
                   ticktext=[r[0] for r in rounds]),
        yaxis_title="Probability of reaching round (%)",
        title="Expected knockout path — line thickness ∝ P(reach round)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: this is a longitudinal view of each team's probability of "
        "reaching successive rounds, not a literal bracket pairing — the "
        "bracket is randomised across simulations so a single 'most-likely "
        "bracket' would over-simplify the joint distribution."
    )


# ---------------------------------------------------------------------------
# Page 4 — Model vs Market
# ---------------------------------------------------------------------------
def page_model_vs_market() -> None:
    st.header("Model vs Market")
    ci = _load_ci_table()
    pw = _load_pairwise()
    mvm = _load_model_vs_market()

    if ci.empty:
        _empty_state("No CI table found. Re-run `python main.py` to regenerate.")
        return

    st.subheader("Bootstrap confidence intervals (95%)")
    pivot_pt = ci.pivot(index="model", columns="metric", values="point_estimate")
    pivot_lo = ci.pivot(index="model", columns="metric", values="ci_low")
    pivot_hi = ci.pivot(index="model", columns="metric", values="ci_high")
    table_rows = []
    for m in pivot_pt.sort_values("brier").index:
        table_rows.append({
            "Model": m,
            "Brier ± 95% CI": (
                f"{pivot_pt.loc[m,'brier']:.4f} "
                f"[{pivot_lo.loc[m,'brier']:.3f}, {pivot_hi.loc[m,'brier']:.3f}]"
            ),
            "RPS ± 95% CI": (
                f"{pivot_pt.loc[m,'rps']:.4f} "
                f"[{pivot_lo.loc[m,'rps']:.3f}, {pivot_hi.loc[m,'rps']:.3f}]"
            ),
            "LogLoss ± 95% CI": (
                f"{pivot_pt.loc[m,'log_loss']:.4f} "
                f"[{pivot_lo.loc[m,'log_loss']:.3f}, {pivot_hi.loc[m,'log_loss']:.3f}]"
            ),
            "Accuracy ± 95% CI": (
                f"{pivot_pt.loc[m,'accuracy']*100:.1f}% "
                f"[{pivot_lo.loc[m,'accuracy']*100:.1f}, {pivot_hi.loc[m,'accuracy']*100:.1f}]"
            ),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    if not pw.empty:
        st.subheader("Pairwise Brier comparisons vs ensemble (paired bootstrap)")
        view = pw.copy()
        view["95% CI"] = view.apply(
            lambda r: f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1
        )
        view["Significant"] = view["significant_at_05"].map(lambda b: "✓" if b else "—")
        st.dataframe(
            view[["model", "mean_diff", "95% CI", "p_value", "Significant"]] \
                .rename(columns={"model": "Other model", "mean_diff": "Brier diff",
                                 "p_value": "p (2-sided)"}),
            use_container_width=True, hide_index=True,
        )

    if not mvm.empty:
        st.subheader("Model vs Market champion probabilities")
        scatter_df = mvm.copy()
        max_p = float(max(scatter_df["p_champion"].max(), scatter_df["market_implied_p"].max())) + 0.02
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, max_p], y=[0, max_p], mode="lines",
            line=dict(color="#6b7280", dash="dash"), name="Equal", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=scatter_df["p_champion"], y=scatter_df["market_implied_p"],
            mode="markers+text", text=scatter_df["team"], textposition="top center",
            marker=dict(size=10, color=scatter_df["delta_pp"], colorscale="RdBu_r",
                        cmid=0, showscale=True, colorbar=dict(title="Δ pp")),
            hovertemplate=(
                "<b>%{text}</b><br>Model: %{x:.2%}<br>Market: %{y:.2%}<extra></extra>"
            ),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Model P(champion)", yaxis_title="Market implied P",
            height=560, title="Above the line = model UNDER; below = model OVER",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top 10 biggest disagreements**")
        st.dataframe(
            mvm.assign(abs_delta=mvm["delta_pp"].abs())
                .sort_values("abs_delta", ascending=False).head(10) \
                [["team", "p_champion", "market_implied_p", "delta_pp", "over_under"]],
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page 5 — Live tournament mode
# ---------------------------------------------------------------------------
def page_live_mode() -> None:
    st.header("Live Tournament Mode")
    st.caption(
        "Record actual results as the tournament unfolds. Each new result "
        "triggers a Bayesian shrinkage update of the affected teams' Dixon-"
        "Coles parameters and a full re-simulation of the remaining fixtures."
    )

    baseline = _load_baseline()
    if baseline.empty:
        _empty_state("No baseline simulation found.")
        return

    live = _load_live_results()
    live_preds = _load_live_predictions()

    teams = sorted({t for ts in WC2026_GROUPS.values() for t in ts})

    st.subheader("Record a result")
    with st.form("record_form", clear_on_submit=True):
        cols = st.columns([3, 3, 1, 1, 2, 1])
        home = cols[0].selectbox("Home", teams, key="live_home")
        away = cols[1].selectbox("Away", [t for t in teams if t != home], key="live_away")
        gh = cols[2].number_input("GH", min_value=0, max_value=15, value=1, step=1, key="live_gh")
        ga = cols[3].number_input("GA", min_value=0, max_value=15, value=1, step=1, key="live_ga")
        date = cols[4].date_input("Date", key="live_date")
        stage = cols[5].selectbox("Stage", ["group", "knockout"], key="live_stage")
        submitted = st.form_submit_button("Record + re-simulate", type="primary",
                                           use_container_width=True)

    if submitted:
        if home == away:
            st.error("Home and away must differ.")
        else:
            append_live_result({
                "home": home, "away": away,
                "goals_home": int(gh), "goals_away": int(ga),
                "date": pd.Timestamp(date), "stage": stage,
            })
            with st.spinner("Updating model and re-simulating …"):
                predictor = _load_predictor()
                if predictor is None:
                    st.error("Predictor not available.")
                else:
                    out = update_and_resimulate(predictor, n_sims=10_000)
                    out["updated_df"].to_parquet(
                        SIM_RESULTS / "live_predictions.parquet", index=False,
                    )
                    _load_live_predictions.clear()
                    _load_live_results.clear()
                    st.success(f"Updated. {out['n_results_applied']} live results applied.")
                    st.rerun()

    st.subheader("Recorded results")
    if live.empty:
        st.info("No live results recorded yet.")
    else:
        st.dataframe(live.sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)

    if st.button("⟲ Reset to pre-tournament", help="Delete all live results"):
        if LIVE_RESULTS_PATH.exists():
            LIVE_RESULTS_PATH.unlink()
        live_pred_path = SIM_RESULTS / "live_predictions.parquet"
        if live_pred_path.exists():
            live_pred_path.unlink()
        _load_live_predictions.clear()
        _load_live_results.clear()
        st.success("Live state cleared.")
        st.rerun()

    if not live_preds.empty:
        merged = baseline[["team", "p_champion"]].rename(columns={"p_champion": "pre"}) \
            .merge(
                live_preds[["team", "p_champion"]].rename(columns={"p_champion": "live"}),
                on="team", how="outer",
            )
        merged["delta_pp"] = (merged["live"] - merged["pre"]) * 100.0
        st.subheader("Predictions: pre-tournament vs current")
        top = merged.sort_values("live", ascending=False).head(16)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Pre-tournament", x=top["team"], y=top["pre"] * 100, marker_color="#6b7280"))
        fig.add_trace(go.Bar(name="Current",        x=top["team"], y=top["live"] * 100, marker_color="#3b82f6"))
        fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE,
                          yaxis_title="P(champion) %", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            merged.assign(pre_pct=lambda d: (d["pre"] * 100).round(2),
                          live_pct=lambda d: (d["live"] * 100).round(2),
                          delta_pp=lambda d: d["delta_pp"].round(2)) \
                  .sort_values("delta_pp", key=lambda s: s.abs(), ascending=False) \
                  [["team", "pre_pct", "live_pct", "delta_pp"]],
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page 6 — Methodology
# ---------------------------------------------------------------------------
def page_methodology() -> None:
    st.header("Methodology & Caveats")
    st.markdown(
        """
        **What this model does well**

        * Cross-confederation discrimination — the empirical confederation
          difficulty fit means UEFA xG is up-weighted relative to CONCACAF xG,
          so qualifying records aren't taken at face value.
        * Calibration after isotonic regression — the XGB head's predicted
          probabilities track the empirical frequencies closely on the held-
          out WC 2018 + 2022 set.
        * Beating a market-derived Bradley-Terry baseline on Brier *on average*
          across the 128-match WC 2018+2022 test set — but see the CI table
          below before celebrating.

        **What this model does poorly**

        * Argentina overconfidence — the ensemble parks Argentina near 20%
          when the market sits around 9%. Likely overfit to a soft group +
          a recent winning streak; the model has no injury, lineup, or
          tactical-matchup awareness.
        * No injury awareness — a Haaland or Bellingham strain doesn't move
          a single parameter unless you manually adjust it in the scenario
          builder.
        * No mid-tournament momentum / fatigue corrections beyond what the
          dynamic update engine pulls in from actual results.
        * Training set skewed toward non-WC matches — qualifying + friendly
          dynamics differ from knockout football, which the model can't
          structurally distinguish.

        **What "beating the market" actually means here**

        The baseline used here is a Bradley-Terry decomposition of the
        pre-tournament outright odds — _not_ a per-match in-running line.
        The n=128 test set comes from WC 2018 + WC 2022. The paired bootstrap
        CI on the Brier difference tells you whether the observed advantage
        survives Monte Carlo resampling.
        """
    )

    ci = _load_ci_table()
    pw = _load_pairwise()
    if not ci.empty:
        st.subheader("Bootstrap CI table (Part A)")
        pivot_pt = ci.pivot(index="model", columns="metric", values="point_estimate")
        pivot_lo = ci.pivot(index="model", columns="metric", values="ci_low")
        pivot_hi = ci.pivot(index="model", columns="metric", values="ci_high")
        rows = []
        for m in pivot_pt.sort_values("brier").index:
            rows.append({
                "Model": m,
                "Brier ± 95% CI": (
                    f"{pivot_pt.loc[m,'brier']:.4f} "
                    f"[{pivot_lo.loc[m,'brier']:.3f}, {pivot_hi.loc[m,'brier']:.3f}]"
                ),
                "RPS ± 95% CI": (
                    f"{pivot_pt.loc[m,'rps']:.4f} "
                    f"[{pivot_lo.loc[m,'rps']:.3f}, {pivot_hi.loc[m,'rps']:.3f}]"
                ),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not pw.empty and "Betting Market" in pw["model"].values:
        m_row = pw.loc[pw["model"] == "Betting Market"].iloc[0]
        # mean_diff = Brier(other) − Brier(reference). Positive ⇒ ref (ensemble) wins.
        if m_row["ci_low"] > 0:
            verdict = (
                f"**Honest verdict:** Ensemble beats the market on Brier by "
                f"{m_row['mean_diff']:.4f} (95% CI "
                f"[{m_row['ci_low']:+.3f}, {m_row['ci_high']:+.3f}], "
                f"p = {m_row['p_value']:.3f})."
            )
        elif m_row["ci_high"] < 0:
            verdict = (
                f"**Honest verdict:** Market beats the ensemble by "
                f"{abs(m_row['mean_diff']):.4f} (CI "
                f"[{m_row['ci_low']:+.3f}, {m_row['ci_high']:+.3f}])."
            )
        else:
            verdict = (
                f"**Honest verdict:** The CI on the model-minus-market Brier "
                f"includes zero (diff {m_row['mean_diff']:+.4f}, CI "
                f"[{m_row['ci_low']:+.3f}, {m_row['ci_high']:+.3f}], "
                f"p = {m_row['p_value']:.3f}) — performance is _comparable_ "
                "to the market, not demonstrably better."
            )
        st.markdown(verdict)

    st.markdown(
        """
        **Sources & methodology overview**

        * Match data — `martj42/international_results` (CSV).
        * ELO — ClubElo when reachable, fallback table otherwise.
        * Outright odds — cached snapshot (no live API dependency in the
          dashboard).
        * Dixon-Coles fit by L-BFGS-B on a Poisson likelihood with low-score
          τ-correction.
        * XGBoost is calibrated with isotonic CV (k=3).
        * 50 000 Monte Carlo simulations per baseline run.

        **GitHub:** project root contains the full pipeline; the file
        `dashboard/app.py` defines this UI.
        """
    )


# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="WC 2026 Predictor", layout="wide", page_icon="⚽",
    )
    st.sidebar.title("WC 2026 Predictor")
    st.sidebar.caption("Live, interactive Monte Carlo dashboard")

    page = st.sidebar.radio(
        "Navigate",
        [
            "1 · Tournament Overview",
            "2 · Scenario Builder",
            "3 · Bracket",
            "4 · Model vs Market",
            "5 · Live Tournament",
            "6 · Methodology",
        ],
    )

    meta = _load_run_meta()
    if meta:
        ts = meta.get("elapsed_total_seconds")
        st.sidebar.caption(f"Total pipeline runtime: {ts}s" if ts else "")

    {
        "1 · Tournament Overview": page_overview,
        "2 · Scenario Builder":    page_scenario_builder,
        "3 · Bracket":             page_bracket,
        "4 · Model vs Market":     page_model_vs_market,
        "5 · Live Tournament":     page_live_mode,
        "6 · Methodology":         page_methodology,
    }[page]()


if __name__ == "__main__":
    main()
