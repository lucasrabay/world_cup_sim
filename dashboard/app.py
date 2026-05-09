"""Streamlit dashboard for the WC 2026 predictor."""
from __future__ import annotations

import sys
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

from src.data_loader import HOST_NATIONS, SQUAD_VALUES, WC2026_GROUPS  # noqa: E402
from src.utils import MODELS_SAVED, SIM_RESULTS  # noqa: E402


CONFEDERATION = {
    # UEFA
    **{t: "UEFA" for t in [
        "England", "France", "Spain", "Germany", "Portugal", "Netherlands",
        "Belgium", "Norway", "Croatia", "Switzerland", "Austria", "Sweden",
        "Turkey", "Bosnia and Herzegovina", "Scotland", "Czechia",
    ]},
    # CONMEBOL
    **{t: "CONMEBOL" for t in ["Brazil", "Argentina", "Colombia", "Uruguay", "Ecuador", "Paraguay"]},
    # CONCACAF
    **{t: "CONCACAF" for t in ["USA", "Mexico", "Canada", "Panama", "Haiti", "Curacao"]},
    # CAF
    **{t: "CAF" for t in [
        "Senegal", "Morocco", "Ivory Coast", "Egypt", "Algeria", "Ghana",
        "Tunisia", "South Africa", "Cape Verde", "DR Congo",
    ]},
    # AFC
    **{t: "AFC" for t in [
        "Japan", "South Korea", "Australia", "Saudi Arabia", "Qatar", "Iran",
        "Jordan", "Iraq", "Uzbekistan",
    ]},
    # OFC
    **{t: "OFC" for t in ["New Zealand"]},
}


@st.cache_data(show_spinner=False)
def _load_baseline() -> pd.DataFrame:
    p = SIM_RESULTS / "baseline.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_scenario(name: str) -> pd.DataFrame:
    p = SIM_RESULTS / f"scenario_{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _list_scenarios() -> list[str]:
    out: list[str] = []
    for p in SIM_RESULTS.glob("scenario_*.parquet"):
        out.append(p.stem.replace("scenario_", ""))
    return sorted(out)


@st.cache_data(show_spinner=False)
def _load_upset_risks() -> pd.DataFrame:
    p = SIM_RESULTS / "upset_risks.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def _empty_state(msg: str) -> None:
    st.warning(msg)
    st.info("Run `python main.py` first to populate the simulation results.")


def page_overview() -> None:
    st.header("Tournament Overview")
    df = _load_baseline()
    if df.empty:
        _empty_state("No baseline simulation found.")
        return

    st.dataframe(
        df.assign(
            p_champion_pct=lambda d: (d["p_champion"] * 100).round(2),
            p_final_pct=lambda d: (d["p_final"] * 100).round(2),
            p_semi_pct=lambda d: (d["p_semi"] * 100).round(2),
        )[["team", "group", "p_champion_pct", "p_final_pct", "p_semi_pct",
            "avg_goals_scored_per_sim", "avg_goals_conceded_per_sim"]],
        use_container_width=True,
        column_config={
            "p_champion_pct": st.column_config.ProgressColumn(
                "P(champion) %", min_value=0, max_value=float(df["p_champion"].max() * 100), format="%.2f"),
            "p_final_pct": st.column_config.ProgressColumn(
                "P(final) %", min_value=0, max_value=float(df["p_final"].max() * 100), format="%.2f"),
        },
    )

    top16 = df.head(16)
    fig = px.bar(top16, x="p_champion", y="team", orientation="h",
                 labels={"p_champion": "P(win tournament)", "team": ""},
                 title="Top 16 — Champion Probability",
                 color="p_champion", color_continuous_scale="Viridis")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    df["confederation"] = df["team"].map(CONFEDERATION).fillna("Other")
    by_conf = df.groupby("confederation")["p_champion"].sum().reset_index()
    fig2 = px.pie(by_conf, names="confederation", values="p_champion",
                  title="Champion Probability by Confederation")
    st.plotly_chart(fig2, use_container_width=True)


def page_groups() -> None:
    st.header("Group Explorer")
    df = _load_baseline()
    if df.empty:
        _empty_state("No baseline simulation found.")
        return

    g = st.selectbox("Group", sorted(WC2026_GROUPS.keys()))
    members = WC2026_GROUPS[g]
    sub = df[df["team"].isin(members)].copy()
    sub["p_advance"] = sub["p_r32"]

    st.dataframe(
        sub[["team", "p_advance", "p_top_group", "p_second_group", "p_third_group",
             "p_qf", "p_semi", "p_final", "p_champion"]]
            .assign(**{c: (sub[c] * 100).round(2) for c in [
                "p_advance", "p_top_group", "p_second_group", "p_third_group",
                "p_qf", "p_semi", "p_final", "p_champion"]})
            .sort_values("p_advance", ascending=False),
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(name="1st in group", x=sub["p_top_group"], y=sub["team"], orientation="h"))
    fig.add_trace(go.Bar(name="2nd in group", x=sub["p_second_group"], y=sub["team"], orientation="h"))
    fig.add_trace(go.Bar(name="3rd in group", x=sub["p_third_group"], y=sub["team"], orientation="h"))
    fig.update_layout(barmode="stack", title=f"Group {g} — Placement Probability",
                       xaxis_title="P(finish in position)")
    st.plotly_chart(fig, use_container_width=True)


def page_h2h() -> None:
    st.header("Head-to-Head")
    df = _load_baseline()
    if df.empty:
        _empty_state("No baseline simulation found.")
        return

    teams = sorted(df["team"].unique())
    col1, col2 = st.columns(2)
    a = col1.selectbox("Team A", teams, index=0)
    b = col2.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)

    if a == b:
        st.info("Pick two different teams.")
        return

    # Match probabilities — use the saved Dixon-Coles model directly.
    from src.models import DixonColesModel

    dc_path = MODELS_SAVED / "dixon_coles.json"
    if not dc_path.exists():
        st.warning("Dixon-Coles model not found.")
        return
    dc = DixonColesModel.load(dc_path)
    probs = dc.predict_outcome_probs(a, b, neutral=True, max_goals=5)

    st.subheader(f"{a}  vs  {b}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{a} wins", f"{probs['home_win']*100:.1f}%")
    c2.metric("Draw", f"{probs['draw']*100:.1f}%")
    c3.metric(f"{b} wins", f"{probs['away_win']*100:.1f}%")

    # Scoreline heatmap (5x5)
    l1, l2 = probs["lambda_home"], probs["lambda_away"]
    from scipy.special import gammaln

    ks = np.arange(6)
    log_ph = ks * np.log(l1) - l1 - gammaln(ks + 1)
    log_pa = ks * np.log(l2) - l2 - gammaln(ks + 1)
    grid = np.outer(np.exp(log_ph), np.exp(log_pa))
    grid = grid / grid.sum()

    fig = go.Figure(data=go.Heatmap(
        z=grid, x=[f"{x}" for x in ks], y=[f"{x}" for x in ks],
        colorscale="YlOrRd", text=(grid * 100).round(1),
        texttemplate="%{text}%", colorbar={"title": "P"},
    ))
    fig.update_layout(title=f"Scoreline probability — {a} (rows) vs {b} (cols)",
                       xaxis_title=f"{b} goals", yaxis_title=f"{a} goals")
    st.plotly_chart(fig, use_container_width=True)


def page_scenarios() -> None:
    st.header("Scenario Simulator")
    baseline = _load_baseline()
    if baseline.empty:
        _empty_state("No baseline simulation found.")
        return

    available = _list_scenarios()
    chosen = st.multiselect("Scenarios", available, default=[s for s in available if s != "baseline"][:2])
    if not chosen:
        st.info("Pick one or more scenarios to compare against baseline.")
        return

    table = baseline[["team", "p_champion"]].rename(columns={"p_champion": "baseline"})
    for name in chosen:
        s = _load_scenario(name)
        if s.empty:
            continue
        table = table.merge(
            s[["team", "p_champion"]].rename(columns={"p_champion": name}),
            on="team",
            how="left",
        )

    for col in chosen:
        if col in table.columns:
            table[f"{col}_delta"] = table[col] - table["baseline"]

    st.dataframe(
        table.assign(**{c: (table[c] * 100).round(3) for c in table.columns if c not in ("team",)})
            .sort_values("baseline", ascending=False),
        use_container_width=True,
    )

    melted = table.melt(id_vars="team", value_vars=["baseline"] + chosen,
                         var_name="scenario", value_name="p_champion")
    top = melted.groupby("team")["p_champion"].max().nlargest(12).index
    fig = px.bar(melted[melted["team"].isin(top)], x="team", y="p_champion",
                 color="scenario", barmode="group",
                 title="Champion probability per scenario (top 12)")
    st.plotly_chart(fig, use_container_width=True)


def page_explainability() -> None:
    st.header("Model Explainability")
    shap_path = MODELS_SAVED / "shap_summary.png"
    cal_path = MODELS_SAVED / "calibration_curve.png"
    cm_path = MODELS_SAVED / "confusion_matrix.png"

    if shap_path.exists():
        st.subheader("SHAP feature contributions")
        st.image(str(shap_path))
    else:
        st.info("SHAP plot not found. Run `python main.py` to generate it.")

    if cal_path.exists():
        st.subheader("Calibration curve (home win)")
        st.image(str(cal_path))
    if cm_path.exists():
        st.subheader("Confusion matrix")
        st.image(str(cm_path))

    metrics_path = MODELS_SAVED / "eval_metrics.json"
    if metrics_path.exists():
        import json
        st.subheader("Held-out metrics (WC 2018 + 2022)")
        st.json(json.loads(metrics_path.read_text()))


def page_upsets() -> None:
    st.header("Match Upset Risk")
    df = _load_upset_risks()
    if df.empty:
        _empty_state("No upset-risk data found.")
        return

    def colour(p: float) -> str:
        if p < 0.15: return "#1b9e77"
        if p < 0.35: return "#ffcc00"
        return "#d95f02"

    df = df.sort_values(["group", "p_upset"], ascending=[True, False]).reset_index(drop=True)
    df["P(upset) %"] = (df["p_upset"] * 100).round(2)

    st.dataframe(
        df.style.applymap(lambda v: f"background-color: {colour(v / 100)}", subset=["P(upset) %"]),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="WC 2026 Predictor", layout="wide")
    st.sidebar.title("WC 2026 Predictor")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Groups", "Head-to-Head", "Scenarios", "Explainability", "Upset Risk"],
    )
    {
        "Overview": page_overview,
        "Groups": page_groups,
        "Head-to-Head": page_h2h,
        "Scenarios": page_scenarios,
        "Explainability": page_explainability,
        "Upset Risk": page_upsets,
    }[page]()


if __name__ == "__main__":
    main()
