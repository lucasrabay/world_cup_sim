"""Feature engineering for the international results dataset.

Builds the per-match feature matrix consumed by the XGBoost classifier and the
match-level metadata required by the Dixon-Coles fitter.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd

from .data_loader import FALLBACK_ELO, SQUAD_VALUES
from .utils import get_logger, load_config, normalise_team

logger = get_logger(__name__)


# Approximate qualifying-campaign xGD (per match) for the 48 WC 2026 teams.
QUALIFYING_XGD: dict[str, float] = {
    "England": +1.8, "France": +1.4, "Spain": +1.6, "Germany": +1.2,
    "Brazil": +0.6, "Portugal": +1.3, "Argentina": +1.1, "Netherlands": +1.0,
    "Belgium": +0.9, "Norway": +1.5, "Colombia": +0.5, "Uruguay": +0.4,
    "Croatia": +0.6, "Japan": +1.1, "Senegal": +0.7, "Morocco": +0.8,
    "USA": +0.8, "Mexico": +0.5, "Canada": +0.7, "Australia": +0.4,
    "South Korea": +0.9, "Switzerland": +0.8, "Austria": +1.0, "Sweden": +1.2,
    "Turkey": +0.7, "Ecuador": +0.3, "Ivory Coast": +0.6, "Egypt": +0.5,
    "Algeria": +0.3, "Ghana": +0.3, "Tunisia": +0.4, "South Africa": +0.2,
    "Saudi Arabia": +0.1, "Qatar": +0.2, "Iran": +0.3, "Paraguay": +0.2,
    "Panama": +0.1, "Bosnia and Herzegovina": +0.4, "Scotland": +0.5,
    "Czechia": +0.3, "Haiti": -0.2, "New Zealand": +0.0,
    "Cape Verde": +0.4, "Jordan": +0.2, "Iraq": +0.3, "Uzbekistan": +0.3,
    "DR Congo": +0.2, "Curacao": +0.1,
}


# Feature columns produced by build_match_features (target excluded).
FEATURE_COLUMNS: list[str] = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "value_home",
    "value_away",
    "value_ratio",
    "is_neutral",
    "is_wc",
    "tournament_weight",
    "days_since_match_home",
    "days_since_match_away",
    "home_form_5",
    "away_form_5",
    "h2h_wc_home_winrate",
    "goals_scored_10_home",
    "goals_conceded_10_home",
    "goals_scored_10_away",
    "goals_conceded_10_away",
    "xg_diff_qualifying_home",
    "xg_diff_qualifying_away",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify_tournament(name: str) -> tuple[float, int]:
    """Return (tournament_weight, is_wc).

    World Cup -> 1.0, Confederations / Nations / continental -> 0.7,
    qualification -> 0.5, friendly / other -> 0.2.
    """
    n = name.lower()
    is_wc = int("world cup" in n and "qualification" not in n)
    if is_wc:
        return 1.0, 1
    if any(s in n for s in ("confederations", "nations league", "uefa euro", "copa america",
                             "africa cup", "asian cup", "concacaf gold", "ofc nations")):
        return 0.7, 0
    if "qualification" in n:
        return 0.5, 0
    return 0.2, 0


def _interpolate_elo(elo_df: pd.DataFrame, team: str, when: pd.Timestamp) -> float:
    """ELO for ``team`` at date ``when`` (last observation carried forward)."""
    if elo_df.empty:
        return FALLBACK_ELO.get(team, 1500.0)
    sub = elo_df[(elo_df["team"] == team) & (elo_df["date"] <= when)]
    if sub.empty:
        team_any = elo_df[elo_df["team"] == team]
        if team_any.empty:
            return FALLBACK_ELO.get(team, 1500.0)
        return float(team_any.iloc[0]["elo"])
    return float(sub.iloc[-1]["elo"])


def _build_elo_lookup(elo_df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pre-index ELO history by team for O(log n) lookups via searchsorted."""
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if elo_df.empty:
        return lookup
    for team, sub in elo_df.sort_values("date").groupby("team"):
        dates = sub["date"].values.astype("datetime64[ns]")
        elos = sub["elo"].to_numpy(dtype=float)
        lookup[team] = (dates, elos)
    return lookup


def _elo_at(lookup: dict[str, tuple[np.ndarray, np.ndarray]], team: str, when: np.datetime64) -> float:
    if team not in lookup:
        return FALLBACK_ELO.get(team, 1500.0)
    dates, elos = lookup[team]
    idx = int(np.searchsorted(dates, when, side="right")) - 1
    if idx < 0:
        return float(elos[0])
    return float(elos[idx])


def _value_for(team: str, value_lookup: dict[str, float]) -> float:
    """Return log squad value, defaulting to a sensible mid-range when unknown."""
    if team in value_lookup:
        return float(np.log(value_lookup[team]))
    return float(np.log(SQUAD_VALUES.get(team, 80.0)))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_match_features(
    results_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    squad_values: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the per-match feature matrix and the target column.

    Parameters
    ----------
    results_df : DataFrame with columns date, home_team, away_team, home_score,
        away_score, tournament, neutral.
    elo_df : long-format DataFrame [team, date, elo].
    squad_values : DataFrame from ``load_squad_values()`` (optional).

    Returns
    -------
    DataFrame with the feature columns above plus
        - date, home_team, away_team
        - home_score, away_score
        - outcome (0 home loss, 1 draw, 2 home win)
        - sample_weight (time-decay weight)
    """
    cfg = load_config()
    df = results_df.copy().sort_values("date").reset_index(drop=True)
    df["home_team"] = df["home_team"].map(normalise_team)
    df["away_team"] = df["away_team"].map(normalise_team)

    # Squad value lookup
    if squad_values is None:
        value_lookup = dict(SQUAD_VALUES)
    else:
        value_lookup = dict(zip(squad_values["team"], squad_values["value_eur_m"]))

    # Pre-index ELO for fast lookup
    elo_lookup = _build_elo_lookup(elo_df)

    # Rolling stats containers — deque per team so we can capture the last N
    # matches' goals, results, and recent dates.
    last_n_results: dict[str, deque] = {}      # last 5 win indicators (home loss/draw/win flags as 0,0.5,1)
    last_n_for: dict[str, deque] = {}          # last 10 goals scored
    last_n_against: dict[str, deque] = {}      # last 10 goals conceded
    last_match_date: dict[str, pd.Timestamp] = {}
    h2h_wc_results: dict[tuple[str, str], deque] = {}

    # Output containers (column-major lists, faster than appending to a DataFrame)
    n = len(df)
    out: dict[str, list] = {col: [] for col in FEATURE_COLUMNS}
    extra: dict[str, list] = {
        "date": [], "home_team": [], "away_team": [],
        "home_score": [], "away_score": [], "outcome": [], "sample_weight": [],
    }

    half_life = float(cfg["model"]["time_decay_half_life_days"])
    decay_lambda = np.log(2.0) / half_life
    latest_date = df["date"].max()

    for row in df.itertuples(index=False):
        h: str = row.home_team
        a: str = row.away_team
        match_date: pd.Timestamp = row.date
        match_dt64 = np.datetime64(match_date.to_datetime64()) if hasattr(match_date, "to_datetime64") else np.datetime64(match_date)

        # --- ELO
        elo_h = _elo_at(elo_lookup, h, match_dt64)
        elo_a = _elo_at(elo_lookup, a, match_dt64)

        # --- Values
        v_h = _value_for(h, value_lookup)
        v_a = _value_for(a, value_lookup)

        # --- Tournament metadata
        weight, is_wc = _classify_tournament(str(row.tournament))
        is_neutral = int(bool(getattr(row, "neutral", False)))

        # --- Days since previous match
        prev_h = last_match_date.get(h)
        prev_a = last_match_date.get(a)
        ds_h = float((match_date - prev_h).days) if prev_h is not None else 30.0
        ds_a = float((match_date - prev_a).days) if prev_a is not None else 30.0

        # --- Form & rolling goal averages
        def _avg(d: deque, default: float) -> float:
            return float(sum(d) / len(d)) if len(d) > 0 else default

        form_h = _avg(last_n_results.setdefault(h, deque(maxlen=5)), 0.5)
        form_a = _avg(last_n_results.setdefault(a, deque(maxlen=5)), 0.5)
        gs10_h = _avg(last_n_for.setdefault(h, deque(maxlen=10)), 1.2)
        gc10_h = _avg(last_n_against.setdefault(h, deque(maxlen=10)), 1.2)
        gs10_a = _avg(last_n_for.setdefault(a, deque(maxlen=10)), 1.2)
        gc10_a = _avg(last_n_against.setdefault(a, deque(maxlen=10)), 1.2)

        # --- H2H WC win rate (home-team perspective). Use canonicalised pair.
        pair = tuple(sorted([h, a]))
        h2h_deque = h2h_wc_results.setdefault(pair, deque(maxlen=5))
        if h2h_deque:
            # Each entry stored as 1 if pair[0] won, 0 if pair[1] won, 0.5 draw.
            wr_pair0 = sum(h2h_deque) / len(h2h_deque)
            h2h_home_wr = wr_pair0 if pair[0] == h else 1.0 - wr_pair0
        else:
            h2h_home_wr = 0.5

        # --- Qualifying xGD
        xg_h = float(QUALIFYING_XGD.get(h, 0.0))
        xg_a = float(QUALIFYING_XGD.get(a, 0.0))

        # --- Outcome
        hs, as_ = int(row.home_score), int(row.away_score)
        if hs > as_:
            outcome = 2
            home_pts = 1.0
            pair0_pts = 1.0 if pair[0] == h else 0.0
        elif hs == as_:
            outcome = 1
            home_pts = 0.5
            pair0_pts = 0.5
        else:
            outcome = 0
            home_pts = 0.0
            pair0_pts = 0.0 if pair[0] == h else 1.0

        # --- Time-decay sample weight
        days_ago = max((latest_date - match_date).days, 0)
        sw = float(np.exp(-decay_lambda * days_ago))

        # --- Append row
        out["elo_home"].append(elo_h)
        out["elo_away"].append(elo_a)
        out["elo_diff"].append(elo_h - elo_a)
        out["value_home"].append(v_h)
        out["value_away"].append(v_a)
        out["value_ratio"].append(v_h - v_a)
        out["is_neutral"].append(is_neutral)
        out["is_wc"].append(is_wc)
        out["tournament_weight"].append(weight)
        out["days_since_match_home"].append(ds_h)
        out["days_since_match_away"].append(ds_a)
        out["home_form_5"].append(form_h)
        out["away_form_5"].append(form_a)
        out["h2h_wc_home_winrate"].append(h2h_home_wr)
        out["goals_scored_10_home"].append(gs10_h)
        out["goals_conceded_10_home"].append(gc10_h)
        out["goals_scored_10_away"].append(gs10_a)
        out["goals_conceded_10_away"].append(gc10_a)
        out["xg_diff_qualifying_home"].append(xg_h)
        out["xg_diff_qualifying_away"].append(xg_a)

        extra["date"].append(match_date)
        extra["home_team"].append(h)
        extra["away_team"].append(a)
        extra["home_score"].append(hs)
        extra["away_score"].append(as_)
        extra["outcome"].append(outcome)
        extra["sample_weight"].append(sw)

        # --- Update rolling state AFTER recording features
        last_n_results[h].append(home_pts)
        last_n_results[a].append(1.0 - home_pts)
        last_n_for[h].append(hs)
        last_n_for[a].append(as_)
        last_n_against[h].append(as_)
        last_n_against[a].append(hs)
        last_match_date[h] = match_date
        last_match_date[a] = match_date
        if is_wc:
            h2h_deque.append(pair0_pts)

    feat_df = pd.DataFrame({**extra, **out})
    # Replace any NaN / inf that might have slipped through
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logger.info("Built feature matrix with %d rows and %d feature columns", len(feat_df), len(FEATURE_COLUMNS))
    return feat_df


def split_features_target(
    feat_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convenience accessor returning (X, y, sample_weights)."""
    X = feat_df[FEATURE_COLUMNS].copy()
    y = feat_df["outcome"].astype(int).copy()
    w = feat_df["sample_weight"].astype(float).copy()
    return X, y, w


__all__ = [
    "QUALIFYING_XGD",
    "FEATURE_COLUMNS",
    "build_match_features",
    "split_features_target",
]
