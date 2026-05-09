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


# ---------------------------------------------------------------------------
# Reference dictionaries
# ---------------------------------------------------------------------------
# Approximate qualifying-campaign xGD (per match) for the 48 WC 2026 teams.
# The raw values are biased upward for confederations whose teams play each
# other repeatedly during qualifying (CONMEBOL especially); the
# CONFEDERATION_DIFFICULTY multiplier below corrects for that mutual inflation.
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


# Confederation difficulty discount for qualifying-campaign metrics.
# CONMEBOL teams play one another every cycle and inflate one another's
# stats; UEFA matches happen against a much wider talent pool, so UEFA is
# the reference 1.0 baseline.
CONFEDERATION_DIFFICULTY: dict[str, float] = {
    "UEFA": 1.0,
    "CONMEBOL": 0.72,
    "AFC": 0.65,
    "CAF": 0.60,
    "CONCACAF": 0.55,
    "OFC": 0.40,
}


# Confederation lookup for all 48 WC 2026 participants.
TEAM_CONFEDERATION: dict[str, str] = {
    # UEFA
    "England": "UEFA", "France": "UEFA", "Spain": "UEFA", "Germany": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Norway": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA", "Sweden": "UEFA",
    "Turkey": "UEFA", "Bosnia and Herzegovina": "UEFA", "Scotland": "UEFA", "Czechia": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    # CONCACAF
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curacao": "CONCACAF",
    # CAF
    "Senegal": "CAF", "Morocco": "CAF", "Ivory Coast": "CAF", "Egypt": "CAF",
    "Algeria": "CAF", "Ghana": "CAF", "Tunisia": "CAF", "South Africa": "CAF",
    "Cape Verde": "CAF", "DR Congo": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Saudi Arabia": "AFC",
    "Qatar": "AFC", "Iran": "AFC", "Jordan": "AFC", "Iraq": "AFC", "Uzbekistan": "AFC",
    # OFC
    "New Zealand": "OFC",
}


# Approximate official FIFA rankings for the 48 WC 2026 participants (May 2026).
FIFA_RANKING_2026: dict[str, int] = {
    "Spain": 1, "Argentina": 2, "France": 3, "England": 4, "Brazil": 5,
    "Portugal": 6, "Netherlands": 7, "Belgium": 8, "Germany": 9, "Croatia": 10,
    "USA": 11, "Colombia": 12, "Morocco": 13, "Japan": 14, "Uruguay": 15,
    "Switzerland": 16, "Senegal": 17, "Iran": 18, "South Korea": 19, "Mexico": 20,
    "Norway": 21, "Sweden": 22, "Egypt": 23, "Australia": 24, "Canada": 25,
    "Tunisia": 26, "Scotland": 27, "Algeria": 28, "Austria": 29, "Czechia": 30,
    "Saudi Arabia": 31, "Turkey": 32, "Ecuador": 33, "Ivory Coast": 34, "Ghana": 35,
    "Paraguay": 36, "South Africa": 37, "Bosnia and Herzegovina": 38, "Iraq": 39,
    "Uzbekistan": 40, "Cape Verde": 41, "Jordan": 42, "DR Congo": 43, "Qatar": 44,
    "Panama": 45, "Curacao": 46, "New Zealand": 47, "Haiti": 48,
}


# Historical knockout-stage win rates at the FIFA World Cup. Values are
# approximations based on historical KO match outcomes (treating shootouts as
# the winning result). Default for teams without WC KO history: 0.20.
WC_KNOCKOUT_WIN_RATE: dict[str, float] = {
    "Brazil": 0.68, "Germany": 0.65, "Argentina": 0.62, "France": 0.58,
    "Spain": 0.55, "Netherlands": 0.52, "Croatia": 0.50, "England": 0.48,
    "Portugal": 0.45, "Uruguay": 0.44, "Czechia": 0.42, "Belgium": 0.40,
    "Sweden": 0.40, "Austria": 0.40, "Morocco": 0.38, "Turkey": 0.38,
    "Colombia": 0.35, "Mexico": 0.33, "Senegal": 0.32, "Switzerland": 0.30,
    "Japan": 0.30, "South Korea": 0.30, "Norway": 0.30, "Ghana": 0.30,
    "Paraguay": 0.30, "USA": 0.28, "Ecuador": 0.25, "DR Congo": 0.20,
    "Ivory Coast": 0.20, "South Africa": 0.20, "Australia": 0.18, "Algeria": 0.15,
    "Bosnia and Herzegovina": 0.15, "Saudi Arabia": 0.12, "Tunisia": 0.10,
    "Egypt": 0.10, "Iran": 0.10, "Scotland": 0.10, "Canada": 0.10,
    "Iraq": 0.10, "Uzbekistan": 0.10, "Cape Verde": 0.10, "Jordan": 0.10,
    "Qatar": 0.10, "Panama": 0.10, "Curacao": 0.10, "New Zealand": 0.10,
    "Haiti": 0.10,
}
DEFAULT_KO_RATE = 0.20


# Squad-value tier (1=minnow ... 5=elite). Trees handle these buckets cleanly.
def _value_tier(value_eur_m: float) -> int:
    if value_eur_m >= 700:
        return 5
    if value_eur_m >= 300:
        return 4
    if value_eur_m >= 150:
        return 3
    if value_eur_m >= 80:
        return 2
    return 1


# Feature columns produced by build_match_features (target excluded).
FEATURE_COLUMNS: list[str] = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "value_home",
    "value_away",
    "value_ratio",
    "value_tier_home",
    "value_tier_away",
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
    "fifa_rank_home",
    "fifa_rank_away",
    "rank_diff",
    "log_rank_ratio",
    "wc_knockout_rate_home",
    "wc_knockout_rate_away",
    "wc_knockout_rate_diff",
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


def _confederation_multiplier(team: str) -> float:
    return float(CONFEDERATION_DIFFICULTY.get(TEAM_CONFEDERATION.get(team, ""), 1.0))


def _qualifying_xgd(team: str) -> float:
    """xGD discounted by the team's confederation difficulty."""
    raw = float(QUALIFYING_XGD.get(team, 0.0))
    return raw * _confederation_multiplier(team)


def _fifa_rank(team: str) -> int:
    """FIFA rank for a team — 99 if unknown (so unknown teams look weak)."""
    return int(FIFA_RANKING_2026.get(team, 99))


def _wc_ko_rate(team: str) -> float:
    return float(WC_KNOCKOUT_WIN_RATE.get(team, DEFAULT_KO_RATE))


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


def _value_raw(team: str, value_lookup: dict[str, float]) -> float:
    return float(value_lookup.get(team, SQUAD_VALUES.get(team, 80.0)))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_match_features(
    results_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    squad_values: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the per-match feature matrix and the target column."""
    cfg = load_config()
    df = results_df.copy().sort_values("date").reset_index(drop=True)
    df["home_team"] = df["home_team"].map(normalise_team)
    df["away_team"] = df["away_team"].map(normalise_team)

    if squad_values is None:
        value_lookup = dict(SQUAD_VALUES)
    else:
        value_lookup = dict(zip(squad_values["team"], squad_values["value_eur_m"]))

    elo_lookup = _build_elo_lookup(elo_df)

    # Rolling stats containers
    last_n_results: dict[str, deque] = {}
    last_n_for: dict[str, deque] = {}
    last_n_against: dict[str, deque] = {}
    last_match_date: dict[str, pd.Timestamp] = {}
    h2h_wc_results: dict[tuple[str, str], deque] = {}

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
        match_dt64 = (
            np.datetime64(match_date.to_datetime64())
            if hasattr(match_date, "to_datetime64")
            else np.datetime64(match_date)
        )

        # ---- ELO
        elo_h = _elo_at(elo_lookup, h, match_dt64)
        elo_a = _elo_at(elo_lookup, a, match_dt64)

        # ---- Squad value
        v_h_log = _value_for(h, value_lookup)
        v_a_log = _value_for(a, value_lookup)
        v_h_raw = _value_raw(h, value_lookup)
        v_a_raw = _value_raw(a, value_lookup)
        tier_h = _value_tier(v_h_raw)
        tier_a = _value_tier(v_a_raw)

        # ---- Tournament metadata
        weight, is_wc = _classify_tournament(str(row.tournament))
        is_neutral = int(bool(getattr(row, "neutral", False)))

        # ---- Days since previous match
        prev_h = last_match_date.get(h)
        prev_a = last_match_date.get(a)
        ds_h = float((match_date - prev_h).days) if prev_h is not None else 30.0
        ds_a = float((match_date - prev_a).days) if prev_a is not None else 30.0

        # ---- Form & rolling goal averages
        def _avg(d: deque, default: float) -> float:
            return float(sum(d) / len(d)) if len(d) > 0 else default

        form_h = _avg(last_n_results.setdefault(h, deque(maxlen=5)), 0.5)
        form_a = _avg(last_n_results.setdefault(a, deque(maxlen=5)), 0.5)
        gs10_h = _avg(last_n_for.setdefault(h, deque(maxlen=10)), 1.2)
        gc10_h = _avg(last_n_against.setdefault(h, deque(maxlen=10)), 1.2)
        gs10_a = _avg(last_n_for.setdefault(a, deque(maxlen=10)), 1.2)
        gc10_a = _avg(last_n_against.setdefault(a, deque(maxlen=10)), 1.2)

        # ---- H2H WC win rate (home perspective)
        pair = tuple(sorted([h, a]))
        h2h_deque = h2h_wc_results.setdefault(pair, deque(maxlen=5))
        if h2h_deque:
            wr_pair0 = sum(h2h_deque) / len(h2h_deque)
            h2h_home_wr = wr_pair0 if pair[0] == h else 1.0 - wr_pair0
        else:
            h2h_home_wr = 0.5

        # ---- Qualifying xGD (confederation-discounted)
        xg_h = _qualifying_xgd(h)
        xg_a = _qualifying_xgd(a)

        # ---- FIFA rank
        rank_h = _fifa_rank(h)
        rank_a = _fifa_rank(a)
        rank_diff = float(rank_a - rank_h)  # positive when home is higher-ranked
        log_rank_ratio = float(np.log((rank_a + 1) / (rank_h + 1)))

        # ---- WC knockout rate
        ko_h = _wc_ko_rate(h)
        ko_a = _wc_ko_rate(a)
        ko_diff = ko_h - ko_a

        # ---- Outcome
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

        # ---- Time-decay sample weight
        days_ago = max((latest_date - match_date).days, 0)
        sw = float(np.exp(-decay_lambda * days_ago))

        # ---- Append row
        out["elo_home"].append(elo_h)
        out["elo_away"].append(elo_a)
        out["elo_diff"].append(elo_h - elo_a)
        out["value_home"].append(v_h_log)
        out["value_away"].append(v_a_log)
        out["value_ratio"].append(v_h_log - v_a_log)
        out["value_tier_home"].append(tier_h)
        out["value_tier_away"].append(tier_a)
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
        out["fifa_rank_home"].append(rank_h)
        out["fifa_rank_away"].append(rank_a)
        out["rank_diff"].append(rank_diff)
        out["log_rank_ratio"].append(log_rank_ratio)
        out["wc_knockout_rate_home"].append(ko_h)
        out["wc_knockout_rate_away"].append(ko_a)
        out["wc_knockout_rate_diff"].append(ko_diff)

        extra["date"].append(match_date)
        extra["home_team"].append(h)
        extra["away_team"].append(a)
        extra["home_score"].append(hs)
        extra["away_score"].append(as_)
        extra["outcome"].append(outcome)
        extra["sample_weight"].append(sw)

        # ---- Update rolling state
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
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logger.info(
        "Built feature matrix with %d rows and %d feature columns (half-life=%dd)",
        len(feat_df),
        len(FEATURE_COLUMNS),
        int(half_life),
    )
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
    "CONFEDERATION_DIFFICULTY",
    "TEAM_CONFEDERATION",
    "FIFA_RANKING_2026",
    "WC_KNOCKOUT_WIN_RATE",
    "DEFAULT_KO_RATE",
    "FEATURE_COLUMNS",
    "build_match_features",
    "split_features_target",
]
