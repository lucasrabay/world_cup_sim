"""Data ingestion layer.

Responsibilities:
    * Download (and cache) the international_results CSV files.
    * Build (or fetch) team ELO ratings.
    * Provide hardcoded squad values and qualifying xGD for the 48 teams
      participating in WC 2026.
    * Encode the WC 2026 group draw and host nations.

All network access is best-effort: if a download fails we fall back to local
caches or computed approximations so the rest of the pipeline keeps working.
"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .utils import (
    DATA_RAW,
    get_logger,
    load_config,
    normalise_team,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------
SQUAD_VALUES: dict[str, float] = {
    "England": 1250, "France": 1180, "Spain": 1050, "Germany": 980,
    "Brazil": 920, "Portugal": 870, "Argentina": 820, "Netherlands": 760,
    "Belgium": 580, "Norway": 520, "Colombia": 410, "Uruguay": 380,
    "Croatia": 340, "Japan": 320, "Senegal": 290, "Morocco": 270,
    "USA": 390, "Mexico": 280, "Canada": 310, "Australia": 180,
    "South Korea": 260, "Switzerland": 430, "Austria": 310, "Sweden": 280,
    "Turkey": 290, "Ecuador": 160, "Ivory Coast": 230, "Egypt": 140,
    "Algeria": 130, "Ghana": 150, "Tunisia": 90, "South Africa": 80,
    "Saudi Arabia": 110, "Qatar": 90, "Iran": 120,
    "Paraguay": 130, "Panama": 70, "Bosnia and Herzegovina": 110,
    "Scotland": 220, "Czechia": 180, "Haiti": 45, "New Zealand": 55,
    "Cape Verde": 40, "Jordan": 50, "Iraq": 60, "Uzbekistan": 65,
    "DR Congo": 70, "Curacao": 35,
}


WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


HOST_NATIONS: list[str] = ["USA", "Mexico", "Canada"]


# Approximate FIFA-rank derived ELO fallback for teams without history.
# Calibrated so a top team sits ~2050 and a minnow ~1300.
FALLBACK_ELO: dict[str, float] = {
    "England": 2050, "France": 2030, "Spain": 2040, "Germany": 1990,
    "Brazil": 2010, "Portugal": 2000, "Argentina": 2020, "Netherlands": 1970,
    "Belgium": 1900, "Norway": 1880, "Colombia": 1830, "Uruguay": 1840,
    "Croatia": 1860, "Japan": 1820, "Senegal": 1810, "Morocco": 1830,
    "USA": 1810, "Mexico": 1790, "Canada": 1780, "Australia": 1740,
    "South Korea": 1780, "Switzerland": 1850, "Austria": 1830, "Sweden": 1790,
    "Turkey": 1780, "Ecuador": 1750, "Ivory Coast": 1770, "Egypt": 1730,
    "Algeria": 1720, "Ghana": 1700, "Tunisia": 1700, "South Africa": 1680,
    "Saudi Arabia": 1670, "Qatar": 1640, "Iran": 1730,
    "Paraguay": 1700, "Panama": 1640, "Bosnia and Herzegovina": 1700,
    "Scotland": 1750, "Czechia": 1720, "Haiti": 1500, "New Zealand": 1530,
    "Cape Verde": 1580, "Jordan": 1560, "Iraq": 1610, "Uzbekistan": 1600,
    "DR Congo": 1620, "Curacao": 1500,
}


# ---------------------------------------------------------------------------
# Results dataset
# ---------------------------------------------------------------------------
def _http_get_with_retry(url: str, retries: int = 3, backoff: float = 3.0) -> bytes | None:
    """Best-effort HTTP GET. Returns bytes on success, None on permanent failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.content
            logger.warning("GET %s -> status %s (attempt %d)", url, resp.status_code, attempt + 1)
        except requests.RequestException as exc:  # pragma: no cover - network only
            logger.warning("GET %s failed: %s (attempt %d)", url, exc, attempt + 1)
        if attempt < retries - 1:
            time.sleep(backoff)
    return None


def download_results() -> pd.DataFrame:
    """Download (and cache) the international football results CSV.

    Filters to matches from ``config.data.min_year`` onwards. Team names are
    normalised to canonical forms used elsewhere in this codebase.
    """
    cfg = load_config()
    cache = DATA_RAW / "results.parquet"

    if cache.exists():
        logger.info("Loading cached results from %s", cache)
        df = pd.read_parquet(cache)
    else:
        url = cfg["data"]["results_url"]
        logger.info("Downloading results from %s", url)
        payload = _http_get_with_retry(url)
        if payload is None:
            logger.error("Could not download results CSV; returning synthetic stub")
            return _synthetic_results_stub()

        df = pd.read_csv(io.BytesIO(payload))
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(cache, index=False)

    # Always re-apply the date filter / normalisation since the cache is raw.
    min_year = int(cfg["data"]["min_year"])
    df = df[df["date"] >= pd.Timestamp(year=min_year, month=1, day=1)].copy()
    df["home_team"] = df["home_team"].map(normalise_team)
    df["away_team"] = df["away_team"].map(normalise_team)
    df["country"] = df["country"].map(normalise_team)
    # Drop unplayed / unknown scores so downstream code can rely on int casts.
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d matches from %d onwards", len(df), min_year)
    return df


def _synthetic_results_stub() -> pd.DataFrame:
    """Tiny synthetic dataset used as a last resort when the download fails."""
    teams = list(SQUAD_VALUES.keys())
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp("2018-01-01")
    for i in range(2000):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append(
            {
                "date": start + pd.Timedelta(days=int(i / 2)),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(1.5)),
                "away_score": int(rng.poisson(1.2)),
                "tournament": rng.choice(["Friendly", "FIFA World Cup qualification"]),
                "city": "Synthetic",
                "country": h,
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ELO ratings
# ---------------------------------------------------------------------------
def _compute_elo_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """Fallback ELO computed iteratively from the results dataset.

    K factors:
        * FIFA World Cup       -> 60
        * Continental finals   -> 50
        * Confederations Cup   -> 40
        * Qualification        -> 30
        * Friendly             -> 20
    """
    teams = pd.unique(pd.concat([results["home_team"], results["away_team"]]))
    elo: dict[str, float] = {t: FALLBACK_ELO.get(t, 1500.0) for t in teams}

    history: list[dict] = []

    def k_for(tournament: str) -> float:
        t = tournament.lower()
        if "world cup" in t and "qualification" not in t:
            return 60.0
        if any(s in t for s in ("uefa euro", "copa america", "africa cup", "asian cup")):
            return 50.0
        if "confederations" in t or "nations league" in t:
            return 40.0
        if "qualification" in t:
            return 30.0
        return 20.0

    for row in results.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh, ra = elo[h], elo[a]
        # Home advantage proxy
        rh_eff = rh if getattr(row, "neutral", False) else rh + 65.0
        e_h = 1.0 / (1.0 + 10 ** ((ra - rh_eff) / 400.0))
        if row.home_score > row.away_score:
            s_h = 1.0
        elif row.home_score == row.away_score:
            s_h = 0.5
        else:
            s_h = 0.0
        # Goal-difference multiplier
        gd = abs(int(row.home_score) - int(row.away_score))
        g_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)

        k = k_for(str(row.tournament)) * g_mult
        elo[h] = rh + k * (s_h - e_h)
        elo[a] = ra + k * ((1 - s_h) - (1 - e_h))

        history.append({"team": h, "date": row.date, "elo": elo[h]})
        history.append({"team": a, "date": row.date, "elo": elo[a]})

    return pd.DataFrame(history)


def load_elo_ratings(teams: Iterable[str], results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a long DataFrame with columns [team, date, elo].

    Strategy: try the clubelo API for each requested team (it actually serves
    only club ratings, so this is best-effort and almost always falls through).
    On failure we compute team ratings iteratively from the historical results.
    """
    cache = DATA_RAW / "elo_ratings.parquet"
    if cache.exists():
        logger.info("Loading cached ELO ratings from %s", cache)
        return pd.read_parquet(cache)

    cfg = load_config()
    template = cfg["data"]["elo_url"]
    api_frames: list[pd.DataFrame] = []
    for team in teams:
        url = template.format(team=team.replace(" ", "%20"))
        payload = _http_get_with_retry(url, retries=1, backoff=3.0)
        if payload is None:
            continue
        try:
            df = pd.read_csv(io.BytesIO(payload))
            if {"From", "Elo"}.issubset(df.columns):
                api_frames.append(
                    pd.DataFrame(
                        {
                            "team": team,
                            "date": pd.to_datetime(df["From"]),
                            "elo": df["Elo"].astype(float),
                        }
                    )
                )
        except Exception as exc:  # pragma: no cover - parsing best-effort
            logger.warning("Could not parse ELO for %s: %s", team, exc)

    api_frames = [f for f in api_frames if not f.empty]
    if api_frames:
        ratings = pd.concat(api_frames, ignore_index=True)
        logger.info("Fetched ELO ratings for %d teams via clubelo API", len(api_frames))
    else:
        if results is None:
            logger.warning("No results provided for ELO fallback; using static FALLBACK_ELO")
            today = pd.Timestamp(date.today())
            ratings = pd.DataFrame(
                [{"team": t, "date": today, "elo": FALLBACK_ELO.get(t, 1500.0)} for t in teams]
            )
        else:
            logger.info("Computing ELO ratings iteratively from results")
            ratings = _compute_elo_from_results(results)

    ratings = ratings.sort_values(["team", "date"]).reset_index(drop=True)
    _warn_if_stale(ratings, list(teams))
    ratings.to_parquet(cache, index=False)
    return ratings


def _warn_if_stale(ratings: pd.DataFrame, requested_teams: list[str], days: int = 60) -> None:
    """Log a warning if the most recent ELO observation for any requested team
    is older than ``days`` days. Stale ratings produce mis-calibrated lambdas."""
    if ratings.empty:
        logger.warning("ELO frame is empty — predictions will fall back to FALLBACK_ELO")
        return
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    latest = ratings.groupby("team")["date"].max()
    stale = latest[latest < cutoff]
    missing = [t for t in requested_teams if t not in latest.index]
    if len(stale) > 0:
        sample = ", ".join(stale.index[:8].tolist()) + ("…" if len(stale) > 8 else "")
        logger.warning("Stale ELO data (>%dd) for %d teams: %s", days, len(stale), sample)
    if missing:
        sample = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
        logger.warning("No ELO history for %d requested teams: %s", len(missing), sample)


def load_squad_values() -> pd.DataFrame:
    """Return a frame of team -> squad value (EUR millions, log)."""
    rows = [
        {"team": team, "value_eur_m": float(v), "log_value": float(np.log(v))}
        for team, v in SQUAD_VALUES.items()
    ]
    df = pd.DataFrame(rows)
    return df


def load_wc2026_fixtures() -> dict:
    """Return WC 2026 group draw plus the host nations list."""
    return {"groups": WC2026_GROUPS, "hosts": HOST_NATIONS}


__all__ = [
    "SQUAD_VALUES",
    "WC2026_GROUPS",
    "HOST_NATIONS",
    "FALLBACK_ELO",
    "download_results",
    "load_elo_ratings",
    "load_squad_values",
    "load_wc2026_fixtures",
]
