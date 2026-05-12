"""Pre-tournament betting-odds integration.

Pulls outright FIFA World Cup winner odds from The Odds API and turns them into
a per-team implied-probability lookup that downstream modules can use as a
feature and as a fourth ensemble component.

Network access is best-effort: if the API key is missing, the call fails, or
quotas are exhausted we fall back silently to a hardcoded mid-May 2026 snapshot
so the pipeline always completes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .utils import DATA_RAW, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mid-May 2026 outright-winner decimal odds (approximate European market).
# ---------------------------------------------------------------------------
FALLBACK_ODDS: dict[str, float] = {
    "Spain": 5.50,
    "France": 6.00,
    "England": 7.00,
    "Argentina": 7.50,
    "Brazil": 8.00,
    "Germany": 12.00,
    "Portugal": 13.00,
    "Netherlands": 15.00,
    "Norway": 17.00,
    "Morocco": 21.00,
    "Colombia": 23.00,
    "Belgium": 26.00,
    "Uruguay": 29.00,
    "Japan": 29.00,
    "Croatia": 34.00,
    "USA": 34.00,
    "Senegal": 41.00,
    "Mexico": 51.00,
    "Canada": 51.00,
    "South Korea": 51.00,
    "Switzerland": 51.00,
    "Austria": 51.00,
    "Sweden": 67.00,
    "Turkey": 67.00,
    "Ecuador": 81.00,
    "Denmark": 81.00,
    "Ivory Coast": 101.00,
    "Egypt": 101.00,
    "Scotland": 126.00,
    "Algeria": 151.00,
    "Ghana": 151.00,
    "Tunisia": 151.00,
    "South Africa": 201.00,
    "Australia": 201.00,
    "Paraguay": 251.00,
    "Bosnia and Herzegovina": 251.00,
    "Czechia": 251.00,
    "Saudi Arabia": 301.00,
    "Iran": 301.00,
    "Qatar": 501.00,
    "DR Congo": 501.00,
    "Uzbekistan": 501.00,
    "Jordan": 501.00,
    "Iraq": 751.00,
    "Cape Verde": 751.00,
    "New Zealand": 1001.00,
    "Panama": 1001.00,
    "Haiti": 1001.00,
    "Curacao": 1001.00,
}


_ODDS_CACHE_PATH = DATA_RAW / "odds.json"
_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours per task spec
_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"


# ---------------------------------------------------------------------------
def odds_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to a raw implied probability (no vig removal)."""
    d = float(decimal_odds)
    if d <= 0:
        return 0.0
    return 1.0 / d


# ---------------------------------------------------------------------------
def _fallback_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"team": t, "decimal_odds": float(o)} for t, o in FALLBACK_ODDS.items()]
    )


def _parse_api_response(payload: list) -> dict[str, float]:
    """Parse The Odds API JSON, return team -> median decimal_odds.

    The API returns one event per market; outright winner odds are nested as
    bookmaker → markets → outcomes. We average across bookmakers when several
    quote the same team.
    """
    accum: dict[str, list[float]] = {}
    for event in payload or []:
        for bookmaker in event.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") not in ("outrights", "h2h"):
                    continue
                for outcome in market.get("outcomes", []) or []:
                    name = outcome.get("name") or outcome.get("description")
                    price = outcome.get("price")
                    if not name or not isinstance(price, (int, float)) or price <= 1.0:
                        continue
                    accum.setdefault(str(name), []).append(float(price))
    return {team: float(sum(prices) / len(prices)) for team, prices in accum.items() if prices}


def fetch_tournament_odds(api_key: str | None = None) -> pd.DataFrame:
    """Return a DataFrame [team, decimal_odds].

    Strategy:
        1. Read the 6h-fresh cache if present.
        2. Otherwise call The Odds API (key from arg or ``ODDS_API_KEY`` env).
        3. On any failure, fall back to ``FALLBACK_ODDS`` and log a warning.
    """
    if _ODDS_CACHE_PATH.exists():
        try:
            cached = json.loads(_ODDS_CACHE_PATH.read_text())
            age = time.time() - float(cached.get("timestamp", 0))
            if age < _CACHE_TTL_SECONDS:
                df = pd.DataFrame(cached["data"])
                if not df.empty:
                    logger.info("Using cached odds (%dm old, %d teams)", int(age / 60), len(df))
                    return df
        except Exception as exc:  # pragma: no cover - cache best-effort
            logger.warning("Could not read odds cache: %s", exc)

    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        logger.warning("No ODDS_API_KEY supplied — falling back to hardcoded odds")
        return _fallback_df()

    try:
        resp = requests.get(
            _API_URL,
            params={
                "regions": "eu",
                "markets": "h2h,outrights",
                "oddsFormat": "decimal",
                "apiKey": api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(
                "Odds API HTTP %s (%s) — falling back to hardcoded odds",
                resp.status_code, (resp.text or "")[:120],
            )
            return _fallback_df()
        team_odds = _parse_api_response(resp.json())
        if not team_odds:
            logger.warning("Odds API returned no usable outrights — falling back")
            return _fallback_df()
        df = pd.DataFrame(
            [{"team": t, "decimal_odds": o} for t, o in team_odds.items()]
        )
        try:
            _ODDS_CACHE_PATH.write_text(json.dumps(
                {"timestamp": time.time(), "data": df.to_dict(orient="records")},
                indent=2,
            ))
        except Exception as exc:  # pragma: no cover - cache write best-effort
            logger.warning("Failed to write odds cache: %s", exc)
        logger.info("Fetched %d outright odds from The Odds API", len(df))
        return df
    except Exception as exc:  # pragma: no cover - network only
        logger.warning("Odds API call failed (%s) — falling back to hardcoded", exc)
        return _fallback_df()


def build_odds_feature(odds_df: pd.DataFrame, teams: list[str]) -> dict[str, float]:
    """Return team -> normalised implied win probability summing to 1.0 across ``teams``.

    Teams with no odds entry are imputed with the mean of the known teams,
    then the whole vector is rescaled so it sums to 1.0 (proportional vig
    removal across the tournament field).
    """
    raw: dict[str, float] = {}
    if odds_df is not None and not odds_df.empty:
        for row in odds_df.itertuples(index=False):
            team = getattr(row, "team", None)
            decimal = getattr(row, "decimal_odds", None)
            if team is None or decimal is None:
                continue
            raw[str(team)] = odds_to_implied_prob(float(decimal))

    out: dict[str, float] = {}
    known_vals: list[float] = []
    missing: list[str] = []
    for t in teams:
        p = raw.get(t)
        if p is None or p <= 0:
            missing.append(t)
        else:
            out[t] = float(p)
            known_vals.append(float(p))

    if missing:
        # Replace missing with average of known implied probs; if everything
        # is missing fall back to uniform.
        mean_known = float(sum(known_vals) / len(known_vals)) if known_vals else 1.0 / max(len(teams), 1)
        for t in missing:
            out[t] = mean_known

    total = float(sum(out.values()))
    if total <= 0:
        u = 1.0 / max(len(teams), 1)
        return {t: u for t in teams}
    return {t: float(p / total) for t, p in out.items()}


__all__ = [
    "FALLBACK_ODDS",
    "odds_to_implied_prob",
    "fetch_tournament_odds",
    "build_odds_feature",
]
