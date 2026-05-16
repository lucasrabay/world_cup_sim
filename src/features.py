"""Feature engineering for the international results dataset.

Builds the per-match feature matrix consumed by the XGBoost classifier and the
match-level metadata required by the Dixon-Coles fitter.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data_loader import FALLBACK_ELO, SQUAD_VALUES
from .odds_loader import FALLBACK_ODDS, odds_to_implied_prob
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


# Confederation lookup. Covers WC 2026 participants plus the historical teams
# that show up in the international_results dataset — needed for
# ``fit_confederation_difficulty`` to actually find cross-confederation rows.
TEAM_CONFEDERATION: dict[str, str] = {
    # ----- UEFA (Europe) -----
    "England": "UEFA", "France": "UEFA", "Spain": "UEFA", "Germany": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Norway": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA", "Sweden": "UEFA",
    "Turkey": "UEFA", "Bosnia and Herzegovina": "UEFA", "Scotland": "UEFA", "Czechia": "UEFA",
    "Italy": "UEFA", "Russia": "UEFA", "Soviet Union": "UEFA", "Yugoslavia": "UEFA",
    "Serbia": "UEFA", "Serbia and Montenegro": "UEFA", "Poland": "UEFA",
    "Romania": "UEFA", "Bulgaria": "UEFA", "Denmark": "UEFA", "Hungary": "UEFA",
    "Greece": "UEFA", "Republic of Ireland": "UEFA", "Northern Ireland": "UEFA",
    "Wales": "UEFA", "Iceland": "UEFA", "Ukraine": "UEFA", "Slovakia": "UEFA",
    "Slovenia": "UEFA", "North Macedonia": "UEFA", "East Germany": "UEFA",
    "West Germany": "UEFA", "Albania": "UEFA", "Belarus": "UEFA",
    "Estonia": "UEFA", "Finland": "UEFA", "Georgia": "UEFA", "Latvia": "UEFA",
    "Lithuania": "UEFA", "Luxembourg": "UEFA", "Malta": "UEFA",
    "Moldova": "UEFA", "Montenegro": "UEFA", "Kosovo": "UEFA", "Cyprus": "UEFA",
    "Andorra": "UEFA", "Armenia": "UEFA", "Azerbaijan": "UEFA", "Faroe Islands": "UEFA",
    "Gibraltar": "UEFA", "Israel": "UEFA", "Kazakhstan": "UEFA", "Liechtenstein": "UEFA",
    "San Marino": "UEFA",
    # ----- CONMEBOL (South America) -----
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL", "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL",
    # ----- CONCACAF (N/C America + Caribbean) -----
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curacao": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "El Salvador": "CONCACAF",
    "Guatemala": "CONCACAF", "Jamaica": "CONCACAF", "Cuba": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "Nicaragua": "CONCACAF",
    "Bermuda": "CONCACAF", "Belize": "CONCACAF",
    # ----- CAF (Africa) -----
    "Senegal": "CAF", "Morocco": "CAF", "Ivory Coast": "CAF", "Egypt": "CAF",
    "Algeria": "CAF", "Ghana": "CAF", "Tunisia": "CAF", "South Africa": "CAF",
    "Cape Verde": "CAF", "DR Congo": "CAF",
    "Cameroon": "CAF", "Nigeria": "CAF", "Angola": "CAF", "Togo": "CAF",
    "Zaire": "CAF", "Sudan": "CAF", "Mali": "CAF", "Burkina Faso": "CAF",
    "Guinea": "CAF", "Zambia": "CAF", "Mozambique": "CAF", "Madagascar": "CAF",
    "Ethiopia": "CAF", "Kenya": "CAF", "Tanzania": "CAF", "Uganda": "CAF",
    "Libya": "CAF", "Mauritania": "CAF", "Benin": "CAF", "Niger": "CAF",
    "Sierra Leone": "CAF", "Liberia": "CAF", "Gabon": "CAF", "Congo": "CAF",
    "Equatorial Guinea": "CAF", "Central African Republic": "CAF",
    "Botswana": "CAF", "Namibia": "CAF", "Zimbabwe": "CAF", "Malawi": "CAF",
    "Burundi": "CAF", "Rwanda": "CAF", "Chad": "CAF", "Comoros": "CAF",
    "Eritrea": "CAF", "Eswatini": "CAF", "Lesotho": "CAF", "Mauritius": "CAF",
    "Sao Tome and Principe": "CAF", "Seychelles": "CAF", "Somalia": "CAF",
    "South Sudan": "CAF", "Djibouti": "CAF", "Gambia": "CAF",
    "Guinea-Bissau": "CAF",
    # ----- AFC (Asia + AUS) -----
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Saudi Arabia": "AFC",
    "Qatar": "AFC", "Iran": "AFC", "Jordan": "AFC", "Iraq": "AFC", "Uzbekistan": "AFC",
    "China PR": "AFC", "China": "AFC", "North Korea": "AFC",
    "Kuwait": "AFC", "United Arab Emirates": "AFC", "Bahrain": "AFC", "Oman": "AFC",
    "Lebanon": "AFC", "Syria": "AFC", "Palestine": "AFC", "Yemen": "AFC",
    "Indonesia": "AFC", "Malaysia": "AFC", "Singapore": "AFC", "Thailand": "AFC",
    "Vietnam": "AFC", "Philippines": "AFC", "Myanmar": "AFC", "Cambodia": "AFC",
    "Hong Kong": "AFC", "Taiwan": "AFC", "Chinese Taipei": "AFC",
    "Tajikistan": "AFC", "Turkmenistan": "AFC", "Kyrgyzstan": "AFC", "Afghanistan": "AFC",
    "Maldives": "AFC", "Nepal": "AFC", "Sri Lanka": "AFC", "Pakistan": "AFC",
    "Bangladesh": "AFC", "India": "AFC", "Bhutan": "AFC", "Brunei": "AFC",
    "Macau": "AFC", "Mongolia": "AFC", "Laos": "AFC", "Timor-Leste": "AFC",
    "Guam": "AFC", "Northern Mariana Islands": "AFC",
    # ----- OFC (Oceania) -----
    "New Zealand": "OFC",
    "Fiji": "OFC", "Papua New Guinea": "OFC", "Solomon Islands": "OFC",
    "Tahiti": "OFC", "Vanuatu": "OFC", "Samoa": "OFC", "Tonga": "OFC",
    "American Samoa": "OFC", "Cook Islands": "OFC",
}


# ---------------------------------------------------------------------------
# Historical WC group draws (for retrofitting path-difficulty features onto
# the held-out WC 2018 + WC 2022 matches). Team names match the canonical
# normalisation produced by :func:`normalise_team`.
# ---------------------------------------------------------------------------
HISTORICAL_WC_GROUPS: dict[int, dict[str, list[str]]] = {
    2018: {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
    2022: {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "USA", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
}

# 32-team format halves (top half = ABCD, bottom = EFGH).
HISTORICAL_BRACKET_HALF: dict[str, int] = {
    "A": 0, "B": 0, "C": 0, "D": 0,
    "E": 1, "F": 1, "G": 1, "H": 1,
}

# 32-team R16 pairings (within-half by adjacent letter).
HISTORICAL_ADJACENT_GROUP: dict[str, str] = {
    "A": "B", "B": "A",
    "C": "D", "D": "C",
    "E": "F", "F": "E",
    "G": "H", "H": "G",
}

# 48-team WC 2026 halves per the task spec.
WC2026_BRACKET_HALF: dict[str, int] = {
    "A": 0, "C": 0, "E": 0, "G": 0, "I": 0, "K": 0,
    "B": 1, "D": 1, "F": 1, "H": 1, "J": 1, "L": 1,
}

# 48-team R32 adjacent group (within the same half) — used to compute the
# "expected R32 opponent". Within each half we pair groups in the order they
# appear in the bracket template.
WC2026_ADJACENT_GROUP: dict[str, str] = {
    # half 0
    "A": "C", "C": "A",
    "E": "G", "G": "E",
    "I": "K", "K": "I",
    # half 1
    "B": "D", "D": "B",
    "F": "H", "H": "F",
    "J": "L", "L": "J",
}


# Date windows for picking the right historical-WC group lookup.
def _historical_wc_year(match_date: pd.Timestamp) -> int | None:
    """Return 2018 / 2022 if the match falls inside the corresponding WC
    fortnight, else None."""
    if pd.Timestamp("2018-06-14") <= match_date <= pd.Timestamp("2018-07-15"):
        return 2018
    if pd.Timestamp("2022-11-20") <= match_date <= pd.Timestamp("2022-12-18"):
        return 2022
    return None


# Path-feature keys — used both to populate FEATURE_COLUMNS and to zero-fill
# the columns for non-WC matches.
PATH_FEATURE_KEYS: tuple[str, ...] = (
    "group_avg_elo_opponents",
    "group_max_elo_opponent",
    "group_elo_rank",
    "bracket_half",
    "expected_r16_opponent_elo",
    "path_to_final_avg_elo",
)


def compute_path_features(
    team: str,
    groups: dict[str, list[str]],
    elo_lookup: dict[str, float],
    bracket_half_map: dict[str, int],
    adjacent_group_map: dict[str, str],
) -> dict[str, float]:
    """Return the six path-difficulty features for ``team`` given a static
    group draw + ELO snapshot.

    Definitions
    -----------
    * ``group_avg_elo_opponents`` — mean ELO of the other 3 teams in the team's
      group.
    * ``group_max_elo_opponent`` — max ELO of the other 3 teams (Group of
      Death ceiling).
    * ``group_elo_rank`` — team's rank within their own group, 1 = highest
      ELO, 4 = lowest.
    * ``bracket_half`` — half-of-bracket assignment (0 or 1).
    * ``expected_r16_opponent_elo`` — ELO of the 2nd-strongest team in the
      adjacent (R32-paired) group — the likely group-stage runner-up that
      would face this team in the early KO round.
    * ``path_to_final_avg_elo`` — round-weighted average opponent ELO across
      the full path including the 3 group games and the 4 KO rounds. Weights:
      group games = 1.0 each, R32 = 1.0, R16 = 1.5, QF = 2.0, SF = 2.5.
      Higher = harder.

      We include the group stage in this aggregate because, with sparse
      cross-confederation data, the strongest signal that bookmakers price in
      for tournament path difficulty is "how forgiving is your group?" — a
      pure KO-only aggregate buries that signal under largely-symmetric half
      strength.
    """
    # Find the team's group.
    own_group: str | None = None
    for letter, members in groups.items():
        if team in members:
            own_group = letter
            break
    if own_group is None:
        return {k: 0.0 for k in PATH_FEATURE_KEYS}

    def _e(t: str) -> float:
        return float(elo_lookup.get(t, FALLBACK_ELO.get(t, 1500.0)))

    members = list(groups[own_group])
    opponents = [t for t in members if t != team]
    opp_elos = [_e(t) for t in opponents]
    own_elo = _e(team)
    group_avg_elo = float(np.mean(opp_elos)) if opp_elos else 0.0
    group_max_elo = float(max(opp_elos)) if opp_elos else 0.0

    # Rank 1 = highest ELO in group.
    sorted_team_elos = sorted(((m, _e(m)) for m in members), key=lambda kv: -kv[1])
    rank_lookup = {m: i + 1 for i, (m, _v) in enumerate(sorted_team_elos)}
    own_rank = float(rank_lookup.get(team, 4))

    half = int(bracket_half_map.get(own_group, 0))

    # Expected R32 opponent — 2nd-strongest team in the adjacent group.
    adj_letter = adjacent_group_map.get(own_group)
    if adj_letter and adj_letter in groups:
        adj_elos_desc = sorted((_e(t) for t in groups[adj_letter]), reverse=True)
        expected_r16_opp = float(adj_elos_desc[1] if len(adj_elos_desc) >= 2 else adj_elos_desc[0])
    else:
        # Fall back to average of all other group winners in the same half.
        fallback_pool: list[float] = []
        for g, mems in groups.items():
            if g == own_group or bracket_half_map.get(g) != half:
                continue
            fallback_pool.append(max(_e(t) for t in mems))
        expected_r16_opp = float(np.mean(fallback_pool)) if fallback_pool else group_avg_elo

    # Round-by-round expected opponents inside the team's half. We use the
    # top-2 ELO of each *other* group in the same half as the candidate pool;
    # rounds further down the bracket draw against successively stronger
    # remaining opponents (mean of pool, then max of pool weighted higher).
    half_pool: list[float] = []
    for g, mems in groups.items():
        if g == own_group or bracket_half_map.get(g) != half:
            continue
        sorted_g = sorted((_e(t) for t in mems), reverse=True)
        half_pool.extend(sorted_g[:2])
    if half_pool:
        half_mean = float(np.mean(half_pool))
        half_pool_sorted = sorted(half_pool, reverse=True)
        # Surrogate "best remaining opponent" rises round-by-round.
        r16_opp = half_mean
        qf_opp = float(np.mean(half_pool_sorted[: max(len(half_pool_sorted) // 2, 1)]))
        sf_opp = float(np.mean(half_pool_sorted[: max(len(half_pool_sorted) // 3, 1)]))
    else:
        r16_opp = qf_opp = sf_opp = group_avg_elo

    # Weighted path average: group games + four KO rounds.
    weights = {
        "group": 1.0,
        "r32": 1.0,
        "r16": 1.5,
        "qf": 2.0,
        "sf": 2.5,
    }
    weighted_sum = (
        weights["group"] * sum(opp_elos)            # 3 group games, weight 1 each
        + weights["r32"] * expected_r16_opp
        + weights["r16"] * r16_opp
        + weights["qf"] * qf_opp
        + weights["sf"] * sf_opp
    )
    total_weight = 3 * weights["group"] + weights["r32"] + weights["r16"] + weights["qf"] + weights["sf"]
    path_to_final = float(weighted_sum / total_weight)

    return {
        "group_avg_elo_opponents": group_avg_elo,
        "group_max_elo_opponent": group_max_elo,
        "group_elo_rank": own_rank,
        "bracket_half": float(half),
        "expected_r16_opponent_elo": float(expected_r16_opp),
        "path_to_final_avg_elo": path_to_final,
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


# Conventional canonical ordering used when post-processing the Bradley-Terry
# fit — empirical fits with limited cross-confederation samples can flip
# adjacent confederations, so we enforce the ordering UEFA > CONMEBOL > AFC >
# CAF > CONCACAF > OFC by sorting the fitted magnitudes into these slots.
_CONFEDERATION_ORDER: tuple[str, ...] = (
    "UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC",
)


# ---------------------------------------------------------------------------
# Confederation difficulty — empirical fit
# ---------------------------------------------------------------------------
def fit_confederation_difficulty(
    results_df: pd.DataFrame,
    tournament_filter: str = "FIFA World Cup",
    min_year: int = 1990,
    min_matches: int = 50,
) -> dict[str, float]:
    """Fit a per-confederation strength scalar from cross-confederation WC matches.

    Method
    ------
    1. Filter to WC matches (excluding qualification) from ``min_year`` onwards.
    2. Keep only matches where home and away teams belong to **different**
       confederations — this is the only unbiased cross-confederation signal.
    3. For each ordered confederation pair (a, b), compute
       ``win_rate_a = (wins_a + 0.5 * draws) / total``.
    4. Anchor UEFA at 1.0 and fit a single scalar per other confederation by
       minimising sum-of-squared-errors between the observed pair win rates
       and Bradley-Terry-implied rates ``s_a / (s_a + s_b)``, weighted by
       sample count.
    5. Clamp results into the plausible range [0.30, 1.00].
    6. Apply the canonical UEFA > CONMEBOL > AFC > CAF > CONCACAF > OFC
       ordering by sorting fitted magnitudes into these slots — this prevents
       small-sample inversions on adjacent confederations.

    Returns
    -------
    dict mapping confederation name → scalar in [0.30, 1.00] with UEFA = 1.0.
    Falls back to the conventional hardcoded values when there are too few
    cross-confederation matches to fit reliably.
    """
    fallback = dict(CONFEDERATION_DIFFICULTY)
    if results_df is None or results_df.empty:
        logger.warning("No results frame supplied; using hardcoded confederation scalars")
        return fallback

    df = results_df.copy()
    df["tournament"] = df["tournament"].astype(str)
    is_wc = (
        df["tournament"].str.contains(tournament_filter, case=False, na=False)
        & ~df["tournament"].str.contains("qualification", case=False, na=False)
    )
    df = df[is_wc].copy()
    df = df[df["date"] >= pd.Timestamp(year=min_year, month=1, day=1)]
    df["home_conf"] = df["home_team"].map(TEAM_CONFEDERATION)
    df["away_conf"] = df["away_team"].map(TEAM_CONFEDERATION)
    df = df.dropna(subset=["home_conf", "away_conf"])
    cross = df[df["home_conf"] != df["away_conf"]]

    if len(cross) < min_matches:
        logger.warning(
            "Only %d cross-confederation WC matches since %d (need %d) — using hardcoded scalars",
            len(cross), min_year, min_matches,
        )
        return fallback

    # Build per-pair (canonical alphabetical) accumulators: total matches and
    # the share won by the first confederation in the sorted pair.
    pair_first_score: dict[tuple[str, str], float] = {}
    pair_total: dict[tuple[str, str], int] = {}
    for row in cross.itertuples(index=False):
        h_conf, a_conf = row.home_conf, row.away_conf
        pair = tuple(sorted([h_conf, a_conf]))
        if row.home_score > row.away_score:
            home_share = 1.0
        elif row.home_score < row.away_score:
            home_share = 0.0
        else:
            home_share = 0.5
        first_share = home_share if h_conf == pair[0] else 1.0 - home_share
        pair_first_score[pair] = pair_first_score.get(pair, 0.0) + first_share
        pair_total[pair] = pair_total.get(pair, 0) + 1

    # Pair observations weighted by sample count.
    pair_obs: list[tuple[str, str, float, int]] = []
    for pair, total in pair_total.items():
        if total < 2:
            continue
        observed = pair_first_score[pair] / total
        pair_obs.append((pair[0], pair[1], observed, total))
    if not pair_obs:
        logger.warning("No usable confederation pairs — using hardcoded scalars")
        return fallback

    confs = sorted({c for c in TEAM_CONFEDERATION.values()} | {p[0] for p in pair_obs} | {p[1] for p in pair_obs})
    # Anchor UEFA at 1.0, fit the rest.
    fitted_confs = [c for c in confs if c != "UEFA"]
    n_params = len(fitted_confs)

    def _scalars(params: np.ndarray) -> dict[str, float]:
        out = {"UEFA": 1.0}
        for i, c in enumerate(fitted_confs):
            out[c] = float(params[i])
        return out

    def _loss(params: np.ndarray) -> float:
        s = _scalars(params)
        sse = 0.0
        for a, b, observed, total in pair_obs:
            sa, sb = s.get(a, 1e-6), s.get(b, 1e-6)
            if sa + sb <= 0:
                continue
            pred = sa / (sa + sb)
            sse += float(total) * (pred - observed) ** 2
        return sse

    x0 = np.full(n_params, 0.7)
    bounds = [(0.10, 1.30)] * n_params
    res = minimize(_loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    raw_fit = _scalars(res.x)

    # Clamp into the allowed band and warn on anything outside [0.30, 1.00].
    clamped: dict[str, float] = {}
    for c, v in raw_fit.items():
        if not (0.30 <= v <= 1.00):
            logger.warning("Confederation scalar for %s out of range (%.3f) — clamping", c, v)
        clamped[c] = float(np.clip(v, 0.30, 1.00))
    clamped["UEFA"] = 1.0  # always preserve the anchor exactly

    # Enforce canonical ordering by reshuffling the fitted magnitudes into the
    # canonical slots — preserves the magnitudes the data wants, but pins
    # which confederation receives which.
    canonical = [c for c in _CONFEDERATION_ORDER if c in clamped]
    sorted_values = sorted(
        (clamped[c] for c in canonical), reverse=True,
    )
    final: dict[str, float] = {}
    for slot, val in zip(canonical, sorted_values):
        final[slot] = val
    # Make sure all six canonical confederations have an entry (fall back if missing).
    for c in _CONFEDERATION_ORDER:
        final.setdefault(c, fallback.get(c, 0.5))
    final["UEFA"] = 1.0  # anchor invariant after sorting
    return final


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
    "odds_implied_prob_home",
    "odds_implied_prob_away",
    "odds_ratio",
    # ---- Path-difficulty (tournament-context) features ----
    # Encoded with the suffix _home/_away — non-WC matches are zero-filled
    # so XGB can route on the existing ``is_wc`` flag.
    "group_avg_elo_opp_home",
    "group_avg_elo_opp_away",
    "group_max_elo_opp_home",
    "group_max_elo_opp_away",
    "group_elo_rank_home",
    "group_elo_rank_away",
    "bracket_half_home",
    "bracket_half_away",
    "expected_r16_opp_elo_home",
    "expected_r16_opp_elo_away",
    "path_to_final_avg_elo_home",
    "path_to_final_avg_elo_away",
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


def _confederation_multiplier(team: str, scalars: dict[str, float] | None = None) -> float:
    table = scalars or CONFEDERATION_DIFFICULTY
    return float(table.get(TEAM_CONFEDERATION.get(team, ""), 1.0))


def _qualifying_xgd(team: str, scalars: dict[str, float] | None = None) -> float:
    """xGD discounted by the team's confederation difficulty.

    ``scalars`` overrides the module-level :data:`CONFEDERATION_DIFFICULTY`
    when supplied — used to plug in empirically fitted values from
    :func:`fit_confederation_difficulty` without mutating global state.
    """
    raw = float(QUALIFYING_XGD.get(team, 0.0))
    return raw * _confederation_multiplier(team, scalars=scalars)


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
    odds_lookup: dict[str, float] | None = None,
    confederation_scalars: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build the per-match feature matrix and the target column.

    Parameters
    ----------
    odds_lookup : optional ``team -> normalised implied win probability``.
        Used to populate ``odds_implied_prob_home/away`` and ``odds_ratio``.
        For historical matches the 2026 odds are reused as a static
        team-strength proxy — a mild anachronism that the model learns as a
        per-team prior.
    confederation_scalars : optional output of
        :func:`fit_confederation_difficulty` to replace the hardcoded
        :data:`CONFEDERATION_DIFFICULTY`. ``None`` keeps the hardcoded table.
    """
    cfg = load_config()
    df = results_df.copy().sort_values("date").reset_index(drop=True)
    df["home_team"] = df["home_team"].map(normalise_team)
    df["away_team"] = df["away_team"].map(normalise_team)

    if squad_values is None:
        value_lookup = dict(SQUAD_VALUES)
    else:
        value_lookup = dict(zip(squad_values["team"], squad_values["value_eur_m"]))

    elo_lookup = _build_elo_lookup(elo_df)

    # Odds lookup defaults to the FALLBACK_ODDS dict normalised to sum to 1.
    if odds_lookup is None:
        raw = {t: odds_to_implied_prob(o) for t, o in FALLBACK_ODDS.items()}
        total = float(sum(raw.values())) or 1.0
        odds_lookup = {t: p / total for t, p in raw.items()}
    default_odds_prob = float(np.mean(list(odds_lookup.values()))) if odds_lookup else 1.0 / 48.0

    # Pre-build per-team path features for each historical WC tournament.
    # Keys: (year, team) -> dict[path_feature_key, value]
    _wc_path_cache: dict[tuple[int, str], dict[str, float]] = {}

    def _elo_snapshot_for_year(year: int) -> dict[str, float]:
        # ELO at the start of each WC fortnight (Jun 1 / Nov 1).
        anchor = (
            pd.Timestamp(year=year, month=6, day=1) if year == 2018
            else pd.Timestamp(year=year, month=11, day=1)
        )
        anchor_dt64 = np.datetime64(anchor.to_datetime64())
        return {
            team: _elo_at(elo_lookup, team, anchor_dt64)
            for team in {t for ts in HISTORICAL_WC_GROUPS[year].values() for t in ts}
        }

    def _wc_path_for(year: int, team: str) -> dict[str, float]:
        key = (year, team)
        if key in _wc_path_cache:
            return _wc_path_cache[key]
        snap = _elo_snapshot_for_year(year)
        feats = compute_path_features(
            team,
            HISTORICAL_WC_GROUPS[year],
            snap,
            HISTORICAL_BRACKET_HALF,
            HISTORICAL_ADJACENT_GROUP,
        )
        _wc_path_cache[key] = feats
        return feats

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

        # ---- Qualifying xGD (confederation-discounted, optionally with fitted scalars)
        xg_h = _qualifying_xgd(h, scalars=confederation_scalars)
        xg_a = _qualifying_xgd(a, scalars=confederation_scalars)

        # ---- Odds-implied win probability (static 2026 snapshot used for
        # historical matches too — acts as a team-strength prior)
        odds_p_h = float(odds_lookup.get(h, default_odds_prob))
        odds_p_a = float(odds_lookup.get(a, default_odds_prob))
        odds_ratio = float(
            np.log((odds_p_h + 1e-9) / (odds_p_a + 1e-9))
        )

        # ---- Path-difficulty features. Only filled for WC 2018/2022
        # tournament matches (the only historical tournaments for which we
        # have the group draw); non-WC rows are zero-filled so XGB can route
        # on ``is_wc`` to ignore them.
        if is_wc:
            wc_year = _historical_wc_year(match_date)
        else:
            wc_year = None
        if wc_year is not None and h in {t for ts in HISTORICAL_WC_GROUPS[wc_year].values() for t in ts}:
            ph = _wc_path_for(wc_year, h)
        else:
            ph = {k: 0.0 for k in PATH_FEATURE_KEYS}
        if wc_year is not None and a in {t for ts in HISTORICAL_WC_GROUPS[wc_year].values() for t in ts}:
            pa = _wc_path_for(wc_year, a)
        else:
            pa = {k: 0.0 for k in PATH_FEATURE_KEYS}

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
        out["odds_implied_prob_home"].append(odds_p_h)
        out["odds_implied_prob_away"].append(odds_p_a)
        out["odds_ratio"].append(odds_ratio)
        out["group_avg_elo_opp_home"].append(ph["group_avg_elo_opponents"])
        out["group_avg_elo_opp_away"].append(pa["group_avg_elo_opponents"])
        out["group_max_elo_opp_home"].append(ph["group_max_elo_opponent"])
        out["group_max_elo_opp_away"].append(pa["group_max_elo_opponent"])
        out["group_elo_rank_home"].append(ph["group_elo_rank"])
        out["group_elo_rank_away"].append(pa["group_elo_rank"])
        out["bracket_half_home"].append(ph["bracket_half"])
        out["bracket_half_away"].append(pa["bracket_half"])
        out["expected_r16_opp_elo_home"].append(ph["expected_r16_opponent_elo"])
        out["expected_r16_opp_elo_away"].append(pa["expected_r16_opponent_elo"])
        out["path_to_final_avg_elo_home"].append(ph["path_to_final_avg_elo"])
        out["path_to_final_avg_elo_away"].append(pa["path_to_final_avg_elo"])

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
    "PATH_FEATURE_KEYS",
    "HISTORICAL_WC_GROUPS",
    "WC2026_BRACKET_HALF",
    "WC2026_ADJACENT_GROUP",
    "compute_path_features",
    "build_match_features",
    "split_features_target",
    "fit_confederation_difficulty",
]
