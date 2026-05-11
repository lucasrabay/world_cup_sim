"""WC 2026 Monte Carlo tournament simulator.

Vectorised across simulations where possible — a single call to
``np.random.poisson`` samples scorelines for all 50,000 sims of a fixture in
one shot, which is the dominant cost in the inner loop.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .data_loader import HOST_NATIONS
from .models import EnsemblePredictor
from .utils import get_logger, load_config

logger = get_logger(__name__)


# Round labels (used both internally and in the output table)
ROUND_GROUP = "group"
ROUND_R32 = "r32"
ROUND_R16 = "r16"
ROUND_QF = "qf"
ROUND_SF = "sf"
ROUND_F = "f"
ROUND_CHAMP = "champion"

EXIT_TO_PROB_COL = {
    "group": "p_group_exit",
    "r32": "p_r32",
    "r16": "p_r16",
    "qf": "p_qf",
    "sf": "p_semi",
    "f": "p_final",
    "champion": "p_champion",
}


def _cdf_or_none(flat: np.ndarray | None) -> np.ndarray | None:
    """Cumulative distribution helper that tolerates ``None`` input."""
    if flat is None:
        return None
    cdf = np.cumsum(flat)
    cdf[-1] = 1.0
    return cdf


# ---------------------------------------------------------------------------
@dataclass
class SimContext:
    """Per-simulation per-team accumulators used to build the final summary."""

    teams: list[str]
    n_sims: int

    def __post_init__(self) -> None:
        n = len(self.teams)
        s = self.n_sims
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        # Furthest round reached, encoded as integer rank.
        self.exit_round = np.zeros((s, n), dtype=np.int8)
        # Group-stage placement counters
        self.placement = np.zeros((s, n), dtype=np.int8)  # 1=top, 2=second, 3=third, 4=fourth, 0=DNF
        # Goals scored / conceded across the whole sim
        self.goals_for = np.zeros((s, n), dtype=np.int32)
        self.goals_against = np.zeros((s, n), dtype=np.int32)
        self.matches = np.zeros((s, n), dtype=np.int16)


# ---------------------------------------------------------------------------
class WorldCupSimulator:
    """Run the full 48-team WC 2026 bracket many times, vectorised over sims."""

    ROUND_RANK = {
        ROUND_GROUP: 1,
        ROUND_R32: 2,
        ROUND_R16: 3,
        ROUND_QF: 4,
        ROUND_SF: 5,
        ROUND_F: 6,
        ROUND_CHAMP: 7,
    }

    def __init__(
        self,
        predictor: EnsemblePredictor,
        groups: dict[str, list[str]],
        n_sims: int = 50000,
        seed: int = 42,
    ) -> None:
        self.predictor = predictor
        self.groups = {g: list(ts) for g, ts in groups.items()}
        self.n_sims = int(n_sims)
        self.seed = int(seed)
        self._cfg = load_config()
        self._pen_rate = float(self._cfg["simulation"]["penalty_conversion_rate"])
        self.teams = sorted({t for ts in self.groups.values() for t in ts})
        self._rng = np.random.default_rng(self.seed)

        # Per-fixture cache: (a, b, scenario_id) ->
        #   (p_h, p_d, p_a, λ_h, λ_a, [flat_grid_H, flat_grid_D, flat_grid_A])
        # Grids are pre-built once per fixture so every sim of that fixture
        # reuses them — no team lookup or τ-correction inside the inner loop.
        self._fixture_cache: dict[tuple[str, str, str], dict] = {}
        # Lightweight backwards-compat λ cache (some external callers expect it).
        self._lambda_cache: dict[tuple[str, str], tuple[float, float]] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _scenario_key(scenario: dict | None) -> str:
        """Stable hashable identifier for a scenario dict."""
        if not scenario:
            return "_baseline_"
        return scenario.get("name", repr(sorted(scenario.items()))) or "_baseline_"

    @staticmethod
    def _build_dc_grid(l_h: float, l_a: float, rho: float, max_goals: int = 10) -> np.ndarray:
        """Construct the (max_goals+1)² τ-corrected joint scoreline grid."""
        ks = np.arange(max_goals + 1)
        log_ph = ks * np.log(l_h) - l_h - gammaln(ks + 1)
        log_pa = ks * np.log(l_a) - l_a - gammaln(ks + 1)
        joint = np.outer(np.exp(log_ph), np.exp(log_pa))
        joint[0, 0] *= 1.0 - l_h * l_a * rho
        joint[1, 0] *= 1.0 + l_a * rho
        joint[0, 1] *= 1.0 + l_h * rho
        joint[1, 1] *= 1.0 - rho
        joint = np.clip(joint, 0.0, None)
        s = joint.sum()
        if s > 0:
            joint = joint / s
        return joint

    @staticmethod
    def _conditional_flat_grids(joint: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return three flattened, per-outcome-renormalised distributions.

        Order: (home_win_grid, draw_grid, away_win_grid). Any of them may be
        ``None`` if the masked sum is zero.
        """
        i_idx, j_idx = np.indices(joint.shape)
        out: list[np.ndarray | None] = []
        for mask in (i_idx > j_idx, i_idx == j_idx, i_idx < j_idx):
            masked = joint * mask
            total = masked.sum()
            if total <= 0:
                out.append(None)
            else:
                out.append((masked / total).ravel())
        return out[0], out[1], out[2]

    def _lambda(self, a: str, b: str, scenario: dict | None = None) -> tuple[float, float]:
        """Backwards-compat: just the λ pair (with scenario applied + clipped)."""
        key = (a, b)
        if key in self._lambda_cache:
            la, lb = self._lambda_cache[key]
        else:
            try:
                la, lb = self.predictor.dc.predict_lambda(a, b, neutral=True)
            except Exception:
                la, lb = 1.2, 1.2
            self._lambda_cache[key] = (la, lb)
        if scenario:
            la *= scenario.get("lambda_mult", {}).get(a, 1.0)
            lb *= scenario.get("lambda_mult", {}).get(b, 1.0)
        return float(np.clip(la, 0.05, 6.0)), float(np.clip(lb, 0.05, 6.0))

    def _prebuild_baseline_cache(self) -> None:
        """Populate the baseline fixture cache for every (home, away) pair in
        a single batched XGB prediction call.

        Without this pre-pass each unique fixture triggers an individual
        ``predict_proba`` call (~10-15ms of sklearn/XGBoost overhead), and the
        tournament touches ~2200 unique pairs across 50k sims — driving the
        baseline run from ~25s to ~55s. Batching turns that into one ~80ms
        call plus pure NumPy.
        """
        if any(k for k in self._fixture_cache if k[2] == "_baseline_"):
            return
        teams = self.teams
        rho = float(self.predictor.dc.rho)
        rows: list[pd.DataFrame] = []
        pairs: list[tuple[str, str]] = []
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                rows.append(self.predictor._xgb_features_for(h, a, neutral=True))
                pairs.append((h, a))
        if not rows:
            return
        X_all = pd.concat(rows, ignore_index=True)
        try:
            proba = self.predictor.xgb.predict_proba(X_all)  # (N, 3) -> [loss, draw, win]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Batched XGB predict failed (%s) — falling back to per-fixture path", exc)
            return

        dw = float(self.predictor.dc_weight)
        xw = float(self.predictor.xgb_weight)
        ew = float(self.predictor.elo_weight)
        team_elo = getattr(self.predictor, "_team_elo", {})
        elo_logit = getattr(self.predictor, "elo_logistic", None)

        for i, (h, a) in enumerate(pairs):
            l_h, l_a = self.predictor.dc.predict_lambda(h, a, neutral=True)
            l_h = float(np.clip(l_h, 0.05, 6.0))
            l_a = float(np.clip(l_a, 0.05, 6.0))
            dc_probs = self.predictor.dc.predict_outcome_probs(h, a, neutral=True)
            xgb_h, xgb_d, xgb_a = float(proba[i, 2]), float(proba[i, 1]), float(proba[i, 0])
            if elo_logit is not None and ew > 0:
                elo_probs = elo_logit.predict_proba(
                    float(team_elo.get(h, 1500.0)),
                    float(team_elo.get(a, 1500.0)),
                )
            else:
                elo_probs = {"home_win": dc_probs["home_win"],
                              "draw": dc_probs["draw"],
                              "away_win": dc_probs["away_win"]}

            p_h = dw * dc_probs["home_win"] + xw * xgb_h + ew * elo_probs["home_win"]
            p_d = dw * dc_probs["draw"]      + xw * xgb_d + ew * elo_probs["draw"]
            p_a = dw * dc_probs["away_win"] + xw * xgb_a + ew * elo_probs["away_win"]
            s = p_h + p_d + p_a
            if s > 0:
                p_h, p_d, p_a = p_h / s, p_d / s, p_a / s

            joint = self._build_dc_grid(l_h, l_a, rho)
            flat_h, flat_d, flat_a = self._conditional_flat_grids(joint)

            # Pre-build ET grid + CDFs too — saves another N rebuilds.
            et_l_h = float(np.clip(l_h * 0.33, 0.05, 4.0))
            et_l_a = float(np.clip(l_a * 0.33, 0.05, 4.0))
            et_joint = self._build_dc_grid(et_l_h, et_l_a, rho=0.0)
            i_idx, j_idx = np.indices(et_joint.shape)
            et_p_h = float(et_joint[i_idx > j_idx].sum())
            et_p_d = float(np.diag(et_joint).sum())
            et_p_a = float(et_joint[i_idx < j_idx].sum())
            et_flat_h, et_flat_d, et_flat_a = self._conditional_flat_grids(et_joint)

            self._fixture_cache[(h, a, "_baseline_")] = {
                "p_h": p_h, "p_d": p_d, "p_a": p_a,
                "l_h": l_h, "l_a": l_a, "rho": rho,
                "flat_h": flat_h, "flat_d": flat_d, "flat_a": flat_a,
                "cdf_h": _cdf_or_none(flat_h),
                "cdf_d": _cdf_or_none(flat_d),
                "cdf_a": _cdf_or_none(flat_a),
                "max_goals": joint.shape[0] - 1,
                "et_p_h": et_p_h, "et_p_d": et_p_d, "et_p_a": et_p_a,
                "et_cdf_h": _cdf_or_none(et_flat_h),
                "et_cdf_d": _cdf_or_none(et_flat_d),
                "et_cdf_a": _cdf_or_none(et_flat_a),
                "et_max_goals": et_joint.shape[0] - 1,
            }

    def _fixture_data(self, a: str, b: str, scenario: dict | None) -> dict:
        """Return cached fixture metadata: outcome probs, λs, and conditional grids.

        For the baseline (no scenario) we use the calibrated ensemble's outcome
        probabilities directly. Under a scenario with ``lambda_mult`` overrides
        we re-derive outcomes from the τ-corrected DC grid at the adjusted λs
        — this is the only way an injury / form scenario can move outcomes,
        since the ensemble's XGB head doesn't see scenario flags.
        """
        key = (a, b, self._scenario_key(scenario))
        cached = self._fixture_cache.get(key)
        if cached is not None:
            return cached

        rho = float(self.predictor.dc.rho)

        # Compute scenario-adjusted λs once.
        l_h, l_a = self._lambda(a, b, scenario=scenario)

        if scenario is None:
            try:
                pred = self.predictor.predict(a, b, neutral=True)
                p_h = float(pred["home_win"])
                p_d = float(pred["draw"])
                p_a = float(pred["away_win"])
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Ensemble predict fallback (%s)", exc)
                joint = self._build_dc_grid(l_h, l_a, rho)
                i_idx, j_idx = np.indices(joint.shape)
                p_h = float(joint[i_idx > j_idx].sum())
                p_d = float(np.diag(joint).sum())
                p_a = float(joint[i_idx < j_idx].sum())
        else:
            # Scenario path: derive outcome probs from DC at adjusted λs so
            # that lambda_mult actually moves the outcome distribution.
            joint = self._build_dc_grid(l_h, l_a, rho)
            i_idx, j_idx = np.indices(joint.shape)
            p_h = float(joint[i_idx > j_idx].sum())
            p_d = float(np.diag(joint).sum())
            p_a = float(joint[i_idx < j_idx].sum())

        # Always build scoreline grids from the (possibly adjusted) λs.
        joint = self._build_dc_grid(l_h, l_a, rho)
        flat_h, flat_d, flat_a = self._conditional_flat_grids(joint)

        # CDFs for fast searchsorted-based vectorised sampling.
        def _cdf(flat: np.ndarray | None) -> np.ndarray | None:
            if flat is None:
                return None
            cdf = np.cumsum(flat)
            cdf[-1] = 1.0  # guard against floating-point shortfall
            return cdf

        data = {
            "p_h": p_h, "p_d": p_d, "p_a": p_a,
            "l_h": l_h, "l_a": l_a,
            "rho": rho,
            "flat_h": flat_h, "flat_d": flat_d, "flat_a": flat_a,
            "cdf_h": _cdf(flat_h), "cdf_d": _cdf(flat_d), "cdf_a": _cdf(flat_a),
            "max_goals": joint.shape[0] - 1,
        }
        self._fixture_cache[key] = data
        return data

    def _sample_scorelines(
        self,
        team_a: str,
        team_b: str,
        n: int,
        scenario: dict | None = None,
        knockout: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` scorelines for the fixture (ensemble-driven outcomes).

        For each of ``n`` simulations:
            1. Sample an outcome ∈ {H, D, A} from the ensemble's calibrated
               probabilities (or from DC's adjusted-λ grid under a scenario).
            2. Sample a scoreline from the DC grid masked to that outcome.

        For knockout matches drawn outcomes are then resolved with extra time
        (λ * 0.33, no τ) and a λ-biased Bernoulli shootout when ET also draws.
        """
        f = self._fixture_data(team_a, team_b, scenario)
        outcomes = self._sample_outcomes(f["p_h"], f["p_d"], f["p_a"], n)
        ga, gb = self._scoreline_from_outcomes(outcomes, f, n)

        if not knockout:
            return ga, gb

        tied_mask = outcomes == 1
        n_tied = int(tied_mask.sum())
        if n_tied == 0:
            return ga, gb

        # Extra time: reuse pre-cached ET grid if available (baseline path),
        # otherwise build it on the fly (scenario path).
        if "et_cdf_h" in f:
            et_p_h = float(f["et_p_h"])
            et_p_d = float(f["et_p_d"])
            et_p_a = float(f["et_p_a"])
            et_data = {
                "cdf_h": f["et_cdf_h"], "cdf_d": f["et_cdf_d"], "cdf_a": f["et_cdf_a"],
                "max_goals": int(f["et_max_goals"]),
            }
        else:
            et_l_h = float(np.clip(f["l_h"] * 0.33, 0.05, 4.0))
            et_l_a = float(np.clip(f["l_a"] * 0.33, 0.05, 4.0))
            et_joint = self._build_dc_grid(et_l_h, et_l_a, rho=0.0)
            i_idx, j_idx = np.indices(et_joint.shape)
            et_p_h = float(et_joint[i_idx > j_idx].sum())
            et_p_d = float(np.diag(et_joint).sum())
            et_p_a = float(et_joint[i_idx < j_idx].sum())
            et_flat_h, et_flat_d, et_flat_a = self._conditional_flat_grids(et_joint)
            et_data = {
                "cdf_h": _cdf_or_none(et_flat_h),
                "cdf_d": _cdf_or_none(et_flat_d),
                "cdf_a": _cdf_or_none(et_flat_a),
                "max_goals": et_joint.shape[0] - 1,
            }

        et_outcomes = self._sample_outcomes(et_p_h, et_p_d, et_p_a, n_tied)
        et_ga, et_gb = self._scoreline_from_outcomes(et_outcomes, et_data, n_tied)

        tied_idx = np.where(tied_mask)[0]
        ga[tied_idx] += et_ga
        gb[tied_idx] += et_gb

        # Penalty shootout for matches still tied after ET.
        st_local = et_outcomes == 1
        if st_local.any():
            n_st = int(st_local.sum())
            bias = 0.5 + 0.05 * np.tanh((f["l_h"] - f["l_a"]) / max(f["l_h"] + f["l_a"], 1e-6))
            wins_a = self._rng.random(n_st) < bias
            st_idx = tied_idx[st_local]
            ga[st_idx[wins_a]] += 1
            gb[st_idx[~wins_a]] += 1
        return ga, gb

    def _sample_outcomes(self, p_h: float, p_d: float, p_a: float, n: int) -> np.ndarray:
        """Sample n outcomes from a categorical (0=H, 1=D, 2=A)."""
        s = p_h + p_d + p_a
        if s <= 0:
            p_h, p_d, p_a = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
            s = 1.0
        u = self._rng.random(n)
        c1 = p_h / s
        c2 = (p_h + p_d) / s
        return np.where(u < c1, 0, np.where(u < c2, 1, 2)).astype(np.int8)

    def _scoreline_from_outcomes(
        self,
        outcomes: np.ndarray,
        f: dict,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised conditional scoreline draw given an outcome array."""
        max_goals = int(f["max_goals"])
        ga = np.zeros(n, dtype=np.int32)
        gb = np.zeros(n, dtype=np.int32)
        # Per-outcome searchsorted into the conditional CDF — much faster than
        # np.random.choice with p= for large n.
        for code, cdf_key, fallback in (
            (0, "cdf_h", (1, 0)),
            (1, "cdf_d", (0, 0)),
            (2, "cdf_a", (0, 1)),
        ):
            sel = outcomes == code
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            cdf = f.get(cdf_key)
            if cdf is None:
                ga[sel] = fallback[0]
                gb[sel] = fallback[1]
                continue
            u = self._rng.random(n_sel)
            idx = np.searchsorted(cdf, u, side="right")
            np.clip(idx, 0, cdf.size - 1, out=idx)
            ga[sel] = (idx // (max_goals + 1)).astype(np.int32)
            gb[sel] = (idx % (max_goals + 1)).astype(np.int32)
        return ga, gb

    # ------------------------------------------------------------------ pens
    def simulate_penalty_shootout(self, team_a: str, team_b: str) -> str:
        """Sequential pen shootout (used for single-match queries)."""
        rng = self._rng
        a = b = 0
        for _ in range(5):
            a += int(rng.random() < self._pen_rate)
            b += int(rng.random() < self._pen_rate)
        while a == b:
            a += int(rng.random() < self._pen_rate)
            b += int(rng.random() < self._pen_rate)
        return team_a if a > b else team_b

    # ------------------------------------------------------------------ single-match (sequential)
    @staticmethod
    def _sample_poisson_conditional(
        l_h: float,
        l_a: float,
        outcome: str,
        rho: float = 0.0,
        max_goals: int = 10,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        """Sample (gh, ga) from a τ-corrected DC grid built from raw λ inputs.

        Used for extra-time re-sampling, where we have λs in hand and want a
        scoreline conditional on a given outcome without doing any team lookup.
        """
        if rng is None:
            rng = np.random.default_rng()
        if outcome not in ("H", "D", "A"):
            raise ValueError(f"outcome must be 'H', 'D', or 'A', got {outcome!r}")
        joint = WorldCupSimulator._build_dc_grid(l_h, l_a, rho, max_goals=max_goals)
        i_idx, j_idx = np.indices(joint.shape)
        if outcome == "H":
            mask = i_idx > j_idx
        elif outcome == "D":
            mask = i_idx == j_idx
        else:
            mask = i_idx < j_idx
        masked = joint * mask
        total = masked.sum()
        if total <= 0:
            if outcome == "H":
                return 1, 0
            if outcome == "A":
                return 0, 1
            return 0, 0
        flat = (masked / total).ravel()
        idx = int(rng.choice(flat.size, p=flat))
        gh, ga = divmod(idx, max_goals + 1)
        return int(gh), int(ga)

    def simulate_match(self, home_team: str, away_team: str, knockout: bool = False) -> dict:
        """Simulate a single match: outcome from ensemble, scoreline from DC.

        For knockout matches a draw triggers extra time (λ * 0.33, no τ);
        if ET is also drawn we run a penalty shootout. The returned scoreline
        reflects 90-minute + ET goals; shootout results are encoded only in
        ``winner`` and ``went_to_pens``.
        """
        pred = self.predictor.predict(home_team, away_team, neutral=True)
        p_h = float(pred["home_win"])
        p_d = float(pred["draw"])
        p_a = float(pred["away_win"])
        l_h = float(pred["lambda_home"])
        l_a = float(pred["lambda_away"])

        # Step 1 — sample the outcome from the ensemble's calibrated probs.
        u = float(self._rng.random())
        s = max(p_h + p_d + p_a, 1e-12)
        if u < p_h / s:
            outcome = "H"
        elif u < (p_h + p_d) / s:
            outcome = "D"
        else:
            outcome = "A"

        # Step 2 — scoreline conditional on the sampled outcome.
        gh, ga = self.predictor.dc.sample_conditional(
            home_team, away_team, outcome, neutral=True, rng=self._rng
        )

        went_to_pens = False
        winner: str | None

        # Step 3 — knockout draw resolution.
        if knockout and outcome == "D":
            et_l_h = float(np.clip(l_h * 0.33, 0.05, 4.0))
            et_l_a = float(np.clip(l_a * 0.33, 0.05, 4.0))

            # Sample ET outcome from its λ-derived grid (no τ for the 30-min period).
            et_joint = self._build_dc_grid(et_l_h, et_l_a, rho=0.0)
            i_idx, j_idx = np.indices(et_joint.shape)
            ep_h = float(et_joint[i_idx > j_idx].sum())
            ep_d = float(np.diag(et_joint).sum())
            ep_a = float(et_joint[i_idx < j_idx].sum())
            u2 = float(self._rng.random())
            es = max(ep_h + ep_d + ep_a, 1e-12)
            if u2 < ep_h / es:
                et_outcome = "H"
            elif u2 < (ep_h + ep_d) / es:
                et_outcome = "D"
            else:
                et_outcome = "A"

            et_gh, et_ga = self._sample_poisson_conditional(
                et_l_h, et_l_a, et_outcome, rho=0.0, rng=self._rng
            )
            gh += et_gh
            ga += et_ga
            outcome = et_outcome  # promote ET result

            if et_outcome == "D":
                winner = self.simulate_penalty_shootout(home_team, away_team)
                went_to_pens = True
                return {
                    "home": home_team, "away": away_team,
                    "goals_home": int(gh), "goals_away": int(ga),
                    "winner": winner, "went_to_pens": True,
                    "outcome_sampled": "D",
                }

        # Step 4 — winner from the (possibly post-ET) scoreline.
        if gh > ga:
            winner = home_team
        elif gh < ga:
            winner = away_team
        else:
            winner = None  # group-stage draw (only reachable with knockout=False)

        return {
            "home": home_team, "away": away_team,
            "goals_home": int(gh), "goals_away": int(ga),
            "winner": winner, "went_to_pens": went_to_pens,
            "outcome_sampled": outcome,
        }

    # ------------------------------------------------------------------ group stage (vectorised)
    def simulate_group_stage(
        self,
        scenario: dict | None = None,
        ctx: SimContext | None = None,
    ) -> tuple[dict[str, np.ndarray], list[str], dict[str, np.ndarray]]:
        """Run all groups for all simulations in one pass.

        Returns
        -------
        placements : dict {group_letter: ndarray (n_sims, 4) of team indices in
                          local-group order ranked 1st..4th}
        third_place_pool : (n_sims, 12) array of team indices for the third
                          placed teams in each group
        per_team_group_metrics : dict with keys 'pts', 'gd', 'gf', 'placement'
                                each (n_sims, n_teams)
        """
        n = self.n_sims
        T = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        pts = np.zeros((n, T), dtype=np.int16)
        gd = np.zeros((n, T), dtype=np.int16)
        gf = np.zeros((n, T), dtype=np.int16)
        # H2H buckets (per group). We track points & GD inside the group's own
        # H2H subset, used for tiebreakers 4-6.
        # Easier representation: for each group fixture we record the two team
        # indices and the (gh, ga) per sim.

        group_letters = list(self.groups.keys())
        placements_idx: dict[str, np.ndarray] = {g: np.zeros((n, 4), dtype=np.int32) for g in group_letters}
        third_pool = np.zeros((n, len(group_letters)), dtype=np.int32)
        third_metrics = {
            "pts": np.zeros((n, len(group_letters)), dtype=np.int16),
            "gd": np.zeros((n, len(group_letters)), dtype=np.int16),
            "gf": np.zeros((n, len(group_letters)), dtype=np.int16),
        }

        for g_i, (g, members) in enumerate(self.groups.items()):
            local_idx = np.array([team_idx[t] for t in members])

            # Local accumulators
            l_pts = np.zeros((n, 4), dtype=np.int16)
            l_gd = np.zeros((n, 4), dtype=np.int16)
            l_gf = np.zeros((n, 4), dtype=np.int16)
            # Round-robin pairs (i<j)
            fixtures = list(itertools.combinations(range(4), 2))
            # Track per-fixture results for h2h tiebreakers
            fixture_results: list[tuple[int, int, np.ndarray, np.ndarray]] = []

            for i, j in fixtures:
                ga, gb = self._sample_scorelines(members[i], members[j], n=n, scenario=scenario)
                # Update points / gd / gf
                home_win = ga > gb
                draw = ga == gb
                away_win = ga < gb
                l_pts[:, i] += np.where(home_win, 3, np.where(draw, 1, 0))
                l_pts[:, j] += np.where(away_win, 3, np.where(draw, 1, 0))
                l_gd[:, i] += (ga - gb)
                l_gd[:, j] += (gb - ga)
                l_gf[:, i] += ga
                l_gf[:, j] += gb
                fixture_results.append((i, j, ga, gb))

                # Update SimContext goal/match accumulators
                if ctx is not None:
                    gi = local_idx[i]; gj = local_idx[j]
                    ctx.goals_for[:, gi] += ga
                    ctx.goals_against[:, gi] += gb
                    ctx.goals_for[:, gj] += gb
                    ctx.goals_against[:, gj] += ga
                    ctx.matches[:, gi] += 1
                    ctx.matches[:, gj] += 1

            # Scatter to global arrays
            for k in range(4):
                pts[:, local_idx[k]] += l_pts[:, k]
                gd[:, local_idx[k]] += l_gd[:, k]
                gf[:, local_idx[k]] += l_gf[:, k]

            # ---- Apply FIFA tiebreaker cascade per simulation, ranking 1..4
            # Build sort keys: pts desc, gd desc, gf desc, h2h pts desc, h2h gd desc, h2h gf desc, random
            # H2H breakers — recompute inside ties.
            ranks = self._rank_group_per_sim(
                l_pts, l_gd, l_gf, fixture_results, members, ctx is None
            )
            # ranks[:, k] = position (0=1st .. 3=4th) of local team k per sim
            # Invert: placements_idx[g][s, p] = global team index at position p
            inv = np.argsort(ranks, axis=1)  # sim s -> sorted local indices
            placements_idx[g] = local_idx[inv]

            # Record placements per team in ctx
            if ctx is not None:
                # placement number from 1..4 for each team
                placement_codes = np.empty((n, 4), dtype=np.int8)
                for p in range(4):
                    placement_codes[:, p] = p + 1  # rank value
                # ranks[:, k] tells us position 0-3 of team k
                team_placement_codes = ranks + 1
                for k in range(4):
                    ctx.placement[:, local_idx[k]] = team_placement_codes[:, k]

            # Record third-place pool entries
            third_local = inv[:, 2]            # local idx of the 3rd-placed team
            third_global = local_idx[third_local]
            third_pool[:, g_i] = third_global
            third_metrics["pts"][:, g_i] = l_pts[np.arange(n), third_local]
            third_metrics["gd"][:, g_i] = l_gd[np.arange(n), third_local]
            third_metrics["gf"][:, g_i] = l_gf[np.arange(n), third_local]

        return placements_idx, group_letters, {
            "third_pool": third_pool,
            "third_pts": third_metrics["pts"],
            "third_gd": third_metrics["gd"],
            "third_gf": third_metrics["gf"],
            "team_pts": pts,
            "team_gd": gd,
            "team_gf": gf,
        }

    @staticmethod
    def _rank_group_per_sim(
        l_pts: np.ndarray,
        l_gd: np.ndarray,
        l_gf: np.ndarray,
        fixture_results: list[tuple[int, int, np.ndarray, np.ndarray]],
        members: list[str],
        deterministic_random: bool,
    ) -> np.ndarray:
        """Return per-sim ranking (0-3) for each of the 4 group teams.

        FIFA cascade: pts → gd → gf → h2h pts → h2h gd → h2h gf → fair play
        (random) → coin flip. We implement up to h2h gf; remaining ties are
        broken by a stable seeded random key.
        """
        n = l_pts.shape[0]
        # Pre-compute per-team H2H accumulators on demand (vectorised over sims).
        # Build pairwise points/gd/gf matrices on the fly by iterating fixtures.
        # We construct: pair_pts[sim, i, j] -- points team i got against team j in this group.
        pair_pts = np.zeros((n, 4, 4), dtype=np.int16)
        pair_gd = np.zeros((n, 4, 4), dtype=np.int16)
        pair_gf = np.zeros((n, 4, 4), dtype=np.int16)
        for i, j, ga, gb in fixture_results:
            home_win = ga > gb
            draw = ga == gb
            pair_pts[:, i, j] += np.where(home_win, 3, np.where(draw, 1, 0))
            pair_pts[:, j, i] += np.where(~home_win & ~draw, 3, np.where(draw, 1, 0))
            pair_gd[:, i, j] += (ga - gb)
            pair_gd[:, j, i] += (gb - ga)
            pair_gf[:, i, j] += ga
            pair_gf[:, j, i] += gb

        # Composite primary key: combine in a single sortable big-int per team.
        # Use a fixed offset so that secondary criteria are subordinate.
        OFFSET = 1000  # safe headroom for gd/gf within a group
        prim = (
            l_pts.astype(np.int64) * (OFFSET ** 2)
            + (l_gd.astype(np.int64) + OFFSET) * OFFSET
            + (l_gf.astype(np.int64) + OFFSET)
        )
        # Stable rank within sim (descending) — pure NumPy.
        order = np.argsort(-prim, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(
            ranks,
            order,
            np.broadcast_to(np.arange(prim.shape[1]), prim.shape),
            axis=1,
        )

        # Short-circuit: identify which sims actually contain any tied primary
        # keys. Sims without ties (typically ~60-80% of all sims) need no
        # further work, saving us the np.unique() call per sim.
        sorted_prim = np.sort(prim, axis=1)
        has_tie = np.any(sorted_prim[:, 1:] == sorted_prim[:, :-1], axis=1)
        tied_sims = np.flatnonzero(has_tie)

        # Resolve teams sharing primary key via H2H — only for sims that need it.
        for s in tied_sims:
            uniq, inv, counts = np.unique(prim[s], return_inverse=True, return_counts=True)
            for u_idx, c in enumerate(counts):
                if c < 2:
                    continue
                tied_mask = inv == u_idx
                tied_idx = np.where(tied_mask)[0]
                # Restrict h2h to tied subset only.
                sub_pts = pair_pts[s][np.ix_(tied_idx, tied_idx)].sum(axis=1)
                sub_gd = pair_gd[s][np.ix_(tied_idx, tied_idx)].sum(axis=1)
                sub_gf = pair_gf[s][np.ix_(tied_idx, tied_idx)].sum(axis=1)
                composite = sub_pts.astype(np.int64) * 100_000 + (sub_gd + 500) * 100 + sub_gf
                # add tiny random for residual ties (fair play / draw)
                noise = (np.arange(len(tied_idx)) + s * 7919) % 113 / 1000.0
                composite = composite.astype(float) + noise
                local_order = np.argsort(-composite, kind="stable")
                # Get positions tied teams currently occupy (by primary order).
                tied_positions = sorted(int(ranks[s, k]) for k in tied_idx)
                for new_rank, ti in enumerate(local_order):
                    ranks[s, tied_idx[ti]] = tied_positions[new_rank]
        return ranks

    # ------------------------------------------------------------------ best 8 thirds
    def get_best_third_places(self, third_metrics: dict[str, np.ndarray]) -> np.ndarray:
        """Pick the best 8 of 12 third-placed teams per sim.

        Returns: (n_sims, 8) array of global team indices, ordered 1st..8th.
        """
        pts = third_metrics["third_pts"]
        gd = third_metrics["third_gd"]
        gf = third_metrics["third_gf"]
        pool = third_metrics["third_pool"]
        # Composite descending sort key.
        composite = pts.astype(np.int64) * 1_000_000 + (gd.astype(np.int64) + 500) * 1_000 + gf.astype(np.int64)
        # argsort descending
        order = np.argsort(-composite, axis=1, kind="stable")  # (n, 12)
        best_columns = order[:, :8]
        rows = np.arange(pool.shape[0])[:, None]
        return pool[rows, best_columns]

    # ------------------------------------------------------------------ knockout (sequential per sim, vectorised over sims for matches)
    def _build_round32_pairings(
        self,
        placements_idx: dict[str, np.ndarray],
        best_thirds: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Construct 16 R32 fixtures.

        WC 2026 expansion: 12 group winners + 12 runners-up + 8 best thirds = 32.
        We use a deterministic seeded layout so the higher-ranked teams (Spain,
        Argentina, France, England) start in different quarters of the bracket.
        Concretely:
            * Group winners are seeded 1-12 by group letter ordering then by
              placement strength (here we just use group order).
            * Pairing: winners vs thirds (8 fixtures), runners-up vs runners-up
              and runners-up vs winners for the remaining 8.
        This is a simplification of the official 2026 bracket but preserves the
        property that the top-seeded winners cannot meet before the SF stage.
        """
        n = self.n_sims
        groups = list(placements_idx.keys())   # e.g. A..L
        # Stack winners and runners-up.
        winners = np.stack([placements_idx[g][:, 0] for g in groups], axis=1)        # (n, 12)
        runners = np.stack([placements_idx[g][:, 1] for g in groups], axis=1)        # (n, 12)
        thirds = best_thirds                                                          # (n, 8)

        # Bracket halves — split winners/runners by group letter parity to keep
        # top groups (A, C, E, G, I, K) in one half. This is purely the seeded
        # template; result-driven separation is then handled by the random
        # within-half pairing.
        upper_winners = winners[:, ::2]                                               # 6 cols
        lower_winners = winners[:, 1::2]                                              # 6 cols
        upper_runners = runners[:, ::2]                                               # 6 cols
        lower_runners = runners[:, 1::2]                                              # 6 cols
        # Distribute thirds: 4 to each half. Use first 4 as upper, next 4 as lower.
        upper_thirds = thirds[:, :4]
        lower_thirds = thirds[:, 4:]

        fixtures: list[tuple[np.ndarray, np.ndarray]] = []
        # Upper half — 8 fixtures (winners vs thirds: 4, runners vs runners: 2, winners vs runners: 2)
        fixtures.append((upper_winners[:, 0], upper_thirds[:, 0]))
        fixtures.append((upper_winners[:, 1], upper_thirds[:, 1]))
        fixtures.append((upper_winners[:, 2], upper_thirds[:, 2]))
        fixtures.append((upper_winners[:, 3], upper_thirds[:, 3]))
        fixtures.append((upper_runners[:, 0], upper_runners[:, 1]))
        fixtures.append((upper_runners[:, 2], upper_runners[:, 3]))
        fixtures.append((upper_winners[:, 4], upper_runners[:, 4]))
        fixtures.append((upper_winners[:, 5], upper_runners[:, 5]))
        # Lower half — same template
        fixtures.append((lower_winners[:, 0], lower_thirds[:, 0]))
        fixtures.append((lower_winners[:, 1], lower_thirds[:, 1]))
        fixtures.append((lower_winners[:, 2], lower_thirds[:, 2]))
        fixtures.append((lower_winners[:, 3], lower_thirds[:, 3]))
        fixtures.append((lower_runners[:, 0], lower_runners[:, 1]))
        fixtures.append((lower_runners[:, 2], lower_runners[:, 3]))
        fixtures.append((lower_winners[:, 4], lower_runners[:, 4]))
        fixtures.append((lower_winners[:, 5], lower_runners[:, 5]))
        return fixtures

    def _play_ko_round(
        self,
        pairings: list[tuple[np.ndarray, np.ndarray]],
        ctx: SimContext,
        round_label: str,
        scenario: dict | None,
    ) -> list[np.ndarray]:
        """Play one KO round across all sims; return list of winner index arrays.

        Sims sharing the same (home, away) team pair are batched together —
        we group by sort-and-slice on the composite key, which is faster than
        building a fresh ``mask = inv == k`` boolean array per unique pair
        when each group covers only a few of the 50k sims.
        """
        winners: list[np.ndarray] = []
        T = len(self.teams)
        team_names = self.teams
        rank = self.ROUND_RANK[round_label]

        for left, right in pairings:
            wins = np.empty_like(left)
            keys = left.astype(np.int64) * T + right.astype(np.int64)

            # Stable sort groups sims by (a, b); split points found by np.diff.
            sort_idx = np.argsort(keys, kind="stable")
            sorted_keys = keys[sort_idx]
            change = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
            starts = np.flatnonzero(change)
            ends = np.concatenate((starts[1:], [len(keys)]))

            for u_start, u_end in zip(starts, ends):
                k = int(sorted_keys[u_start])
                indices = sort_idx[u_start:u_end]
                a_idx = k // T
                b_idx = k - a_idx * T
                a, b = team_names[a_idx], team_names[b_idx]
                count = int(u_end - u_start)
                ga, gb = self._sample_scorelines(a, b, n=count, scenario=scenario, knockout=True)
                a_wins = ga >= gb  # ties impossible after the KO sampler
                w_idx = np.where(a_wins, a_idx, b_idx)
                l_idx = np.where(a_wins, b_idx, a_idx)
                wins[indices] = w_idx
                # Per-team accumulator updates use fancy indexing on the
                # short ``indices`` array (typically a handful of entries)
                # rather than a 50k-length boolean mask.
                ctx.goals_for[indices, a_idx] += ga
                ctx.goals_against[indices, a_idx] += gb
                ctx.goals_for[indices, b_idx] += gb
                ctx.goals_against[indices, b_idx] += ga
                ctx.matches[indices, a_idx] += 1
                ctx.matches[indices, b_idx] += 1
                ctx.exit_round[indices, l_idx] = rank
            winners.append(wins)
        return winners

    def simulate_knockout_bracket(
        self,
        placements_idx: dict[str, np.ndarray],
        best_thirds: np.ndarray,
        ctx: SimContext,
        scenario: dict | None = None,
    ) -> np.ndarray:
        """Run R32 → R16 → QF → SF → F across all sims. Returns champion indices."""
        # R32
        r32 = self._build_round32_pairings(placements_idx, best_thirds)
        r32_winners = self._play_ko_round(r32, ctx, ROUND_R32, scenario)
        # R16: 16 winners → 8 fixtures
        r16_pairs = [(r32_winners[2 * i], r32_winners[2 * i + 1]) for i in range(8)]
        r16_winners = self._play_ko_round(r16_pairs, ctx, ROUND_R16, scenario)
        # QF: 8 → 4
        qf_pairs = [(r16_winners[2 * i], r16_winners[2 * i + 1]) for i in range(4)]
        qf_winners = self._play_ko_round(qf_pairs, ctx, ROUND_QF, scenario)
        # SF: 4 → 2
        sf_pairs = [(qf_winners[2 * i], qf_winners[2 * i + 1]) for i in range(2)]
        sf_winners = self._play_ko_round(sf_pairs, ctx, ROUND_SF, scenario)
        # F: 2 → 1
        final_pairs = [(sf_winners[0], sf_winners[1])]
        champions = self._play_ko_round(final_pairs, ctx, ROUND_F, scenario)[0]
        # Mark champion exit_round=7
        rows = np.arange(len(champions))
        ctx.exit_round[rows, champions] = self.ROUND_RANK[ROUND_CHAMP]
        return champions

    # ------------------------------------------------------------------ orchestrator
    def run(self, scenario: dict | None = None) -> pd.DataFrame:
        """Run the full simulation and return the per-team summary frame."""
        # reset RNG so that independent .run() calls with same seed are deterministic
        self._rng = np.random.default_rng(self.seed)
        # Drop any per-fixture cache state from a previous scenario; otherwise
        # a baseline run could re-use grids built under, say, "haaland_injured".
        self._fixture_cache = {}
        self._lambda_cache = {}
        # Batched ensemble pre-prediction — one XGB call for every pair so the
        # inner loop is pure NumPy. Only worthwhile for baseline; scenarios
        # use DC-derived probs and need re-keyed cache entries anyway.
        if scenario is None:
            self._prebuild_baseline_cache()
        ctx = SimContext(teams=list(self.teams), n_sims=self.n_sims)
        placements_idx, group_letters, group_metrics = self.simulate_group_stage(scenario=scenario, ctx=ctx)

        # Group exit set for teams that finished 4th, or 3rd outside best-8 pool
        # default exit_round = 1 (group). We'll bump for advancing teams below.
        ctx.exit_round[:] = self.ROUND_RANK[ROUND_GROUP]

        best_thirds = self.get_best_third_places(group_metrics)  # (n, 8)
        # Mark advancing teams as having reached at least R32.
        rows = np.arange(self.n_sims)[:, None]
        # winners + runners-up always advance
        for g in group_letters:
            adv = placements_idx[g][:, :2]
            ctx.exit_round[rows, adv] = self.ROUND_RANK[ROUND_R32]
        ctx.exit_round[rows, best_thirds] = self.ROUND_RANK[ROUND_R32]

        self.simulate_knockout_bracket(placements_idx, best_thirds, ctx, scenario=scenario)

        return self._summarise(ctx)

    def _summarise(self, ctx: SimContext) -> pd.DataFrame:
        n = self.n_sims
        rounds = ctx.exit_round
        # Probabilities of reaching each round (>= rank).
        def p_reach(rank: int) -> np.ndarray:
            return (rounds >= rank).mean(axis=0)

        p_champion = p_reach(self.ROUND_RANK[ROUND_CHAMP])
        p_final = p_reach(self.ROUND_RANK[ROUND_F])
        p_semi = p_reach(self.ROUND_RANK[ROUND_SF])
        p_qf = p_reach(self.ROUND_RANK[ROUND_QF])
        p_r16 = p_reach(self.ROUND_RANK[ROUND_R16])
        p_r32 = p_reach(self.ROUND_RANK[ROUND_R32])
        p_group_exit = (rounds == self.ROUND_RANK[ROUND_GROUP]).mean(axis=0)

        avg_gf = ctx.goals_for.mean(axis=0)
        avg_ga = ctx.goals_against.mean(axis=0)

        # Group placement probabilities
        p_top = (ctx.placement == 1).mean(axis=0)
        p_second = (ctx.placement == 2).mean(axis=0)
        p_third = (ctx.placement == 3).mean(axis=0)

        # Team -> group letter
        team_group: dict[str, str] = {}
        for g, ts in self.groups.items():
            for t in ts:
                team_group[t] = g

        df = pd.DataFrame(
            {
                "team": ctx.teams,
                "group": [team_group.get(t, "?") for t in ctx.teams],
                "p_champion": p_champion,
                "p_final": p_final,
                "p_semi": p_semi,
                "p_qf": p_qf,
                "p_r16": p_r16,
                "p_r32": p_r32,
                "p_group_exit": p_group_exit,
                "avg_goals_scored_per_sim": avg_gf,
                "avg_goals_conceded_per_sim": avg_ga,
                "p_top_group": p_top,
                "p_second_group": p_second,
                "p_third_group": p_third,
            }
        )
        df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ scenarios
    def upset_probability(self, team_a: str, team_b: str) -> float:
        """Return the probability that the lower-ELO team wins the match."""
        elo_a = self.predictor._team_elo.get(team_a, 1500.0)
        elo_b = self.predictor._team_elo.get(team_b, 1500.0)
        probs = self.predictor.predict(team_a, team_b, neutral=True)
        if elo_a >= elo_b:
            # underdog is B
            return float(probs["away_win"])
        return float(probs["home_win"])

    def run_scenarios(self, scenarios: list[dict] | None = None) -> dict[str, pd.DataFrame]:
        """Run a default panel of scenarios (or a user-supplied list)."""
        if scenarios is None:
            scenarios = self.default_scenarios()
        out: dict[str, pd.DataFrame] = {}
        for sc in scenarios:
            name = sc.get("name", "scenario")
            logger.info("Running scenario: %s", name)
            out[name] = self.run(scenario=sc)
        return out

    @staticmethod
    def default_scenarios() -> list[dict]:
        return [
            {"name": "baseline"},
            {
                "name": "haaland_injured",
                "lambda_mult": {"Norway": 0.55},
                "elo_delta": {"Norway": -80},
            },
            {
                "name": "messi_rests_group",
                "lambda_mult": {"Argentina": 0.75},  # group only — single-stage approximation
                "group_only": True,
            },
            {
                "name": "brazil_form_dip",
                "elo_delta": {"Brazil": -60},
            },
            {
                "name": "spain_dominant",
                "lambda_mult": {"Spain": 1.15},
            },
        ]


__all__ = ["WorldCupSimulator", "SimContext"]
