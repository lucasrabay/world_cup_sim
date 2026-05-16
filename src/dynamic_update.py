"""Dynamic, in-tournament re-prediction engine.

During the World Cup the model should respond to real results: a heavy
Argentina loss in their opening match should visibly lower Argentina's
championship probability before the next group game kicks off.

This module exposes two surfaces:

* :func:`apply_match_result` — Bayesian-style shrinkage of a single team's
  attack / defence parameters after observing one fixture.
* :func:`update_and_resimulate` — load all recorded live results, apply the
  shrinkage cascade, and run the Monte Carlo on the *remaining* fixtures with
  the played ones pinned via ``simulator.run(fixed_results=...)``.

The shrinkage approach is deliberately simple. Full Bayesian inference would
be more principled but is heavier to fit, harder to explain, and (with the
relatively small per-match evidence we get) would land in a similar
neighbourhood anyway. Each new match nudges the relevant parameters 30% of
the way toward the value the observation alone would suggest.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_loader import WC2026_GROUPS
from .models import DixonColesModel, EnsemblePredictor
from .monte_carlo import WorldCupSimulator
from .utils import DATA_RAW, get_logger

logger = get_logger(__name__)

LIVE_RESULTS_PATH = DATA_RAW.parent / "live_results.parquet"


# ---------------------------------------------------------------------------
def apply_match_result(
    team_strengths: dict,
    result: dict,
    shrinkage_alpha: float = 0.3,
) -> dict:
    """Update ``team_strengths`` after observing one match.

    ``team_strengths`` must be the dict form ``{team: {'alpha': float,
    'beta': float}}``. The function returns a NEW dict — the input is left
    unchanged so callers can roll back if needed.

    Parameters
    ----------
    team_strengths : current attack (α) / defence (β) snapshot for every team.
    result : dict with keys ``home``, ``away``, ``goals_home``, ``goals_away``,
        ``date`` (optional), and an optional ``stage`` ∈ {"group", "knockout"}.
        Knockout matches are weighted 1.5× to reflect that those samples are
        played at higher intensity and therefore more informative.
    shrinkage_alpha : 0.0 = ignore the result, 1.0 = pure observation. 0.3 is
        the default — it takes ~5 matches to fully overwrite a prior.

    Update rule
    -----------
    Observed offensive signal: log(goals_scored + 0.5) − β_opp  (the opposing
    team's defence is baked into the observed goal rate, so we subtract it
    to isolate the home team's attacking contribution). The defensive signal
    is the symmetric construction. Each parameter then moves toward the
    observed signal by ``shrinkage_alpha`` (scaled by the stage weight).
    """
    if shrinkage_alpha < 0.0 or shrinkage_alpha > 1.0:
        raise ValueError("shrinkage_alpha must lie in [0, 1]")
    updated = copy.deepcopy(team_strengths)
    home = str(result["home"]); away = str(result["away"])
    gh = float(result["goals_home"]); ga = float(result["goals_away"])
    stage_weight = 1.5 if str(result.get("stage", "group")).startswith("knock") else 1.0

    if home not in updated:
        updated[home] = {"alpha": 0.0, "beta": 0.0}
    if away not in updated:
        updated[away] = {"alpha": 0.0, "beta": 0.0}

    h_prev = updated[home]
    a_prev = updated[away]

    # Observed attacking / defensive signals on the natural log scale.
    obs_home_attack = float(np.log(gh + 0.5)) - float(a_prev["beta"])
    obs_home_defence = float(a_prev["alpha"]) - float(np.log(ga + 0.5))
    obs_away_attack = float(np.log(ga + 0.5)) - float(h_prev["beta"])
    obs_away_defence = float(h_prev["alpha"]) - float(np.log(gh + 0.5))

    eff_alpha = min(1.0, shrinkage_alpha * stage_weight)

    updated[home] = {
        "alpha": (1.0 - eff_alpha) * h_prev["alpha"] + eff_alpha * obs_home_attack,
        "beta":  (1.0 - eff_alpha) * h_prev["beta"]  + eff_alpha * obs_home_defence,
    }
    updated[away] = {
        "alpha": (1.0 - eff_alpha) * a_prev["alpha"] + eff_alpha * obs_away_attack,
        "beta":  (1.0 - eff_alpha) * a_prev["beta"]  + eff_alpha * obs_away_defence,
    }
    return updated


# ---------------------------------------------------------------------------
def strengths_from_dc(dc: DixonColesModel) -> dict[str, dict[str, float]]:
    """Convert the fitted Dixon-Coles model into the team-strengths dict form
    used by :func:`apply_match_result`."""
    teams = set(dc.attack_params) | set(dc.defence_params)
    return {
        t: {
            "alpha": float(dc.attack_params.get(t, 0.0)),
            "beta": float(dc.defence_params.get(t, 0.0)),
        }
        for t in teams
    }


def apply_strengths_to_dc(dc: DixonColesModel, strengths: dict) -> DixonColesModel:
    """Return a shallow-copied DixonColesModel with α/β replaced by ``strengths``.

    The other parameters (γ, ρ, teams) are preserved. The original instance is
    not mutated, which keeps the predictor cached at module level safe to
    re-use after the user resets the dashboard.
    """
    new_dc = DixonColesModel(
        attack_params=dict(dc.attack_params),
        defence_params=dict(dc.defence_params),
        home_advantage=float(dc.home_advantage),
        rho=float(dc.rho),
        teams_=list(dc.teams_),
        fitted_=bool(dc.fitted_),
    )
    for team, ab in strengths.items():
        new_dc.attack_params[team] = float(ab["alpha"])
        new_dc.defence_params[team] = float(ab["beta"])
    return new_dc


# ---------------------------------------------------------------------------
def _load_live_results(path: Path | None = None) -> pd.DataFrame:
    """Read the live-results frame, returning an empty frame if missing."""
    p = Path(path) if path is not None else LIVE_RESULTS_PATH
    if not p.exists():
        return pd.DataFrame(columns=["home", "away", "goals_home", "goals_away", "date", "stage"])
    return pd.read_parquet(p)


def append_live_result(result: dict, path: Path | None = None) -> pd.DataFrame:
    """Append one row to the live-results parquet and return the updated frame."""
    p = Path(path) if path is not None else LIVE_RESULTS_PATH
    df = _load_live_results(p)
    new_row = {
        "home": str(result["home"]),
        "away": str(result["away"]),
        "goals_home": int(result["goals_home"]),
        "goals_away": int(result["goals_away"]),
        "date": pd.Timestamp(result.get("date", pd.Timestamp.utcnow())),
        "stage": str(result.get("stage", "group")),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def update_and_resimulate(
    base_predictor: EnsemblePredictor,
    current_results: pd.DataFrame | None = None,
    n_sims: int = 50_000,
    groups: dict[str, list[str]] | None = None,
    shrinkage_alpha: float = 0.3,
    seed: int = 42,
) -> dict:
    """Full re-prediction pipeline given the currently-observed live results.

    1. Load (or accept) the live-results frame.
    2. Apply ``apply_match_result`` for every row sequentially.
    3. Apply the updated parameters to a copy of the predictor.
    4. Run Monte Carlo over the WC 2026 bracket with the played fixtures pinned.

    Returns a dict with keys:

    * ``baseline_df`` — predictions BEFORE applying any updates (handy for the
      dashboard's "Δ since start" display).
    * ``updated_df`` — predictions AFTER updates.
    * ``strengths`` — final team-strengths dict.
    * ``n_results_applied`` — count of live results folded in.
    """
    groups = groups or WC2026_GROUPS
    results_df = current_results if current_results is not None else _load_live_results()

    # ---- Step 1: baseline run (no updates) ----------------------------
    baseline_sim = WorldCupSimulator(base_predictor, groups, n_sims=n_sims, seed=seed)
    baseline_df = baseline_sim.run()

    if results_df is None or results_df.empty:
        return {
            "baseline_df": baseline_df,
            "updated_df": baseline_df.copy(),
            "strengths": strengths_from_dc(base_predictor.dc),
            "n_results_applied": 0,
        }

    # ---- Step 2: cascade shrinkage updates ----------------------------
    strengths = strengths_from_dc(base_predictor.dc)
    for _, row in results_df.iterrows():
        strengths = apply_match_result(
            strengths,
            {
                "home": row["home"], "away": row["away"],
                "goals_home": row["goals_home"], "goals_away": row["goals_away"],
                "stage": row.get("stage", "group") if "stage" in row else "group",
            },
            shrinkage_alpha=shrinkage_alpha,
        )

    # ---- Step 3: rebuild the predictor's DC head with updated params --
    updated_dc = apply_strengths_to_dc(base_predictor.dc, strengths)
    # Copy the rest of the ensemble verbatim — only DC moves with shrinkage.
    updated_predictor = copy.copy(base_predictor)
    updated_predictor.dc = updated_dc

    # ---- Step 4: build fixed_results list and re-simulate ------------
    fixed = [
        {
            "home": r["home"], "away": r["away"],
            "goals_home": int(r["goals_home"]), "goals_away": int(r["goals_away"]),
        }
        for _, r in results_df.iterrows()
    ]
    new_sim = WorldCupSimulator(updated_predictor, groups, n_sims=n_sims, seed=seed)
    updated_df = new_sim.run(fixed_results=fixed)

    return {
        "baseline_df": baseline_df,
        "updated_df": updated_df,
        "strengths": strengths,
        "n_results_applied": int(len(results_df)),
    }


def most_changed_teams(
    baseline_df: pd.DataFrame,
    updated_df: pd.DataFrame,
    top_n: int = 10,
    key: str = "p_champion",
) -> pd.DataFrame:
    """Return the ``top_n`` teams sorted by absolute Δ on ``key``."""
    merged = baseline_df[["team", key]].merge(
        updated_df[["team", key]], on="team", how="outer", suffixes=("_before", "_after")
    )
    merged[f"{key}_delta"] = merged[f"{key}_after"] - merged[f"{key}_before"]
    merged["abs_delta"] = merged[f"{key}_delta"].abs()
    return merged.sort_values("abs_delta", ascending=False).head(top_n).reset_index(drop=True)


__all__ = [
    "apply_match_result",
    "append_live_result",
    "update_and_resimulate",
    "most_changed_teams",
    "strengths_from_dc",
    "apply_strengths_to_dc",
    "LIVE_RESULTS_PATH",
]
