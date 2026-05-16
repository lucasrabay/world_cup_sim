"""Systematic comparison of every model component against the same held-out
WC 2018 + WC 2022 test set.

Implements Brier / log-loss / accuracy / RPS / calibration-error metrics, a
multi-class calibration plot, and a `ModelEvaluator` that takes a dict of
named predictors and produces a sorted comparison DataFrame.

Predictors only need a single method::

    predict(home_team: str, away_team: str, neutral: bool) -> dict
        # returns {'home_win': p, 'draw': p, 'away_win': p}

This is the same interface produced by every model in :mod:`src.models`, so
DC, XGB, ELO logistic, the OddsBaseline and the full ensemble all slot in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .utils import MODELS_SAVED, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multi-class Ranked Probability Score for ordered outcomes.

    Parameters
    ----------
    probs : array shape (n, 3) — [p_home_win, p_draw, p_away_win] per match.
    outcomes : array shape (n, 3) — one-hot encoded actual results in the
        same ordering as ``probs``.

    Returns
    -------
    float — mean RPS over the n matches. RPS = 0 for a perfect prediction,
    1 for a worst (confidently wrong, opposite-corner) prediction.

    Formula
    -------
    For each match::

        RPS = (1/2) * Σ_{k=1..2} (Σ_{j≤k} p_j - Σ_{j≤k} o_j)^2

    where the sum runs over k ∈ {1, 2} (i.e. the first two cumulative slots,
    since the third is always 1 by construction).
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probs.shape != outcomes.shape:
        raise ValueError(f"shape mismatch: {probs.shape} vs {outcomes.shape}")
    if probs.shape[1] != 3:
        raise ValueError("RPS expects 3-class outcomes [home, draw, away]")
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(outcomes, axis=1)
    diff_sq = (cum_p[:, :2] - cum_o[:, :2]) ** 2
    return float(0.5 * diff_sq.sum(axis=1).mean())


def _multiclass_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean one-vs-rest Brier across the 3 outcome classes."""
    total = 0.0
    for c in (0, 1, 2):
        total += brier_score_loss(outcomes[:, c], probs[:, c])
    return float(total / 3.0)


def _calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute deviation between empirical and predicted frequencies
    across 10 equal-width probability bins, averaged over the 3 classes."""
    errs: list[float] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for c in (0, 1, 2):
        p = probs[:, c]
        y = outcomes[:, c]
        per_bin: list[float] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() < 2:
                continue
            per_bin.append(abs(p[mask].mean() - y[mask].mean()))
        if per_bin:
            errs.append(float(np.mean(per_bin)))
    return float(np.mean(errs)) if errs else float("nan")


def calibration_curve_multi(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Return (bin_mid, frac_pos_h, frac_pos_d, frac_pos_a) suitable for a
    multi-class calibration plot."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    fracs = []
    for c in (0, 1, 2):
        p = probs[:, c]
        y = outcomes[:, c]
        per_bin = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p < hi)
            per_bin.append(float(y[mask].mean()) if mask.sum() >= 2 else float("nan"))
        fracs.append(np.array(per_bin))
    return mids, fracs[0], fracs[1], fracs[2]


# ---------------------------------------------------------------------------
# Dumb baselines (no class needed)
# ---------------------------------------------------------------------------
class _UniformBaseline:
    """Always predicts (1/3, 1/3, 1/3)."""

    def predict(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        return {"home_win": 1.0 / 3.0, "draw": 1.0 / 3.0, "away_win": 1.0 / 3.0}


class _HomeWinBaseline:
    """Always predicts the historical international-football base rate
    (~45/27/28). For neutral-ground tournament matches the home advantage is
    arguably smaller, but the model only needs a stable population prior."""

    def __init__(self, p_home: float = 0.45, p_draw: float = 0.27, p_away: float = 0.28) -> None:
        total = p_home + p_draw + p_away
        self.p_home = p_home / total
        self.p_draw = p_draw / total
        self.p_away = p_away / total

    def predict(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        return {"home_win": self.p_home, "draw": self.p_draw, "away_win": self.p_away}


# ---------------------------------------------------------------------------
# Adapters for raw / un-calibrated XGB and ELO logistic which need a feature
# row, not just team names.
# ---------------------------------------------------------------------------
class _XGBFeatureAdapter:
    """Wrap a fitted XGBClassifier (uncalibrated *or* calibrated) so that
    ``predict`` can be called with team names. Features come from one of two
    sources:

    * If ``feat_df`` is provided, we look up the matching row in the stored
      training/eval frame — this is the correct path for historical matches
      because their path-difficulty features were computed against the actual
      tournament's group draw, not the WC 2026 one.
    * Otherwise we fall back to a ``build_row`` callable (the EnsemblePredictor's
      ``_xgb_features_for``) which always builds against the WC 2026 context —
      this is the right path for forward predictions.
    """

    def __init__(
        self,
        classifier,
        build_row: Callable[[str, str, bool], pd.DataFrame],
        feature_columns: list[str] | None = None,
        feat_df: pd.DataFrame | None = None,
    ) -> None:
        self.classifier = classifier
        self.build_row = build_row
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.feat_df = feat_df

    def _lookup_row(self, home_team: str, away_team: str) -> pd.DataFrame | None:
        if self.feat_df is None:
            return None
        m = (self.feat_df["home_team"] == home_team) & (self.feat_df["away_team"] == away_team)
        sub = self.feat_df[m]
        if sub.empty:
            return None
        # If multiple meetings exist, prefer the latest (closest to today).
        sub = sub.sort_values("date").tail(1)
        if self.feature_columns is not None:
            return sub[self.feature_columns]
        return sub

    def predict(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        X = self._lookup_row(home_team, away_team)
        if X is None:
            X = self.build_row(home_team, away_team, neutral)
            if self.feature_columns is not None:
                X = X[self.feature_columns]
        proba = self.classifier.predict_proba(X)[0]
        return {"home_win": float(proba[2]), "draw": float(proba[1]), "away_win": float(proba[0])}


class _DCAdapter:
    """Wraps a fitted DixonColesModel to expose the standard predict() shape."""

    def __init__(self, dc) -> None:
        self.dc = dc

    def predict(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        probs = self.dc.predict_outcome_probs(home_team, away_team, neutral=neutral)
        return {"home_win": probs["home_win"], "draw": probs["draw"], "away_win": probs["away_win"]}


class _ELOLogisticAdapter:
    """Wraps an ELOLogisticModel so it can be queried by team names."""

    def __init__(self, model, team_elo: dict[str, float]) -> None:
        self.model = model
        self.team_elo = dict(team_elo)

    def predict(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        eh = float(self.team_elo.get(home_team, 1500.0))
        ea = float(self.team_elo.get(away_team, 1500.0))
        return self.model.predict_proba(eh, ea)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """Score every model on the same WC 2018 + WC 2022 held-out set."""

    def __init__(self, results_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
        self.results_df = results_df
        self.features_df = features_df

    def _select_test_matches(self, tournaments: list[str]) -> pd.DataFrame:
        """Return the WC 2018 + WC 2022 matches with their actual outcomes."""
        df = self.features_df.copy()
        mask = (
            df["is_wc"].astype(int) == 1
        ) & (
            df["date"] >= pd.Timestamp("2018-01-01")
        ) & (
            df["date"] <= pd.Timestamp("2023-01-01")
        )
        df = df[mask]
        # Optional name-based filter for safety
        if tournaments:
            keep = df["home_team"].notna()  # default truthy mask
            self.results_df  # noqa - kept for parity with API
        return df.reset_index(drop=True)

    @staticmethod
    def _one_hot(outcomes: Iterable[int]) -> np.ndarray:
        out = np.zeros((len(outcomes), 3), dtype=float)
        for i, o in enumerate(outcomes):
            # outcome encoding: 0=away_win, 1=draw, 2=home_win
            # we want columns ordered [home, draw, away] for RPS, so map:
            #   home(2) -> col 0,  draw(1) -> col 1,  away(0) -> col 2.
            col = {2: 0, 1: 1, 0: 2}[int(o)]
            out[i, col] = 1.0
        return out

    @staticmethod
    def _to_homedrawaway(probs_dict: dict[str, float]) -> tuple[float, float, float]:
        return (
            float(probs_dict.get("home_win", 0.0)),
            float(probs_dict.get("draw", 0.0)),
            float(probs_dict.get("away_win", 0.0)),
        )

    def _predict_matrix(self, model, matches: pd.DataFrame) -> np.ndarray:
        """Run a model over every match; return (n, 3) [home, draw, away] probs.

        For models that accept a ``feature_row`` argument (like the
        EnsemblePredictor), we pass the actual feat_df row so XGB sees the
        historical-tournament features rather than a regenerated WC 2026 row.
        """
        import inspect

        n = len(matches)
        probs = np.zeros((n, 3), dtype=float)
        try:
            accepts_row = "feature_row" in inspect.signature(model.predict).parameters
        except (TypeError, ValueError):
            accepts_row = False

        for i, row in enumerate(matches.itertuples(index=False)):
            try:
                if accepts_row:
                    feature_row = matches.iloc[[i]]
                    p = model.predict(
                        row.home_team, row.away_team,
                        neutral=True, feature_row=feature_row,
                    )
                else:
                    p = model.predict(row.home_team, row.away_team, neutral=True)
            except Exception as exc:  # pragma: no cover - safety only
                logger.warning(
                    "Model %s predict failed (%s) for %s vs %s",
                    model, exc, row.home_team, row.away_team,
                )
                p = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
            probs[i] = self._to_homedrawaway(p)
        return probs

    @staticmethod
    def _score_single(probs: np.ndarray, outcomes_oh: np.ndarray, outcomes_int: np.ndarray) -> dict[str, float]:
        if len(probs) == 0:
            return {k: float("nan") for k in ("brier_score", "log_loss", "accuracy", "rps", "calibration_error")}
        ll = log_loss(outcomes_int, probs, labels=[0, 1, 2])
        # outcomes_int encodes [away=0, draw=1, home=2] (the existing encoding).
        # We compare argmax(probs[:, [home,draw,away]]) to the encoded outcome
        # — translate the argmax back through the same {2:0, 1:1, 0:2} map.
        argmax_cols = probs.argmax(axis=1)
        pred_int = np.where(argmax_cols == 0, 2, np.where(argmax_cols == 1, 1, 0))
        acc = accuracy_score(outcomes_int, pred_int)
        brier = _multiclass_brier(probs, outcomes_oh)
        rps = ranked_probability_score(probs, outcomes_oh)
        cal = _calibration_error(probs, outcomes_oh)
        return {
            "brier_score": float(brier),
            "log_loss": float(ll),
            "accuracy": float(acc),
            "rps": float(rps),
            "calibration_error": float(cal),
        }

    def evaluate_all(
        self,
        models: Mapping[str, object],
        test_tournaments: list[str] | None = None,
    ) -> pd.DataFrame:
        """Score every model. Returns a DataFrame indexed by model name."""
        tournaments = test_tournaments or ["FIFA World Cup"]
        matches = self._select_test_matches(tournaments)
        if matches.empty:
            logger.warning("Held-out match set is empty; returning blank evaluation frame")
            return pd.DataFrame()
        outcomes_int = matches["outcome"].astype(int).to_numpy()
        outcomes_oh = self._one_hot(outcomes_int)

        rows: list[dict] = []
        per_model_probs: dict[str, np.ndarray] = {}
        for name, model in models.items():
            probs = self._predict_matrix(model, matches)
            per_model_probs[name] = probs
            scores = self._score_single(probs, outcomes_oh, outcomes_int)
            scores["model"] = name
            rows.append(scores)

        df = pd.DataFrame(rows).set_index("model").sort_values("brier_score")
        self._last_probs = per_model_probs
        self._last_matches = matches
        return df

    def evaluate_split(
        self,
        models: Mapping[str, object],
    ) -> pd.DataFrame:
        """Per-tournament (2018 vs 2022) Brier split for every model."""
        matches = self._select_test_matches(["FIFA World Cup"])
        outcomes_int = matches["outcome"].astype(int).to_numpy()
        outcomes_oh = self._one_hot(outcomes_int)
        is_2018 = (matches["date"] >= pd.Timestamp("2018-01-01")) & (matches["date"] <= pd.Timestamp("2018-12-31"))
        is_2022 = (matches["date"] >= pd.Timestamp("2022-01-01")) & (matches["date"] <= pd.Timestamp("2023-01-01"))

        rows: list[dict] = []
        for name, model in models.items():
            probs = self._predict_matrix(model, matches)
            b2018 = _multiclass_brier(probs[is_2018.to_numpy()], outcomes_oh[is_2018.to_numpy()])
            b2022 = _multiclass_brier(probs[is_2022.to_numpy()], outcomes_oh[is_2022.to_numpy()])
            rows.append({
                "model": name,
                "brier_2018": float(b2018),
                "brier_2022": float(b2022),
                "delta": float(b2022 - b2018),
            })
        return pd.DataFrame(rows).set_index("model")

    def plot_calibration_grid(self, out_path: Path | None = None) -> Path | None:
        """One subplot per scored model, three calibration lines per subplot.

        Requires ``evaluate_all`` to have been called first (caches probs).
        """
        if not hasattr(self, "_last_probs") or not self._last_probs:
            return None
        matches = self._last_matches
        outcomes_oh = self._one_hot(matches["outcome"].astype(int).to_numpy())

        n = len(self._last_probs)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
        flat = axes.ravel()
        for ax, (name, probs) in zip(flat, self._last_probs.items()):
            mids, fh, fd, fa = calibration_curve_multi(probs, outcomes_oh)
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
            ax.plot(mids, fh, "o-", label="home win", color="#1b9e77")
            ax.plot(mids, fd, "s-", label="draw", color="#7570b3")
            ax.plot(mids, fa, "^-", label="away win", color="#d95f02")
            ax.set_title(name, fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
        for ax in flat[len(self._last_probs):]:
            ax.axis("off")
        flat[0].legend(loc="upper left", fontsize=8)
        fig.suptitle("Multi-class calibration — held-out WC matches", y=1.0)
        fig.tight_layout()
        if out_path is None:
            out_path = MODELS_SAVED / "calibration_all_models.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path


__all__ = [
    "ranked_probability_score",
    "calibration_curve_multi",
    "ModelEvaluator",
    "_UniformBaseline",
    "_HomeWinBaseline",
    "_XGBFeatureAdapter",
    "_DCAdapter",
    "_ELOLogisticAdapter",
]
