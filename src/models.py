"""Match-outcome models and ensembling.

Three model classes are provided:

    * DixonColesModel     – classical Poisson scoreline model with low-score
      correction (Dixon & Coles, 1997).
    * XGBMatchPredictor   – XGBoost multi-class classifier with isotonic
      probability calibration.
    * EnsemblePredictor   – simple convex blend of the two, exposing the
      interface the Monte Carlo engine needs (probabilities + λ + sample).

The module also includes ``evaluate_model`` which scores a calibrated
predictor on held-out WC 2018 + WC 2022 matches.
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLUMNS, split_features_target
from .utils import MODELS_SAVED, get_logger, load_config

logger = get_logger(__name__)

# Silence sklearn / xgboost informational warnings that crowd the log.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Dixon-Coles
# ---------------------------------------------------------------------------
def _dc_tau(x: np.ndarray, y: np.ndarray, l1: np.ndarray, l2: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction τ(x, y, λ1, λ2, ρ), vectorised."""
    tau = np.ones_like(l1, dtype=float)
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - l1[m00] * l2[m00] * rho
    tau[m10] = 1.0 + l2[m10] * rho
    tau[m01] = 1.0 + l1[m01] * rho
    tau[m11] = 1.0 - rho
    # τ may dip below zero for extreme λ; floor at a tiny positive.
    return np.clip(tau, 1e-9, None)


def _poisson_log_pmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lam_safe = np.clip(lam, 1e-9, None)
    return k * np.log(lam_safe) - lam_safe - gammaln(k + 1.0)


@dataclass
class DixonColesModel:
    """Classical Dixon-Coles bivariate Poisson model."""

    attack_params: dict[str, float] = field(default_factory=dict)
    defence_params: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.25
    rho: float = -0.05
    teams_: list[str] = field(default_factory=list)
    fitted_: bool = False

    # ---------------- Fitting ----------------
    def fit(self, matches_df: pd.DataFrame, time_decay: bool = True) -> "DixonColesModel":
        """Fit attack/defence/home/rho by minimising the negative log-likelihood."""
        df = matches_df.copy()
        teams = sorted(set(df["home_team"]).union(df["away_team"]))
        self.teams_ = teams
        team_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        h_idx = df["home_team"].map(team_idx).to_numpy()
        a_idx = df["away_team"].map(team_idx).to_numpy()
        hs = df["home_score"].to_numpy(dtype=float)
        as_ = df["away_score"].to_numpy(dtype=float)
        if time_decay and "sample_weight" in df.columns:
            w = df["sample_weight"].to_numpy(dtype=float)
        else:
            w = np.ones(len(df))

        # neutral flag — when set we drop home advantage for that match.
        neutral = (
            df["is_neutral"].to_numpy(dtype=bool)
            if "is_neutral" in df.columns
            else np.zeros(len(df), dtype=bool)
        )

        # parameter vector layout: [α_1..α_T, β_1..β_T, γ, ρ]
        # Identifiability: pin α_0 = 0 and β_0 = 0 (canonical reference team).
        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            alpha = np.zeros(n_teams)
            beta = np.zeros(n_teams)
            alpha[1:] = params[: n_teams - 1]
            beta[1:] = params[n_teams - 1 : 2 * (n_teams - 1)]
            gamma = params[-2]
            rho = params[-1]
            return alpha, beta, float(gamma), float(rho)

        def neg_log_lik(params: np.ndarray) -> float:
            alpha, beta, gamma, rho = unpack(params)
            # λ_home = exp(α_home - β_away + γ * (1 - neutral))
            l1 = np.exp(alpha[h_idx] - beta[a_idx] + gamma * (~neutral).astype(float))
            l2 = np.exp(alpha[a_idx] - beta[h_idx])
            ll = _poisson_log_pmf(hs, l1) + _poisson_log_pmf(as_, l2)
            tau = _dc_tau(hs.astype(int), as_.astype(int), l1, l2, rho)
            ll = ll + np.log(tau)
            return -float(np.sum(w * ll))

        x0 = np.concatenate(
            [
                np.zeros(n_teams - 1),  # α (excluding pinned ref)
                np.zeros(n_teams - 1),  # β
                [0.25, -0.05],          # γ, ρ
            ]
        )
        bounds = (
            [(-3.0, 3.0)] * (n_teams - 1)
            + [(-3.0, 3.0)] * (n_teams - 1)
            + [(-0.5, 1.5), (-0.2, 0.2)]
        )

        logger.info("Fitting Dixon-Coles on %d matches / %d teams", len(df), n_teams)
        res = minimize(
            neg_log_lik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-7},
        )

        alpha, beta, gamma, rho = unpack(res.x)
        # Re-centre attack/defence parameters around zero (post-hoc; this is a
        # reparameterisation that doesn't change λ since both shift equally).
        alpha = alpha - alpha.mean()
        beta = beta - beta.mean()

        self.attack_params = {teams[i]: float(alpha[i]) for i in range(n_teams)}
        self.defence_params = {teams[i]: float(beta[i]) for i in range(n_teams)}
        self.home_advantage = float(gamma)
        self.rho = float(rho)
        self.fitted_ = True
        logger.info(
            "Dixon-Coles fit complete (neg-LL=%.2f, γ=%.3f, ρ=%.3f)",
            res.fun,
            gamma,
            rho,
        )
        return self

    # ---------------- Inference ----------------
    def _alpha(self, team: str) -> float:
        return float(self.attack_params.get(team, 0.0))

    def _beta(self, team: str) -> float:
        return float(self.defence_params.get(team, 0.0))

    def predict_lambda(self, home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
        if not self.fitted_:
            raise RuntimeError("Model not fitted")
        gamma = 0.0 if neutral else self.home_advantage
        l1 = float(np.exp(self._alpha(home_team) - self._beta(away_team) + gamma))
        l2 = float(np.exp(self._alpha(away_team) - self._beta(home_team)))
        # Sane clipping — extreme parameter pairs can blow up.
        l1 = float(np.clip(l1, 0.05, 6.0))
        l2 = float(np.clip(l2, 0.05, 6.0))
        return l1, l2

    def predict_outcome_probs(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        max_goals: int = 10,
    ) -> dict[str, float]:
        l1, l2 = self.predict_lambda(home_team, away_team, neutral=neutral)
        # Build the joint scoreline probability grid with τ correction.
        ks = np.arange(max_goals + 1)
        log_ph = ks * np.log(l1) - l1 - gammaln(ks + 1)
        log_pa = ks * np.log(l2) - l2 - gammaln(ks + 1)
        ph = np.exp(log_ph)
        pa = np.exp(log_pa)
        joint = np.outer(ph, pa)
        # τ correction on the four low-score cells.
        joint[0, 0] *= 1.0 - l1 * l2 * self.rho
        joint[1, 0] *= 1.0 + l2 * self.rho
        joint[0, 1] *= 1.0 + l1 * self.rho
        joint[1, 1] *= 1.0 - self.rho
        joint = np.clip(joint, 0.0, None)
        joint = joint / joint.sum()

        i, j = np.indices(joint.shape)
        p_home = float(joint[i > j].sum())
        p_draw = float(np.diag(joint).sum())
        p_away = float(joint[i < j].sum())
        return {
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away,
            "lambda_home": l1,
            "lambda_away": l2,
        }

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        rng = rng or np.random.default_rng()
        l1, l2 = self.predict_lambda(home_team, away_team, neutral=neutral)
        return int(rng.poisson(l1)), int(rng.poisson(l2))

    def sample_conditional(
        self,
        home_team: str,
        away_team: str,
        outcome: str,
        neutral: bool = True,
        max_goals: int = 10,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        """Sample a scoreline from the τ-corrected Dixon-Coles grid conditioned
        on a specified match outcome.

        Parameters
        ----------
        home_team, away_team : team names known to the model.
        outcome : one of 'H' (home win), 'D' (draw), 'A' (away win).
        neutral : if True, drop the home-advantage term γ from λ_home.
        max_goals : grid is built over (max_goals+1)² scoreline cells.
        rng : numpy Generator. ``np.random`` is used if not supplied.

        Returns
        -------
        (goals_home, goals_away) : non-negative integers consistent with
            ``outcome`` (e.g. for 'H' we are guaranteed goals_home > goals_away).

        Notes
        -----
        Cells outside the requested-outcome region are zeroed and the remainder
        is renormalised before sampling. If the masked grid sums to zero (an
        extreme λ pair could in theory produce that), we return a deterministic
        minimum-margin scoreline matching the outcome and emit a warning.
        """
        if outcome not in ("H", "D", "A"):
            raise ValueError(f"outcome must be 'H', 'D', or 'A', got {outcome!r}")
        if rng is None:
            rng = np.random.default_rng()

        l1, l2 = self.predict_lambda(home_team, away_team, neutral=neutral)

        # Build the τ-corrected joint scoreline distribution.
        ks = np.arange(max_goals + 1)
        log_ph = ks * np.log(l1) - l1 - gammaln(ks + 1)
        log_pa = ks * np.log(l2) - l2 - gammaln(ks + 1)
        joint = np.outer(np.exp(log_ph), np.exp(log_pa))
        joint[0, 0] *= 1.0 - l1 * l2 * self.rho
        joint[1, 0] *= 1.0 + l2 * self.rho
        joint[0, 1] *= 1.0 + l1 * self.rho
        joint[1, 1] *= 1.0 - self.rho
        joint = np.clip(joint, 0.0, None)

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
            logger.warning(
                "Conditional grid sum is zero for %s vs %s outcome=%s; using fallback",
                home_team, away_team, outcome,
            )
            if outcome == "H":
                return 1, 0
            if outcome == "A":
                return 0, 1
            return 0, 0

        flat = (masked / total).ravel()
        idx = int(rng.choice(flat.size, p=flat))
        gh, ga = divmod(idx, max_goals + 1)
        return int(gh), int(ga)

    # ---------------- Persistence ----------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "attack_params": self.attack_params,
                    "defence_params": self.defence_params,
                    "home_advantage": self.home_advantage,
                    "rho": self.rho,
                    "teams_": self.teams_,
                    "fitted_": self.fitted_,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "DixonColesModel":
        data = json.loads(Path(path).read_text())
        m = cls()
        m.attack_params = data["attack_params"]
        m.defence_params = data["defence_params"]
        m.home_advantage = float(data["home_advantage"])
        m.rho = float(data["rho"])
        m.teams_ = list(data["teams_"])
        m.fitted_ = bool(data["fitted_"])
        return m


# ---------------------------------------------------------------------------
# XGBoost classifier with isotonic calibration
# ---------------------------------------------------------------------------
class XGBMatchPredictor:
    """XGBoost multi-class classifier with optuna-tuned hyperparameters and
    isotonic-calibrated output probabilities."""

    def __init__(self, n_trials: int | None = None, cv_folds: int | None = None) -> None:
        cfg = load_config()
        self.n_trials = int(n_trials if n_trials is not None else cfg["model"]["n_trials_optuna"])
        self.cv_folds = int(cv_folds if cv_folds is not None else cfg["model"]["cv_folds"])
        self.calibration_method = cfg["model"]["calibration_method"]
        self.best_params: dict[str, Any] = {}
        self.calibrated_model: CalibratedClassifierCV | None = None
        self.feature_names_: list[str] = list(FEATURE_COLUMNS)
        self.raw_model_ = None

    # ------- Optuna objective -------
    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, w: pd.Series):
        import xgboost as xgb

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "verbosity": 0,
            "random_state": 42,
            "n_jobs": 1,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = w.iloc[train_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            proba = model.predict_proba(X_va)
            try:
                scores.append(log_loss(y_va, proba, labels=[0, 1, 2]))
            except ValueError:
                scores.append(np.inf)
        return float(np.mean(scores))

    # ------- Fit -------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> "XGBMatchPredictor":
        import xgboost as xgb

        if sample_weight is None:
            sample_weight = pd.Series(np.ones(len(X)), index=X.index)

        # Drop matches whose time-decay weight has effectively underflowed.
        # With a 180-day half-life the oldest matches in our dataset weigh
        # ~1e-19, which causes XGBoost's calibrator folds to fail the
        # sum_weight >= 1e-6 sanity check. A 1e-4 floor is plenty since those
        # samples contribute nothing to the gradient anyway.
        WEIGHT_FLOOR = 1e-4
        keep_mask = sample_weight.to_numpy(dtype=float) > WEIGHT_FLOOR
        if not keep_mask.all():
            dropped = int((~keep_mask).sum())
            X = X.iloc[keep_mask].reset_index(drop=True)
            y = y.iloc[keep_mask].reset_index(drop=True)
            sample_weight = sample_weight.iloc[keep_mask].reset_index(drop=True)
            logger.info("Dropped %d matches with weight < %.0e", dropped, WEIGHT_FLOOR)
        # Normalise weights so their sum equals N — preserves relative
        # weighting but keeps fold sums comfortably above XGBoost's epsilon.
        sw_arr = sample_weight.to_numpy(dtype=float)
        if sw_arr.sum() > 0:
            sample_weight = pd.Series(sw_arr * len(sw_arr) / sw_arr.sum(), index=sample_weight.index)

        # Optuna search — seeded for full determinism so downstream evaluation
        # and Monte Carlo simulations produce the same results across runs.
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction="minimize",
                study_name="xgb_match",
                sampler=sampler,
            )
            study.optimize(
                lambda trial: self._objective(trial, X, y, sample_weight),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )
            self.best_params = dict(study.best_params)
            logger.info("Optuna best params: %s (logloss=%.4f)", self.best_params, study.best_value)
        except Exception as exc:  # pragma: no cover - tuning best-effort
            logger.warning("Optuna tuning failed (%s); falling back to defaults", exc)
            self.best_params = {
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 400,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 3,
            }

        base = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
            random_state=42,
            n_jobs=1,
            **self.best_params,
        )
        base.fit(X, y, sample_weight=sample_weight)
        self.raw_model_ = base
        # Calibration on the same training data via cross-validation.
        # Use prefit-on-folds approach with `cv` to avoid leakage from a single fit.
        calibrator = CalibratedClassifierCV(
            estimator=xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                verbosity=0,
                random_state=42,
                n_jobs=1,
                **self.best_params,
            ),
            method=self.calibration_method,
            cv=3,
        )
        calibrator.fit(X, y, sample_weight=sample_weight)
        self.calibrated_model = calibrator
        logger.info("XGBoost trained and calibrated (%s)", self.calibration_method)
        return self

    # ------- Inference -------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated_model is None:
            raise RuntimeError("Model not fitted")
        return self.calibrated_model.predict_proba(X[self.feature_names_])

    # ------- SHAP explanation -------
    def explain(self, X: pd.DataFrame, max_samples: int = 500) -> Path | None:
        if self.raw_model_ is None:
            logger.warning("explain() called before fit"); return None
        try:
            import shap
        except ImportError:  # pragma: no cover
            logger.warning("shap not installed; skipping SHAP plot")
            return None

        sample = X[self.feature_names_].head(max_samples)
        try:
            explainer = shap.TreeExplainer(self.raw_model_)
            shap_values = explainer.shap_values(sample)
            plt.figure(figsize=(10, 6))
            # multi-class shap_values -> list per class; plot on home-win class (idx 2)
            sv = shap_values[2] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(sv, sample, show=False)
            out = MODELS_SAVED / "shap_summary.png"
            plt.tight_layout()
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close("all")
            logger.info("Saved SHAP summary -> %s", out)
            return out
        except Exception as exc:  # pragma: no cover
            logger.warning("SHAP explanation failed: %s", exc)
            return None

    # ------- Persistence -------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "calibrated_model": self.calibrated_model,
                "raw_model": self.raw_model_,
                "best_params": self.best_params,
                "feature_names_": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "XGBMatchPredictor":
        blob = joblib.load(path)
        m = cls()
        m.calibrated_model = blob["calibrated_model"]
        m.raw_model_ = blob.get("raw_model")
        m.best_params = blob.get("best_params", {})
        m.feature_names_ = blob.get("feature_names_", list(FEATURE_COLUMNS))
        return m


# ---------------------------------------------------------------------------
# Lightweight ELO-only logistic regression
# ---------------------------------------------------------------------------
class ELOLogisticModel:
    """Multinomial logistic regression on a single feature: ``elo_diff``.

    Acts as a conservative regularising anchor inside the ensemble — it has no
    capacity to overfit team-specific quirks the way XGBoost or DC can.
    """

    def __init__(self) -> None:
        self.clf: LogisticRegression | None = None

    def fit(self, feat_df: pd.DataFrame) -> "ELOLogisticModel":
        if "elo_diff" not in feat_df.columns or "outcome" not in feat_df.columns:
            raise ValueError("feat_df must contain 'elo_diff' and 'outcome' columns")
        X = feat_df[["elo_diff"]].to_numpy(dtype=float)
        y = feat_df["outcome"].astype(int).to_numpy()
        w = (
            feat_df["sample_weight"].to_numpy(dtype=float)
            if "sample_weight" in feat_df.columns
            else np.ones(len(feat_df))
        )
        self.clf = LogisticRegression(solver="lbfgs", max_iter=500)
        self.clf.fit(X, y, sample_weight=w)
        logger.info("ELO logistic anchor fitted on %d matches", len(feat_df))
        return self

    def predict_proba(self, elo_home: float, elo_away: float) -> dict[str, float]:
        if self.clf is None:
            return {"home_win": 0.34, "draw": 0.33, "away_win": 0.33}
        x = np.array([[elo_home - elo_away]], dtype=float)
        proba = self.clf.predict_proba(x)[0]
        # sklearn gives ordered classes; classes_ tells us which.
        cls = list(self.clf.classes_)
        out = {0: 0.0, 1: 0.0, 2: 0.0}
        for c, p in zip(cls, proba):
            out[int(c)] = float(p)
        return {"home_win": out[2], "draw": out[1], "away_win": out[0]}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf}, path)

    @classmethod
    def load(cls, path: Path) -> "ELOLogisticModel":
        m = cls()
        m.clf = joblib.load(path)["clf"]
        return m


# ---------------------------------------------------------------------------
# Pre-tournament odds baseline
# ---------------------------------------------------------------------------
class OddsBaselineModel:
    """A trivial outright-odds-driven head.

    Reads the team's pre-tournament tournament-winner implied probability and
    converts it into a head-to-head match outcome via Bradley-Terry. A fixed
    historical World Cup draw rate of 23% is reserved; the remaining 77% is
    split between the two sides proportionally to their outright strength.

    This model is read-only — it requires no training, just a lookup table.
    """

    def __init__(self, odds_dict: dict[str, float], draw_prob: float = 0.23) -> None:
        # Store a defensive copy; values should be non-negative.
        self.odds_dict: dict[str, float] = {str(t): float(p) for t, p in odds_dict.items()}
        self.draw_prob: float = float(np.clip(draw_prob, 0.0, 0.99))
        self._uniform = 1.0 / max(len(self.odds_dict), 1)

    def _lookup(self, team: str) -> float:
        p = self.odds_dict.get(team)
        if p is None or p <= 0:
            return self._uniform
        return float(p)

    def predict(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        """Return ``{home_win, draw, away_win}`` summing to 1.0."""
        p_h_out = self._lookup(home_team)
        p_a_out = self._lookup(away_team)
        denom = p_h_out + p_a_out
        if denom <= 0:
            return {"home_win": (1.0 - self.draw_prob) / 2.0,
                    "draw": self.draw_prob,
                    "away_win": (1.0 - self.draw_prob) / 2.0}
        bt_home = p_h_out / denom
        non_draw = 1.0 - self.draw_prob
        return {
            "home_win": float(bt_home * non_draw),
            "draw": float(self.draw_prob),
            "away_win": float((1.0 - bt_home) * non_draw),
        }


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
class EnsemblePredictor:
    """Convex combination of up to four heads:

        * Dixon-Coles            — scoreline-aware Poisson model.
        * XGBoost (calibrated)   — feature-rich tabular classifier.
        * ELO logistic           — conservative single-feature anchor.
        * Odds baseline          — pre-tournament market consensus.

    The XGBoost classifier reads a single-row feature frame built lazily from
    the team context — call :meth:`set_context` once after training to set
    ELO, squad value, and odds-implied probabilities for the 48 teams.

    Default 4-way weights are ``(DC=0.25, XGB=0.45, ELO=0.10, Odds=0.20)`` —
    the trained model still dominates, but the market gets a meaningful pull
    so blatantly mis-priced model bets get reined in.

    Backwards compatibility
    -----------------------
    Older callers that pass only ``dc_weight=...`` (or omit ``odds_baseline``)
    are still supported: missing components simply drop out of the blend, with
    their weight absorbed into the remaining components and renormalised.
    """

    def __init__(
        self,
        dc_model: DixonColesModel,
        xgb_model: XGBMatchPredictor,
        dc_weight: float | None = None,
        elo_logistic: "ELOLogisticModel | None" = None,
        elo_weight: float | None = None,
        odds_baseline: "OddsBaselineModel | None" = None,
        odds_weight: float | None = None,
        weights: tuple[float, ...] | None = None,
    ) -> None:
        self.dc = dc_model
        self.xgb = xgb_model
        self.elo_logistic = elo_logistic
        self.odds_baseline = odds_baseline

        # ---- resolve four weights -----------------------------------------
        if weights is not None:
            if len(weights) == 4:
                dc_w, xgb_w, elo_w, odds_w = (float(w) for w in weights)
            elif len(weights) == 3:
                dc_w, xgb_w, elo_w = (float(w) for w in weights)
                odds_w = 0.0
            else:
                raise ValueError(
                    f"weights tuple must have length 3 or 4, got {len(weights)}"
                )
        elif (
            dc_weight is None and elo_weight is None and odds_weight is None
            and elo_logistic is not None and odds_baseline is not None
        ):
            # Canonical four-component default.
            dc_w, xgb_w, elo_w, odds_w = 0.25, 0.45, 0.10, 0.20
        elif (
            dc_weight is None and elo_weight is None and odds_weight is None
            and elo_logistic is not None and odds_baseline is None
        ):
            # Canonical three-component default.
            dc_w, xgb_w, elo_w, odds_w = 0.30, 0.55, 0.15, 0.0
        else:
            dc_w = float(dc_weight) if dc_weight is not None else 0.5
            elo_w = float(elo_weight) if (elo_weight is not None and elo_logistic is not None) else 0.0
            odds_w = float(odds_weight) if (odds_weight is not None and odds_baseline is not None) else 0.0
            xgb_w = max(0.0, 1.0 - dc_w - elo_w - odds_w)

        # Force-zero weights for missing heads.
        if elo_logistic is None:
            elo_w = 0.0
        if odds_baseline is None:
            odds_w = 0.0

        total = dc_w + xgb_w + elo_w + odds_w
        if total <= 0:
            dc_w, xgb_w, elo_w, odds_w = 0.5, 0.5, 0.0, 0.0
            total = 1.0
        self.dc_weight = dc_w / total
        self.xgb_weight = xgb_w / total
        self.elo_weight = elo_w / total
        self.odds_weight = odds_w / total

        self._team_elo: dict[str, float] = {}
        self._team_value: dict[str, float] = {}
        self._team_odds: dict[str, float] = {}
        self._path_features: dict[str, dict[str, float]] = {}

    def set_context(
        self,
        team_elo: dict[str, float],
        team_value_eur_m: dict[str, float],
        team_odds: dict[str, float] | None = None,
        path_features: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Set per-team static context for runtime feature construction."""
        self._team_elo = dict(team_elo)
        self._team_value = dict(team_value_eur_m)
        if team_odds is not None:
            self._team_odds = dict(team_odds)
        if path_features is not None:
            self._path_features = {t: dict(v) for t, v in path_features.items()}

    def _xgb_features_for(self, home_team: str, away_team: str, neutral: bool) -> pd.DataFrame:
        """Construct a single-row feature frame matching FEATURE_COLUMNS."""
        from .features import (
            DEFAULT_KO_RATE,
            FIFA_RANKING_2026,
            QUALIFYING_XGD,
            TEAM_CONFEDERATION,
            CONFEDERATION_DIFFICULTY,
            WC_KNOCKOUT_WIN_RATE,
            PATH_FEATURE_KEYS,
            _value_tier,
        )

        # Path-difficulty features are pre-computed per team once at
        # set_context() time; here we just look them up.
        ph = self._path_features.get(home_team, {k: 0.0 for k in PATH_FEATURE_KEYS})
        pa = self._path_features.get(away_team, {k: 0.0 for k in PATH_FEATURE_KEYS})

        elo_h = self._team_elo.get(home_team, 1500.0)
        elo_a = self._team_elo.get(away_team, 1500.0)
        val_h = self._team_value.get(home_team, 80.0)
        val_a = self._team_value.get(away_team, 80.0)
        v_h = float(np.log(val_h))
        v_a = float(np.log(val_a))

        rank_h = int(FIFA_RANKING_2026.get(home_team, 99))
        rank_a = int(FIFA_RANKING_2026.get(away_team, 99))

        ko_h = float(WC_KNOCKOUT_WIN_RATE.get(home_team, DEFAULT_KO_RATE))
        ko_a = float(WC_KNOCKOUT_WIN_RATE.get(away_team, DEFAULT_KO_RATE))

        # Confederation-discounted xGD
        cm_h = float(CONFEDERATION_DIFFICULTY.get(TEAM_CONFEDERATION.get(home_team, ""), 1.0))
        cm_a = float(CONFEDERATION_DIFFICULTY.get(TEAM_CONFEDERATION.get(away_team, ""), 1.0))
        xg_h = float(QUALIFYING_XGD.get(home_team, 0.0)) * cm_h
        xg_a = float(QUALIFYING_XGD.get(away_team, 0.0)) * cm_a

        # Odds-implied probabilities — fall back to a uniform 1/48 if missing
        default_odds = 1.0 / 48.0
        odds_p_h = float(self._team_odds.get(home_team, default_odds))
        odds_p_a = float(self._team_odds.get(away_team, default_odds))
        odds_ratio = float(np.log((odds_p_h + 1e-9) / (odds_p_a + 1e-9)))

        row = {
            "elo_home": elo_h,
            "elo_away": elo_a,
            "elo_diff": elo_h - elo_a,
            "value_home": v_h,
            "value_away": v_a,
            "value_ratio": v_h - v_a,
            "value_tier_home": _value_tier(val_h),
            "value_tier_away": _value_tier(val_a),
            "is_neutral": int(bool(neutral)),
            "is_wc": 1,
            "tournament_weight": 1.0,
            "days_since_match_home": 4.0,
            "days_since_match_away": 4.0,
            "home_form_5": 0.55,
            "away_form_5": 0.55,
            "h2h_wc_home_winrate": 0.5,
            "goals_scored_10_home": 1.4,
            "goals_conceded_10_home": 1.0,
            "goals_scored_10_away": 1.3,
            "goals_conceded_10_away": 1.0,
            "xg_diff_qualifying_home": xg_h,
            "xg_diff_qualifying_away": xg_a,
            "fifa_rank_home": rank_h,
            "fifa_rank_away": rank_a,
            "rank_diff": float(rank_a - rank_h),
            "log_rank_ratio": float(np.log((rank_a + 1) / (rank_h + 1))),
            "wc_knockout_rate_home": ko_h,
            "wc_knockout_rate_away": ko_a,
            "wc_knockout_rate_diff": ko_h - ko_a,
            "odds_implied_prob_home": odds_p_h,
            "odds_implied_prob_away": odds_p_a,
            "odds_ratio": odds_ratio,
            "group_avg_elo_opp_home": ph["group_avg_elo_opponents"],
            "group_avg_elo_opp_away": pa["group_avg_elo_opponents"],
            "group_max_elo_opp_home": ph["group_max_elo_opponent"],
            "group_max_elo_opp_away": pa["group_max_elo_opponent"],
            "group_elo_rank_home": ph["group_elo_rank"],
            "group_elo_rank_away": pa["group_elo_rank"],
            "bracket_half_home": ph["bracket_half"],
            "bracket_half_away": pa["bracket_half"],
            "expected_r16_opp_elo_home": ph["expected_r16_opponent_elo"],
            "expected_r16_opp_elo_away": pa["expected_r16_opponent_elo"],
            "path_to_final_avg_elo_home": ph["path_to_final_avg_elo"],
            "path_to_final_avg_elo_away": pa["path_to_final_avg_elo"],
        }
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        feature_row: "pd.DataFrame | None" = None,
    ) -> dict[str, float]:
        """Blend the four heads. When evaluating historical fixtures pass the
        matching ``feature_row`` from the training feat_df so XGB sees the
        actual tournament's path-difficulty features, not the WC 2026 ones."""
        dc = self.dc.predict_outcome_probs(home_team, away_team, neutral=neutral)
        try:
            X = feature_row if feature_row is not None else self._xgb_features_for(home_team, away_team, neutral)
            if hasattr(self.xgb, "feature_names_"):
                X = X[self.xgb.feature_names_]
            proba = self.xgb.predict_proba(X)[0]  # [loss, draw, win] for home
            xgb_pred = {"home_win": float(proba[2]), "draw": float(proba[1]), "away_win": float(proba[0])}
        except Exception as exc:
            logger.debug("XGB predict fallback (%s)", exc)
            xgb_pred = {"home_win": dc["home_win"], "draw": dc["draw"], "away_win": dc["away_win"]}

        # ELO-logistic anchor (or fall back to DC if no logistic provided)
        if self.elo_logistic is not None and self.elo_weight > 0:
            elo_h = self._team_elo.get(home_team, 1500.0)
            elo_a = self._team_elo.get(away_team, 1500.0)
            elo_pred = self.elo_logistic.predict_proba(elo_h, elo_a)
        else:
            elo_pred = {"home_win": dc["home_win"], "draw": dc["draw"], "away_win": dc["away_win"]}

        # Odds baseline (or fall back to DC if not provided)
        if self.odds_baseline is not None and self.odds_weight > 0:
            odds_pred = self.odds_baseline.predict(home_team, away_team, neutral=neutral)
        else:
            odds_pred = {"home_win": dc["home_win"], "draw": dc["draw"], "away_win": dc["away_win"]}

        blended = {
            "home_win": (
                self.dc_weight * dc["home_win"]
                + self.xgb_weight * xgb_pred["home_win"]
                + self.elo_weight * elo_pred["home_win"]
                + self.odds_weight * odds_pred["home_win"]
            ),
            "draw": (
                self.dc_weight * dc["draw"]
                + self.xgb_weight * xgb_pred["draw"]
                + self.elo_weight * elo_pred["draw"]
                + self.odds_weight * odds_pred["draw"]
            ),
            "away_win": (
                self.dc_weight * dc["away_win"]
                + self.xgb_weight * xgb_pred["away_win"]
                + self.elo_weight * elo_pred["away_win"]
                + self.odds_weight * odds_pred["away_win"]
            ),
            "lambda_home": dc["lambda_home"],
            "lambda_away": dc["lambda_away"],
        }
        s = blended["home_win"] + blended["draw"] + blended["away_win"]
        if s > 0:
            for k in ("home_win", "draw", "away_win"):
                blended[k] /= s
        return blended

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        return self.dc.simulate_match(home_team, away_team, neutral=neutral, rng=rng)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    predictor: EnsemblePredictor | XGBMatchPredictor,
    feat_df: pd.DataFrame,
    tournament_filter: str = "FIFA World Cup",
) -> dict[str, float]:
    """Evaluate on WC 2018 + WC 2022 matches.

    Train cutoff is Jan 1 2018 — matches after that, classified as ``is_wc``,
    form the held-out test set.
    """
    test = feat_df[
        (feat_df["date"] >= "2018-01-01")
        & (feat_df["date"] <= "2023-01-01")
        & (feat_df["xg_diff_qualifying_home"].notna())
        & (feat_df["xg_diff_qualifying_away"].notna())
    ].copy()
    # Restrict to actual World Cup matches via the is_wc flag (more robust than name match).
    test = test[test.get("is_wc", 0) == 1] if "is_wc" in test.columns else test
    if test.empty:
        logger.warning("No held-out WC matches found; skipping evaluation")
        return {"brier": float("nan"), "logloss": float("nan"), "accuracy": float("nan")}

    X_test, y_test, _ = split_features_target(test)
    if isinstance(predictor, EnsemblePredictor):
        # Reuse the calibrated XGB head as the evaluation surface.
        proba = predictor.xgb.predict_proba(X_test)
    else:
        proba = predictor.predict_proba(X_test)

    # Brier score for multi-class: mean over classes of one-vs-rest Brier.
    brier_total = 0.0
    for c in (0, 1, 2):
        brier_total += brier_score_loss((y_test == c).astype(int), proba[:, c])
    brier = brier_total / 3.0

    ll = log_loss(y_test, proba, labels=[0, 1, 2])
    acc = accuracy_score(y_test, np.argmax(proba, axis=1))

    # Calibration curve (home-win class)
    try:
        prob_home = proba[:, 2]
        true_home = (y_test == 2).astype(int)
        frac_pos, mean_pred = calibration_curve(true_home, prob_home, n_bins=8)
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
        plt.plot(mean_pred, frac_pos, "o-", label="Model")
        plt.xlabel("Predicted probability")
        plt.ylabel("Empirical frequency")
        plt.title("Calibration curve – home win")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(MODELS_SAVED / "calibration_curve.png", dpi=120)
        plt.close("all")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not plot calibration curve: %s", exc)

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, np.argmax(proba, axis=1), labels=[0, 1, 2])
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1, 2], ["Loss", "Draw", "Win"])
        plt.yticks([0, 1, 2], ["Loss", "Draw", "Win"])
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(MODELS_SAVED / "confusion_matrix.png", dpi=120)
        plt.close("all")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not plot confusion matrix: %s", exc)

    summary = {
        "brier": float(brier),
        "logloss": float(ll),
        "accuracy": float(acc),
        "n_test": int(len(y_test)),
    }
    logger.info(
        "Eval on %d WC matches: Brier=%.4f  LogLoss=%.4f  Acc=%.3f",
        summary["n_test"], summary["brier"], summary["logloss"], summary["accuracy"],
    )
    return summary


__all__ = [
    "DixonColesModel",
    "XGBMatchPredictor",
    "ELOLogisticModel",
    "OddsBaselineModel",
    "EnsemblePredictor",
    "evaluate_model",
]
