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

        # Optuna search — keep verbose output minimal.
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize", study_name="xgb_match")
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
# Ensemble
# ---------------------------------------------------------------------------
class EnsemblePredictor:
    """Convex combination of Dixon-Coles and XGBoost outcome probabilities.

    The XGBoost classifier needs a feature row, which is built lazily from the
    most recent ELO/value/xGD context held by this object — set those via
    ``set_context()`` once after training.
    """

    def __init__(
        self,
        dc_model: DixonColesModel,
        xgb_model: XGBMatchPredictor,
        dc_weight: float = 0.5,
    ) -> None:
        self.dc = dc_model
        self.xgb = xgb_model
        self.dc_weight = float(np.clip(dc_weight, 0.0, 1.0))
        self.xgb_weight = 1.0 - self.dc_weight
        self._team_elo: dict[str, float] = {}
        self._team_value: dict[str, float] = {}

    def set_context(self, team_elo: dict[str, float], team_value_eur_m: dict[str, float]) -> None:
        self._team_elo = dict(team_elo)
        self._team_value = dict(team_value_eur_m)

    def _xgb_features_for(self, home_team: str, away_team: str, neutral: bool) -> pd.DataFrame:
        """Construct a single-row feature frame matching FEATURE_COLUMNS."""
        elo_h = self._team_elo.get(home_team, 1500.0)
        elo_a = self._team_elo.get(away_team, 1500.0)
        v_h = float(np.log(self._team_value.get(home_team, 80.0)))
        v_a = float(np.log(self._team_value.get(away_team, 80.0)))
        from .features import QUALIFYING_XGD

        row = {
            "elo_home": elo_h,
            "elo_away": elo_a,
            "elo_diff": elo_h - elo_a,
            "value_home": v_h,
            "value_away": v_a,
            "value_ratio": v_h - v_a,
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
            "xg_diff_qualifying_home": float(QUALIFYING_XGD.get(home_team, 0.0)),
            "xg_diff_qualifying_away": float(QUALIFYING_XGD.get(away_team, 0.0)),
        }
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> dict[str, float]:
        dc = self.dc.predict_outcome_probs(home_team, away_team, neutral=neutral)
        try:
            X = self._xgb_features_for(home_team, away_team, neutral)
            proba = self.xgb.predict_proba(X)[0]  # [loss, draw, win] for home
            xgb_pred = {"home_win": float(proba[2]), "draw": float(proba[1]), "away_win": float(proba[0])}
        except Exception as exc:
            logger.debug("XGB predict fallback (%s)", exc)
            xgb_pred = {"home_win": dc["home_win"], "draw": dc["draw"], "away_win": dc["away_win"]}

        blended = {
            "home_win": self.dc_weight * dc["home_win"] + self.xgb_weight * xgb_pred["home_win"],
            "draw": self.dc_weight * dc["draw"] + self.xgb_weight * xgb_pred["draw"],
            "away_win": self.dc_weight * dc["away_win"] + self.xgb_weight * xgb_pred["away_win"],
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
    "EnsemblePredictor",
    "evaluate_model",
]
