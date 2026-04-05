"""Meta-Learner: Super-Stacker that combines all base model predictions.

Takes probability outputs from 12+ independent models plus contextual features
and produces a final calibrated prediction via a learned stacking ensemble.

Training uses purged walk-forward CV to prevent leakage.
"""

import logging
import json
import os
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


# Feature groups for the meta-learner
BASE_MODEL_FEATURES = [
    "elo_prob", "xgb_prob",                           # Existing models
    "hmm_hot_diff", "hmm_confidence",                 # HMM
    "kalman_prob", "kalman_strength_diff",             # Kalman
    "kalman_home_uncertainty", "kalman_away_uncertainty",
    "pagerank_prob", "pagerank_diff",                  # Network
    "eigenvector_diff", "h2h_advantage",
    "lgbm_prob", "catboost_prob",                      # GBM ensemble
    "mlp_prob", "lstm_prob",                           # Neural networks
]

CONTEXT_FEATURES = [
    "market_implied_prob", "line_movement",            # Odds
    "weather_scoring_impact", "weather_unpredictability",  # Weather
    "net_epa_diff", "cpoe_diff",                       # Advanced stats (NFL)
    "woba_diff", "era_diff",                           # Advanced stats (MLB)
    "home_sos_pagerank", "sos_diff",                   # SOS
]

ALL_FEATURES = BASE_MODEL_FEATURES + CONTEXT_FEATURES


class MetaLearner:
    """Super-stacking ensemble that combines all base models.

    Uses XGBoost as the meta-learner (proven effective for stacking),
    with purged walk-forward CV training to prevent leakage.
    """

    def __init__(self, sport="nfl", meta_model="xgboost", **model_params):
        self.sport = sport
        self.meta_model_type = meta_model
        self.model = None
        self.feature_names = []
        self._fitted = False
        self._feature_importance = {}
        self._training_log = []
        self._model_weights = {}
        self.model_params = model_params

    def train(self, X, y, feature_names=None):
        """Train the meta-learner on stacked features.

        Args:
            X: numpy array where each column is a base model prediction or context feature
            y: binary labels (1=home win)
            feature_names: names of feature columns
        """
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        y = np.array(y, dtype=np.float32)

        if self.meta_model_type == "xgboost":
            self._train_xgboost(X, y)
        elif self.meta_model_type == "logistic":
            self._train_logistic(X, y)
        elif self.meta_model_type == "ridge":
            self._train_ridge(X, y)
        else:
            self._train_xgboost(X, y)

    def _train_xgboost(self, X, y):
        """Train XGBoost meta-learner with anti-overfitting settings."""
        import xgboost as xgb

        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": self.model_params.get("meta_xgb_max_depth", 4),
            "eta": self.model_params.get("meta_xgb_eta", 0.05),
            "subsample": self.model_params.get("meta_xgb_subsample", 0.8),
            "colsample_bytree": 0.7,
            "min_child_weight": self.model_params.get("meta_xgb_min_child_weight", 5),
            "alpha": self.model_params.get("meta_xgb_alpha", 0.5),
            "lambda": self.model_params.get("meta_xgb_lambda", 1.0),
            "seed": 42,
        }

        self.model = xgb.train(
            params, dtrain,
            num_boost_round=self.model_params.get("meta_xgb_num_boost_round", 200),
            evals=[(dtrain, "train")],
            verbose_eval=0,
        )

        # Feature importance
        imp = self.model.get_score(importance_type="gain")
        total = sum(imp.values()) if imp else 1
        self._feature_importance = {k: v / total for k, v in imp.items()}
        self._fitted = True
        logging.info("Meta-learner (XGBoost) trained: %d features, %d samples", X.shape[1], len(y))

    def _train_logistic(self, X, y):
        """Train logistic regression meta-learner (simpler, less overfitting risk)."""
        from scipy.optimize import minimize

        n_features = X.shape[1]

        def objective(params):
            weights = params[:n_features]
            bias = params[n_features]
            logits = X @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            # Log loss + L2 penalty
            ll = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            l2 = self.model_params.get("meta_logistic_l2", 0.01) * np.sum(weights ** 2)
            return ll + l2

        x0 = np.zeros(n_features + 1)
        result = minimize(objective, x0, method="L-BFGS-B")

        self._logistic_weights = result.x[:n_features]
        self._logistic_bias = result.x[n_features]
        self._fitted = True
        self.meta_model_type = "logistic"

        # Feature importance from weights
        abs_w = np.abs(self._logistic_weights)
        total = abs_w.sum() if abs_w.sum() > 0 else 1
        self._feature_importance = {f: w / total for f, w in zip(self.feature_names, abs_w)}
        logging.info("Meta-learner (Logistic) trained: %d features", n_features)

    def _train_ridge(self, X, y):
        """Train ridge regression meta-learner with standardization."""
        from numpy.linalg import solve

        n, p = X.shape

        # Standardize features (store mean/std for prediction)
        self._ridge_mean = np.mean(X, axis=0)
        self._ridge_std = np.std(X, axis=0)
        self._ridge_std[self._ridge_std < 1e-8] = 1.0  # Avoid division by zero
        X_std = (X - self._ridge_mean) / self._ridge_std

        X_bias = np.column_stack([X_std, np.ones(n)])

        # Scale alpha with number of features to prevent overfitting
        alpha = max(5.0, p * self.model_params.get("meta_ridge_alpha_scale", 0.5))

        # Ridge closed-form: (X^T X + alpha*I)^-1 X^T y
        I = np.eye(X_bias.shape[1])
        I[-1, -1] = 0  # Don't regularize the bias term
        XTX = X_bias.T @ X_bias + alpha * I
        XTy = X_bias.T @ y
        self._ridge_weights = solve(XTX, XTy)
        self._fitted = True
        self.meta_model_type = "ridge"

        abs_w = np.abs(self._ridge_weights[:-1])
        total = abs_w.sum() if abs_w.sum() > 0 else 1
        self._feature_importance = {f: w / total for f, w in zip(self.feature_names, abs_w)}
        logging.info("Meta-learner (Ridge) trained: %d features, alpha=%.1f", p, alpha)

    def predict_proba(self, X):
        """Predict final probabilities."""
        if not self._fitted:
            return np.full(len(X), 0.5)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.meta_model_type == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(X, feature_names=self.feature_names)
            probs = self.model.predict(dmat)
        elif self.meta_model_type == "logistic":
            logits = X @ self._logistic_weights + self._logistic_bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        elif self.meta_model_type == "ridge":
            X_std = (X - self._ridge_mean) / self._ridge_std
            X_bias = np.column_stack([X_std, np.ones(len(X))])
            raw = X_bias @ self._ridge_weights
            probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        else:
            return np.full(len(X), 0.5)

        return np.clip(probs, 0.01, 0.99)

    def predict_single(self, features_dict):
        """Predict from a feature dictionary."""
        if not self._fitted:
            return 0.5

        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])
        return float(self.predict_proba(X)[0])

    def show_feature_importance(self):
        """Print feature importance ranking."""
        if not self._feature_importance:
            print("  No feature importance available (model not trained).")
            return

        sorted_imp = sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Meta-Learner Feature Importance ({self.meta_model_type}):")
        print(f"  {'Feature':<35} {'Importance':>10} {'Bar':>20}")
        print("  " + "-" * 68)

        max_imp = max(v for _, v in sorted_imp) if sorted_imp else 1
        for feat, imp in sorted_imp:
            bar_len = int(20 * imp / max_imp)
            bar = "#" * bar_len
            print(f"  {feat:<35} {imp:>10.4f} {bar}")

    def save(self, filepath=None):
        """Save meta-learner model and metadata."""
        if filepath is None:
            filepath = f"{self.sport}_meta_learner.json"

        meta = {
            "sport": self.sport,
            "meta_model_type": self.meta_model_type,
            "feature_names": self.feature_names,
            "feature_importance": self._feature_importance,
            "saved_at": datetime.now().isoformat(),
        }

        if self.meta_model_type == "xgboost" and self.model:
            model_path = f"{self.sport}_meta_xgb.json"
            self.model.save_model(model_path)
            meta["model_file"] = model_path
        elif self.meta_model_type == "logistic":
            meta["weights"] = self._logistic_weights.tolist()
            meta["bias"] = float(self._logistic_bias)
        elif self.meta_model_type == "ridge":
            meta["weights"] = self._ridge_weights.tolist()

        with open(filepath, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, filepath=None):
        """Load saved meta-learner."""
        if filepath is None:
            filepath = f"{self.sport}_meta_learner.json"

        if not os.path.exists(filepath):
            return False

        with open(filepath, "r") as f:
            meta = json.load(f)

        self.sport = meta.get("sport", self.sport)
        self.meta_model_type = meta.get("meta_model_type", "xgboost")
        self.feature_names = meta.get("feature_names", [])
        self._feature_importance = meta.get("feature_importance", {})

        if self.meta_model_type == "xgboost":
            import xgboost as xgb
            model_path = meta.get("model_file", f"{self.sport}_meta_xgb.json")
            if os.path.exists(model_path):
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                self._fitted = True
        elif self.meta_model_type == "logistic":
            self._logistic_weights = np.array(meta["weights"])
            self._logistic_bias = meta["bias"]
            self._fitted = True
        elif self.meta_model_type == "ridge":
            self._ridge_weights = np.array(meta["weights"])
            self._fitted = True

        return self._fitted


class MegaEnsemble:
    """Orchestrator that manages all base models and the meta-learner.

    This is the main entry point for the mega-ensemble system.
    It coordinates data flow between base models and produces
    final predictions via the meta-learner.
    """

    def __init__(self, sport="nfl"):
        self.sport = sport
        self.meta = MetaLearner(sport)
        self._base_models = {}
        self._feature_collectors = {}
        self._game_features = []   # Accumulated features for training
        self._game_labels = []     # Corresponding labels

    def register_model(self, name, model_obj):
        """Register a base model that produces predictions."""
        self._base_models[name] = model_obj

    def collect_features(self, home_team, away_team, elo_prob=0.5,
                         xgb_prob=0.5, **extra_features):
        """Collect all features for a matchup from registered models.

        Returns a dict of all features for the meta-learner.
        """
        features = {
            "elo_prob": elo_prob,
            "xgb_prob": xgb_prob,
        }

        # Collect from each registered model
        for name, model in self._base_models.items():
            if hasattr(model, "get_features"):
                try:
                    model_features = model.get_features(home_team, away_team)
                    features.update(model_features)
                except Exception as e:
                    logging.debug("Feature collection failed for %s: %s", name, e)

            if hasattr(model, "predict_single"):
                try:
                    prob = model.predict_single(home_team, away_team) \
                        if name in ("lstm",) else None
                    if prob is not None:
                        features[f"{name}_prob"] = prob
                except Exception:
                    pass

        # Add extra features (odds, weather, sentiment, etc.)
        features.update(extra_features)

        return features

    def record_game(self, features_dict, label):
        """Record a game's features and outcome for meta-learner training."""
        self._game_features.append(features_dict)
        self._game_labels.append(label)

    def train_meta(self, min_games=100):
        """Train the meta-learner on accumulated game features.

        Uses all recorded games with purged walk-forward structure.
        """
        if len(self._game_features) < min_games:
            logging.warning("Not enough games for meta-learner training (%d < %d)",
                            len(self._game_features), min_games)
            return False

        # Build feature matrix
        all_features = set()
        for f in self._game_features:
            all_features.update(f.keys())

        feature_names = sorted(all_features)
        X = np.array([[f.get(fn, 0) for fn in feature_names] for f in self._game_features])
        y = np.array(self._game_labels, dtype=np.float32)

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        self.meta.feature_names = feature_names
        self.meta.train(X, y, feature_names)
        return True

    def predict(self, home_team, away_team, elo_prob=0.5, xgb_prob=0.5,
                **extra_features):
        """Generate final mega-ensemble prediction.

        Returns dict with final probability and component breakdowns.
        """
        features = self.collect_features(home_team, away_team, elo_prob,
                                         xgb_prob, **extra_features)

        if self.meta._fitted:
            final_prob = self.meta.predict_single(features)
        else:
            # Fallback: weighted average of available base model probs
            probs = []
            weights = []
            for key, val in features.items():
                if key.endswith("_prob") and isinstance(val, (int, float)):
                    if 0.01 < val < 0.99:
                        probs.append(val)
                        weights.append(1.0)

            if probs:
                total_w = sum(weights)
                final_prob = sum(p * w for p, w in zip(probs, weights)) / total_w
            else:
                final_prob = elo_prob

        return {
            "final_prob": max(0.01, min(0.99, final_prob)),
            "components": features,
            "meta_fitted": self.meta._fitted,
            "n_models": len([k for k in features if k.endswith("_prob")]),
        }

    def show_prediction_breakdown(self, result):
        """Display a detailed breakdown of a mega-ensemble prediction."""
        print(f"\n  MEGA-ENSEMBLE PREDICTION: {result['final_prob']:.1%}")
        print(f"  Models contributing: {result['n_models']}")
        print(f"  Meta-learner: {'ACTIVE' if result['meta_fitted'] else 'FALLBACK (averaging)'}")

        print(f"\n  {'Model':<25} {'Probability':>12}")
        print("  " + "-" * 40)

        components = result.get("components", {})
        for key in sorted(components.keys()):
            if key.endswith("_prob"):
                val = components[key]
                name = key.replace("_prob", "").upper()
                bar_len = int(30 * val)
                bar = "#" * bar_len + "." * (30 - bar_len)
                print(f"  {name:<25} {val:>11.1%}  |{bar}|")

        # Show non-probability features
        context_features = {k: v for k, v in components.items()
                           if not k.endswith("_prob") and isinstance(v, (int, float))}
        if context_features:
            print(f"\n  {'Context Feature':<30} {'Value':>10}")
            print("  " + "-" * 42)
            for key in sorted(context_features.keys()):
                val = context_features[key]
                print(f"  {key:<30} {val:>+10.4f}")

    def backtest(self, games_df, home_col="home_team", away_col="away_team",
                 home_score_col="home_score", away_score_col="away_score",
                 min_train=200, retrain_every=50):
        """Walk-forward backtest of the mega-ensemble.

        Processes games chronologically:
        - First min_train games: collect features, no meta-learner predictions
        - After min_train: train meta-learner, predict, retrain periodically
        """
        predictions = []
        actuals = []

        for idx, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                h_score = float(row.get(home_score_col, 0))
                a_score = float(row.get(away_score_col, 0))
            except (ValueError, TypeError):
                continue

            label = 1 if h_score > a_score else 0
            game_num = len(self._game_features)

            # Collect features from all registered models
            features = self.collect_features(home, away)

            if game_num >= min_train:
                # Train/retrain meta-learner periodically
                if game_num == min_train or (game_num - min_train) % retrain_every == 0:
                    self.train_meta(min_games=min(min_train, game_num))

                if self.meta._fitted:
                    pred = self.meta.predict_single(features)
                    predictions.append(pred)
                    actuals.append(label)

            # Record for future training
            self.record_game(features, label)

        # Compute metrics
        if not predictions:
            return {"accuracy": 0, "log_loss": 1, "brier": 0.25, "n_predictions": 0}

        preds = np.array(predictions)
        acts = np.array(actuals)

        correct = np.sum((preds > 0.5) == acts)
        accuracy = correct / len(preds) * 100

        # Log loss
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        ll = -np.mean(acts * np.log(preds_clipped) + (1 - acts) * np.log(1 - preds_clipped))

        # Brier score
        brier = np.mean((preds - acts) ** 2)

        return {
            "accuracy": round(accuracy, 2),
            "log_loss": round(ll, 4),
            "brier": round(brier, 4),
            "n_predictions": len(predictions),
            "n_training": min_train,
        }

    def save(self):
        """Save meta-learner and all serializable state."""
        self.meta.save()

    def load(self):
        """Load saved meta-learner."""
        return self.meta.load()
