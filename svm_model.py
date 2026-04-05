"""Support Vector Machine model for sports predictions.

Historical origin: Vladimir Vapnik & colleagues (1990s).
SVM finds the maximum-margin hyperplane separating classes.

Why add this when we have LightGBM/CatBoost/RF:
  - SVM uses a fundamentally different approach (margin maximization)
  - RBF kernel captures non-linear boundaries that trees may miss
  - More robust to feature scaling issues
  - Uncorrelated errors with tree-based models = better ensemble diversity

Packages: scikit-learn (optional, falls back to numpy linear approximation)
"""

import logging
from collections import defaultdict

import numpy as np

# Try sklearn for full SVM (preferred)
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SVMPredictor:
    """SVM classifier for sports win probability prediction.

    Uses sklearn SVC with probability=True if available,
    otherwise falls back to a numpy-based linear approximation
    using random Fourier features + logistic regression.
    """

    def __init__(self, sport="nfl", svm_C=1.0, svm_kernel="rbf",
                 svm_gamma="scale", svm_max_iter=1000, **kwargs):
        self.sport = sport
        self.C = float(svm_C)
        self.kernel = svm_kernel
        self.gamma = svm_gamma
        self.max_iter = int(svm_max_iter)
        self._fitted = False
        self.feature_names = []

        # sklearn SVM
        self._svm = None
        self._scaler = None

        # Fallback: random Fourier features + logistic regression
        self._rff_W = None
        self._rff_b = None
        self._lr_weights = None
        self._lr_bias = 0.0
        self._fallback_mean = None
        self._fallback_std = None

    def train(self, X, y, feature_names=None):
        """Train SVM on feature matrix."""
        if isinstance(X, np.ndarray):
            pass
        else:
            X = np.array(X)

        y = np.array(y, dtype=np.float32)

        if len(y) < 30:
            return

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Subsample if too large (SVM is O(n^2) in training)
        max_samples = 2000
        if len(X) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X_train = X[idx]
            y_train = y[idx]
        else:
            X_train = X
            y_train = y

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)

        if HAS_SKLEARN:
            self._train_sklearn(X_train, y_train)
        else:
            self._train_fallback(X_train, y_train)

    def _train_sklearn(self, X, y):
        """Train using sklearn SVC."""
        try:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            gamma = self.gamma
            if gamma == "scale":
                gamma = "scale"
            elif gamma == "auto":
                gamma = "auto"
            else:
                gamma = float(gamma)

            self._svm = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=gamma,
                probability=True,
                max_iter=self.max_iter,
                random_state=42,
                class_weight="balanced",
            )
            self._svm.fit(X_scaled, y.astype(int))
            self._fitted = True
            logging.debug("SVM (sklearn) trained: %d samples, %d features, C=%.1f",
                          len(X), X.shape[1], self.C)
        except Exception as e:
            logging.debug("sklearn SVM failed, trying fallback: %s", e)
            self._train_fallback(X, y)

    def _train_fallback(self, X, y):
        """Fallback: Random Fourier Features + logistic regression.

        Approximates RBF kernel SVM using Rahimi & Recht (2007).
        """
        from scipy.optimize import minimize

        n, d = X.shape

        # Standardize
        self._fallback_mean = np.mean(X, axis=0)
        self._fallback_std = np.std(X, axis=0)
        self._fallback_std[self._fallback_std < 1e-8] = 1.0
        X_std = (X - self._fallback_mean) / self._fallback_std

        # Random Fourier Features: z(x) = sqrt(2/D) * cos(Wx + b)
        n_rff = min(200, d * 4)
        rng = np.random.RandomState(42)

        # Gamma for RBF
        if self.gamma == "scale":
            gamma = 1.0 / (d * np.var(X_std))
        elif self.gamma == "auto":
            gamma = 1.0 / d
        else:
            gamma = float(self.gamma)

        self._rff_W = rng.normal(0, np.sqrt(2 * gamma), (d, n_rff))
        self._rff_b = rng.uniform(0, 2 * np.pi, n_rff)

        # Transform
        Z = np.sqrt(2.0 / n_rff) * np.cos(X_std @ self._rff_W + self._rff_b)

        # Logistic regression on RFF features
        p = Z.shape[1]

        def objective(params):
            w = params[:p]
            b = params[p]
            logits = Z @ w + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            ll = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            reg = (1.0 / (2.0 * self.C * n)) * np.sum(w ** 2)
            return ll + reg

        x0 = np.zeros(p + 1)
        result = minimize(objective, x0, method="L-BFGS-B",
                          options={"maxiter": self.max_iter})

        self._lr_weights = result.x[:p]
        self._lr_bias = result.x[p]
        self._fitted = True
        logging.debug("SVM (RFF fallback) trained: %d samples, %d RFF features",
                      len(X), n_rff)

    def predict_proba(self, X):
        """Predict probabilities for feature matrix."""
        if not self._fitted:
            return np.full(len(X), 0.5)

        X = np.nan_to_num(np.array(X, dtype=np.float64),
                          nan=0.0, posinf=1.0, neginf=-1.0)

        if self._svm is not None and HAS_SKLEARN:
            X_scaled = self._scaler.transform(X)
            probs = self._svm.predict_proba(X_scaled)[:, 1]
        else:
            # RFF fallback
            X_std = (X - self._fallback_mean) / self._fallback_std
            Z = np.sqrt(2.0 / self._rff_W.shape[1]) * np.cos(
                X_std @ self._rff_W + self._rff_b)
            logits = Z @ self._lr_weights + self._lr_bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

        return np.clip(probs, 0.01, 0.99)
