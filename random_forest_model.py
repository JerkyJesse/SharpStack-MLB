"""Random Forest model for sports predictions.

Historical origin: Leo Breiman (2001) - Random Forests.
Extension of bagging (1996) + random subspace method (1998).

Key insight: DIVERSITY in the ensemble is what matters.
Each tree sees different random subsets of features and data.
The forest's aggregate is more robust than any individual tree.

Why add this when we have LightGBM/CatBoost:
  - RF uses bagging (parallel), GBMs use boosting (sequential)
  - RF is more robust to noisy features (doesn't chase residuals)
  - RF provides natural confidence intervals via tree agreement
  - Uncorrelated errors with GBMs = better ensemble diversity

Packages: scikit-learn (pip install scikit-learn)
"""

import logging
from collections import defaultdict

import numpy as np


class RandomForestPredictor:
    """Random Forest classifier using pure numpy (no sklearn dependency)."""

    def __init__(self, sport="nfl", n_trees=100, max_depth=5,
                 min_samples_leaf=10, max_features="sqrt", **kwargs):
        self.sport = sport
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self._trees = []
        self._feature_indices = []  # Which features each tree uses
        self._fitted = False
        self.feature_names = []

    def _build_tree(self, X, y, rng, depth=0):
        """Build a single decision tree recursively."""
        n, p = X.shape

        # Leaf conditions
        if (depth >= self.max_depth or n <= self.min_samples_leaf * 2
                or len(np.unique(y)) <= 1):
            return {"leaf": True, "value": float(np.mean(y))}

        # Random feature subset
        if self.max_features == "sqrt":
            n_features = max(1, int(np.sqrt(p)))
        elif isinstance(self.max_features, int):
            n_features = min(self.max_features, p)
        else:
            n_features = p

        feature_subset = rng.choice(p, n_features, replace=False)

        best_gain = -1
        best_feature = 0
        best_threshold = 0

        parent_impurity = self._gini(y)

        for feat in feature_subset:
            values = X[:, feat]
            thresholds = np.unique(values)
            if len(thresholds) > 20:
                thresholds = np.percentile(values, np.linspace(10, 90, 10))

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                weighted_impurity = (left_mask.sum() * left_impurity +
                                     right_mask.sum() * right_impurity) / n

                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh

        if best_gain <= 0:
            return {"leaf": True, "value": float(np.mean(y))}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], rng, depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], rng, depth + 1),
        }

    def _gini(self, y):
        """Gini impurity."""
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 2 * p * (1 - p)

    def _predict_tree(self, tree, x):
        """Predict a single sample through one tree."""
        if tree["leaf"]:
            return tree["value"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        return self._predict_tree(tree["right"], x)

    def train(self, X, y, feature_names=None):
        """Train the random forest."""
        if isinstance(X, np.ndarray):
            pass
        else:
            X = np.array(X)

        y = np.array(y, dtype=np.float32)
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Scale complexity for small datasets
        n_samples = X.shape[0]
        if n_samples < 200:
            actual_trees = min(self.n_trees, 30)
            actual_depth = min(self.max_depth, 3)
            actual_min_leaf = max(self.min_samples_leaf, n_samples // 10)
        elif n_samples < 500:
            actual_trees = min(self.n_trees, 50)
            actual_depth = min(self.max_depth, 4)
            actual_min_leaf = self.min_samples_leaf
        else:
            actual_trees = self.n_trees
            actual_depth = self.max_depth
            actual_min_leaf = self.min_samples_leaf

        self._trees = []
        n = len(X)

        for i in range(actual_trees):
            rng = np.random.RandomState(42 + i)

            # Bootstrap sample
            indices = rng.randint(0, n, n)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = self._build_tree(X_boot, y_boot, rng, depth=0)
            self._trees.append(tree)

        self._fitted = True
        logging.debug("Random Forest trained: %d trees, %d features, %d samples",
                      len(self._trees), X.shape[1], n)

    def predict_proba(self, X):
        """Predict probabilities (average of all trees)."""
        if not self._fitted:
            return np.full(len(X), 0.5)

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        preds = np.zeros(len(X))
        for tree in self._trees:
            for i in range(len(X)):
                preds[i] += self._predict_tree(tree, X[i])
        preds /= len(self._trees)
        return np.clip(preds, 0.01, 0.99)

    def tree_agreement(self, X):
        """How much do trees agree? High agreement = high confidence.

        Returns std of tree predictions. Low std = high agreement.
        """
        if not self._fitted:
            return np.full(len(X), 0.25)

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        all_preds = np.zeros((len(self._trees), len(X)))
        for t, tree in enumerate(self._trees):
            for i in range(len(X)):
                all_preds[t, i] = self._predict_tree(tree, X[i])

        return np.std(all_preds, axis=0)
