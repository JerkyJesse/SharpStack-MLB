"""Team Archetype Clustering for sports predictions.

Historical origin: k-means (Lloyd 1957/1982), DBSCAN (1996).
Applied to sports:
  - Cluster teams by playing style (offensive vs defensive, etc.)
  - Certain archetypes match up well/poorly against others
  - Distance from cluster center = how "typical" a team is
  - Cluster transitions reveal strategic shifts

Packages: scikit-learn (pip install scikit-learn)
"""

import logging
from collections import defaultdict

import numpy as np


class TeamProfile:
    """Multi-dimensional team profile for clustering."""

    def __init__(self, min_games=10):
        self.min_games = min_games
        self.scores_for = []
        self.scores_against = []
        self.margins = []
        self.results = []

    def add_game(self, score_for, score_against, won):
        self.scores_for.append(score_for)
        self.scores_against.append(score_against)
        self.margins.append(score_for - score_against)
        self.results.append(won)

    def get_profile_vector(self, window=None):
        """Build a feature vector describing the team's playing style."""
        sf = self.scores_for[-window:] if window else self.scores_for
        sa = self.scores_against[-window:] if window else self.scores_against
        margins = self.margins[-window:] if window else self.margins
        results = self.results[-window:] if window else self.results

        if len(sf) < self.min_games:
            return None

        sf_arr = np.array(sf, dtype=float)
        sa_arr = np.array(sa, dtype=float)
        margins_arr = np.array(margins, dtype=float)

        return np.array([
            np.mean(sf_arr),                    # Offensive output
            np.mean(sa_arr),                    # Defensive output
            np.std(sf_arr),                     # Offensive consistency
            np.std(sa_arr),                     # Defensive consistency
            np.mean(results),                   # Win rate
            np.mean(margins_arr),               # Average margin
            np.std(margins_arr),                # Margin volatility
            np.mean(sf_arr) / max(np.mean(sa_arr), 0.1),  # Off/Def ratio
        ])

    FEATURE_NAMES = [
        "offense", "defense", "off_consistency", "def_consistency",
        "win_rate", "avg_margin", "margin_volatility", "off_def_ratio",
    ]


class LeagueClustering:
    """Clusters teams into archetypes and computes matchup features."""

    def __init__(self, n_clusters=4, min_games=10, **kwargs):
        self.n_clusters = n_clusters
        self.min_games = min_games
        self.teams = defaultdict(lambda: TeamProfile(min_games))
        self._cluster_labels = {}
        self._cluster_centers = None
        self._team_distances = {}  # Distance from cluster center
        self._fitted = False
        # Matchup advantage matrix: which archetypes beat which
        self._matchup_matrix = np.zeros((n_clusters, n_clusters))

    def add_game(self, team, score_for, score_against, won):
        self.teams[team].add_game(score_for, score_against, won)

    def fit(self, window=None):
        """Cluster teams based on their playing style profiles."""
        profiles = {}
        for team, prof in self.teams.items():
            vec = prof.get_profile_vector(window)
            if vec is not None:
                profiles[team] = vec

        if len(profiles) < self.n_clusters + 1:
            return False

        team_names = list(profiles.keys())
        X = np.array([profiles[t] for t in team_names])

        # Standardize features
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        X_std = (X - mean) / std

        # k-means clustering (manual to avoid sklearn dependency)
        labels, centers = self._kmeans(X_std, self.n_clusters)

        for i, team in enumerate(team_names):
            self._cluster_labels[team] = int(labels[i])
            self._team_distances[team] = float(np.linalg.norm(X_std[i] - centers[labels[i]]))

        self._cluster_centers = centers
        self._fitted = True
        return True

    def _kmeans(self, X, k, max_iter=50):
        """Simple k-means clustering."""
        n = len(X)
        rng = np.random.RandomState(42)

        # Initialize with k-means++
        centers = [X[rng.randint(n)]]
        for _ in range(k - 1):
            dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centers) for x in X])
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centers.append(X[idx])
        centers = np.array(centers)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            new_labels = np.array([np.argmin([np.linalg.norm(x - c) for c in centers]) for x in X])
            if np.all(new_labels == labels):
                break
            labels = new_labels

            # Update
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centers[j] = X[mask].mean(axis=0)

        return labels, centers

    def compute_matchup_advantages(self, games_df, home_col="home_team", away_col="away_team",
                                    home_score_col="home_score", away_score_col="away_score"):
        """Learn which cluster archetypes beat which."""
        if not self._fitted:
            return

        matchup_wins = np.zeros((self.n_clusters, self.n_clusters))
        matchup_total = np.zeros((self.n_clusters, self.n_clusters))

        for _, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")

            if home not in self._cluster_labels or away not in self._cluster_labels:
                continue

            h_cluster = self._cluster_labels[home]
            a_cluster = self._cluster_labels[away]
            try:
                home_won = float(row[home_score_col]) > float(row[away_score_col])
            except (ValueError, TypeError):
                continue

            matchup_total[h_cluster, a_cluster] += 1
            if home_won:
                matchup_wins[h_cluster, a_cluster] += 1

        # Win rate matrix
        matchup_total[matchup_total == 0] = 1
        self._matchup_matrix = matchup_wins / matchup_total

    def get_features(self, home_team, away_team):
        """Get clustering features for a matchup."""
        if not self._fitted:
            return {
                "archetype_matchup_prob": 0.5,
                "home_typicality": 0.0,
                "away_typicality": 0.0,
                "same_archetype": 0,
            }

        h_cluster = self._cluster_labels.get(home_team, 0)
        a_cluster = self._cluster_labels.get(away_team, 0)
        h_dist = self._team_distances.get(home_team, 1.0)
        a_dist = self._team_distances.get(away_team, 1.0)

        matchup_prob = float(self._matchup_matrix[h_cluster, a_cluster]) if self._matchup_matrix.sum() > 0 else 0.5

        return {
            "archetype_matchup_prob": matchup_prob,
            "home_typicality": h_dist,  # Lower = more typical of archetype
            "away_typicality": a_dist,
            "typicality_diff": a_dist - h_dist,
            "same_archetype": 1 if h_cluster == a_cluster else 0,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                h_score = float(row[home_score_col])
                a_score = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(home, h_score, a_score, h_score > a_score)
            self.add_game(away, a_score, h_score, a_score > h_score)

        self.fit()
        self.compute_matchup_advantages(games_df, home_col, away_col,
                                        home_score_col, away_score_col)
        return len(self.teams)
