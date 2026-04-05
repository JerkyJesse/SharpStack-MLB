"""Bradley-Terry Paired Comparison Model.

Historical origin: Ralph Bradley & Milton Terry (1952).
Thurstone's model of comparative judgment (1927) as precursor.

The most mathematically principled way to rank teams from paired comparisons.
P(i beats j) = pi_i / (pi_i + pi_j)

Unlike Elo which updates sequentially, Bradley-Terry fits ALL games simultaneously
via maximum likelihood — producing globally optimal strength parameters.

Key advantages over Elo:
  - No path dependence (game order doesn't matter)
  - MLE guarantees optimality
  - Natural confidence intervals via Fisher information

Packages: numpy, scipy (built-in)
"""

import math
from collections import defaultdict

import numpy as np
from scipy import optimize


class BradleyTerryModel:
    """Maximum likelihood Bradley-Terry model."""

    def __init__(self, home_advantage=True, decay=0.99, min_games=20, **kwargs):
        self.home_advantage = home_advantage
        self.decay = decay
        self.min_games = min_games
        self.teams = set()
        self._games = []  # (home, away, home_won, weight)
        self._strengths = {}  # team -> log-strength
        self._home_adv = 0.0  # log home advantage parameter
        self._fitted = False

    def add_game(self, home, away, home_won, game_idx=0):
        """Record a game result."""
        self.teams.add(home)
        self.teams.add(away)
        n = len(self._games)
        weight = self.decay ** max(0, n - game_idx)
        self._games.append((home, away, int(home_won), weight))

    def fit(self):
        """Fit Bradley-Terry model via maximum likelihood.

        Uses iterative proportional fitting (MM algorithm):
        pi_i = wins_i / sum_j(n_ij / (pi_i + pi_j))
        """
        if len(self._games) < self.min_games:
            return False

        team_list = sorted(self.teams)
        n_teams = len(team_list)
        team_idx = {t: i for i, t in enumerate(team_list)}

        # Count weighted wins and matchup weights
        wins = np.zeros(n_teams)
        matchup_weights = np.zeros((n_teams, n_teams))

        for home, away, home_won, w in self._games:
            hi, ai = team_idx[home], team_idx[away]
            if home_won:
                wins[hi] += w
            else:
                wins[ai] += w
            matchup_weights[hi, ai] += w
            matchup_weights[ai, hi] += w

        # Initialize strengths
        pi = np.ones(n_teams)

        # MM algorithm iterations
        for iteration in range(100):
            pi_old = pi.copy()

            for i in range(n_teams):
                denom = 0.0
                for j in range(n_teams):
                    if i == j:
                        continue
                    n_ij = matchup_weights[i, j] + matchup_weights[j, i]
                    if n_ij > 0:
                        denom += n_ij / (pi[i] + pi[j])

                if denom > 0 and wins[i] > 0:
                    pi[i] = wins[i] / denom
                else:
                    pi[i] = max(pi[i], 0.001)

            # Normalize (geometric mean = 1)
            pi /= np.exp(np.mean(np.log(np.maximum(pi, 1e-10))))

            # Check convergence
            if np.max(np.abs(pi - pi_old)) < 1e-6:
                break

        # Store as log-strengths
        for t in team_list:
            self._strengths[t] = math.log(max(pi[team_idx[t]], 1e-10))

        # Estimate home advantage from data
        if self.home_advantage:
            home_wins = sum(w for _, _, hw, w in self._games if hw)
            total = sum(w for _, _, _, w in self._games)
            if total > 0:
                hw_rate = home_wins / total
                self._home_adv = math.log(max(hw_rate / max(1 - hw_rate, 0.01), 0.1))
            else:
                self._home_adv = 0.0

        self._fitted = True
        return True

    def win_prob(self, home, away):
        """P(home beats away) under the Bradley-Terry model."""
        if not self._fitted:
            return 0.5

        s_h = self._strengths.get(home, 0.0)
        s_a = self._strengths.get(away, 0.0)

        logit = s_h - s_a
        if self.home_advantage:
            logit += self._home_adv

        prob = 1.0 / (1.0 + math.exp(-logit))
        return float(np.clip(prob, 0.01, 0.99))

    def get_features(self, home, away):
        """Get Bradley-Terry features for a matchup."""
        prob = self.win_prob(home, away)
        s_h = self._strengths.get(home, 0.0)
        s_a = self._strengths.get(away, 0.0)

        return {
            "bt_prob": prob,
            "bt_strength_diff": s_h - s_a,
            "bt_home_strength": s_h,
            "bt_away_strength": s_a,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for idx, row in games_df.iterrows():
            try:
                h_score = float(row[home_score_col])
                a_score = float(row[away_score_col])
                self.add_game(row[home_col], row[away_col], h_score > a_score, game_idx=idx)
            except (ValueError, TypeError, KeyError):
                continue
        return self.fit()
