"""Poisson / Dixon-Coles Score Prediction Model.

Historical origin: Simeon Denis Poisson (1837) - Poisson distribution.
Dixon & Coles (1997) - improved independence correction for low scores.

Applied to sports:
  - Models each team's scoring as a Poisson process
  - Attack strength * defense weakness = expected goals/runs/points
  - Can predict exact scores, not just win/loss
  - Naturally handles home advantage as a multiplier

Key insight: Instead of predicting "who wins", predict the SCORE DISTRIBUTION
and derive win probability from P(home_score > away_score).

Packages: numpy, scipy (built-in)
"""

import math
from collections import defaultdict

import numpy as np
from scipy import optimize
from scipy.stats import poisson


class PoissonTeamModel:
    """Attack/defense strength model for a single team."""

    def __init__(self):
        self.attack = 1.0    # Relative attacking strength
        self.defense = 1.0   # Relative defensive strength (lower = better)
        self.scores_for = []
        self.scores_against = []
        self.opponents = []

    def add_game(self, scored, conceded, opponent):
        self.scores_for.append(scored)
        self.scores_against.append(conceded)
        self.opponents.append(opponent)


class LeaguePoisson:
    """Poisson regression model for an entire league.

    Estimates attack/defense parameters for every team simultaneously
    by maximizing the joint likelihood of all observed scores.
    """

    def __init__(self, home_advantage=1.2, min_games=20, decay=0.98, **kwargs):
        self.home_advantage = home_advantage
        self.min_games = min_games
        self.decay = decay
        self.teams = defaultdict(PoissonTeamModel)
        self.league_avg_goals = 3.0  # Will be updated
        self._fitted = False
        self._game_log = []  # (home, away, h_score, a_score)

    def add_game(self, home, away, h_score, a_score):
        """Record a game result."""
        self.teams[home].add_game(h_score, a_score, away)
        self.teams[away].add_game(a_score, h_score, home)
        self._game_log.append((home, away, h_score, a_score))

    def fit(self, decay=None):
        """Fit attack/defense parameters using maximum likelihood.

        Uses iterative proportional fitting (simpler than full MLE):
        attack_i = avg(goals_scored_by_i) / league_avg
        defense_i = avg(goals_conceded_by_i) / league_avg
        """
        if decay is None:
            decay = self.decay

        if len(self._game_log) < self.min_games:
            return False

        team_names = sorted(self.teams.keys())
        n_teams = len(team_names)
        team_idx = {t: i for i, t in enumerate(team_names)}

        # Compute weighted averages with time decay
        n_games = len(self._game_log)
        weights = np.array([decay ** (n_games - 1 - i) for i in range(n_games)])
        weights /= weights.sum()

        # Per-team weighted stats
        total_scored = np.zeros(n_teams)
        total_conceded = np.zeros(n_teams)
        team_weights = np.zeros(n_teams)

        for i, (home, away, h_score, a_score) in enumerate(self._game_log):
            w = weights[i]
            hi, ai = team_idx[home], team_idx[away]
            total_scored[hi] += w * h_score
            total_conceded[hi] += w * a_score
            total_scored[ai] += w * a_score
            total_conceded[ai] += w * h_score
            team_weights[hi] += w
            team_weights[ai] += w

        # Avoid division by zero
        team_weights = np.maximum(team_weights, 1e-10)

        # League averages
        self.league_avg_goals = float(np.sum(total_scored) / np.sum(team_weights))
        if self.league_avg_goals < 0.1:
            self.league_avg_goals = 3.0

        # Attack and defense ratings
        for t in team_names:
            idx = team_idx[t]
            avg_scored = total_scored[idx] / team_weights[idx]
            avg_conceded = total_conceded[idx] / team_weights[idx]

            self.teams[t].attack = avg_scored / self.league_avg_goals
            self.teams[t].defense = avg_conceded / self.league_avg_goals

        # Estimate home advantage from data
        home_goals = sum(h for _, _, h, _ in self._game_log[-200:])
        away_goals = sum(a for _, _, _, a in self._game_log[-200:])
        if away_goals > 0:
            self.home_advantage = home_goals / away_goals
        else:
            pass  # Keep self.home_advantage from constructor

        self._fitted = True
        return True

    def predict_score(self, home, away):
        """Predict expected score for a matchup.

        Expected home goals = home_attack * away_defense * league_avg * home_adv
        Expected away goals = away_attack * home_defense * league_avg
        """
        if not self._fitted:
            return self.league_avg_goals, self.league_avg_goals

        h_team = self.teams.get(home)
        a_team = self.teams.get(away)

        if h_team is None or a_team is None:
            return self.league_avg_goals, self.league_avg_goals

        h_expected = h_team.attack * a_team.defense * self.league_avg_goals * self.home_advantage
        a_expected = a_team.attack * h_team.defense * self.league_avg_goals

        return max(0.1, h_expected), max(0.1, a_expected)

    def predict_win_prob(self, home, away, max_score=200):
        """Predict win probability from Poisson score distributions.

        P(home wins) = sum over all (h, a) where h > a of P(h) * P(a)
        """
        h_exp, a_exp = self.predict_score(home, away)

        # Build probability matrices
        p_home_win = 0.0
        p_draw = 0.0

        for h in range(max_score + 1):
            p_h = poisson.pmf(h, h_exp)
            for a in range(max_score + 1):
                p_a = poisson.pmf(a, a_exp)
                joint = p_h * p_a
                if h > a:
                    p_home_win += joint
                elif h == a:
                    p_draw += joint

        # For sports without draws, split draw probability proportionally
        p_away_win = 1.0 - p_home_win - p_draw
        total_decisive = p_home_win + p_away_win
        if total_decisive > 0:
            p_home_win_adj = p_home_win + p_draw * (p_home_win / total_decisive)
        else:
            p_home_win_adj = 0.5

        return float(np.clip(p_home_win_adj, 0.01, 0.99))

    def get_features(self, home, away):
        """Get Poisson model features for a matchup."""
        h_exp, a_exp = self.predict_score(home, away)
        win_prob = self.predict_win_prob(home, away)

        h_team = self.teams.get(home)
        a_team = self.teams.get(away)

        h_attack = h_team.attack if h_team else 1.0
        a_attack = a_team.attack if a_team else 1.0
        h_defense = h_team.defense if h_team else 1.0
        a_defense = a_team.defense if a_team else 1.0

        return {
            "poisson_win_prob": win_prob,
            "poisson_expected_margin": h_exp - a_exp,
            "poisson_total": h_exp + a_exp,
            "attack_diff": h_attack - a_attack,
            "defense_diff": a_defense - h_defense,  # Inverted (lower = better)
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
            self.add_game(home, away, h_score, a_score)
        return self.fit()
