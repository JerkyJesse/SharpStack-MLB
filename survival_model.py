"""Survival Analysis for sports predictions.

Models the "lifetime" of win/loss streaks using hazard functions.
Some teams have durable streaks (low hazard); others are fragile (high hazard).
The hazard rate = probability a streak ends in the next game.

Also models time-to-event for regression to mean after extreme performances.

Package: lifelines (pip install lifelines) or numpy fallback
"""

import logging
import math
from collections import defaultdict

import numpy as np


class TeamSurvival:
    """Survival model for a single team's streak patterns."""

    def __init__(self, min_streaks=5):
        self.min_streaks = min_streaks
        self.win_streaks = []     # Completed win streak lengths
        self.loss_streaks = []    # Completed loss streak lengths
        self.current_streak = 0  # Positive=winning, negative=losing
        self._hazard_rate = None
        self._mean_win_streak = None
        self._mean_loss_streak = None

    def add_game(self, won):
        """Record a game result."""
        if won:
            if self.current_streak > 0:
                self.current_streak += 1
            else:
                if self.current_streak < 0:
                    self.loss_streaks.append(abs(self.current_streak))
                self.current_streak = 1
        else:
            if self.current_streak < 0:
                self.current_streak -= 1
            else:
                if self.current_streak > 0:
                    self.win_streaks.append(self.current_streak)
                self.current_streak = -1

    def fit(self):
        """Compute hazard rates from streak data."""
        if len(self.win_streaks) < self.min_streaks:
            return False

        self._mean_win_streak = np.mean(self.win_streaks) if self.win_streaks else 2.0
        self._mean_loss_streak = np.mean(self.loss_streaks) if self.loss_streaks else 2.0

        # Exponential hazard: h(t) = 1/mean_streak_length
        # Higher hazard = streaks end sooner = less consistent team
        self._hazard_rate = 1.0 / self._mean_win_streak if self._mean_win_streak > 0 else 0.5

        return True

    def streak_break_prob(self):
        """Probability the current streak ends next game.

        Uses a simple exponential survival model.
        P(streak ends | streak length = t) increases with t.
        """
        if self._hazard_rate is None:
            self.fit()
        if self._hazard_rate is None:
            return 0.5

        t = abs(self.current_streak)
        if t == 0:
            return 0.5

        # Increasing hazard with streak length (Weibull-like)
        # Teams on long streaks are more likely to regress
        if self.current_streak > 0:
            mean_len = self._mean_win_streak or 2.0
        else:
            mean_len = self._mean_loss_streak or 2.0

        # P(break) increases as streak exceeds team's typical length
        ratio = t / mean_len
        prob = 1.0 - math.exp(-ratio)
        return max(0.05, min(0.95, prob))

    def get_features(self):
        """Get survival-based features."""
        self.fit()
        return {
            "current_streak": self.current_streak,
            "streak_break_prob": self.streak_break_prob(),
            "mean_win_streak": self._mean_win_streak or 2.0,
            "mean_loss_streak": self._mean_loss_streak or 2.0,
            "streak_durability": (self._mean_win_streak or 2.0) / max(self._mean_loss_streak or 2.0, 0.1),
        }


class LeagueSurvival:
    """Manages survival models for all teams."""

    def __init__(self, min_streaks=5):
        self.min_streaks = min_streaks
        self.teams = defaultdict(lambda: TeamSurvival(min_streaks))

    def add_game(self, team, won):
        self.teams[team].add_game(won)

    def get_features(self, home_team, away_team):
        """Get survival features for a matchup."""
        home_feats = self.teams[home_team].get_features() if home_team in self.teams else {}
        away_feats = self.teams[away_team].get_features() if away_team in self.teams else {}

        h_break = home_feats.get("streak_break_prob", 0.5)
        a_break = away_feats.get("streak_break_prob", 0.5)
        h_streak = home_feats.get("current_streak", 0)
        a_streak = away_feats.get("current_streak", 0)
        h_dur = home_feats.get("streak_durability", 1.0)
        a_dur = away_feats.get("streak_durability", 1.0)

        # If home team is on a winning streak with high break probability,
        # regression is likely — reduces their edge
        home_regression_risk = h_break if h_streak > 0 else 0
        away_regression_risk = a_break if a_streak > 0 else 0

        # If away team is on a losing streak with high break probability,
        # bounce-back is likely — increases their chances
        home_bounce_back = h_break if h_streak < 0 else 0
        away_bounce_back = a_break if a_streak < 0 else 0

        return {
            "home_streak": h_streak,
            "away_streak": a_streak,
            "streak_diff": h_streak - a_streak,
            "home_break_prob": h_break,
            "away_break_prob": a_break,
            "durability_diff": h_dur - a_dur,
            "regression_risk_diff": home_regression_risk - away_regression_risk,
            "bounce_back_diff": home_bounce_back - away_bounce_back,
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
            self.add_game(home, h_score > a_score)
            self.add_game(away, a_score > h_score)
        return len(self.teams)
