"""Copula model for joint offense/defense dependency modeling.

Models the joint distribution of team offense and defense.
Captures tail dependencies — when offense collapses, does defense too?

Uses Gaussian copula (no extra dependencies beyond numpy/scipy).
"""

import logging
import math
from collections import defaultdict

import numpy as np
from scipy import stats


class TeamCopula:
    """Models joint offense-defense dependency for a single team."""

    def __init__(self, min_games=15):
        self.min_games = min_games
        self.scores_for = []
        self.scores_against = []
        self._rho = 0.0           # Correlation parameter
        self._tail_dep = 0.0      # Lower tail dependence
        self._fitted = False

    def add_game(self, points_for, points_against):
        self.scores_for.append(points_for)
        self.scores_against.append(points_against)

    def fit(self):
        """Fit Gaussian copula to offense-defense relationship."""
        if len(self.scores_for) < self.min_games:
            return False

        off = np.array(self.scores_for, dtype=float)
        defe = np.array(self.scores_against, dtype=float)

        # Rank-based correlation (Spearman) — more robust than Pearson
        rho, _ = stats.spearmanr(off, defe)
        if np.isnan(rho):
            rho = 0.0
        self._rho = float(rho)

        # Estimate lower tail dependence
        # (probability both offense AND defense are bad simultaneously)
        n = len(off)
        off_rank = stats.rankdata(off) / (n + 1)
        def_rank = stats.rankdata(defe) / (n + 1)

        # Lower tail: both below 25th percentile
        threshold = 0.25
        both_bad = np.sum((off_rank < threshold) & (def_rank < threshold)) / n
        self._tail_dep = float(both_bad / threshold) if threshold > 0 else 0

        self._fitted = True
        return True

    def collapse_probability(self):
        """Probability of simultaneous offense/defense failure.

        High value = team tends to collapse completely (bad offense + bad defense).
        """
        if not self._fitted:
            return 0.1

        # Positive correlation between low scoring and high allowing = collapse risk
        if self._rho > 0:
            return min(0.5, 0.1 + self._rho * 0.3)
        return max(0.02, 0.1 + self._rho * 0.1)

    def get_features(self):
        """Get copula-based features."""
        self.fit()
        return {
            "off_def_correlation": self._rho,
            "tail_dependence": self._tail_dep,
            "collapse_prob": self.collapse_probability(),
        }


class LeagueCopula:
    """Manages copula models for all teams."""

    def __init__(self, min_games=15):
        self.min_games = min_games
        self.teams = defaultdict(lambda: TeamCopula(min_games))

    def add_game(self, team, points_for, points_against):
        self.teams[team].add_game(points_for, points_against)

    def get_features(self, home_team, away_team):
        """Get copula features for a matchup."""
        home_feats = self.teams[home_team].get_features() if home_team in self.teams else {}
        away_feats = self.teams[away_team].get_features() if away_team in self.teams else {}

        h_collapse = home_feats.get("collapse_prob", 0.1)
        a_collapse = away_feats.get("collapse_prob", 0.1)
        h_corr = home_feats.get("off_def_correlation", 0)
        a_corr = away_feats.get("off_def_correlation", 0)
        h_tail = home_feats.get("tail_dependence", 0)
        a_tail = away_feats.get("tail_dependence", 0)

        return {
            "collapse_prob_diff": h_collapse - a_collapse,
            "off_def_corr_diff": h_corr - a_corr,
            "tail_dep_diff": h_tail - a_tail,
            "combined_collapse_risk": h_collapse + a_collapse,
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
            self.add_game(home, h_score, a_score)
            self.add_game(away, a_score, h_score)
        return len(self.teams)
