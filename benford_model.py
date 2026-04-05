"""Benford's Law model for sports predictions.

Historical origin: Simon Newcomb (1881), Frank Benford (1938).
First-digit distribution in naturally occurring data: P(d) = log10(1 + 1/d).

Applied to sports:
  - Team scoring patterns can deviate from expected digit distributions
  - High chi-squared deviation = anomalous scoring (unusual form)
  - Conformity to Benford's Law correlates with "natural" performance
  - Deviations may indicate unsustainable streaks (hot/cold)

For low-scoring sports (NHL: 0-8), we use cumulative multi-game
score windows to produce multi-digit numbers.

Packages: numpy, scipy (built-in)
"""

import math
from collections import defaultdict

import numpy as np


# Benford's expected first-digit distribution
BENFORD_EXPECTED = {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def _leading_digit(n):
    """Extract the leading digit of a positive number."""
    if n <= 0:
        return None
    n = abs(n)
    while n >= 10:
        n //= 10
    return int(n)


def _chi_squared_benford(digit_counts, total):
    """Compute chi-squared statistic against Benford's distribution."""
    if total < 5:
        return 0.0
    chi_sq = 0.0
    for d in range(1, 10):
        observed = digit_counts.get(d, 0)
        expected = BENFORD_EXPECTED[d] * total
        if expected > 0:
            chi_sq += (observed - expected) ** 2 / expected
    return chi_sq


class TeamBenford:
    """Benford's Law tracking for a single team."""

    def __init__(self, min_games=15, window=20):
        self.min_games = min_games
        self.window = window
        self.scores = []           # All scores (points for)
        self.opp_scores = []       # Opponent scores (points against)
        self.results = []          # Win/loss history
        self.cumulative_scores = []  # Rolling multi-game sums (for low-scoring sports)

    def add_game(self, score_for, score_against, won):
        """Add a game result."""
        self.scores.append(int(score_for))
        self.opp_scores.append(int(score_against))
        self.results.append(1 if won else 0)

        # Maintain rolling cumulative sums (windows of 3 games)
        if len(self.scores) >= 3:
            cum = sum(self.scores[-3:])
            self.cumulative_scores.append(cum)

    def get_features(self):
        """Get Benford's Law features."""
        if len(self.scores) < self.min_games:
            return {
                "anomaly": 0.0,
                "conformity": 0.0,
                "entropy": 0.0,
            }

        # Collect digits from both individual scores and cumulative scores
        digit_counts = defaultdict(int)
        total = 0

        # Use individual scores >= 10 (single digits excluded for Benford)
        for s in self.scores[-self.window:]:
            d = _leading_digit(s)
            if d is not None and s >= 10:
                digit_counts[d] += 1
                total += 1

        # Also use cumulative 3-game sums (always multi-digit for most sports)
        recent_cum = self.cumulative_scores[-self.window:] if self.cumulative_scores else []
        for s in recent_cum:
            d = _leading_digit(s)
            if d is not None:
                digit_counts[d] += 1
                total += 1

        # Chi-squared deviation from Benford's law
        chi_sq = _chi_squared_benford(digit_counts, total)

        # Normalize: chi-sq / degrees_of_freedom (8 for 9 digits - 1)
        # Values > 2 are significantly non-Benford
        anomaly = min(chi_sq / 8.0, 3.0) if total >= 5 else 0.0

        # Conformity: 1 - normalized deviation (1.0 = perfect Benford, 0 = max deviation)
        conformity = max(0.0, 1.0 - anomaly / 3.0)

        # Digit entropy: how spread/concentrated the digit distribution is
        entropy = 0.0
        if total > 0:
            for d in range(1, 10):
                p = digit_counts.get(d, 0) / total
                if p > 0:
                    entropy -= p * math.log2(p)
            # Normalize by max entropy (log2(9) ~ 3.17)
            entropy /= math.log2(9)

        return {
            "anomaly": float(anomaly),
            "conformity": float(conformity),
            "entropy": float(entropy),
        }


class LeagueBenford:
    """League-wide Benford's Law tracking."""

    def __init__(self, min_games=15, benford_window=None, **kwargs):
        self.min_games = min_games
        self.window = benford_window if benford_window is not None else 20
        self.teams = {}

    def _get_team(self, team):
        if team not in self.teams:
            self.teams[team] = TeamBenford(
                min_games=self.min_games,
                window=self.window,
            )
        return self.teams[team]

    def add_game(self, team, score_for, score_against, won):
        """Add a game result for a team."""
        self._get_team(team).add_game(score_for, score_against, won)

    def get_features(self, home_team, away_team):
        """Get differential Benford features for a matchup."""
        h = self._get_team(home_team).get_features()
        a = self._get_team(away_team).get_features()

        return {
            "benford_anomaly_diff": h["anomaly"] - a["anomaly"],
            "benford_conformity_diff": h["conformity"] - a["conformity"],
            "benford_entropy_diff": h["entropy"] - a["entropy"],
        }
