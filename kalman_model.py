"""Kalman Filter for dynamic team strength tracking.

Shared module across sports. Separates true team strength (signal) from
game-to-game variance (noise). Adapts learning rate automatically via
the Kalman gain — uncertain teams learn faster from new results.

Package: filterpy (pip install filterpy)
"""

import logging
import math
from collections import defaultdict

import numpy as np


class TeamKalman:
    """Kalman filter tracking a team's offensive and defensive strength.

    State vector: [offensive_strength, defensive_strength]
    Observation: game score margin (offense - defense relative to opponent)
    """

    def __init__(self, process_noise=0.5, measurement_noise=13.0,
                 initial_strength=0.0, initial_uncertainty=100.0):
        """
        Args:
            process_noise: how much team strength can change per game (Q)
            measurement_noise: single-game score variance (R) — ~13 for NFL, ~4 for MLB
            initial_strength: starting strength estimate
            initial_uncertainty: starting uncertainty (high = learn fast)
        """
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State: [off_strength, def_strength]
        self.kf.x = np.array([[initial_strength], [initial_strength]])

        # State covariance (uncertainty)
        self.kf.P = np.eye(2) * initial_uncertainty

        # State transition (strength persists with some noise)
        self.kf.F = np.eye(2)

        # Process noise (how much strength can change per game)
        self.kf.Q = np.eye(2) * process_noise

        # Measurement noise (single game variance)
        self.kf.R = np.array([[measurement_noise ** 2]])

        # Measurement matrix: observe (off - opponent_def), set per game
        self.kf.H = np.array([[1.0, -1.0]])

        self.n_games = 0
        self.history = []

    @property
    def off_strength(self):
        return float(self.kf.x[0])

    @property
    def def_strength(self):
        return float(self.kf.x[1])

    @property
    def total_strength(self):
        return self.off_strength - self.def_strength

    @property
    def uncertainty(self):
        """Overall uncertainty (trace of covariance matrix)."""
        return float(np.trace(self.kf.P))

    @property
    def kalman_gain(self):
        """Current Kalman gain magnitude (how fast the filter is learning)."""
        return float(np.mean(np.abs(self.kf.K))) if hasattr(self.kf, 'K') and self.kf.K is not None else 0.5

    def predict(self):
        """Predict next state (no observation yet)."""
        self.kf.predict()

    def update_offense(self, margin, opponent_def_strength=0.0):
        """Update after observing a game where this team was on offense.

        margin: points scored minus points allowed
        opponent_def_strength: opponent's defensive Kalman strength
        """
        # Adjust observation for opponent quality
        adjusted_margin = margin - opponent_def_strength
        self.kf.predict()
        self.kf.update(np.array([[adjusted_margin]]))
        self.n_games += 1
        self.history.append({
            "off": self.off_strength,
            "def": self.def_strength,
            "uncertainty": self.uncertainty,
            "gain": self.kalman_gain,
        })

    def update_game(self, margin):
        """Simplified update using just the score margin."""
        self.kf.predict()
        self.kf.update(np.array([[margin]]))
        self.n_games += 1
        self.history.append({
            "off": self.off_strength,
            "def": self.def_strength,
            "uncertainty": self.uncertainty,
            "gain": self.kalman_gain,
        })


class LeagueKalman:
    """Manages Kalman filters for all teams in a league."""

    def __init__(self, process_noise=0.5, measurement_noise=13.0,
                 season_regression=0.33, **kwargs):
        """
        Args:
            process_noise: per-game strength volatility
            measurement_noise: single-game score std dev
                NFL: ~13, MLB: ~4, NBA: ~11, NHL: ~2.5
            season_regression: fraction to regress to mean at season boundaries
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.season_regression = season_regression
        self.teams = {}

    def _get_or_create(self, team):
        if team not in self.teams:
            self.teams[team] = TeamKalman(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
        return self.teams[team]

    def update_game(self, home_team, away_team, home_score, away_score):
        """Update both teams' Kalman filters after a game."""
        margin = home_score - away_score

        home_kf = self._get_or_create(home_team)
        away_kf = self._get_or_create(away_team)

        # Update home team with positive margin
        home_kf.update_game(margin)
        # Update away team with negative margin
        away_kf.update_game(-margin)

    def regress_to_mean(self):
        """Apply season regression to all teams."""
        if not self.teams:
            return

        mean_off = np.mean([t.off_strength for t in self.teams.values()])
        mean_def = np.mean([t.def_strength for t in self.teams.values()])

        for team in self.teams.values():
            team.kf.x[0] = team.kf.x[0] * (1 - self.season_regression) + mean_off * self.season_regression
            team.kf.x[1] = team.kf.x[1] * (1 - self.season_regression) + mean_def * self.season_regression
            # Increase uncertainty at season boundary
            team.kf.P *= 1.5

    def win_probability(self, home_team, away_team, home_advantage=0.0):
        """Compute win probability using Kalman-filtered strengths.

        Uses a logistic model on the strength differential.
        """
        home_kf = self._get_or_create(home_team)
        away_kf = self._get_or_create(away_team)

        # Strength differential (home offense vs away defense, minus away offense vs home defense)
        diff = (home_kf.off_strength - away_kf.def_strength) - \
               (away_kf.off_strength - home_kf.def_strength) + \
               home_advantage

        # Combined uncertainty
        total_var = home_kf.uncertainty + away_kf.uncertainty + self.measurement_noise ** 2

        # Logistic with uncertainty scaling
        scale = math.sqrt(3.0 / (math.pi ** 2) * total_var) if total_var > 0 else 1.0
        prob = 1.0 / (1.0 + math.exp(-diff / max(scale, 0.01)))

        return max(0.01, min(0.99, prob))

    def get_features(self, home_team, away_team):
        """Get Kalman-based features for a matchup.

        Returns dict of features for the meta-learner.
        """
        home_kf = self._get_or_create(home_team)
        away_kf = self._get_or_create(away_team)

        home_total = home_kf.off_strength - home_kf.def_strength
        away_total = away_kf.off_strength - away_kf.def_strength

        return {
            "kalman_home_off": home_kf.off_strength,
            "kalman_home_def": home_kf.def_strength,
            "kalman_away_off": away_kf.off_strength,
            "kalman_away_def": away_kf.def_strength,
            "kalman_strength_diff": home_total - away_total,
            "kalman_home_uncertainty": home_kf.uncertainty,
            "kalman_away_uncertainty": away_kf.uncertainty,
            "kalman_home_gain": home_kf.kalman_gain,
            "kalman_away_gain": away_kf.kalman_gain,
            "kalman_prob": self.win_probability(home_team, away_team),
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score",
                         season_col=None):
        """Build Kalman filters from a games DataFrame.

        Processes games chronologically, with season regression at boundaries.
        """
        last_season = None

        for _, row in games_df.iterrows():
            # Season regression at boundaries
            if season_col and season_col in row.index:
                current_season = row[season_col]
                if last_season is not None and current_season != last_season:
                    self.regress_to_mean()
                last_season = current_season

            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                h_score = float(row.get(home_score_col, 0))
                a_score = float(row.get(away_score_col, 0))
            except (ValueError, TypeError):
                continue

            self.update_game(home, away, h_score, a_score)

        return len(self.teams)

    def show_rankings(self):
        """Print team rankings by Kalman-filtered strength."""
        if not self.teams:
            print("  No Kalman data available.")
            return

        ranked = sorted(self.teams.items(),
                        key=lambda x: x[1].off_strength - x[1].def_strength,
                        reverse=True)

        print(f"\n  {'Rank':>4} {'Team':<25} {'Off':>7} {'Def':>7} {'Total':>7} {'Uncert':>7} {'Games':>6}")
        print("  " + "-" * 67)

        for i, (team, kf) in enumerate(ranked, 1):
            total = kf.off_strength - kf.def_strength
            print(f"  {i:>4} {team:<25} {kf.off_strength:>+7.2f} {kf.def_strength:>+7.2f} "
                  f"{total:>+7.2f} {kf.uncertainty:>7.1f} {kf.n_games:>6}")
