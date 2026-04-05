"""Physics-inspired Momentum & Inertia model for sports predictions.

Historical origin: Newton's Laws of Motion (1687).
Applied to sports:
  - Momentum = mass (talent) * velocity (recent form change rate)
  - Inertia = resistance to change in current state (teams in motion stay in motion)
  - Angular momentum = rotational (cyclical) patterns in performance
  - Friction = regression force pulling toward league mean

Packages: numpy, scipy (built-in)
"""

import math
from collections import defaultdict

import numpy as np


class TeamMomentum:
    """Physics-based momentum tracking for a single team."""

    def __init__(self, mass=1.0, friction=0.05, min_games=5,
                 velocity_window=5, impulse_window=3):
        self.mass = mass          # Proxy for talent level
        self.friction = friction  # Regression-to-mean force
        self.min_games = min_games
        self.velocity_window = velocity_window
        self.impulse_window = impulse_window
        self.margins = []
        self.velocities = []      # Rate of change in performance
        self.momentum = 0.0       # mass * velocity
        self.kinetic_energy = 0.0 # 0.5 * mass * velocity^2

    def add_game(self, margin):
        self.margins.append(margin)
        self._update()

    def _update(self):
        """Recalculate momentum from recent performance."""
        if len(self.margins) < 3:
            return

        # Velocity = rate of change in rolling average margin
        window = min(self.velocity_window, len(self.margins))
        recent = np.array(self.margins[-window:], dtype=float)

        # First derivative: acceleration of performance
        if len(recent) >= 3:
            velocity = np.mean(np.diff(recent))  # Average change per game
        else:
            velocity = 0.0

        # Apply friction (regression toward mean)
        velocity *= (1 - self.friction)

        self.velocities.append(velocity)
        self.momentum = self.mass * velocity
        self.kinetic_energy = 0.5 * self.mass * velocity ** 2

    def get_impulse(self, window=None):
        """Impulse = sudden change in momentum (force * time).

        High impulse = team has suddenly shifted trajectory.
        """
        if window is None:
            window = self.impulse_window
        if len(self.velocities) < window + 1:
            return 0.0

        old_v = np.mean(self.velocities[-(window + 1):-1])
        new_v = self.velocities[-1]
        return self.mass * (new_v - old_v)

    def get_angular_momentum(self, window=10):
        """Angular momentum: detects cyclical patterns.

        Teams that oscillate (win-loss-win-loss) have high angular momentum.
        Teams with stable streaks have low angular momentum.
        """
        if len(self.margins) < window:
            return 0.0

        recent = np.array(self.margins[-window:], dtype=float)
        # Count sign changes (direction reversals)
        signs = np.sign(recent - np.mean(recent))
        reversals = np.sum(np.abs(np.diff(signs)) > 0)
        max_reversals = window - 1
        return float(reversals / max_reversals) if max_reversals > 0 else 0.0

    def get_jerk(self):
        """Jerk = rate of change of acceleration.

        High jerk = performance trajectory is changing rapidly.
        Unstable teams have high jerk.
        """
        if len(self.velocities) < 4:
            return 0.0

        accels = np.diff(self.velocities[-4:])
        return float(np.mean(np.diff(accels)))

    def get_features(self):
        return {
            "momentum": self.momentum,
            "kinetic_energy": self.kinetic_energy,
            "impulse": self.get_impulse(),
            "angular_momentum": self.get_angular_momentum(),
            "jerk": self.get_jerk(),
            "velocity": self.velocities[-1] if self.velocities else 0.0,
        }


class LeagueMomentum:
    """Manages momentum models for all teams in a league."""

    def __init__(self, friction=0.05, min_games=5, velocity_window=5, impulse_window=3, **kwargs):
        self.friction = friction
        self.min_games = min_games
        self._velocity_window = velocity_window
        self._impulse_window = impulse_window
        self.teams = defaultdict(lambda: TeamMomentum(
            friction=friction, min_games=min_games,
            velocity_window=self._velocity_window, impulse_window=self._impulse_window))
        # Track "mass" updates based on overall strength
        self._mass_history = defaultdict(list)

    def add_game(self, team, margin, elo_rating=None):
        """Add game result and optionally update team's 'mass' from Elo."""
        if elo_rating is not None:
            # Mass proportional to team strength (normalized Elo)
            mass = max(0.5, elo_rating / 1500)
            self.teams[team].mass = mass

        self.teams[team].add_game(margin)

    def get_features(self, home_team, away_team):
        """Get momentum features for a matchup."""
        h_feats = self.teams[home_team].get_features() if home_team in self.teams else {}
        a_feats = self.teams[away_team].get_features() if away_team in self.teams else {}

        h_mom = h_feats.get("momentum", 0)
        a_mom = a_feats.get("momentum", 0)
        h_ke = h_feats.get("kinetic_energy", 0)
        a_ke = a_feats.get("kinetic_energy", 0)
        h_imp = h_feats.get("impulse", 0)
        a_imp = a_feats.get("impulse", 0)
        h_ang = h_feats.get("angular_momentum", 0)
        a_ang = a_feats.get("angular_momentum", 0)

        return {
            "momentum_diff": h_mom - a_mom,
            "kinetic_energy_diff": h_ke - a_ke,
            "impulse_diff": h_imp - a_imp,
            "angular_momentum_diff": h_ang - a_ang,
            "combined_momentum": abs(h_mom) + abs(a_mom),
            "momentum_advantage": 1 if h_mom > a_mom else -1 if a_mom > h_mom else 0,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                margin = float(row[home_score_col]) - float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(home, margin)
            self.add_game(away, -margin)
        return len(self.teams)
