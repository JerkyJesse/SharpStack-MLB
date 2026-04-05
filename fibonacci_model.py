"""Fibonacci Retracement model for sports predictions.

Historical origin: Leonardo of Pisa (Fibonacci), ~1202.
Applied in technical analysis since the 1930s.

Key Fibonacci ratios: 0.236, 0.382, 0.500, 0.618, 0.786
Applied to sports:
  - Track EMA-smoothed performance margins over time
  - Detect swing highs (peak performance) and swing lows (troughs)
  - Compute retracement levels between swings
  - Teams near 0.618 support after decline may bounce back
  - Teams near 0.618 resistance after rise may pull back

Packages: numpy (built-in)
"""

import math
from collections import defaultdict

import numpy as np


# Fibonacci ratios
FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]


class TeamFibonacci:
    """Fibonacci retracement tracking for a single team."""

    def __init__(self, min_games=10, ema_alpha=0.15, swing_window=10):
        self.min_games = min_games
        self.ema_alpha = ema_alpha
        self.swing_window = swing_window
        self.margins = []
        self.ema_values = []
        self.swing_high = None
        self.swing_low = None
        self.fib_levels = {}

    def add_game(self, margin):
        """Add a game result and update EMA + Fibonacci levels."""
        self.margins.append(margin)

        # Update EMA
        if not self.ema_values:
            self.ema_values.append(float(margin))
        else:
            prev = self.ema_values[-1]
            new_ema = self.ema_alpha * margin + (1 - self.ema_alpha) * prev
            self.ema_values.append(new_ema)

        # Detect swings once we have enough data
        if len(self.ema_values) >= self.swing_window:
            self._detect_swings()

    def _detect_swings(self):
        """Detect swing high and swing low from EMA values."""
        w = self.swing_window
        recent = self.ema_values[-w:]

        # Find local max and min in the window
        max_val = max(recent)
        min_val = min(recent)
        max_idx = len(self.ema_values) - w + recent.index(max_val)
        min_idx = len(self.ema_values) - w + recent.index(min_val)

        # Only update swings if there's meaningful amplitude
        amplitude = max_val - min_val
        if amplitude < 0.5:  # Minimum swing size
            return

        # Determine swing direction: most recent extreme defines the trend
        if max_idx > min_idx:
            # Upswing: low came first, high came after
            self.swing_low = min_val
            self.swing_high = max_val
        else:
            # Downswing: high came first, low came after
            self.swing_high = max_val
            self.swing_low = min_val

        # Compute Fibonacci retracement levels
        self.fib_levels = {}
        for ratio in FIB_RATIOS:
            self.fib_levels[ratio] = self.swing_low + ratio * (self.swing_high - self.swing_low)

    def get_features(self):
        """Get Fibonacci-based features."""
        if (len(self.ema_values) < self.min_games or
                self.swing_high is None or self.swing_low is None):
            return {
                "retrace_depth": 0.0,
                "support_dist": 0.0,
                "trend_strength": 0.0,
                "bounce_signal": 0.0,
            }

        current = self.ema_values[-1]
        swing_range = self.swing_high - self.swing_low
        if swing_range < 0.01:
            swing_range = 0.01

        # Retracement depth: where current sits relative to the swing (0=low, 1=high)
        retrace_depth = (current - self.swing_low) / swing_range
        retrace_depth = max(0.0, min(2.0, retrace_depth))  # Allow extensions beyond 1.0

        # Distance to nearest Fibonacci support level (below current)
        support_dist = 0.0
        for ratio in sorted(FIB_RATIOS, reverse=True):
            level = self.fib_levels[ratio]
            if level <= current:
                support_dist = (current - level) / swing_range
                break

        # Trend strength: derived from position in Fibonacci levels
        # Near 0.618 or above = strong trend continuation
        trend_strength = 0.0
        if retrace_depth >= 0.618:
            trend_strength = min(1.0, (retrace_depth - 0.618) / 0.382)
        elif retrace_depth <= 0.382:
            trend_strength = -min(1.0, (0.382 - retrace_depth) / 0.382)

        # Bounce signal: proximity to key Fibonacci levels
        # Strong signal when near 0.382 or 0.618 (common reversal points)
        bounce_signal = 0.0
        for key_level in [0.382, 0.618]:
            fib_val = self.fib_levels.get(key_level, current)
            dist = abs(current - fib_val) / swing_range if swing_range > 0 else 1.0
            if dist < 0.05:  # Within 5% of a key level
                # Positive bounce at support (0.618 during retracement)
                # Negative at resistance
                bounce_signal = 0.5 * (1 - dist / 0.05)

        return {
            "retrace_depth": float(retrace_depth),
            "support_dist": float(support_dist),
            "trend_strength": float(trend_strength),
            "bounce_signal": float(bounce_signal),
        }


class LeagueFibonacci:
    """League-wide Fibonacci retracement tracking."""

    def __init__(self, min_games=10, ema_alpha=0.15, swing_window=10,
                 fib_ema_alpha=None, fib_swing_window=None, **kwargs):
        # Allow mega_config hyperparams to override
        self.min_games = min_games
        self.ema_alpha = fib_ema_alpha if fib_ema_alpha is not None else ema_alpha
        self.swing_window = fib_swing_window if fib_swing_window is not None else swing_window
        self.teams = {}

    def _get_team(self, team):
        if team not in self.teams:
            self.teams[team] = TeamFibonacci(
                min_games=self.min_games,
                ema_alpha=self.ema_alpha,
                swing_window=self.swing_window,
            )
        return self.teams[team]

    def add_game(self, team, margin):
        """Add a game result for a team."""
        self._get_team(team).add_game(margin)

    def get_features(self, home_team, away_team):
        """Get differential Fibonacci features for a matchup."""
        h = self._get_team(home_team).get_features()
        a = self._get_team(away_team).get_features()

        return {
            "fib_retrace_depth_diff": h["retrace_depth"] - a["retrace_depth"],
            "fib_support_dist_diff": h["support_dist"] - a["support_dist"],
            "fib_trend_strength_diff": h["trend_strength"] - a["trend_strength"],
            "fib_bounce_signal": h["bounce_signal"] - a["bounce_signal"],
        }
