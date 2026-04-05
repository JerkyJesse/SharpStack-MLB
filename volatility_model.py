"""Volatility & Chaos Theory models for sports predictions.

GARCH: Models time-varying volatility in team performance.
High-volatility teams are less predictable — widen confidence intervals.
Lyapunov exponents: Measure chaos in team performance time series.

Packages: arch (pip install arch), nolds (pip install nolds)
"""

import logging
import math
from collections import defaultdict

import numpy as np


class TeamVolatility:
    """GARCH(1,1) volatility model for a single team."""

    def __init__(self, min_games=15, alpha=0.10, beta=0.80):
        self.min_games = min_games
        self.alpha = alpha
        self.beta = beta
        self.margins = []
        self.current_vol = None
        self.long_run_vol = None
        self._fitted = False

    def add_game(self, margin):
        self.margins.append(margin)

    def fit(self):
        """Fit GARCH(1,1) model on team's margin series.

        Uses a simplified GARCH without the arch package for robustness.
        GARCH(1,1): sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
        """
        if len(self.margins) < self.min_games:
            return False

        margins = np.array(self.margins, dtype=float)
        mean_margin = np.mean(margins)
        residuals = margins - mean_margin

        # Simplified GARCH(1,1) estimation
        # Typical sports values: alpha~0.1, beta~0.8, omega from unconditional var
        alpha = self.alpha
        beta = self.beta
        omega = np.var(residuals) * (1 - alpha - beta)

        # Forward filter
        sigma2 = np.var(residuals)  # Initial variance
        vol_series = []

        for e in residuals:
            sigma2 = omega + alpha * e**2 + beta * sigma2
            vol_series.append(math.sqrt(max(sigma2, 0.01)))

        self.current_vol = vol_series[-1]
        self.long_run_vol = math.sqrt(omega / max(1 - alpha - beta, 0.01))
        self._fitted = True
        return True

    def get_volatility_ratio(self):
        """Current volatility relative to long-run average.

        > 1.0: team is more volatile than usual (unpredictable)
        < 1.0: team is more stable than usual (predictable)
        """
        if not self._fitted:
            return 1.0
        if self.long_run_vol <= 0:
            return 1.0
        return self.current_vol / self.long_run_vol


class LeagueVolatility:
    """Manages volatility models for all teams."""

    def __init__(self, min_games=15, alpha=0.10, beta=0.80, **kwargs):
        self.min_games = min_games
        self._alpha = alpha
        self._beta = beta
        self.teams = defaultdict(lambda: TeamVolatility(min_games, alpha=self._alpha, beta=self._beta))

    def add_game(self, team, margin):
        self.teams[team].add_game(margin)

    def fit_all(self):
        fitted = 0
        for team, vol in self.teams.items():
            if vol.fit():
                fitted += 1
        return fitted

    def get_features(self, home_team, away_team):
        """Get volatility features for a matchup."""
        home_vol = self.teams.get(home_team)
        away_vol = self.teams.get(away_team)

        h_ratio = home_vol.get_volatility_ratio() if home_vol and home_vol._fitted else 1.0
        a_ratio = away_vol.get_volatility_ratio() if away_vol and away_vol._fitted else 1.0
        h_vol = home_vol.current_vol if home_vol and home_vol._fitted else 10.0
        a_vol = away_vol.current_vol if away_vol and away_vol._fitted else 10.0

        return {
            "home_volatility": h_vol,
            "away_volatility": a_vol,
            "volatility_ratio_diff": h_ratio - a_ratio,
            "combined_volatility": h_vol + a_vol,
            "unpredictability": max(h_ratio, a_ratio),
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
        return self.fit_all()


def compute_lyapunov_exponent(margins, emb_dim=3, lag=1):
    """Compute largest Lyapunov exponent for a team's margin series.

    Positive Lyapunov = chaotic (inherently unpredictable).
    Near zero = periodic or quasi-periodic.
    Negative = converging (stable/predictable).

    Uses Rosenstein's algorithm (simplified).
    """
    if len(margins) < 20:
        return 0.0

    try:
        import nolds
        le = nolds.lyap_r(np.array(margins, dtype=float), emb_dim=emb_dim, lag=lag)
        return float(le) if not np.isnan(le) else 0.0
    except ImportError:
        # Simplified estimation without nolds
        x = np.array(margins, dtype=float)
        n = len(x)
        if n < 10:
            return 0.0

        # Basic divergence rate estimation
        divergences = []
        for i in range(n - 2):
            for j in range(i + 1, min(i + 5, n - 1)):
                d0 = abs(x[i] - x[j])
                if d0 > 0.1:
                    d1 = abs(x[i + 1] - x[j + 1]) if j + 1 < n and i + 1 < n else d0
                    if d1 > 0:
                        divergences.append(math.log(d1 / d0))

        return float(np.mean(divergences)) if divergences else 0.0
    except Exception:
        return 0.0


def compute_hurst_exponent(margins):
    """Compute Hurst exponent of a team's margin series.

    H > 0.5: trending (momentum)
    H = 0.5: random walk
    H < 0.5: mean-reverting (regression to mean)
    """
    if len(margins) < 20:
        return 0.5

    try:
        import nolds
        h = nolds.hurst_rs(np.array(margins, dtype=float))
        return float(h) if not np.isnan(h) else 0.5
    except ImportError:
        # R/S analysis simplified
        x = np.array(margins, dtype=float)
        n = len(x)
        max_k = min(n // 2, 50)
        if max_k < 4:
            return 0.5

        rs_values = []
        ns = []
        for k in range(4, max_k + 1):
            rs_list = []
            for start in range(0, n - k + 1, k):
                chunk = x[start:start + k]
                mean_c = np.mean(chunk)
                devs = np.cumsum(chunk - mean_c)
                r = np.max(devs) - np.min(devs)
                s = np.std(chunk)
                if s > 0:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append(np.mean(rs_list))
                ns.append(k)

        if len(rs_values) < 3:
            return 0.5

        log_n = np.log(ns)
        log_rs = np.log(rs_values)
        slope, _ = np.polyfit(log_n, log_rs, 1)
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5
