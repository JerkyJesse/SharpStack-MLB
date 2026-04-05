"""Extreme Value Theory (EVT) model for sports predictions.

Historical origin: Fisher & Tippett (1928), Gnedenko (1943).
Models the tails of distributions -- extreme outcomes.

Applied to sports:
  - Fit Generalized Pareto Distribution (GPD) to margin exceedances
  - Shape parameter xi indicates tail heaviness:
    xi > 0: heavy tail (extreme outcomes more likely)
    xi = 0: exponential tail
    xi < 0: bounded tail (extreme outcomes less likely)
  - Predict blowout probability, upset risk
  - Teams with heavy tails are more volatile in extreme situations

Packages: numpy, scipy.stats (built-in)
"""

import math
import logging
from collections import defaultdict

import numpy as np


class TeamEVT:
    """Extreme Value Theory tracking for a single team."""

    def __init__(self, min_games=20, threshold_quantile=0.90,
                 min_exceedances=5):
        self.min_games = min_games
        self.threshold_quantile = threshold_quantile
        self.min_exceedances = min_exceedances
        self.margins = []
        self.threshold = None
        self.shape = 0.0    # GPD shape parameter (xi)
        self.scale = 1.0    # GPD scale parameter (sigma)
        self._fitted = False

    def add_game(self, margin):
        """Add a game margin result."""
        self.margins.append(float(margin))

    def fit(self):
        """Fit GPD to tail exceedances above threshold."""
        if len(self.margins) < self.min_games:
            return

        abs_margins = np.abs(self.margins)

        # Set threshold at quantile of absolute margins
        self.threshold = float(np.quantile(abs_margins, self.threshold_quantile))
        if self.threshold < 0.5:
            self.threshold = 0.5

        # Get exceedances (values above threshold)
        exceedances = abs_margins[abs_margins > self.threshold] - self.threshold

        if len(exceedances) < self.min_exceedances:
            return

        # Fit GPD using method of moments (robust, no scipy dependency)
        # For GPD: mean = sigma/(1-xi), var = sigma^2/((1-xi)^2*(1-2*xi))
        # Method of moments estimators:
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)

        if mean_exc <= 0 or var_exc <= 0:
            self.shape = 0.0
            self.scale = max(0.1, mean_exc)
            self._fitted = True
            return

        # From method of moments:
        # xi = 0.5 * (1 - mean^2/var)
        # sigma = mean * (1 - xi)
        ratio = mean_exc ** 2 / var_exc
        self.shape = 0.5 * (1 - ratio)

        # Bound shape parameter to reasonable range
        self.shape = max(-0.5, min(0.5, self.shape))

        self.scale = mean_exc * (1 - self.shape)
        self.scale = max(0.1, self.scale)

        self._fitted = True

        # Try scipy for better fit if available
        try:
            from scipy.stats import genpareto
            xi, _, sigma = genpareto.fit(exceedances, floc=0)
            if -0.5 <= xi <= 0.5 and sigma > 0:
                self.shape = xi
                self.scale = sigma
        except Exception:
            pass  # Keep method-of-moments estimate

    def get_features(self):
        """Get EVT-based features."""
        if not self._fitted or len(self.margins) < self.min_games:
            return {
                "tail_index": 0.0,
                "blowout_prob": 0.0,
                "tail_heaviness": 0.0,
                "extreme_ratio": 0.0,
            }

        # Tail index (shape parameter): positive = heavy tail
        tail_index = float(self.shape)

        # Blowout probability: P(|margin| > 2 * threshold)
        # Using GPD survival function: P(X > x) = (1 + xi*x/sigma)^(-1/xi)
        x = self.threshold  # Exceedance of 1x threshold above threshold
        blowout_prob = 0.0
        if self.shape != 0 and self.scale > 0:
            arg = 1 + self.shape * x / self.scale
            if arg > 0:
                blowout_prob = float(arg ** (-1.0 / self.shape))
                blowout_prob = max(0.0, min(1.0, blowout_prob))
        elif self.scale > 0:
            # Exponential case (xi = 0)
            blowout_prob = float(math.exp(-x / self.scale))

        # Tail heaviness: normalized shape parameter
        # Positive = heavier than exponential, negative = lighter
        tail_heaviness = min(1.0, max(-1.0, tail_index * 3))

        # Extreme ratio: fraction of games with extreme margins
        abs_margins = np.abs(self.margins[-50:]) if len(self.margins) > 50 else np.abs(self.margins)
        extreme_ratio = float(np.mean(abs_margins > self.threshold))

        return {
            "tail_index": tail_index,
            "blowout_prob": blowout_prob,
            "tail_heaviness": tail_heaviness,
            "extreme_ratio": extreme_ratio,
        }


class LeagueEVT:
    """League-wide Extreme Value Theory tracking."""

    def __init__(self, min_games=20, threshold_quantile=0.90,
                 min_exceedances=5,
                 evt_threshold_quantile=None, evt_min_exceedances=None,
                 **kwargs):
        self.min_games = min_games
        self.threshold_quantile = (evt_threshold_quantile
                                   if evt_threshold_quantile is not None
                                   else threshold_quantile)
        self.min_exceedances = (evt_min_exceedances
                                if evt_min_exceedances is not None
                                else min_exceedances)
        self.teams = {}

    def _get_team(self, team):
        if team not in self.teams:
            self.teams[team] = TeamEVT(
                min_games=self.min_games,
                threshold_quantile=self.threshold_quantile,
                min_exceedances=self.min_exceedances,
            )
        return self.teams[team]

    def add_game(self, team, margin):
        """Add a game result for a team."""
        self._get_team(team).add_game(margin)

    def fit_all(self):
        """Refit GPD for all teams."""
        for team in self.teams.values():
            try:
                team.fit()
            except Exception as e:
                logging.debug("EVT fit failed for a team: %s", e)

    def get_features(self, home_team, away_team):
        """Get differential EVT features for a matchup."""
        h = self._get_team(home_team).get_features()
        a = self._get_team(away_team).get_features()

        # Upset risk: away team's blowout probability advantage
        # (heavier tail + higher blowout prob = more upset potential)
        upset_risk = a["blowout_prob"] - h["blowout_prob"]

        return {
            "evt_tail_index_diff": h["tail_index"] - a["tail_index"],
            "evt_blowout_prob_diff": h["blowout_prob"] - a["blowout_prob"],
            "evt_upset_risk": float(upset_risk),
            "evt_tail_heaviness_diff": h["tail_heaviness"] - a["tail_heaviness"],
        }
