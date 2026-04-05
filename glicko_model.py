"""Glicko-2 Rating System for sports predictions.

Historical origin: Mark Glickman (1995, updated 2001).
Improvement over Elo that tracks UNCERTAINTY in ratings.

Key innovations:
  - Rating deviation (RD): how uncertain we are about a team's strength
  - Volatility (sigma): how erratic a team's performance is
  - RD increases during inactivity (uncertainty grows without data)
  - Strong results against strong opponents reduce RD faster

This gives us not just "who is better" but "how confident are we?"

Packages: numpy (built-in)
"""

import math
from collections import defaultdict

import numpy as np


# Glicko-2 constants
TAU = 0.5       # System constant (constrains volatility change)
EPSILON = 0.000001  # Convergence tolerance


def _g(phi):
    """Glicko-2 g function: reduces impact of games with uncertain opponents."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _E(mu, mu_j, phi_j):
    """Expected score against opponent j."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


class GlickoTeam:
    """Glicko-2 rating for a single team."""

    def __init__(self, rating=1500, rd=350, vol=0.06):
        # Glicko-1 scale
        self.rating = rating
        self.rd = rd
        self.vol = vol

        # Glicko-2 scale (internal)
        self.mu = (rating - 1500) / 173.7178
        self.phi = rd / 173.7178
        self.sigma = vol

    def to_glicko1(self):
        """Convert internal Glicko-2 to display scale."""
        self.rating = self.mu * 173.7178 + 1500
        self.rd = self.phi * 173.7178

    def pre_period(self):
        """Increase RD at start of rating period (uncertainty grows without games)."""
        self.phi = math.sqrt(self.phi**2 + self.sigma**2)

    def update(self, opponents_mu, opponents_phi, scores):
        """Update rating based on period results.

        opponents_mu: list of opponent mu values
        opponents_phi: list of opponent phi values
        scores: list of results (1=win, 0.5=draw, 0=loss)
        """
        if not opponents_mu:
            self.pre_period()
            self.to_glicko1()
            return

        # Step 3: Compute variance v
        v_inv = 0.0
        for mu_j, phi_j in zip(opponents_mu, opponents_phi):
            g_phi = _g(phi_j)
            e = _E(self.mu, mu_j, phi_j)
            v_inv += g_phi**2 * e * (1 - e)

        if v_inv < EPSILON:
            self.to_glicko1()
            return
        v = 1.0 / v_inv

        # Step 4: Compute delta
        delta = 0.0
        for mu_j, phi_j, s in zip(opponents_mu, opponents_phi, scores):
            g_phi = _g(phi_j)
            e = _E(self.mu, mu_j, phi_j)
            delta += g_phi * (s - e)
        delta *= v

        # Step 5: Compute new volatility (Illinois algorithm)
        a = math.log(self.sigma**2)
        A = a
        if delta**2 > self.phi**2 + v:
            B = math.log(delta**2 - self.phi**2 - v)
        else:
            k = 1
            while True:
                val = a - k * TAU
                f_val = (math.exp(val) * (delta**2 - self.phi**2 - v - math.exp(val))) / \
                        (2 * (self.phi**2 + v + math.exp(val))**2) - (val - a) / TAU**2
                if f_val < 0:
                    break
                k += 1
                if k > 100:
                    break
            B = a - k * TAU

        f_A = (math.exp(A) * (delta**2 - self.phi**2 - v - math.exp(A))) / \
              (2 * (self.phi**2 + v + math.exp(A))**2) - (A - a) / TAU**2
        f_B = (math.exp(B) * (delta**2 - self.phi**2 - v - math.exp(B))) / \
              (2 * (self.phi**2 + v + math.exp(B))**2) - (B - a) / TAU**2

        for _ in range(50):
            if abs(B - A) < EPSILON:
                break
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = (math.exp(C) * (delta**2 - self.phi**2 - v - math.exp(C))) / \
                  (2 * (self.phi**2 + v + math.exp(C))**2) - (C - a) / TAU**2

            if f_C * f_B <= 0:
                A, f_A = B, f_B
            else:
                f_A /= 2.0
            B, f_B = C, f_C

        new_sigma = math.exp(A / 2)
        self.sigma = new_sigma

        # Step 6: Update phi and mu
        phi_star = math.sqrt(self.phi**2 + new_sigma**2)
        self.phi = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        self.mu += self.phi**2 * delta / v

        self.to_glicko1()


class LeagueGlicko:
    """Manages Glicko-2 ratings for all teams in a league."""

    def __init__(self, initial_rating=1500, initial_rd=200, initial_vol=0.06, **kwargs):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        self.teams = {}
        self._period_games = defaultdict(list)  # team -> [(opp_mu, opp_phi, score)]
        self._game_count = 0
        self._period_size = 10  # Games per rating period

    def _get_or_create(self, team):
        if team not in self.teams:
            self.teams[team] = GlickoTeam(self.initial_rating, self.initial_rd, self.initial_vol)
        return self.teams[team]

    def add_game(self, home, away, h_score, a_score):
        """Add a game and update ratings."""
        h_team = self._get_or_create(home)
        a_team = self._get_or_create(away)

        score_h = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
        score_a = 1.0 - score_h

        # Accumulate games for the rating period
        self._period_games[home].append((a_team.mu, a_team.phi, score_h))
        self._period_games[away].append((h_team.mu, h_team.phi, score_a))

        self._game_count += 1

        # Process rating period
        if self._game_count % self._period_size == 0:
            self._process_period()

    def _process_period(self):
        """Update all teams that played during this period."""
        for team, games in self._period_games.items():
            if not games:
                continue
            t = self._get_or_create(team)
            opp_mu = [g[0] for g in games]
            opp_phi = [g[1] for g in games]
            scores = [g[2] for g in games]
            t.update(opp_mu, opp_phi, scores)

        self._period_games.clear()

    def win_prob(self, home, away):
        """Predict win probability using Glicko-2 ratings."""
        h = self._get_or_create(home)
        a = self._get_or_create(away)

        # Expected score incorporating both uncertainties
        combined_phi = math.sqrt(h.phi**2 + a.phi**2)
        prob = _E(h.mu, a.mu, combined_phi)
        return float(np.clip(prob, 0.01, 0.99))

    def get_features(self, home, away):
        """Get Glicko-2 features for a matchup."""
        h = self._get_or_create(home)
        a = self._get_or_create(away)

        prob = self.win_prob(home, away)

        return {
            "glicko_prob": prob,
            "glicko_rating_diff": h.rating - a.rating,
            "glicko_rd_diff": h.rd - a.rd,
            "glicko_combined_uncertainty": h.rd + a.rd,
            "glicko_vol_diff": h.vol - a.vol,
        }

    def regress_to_mean(self, factor=0.33):
        """Season boundary regression."""
        for team in self.teams.values():
            team.rating = team.rating * (1 - factor) + 1500 * factor
            team.mu = (team.rating - 1500) / 173.7178
            team.rd = min(350, team.rd * 1.5)  # Increase uncertainty
            team.phi = team.rd / 173.7178

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            try:
                self.add_game(row[home_col], row[away_col],
                              float(row[home_score_col]), float(row[away_score_col]))
            except (ValueError, TypeError, KeyError):
                continue
        self._process_period()  # Flush remaining
        return len(self.teams)
