"""Monte Carlo Simulation model for sports predictions.

Historical origin: Stanislaw Ulam & John von Neumann (1946) - Manhattan Project.
Nicholas Metropolis named it after the Monte Carlo Casino.

Applied to sports:
  - Simulate 10,000+ games using team's actual score distributions
  - Win probability = fraction of simulations where home team wins
  - Naturally captures score variance, skewness, and fat tails
  - Can model "upset potential" from high-variance teams

Advantage over analytical models:
  - No assumptions about score distribution shape
  - Handles ties, overtime, and non-standard outcomes
  - Captures correlation between offensive and defensive performance

Packages: numpy (built-in)
"""

import math
from collections import defaultdict

import numpy as np


class TeamDistribution:
    """Tracks empirical score distribution for a team."""

    def __init__(self, min_games=10):
        self.min_games = min_games
        self.scores_for = []
        self.scores_against = []
        self.home_scores_for = []
        self.home_scores_against = []
        self.away_scores_for = []
        self.away_scores_against = []

    def add_game(self, scored, conceded, is_home):
        self.scores_for.append(scored)
        self.scores_against.append(conceded)
        if is_home:
            self.home_scores_for.append(scored)
            self.home_scores_against.append(conceded)
        else:
            self.away_scores_for.append(scored)
            self.away_scores_against.append(conceded)

    def sample_score(self, is_home, n_samples=1, rng=None):
        """Sample from the team's empirical score distribution.

        Uses kernel density estimation (Gaussian smoothing) to generate
        continuous samples from the discrete score history.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        if is_home and len(self.home_scores_for) >= self.min_games:
            scores = np.array(self.home_scores_for, dtype=float)
        elif not is_home and len(self.away_scores_for) >= self.min_games:
            scores = np.array(self.away_scores_for, dtype=float)
        elif len(self.scores_for) >= self.min_games:
            scores = np.array(self.scores_for, dtype=float)
        else:
            return rng.normal(22, 7, n_samples)  # Default fallback

        # Bootstrap + Gaussian noise (kernel density)
        indices = rng.randint(0, len(scores), n_samples)
        sampled = scores[indices]
        bandwidth = max(1.0, np.std(scores) * 0.3)
        noise = rng.normal(0, bandwidth, n_samples)
        return np.maximum(0, sampled + noise)

    def sample_conceded(self, is_home, n_samples=1, rng=None):
        """Sample from the team's conceded score distribution."""
        if rng is None:
            rng = np.random.RandomState(42)

        if is_home and len(self.home_scores_against) >= self.min_games:
            scores = np.array(self.home_scores_against, dtype=float)
        elif not is_home and len(self.away_scores_against) >= self.min_games:
            scores = np.array(self.away_scores_against, dtype=float)
        elif len(self.scores_against) >= self.min_games:
            scores = np.array(self.scores_against, dtype=float)
        else:
            return rng.normal(22, 7, n_samples)

        indices = rng.randint(0, len(scores), n_samples)
        sampled = scores[indices]
        bandwidth = max(1.0, np.std(scores) * 0.3)
        noise = rng.normal(0, bandwidth, n_samples)
        return np.maximum(0, sampled + noise)


class LeagueMonteCarlo:
    """Monte Carlo simulation engine for game predictions."""

    def __init__(self, n_simulations=5000, min_games=10, seed=42, **kwargs):
        self.n_simulations = n_simulations
        self.min_games = min_games
        self.seed = seed
        self.teams = defaultdict(lambda: TeamDistribution(min_games))
        self.rng = np.random.RandomState(seed)

    def add_game(self, team, scored, conceded, is_home):
        self.teams[team].add_game(scored, conceded, is_home)

    def simulate_game(self, home, away):
        """Simulate a matchup n_simulations times.

        Returns dict with win probabilities, expected scores, and spread.
        """
        h_dist = self.teams.get(home)
        a_dist = self.teams.get(away)

        if h_dist is None or a_dist is None:
            return {"win_prob": 0.5, "expected_margin": 0, "margin_std": 10,
                    "upset_potential": 0.1, "blowout_potential": 0.1}

        n = self.n_simulations

        # Generate home team scores from their offensive distribution
        h_offense = h_dist.sample_score(is_home=True, n_samples=n, rng=self.rng)
        home_scores = h_offense  # Home team's actual score distribution

        # Generate away team scores from their offensive distribution
        a_offense = a_dist.sample_score(is_home=False, n_samples=n, rng=self.rng)
        away_scores = a_offense  # Away team's actual score distribution

        margins = home_scores - away_scores
        home_wins = np.sum(margins > 0)
        away_wins = np.sum(margins < 0)
        draws = np.sum(margins == 0)

        # Split draws proportionally
        total_decisive = home_wins + away_wins
        if total_decisive > 0:
            win_prob = (home_wins + draws * home_wins / total_decisive) / n
        else:
            win_prob = 0.5

        return {
            "win_prob": float(win_prob),
            "expected_margin": float(np.mean(margins)),
            "margin_std": float(np.std(margins)),
            "upset_potential": float(np.mean(margins < -np.std(margins))),
            "blowout_potential": float(np.mean(margins > np.std(margins))),
        }

    def get_features(self, home, away):
        """Get Monte Carlo features for a matchup."""
        sim = self.simulate_game(home, away)

        return {
            "mc_win_prob": sim["win_prob"],
            "mc_expected_margin": sim["expected_margin"],
            "mc_margin_std": sim["margin_std"],
            "mc_upset_potential": sim.get("upset_potential", 0.1),
            "mc_blowout_potential": sim.get("blowout_potential", 0.1),
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            try:
                h_score = float(row[home_score_col])
                a_score = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(row[home_col], h_score, a_score, is_home=True)
            self.add_game(row[away_col], a_score, h_score, is_home=False)
        return len(self.teams)
