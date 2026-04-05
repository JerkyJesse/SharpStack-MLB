"""Markov Chain model for sports predictions.

Historical origin: Andrey Markov (1906) - stochastic processes.
Applied to sports:
  - Transition matrix: P(next_state | current_state)
  - States: win_big, win_close, loss_close, loss_big
  - Steady-state distribution reveals long-term expected performance
  - Absorbing states analysis: probability of reaching certain outcomes

Packages: numpy (built-in)
"""

import math
from collections import defaultdict

import numpy as np


# Performance states based on margin
STATES = ["big_loss", "close_loss", "close_win", "big_win"]
N_STATES = len(STATES)


def _margin_to_state(margin, sport="nfl"):
    """Convert a game margin to a discrete state."""
    # Sport-specific thresholds for "close" vs "big"
    thresholds = {
        "nfl": 7,    # One score (TD + PAT)
        "nba": 8,    # One possession + free throws
        "mlb": 3,    # Typical close game
        "nhl": 2,    # One goal margin
    }
    threshold = thresholds.get(sport, 7)

    if margin <= -threshold:
        return 0   # big_loss
    elif margin < 0:
        return 1   # close_loss
    elif margin < threshold:
        return 2   # close_win
    else:
        return 3   # big_win


class TeamMarkovChain:
    """Markov chain model of a team's performance transitions."""

    def __init__(self, sport="nfl", min_games=8):
        self.sport = sport
        self.min_games = min_games
        self.transitions = np.zeros((N_STATES, N_STATES))
        self.state_counts = np.zeros(N_STATES)
        self.states = []
        self._transition_matrix = None
        self._steady_state = None

    def add_game(self, margin):
        state = _margin_to_state(margin, self.sport)
        self.state_counts[state] += 1

        if self.states:
            prev = self.states[-1]
            self.transitions[prev, state] += 1

        self.states.append(state)

    def compute_transition_matrix(self):
        """Normalize transitions into probabilities."""
        if len(self.states) < self.min_games:
            return None

        T = self.transitions.copy()
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self._transition_matrix = T / row_sums
        return self._transition_matrix

    def compute_steady_state(self):
        """Find steady-state distribution (long-run probability of each state).

        Solves pi * T = pi, sum(pi) = 1
        """
        T = self.compute_transition_matrix()
        if T is None:
            return np.array([0.25, 0.25, 0.25, 0.25])

        # Power iteration method
        pi = np.ones(N_STATES) / N_STATES
        for _ in range(100):
            pi_new = pi @ T
            if np.allclose(pi, pi_new, atol=1e-8):
                break
            pi = pi_new

        self._steady_state = pi / pi.sum()
        return self._steady_state

    def next_state_probs(self):
        """Probability distribution over next game's state."""
        T = self.compute_transition_matrix()
        if T is None or not self.states:
            return np.array([0.25, 0.25, 0.25, 0.25])

        current = self.states[-1]
        return T[current]

    def win_probability(self):
        """P(win) based on current state transition probabilities."""
        probs = self.next_state_probs()
        # close_win + big_win
        return float(probs[2] + probs[3])

    def momentum_metric(self):
        """Ratio of actual recent wins to steady-state expectation.

        > 1: team is over-performing its long-run average (hot)
        < 1: team is under-performing (cold)
        """
        steady = self.compute_steady_state()
        steady_win_rate = steady[2] + steady[3]
        if steady_win_rate < 0.01:
            return 1.0

        recent_n = min(10, len(self.states))
        if recent_n < 3:
            return 1.0

        recent_wins = sum(1 for s in self.states[-recent_n:] if s >= 2)
        recent_rate = recent_wins / recent_n
        return float(recent_rate / steady_win_rate)

    def get_features(self):
        probs = self.next_state_probs()
        steady = self.compute_steady_state()

        return {
            "markov_win_prob": float(probs[2] + probs[3]),
            "markov_big_win_prob": float(probs[3]),
            "markov_big_loss_prob": float(probs[0]),
            "markov_momentum": self.momentum_metric(),
            "steady_win_rate": float(steady[2] + steady[3]),
            "current_state": self.states[-1] if self.states else 2,
        }


class LeagueMarkovChain:
    """Manages Markov chain models for all teams."""

    def __init__(self, sport="nfl", min_games=8):
        self.sport = sport
        self.min_games = min_games
        self.teams = defaultdict(lambda: TeamMarkovChain(sport, min_games))

    def add_game(self, team, margin):
        self.teams[team].add_game(margin)

    def get_features(self, home_team, away_team):
        """Get Markov chain features for a matchup."""
        h_feats = self.teams[home_team].get_features() if home_team in self.teams else {}
        a_feats = self.teams[away_team].get_features() if away_team in self.teams else {}

        h_wp = h_feats.get("markov_win_prob", 0.5)
        a_wp = a_feats.get("markov_win_prob", 0.5)
        h_mom = h_feats.get("markov_momentum", 1.0)
        a_mom = a_feats.get("markov_momentum", 1.0)
        h_big_win = h_feats.get("markov_big_win_prob", 0.25)
        a_big_win = a_feats.get("markov_big_win_prob", 0.25)
        h_big_loss = h_feats.get("markov_big_loss_prob", 0.25)
        a_big_loss = a_feats.get("markov_big_loss_prob", 0.25)

        # Convert to matchup probability
        if h_wp + (1 - a_wp) > 0:
            matchup_prob = h_wp / (h_wp + (1 - a_wp))
        else:
            matchup_prob = 0.5

        return {
            "markov_matchup_prob": matchup_prob,
            "markov_win_prob_diff": h_wp - a_wp,
            "markov_momentum_diff": h_mom - a_mom,
            "markov_dominance_diff": h_big_win - a_big_win,
            "markov_vulnerability_diff": a_big_loss - h_big_loss,
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
