"""Information Theory model for sports predictions.

Historical origin: Claude Shannon (1948) - entropy as measure of uncertainty.
Applied to sports: Teams with high-entropy performance are less predictable.
Mutual information reveals which features actually predict outcomes.
KL divergence measures how much a team's recent form differs from their baseline.

Packages: numpy, scipy (built-in)
"""

import math
from collections import defaultdict

import numpy as np
from scipy import stats


class TeamInformation:
    """Information-theoretic analysis of a single team."""

    def __init__(self, min_games=10, n_bins=5):
        self.min_games = min_games
        self.n_bins = n_bins
        self.margins = []
        self.scores_for = []
        self.scores_against = []
        self.results = []  # 1=win, 0=loss

    def add_game(self, margin, score_for, score_against, won):
        self.margins.append(margin)
        self.scores_for.append(score_for)
        self.scores_against.append(score_against)
        self.results.append(won)

    def score_entropy(self, window=None):
        """Shannon entropy of the scoring distribution.

        Higher entropy = more varied scoring = less predictable.
        """
        scores = self.scores_for[-window:] if window else self.scores_for
        if len(scores) < self.min_games:
            return 0.5  # Neutral default

        # Bin scores and compute entropy
        hist, _ = np.histogram(scores, bins=self.n_bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(self.n_bins)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.5

    def margin_entropy(self, window=None):
        """Entropy of margin distribution."""
        margins = self.margins[-window:] if window else self.margins
        if len(margins) < self.min_games:
            return 0.5

        hist, _ = np.histogram(margins, bins=self.n_bins, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(self.n_bins)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.5

    def kl_divergence_recent_vs_season(self, window=10):
        """KL divergence between recent margins and full-season margins.

        High KL = team's recent form is very different from season norm.
        Could signal momentum shift or regression-to-mean opportunity.
        """
        if len(self.margins) < self.min_games + window:
            return 0.0

        full = np.array(self.margins, dtype=float)
        recent = np.array(self.margins[-window:], dtype=float)

        # Build distributions
        bins = np.linspace(min(full.min(), recent.min()) - 1,
                           max(full.max(), recent.max()) + 1, self.n_bins + 1)

        p_full, _ = np.histogram(full, bins=bins, density=True)
        p_recent, _ = np.histogram(recent, bins=bins, density=True)

        # Add smoothing
        p_full = p_full + 1e-10
        p_recent = p_recent + 1e-10
        p_full = p_full / p_full.sum()
        p_recent = p_recent / p_recent.sum()

        # KL(recent || season)
        kl = float(np.sum(p_recent * np.log(p_recent / p_full)))
        return max(0, kl)

    def conditional_entropy(self, window=None):
        """Conditional entropy H(result | previous_result).

        Low = results are predictable from previous result (momentum/anti-momentum).
        High = results are independent of previous result (random walk).
        """
        results = self.results[-window:] if window else self.results
        if len(results) < self.min_games:
            return 1.0

        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            transitions[prev][curr] += 1

        # H(current | previous)
        total = len(results) - 1
        h_cond = 0
        for prev_state in [0, 1]:
            prev_count = sum(transitions[prev_state].values())
            if prev_count == 0:
                continue
            p_prev = prev_count / total
            for curr_state in [0, 1]:
                p_joint = transitions[prev_state][curr_state] / total
                if p_joint > 0:
                    p_cond = transitions[prev_state][curr_state] / prev_count
                    h_cond -= p_joint * math.log2(p_cond)

        return float(h_cond)

    def get_features(self, window=None):
        """Get all information-theoretic features."""
        return {
            "score_entropy": self.score_entropy(window),
            "margin_entropy": self.margin_entropy(window),
            "kl_divergence": self.kl_divergence_recent_vs_season(),
            "conditional_entropy": self.conditional_entropy(window),
        }


class LeagueInformationTheory:
    """Information theory analysis for all teams in a league."""

    def __init__(self, min_games=10):
        self.min_games = min_games
        self.teams = defaultdict(lambda: TeamInformation(min_games))

    def add_game(self, team, margin, score_for, score_against, won):
        self.teams[team].add_game(margin, score_for, score_against, won)

    def get_features(self, home_team, away_team, window=None):
        """Get information theory features for a matchup."""
        h_feats = self.teams[home_team].get_features(window) if home_team in self.teams else {}
        a_feats = self.teams[away_team].get_features(window) if away_team in self.teams else {}

        h_score_ent = h_feats.get("score_entropy", 0.5)
        a_score_ent = a_feats.get("score_entropy", 0.5)
        h_margin_ent = h_feats.get("margin_entropy", 0.5)
        a_margin_ent = a_feats.get("margin_entropy", 0.5)
        h_kl = h_feats.get("kl_divergence", 0)
        a_kl = a_feats.get("kl_divergence", 0)
        h_cond = h_feats.get("conditional_entropy", 1.0)
        a_cond = a_feats.get("conditional_entropy", 1.0)

        return {
            "score_entropy_diff": h_score_ent - a_score_ent,
            "margin_entropy_diff": h_margin_ent - a_margin_ent,
            "kl_divergence_diff": h_kl - a_kl,
            "combined_kl": h_kl + a_kl,
            "conditional_entropy_diff": h_cond - a_cond,
            "predictability_diff": (1 - h_score_ent) - (1 - a_score_ent),
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
            margin = h_score - a_score
            self.add_game(home, margin, h_score, a_score, h_score > a_score)
            self.add_game(away, -margin, a_score, h_score, a_score > h_score)
        return len(self.teams)
