"""Game Theory model for sports predictions.

Historical origin: John von Neumann & Oskar Morgenstern (1944),
John Nash (1950) - Nash equilibrium.

Applied to sports:
  - Rock-paper-scissors dynamics: some playstyles counter others
  - Minimax: optimal strategy against a rational opponent
  - Repeated games: track how teams adapt to opponents over time
  - Payoff matrices from historical head-to-head records

Packages: numpy (built-in)
"""

import math
from collections import defaultdict

import numpy as np


class HeadToHeadTracker:
    """Tracks head-to-head records and adaptation patterns."""

    def __init__(self, min_matchups=2):
        self.min_matchups = min_matchups
        # h2h_record[(team_a, team_b)] = list of (margin, date_idx)
        self.h2h_record = defaultdict(list)
        # adaptation[team] = how much they improve in rematches
        self.adaptation_scores = defaultdict(list)

    def add_game(self, home, away, margin, game_idx=0):
        key = (home, away)
        self.h2h_record[key].append((margin, game_idx))

        # Track adaptation: did team improve in rematches?
        rev_key = (away, home)
        prev_as_away = self.h2h_record.get(rev_key, [])
        if prev_as_away:
            # This team previously played as away against this opponent
            last_margin = prev_as_away[-1][0]
            improvement = margin - (-last_margin)  # Compare margins (flip sign)
            self.adaptation_scores[home].append(improvement)

    def get_h2h_advantage(self, home, away):
        """Head-to-head win rate for home team against away team."""
        key = (home, away)
        rev_key = (away, home)

        # All games between these teams (in either venue)
        home_margins = [m for m, _ in self.h2h_record.get(key, [])]
        away_margins = [-m for m, _ in self.h2h_record.get(rev_key, [])]
        all_margins = home_margins + away_margins

        if len(all_margins) < self.min_matchups:
            return 0.0, 0  # No advantage, no data

        wins = sum(1 for m in all_margins if m > 0)
        return wins / len(all_margins) - 0.5, len(all_margins)

    def get_adaptation_rate(self, team):
        """How much does a team improve in rematches?

        Positive = team gets better against opponents they've faced before.
        """
        scores = self.adaptation_scores.get(team, [])
        if len(scores) < 2:
            return 0.0
        return float(np.mean(scores))


class StyleMatchup:
    """Models rock-paper-scissors dynamics between team styles.

    Team archetypes:
    - Offensive powerhouse (high scoring, poor defense)
    - Defensive fortress (low scoring, great defense)
    - Balanced (moderate in both)
    - Volatile (inconsistent, high variance)

    Certain styles have natural advantages over others.
    """

    def __init__(self, min_games=10):
        self.min_games = min_games
        self.team_styles = {}
        self.style_matchup_matrix = np.ones((4, 4)) * 0.5  # 4 styles

    def classify_style(self, ppg, papg, ppg_std, papg_std, league_ppg=None):
        """Classify a team's playing style."""
        if league_ppg is None:
            league_ppg = 22  # Default

        off_rating = ppg / max(league_ppg, 1)
        def_rating = papg / max(league_ppg, 1)

        if off_rating > 1.1 and def_rating > 1.0:
            return 0  # Offensive powerhouse (scores a lot, gives up a lot)
        elif off_rating < 0.95 and def_rating < 0.95:
            return 1  # Defensive fortress (doesn't score much, doesn't give up much)
        elif ppg_std > league_ppg * 0.4 or papg_std > league_ppg * 0.4:
            return 3  # Volatile
        else:
            return 2  # Balanced

    def update_team_style(self, team, scores_for, scores_against, league_ppg=None):
        """Update a team's style classification."""
        if len(scores_for) < self.min_games:
            return

        ppg = np.mean(scores_for[-15:])
        papg = np.mean(scores_against[-15:])
        ppg_std = np.std(scores_for[-15:])
        papg_std = np.std(scores_against[-15:])

        self.team_styles[team] = self.classify_style(ppg, papg, ppg_std, papg_std, league_ppg)

    def update_matchup_matrix(self, home_style, away_style, home_won):
        """Update the style matchup matrix with a game result."""
        # Exponential moving average update
        alpha = 0.05
        current = self.style_matchup_matrix[home_style, away_style]
        self.style_matchup_matrix[home_style, away_style] = (1 - alpha) * current + alpha * home_won

    def get_style_advantage(self, home_team, away_team):
        """Get the style-based matchup advantage."""
        h_style = self.team_styles.get(home_team)
        a_style = self.team_styles.get(away_team)

        if h_style is None or a_style is None:
            return 0.0

        return float(self.style_matchup_matrix[h_style, a_style] - 0.5)


class LeagueGameTheory:
    """Complete game theory model for a league."""

    def __init__(self, min_matchups=2, min_games=10):
        self.h2h = HeadToHeadTracker(min_matchups)
        self.style = StyleMatchup(min_games)
        self.teams_scores_for = defaultdict(list)
        self.teams_scores_against = defaultdict(list)

    def add_game(self, home, away, h_score, a_score, game_idx=0):
        margin = h_score - a_score
        self.h2h.add_game(home, away, margin, game_idx)

        self.teams_scores_for[home].append(h_score)
        self.teams_scores_for[away].append(a_score)
        self.teams_scores_against[home].append(a_score)
        self.teams_scores_against[away].append(h_score)

        # Update styles
        self.style.update_team_style(home, self.teams_scores_for[home],
                                     self.teams_scores_against[home])
        self.style.update_team_style(away, self.teams_scores_for[away],
                                     self.teams_scores_against[away])

        # Update matchup matrix
        h_style = self.style.team_styles.get(home)
        a_style = self.style.team_styles.get(away)
        if h_style is not None and a_style is not None:
            self.style.update_matchup_matrix(h_style, a_style, int(h_score > a_score))

    def get_features(self, home_team, away_team):
        """Get game theory features for a matchup."""
        h2h_adv, h2h_n = self.h2h.get_h2h_advantage(home_team, away_team)
        h_adapt = self.h2h.get_adaptation_rate(home_team)
        a_adapt = self.h2h.get_adaptation_rate(away_team)
        style_adv = self.style.get_style_advantage(home_team, away_team)

        return {
            "h2h_advantage": h2h_adv,
            "h2h_n_games": h2h_n,
            "adaptation_diff": h_adapt - a_adapt,
            "style_advantage": style_adv,
            "rematch_factor": 1 if h2h_n >= 2 else 0,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for idx, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                h_score = float(row[home_score_col])
                a_score = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(home, away, h_score, a_score, game_idx=idx)
        return len(self.teams_scores_for)
