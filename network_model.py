"""Network/Graph model for sports predictions.

Uses PageRank, eigenvector centrality, and graph-based features
to capture transitive dominance in win-loss networks.

Package: networkx (pip install networkx)
"""

import logging
import math
from collections import defaultdict

import numpy as np
import networkx as nx


class LeagueNetwork:
    """Graph-based model of league competitive dynamics.

    Teams are nodes, games are weighted directed edges.
    Edge weight = number of wins (or margin-weighted).
    """

    def __init__(self, decay=0.95, margin_weighted=True, damping=0.85, **kwargs):
        """
        Args:
            decay: temporal decay factor (recent games weighted more)
            margin_weighted: weight edges by score margin (vs binary win/loss)
            damping: PageRank damping factor (default 0.85)
        """
        self.decay = decay
        self.margin_weighted = margin_weighted
        self.damping = damping
        self.graph = nx.DiGraph()
        self.n_games = 0
        self._pagerank = {}
        self._eigenvector = {}
        self._hub_scores = {}
        self._authority_scores = {}
        self._betweenness = {}

    def add_game(self, winner, loser, margin=1, game_idx=0):
        """Add a game result to the network.

        Creates/updates a directed edge from winner to loser.
        """
        weight = margin if self.margin_weighted else 1.0

        # Apply temporal decay
        if self.decay < 1.0 and self.n_games > 0:
            age = self.n_games - 1 - game_idx
            weight *= self.decay ** age

        if self.graph.has_edge(winner, loser):
            self.graph[winner][loser]["weight"] += weight
            self.graph[winner][loser]["games"] += 1
        else:
            self.graph.add_edge(winner, loser, weight=weight, games=1)

        # Ensure both teams are nodes
        if winner not in self.graph:
            self.graph.add_node(winner)
        if loser not in self.graph:
            self.graph.add_node(loser)

        self.n_games += 1

    def compute_centralities(self):
        """Compute all centrality measures.

        PageRank: like Google - beating strong teams matters more.
        Eigenvector: similar to PageRank but different normalization.
        HITS: hub (beats many) and authority (beaten by few) scores.
        Betweenness: how central a team is in the win-loss network.
        """
        if len(self.graph) < 3:
            return

        try:
            self._pagerank = nx.pagerank(self.graph, alpha=self.damping, weight="weight")
        except nx.PowerIterationFailedConvergence:
            self._pagerank = {n: 1.0 / len(self.graph) for n in self.graph.nodes()}

        try:
            self._eigenvector = nx.eigenvector_centrality_numpy(self.graph, weight="weight")
        except Exception:
            self._eigenvector = {n: 0.5 for n in self.graph.nodes()}

        try:
            hubs, authorities = nx.hits(self.graph, max_iter=100)
            self._hub_scores = hubs
            self._authority_scores = authorities
        except Exception:
            self._hub_scores = {n: 0.5 for n in self.graph.nodes()}
            self._authority_scores = {n: 0.5 for n in self.graph.nodes()}

        try:
            self._betweenness = nx.betweenness_centrality(self.graph, weight="weight")
        except Exception:
            self._betweenness = {n: 0.0 for n in self.graph.nodes()}

    def win_probability(self, home_team, away_team, home_advantage=0.03):
        """Compute win probability using PageRank differential.

        Higher PageRank = stronger team in the network.
        """
        if not self._pagerank:
            self.compute_centralities()

        home_pr = self._pagerank.get(home_team, 0.5 / max(1, len(self.graph)))
        away_pr = self._pagerank.get(away_team, 0.5 / max(1, len(self.graph)))

        total = home_pr + away_pr
        if total <= 0:
            return 0.5

        raw_prob = home_pr / total + home_advantage

        return max(0.01, min(0.99, raw_prob))

    def get_features(self, home_team, away_team):
        """Get graph-based features for a matchup.

        Returns dict of features for the meta-learner.
        """
        if not self._pagerank:
            self.compute_centralities()

        default_pr = 1.0 / max(1, len(self.graph))

        home_pr = self._pagerank.get(home_team, default_pr)
        away_pr = self._pagerank.get(away_team, default_pr)
        home_eig = self._eigenvector.get(home_team, 0.5)
        away_eig = self._eigenvector.get(away_team, 0.5)
        home_hub = self._hub_scores.get(home_team, 0.5)
        away_hub = self._hub_scores.get(away_team, 0.5)
        home_auth = self._authority_scores.get(home_team, 0.5)
        away_auth = self._authority_scores.get(away_team, 0.5)
        home_btw = self._betweenness.get(home_team, 0.0)
        away_btw = self._betweenness.get(away_team, 0.0)

        # Direct head-to-head record
        h2h_home = 0
        h2h_away = 0
        if self.graph.has_edge(home_team, away_team):
            h2h_home = self.graph[home_team][away_team].get("games", 0)
        if self.graph.has_edge(away_team, home_team):
            h2h_away = self.graph[away_team][home_team].get("games", 0)

        h2h_total = h2h_home + h2h_away
        h2h_advantage = (h2h_home - h2h_away) / max(1, h2h_total)

        # Common opponents analysis
        home_beaten = set(self.graph.successors(home_team)) if home_team in self.graph else set()
        away_beaten = set(self.graph.successors(away_team)) if away_team in self.graph else set()
        common = home_beaten & away_beaten
        common_advantage = 0.0
        if common:
            for opp in common:
                hw = self.graph[home_team][opp]["weight"] if self.graph.has_edge(home_team, opp) else 0
                aw = self.graph[away_team][opp]["weight"] if self.graph.has_edge(away_team, opp) else 0
                common_advantage += (hw - aw)
            common_advantage /= len(common)

        return {
            "pagerank_home": home_pr,
            "pagerank_away": away_pr,
            "pagerank_diff": home_pr - away_pr,
            "pagerank_prob": self.win_probability(home_team, away_team, 0),
            "eigenvector_diff": home_eig - away_eig,
            "hub_diff": home_hub - away_hub,
            "authority_diff": home_auth - away_auth,
            "betweenness_diff": home_btw - away_btw,
            "h2h_advantage": h2h_advantage,
            "common_opp_advantage": common_advantage,
            "n_common_opponents": len(common),
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        """Build the network from a games DataFrame."""
        for idx, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                h_score = float(row.get(home_score_col, 0))
                a_score = float(row.get(away_score_col, 0))
            except (ValueError, TypeError):
                continue

            margin = abs(h_score - a_score)
            if h_score > a_score:
                self.add_game(home, away, margin=max(1, margin), game_idx=idx)
            elif a_score > h_score:
                self.add_game(away, home, margin=max(1, margin), game_idx=idx)
            # Ties: no edge added

        self.compute_centralities()
        return len(self.graph.nodes())

    def show_rankings(self):
        """Print team rankings by PageRank."""
        if not self._pagerank:
            self.compute_centralities()

        if not self._pagerank:
            print("  No network data available.")
            return

        ranked = sorted(self._pagerank.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  {'Rank':>4} {'Team':<25} {'PageRank':>9} {'Eigenvec':>9} {'Hub':>7} {'Auth':>7}")
        print("  " + "-" * 65)

        for i, (team, pr) in enumerate(ranked, 1):
            eig = self._eigenvector.get(team, 0)
            hub = self._hub_scores.get(team, 0)
            auth = self._authority_scores.get(team, 0)
            print(f"  {i:>4} {team:<25} {pr:>9.4f} {eig:>9.4f} {hub:>7.4f} {auth:>7.4f}")


class StrengthOfSchedule:
    """Compute strength of schedule from the network."""

    def __init__(self, network):
        self.network = network

    def compute_sos(self, team):
        """Compute strength of schedule as average opponent PageRank."""
        if not self.network._pagerank:
            self.network.compute_centralities()

        opponents = set()
        if team in self.network.graph:
            opponents |= set(self.network.graph.successors(team))
            opponents |= set(self.network.graph.predecessors(team))

        if not opponents:
            return 0.5

        opp_ranks = [self.network._pagerank.get(opp, 0) for opp in opponents]
        return np.mean(opp_ranks) if opp_ranks else 0.5

    def get_sos_features(self, home_team, away_team):
        """Get SOS features for a matchup."""
        home_sos = self.compute_sos(home_team)
        away_sos = self.compute_sos(away_team)
        return {
            "home_sos_pagerank": home_sos,
            "away_sos_pagerank": away_sos,
            "sos_diff": home_sos - away_sos,
        }
