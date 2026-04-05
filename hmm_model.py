"""Hidden Markov Model for team state detection.

Shared module across sports. Detects hidden team states (hot, cold, normal, transition)
from observable game results using the Baum-Welch algorithm.

Package: hmmlearn (pip install hmmlearn)
"""

import logging
import math
from collections import defaultdict

import numpy as np

# Number of hidden states
N_STATES = 4  # hot, cold, normal, transition
STATE_NAMES = ["HOT", "COLD", "NORMAL", "TRANSITION"]


class TeamHMM:
    """Hidden Markov Model for a single team's state trajectory."""

    def __init__(self, n_states=N_STATES, min_games=10, **kwargs):
        self.n_states = n_states
        self.min_games = min_games
        self.model = None
        self._fitted = False

    def _build_observations(self, margins, is_home_list=None):
        """Convert game margins into observation matrix for HMM.

        Observations: [normalized_margin, win_flag, margin_trend]
        """
        if len(margins) < self.min_games:
            return None

        obs = []
        for i, m in enumerate(margins):
            # Normalize margin to roughly [-3, 3] range
            norm_margin = np.clip(m / 10.0, -3, 3)
            win = 1.0 if m > 0 else 0.0

            # 3-game rolling trend
            if i >= 3:
                recent = margins[i-3:i+1]
                trend = (np.mean(recent) - np.mean(margins[max(0, i-10):i+1])) / 10.0
            else:
                trend = 0.0

            obs.append([norm_margin, win, trend])

        return np.array(obs)

    def fit(self, margins, is_home_list=None):
        """Fit HMM on a team's historical game margins.

        Args:
            margins: list of score margins (positive = win)
            is_home_list: optional list of booleans (True if home game)
        """
        from hmmlearn.hmm import GaussianHMM

        obs = self._build_observations(margins, is_home_list)
        if obs is None:
            return False

        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
                init_params="stmc",
            )
            self.model.fit(obs)
            self._fitted = True
            return True
        except Exception as e:
            logging.debug("HMM fit failed: %s", e)
            self._fitted = False
            return False

    def predict_state(self, margins):
        """Predict current hidden state from recent margins.

        Returns dict: {state_name, state_probs, is_hot, is_cold, confidence}
        """
        if not self._fitted or self.model is None:
            return {"state_name": "UNKNOWN", "state_probs": [], "is_hot": False,
                    "is_cold": False, "confidence": 0}

        obs = self._build_observations(margins)
        if obs is None:
            return {"state_name": "UNKNOWN", "state_probs": [], "is_hot": False,
                    "is_cold": False, "confidence": 0}

        try:
            # Get state sequence and probabilities
            log_prob, state_seq = self.model.decode(obs, algorithm="viterbi")
            state_probs = self.model.predict_proba(obs)

            # Current state (last observation)
            current_state = state_seq[-1]
            current_probs = state_probs[-1]

            # Identify which state is "hot" and "cold" by mean emission
            state_means = self.model.means_[:, 0]  # First feature: normalized margin
            hot_state = int(np.argmax(state_means))
            cold_state = int(np.argmin(state_means))

            return {
                "state_idx": int(current_state),
                "state_name": STATE_NAMES[current_state] if current_state < len(STATE_NAMES) else f"STATE_{current_state}",
                "state_probs": current_probs.tolist(),
                "is_hot": current_state == hot_state,
                "is_cold": current_state == cold_state,
                "hot_prob": float(current_probs[hot_state]),
                "cold_prob": float(current_probs[cold_state]),
                "confidence": float(max(current_probs)),
                "state_means": state_means.tolist(),
            }
        except Exception as e:
            logging.debug("HMM predict failed: %s", e)
            return {"state_name": "ERROR", "state_probs": [], "is_hot": False,
                    "is_cold": False, "confidence": 0}


class LeagueHMM:
    """Manages HMMs for all teams in a league."""

    def __init__(self, n_states=N_STATES, min_games=10, **kwargs):
        self.n_states = n_states
        self.min_games = min_games
        self.team_models = {}
        self.team_margins = defaultdict(list)
        self.team_home = defaultdict(list)

    def add_game(self, team, margin, is_home=True):
        """Record a game result for a team."""
        self.team_margins[team].append(margin)
        self.team_home[team].append(is_home)

    def fit_all(self):
        """Fit HMM for every team with enough data."""
        fitted = 0
        for team, margins in self.team_margins.items():
            if len(margins) < self.min_games:
                continue
            hmm = TeamHMM(self.n_states, self.min_games)
            if hmm.fit(margins, self.team_home.get(team)):
                self.team_models[team] = hmm
                fitted += 1
        logging.debug("HMM fitted for %d/%d teams", fitted, len(self.team_margins))
        return fitted

    def get_state(self, team):
        """Get current HMM state for a team."""
        if team not in self.team_models:
            return {"state_name": "NO_MODEL", "state_probs": [],
                    "is_hot": False, "is_cold": False, "confidence": 0}

        margins = self.team_margins.get(team, [])
        return self.team_models[team].predict_state(margins)

    def get_features(self, home_team, away_team):
        """Get HMM-based features for a matchup.

        Returns dict of features for the meta-learner.
        """
        home_state = self.get_state(home_team)
        away_state = self.get_state(away_team)

        return {
            "home_hmm_hot_prob": home_state.get("hot_prob", 0.25),
            "home_hmm_cold_prob": home_state.get("cold_prob", 0.25),
            "away_hmm_hot_prob": away_state.get("hot_prob", 0.25),
            "away_hmm_cold_prob": away_state.get("cold_prob", 0.25),
            "hmm_hot_diff": home_state.get("hot_prob", 0.25) - away_state.get("hot_prob", 0.25),
            "hmm_confidence": min(home_state.get("confidence", 0), away_state.get("confidence", 0)),
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        """Build HMMs from a games DataFrame.

        Processes games chronologically, accumulating margins per team,
        then fits all team HMMs.
        """
        for _, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                h_score = float(row.get(home_score_col, 0))
                a_score = float(row.get(away_score_col, 0))
            except (ValueError, TypeError):
                continue

            margin = h_score - a_score
            self.add_game(home, margin, is_home=True)
            self.add_game(away, -margin, is_home=False)

        return self.fit_all()
