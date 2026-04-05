"""Classical sports rating and prediction models.

Six time-tested methods collected as shared modules for the mega-ensemble:

1. SimpleRatingSystem (SRS) - Pro-Football-Reference style margin-adjusted
   strength of schedule. Iteratively solves r_i = avg_margin_i + avg(r_opponents).
   Origin: Doug Drinen / PFR, popularised ~2006.

2. ColleyMatrix - Bias-free ranking from the BCS era (2001-2013).
   Solves (2I + H)r = 1 + (w-l)/2 where H encodes head-to-head matchups.
   Origin: Dr. Wesley Colley, 1998 PhD thesis / 2002 BCS component.

3. Log5 - Bill James' formula for head-to-head win probability from
   each team's overall win rate. P(A>B) = pA(1-pB) / [pA(1-pB) + pB(1-pA)].
   Origin: Bill James, 1981 Baseball Abstract.

4. PythagenPat - Dynamic-exponent Pythagorean expectation.
   Instead of a fixed sport exponent, uses exp = (runs_per_game)^0.287.
   Origin: David Smyth (Pythagenpat, 2004) building on Bill James (1980).

5. ExponentialSmoother - Holt double exponential smoothing on points-for
   and points-against per game. Captures level and trend simultaneously.
   Origin: C. C. Holt (1957), applied to sports by various sabermetric writers.

6. MeanReversionDetector - Bollinger-band style z-scores detecting teams
   whose recent performance deviates from their season baseline.
   Overperforming teams (high z) regress; underperforming teams bounce back.
   Origin: John Bollinger (1983) financial bands, adapted for sports analytics.

Dependencies: numpy, scipy (both standard scientific Python, no paid APIs).
"""

import logging
import math
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Simple Rating System (SRS)
# ---------------------------------------------------------------------------

class SimpleRatingSystem:
    """Margin-adjusted strength-of-schedule rating.

    Each team's SRS rating = average margin of victory + average opponent SRS.
    Solved iteratively until convergence (equivalent to a linear system).

    A team with SRS +5 is expected to beat an average (SRS 0) team by 5 points.
    """

    def __init__(self, min_games=5, max_iter=100, tol=1e-4):
        self.min_games = min_games
        self.max_iter = max_iter
        self.tol = tol
        # Accumulators
        self._margins = defaultdict(list)       # team -> [margins]
        self._opponents = defaultdict(list)     # team -> [opponent names]
        self._ratings = {}
        self._fitted = False

    def add_game(self, home_team, away_team, home_score, away_score):
        """Record a game result."""
        margin = home_score - away_score
        self._margins[home_team].append(margin)
        self._margins[away_team].append(-margin)
        self._opponents[home_team].append(away_team)
        self._opponents[away_team].append(home_team)

    def fit(self):
        """Iteratively solve the SRS system."""
        teams = [t for t in self._margins if len(self._margins[t]) >= self.min_games]
        if not teams:
            self._fitted = False
            return False

        # Initialise all ratings to average margin
        ratings = {}
        avg_margins = {}
        for t in teams:
            avg_margins[t] = float(np.mean(self._margins[t]))
            ratings[t] = avg_margins[t]

        for iteration in range(self.max_iter):
            new_ratings = {}
            for t in teams:
                opp_list = self._opponents[t]
                opp_ratings = [ratings.get(o, 0.0) for o in opp_list]
                sos = float(np.mean(opp_ratings)) if opp_ratings else 0.0
                new_ratings[t] = avg_margins[t] + sos

            # Centre ratings (mean = 0)
            mean_r = np.mean(list(new_ratings.values()))
            for t in new_ratings:
                new_ratings[t] -= mean_r

            # Check convergence
            max_delta = max(abs(new_ratings[t] - ratings.get(t, 0.0)) for t in teams)
            ratings = new_ratings
            if max_delta < self.tol:
                break

        self._ratings = ratings
        self._fitted = True
        return True

    def get_features(self, home_team, away_team):
        """Return SRS-based features for a matchup."""
        if not self._fitted:
            self.fit()

        h_srs = self._ratings.get(home_team, 0.0)
        a_srs = self._ratings.get(away_team, 0.0)

        # Strength of schedule: mean opponent rating
        h_opps = self._opponents.get(home_team, [])
        a_opps = self._opponents.get(away_team, [])
        h_sos = float(np.mean([self._ratings.get(o, 0.0) for o in h_opps])) if h_opps else 0.0
        a_sos = float(np.mean([self._ratings.get(o, 0.0) for o in a_opps])) if a_opps else 0.0

        return {
            "srs_home": h_srs,
            "srs_away": a_srs,
            "srs_diff": h_srs - a_srs,
            "srs_sos_diff": h_sos - a_sos,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        """Ingest a DataFrame of games and fit the model."""
        for _, row in games_df.iterrows():
            try:
                hs = float(row[home_score_col])
                as_ = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(row[home_col], row[away_col], hs, as_)
        self.fit()
        return len(self._ratings)


# ---------------------------------------------------------------------------
# 2. Colley Matrix
# ---------------------------------------------------------------------------

class ColleyMatrix:
    """Bias-free ranking system from the BCS era.

    Solves the linear system  C * r = b  where:
        C = 2*I + H   (H_ij = -n_ij = negative number of games between i,j)
        b_i = 1 + (wins_i - losses_i) / 2

    All teams start at 0.5 by construction -- no preseason bias.
    """

    def __init__(self, min_games=3):
        self.min_games = min_games
        self._teams = set()
        self._wins = defaultdict(int)
        self._losses = defaultdict(int)
        self._matchups = defaultdict(int)   # (teamA, teamB) -> count
        self._ratings = {}
        self._fitted = False

    def add_game(self, home_team, away_team, home_score, away_score):
        """Record a game result (winner gets a win, loser gets a loss)."""
        self._teams.add(home_team)
        self._teams.add(away_team)
        if home_score > away_score:
            self._wins[home_team] += 1
            self._losses[away_team] += 1
        elif away_score > home_score:
            self._wins[away_team] += 1
            self._losses[home_team] += 1
        else:
            # Tie: half-win, half-loss each
            self._wins[home_team] += 0.5
            self._losses[home_team] += 0.5
            self._wins[away_team] += 0.5
            self._losses[away_team] += 0.5

        # Symmetric matchup count
        key = tuple(sorted([home_team, away_team]))
        self._matchups[key] += 1

    def fit(self):
        """Solve the Colley system."""
        teams = sorted([t for t in self._teams
                        if self._wins[t] + self._losses[t] >= self.min_games])
        n = len(teams)
        if n < 2:
            self._fitted = False
            return False

        idx = {t: i for i, t in enumerate(teams)}

        # Build C matrix and b vector
        C = np.zeros((n, n))
        b = np.zeros(n)

        for i, t in enumerate(teams):
            total_games = self._wins[t] + self._losses[t]
            C[i, i] = 2.0 + total_games
            b[i] = 1.0 + (self._wins[t] - self._losses[t]) / 2.0

        for (t1, t2), count in self._matchups.items():
            if t1 in idx and t2 in idx:
                i, j = idx[t1], idx[t2]
                C[i, j] -= count
                C[j, i] -= count

        try:
            r = np.linalg.solve(C, b)
        except np.linalg.LinAlgError:
            logger.warning("Colley matrix singular, using least-squares")
            r, _, _, _ = np.linalg.lstsq(C, b, rcond=None)

        self._ratings = {teams[i]: float(r[i]) for i in range(n)}
        self._fitted = True
        return True

    def get_features(self, home_team, away_team):
        """Return Colley-based features for a matchup."""
        if not self._fitted:
            self.fit()

        h_r = self._ratings.get(home_team, 0.5)
        a_r = self._ratings.get(away_team, 0.5)

        return {
            "colley_home": h_r,
            "colley_away": a_r,
            "colley_diff": h_r - a_r,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            try:
                hs = float(row[home_score_col])
                as_ = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(row[home_col], row[away_col], hs, as_)
        self.fit()
        return len(self._ratings)


# ---------------------------------------------------------------------------
# 3. Log5
# ---------------------------------------------------------------------------

class Log5:
    """Bill James' Log5 formula for head-to-head win probability.

    Given each team's overall win percentage (pA, pB against the league),
    the probability that A beats B is:

        P(A>B) = pA*(1-pB) / [pA*(1-pB) + pB*(1-pA)]

    When pA = pB = 0.5 the result is exactly 0.5. The formula naturally
    handles mismatches: a .700 team vs .400 team yields ~.724 for the favourite.
    """

    def __init__(self, min_games=5, floor=0.01):
        self.min_games = min_games
        self.floor = floor  # prevent division by zero
        self._wins = defaultdict(int)
        self._games = defaultdict(int)
        self._home_wins = defaultdict(int)
        self._home_games = defaultdict(int)

    def add_game(self, home_team, away_team, home_score, away_score):
        """Record a game result."""
        self._games[home_team] += 1
        self._games[away_team] += 1
        self._home_games[home_team] += 1

        if home_score > away_score:
            self._wins[home_team] += 1
            self._home_wins[home_team] += 1
        elif away_score > home_score:
            self._wins[away_team] += 1
        else:
            # Ties: half-win each
            self._wins[home_team] += 0.5
            self._wins[away_team] += 0.5
            self._home_wins[home_team] += 0.5

    def _win_pct(self, team):
        g = self._games.get(team, 0)
        if g < self.min_games:
            return 0.5
        return max(self.floor, min(1 - self.floor,
                                    self._wins[team] / g))

    def _home_win_pct(self, team):
        g = self._home_games.get(team, 0)
        if g < self.min_games:
            return 0.5
        return max(self.floor, min(1 - self.floor,
                                    self._home_wins[team] / g))

    @staticmethod
    def log5_prob(pA, pB):
        """Core Log5 calculation."""
        num = pA * (1 - pB)
        den = num + pB * (1 - pA)
        if den == 0:
            return 0.5
        return num / den

    def get_features(self, home_team, away_team):
        """Return Log5-based features for a matchup."""
        pA = self._win_pct(home_team)
        pB = self._win_pct(away_team)
        prob = self.log5_prob(pA, pB)

        # Home-adjusted variant
        h_home_pct = self._home_win_pct(home_team)
        prob_home_adj = self.log5_prob(h_home_pct, pB)

        return {
            "log5_prob": prob,
            "log5_home_adj": prob_home_adj,
            "log5_wp_diff": pA - pB,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            try:
                hs = float(row[home_score_col])
                as_ = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(row[home_col], row[away_col], hs, as_)
        return len(set(self._games.keys()))


# ---------------------------------------------------------------------------
# 4. PythagenPat (dynamic exponent Pythagorean)
# ---------------------------------------------------------------------------

class PythagenPat:
    """Dynamic-exponent Pythagorean win expectation.

    Standard Pythagorean: WinPct = PF^exp / (PF^exp + PA^exp)
    Fixed exponents per sport: MLB 1.83, NBA 14, NFL 2.37, NHL 2.05.

    PythagenPat (David Smyth, 2004) replaces the fixed exponent with:
        exp = ((PF + PA) / G) ^ 0.287
    which auto-adjusts for run environment -- higher-scoring leagues/eras
    get a higher exponent, making blowouts count more.
    """

    def __init__(self, min_games=5, smyth_exp=0.287):
        self.min_games = min_games
        self.smyth_exp = smyth_exp
        self._pf = defaultdict(float)     # points for
        self._pa = defaultdict(float)     # points against
        self._games = defaultdict(int)
        self._wins = defaultdict(float)

    def add_game(self, home_team, away_team, home_score, away_score):
        """Record a game result."""
        self._pf[home_team] += home_score
        self._pa[home_team] += away_score
        self._pf[away_team] += away_score
        self._pa[away_team] += home_score
        self._games[home_team] += 1
        self._games[away_team] += 1
        if home_score > away_score:
            self._wins[home_team] += 1
        elif away_score > home_score:
            self._wins[away_team] += 1
        else:
            self._wins[home_team] += 0.5
            self._wins[away_team] += 0.5

    def _pythagorean(self, team):
        """Compute PythagenPat expected win% for a team."""
        g = self._games.get(team, 0)
        if g < self.min_games:
            return 0.5, 2.0  # default
        pf = self._pf[team]
        pa = self._pa[team]
        if pf <= 0 or pa <= 0:
            return 0.5, 2.0

        rpg = (pf + pa) / g  # total points per game (both teams combined)
        exp = rpg ** self.smyth_exp  # PythagenPat: exp = RPG^0.287 (~14 for NBA)
        pyth_wp = pf ** exp / (pf ** exp + pa ** exp)
        return float(pyth_wp), float(exp)

    def get_features(self, home_team, away_team):
        """Return PythagenPat features for a matchup."""
        h_pyth, h_exp = self._pythagorean(home_team)
        a_pyth, a_exp = self._pythagorean(away_team)

        # Pythagorean residual = actual W% - expected W%  (luck indicator)
        h_g = self._games.get(home_team, 1)
        a_g = self._games.get(away_team, 1)
        h_actual = self._wins.get(home_team, 0) / max(h_g, 1)
        a_actual = self._wins.get(away_team, 0) / max(a_g, 1)
        h_residual = h_actual - h_pyth
        a_residual = a_actual - a_pyth

        return {
            "pythpat_home": h_pyth,
            "pythpat_away": a_pyth,
            "pythpat_diff": h_pyth - a_pyth,
            "pythpat_exp_home": h_exp,
            "pythpat_exp_away": a_exp,
            "pythpat_luck_home": h_residual,
            "pythpat_luck_away": a_residual,
            "pythpat_luck_diff": h_residual - a_residual,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            try:
                hs = float(row[home_score_col])
                as_ = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(row[home_col], row[away_col], hs, as_)
        return len(set(self._games.keys()))


# ---------------------------------------------------------------------------
# 5. Exponential Smoother (Holt double exponential)
# ---------------------------------------------------------------------------

class ExponentialSmoother:
    """Holt double exponential smoothing on per-game scoring.

    Tracks two components per team:
        level (L):  smoothed current scoring rate
        trend (T):  smoothed rate of change

    Update equations (after each game):
        L_t = alpha * x_t + (1-alpha) * (L_{t-1} + T_{t-1})
        T_t = beta  * (L_t - L_{t-1}) + (1-beta) * T_{t-1}

    Forecast for next game: L_t + T_t

    Applied independently to points-for (PF) and points-against (PA).
    """

    def __init__(self, alpha=0.3, beta=0.1, min_games=5):
        self.alpha = alpha
        self.beta = beta
        self.min_games = min_games
        # Per-team state: (level_pf, trend_pf, level_pa, trend_pa, n_games)
        self._state = {}

    def add_game(self, home_team, away_team, home_score, away_score):
        """Update exponential smoothing state for both teams."""
        self._update_team(home_team, home_score, away_score)
        self._update_team(away_team, away_score, home_score)

    def _update_team(self, team, pf, pa):
        if team not in self._state:
            # Initialise: level = first observation, trend = 0
            self._state[team] = {
                "L_pf": float(pf), "T_pf": 0.0,
                "L_pa": float(pa), "T_pa": 0.0,
                "n": 1,
            }
            return

        s = self._state[team]
        # Points for
        prev_L_pf = s["L_pf"]
        s["L_pf"] = self.alpha * pf + (1 - self.alpha) * (s["L_pf"] + s["T_pf"])
        s["T_pf"] = self.beta * (s["L_pf"] - prev_L_pf) + (1 - self.beta) * s["T_pf"]
        # Points against
        prev_L_pa = s["L_pa"]
        s["L_pa"] = self.alpha * pa + (1 - self.alpha) * (s["L_pa"] + s["T_pa"])
        s["T_pa"] = self.beta * (s["L_pa"] - prev_L_pa) + (1 - self.beta) * s["T_pa"]
        s["n"] += 1

    def _forecast(self, team):
        """Forecast next-game PF and PA."""
        if team not in self._state:
            return None
        s = self._state[team]
        if s["n"] < self.min_games:
            return None
        fc_pf = s["L_pf"] + s["T_pf"]
        fc_pa = s["L_pa"] + s["T_pa"]
        return fc_pf, fc_pa, s["T_pf"], s["T_pa"]

    def get_features(self, home_team, away_team):
        """Return exponential-smoothing features for a matchup."""
        h_fc = self._forecast(home_team)
        a_fc = self._forecast(away_team)

        if h_fc is None:
            h_pf, h_pa, h_trend_pf, h_trend_pa = 100.0, 100.0, 0.0, 0.0
        else:
            h_pf, h_pa, h_trend_pf, h_trend_pa = h_fc

        if a_fc is None:
            a_pf, a_pa, a_trend_pf, a_trend_pa = 100.0, 100.0, 0.0, 0.0
        else:
            a_pf, a_pa, a_trend_pf, a_trend_pa = a_fc

        # Expected margin: home's forecasted PF minus away's forecasted PF
        # (home scores vs away scores projected)
        exp_home_margin = h_pf - a_pf

        return {
            "es_home_pf": h_pf,
            "es_home_pa": h_pa,
            "es_away_pf": a_pf,
            "es_away_pa": a_pa,
            "es_margin_forecast": exp_home_margin,
            "es_home_pf_trend": h_trend_pf,
            "es_away_pf_trend": a_trend_pf,
            "es_trend_diff": h_trend_pf - a_trend_pf,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            try:
                hs = float(row[home_score_col])
                as_ = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(row[home_col], row[away_col], hs, as_)
        return len(self._state)


# ---------------------------------------------------------------------------
# 6. Mean Reversion Detector (Bollinger bands for sports)
# ---------------------------------------------------------------------------

class MeanReversionDetector:
    """Bollinger-band style overperformance/underperformance detector.

    For each team, tracks season-long mean and std of scoring margin.
    Computes a z-score of recent performance (last N games) vs the full season.

    High positive z  -> team running hot, expect regression downward
    High negative z  -> team running cold, expect regression upward
    Near zero        -> performing as expected

    The "Bollinger width" (2 * season_std) measures overall volatility.
    """

    def __init__(self, window=10, min_games=15):
        self.window = window        # recent-window size
        self.min_games = min_games
        self._margins = defaultdict(list)

    def add_game(self, home_team, away_team, home_score, away_score):
        """Record a game result."""
        margin = home_score - away_score
        self._margins[home_team].append(margin)
        self._margins[away_team].append(-margin)

    def _z_score(self, team):
        """Compute z-score of recent window vs full season."""
        margins = self._margins.get(team, [])
        if len(margins) < self.min_games:
            return 0.0, 0.0, 0.0

        full = np.array(margins, dtype=float)
        season_mean = np.mean(full)
        season_std = np.std(full)
        if season_std < 0.01:
            season_std = 0.01

        recent = full[-self.window:]
        recent_mean = np.mean(recent)

        z = (recent_mean - season_mean) / season_std
        return float(z), float(season_std), float(recent_mean - season_mean)

    def get_features(self, home_team, away_team):
        """Return mean-reversion features for a matchup."""
        h_z, h_bw, h_dev = self._z_score(home_team)
        a_z, a_bw, a_dev = self._z_score(away_team)

        return {
            "mr_home_zscore": h_z,
            "mr_away_zscore": a_z,
            "mr_zscore_diff": h_z - a_z,
            "mr_home_bollinger_width": h_bw,
            "mr_away_bollinger_width": a_bw,
            "mr_home_deviation": h_dev,
            "mr_away_deviation": a_dev,
            # Reversion signal: negative z means expect bounce-back (positive edge)
            "mr_reversion_edge": -h_z + a_z,
        }

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            try:
                hs = float(row[home_score_col])
                as_ = float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(row[home_col], row[away_col], hs, as_)
        return len(self._margins)
