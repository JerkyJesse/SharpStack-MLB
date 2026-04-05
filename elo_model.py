"""MLBElo model class."""

import os
import json
import math
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from difflib import get_close_matches

from config import (
    TEAM_ABBR, RATINGS_FILE,
    get_season_label, current_timestamp,
)
from color_helpers import cok, cwarn, cdim, chi, div, hdr, cred, cgrn
from platt import load_platt_scaler, apply_platt

# Park factors: run-scoring multiplier for each team's home park (1.0 = average)
# Source: FanGraphs 5-year park factors (2020-2024 aggregate)
# Athletics relocated to Sacramento for 2025; neutral (1.00) pending venue data
PARK_FACTORS = {
    "Colorado Rockies": 1.27,       # Coors Field (extreme altitude + thin air)
    "Boston Red Sox": 1.07,         # Fenway Park (Green Monster, short dimensions)
    "Arizona Diamondbacks": 1.05,   # Chase Field (dry air, retractable roof)
    "Texas Rangers": 1.04,          # Globe Life Field
    "Cincinnati Reds": 1.04,        # Great American Ball Park (short RF)
    "Philadelphia Phillies": 1.03,  # Citizens Bank Park
    "Chicago Cubs": 1.02,           # Wrigley Field (wind effects)
    "Milwaukee Brewers": 1.02,      # American Family Field
    "Toronto Blue Jays": 1.01,      # Rogers Centre
    "Minnesota Twins": 1.01,        # Target Field
    "Los Angeles Angels": 1.00,     # Angel Stadium
    "Baltimore Orioles": 1.00,      # Camden Yards
    "Cleveland Guardians": 0.99,    # Progressive Field
    "Detroit Tigers": 0.99,         # Comerica Park
    "Atlanta Braves": 0.99,         # Truist Park
    "New York Yankees": 0.99,       # Yankee Stadium
    "Kansas City Royals": 0.98,     # Kauffman Stadium
    "Chicago White Sox": 0.98,      # Guaranteed Rate Field
    "Houston Astros": 0.97,         # Minute Maid Park
    "Pittsburgh Pirates": 0.97,     # PNC Park
    "St. Louis Cardinals": 0.97,    # Busch Stadium
    "Washington Nationals": 0.96,   # Nationals Park
    "New York Mets": 0.96,          # Citi Field
    "Los Angeles Dodgers": 0.96,    # Dodger Stadium
    "Tampa Bay Rays": 0.96,         # Tropicana Field
    "San Diego Padres": 0.95,       # Petco Park
    "San Francisco Giants": 0.95,   # Oracle Park
    "Seattle Mariners": 0.94,       # T-Mobile Park
    "Miami Marlins": 0.93,          # loanDepot park
    "Athletics": 1.00,              # Short-name fallback
    "Sacramento Athletics": 1.00,   # New venue
}

# Timezone offset (hours from UTC) for each team's home city
TEAM_TIMEZONE = {
    "Arizona Diamondbacks": -7, "Atlanta Braves": -5,
    "Baltimore Orioles": -5, "Boston Red Sox": -5,
    "Chicago Cubs": -6, "Chicago White Sox": -6,
    "Cincinnati Reds": -5, "Cleveland Guardians": -5,
    "Colorado Rockies": -7, "Detroit Tigers": -5,
    "Houston Astros": -6, "Kansas City Royals": -6,
    "Los Angeles Angels": -8, "Los Angeles Dodgers": -8,
    "Miami Marlins": -5, "Milwaukee Brewers": -6,
    "Minnesota Twins": -6, "New York Mets": -5,
    "New York Yankees": -5, "Athletics": -8, "Sacramento Athletics": -8,
    "Philadelphia Phillies": -5, "Pittsburgh Pirates": -5,
    "San Diego Padres": -8, "San Francisco Giants": -8,
    "Seattle Mariners": -8, "St. Louis Cardinals": -6,
    "Tampa Bay Rays": -5, "Texas Rangers": -6,
    "Toronto Blue Jays": -5, "Washington Nationals": -5,
}


class MLBElo:
    K_PITCHER = 6  # pitcher rating update speed

    def __init__(self, base_rating=1500.0, k=1.0, home_adv=23.47,
                 use_mov=True, player_boost=2.54, starter_boost=9.61,
                 rest_factor=0.0,
                 form_weight=0.0, travel_factor=0.0, sos_factor=0.0,
                 playoff_hca_factor=0.934, pace_factor=5.0,
                 division_factor=0.0, mean_reversion=10.0,
                 pyth_factor=16.0, home_road_factor=4.04,
                 mov_base=0.3,
                 b2b_penalty=26.51, road_trip_factor=0.0,
                 homestand_factor=1.41, win_streak_factor=0.0,
                 altitude_factor=12.48, season_phase_factor=9.35,
                 scoring_consistency_factor=0.0, rest_advantage_cap=4.14,
                 park_factor_weight=0.0,
                 mov_cap=19.9, east_travel_penalty=0.0,
                 series_adaptation=3.92, interleague_factor=2.04,
                 bullpen_factor=6.43, opp_pitcher_factor=18.0,
                 k_decay=2.07, surprise_k=0.0,
                 elo_scale=400.0):
        self.base_rating   = base_rating
        self.k             = k
        self.home_adv      = home_adv
        self.use_mov       = use_mov
        self.player_boost  = player_boost     # team-level player strength
        self.starter_boost = starter_boost    # starting pitcher quality adjustment
        self.rest_factor   = rest_factor
        self.form_weight   = form_weight
        self.travel_factor   = travel_factor    # Elo penalty per timezone crossed
        self.sos_factor      = sos_factor       # strength-of-schedule weight
        self.playoff_hca_factor = playoff_hca_factor  # multiply home_adv in playoffs (< 1 = reduced)
        self.pace_factor     = pace_factor      # run-environment mismatch adjustment
        self.division_factor = division_factor  # reduce confidence for divisional games
        self.mean_reversion  = mean_reversion   # regression after extreme results
        self.pyth_factor     = pyth_factor      # Pythagorean expected W% adjustment
        self.home_road_factor = home_road_factor # team-specific home/road split bonus
        self.mov_base        = mov_base          # MOV multiplier constant (shifts log curve)
        self.b2b_penalty     = b2b_penalty      # extra penalty for back-to-back / no off day
        self.road_trip_factor = road_trip_factor  # penalty for extended road trips
        self.homestand_factor = homestand_factor  # bonus for extended homestands
        self.win_streak_factor = win_streak_factor  # momentum from win/loss streaks
        self.altitude_factor = altitude_factor    # multiplier on altitude bonus (0=raw)
        self.season_phase_factor = season_phase_factor  # early-season dampener
        self.scoring_consistency_factor = scoring_consistency_factor  # penalty for volatile scoring
        self.rest_advantage_cap = rest_advantage_cap  # max rest days counted (0=uncapped)
        self.park_factor_weight = park_factor_weight  # park factor adjustment weight
        self.mov_cap         = mov_cap              # cap blowout margins at N runs (0=uncapped)
        self.east_travel_penalty = east_travel_penalty  # extra penalty for eastward travel
        self.series_adaptation = series_adaptation  # visiting team adapts in series (game 2/3)
        self.interleague_factor = interleague_factor  # adjustment for interleague games
        self.bullpen_factor = bullpen_factor        # bullpen quality weight
        self.opp_pitcher_factor = opp_pitcher_factor  # opposing pitcher quality weight
        self.k_decay       = k_decay                # K-factor decay: k shrinks as team plays more games
        self.surprise_k    = surprise_k             # extra K after surprising results (autocorrelation)
        self.elo_scale     = elo_scale              # probability scaling divisor (400=standard, lower=more extreme)
        self.ratings       = defaultdict(lambda: base_rating)
        self.team_names    = []
        self._team_lookup  = {}
        self._player_scores = {}
        self._last_game_date = {}          # team -> datetime of last game
        self._recent_results = defaultdict(list)  # team -> list of last N results (1=win, 0=loss)
        self._altitude_bonus = {}          # team -> extra home Elo (auto-calculated from data)
        self._last_game_location = {}      # team -> team name of venue city (for travel calc)
        self._opponent_elos = defaultdict(list)   # team -> list of opponent Elo at game time (SOS)
        self._team_scores = defaultdict(list)     # team -> list of recent (runs_for, runs_against)
        self._last_margin = {}                   # team -> margin of last game (for mean reversion)
        self._home_results = defaultdict(list)    # team -> list of home game results (1=win, 0=loss)
        self._road_results = defaultdict(list)    # team -> list of road game results
        self._consecutive_away = defaultdict(int)  # team -> current away game streak
        self._consecutive_home = defaultdict(int)  # team -> current home game streak
        self._game_number = defaultdict(int)       # team -> games played this season
        self._pitcher_ratings = {}         # pitcher_name -> cumulative Elo-like rating
        self._pitcher_starts = defaultdict(int)  # pitcher_name -> number of starts
        self._pitcher_game_scores = defaultdict(list)  # pitcher -> rolling game scores
        self._team_bullpen_rating = defaultdict(float)  # team -> cumulative bullpen Elo
        self._last_opponent = {}           # team -> last opponent name (for series detection)
        self._series_game_num = defaultdict(int)  # (home, away) -> game number in series
        self._league_membership = {}       # team -> "AL" or "NL" (for interleague)
        self._platt_scaler = load_platt_scaler()
        self._xgb_model = None    # XGBoost booster (loaded from enhanced model)
        self._xgb_meta  = None    # XGBoost metadata (feature_cols, elo_weight)
        self._mega_predictor = None  # MegaPredictor (31-model ensemble)
        self.metadata = {
            "season_label": get_season_label(), "trained_games": 0,
            "saved_at": None, "source_file": None, "settings": self.settings_dict(),
        }

    def settings_dict(self):
        return {
            "base_rating": self.base_rating, "k": self.k,
            "home_adv": self.home_adv, "use_mov": self.use_mov,
            "player_boost": self.player_boost,
            "starter_boost": self.starter_boost,
            "rest_factor": self.rest_factor,
            "form_weight": self.form_weight,
            "travel_factor": self.travel_factor,
            "sos_factor": self.sos_factor,
            "playoff_hca_factor": self.playoff_hca_factor,
            "pace_factor": self.pace_factor,
            "division_factor": self.division_factor,
            "mean_reversion": self.mean_reversion,
            "pyth_factor": self.pyth_factor,
            "home_road_factor": self.home_road_factor,
            "mov_base": self.mov_base,
            "b2b_penalty": self.b2b_penalty,
            "road_trip_factor": self.road_trip_factor,
            "homestand_factor": self.homestand_factor,
            "win_streak_factor": self.win_streak_factor,
            "altitude_factor": self.altitude_factor,
            "season_phase_factor": self.season_phase_factor,
            "scoring_consistency_factor": self.scoring_consistency_factor,
            "rest_advantage_cap": self.rest_advantage_cap,
            "park_factor_weight": self.park_factor_weight,
            "mov_cap": self.mov_cap,
            "east_travel_penalty": self.east_travel_penalty,
            "series_adaptation": self.series_adaptation,
            "interleague_factor": self.interleague_factor,
            "bullpen_factor": self.bullpen_factor,
            "opp_pitcher_factor": self.opp_pitcher_factor,
            "k_decay": self.k_decay,
            "surprise_k": self.surprise_k,
        }

    def set_pitcher_priors(self, advanced_df):
        """Initialize pitcher ratings from external stats (ERA, K, W).

        Pitchers with low ERA get positive initial ratings (good pitchers).
        League average ERA (~4.50) maps to 0. Scaled so elite (~2.50 ERA)
        gets about +3.0 rating points.
        """
        if advanced_df is None or advanced_df.empty:
            return
        for _, row in advanced_df.iterrows():
            name = str(row.get("Player", "")).strip()
            if not name:
                continue
            era = float(row.get("ERA", 0.0) or 0.0)
            k = float(row.get("K", 0.0) or 0.0)
            w = float(row.get("W", 0.0) or 0.0)
            if era <= 0 and k <= 0 and w <= 0:
                continue
            # ERA component: 4.50 is league average, lower = better
            era_component = 0.0
            if era > 0:
                era_component = (4.50 - era) * 1.5  # ~3.0 for elite, -2.25 for 6.0 ERA
            # Strikeout bonus (more K = better)
            k_component = 0.0
            if k > 0:
                k_component = min(k / 100.0, 1.5)  # cap at 1.5 for 150+ K
            # Wins bonus
            w_component = 0.0
            if w > 0:
                w_component = min(w / 10.0, 1.0)  # cap at 1.0 for 10+ W
            initial_rating = era_component + k_component + w_component
            if name not in self._pitcher_ratings:
                self._pitcher_ratings[name] = initial_rating
                self._pitcher_starts[name] = 2  # Pretend 2 starts so adjustment kicks in
        logging.info("Initialized %d pitcher priors from stats", len(self._pitcher_ratings))

    def set_fip_priors(self, fangraphs_df, season=None, only_new=True):
        """Initialize pitcher ratings from FanGraphs FIP/xFIP data.

        FIP (Fielding Independent Pitching) is more predictive than ERA
        because it removes defense and luck. xFIP normalizes HR/FB rate.
        Uses weighted combo: 60% FIP + 40% xFIP.
        """
        if fangraphs_df is None or fangraphs_df.empty:
            return 0
        df = fangraphs_df
        if season is not None and "season" in df.columns:
            df = df[df["season"] == season]
        if df.empty:
            return 0
        count = 0
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            if only_new and name in self._pitcher_ratings:
                continue
            fip = float(row.get("FIP", 0.0) or 0.0)
            xfip = float(row.get("xFIP", 0.0) or 0.0)
            ip = float(row.get("IP", 0.0) or 0.0)
            kbb = float(row.get("K-BB%", 0.0) or 0.0)
            if fip <= 0 and xfip <= 0:
                continue
            # Weighted FIP: 60% FIP + 40% xFIP (xFIP stabilizes faster)
            weighted_fip = 0.6 * fip + 0.4 * xfip if xfip > 0 else fip
            # Convert to rating: league avg FIP ~4.00, lower = better
            fip_component = (4.00 - weighted_fip) * 1.5
            # K-BB% bonus: elite K-BB% (~25%+) more predictable
            kbb_component = 0.0
            if kbb > 0:
                kbb_component = (kbb - 0.12) * 3.0  # ±0.5-1.0 for extremes
            # IP-based confidence: more innings = more reliable prior
            ip_confidence = min(ip / 150.0, 1.0) if ip > 0 else 0.5
            initial_rating = (fip_component + kbb_component) * ip_confidence
            self._pitcher_ratings[name] = initial_rating
            self._pitcher_starts[name] = 2  # Pretend 2 starts
            count += 1
        if count > 0:
            logging.info("Set %d FIP-based pitcher priors (season=%s)", count, season)
        return count

    def set_player_stats(self, player_df):
        from data_players import build_league_player_scores
        self._player_scores = build_league_player_scores(player_df)
        if self.player_boost > 0 and len(self._player_scores) < 25:
            logging.warning(
                "player_boost=%.1f but only %d teams have scores -- "
                "consider set boost=0 until data is stable.",
                self.player_boost, len(self._player_scores)
            )

    def _rebuild_lookup(self):
        self._team_lookup = {t.lower(): t for t in self.team_names}

    def rest_days(self, team, game_date):
        """Days since team's last game. Returns None if unknown."""
        last = self._last_game_date.get(team)
        if last is None or game_date is None:
            return None
        delta = (game_date - last).days
        return max(0, delta)

    def rest_adjustment(self, team, game_date):
        """Elo adjustment for rest. B2B games are rare in MLB but off-days help.
        0 days rest (doubleheader) = penalty, 1 day = normal, 2+ = slight bonus."""
        if self.rest_factor == 0:
            return 0.0
        rd = self.rest_days(team, game_date)
        if rd is None:
            return 0.0
        cap = int(self.rest_advantage_cap) if self.rest_advantage_cap > 0 else 3
        adj = self.rest_factor * (min(rd, cap) - 1)
        return adj

    def form_adjustment(self, team):
        """Elo adjustment based on recent form (last 10 games win%)."""
        if self.form_weight == 0:
            return 0.0
        results = self._recent_results.get(team, [])
        if len(results) < 5:
            return 0.0
        win_pct = sum(results[-10:]) / len(results[-10:])
        return self.form_weight * (win_pct - 0.5)

    def travel_adjustment(self, team, venue_team):
        """Elo penalty for timezone crossings since last game.
        venue_team = the home team of the current game (determines where team is playing)."""
        if self.travel_factor == 0:
            return 0.0
        last_loc = self._last_game_location.get(team)
        if last_loc is None:
            return 0.0
        prev_tz = TEAM_TIMEZONE.get(last_loc, -6)
        curr_tz = TEAM_TIMEZONE.get(venue_team, -6)
        tz_diff = abs(curr_tz - prev_tz)
        if tz_diff == 0:
            return 0.0
        return -self.travel_factor * tz_diff

    def sos_adjustment(self, team):
        """Elo adjustment for strength of schedule.
        Teams facing tougher opponents get a bonus (their rating is battle-tested)."""
        if self.sos_factor == 0:
            return 0.0
        opp_elos = self._opponent_elos.get(team, [])
        if len(opp_elos) < 5:
            return 0.0
        recent = opp_elos[-10:]
        avg_opp = sum(recent) / len(recent)
        return self.sos_factor * (avg_opp - self.base_rating) / 100.0

    def playoff_home_adv(self, game_date):
        """Return adjusted home advantage for playoffs.
        Playoffs (October) have reduced home field advantage."""
        if game_date is None:
            return self.home_adv
        month = game_date.month
        if month >= 10:
            return self.home_adv * self.playoff_hca_factor
        return self.home_adv

    def is_playoff_game(self, game_date):
        """Detect if a game is likely a playoff game based on date."""
        if game_date is None:
            return False
        return game_date.month >= 10

    def pace_adjustment(self, team_a, team_b):
        """Elo adjustment based on run-environment mismatch.
        High-scoring teams benefit when playing other high-scoring teams.
        Low-scoring/pitching-dominant teams benefit from controlling the game pace."""
        if self.pace_factor == 0:
            return 0.0, 0.0
        scores_a = self._team_scores.get(team_a, [])
        scores_b = self._team_scores.get(team_b, [])
        if len(scores_a) < 5 or len(scores_b) < 5:
            return 0.0, 0.0
        recent_a = scores_a[-10:]
        recent_b = scores_b[-10:]
        # Estimated run environment = (runs_for + runs_against) / 2 per game
        pace_a = sum(rf + ra for rf, ra in recent_a) / (2.0 * len(recent_a))
        pace_b = sum(rf + ra for rf, ra in recent_b) / (2.0 * len(recent_b))
        # Pace differential: positive = A scores/allows more
        pace_diff = pace_a - pace_b
        # The lower-scoring team benefits from pace mismatch (pitching dominance)
        adj_a = -self.pace_factor * pace_diff / 10.0
        adj_b = self.pace_factor * pace_diff / 10.0
        return adj_a, adj_b

    def division_adjustment(self, team_a, team_b):
        """Reduce prediction confidence for divisional games"""
        if self.division_factor == 0:
            return 0.0, 0.0
        from config import same_division
        if same_division(team_a, team_b):
            ra, rb = self.ratings[team_a], self.ratings[team_b]
            diff = ra - rb
            adj = self.division_factor * diff / 100.0
            return -adj, adj
        return 0.0, 0.0

    def mean_reversion_adjustment(self, team):
        """After extreme results, expect regression to mean"""
        if self.mean_reversion == 0 or team not in self._last_margin:
            return 0
        margin = self._last_margin[team]
        if abs(margin) > 5:  # MLB blowout threshold: 5+ runs
            return -self.mean_reversion * margin / 100.0
        return 0

    def pythagorean_adjustment(self, team):
        """Elo adjustment based on Pythagorean expected win rate.
        Uses (RS^1.83) / (RS^1.83 + RA^1.83) which is more predictive
        than actual win percentage in baseball."""
        if self.pyth_factor == 0:
            return 0.0
        scores = self._team_scores.get(team, [])
        if len(scores) < 10:
            return 0.0
        recent = scores[-15:]  # Use 15-game window
        rs = max(sum(rf for rf, _ in recent), 0.1)
        ra = max(sum(ra_val for _, ra_val in recent), 0.1)
        # Pythagorean exponent 1.83 (Davenport/BP standard for baseball)
        exp = 1.83
        rs_exp = rs ** exp
        ra_exp = ra ** exp
        if rs_exp + ra_exp == 0:
            return 0.0
        pyth_wpct = rs_exp / (rs_exp + ra_exp)
        # Convert to Elo-scale adjustment: above .500 = positive, below = negative
        return self.pyth_factor * (pyth_wpct - 0.5)

    def home_road_adjustment(self, team, is_home):
        """Elo adjustment based on team's home/road split performance.
        Teams that consistently over/under-perform at home get an adjustment."""
        if self.home_road_factor == 0:
            return 0.0
        results = self._home_results.get(team, []) if is_home else self._road_results.get(team, [])
        if len(results) < 10:
            return 0.0
        recent = results[-20:]
        split_wpct = sum(recent) / len(recent)
        # Positive adjustment if team is above-average at home/road
        return self.home_road_factor * (split_wpct - 0.5)

    def b2b_penalty_adjustment(self, team, game_date):
        """Extra penalty for back-to-back games (no off day, rest == 0)."""
        if self.b2b_penalty == 0:
            return 0.0
        rd = self.rest_days(team, game_date)
        if rd is not None and rd == 0:
            return -self.b2b_penalty
        return 0.0

    def road_trip_adjustment(self, team):
        """Penalty for extended road trips (3+ consecutive away games)."""
        if self.road_trip_factor == 0:
            return 0.0
        consec = self._consecutive_away.get(team, 0)
        if consec >= 3:
            return -self.road_trip_factor * min(consec - 2, 5)
        return 0.0

    def homestand_adjustment(self, team):
        """Bonus for extended homestands (3+ consecutive home games)."""
        if self.homestand_factor == 0:
            return 0.0
        consec = self._consecutive_home.get(team, 0)
        if consec >= 3:
            return self.homestand_factor * min(consec - 2, 5)
        return 0.0

    def win_streak_adjustment(self, team):
        """Momentum bonus for win streaks, penalty for loss streaks."""
        if self.win_streak_factor == 0:
            return 0.0
        results = self._recent_results.get(team, [])
        if len(results) < 2:
            return 0.0
        # Count streak from end
        streak = 0
        last = results[-1]
        for r in reversed(results):
            if r == last:
                streak += 1
            else:
                break
        streak = min(streak, 5)
        if last >= 0.5:
            return self.win_streak_factor * streak / 5.0
        else:
            return -self.win_streak_factor * streak / 5.0

    def season_phase_adjustment(self, team_a, team_b):
        """Reduce confidence in early-season predictions."""
        if self.season_phase_factor == 0:
            return 0.0
        gn_a = self._game_number.get(team_a, 0)
        gn_b = self._game_number.get(team_b, 0)
        avg_gn = (gn_a + gn_b) / 2.0
        game_frac = min(avg_gn / 162.0, 1.0)
        if game_frac < 0.20:
            # Early season: dampen rating difference
            return self.season_phase_factor * (0.20 - game_frac)
        return 0.0

    def scoring_consistency_adjustment(self, team):
        """Penalty for teams with volatile scoring (high std of recent scores)."""
        if self.scoring_consistency_factor == 0:
            return 0.0
        scores = self._team_scores.get(team, [])
        if len(scores) < 5:
            return 0.0
        recent_rf = [rf for rf, ra in scores[-10:]]
        std = float(np.std(recent_rf))
        # MLB avg scoring std ~3 runs; penalize above-average volatility
        return -self.scoring_consistency_factor * (std - 3.0) / 10.0

    def directional_travel_adjustment(self, team, venue_team):
        """Extra penalty for eastward travel (confirmed by PNAS research).
        Eastward travel disrupts circadian rhythm more than westward."""
        if self.east_travel_penalty == 0:
            return 0.0
        last_loc = self._last_game_location.get(team)
        if last_loc is None:
            return 0.0
        prev_tz = TEAM_TIMEZONE.get(last_loc, -6)
        curr_tz = TEAM_TIMEZONE.get(venue_team, -6)
        tz_diff = curr_tz - prev_tz  # positive = traveling east
        if tz_diff > 0:  # eastward (harder)
            return -self.east_travel_penalty * tz_diff
        elif tz_diff < 0:  # westward (easier) - half penalty
            return -self.east_travel_penalty * abs(tz_diff) * 0.3
        return 0.0

    def series_context_adjustment(self, team_a, team_b, team_a_home):
        """Visiting team adapts during a series (games 2/3 are closer).
        Research shows visiting teams adjust after seeing the host's pitching."""
        if self.series_adaptation == 0:
            return 0.0, 0.0
        home = team_a if team_a_home else team_b
        away = team_b if team_a_home else team_a
        key = (home, away)
        game_num = self._series_game_num.get(key, 1)
        if game_num >= 2:
            # Visiting team adapts: reduce home advantage effect
            adapt = self.series_adaptation * min(game_num - 1, 3) / 100.0
            if team_a_home:
                return -adapt, adapt  # reduce A's advantage, boost B
            else:
                return adapt, -adapt
        return 0.0, 0.0

    def interleague_adjustment(self, team_a, team_b):
        """Interleague games are less predictable (different league dynamics).
        Shrink the rating gap for AL vs NL matchups."""
        if self.interleague_factor == 0:
            return 0.0, 0.0
        league_a = self._league_membership.get(team_a, "")
        league_b = self._league_membership.get(team_b, "")
        if league_a and league_b and league_a != league_b:
            diff = self.ratings[team_a] - self.ratings[team_b]
            adj = self.interleague_factor * diff / 100.0
            return -adj, adj  # shrink the gap
        return 0.0, 0.0

    def bullpen_quality_adjustment(self, team):
        """Adjust for bullpen quality. Strong bullpens protect leads."""
        if self.bullpen_factor == 0:
            return 0.0
        bp_rating = self._team_bullpen_rating.get(team, 0.0)
        return self.bullpen_factor * bp_rating / 100.0

    def opp_pitcher_quality_adjustment(self, pitcher_name):
        """Adjust for opposing starting pitcher quality.
        Facing an ace = lower win probability."""
        if self.opp_pitcher_factor == 0 or not pitcher_name:
            return 0.0
        starts = self._pitcher_starts.get(pitcher_name, 0)
        if starts < 5:
            return 0.0
        rating = self._pitcher_ratings.get(pitcher_name, 0.0)
        # Negative rating pitcher = good for the team facing them
        return -self.opp_pitcher_factor * rating / 100.0

    def compute_game_score(self, innings_pitched=6.0, hits=6, runs=3,
                           earned_runs=3, walks=2, strikeouts=5,
                           home_runs=1):
        """FiveThirtyEight-style game score (Tangotiger variant).
        Higher = better pitching performance."""
        outs = innings_pitched * 3
        gs = 47.4 + strikeouts + (outs * 1.5) - (walks * 2) - (hits * 2) - (runs * 3) - (home_runs * 4)
        return gs

    def park_factor_adjustment(self, home_team):
        """Elo adjustment based on park factor for the home team's ballpark."""
        if self.park_factor_weight <= 0:
            return 0.0
        pf = PARK_FACTORS.get(home_team, 1.0)
        return self.park_factor_weight * (pf - 1.0) * 100.0

    def starter_adjustment(self, pitcher_name):
        """Elo adjustment based on starting pitcher quality.
        Returns the pitcher's accumulated rating scaled by starter_boost.
        Pitchers with < 8 starts get 0 adjustment (noisy ratings hurt accuracy)."""
        if not pitcher_name or self.starter_boost == 0:
            return 0.0
        starts = self._pitcher_starts.get(pitcher_name, 0)
        if starts < 8:
            return 0.0
        rating = self._pitcher_ratings.get(pitcher_name, 0.0)
        return self.starter_boost * rating / 100.0

    def update_pitcher_rating(self, pitcher_name, team_won, score_diff):
        """Update pitcher's cumulative rating after a start."""
        if not pitcher_name:
            return
        self._pitcher_starts[pitcher_name] += 1
        outcome = 1.0 if team_won else 0.0
        mov = math.log(abs(score_diff) + 1.0) + 0.8
        update = self.K_PITCHER * (outcome - 0.5) * mov
        self._pitcher_ratings[pitcher_name] = self._pitcher_ratings.get(pitcher_name, 0.0) + update

    def regress_pitcher_ratings(self, factor=0.5):
        """Regress all pitcher ratings toward 0 at season boundaries."""
        for name in list(self._pitcher_ratings.keys()):
            self._pitcher_ratings[name] *= (1.0 - factor)

    def effective_k(self, team):
        """Variable K-factor: higher early season, decays as team plays more games.
        Also spikes after surprise results (autocorrelation-based learning)."""
        k = self.k
        # Season-adaptive K: early season K is 2x, decays to 1x by game 81
        if self.k_decay > 0:
            games = self._game_number.get(team, 0)
            # K multiplier: starts at (1 + k_decay), decays to 1.0 by mid-season
            decay_mult = 1.0 + self.k_decay * max(0, 1.0 - games / 81.0)
            k *= decay_mult
        # Surprise-adaptive K: if last result was surprising, increase K
        if self.surprise_k > 0:
            results = self._recent_results.get(team, [])
            if len(results) >= 2:
                # Check if last result was a streak-breaker
                last = results[-1]
                prev = results[-2]
                if last != prev:  # direction changed (win->loss or loss->win)
                    k += self.surprise_k
        return k

    def expected_score(self, ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / self.elo_scale))

    def update_game(self, home_team, away_team, home_score, away_score,
                    neutral_site=False, game_date=None,
                    home_starter="", away_starter=""):
        # Track opponent Elo at game time (for SOS)
        self._opponent_elos[home_team].append(self.ratings[away_team])
        self._opponent_elos[away_team].append(self.ratings[home_team])
        for team in (home_team, away_team):
            if len(self._opponent_elos[team]) > 10:
                self._opponent_elos[team] = self._opponent_elos[team][-10:]
        # Track scores for pace estimation
        self._team_scores[home_team].append((float(home_score), float(away_score)))
        self._team_scores[away_team].append((float(away_score), float(home_score)))
        for team in (home_team, away_team):
            if len(self._team_scores[team]) > 10:
                self._team_scores[team] = self._team_scores[team][-10:]

        ra = self.ratings[home_team] + (0.0 if neutral_site else self.home_adv)
        rb = self.ratings[away_team]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        if home_score > away_score:
            sa, sb = 1.0, 0.0
        elif home_score < away_score:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        # MLB MOV: shift log up so 1-run games (most common) produce meaningful updates.
        score_diff_for_mov = abs(home_score - away_score)
        if self.mov_cap > 0:
            score_diff_for_mov = min(score_diff_for_mov, self.mov_cap)
        mov = (math.log(score_diff_for_mov + 1.0) + self.mov_base) if self.use_mov else 1.0
        k_home = self.effective_k(home_team)
        k_away = self.effective_k(away_team)
        self.ratings[home_team] += k_home * mov * (sa - ea)
        self.ratings[away_team] += k_away * mov * (sb - eb)
        # Update starting pitcher ratings
        score_diff = home_score - away_score
        self.update_pitcher_rating(home_starter, home_score > away_score, score_diff)
        self.update_pitcher_rating(away_starter, away_score > home_score, -score_diff)
        # Track last game date for rest-day calculations
        if game_date is not None:
            self._last_game_date[home_team] = game_date
            self._last_game_date[away_team] = game_date
        # Track game location (home team's city = venue)
        self._last_game_location[home_team] = home_team
        self._last_game_location[away_team] = home_team
        # Track last margin for mean reversion
        self._last_margin[home_team] = home_score - away_score
        self._last_margin[away_team] = away_score - home_score
        # Track recent results for form
        self._recent_results[home_team].append(sa)
        self._recent_results[away_team].append(sb)
        for team in (home_team, away_team):
            if len(self._recent_results[team]) > 10:
                self._recent_results[team] = self._recent_results[team][-10:]
        # Track home/road results
        self._home_results[home_team].append(sa)
        self._road_results[away_team].append(sb)
        for team in (home_team, away_team):
            if len(self._home_results.get(team, [])) > 20:
                self._home_results[team] = self._home_results[team][-20:]
            if len(self._road_results.get(team, [])) > 20:
                self._road_results[team] = self._road_results[team][-20:]
        # Track consecutive home/away games
        self._consecutive_home[home_team] = self._consecutive_home.get(home_team, 0) + 1
        self._consecutive_away[home_team] = 0
        self._consecutive_away[away_team] = self._consecutive_away.get(away_team, 0) + 1
        self._consecutive_home[away_team] = 0
        # Track game number per team
        self._game_number[home_team] = self._game_number.get(home_team, 0) + 1
        self._game_number[away_team] = self._game_number.get(away_team, 0) + 1
        # Track series context (same home/away matchup on consecutive days)
        key = (home_team, away_team)
        last_opp_home = self._last_opponent.get(home_team)
        if last_opp_home == away_team:
            self._series_game_num[key] = self._series_game_num.get(key, 0) + 1
        else:
            self._series_game_num[key] = 1
        self._last_opponent[home_team] = away_team
        self._last_opponent[away_team] = home_team
        # Update bullpen rating (proxy: if starter gave up many runs, bullpen was tested)
        # Simple heuristic: winning team's bullpen gets credit, losing team's debited
        bp_update = 0.3
        if home_score > away_score:
            self._team_bullpen_rating[home_team] = self._team_bullpen_rating.get(home_team, 0.0) + bp_update
            self._team_bullpen_rating[away_team] = self._team_bullpen_rating.get(away_team, 0.0) - bp_update
        elif away_score > home_score:
            self._team_bullpen_rating[away_team] = self._team_bullpen_rating.get(away_team, 0.0) + bp_update
            self._team_bullpen_rating[home_team] = self._team_bullpen_rating.get(home_team, 0.0) - bp_update
        # Decay bullpen ratings toward 0 slowly
        for t in (home_team, away_team):
            self._team_bullpen_rating[t] = self._team_bullpen_rating.get(t, 0.0) * 0.98

    def win_prob(self, team_a, team_b, team_a_home=True, neutral_site=False,
                 calibrated=True, game_date=None, use_injuries=True,
                 home_starter="", away_starter=""):
        """
        Win probability for team_a.
        calibrated=True applies Platt scaler if fitted.
        calibrated=False returns raw Elo probability (used internally for fitting).
        home_starter/away_starter: pitcher names for starter quality adjustment.
        """
        ra = self.ratings[team_a]
        rb = self.ratings[team_b]
        # Home field advantage (reduced in playoffs)
        if not neutral_site and team_a_home is not None:
            hca = self.playoff_home_adv(game_date)
            ra += hca if team_a_home else 0.0
            rb += 0.0 if team_a_home else hca
            # Altitude bonus (only for the home team)
            if self._altitude_bonus:
                home_team = team_a if team_a_home else team_b
                alt_bonus = self._altitude_bonus.get(home_team, 0.0)
                if alt_bonus > 0:
                    if self.altitude_factor > 0:
                        alt_bonus = alt_bonus * self.altitude_factor
                    if team_a_home:
                        ra += alt_bonus
                    else:
                        rb += alt_bonus
        if self._player_scores and self.player_boost > 0.0:
            ra += self._player_scores.get(team_a, 0.0) * self.player_boost
            rb += self._player_scores.get(team_b, 0.0) * self.player_boost
        # Starting pitcher quality adjustment
        if self.starter_boost > 0:
            h_sp = home_starter if team_a_home else away_starter
            a_sp = away_starter if team_a_home else home_starter
            ra += self.starter_adjustment(h_sp)
            rb += self.starter_adjustment(a_sp)
        # Rest day adjustments
        ra += self.rest_adjustment(team_a, game_date)
        rb += self.rest_adjustment(team_b, game_date)
        # Recent form adjustments
        ra += self.form_adjustment(team_a)
        rb += self.form_adjustment(team_b)
        # Travel fatigue adjustments
        if self.travel_factor > 0 and not neutral_site and team_a_home is not None:
            venue_team = team_a if team_a_home else team_b
            ra += self.travel_adjustment(team_a, venue_team)
            rb += self.travel_adjustment(team_b, venue_team)
        # Strength of schedule adjustments
        ra += self.sos_adjustment(team_a)
        rb += self.sos_adjustment(team_b)
        # Pace mismatch adjustments
        pace_a, pace_b = self.pace_adjustment(team_a, team_b)
        ra += pace_a
        rb += pace_b
        # Divisional rivalry adjustments
        div_a, div_b = self.division_adjustment(team_a, team_b)
        ra += div_a
        rb += div_b
        # Mean reversion adjustments (after blowouts)
        ra += self.mean_reversion_adjustment(team_a)
        rb += self.mean_reversion_adjustment(team_b)
        # Pythagorean expected win rate adjustments
        ra += self.pythagorean_adjustment(team_a)
        rb += self.pythagorean_adjustment(team_b)
        # Home/road split adjustments
        if not neutral_site and team_a_home is not None:
            a_is_home = bool(team_a_home)
            ra += self.home_road_adjustment(team_a, a_is_home)
            rb += self.home_road_adjustment(team_b, not a_is_home)
        # Back-to-back penalty (no off day)
        ra += self.b2b_penalty_adjustment(team_a, game_date)
        rb += self.b2b_penalty_adjustment(team_b, game_date)
        # Road trip / homestand
        ra += self.road_trip_adjustment(team_a)
        rb += self.road_trip_adjustment(team_b)
        ra += self.homestand_adjustment(team_a)
        rb += self.homestand_adjustment(team_b)
        # Win streak momentum
        ra += self.win_streak_adjustment(team_a)
        rb += self.win_streak_adjustment(team_b)
        # Scoring consistency
        ra += self.scoring_consistency_adjustment(team_a)
        rb += self.scoring_consistency_adjustment(team_b)
        # Season phase dampener (reduces diff in early season)
        phase_adj = self.season_phase_adjustment(team_a, team_b)
        if phase_adj > 0:
            diff = ra - rb
            ra -= phase_adj * diff / 100.0
            rb += phase_adj * diff / 100.0
        # Park factor adjustment (home team only)
        if not neutral_site and team_a_home is not None:
            home_team = team_a if team_a_home else team_b
            pf_adj = self.park_factor_adjustment(home_team)
            if team_a_home:
                ra += pf_adj
            else:
                rb += pf_adj
        # Directional travel (eastward worse than westward)
        if self.east_travel_penalty > 0 and not neutral_site and team_a_home is not None:
            venue_team = team_a if team_a_home else team_b
            ra += self.directional_travel_adjustment(team_a, venue_team)
            rb += self.directional_travel_adjustment(team_b, venue_team)
        # Series context (visiting team adapts in games 2/3)
        if self.series_adaptation > 0 and team_a_home is not None:
            ser_a, ser_b = self.series_context_adjustment(team_a, team_b, team_a_home)
            ra += ser_a
            rb += ser_b
        # Interleague adjustment (AL vs NL less predictable)
        il_a, il_b = self.interleague_adjustment(team_a, team_b)
        ra += il_a
        rb += il_b
        # Bullpen quality
        ra += self.bullpen_quality_adjustment(team_a)
        rb += self.bullpen_quality_adjustment(team_b)
        # Opposing pitcher quality (facing an ace hurts you)
        if self.opp_pitcher_factor > 0:
            # team_a is hurt by away_starter's quality, team_b by home_starter's
            h_sp = home_starter if team_a_home else away_starter
            a_sp = away_starter if team_a_home else home_starter
            ra += self.opp_pitcher_quality_adjustment(a_sp)  # A hurt by B's pitcher
            rb += self.opp_pitcher_quality_adjustment(h_sp)  # B hurt by A's pitcher
        # Injury adjustments (live predictions only)
        if use_injuries:
            from injuries import get_team_injuries, calc_injury_impact
            for team, rating_ref in ((team_a, "a"), (team_b, "b")):
                out = get_team_injuries(team)
                if out:
                    impact = calc_injury_impact(team, out)
                    if rating_ref == "a":
                        ra += impact
                    else:
                        rb += impact
        raw_p = self.expected_score(ra, rb)
        # XGBoost ensemble (if trained and available)
        if calibrated and self._xgb_model is not None and self._xgb_meta is not None:
            xgb_prob = self._xgb_predict(team_a, team_b, raw_p, ra - rb, game_date,
                                         home_starter, away_starter)
            if xgb_prob is not None:
                elo_w = self._xgb_meta.get("elo_weight", 0.8)
                raw_p = elo_w * raw_p + (1.0 - elo_w) * xgb_prob
        # Mega-ensemble adjustment (if trained and available)
        if calibrated and self._mega_predictor is not None:
            mega_adj = self._mega_predictor.predict(
                team_a, team_b, raw_p, ra - rb, game_date
            )
            raw_p = max(0.02, min(0.98, raw_p + mega_adj))
        if calibrated and self._platt_scaler is not None:
            return apply_platt(raw_p, self._platt_scaler)
        return raw_p

    def _xgb_predict(self, team_a, team_b, elo_prob, elo_diff, game_date,
                     home_starter="", away_starter=""):
        """Get XGBoost probability for a matchup using rolling stats."""
        try:
            import xgboost as xgb
            from enhanced_model import build_game_features
            scores_a = self._team_scores.get(team_a, [])
            scores_b = self._team_scores.get(team_b, [])
            if len(scores_a) < 3 or len(scores_b) < 3:
                return None

            def _rolling(scores, team):
                rf = [s[0] for s in scores[-10:]]
                ra = [s[1] for s in scores[-10:]]
                res = [1.0 if s[0] > s[1] else 0.0 for s in scores[-10:]]
                margins = [s[0] - s[1] for s in scores[-10:]]
                rd = self.rest_days(team, game_date)
                return {
                    "ppg": np.mean(rf), "papg": np.mean(ra),
                    "win_pct": np.mean(res), "avg_margin": np.mean(margins),
                    "off_rating": np.mean(rf), "def_rating": np.mean(ra),
                    "rest_days": rd if rd is not None else 1,
                    "games_played": len(rf),
                }

            home_feats = _rolling(scores_a, team_a)
            away_feats = _rolling(scores_b, team_b)
            rd_a = self.rest_days(team_a, game_date)
            rd_b = self.rest_days(team_b, game_date)
            home_feats["rest_days"] = rd_a if rd_a is not None else 1
            away_feats["rest_days"] = rd_b if rd_b is not None else 1

            player_diff = (self._player_scores.get(team_a, 0.0)
                           - self._player_scores.get(team_b, 0.0))
            h_pr = self._pitcher_ratings.get(home_starter, 0.0)
            a_pr = self._pitcher_ratings.get(away_starter, 0.0)
            fdict = build_game_features(home_feats, away_feats, elo_prob,
                                        elo_diff, player_diff,
                                        h_pr - a_pr, h_pr, a_pr)
            if fdict is None:
                return None
            feature_cols = self._xgb_meta.get("feature_cols")
            if not feature_cols:
                return None
            fvec = np.array([[fdict[c] for c in feature_cols]])
            dmat = xgb.DMatrix(fvec, feature_names=feature_cols)
            return float(self._xgb_model.predict(dmat)[0])
        except Exception as e:
            logging.debug("XGB predict failed: %s", e)
            return None

    def pick_winner(self, team_a, team_b, team_a_home=True, neutral_site=False,
                    game_date=None):
        if game_date is None:
            game_date = datetime.now()
        pa = self.win_prob(team_a, team_b, team_a_home, neutral_site,
                           game_date=game_date)
        return (team_a, pa) if pa >= 0.5 else (team_b, 1.0 - pa)

    def find_team(self, query):
        query = str(query).strip().lower()
        if not query:
            return None
        if query in self._team_lookup:
            return self._team_lookup[query]
        query_upper = query.upper()
        for full_name, abbr in TEAM_ABBR.items():
            if abbr == query_upper:
                found = self._team_lookup.get(full_name.lower())
                if found:
                    return found
                for key, name in self._team_lookup.items():
                    if any(part in key for part in full_name.lower().split() if len(part) > 3):
                        return name
        for key, name in self._team_lookup.items():
            if query in key:
                return name
        matches = get_close_matches(query, list(self._team_lookup.keys()), n=1, cutoff=0.6)
        if matches:
            return self._team_lookup[matches[0]]
        return None

    def save(self, filename=RATINGS_FILE):
        try:
            self.metadata["saved_at"] = current_timestamp()
            self.metadata["settings"] = self.settings_dict()
            payload = {"regular": dict(self.ratings), "metadata": self.metadata}
            with open(filename, "w") as f:
                json.dump(payload, f, indent=2)
            logging.info("Saved %d ratings -> %s", len(self.ratings), filename)
        except Exception as e:
            logging.warning("Save failed: %s", e)

    def load(self, filename=RATINGS_FILE):
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "regular" in data:
                self.ratings = defaultdict(lambda: self.base_rating, data.get("regular", {}))
                self.metadata.update(data.get("metadata", {}))
            else:
                self.ratings = defaultdict(lambda: self.base_rating, data)
            self.team_names = sorted(self.ratings.keys())
            self._rebuild_lookup()
            self._platt_scaler = load_platt_scaler()
            if self._platt_scaler:
                logging.info("Platt scaler active (n=%s samples)", self._platt_scaler.get("n_samples", "?"))
            logging.info("Loaded %d team ratings", len(self.team_names))
            return True
        except Exception as e:
            logging.warning("Load failed: %s", e)
            return False

    def show_settings(self):
        from config import load_elo_settings
        hdr("ELO SETTINGS")
        div()
        from enhanced_model import load_enhanced_model
        xgb_model, xgb_meta = load_enhanced_model()
        rows = [
            ("Base Rating",   self.base_rating),
            ("K Factor",      self.k),
            ("Home Adv",      self.home_adv),
            ("MOV Enabled",   self.use_mov),
            ("Player Boost",  self.player_boost),
            ("Starter Boost", self.starter_boost),
            ("Pitcher Ratings", "%d pitchers tracked" % len(self._pitcher_ratings)),
            ("Rest Factor",   self.rest_factor),
            ("Travel Factor", self.travel_factor),
            ("SOS Factor",    self.sos_factor),
            ("Playoff HFA",  "%.0f%% of normal" % (self.playoff_hca_factor * 100)),
            ("Pace Factor",   self.pace_factor),
            ("B2B Penalty",   self.b2b_penalty),
            ("Road Trip",     self.road_trip_factor),
            ("Homestand",     self.homestand_factor),
            ("Win Streak",    self.win_streak_factor),
            ("Altitude Fac",  self.altitude_factor if self.altitude_factor > 0 else "raw"),
            ("Season Phase",  self.season_phase_factor),
            ("Score Consist", self.scoring_consistency_factor),
            ("Rest Cap",      int(self.rest_advantage_cap) if self.rest_advantage_cap > 0 else "off"),
            ("Park Factor W", self.park_factor_weight if self.park_factor_weight > 0 else "off"),
            ("Altitude Bonus", ", ".join("%s +%.1f" % (t, b) for t, b in self._altitude_bonus.items()) if self._altitude_bonus else "none (no data)"),
            ("Player Scores", "%d teams loaded" % len(self._player_scores)),
            ("XGBoost",       "ACTIVE (Elo=80% XGB=20%)" if xgb_model else "not trained -- run 'enhanced'"),
            ("Mega-Ensemble", self._mega_predictor.get_status() if self._mega_predictor else "not trained -- run 'mega' first"),
            ("Auto-Resolve",  "ON" if load_elo_settings().get("autoresolve_enabled") else "OFF"),
        ]
        for label, val in rows:
            print("  %-16s: %s" % (chi(label), cok(val)))
        if self._platt_scaler:
            ps = self._platt_scaler
            platt_s = cok("ACTIVE") + cdim(
                " (n=%s, coef=%.3f, intercept=%.3f, fitted %s)"
                % (ps.get("n_samples", "?"), ps.get("coef", 0),
                   ps.get("intercept", 0), ps.get("fitted_at", "?"))
            )
        else:
            platt_s = cwarn("not fitted") + cdim(" -- run 'backtest' to fit")
        print("  %-16s: %s" % (chi("Platt Scaler"), platt_s))
        div()

    def show_all_teams(self):
        sorted_teams = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        hdr("ALL %d MLB TEAMS - Elo Ratings" % len(sorted_teams))
        div()
        for i, (team, rating) in enumerate(sorted_teams, 1):
            if rating >= 1550:
                rs = cok("%.1f" % rating)
            elif rating >= 1450:
                rs = cwarn("%.1f" % rating)
            else:
                rs = cred("%.1f" % rating)
            bar = cdim("#" * int((rating - 1400) / 10))
            print("  %s %-25s  %s  %s" % (cdim("%3d." % i), team, rs, bar))
        div()
