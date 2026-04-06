"""Constants, team abbreviations, settings I/O, and shared helpers."""

import os
import sys
import json
import logging
from datetime import datetime

try:
    from cache_utils import smart_cache_stale, get_cache_age_str
    HAS_SMART_CACHE = True
except ImportError:
    HAS_SMART_CACHE = False

SETTINGS_FILE          = "mlb_elo_settings.json"
PREDICTS_FILE = "predicts_lots.csv"
PLAYER_STATS_FILE      = "mlb_player_stats.csv"
GAMES_FILE             = "mlb_recent_games.csv"
RATINGS_FILE           = "mlb_elo_ratings.json"
ADVANCED_STATS_FILE    = "mlb_advanced_stats.csv"
PLATT_SCALER_FILE      = "mlb_platt_scaler.json"
CACHE_MAX_AGE_HOURS    = 6

TEAM_ABBR = {
    "Arizona Diamondbacks": "AZ",    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",      "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",           "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",        "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",       "Detroit Tigers": "DET",
    "Houston Astros": "HOU",         "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",     "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",          "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",        "New York Mets": "NYM",
    "New York Yankees": "NYY",       "Oakland Athletics": "ATH",
    "Athletics": "ATH",              "Sacramento Athletics": "ATH",
    "Philadelphia Phillies": "PHI",  "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",        "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",       "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",          "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",      "Washington Nationals": "WSH",
}

MLB_DIVISIONS = {
    "AL East": ["Baltimore Orioles", "Boston Red Sox", "New York Yankees", "Tampa Bay Rays", "Toronto Blue Jays"],
    "AL Central": ["Chicago White Sox", "Cleveland Guardians", "Detroit Tigers", "Kansas City Royals", "Minnesota Twins"],
    "AL West": ["Houston Astros", "Los Angeles Angels", "Oakland Athletics", "Seattle Mariners", "Texas Rangers"],
    "NL East": ["Atlanta Braves", "Miami Marlins", "New York Mets", "Philadelphia Phillies", "Washington Nationals"],
    "NL Central": ["Chicago Cubs", "Cincinnati Reds", "Milwaukee Brewers", "Pittsburgh Pirates", "St. Louis Cardinals"],
    "NL West": ["Arizona Diamondbacks", "Colorado Rockies", "Los Angeles Dodgers", "San Diego Padres", "San Francisco Giants"],
}


def same_division(team_a, team_b):
    for div_teams in MLB_DIVISIONS.values():
        if team_a in div_teams and team_b in div_teams:
            return True
    return False


def get_league(team):
    """Return 'AL' or 'NL' for a team, or '' if unknown."""
    for div_name, div_teams in MLB_DIVISIONS.items():
        if team in div_teams:
            return "AL" if div_name.startswith("AL") else "NL"
    return ""


def build_league_map():
    """Return dict of team_name -> 'AL'/'NL' for all teams."""
    league_map = {}
    for div_name, div_teams in MLB_DIVISIONS.items():
        league = "AL" if div_name.startswith("AL") else "NL"
        for team in div_teams:
            league_map[team] = league
    return league_map


def get_team_abbr(full_name):
    return TEAM_ABBR.get(str(full_name).strip(), str(full_name)[:3].upper())


def get_current_season_year():
    """MLB season runs within a single calendar year (April-October)."""
    return datetime.now().year


def get_season_label(year=None):
    year = year or get_current_season_year()
    return str(year)


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_cache_stale(filepath, max_age_hours=CACHE_MAX_AGE_HOURS, data_type="games"):
    """Check if cached data needs refreshing. Uses smart season-aware logic when available."""
    if HAS_SMART_CACHE:
        return smart_cache_stale(filepath, "mlb", data_type, max_age_hours=max_age_hours)
    # Fallback: simple age check
    # Files under 500 bytes are empty/corrupt stubs (valid CSV needs headers + rows)
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 500:
        return True
    age_hours = (datetime.now().timestamp() - os.path.getmtime(filepath)) / 3600
    return age_hours > max_age_hours


def load_elo_settings(filename=SETTINGS_FILE):
    defaults = {
        "base_rating": 1500.0, "k": 2.62, "home_adv": 37.63,
        "use_mov": True, "autoresolve_enabled": False,
        "player_boost": 19.53, "starter_boost": 88.08,
        "rest_factor": 2.80, "form_weight": 7.08,
        "travel_factor": 13.71, "sos_factor": 2.45,
        "playoff_hca_factor": 0.82, "pace_factor": 41.87,
        "division_factor": 6.64, "mean_reversion": 34.23,
        "pyth_factor": 16.0, "home_road_factor": 4.04, "mov_base": 0.3,
        "season_regress": 0.33,
        "b2b_penalty": 26.51, "road_trip_factor": 0.0,
        "homestand_factor": 1.41, "win_streak_factor": 0.0,
        "altitude_factor": 12.48, "season_phase_factor": 9.35,
        "scoring_consistency_factor": 0.0, "rest_advantage_cap": 4.14,
        "park_factor_weight": 0.0,
        "mov_cap": 19.9, "east_travel_penalty": 0.0,
        "series_adaptation": 3.92, "interleague_factor": 2.04,
        "bullpen_factor": 6.43, "opp_pitcher_factor": 18.0,
        "k_decay": 2.07, "surprise_k": 0.0,
        "elo_scale": 400.0,
        "starting_balance": 125.0, "kelly_fraction": 0.50,
        "auto_kalshi": False,
    }
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
            logging.info("Loaded settings: K=%s, HomeAdv=%s", defaults["k"], defaults["home_adv"])
        except Exception as e:
            logging.warning("Settings load failed: %s", e)
    else:
        logging.info("Using default settings")
    return defaults


def save_elo_settings(settings, filename=SETTINGS_FILE):
    try:
        with open(filename, "w") as f:
            json.dump(settings, f, indent=2)
        logging.info("Saved settings -> %s", filename)
    except Exception as e:
        logging.warning("Save failed: %s", e)
