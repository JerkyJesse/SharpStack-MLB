"""Smart caching utilities for API data.

Principle: Only call an API if the data is likely to have changed.
- Game scores: only change when games are being played
- Player stats: only change during active season, weekly at most
- Injuries: change daily during season, irrelevant in offseason
- Odds: change by the minute on game days, stale otherwise
- Weather: only matters for games today/tomorrow
- Sentiment: changes slowly, 4-6 hour cache is fine
- Advanced stats (EPA/Statcast): updated weekly during season

This module provides sport-aware cache staleness checks.
"""

import os
import json
import logging
from datetime import datetime, timedelta

# ── Season calendars ───────────────────────────────────────────────
# (start_month, end_month) — when games are actively being played
SEASON_MONTHS = {
    "nfl": [(9, 12), (1, 2)],    # Sep-Feb (cross-year)
    "mlb": [(3, 10)],            # Mar-Oct
    "nba": [(10, 12), (1, 6)],   # Oct-Jun (cross-year)
    "nhl": [(10, 12), (1, 6)],   # Oct-Jun (cross-year)
}

# Days of week when games are commonly played
GAME_DAYS = {
    "nfl": [0, 3, 6],       # Mon, Thu, Sun
    "mlb": [0, 1, 2, 3, 4, 5, 6],  # Every day
    "nba": [0, 1, 2, 3, 4, 5, 6],  # Most days
    "nhl": [0, 1, 2, 3, 4, 5, 6],  # Most days
}


def is_in_season(sport, dt=None):
    """Check if the given date is during the sport's active season."""
    if dt is None:
        dt = datetime.now()
    month = dt.month
    ranges = SEASON_MONTHS.get(sport.lower(), [(1, 12)])
    for start, end in ranges:
        if start <= end:
            if start <= month <= end:
                return True
        else:  # Cross-year range like (9, 2)
            if month >= start or month <= end:
                return True
    return False


def is_game_day(sport, dt=None):
    """Check if today is a typical game day for this sport."""
    if dt is None:
        dt = datetime.now()
    return dt.weekday() in GAME_DAYS.get(sport.lower(), list(range(7)))


def smart_cache_stale(filepath, sport, data_type="games",
                       min_age_hours=None, max_age_hours=None):
    """Smart staleness check that adapts to sport calendar.

    Args:
        filepath: path to cached data file
        sport: 'nfl', 'mlb', 'nba', 'nhl'
        data_type: what kind of data (affects staleness rules)
            'games' - game scores (stale fast on game days, slow otherwise)
            'players' - player stats (stale weekly during season)
            'injuries' - injury reports (stale every 4h during season)
            'odds' - betting odds (stale every 15min on game days)
            'weather' - weather forecasts (stale every 2h on game days)
            'advanced' - advanced stats (stale every 24h during season)
        min_age_hours: override minimum age before considering stale
        max_age_hours: override maximum age (always stale after this)

    Returns True if data should be re-fetched.
    """
    # File doesn't exist or is too small
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 100:
        return True

    file_age_hours = (datetime.now().timestamp() - os.path.getmtime(filepath)) / 3600
    in_season = is_in_season(sport)
    game_day = is_game_day(sport)

    # ── Data-type specific rules ──

    if data_type == "games":
        if max_age_hours is not None:
            return file_age_hours > max_age_hours
        if not in_season:
            return file_age_hours > 168  # 1 week in offseason
        if game_day:
            return file_age_hours > 4    # 4h on game days
        return file_age_hours > 12       # 12h on non-game days

    elif data_type == "players":
        if max_age_hours is not None:
            return file_age_hours > max_age_hours
        if not in_season:
            return file_age_hours > 720  # 30 days in offseason
        return file_age_hours > 48       # 2 days during season (stats update ~daily)

    elif data_type == "injuries":
        if not in_season:
            return file_age_hours > 720  # 30 days in offseason
        if game_day:
            return file_age_hours > 2    # 2h on game days (lineups change)
        return file_age_hours > 6        # 6h otherwise

    elif data_type == "odds":
        if not in_season:
            return False  # Never fetch odds in offseason
        if game_day:
            return file_age_hours > 0.25  # 15 minutes on game days
        return file_age_hours > 4         # 4h otherwise

    elif data_type == "weather":
        if not in_season:
            return False  # Never fetch weather in offseason
        if game_day:
            return file_age_hours > 2     # 2h on game days
        return file_age_hours > 12        # 12h otherwise

    elif data_type == "sentiment":
        if not in_season:
            return file_age_hours > 168   # 1 week in offseason
        return file_age_hours > 4         # 4h during season

    elif data_type == "advanced":
        if not in_season:
            return file_age_hours > 720   # 30 days in offseason
        return file_age_hours > 24        # 24h during season (weekly data)

    # Default: 6 hours
    return file_age_hours > (max_age_hours or 6)


def get_cache_age_str(filepath):
    """Get human-readable age of a cache file."""
    if not os.path.exists(filepath):
        return "never cached"
    age_seconds = datetime.now().timestamp() - os.path.getmtime(filepath)
    if age_seconds < 60:
        return "%ds ago" % age_seconds
    if age_seconds < 3600:
        return "%dm ago" % (age_seconds / 60)
    if age_seconds < 86400:
        return "%.1fh ago" % (age_seconds / 3600)
    return "%.1fd ago" % (age_seconds / 86400)
