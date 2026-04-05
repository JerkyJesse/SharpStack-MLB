"""MLB Advanced Stats via pybaseball.

Pulls Statcast data, xwOBA, barrel rate, xERA, sprint speed.
All data is 100% free from Baseball Savant / FanGraphs (no API key).
"""

import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

STATCAST_CACHE_FILE = "mlb_statcast_cache.csv"
TEAM_STATS_FILE = "mlb_team_advanced.csv"
PITCHER_ADVANCED_FILE = "mlb_pitcher_advanced.csv"
CACHE_MAX_AGE_HOURS = 12


def _is_cache_stale(filepath, max_hours=CACHE_MAX_AGE_HOURS):
    try:
        import sys
        from cache_utils import smart_cache_stale
        return smart_cache_stale(filepath, "mlb", "advanced", max_age_hours=max_hours)
    except ImportError:
        pass
    if not os.path.exists(filepath):
        return True
    size = os.path.getsize(filepath)
    if size < 500:
        return True
    age = datetime.now().timestamp() - os.path.getmtime(filepath)
    return age > max_hours * 3600


# FanGraphs team abbreviation mapping to full names
FANGRAPHS_TO_FULL = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CHW": "Chicago White Sox", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KCR": "Kansas City Royals", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "ATH": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres", "SD": "San Diego Padres",
    "SFG": "San Francisco Giants", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals", "WSH": "Washington Nationals",
}


def download_team_batting_stats(season=None):
    """Download team-level batting stats from FanGraphs via pybaseball.

    Returns DataFrame with team batting metrics including advanced stats.
    """
    from pybaseball import team_batting

    if season is None:
        season = datetime.now().year
        if datetime.now().month < 4:
            season -= 1

    try:
        logging.info("Downloading team batting stats for %d...", season)
        df = team_batting(season)
        logging.info("Downloaded batting stats for %d teams", len(df))
        return df
    except Exception as e:
        logging.error("Failed to download team batting: %s", e)
        return None


def download_team_pitching_stats(season=None):
    """Download team-level pitching stats from FanGraphs via pybaseball."""
    from pybaseball import team_pitching

    if season is None:
        season = datetime.now().year
        if datetime.now().month < 4:
            season -= 1

    try:
        logging.info("Downloading team pitching stats for %d...", season)
        df = team_pitching(season)
        logging.info("Downloaded pitching stats for %d teams", len(df))
        return df
    except Exception as e:
        logging.error("Failed to download team pitching: %s", e)
        return None


def download_pitcher_stats(season=None, min_ip=20):
    """Download individual pitcher stats from FanGraphs.

    Returns DataFrame with pitcher-level advanced metrics.
    """
    from pybaseball import pitching_stats

    if season is None:
        season = datetime.now().year
        if datetime.now().month < 4:
            season -= 1

    try:
        logging.info("Downloading pitcher stats for %d (min %d IP)...", season, min_ip)
        df = pitching_stats(season, qual=min_ip)
        logging.info("Downloaded stats for %d pitchers", len(df))
        return df
    except Exception as e:
        logging.error("Failed to download pitcher stats: %s", e)
        return None


def compute_team_advanced(batting_df, pitching_df):
    """Compute team-level advanced stats from FanGraphs data.

    Returns dict: team_name -> {
        xwoba, barrel_rate, hard_hit_rate, k_rate, bb_rate,
        xera, pitch_k_rate, pitch_bb_rate, whip, fip,
        war_batting, war_pitching
    }
    """
    team_stats = {}

    if batting_df is not None and len(batting_df) > 0:
        for _, row in batting_df.iterrows():
            team_abbr = str(row.get("Team", row.get("team", ""))).strip()
            team_name = FANGRAPHS_TO_FULL.get(team_abbr, team_abbr)

            stats = {}

            # Batting advanced metrics
            for col, key in [
                ("wOBA", "woba"), ("xwOBA", "xwoba"),
                ("Barrel%", "barrel_rate"), ("HardHit%", "hard_hit_rate"),
                ("K%", "k_rate"), ("BB%", "bb_rate"),
                ("ISO", "iso"), ("BABIP", "babip"),
                ("wRC+", "wrc_plus"), ("WAR", "war_batting"),
                ("OPS", "ops"), ("SLG", "slg"), ("OBP", "obp"),
            ]:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        stats[key] = float(val)
                    except (ValueError, TypeError):
                        pass

            if stats:
                team_stats[team_name] = stats

    if pitching_df is not None and len(pitching_df) > 0:
        for _, row in pitching_df.iterrows():
            team_abbr = str(row.get("Team", row.get("team", ""))).strip()
            team_name = FANGRAPHS_TO_FULL.get(team_abbr, team_abbr)

            if team_name not in team_stats:
                team_stats[team_name] = {}

            for col, key in [
                ("ERA", "era"), ("xERA", "xera"), ("FIP", "fip"),
                ("xFIP", "xfip"), ("SIERA", "siera"),
                ("K%", "pitch_k_rate"), ("BB%", "pitch_bb_rate"),
                ("WHIP", "whip"), ("WAR", "war_pitching"),
                ("HR/9", "hr_per_9"), ("K/9", "k_per_9"),
                ("LOB%", "lob_pct"),
            ]:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        team_stats[team_name]["pitch_" + key if not key.startswith("pitch_") and key not in ("era", "xera", "fip", "xfip", "siera", "whip", "war_pitching", "hr_per_9", "k_per_9", "lob_pct") else key] = float(val)
                    except (ValueError, TypeError):
                        pass

    return team_stats


def get_starting_pitcher_stats(pitcher_name, pitcher_stats_df):
    """Look up a starting pitcher's advanced stats.

    Returns dict with xERA, FIP, K%, BB%, xFIP, SIERA if available.
    """
    if pitcher_stats_df is None or len(pitcher_stats_df) == 0:
        return {}

    # Fuzzy match on name
    name_lower = pitcher_name.lower().strip()
    for _, row in pitcher_stats_df.iterrows():
        p_name = str(row.get("Name", row.get("name", ""))).lower().strip()
        if name_lower in p_name or p_name in name_lower:
            stats = {}
            for col, key in [
                ("ERA", "era"), ("xERA", "xera"), ("FIP", "fip"),
                ("xFIP", "xfip"), ("SIERA", "siera"),
                ("K%", "k_rate"), ("BB%", "bb_rate"),
                ("WHIP", "whip"), ("IP", "innings"),
            ]:
                val = row.get(col)
                if val is not None:
                    try:
                        stats[key] = float(val)
                    except (ValueError, TypeError):
                        pass
            return stats

    return {}


def get_advanced_features(home_team, away_team, team_stats):
    """Get advanced stat features for a matchup.

    Returns dict of features suitable for the meta-learner.
    """
    home = team_stats.get(home_team, {})
    away = team_stats.get(away_team, {})

    if not home or not away:
        return {}

    features = {
        # Batting differentials
        "woba_diff": home.get("woba", 0.320) - away.get("woba", 0.320),
        "barrel_diff": home.get("barrel_rate", 0.06) - away.get("barrel_rate", 0.06),
        "hard_hit_diff": home.get("hard_hit_rate", 0.35) - away.get("hard_hit_rate", 0.35),
        "iso_diff": home.get("iso", 0.150) - away.get("iso", 0.150),

        # Pitching differentials (lower ERA = better, so away - home)
        "era_diff": away.get("era", 4.00) - home.get("era", 4.00),
        "fip_diff": away.get("fip", 4.00) - home.get("fip", 4.00),
        "whip_diff": away.get("whip", 1.30) - home.get("whip", 1.30),

        # K/BB differentials
        "k_rate_diff": home.get("k_rate", 0.22) - away.get("k_rate", 0.22),
        "pitch_k_diff": home.get("k_per_9", 8.5) - away.get("k_per_9", 8.5),

        # WAR
        "war_bat_diff": home.get("war_batting", 0) - away.get("war_batting", 0),
        "war_pitch_diff": home.get("war_pitching", 0) - away.get("war_pitching", 0),

        # Raw values for context
        "home_woba": home.get("woba", 0.320),
        "away_woba": away.get("woba", 0.320),
        "home_era": home.get("era", 4.00),
        "away_era": away.get("era", 4.00),
    }

    return features


def load_or_download_advanced(force=False):
    """Load cached advanced stats or download fresh data.

    Returns (team_stats_dict, pitcher_stats_df).
    """
    if not force and not _is_cache_stale(TEAM_STATS_FILE):
        try:
            df = pd.read_csv(TEAM_STATS_FILE)
            # Reconstruct dict from CSV
            team_stats = {}
            for _, row in df.iterrows():
                team = row.get("team", "")
                if team:
                    stats = {k: v for k, v in row.items() if k != "team" and not pd.isna(v)}
                    team_stats[team] = stats

            pitcher_df = None
            if not _is_cache_stale(PITCHER_ADVANCED_FILE):
                pitcher_df = pd.read_csv(PITCHER_ADVANCED_FILE)

            logging.info("Loaded cached advanced stats (%d teams)", len(team_stats))
            return team_stats, pitcher_df
        except Exception:
            pass

    # Download fresh
    batting = download_team_batting_stats()
    pitching = download_team_pitching_stats()
    pitcher_df = download_pitcher_stats()

    team_stats = compute_team_advanced(batting, pitching)

    # Cache results
    if team_stats:
        rows = []
        for team, stats in team_stats.items():
            row = {"team": team}
            row.update(stats)
            rows.append(row)
        pd.DataFrame(rows).to_csv(TEAM_STATS_FILE, index=False)

    if pitcher_df is not None and len(pitcher_df) > 0:
        pitcher_df.to_csv(PITCHER_ADVANCED_FILE, index=False)

    return team_stats, pitcher_df


def show_team_rankings(team_stats):
    """Display team advanced stat rankings."""
    if not team_stats:
        print("  No advanced stats available. Run 'advstats' to download.")
        return

    # Sort by wOBA
    sorted_teams = sorted(team_stats.items(),
                          key=lambda x: x[1].get("woba", 0), reverse=True)

    print(f"\n  {'Rank':>4} {'Team':<25} {'wOBA':>6} {'Barrel%':>8} {'ERA':>6} {'FIP':>6} {'WHIP':>6} {'WAR(B)':>7} {'WAR(P)':>7}")
    print("  " + "-" * 82)

    for i, (team, stats) in enumerate(sorted_teams, 1):
        print(f"  {i:>4} {team:<25} "
              f"{stats.get('woba', 0):.3f}  "
              f"{stats.get('barrel_rate', 0):>7.1%}  "
              f"{stats.get('era', 0):>5.2f}  "
              f"{stats.get('fip', 0):>5.2f}  "
              f"{stats.get('whip', 0):>5.2f}  "
              f"{stats.get('war_batting', 0):>6.1f}  "
              f"{stats.get('war_pitching', 0):>6.1f}")
