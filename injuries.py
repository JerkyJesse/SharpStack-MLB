"""MLB injury report via ESPN public API + impact scoring."""

import os
import json
import logging
from datetime import datetime

import requests
import pandas as pd

from config import current_timestamp, get_team_abbr, TEAM_ABBR
from data_players import load_player_stats, load_advanced_stats

INJURY_CACHE_FILE = "mlb_injuries.json"
INJURY_CACHE_MAX_HOURS = 4
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries"

# ESPN team names that differ from ours
ESPN_TEAM_MAP = {
}


def fetch_injury_report():
    """
    Fetch MLB injury report from ESPN public API.
    Returns list of {player, team, status, detail}.
    Uses cache if fresh enough.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    injuries = _fetch_from_espn()
    if injuries:
        _save_cache(injuries)
        return injuries

    return []


def _fetch_from_espn():
    """Fetch structured injury data from ESPN's public JSON API."""
    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning("ESPN injury fetch failed: %s", e)
        return None

    injuries = []
    for team_block in data.get("injuries", []):
        espn_team = team_block.get("displayName", "")
        team_name = ESPN_TEAM_MAP.get(espn_team, espn_team)

        for entry in team_block.get("injuries", []):
            status = entry.get("status", "")
            athlete = entry.get("athlete", {})
            player = athlete.get("displayName", "")
            detail = entry.get("shortComment", "")

            if player and status:
                injuries.append({
                    "player": player,
                    "team": team_name,
                    "status": status,
                    "detail": detail,
                })

    logging.info("ESPN injuries: %d entries across %d teams",
                 len(injuries), len(data.get("injuries", [])))
    return injuries


def _save_cache(injuries):
    cache = {"fetched_at": current_timestamp(), "injuries": injuries}
    try:
        with open(INJURY_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logging.warning("Injury cache save failed: %s", e)


def _load_cache():
    if not os.path.exists(INJURY_CACHE_FILE):
        return None
    try:
        with open(INJURY_CACHE_FILE, "r") as f:
            cache = json.load(f)
        fetched = datetime.strptime(cache["fetched_at"], "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - fetched).total_seconds() / 3600
        if age > INJURY_CACHE_MAX_HOURS:
            return None
        return cache.get("injuries", [])
    except Exception as e:
        logging.debug("Injury cache load failed: %s", e)
        return None


def manual_set_injuries(team_name, out_players):
    """Manually mark players as out. Merges with existing cache."""
    injuries = _load_cache() or []
    for player in out_players:
        injuries.append({
            "player": player.strip(),
            "team": team_name,
            "status": "Out",
            "detail": "manual",
        })
    _save_cache(injuries)
    return injuries


def get_team_injuries(team_name, injuries=None):
    """Get list of OUT/DOUBTFUL players for a team."""
    if injuries is None:
        injuries = fetch_injury_report()
    if not injuries:
        return []
    abbr = get_team_abbr(team_name)
    team_lower = team_name.lower()
    out = []
    for inj in injuries:
        inj_team = str(inj.get("team", "")).strip()
        status = str(inj.get("status", "")).strip()
        if status not in ("Out", "Doubtful", "60-Day IL", "15-Day IL", "10-Day IL"):
            continue
        if (inj_team.upper() == abbr or
                inj_team.lower() == team_lower or
                team_lower in inj_team.lower() or
                inj_team.lower() in team_lower):
            out.append(inj["player"])
    return out


def calc_injury_impact(team_name, out_players):
    """
    Calculate Elo adjustment for missing players.
    Returns negative value (penalty) for the injured team.
    """
    if not out_players:
        return 0.0

    player_df = load_player_stats()
    if player_df.empty:
        return -12.0 * len(out_players)

    abbr = get_team_abbr(team_name)
    team_players = player_df[player_df["Tm"] == abbr].copy()
    if team_players.empty:
        return -12.0 * len(out_players)

    adv_df = load_advanced_stats()

    total_impact = 0.0
    for player_name in out_players:
        matches = team_players[team_players["Player"].str.contains(
            player_name, case=False, na=False)]
        if matches.empty:
            for part in player_name.split():
                if len(part) > 3:
                    matches = team_players[team_players["Player"].str.contains(
                        part, case=False, na=False)]
                    if not matches.empty:
                        break
        if matches.empty:
            total_impact -= 8.0
            continue

        row = matches.iloc[0]
        hr = float(row.get("HR", 0) or 0)
        rbi = float(row.get("RBI", 0) or 0)

        # Impact based on offensive production (HR + RBI proxy)
        if hr >= 30:
            impact = -30.0
        elif hr >= 20:
            impact = -22.0
        elif hr >= 10:
            impact = -15.0
        elif rbi >= 50:
            impact = -12.0
        else:
            impact = -6.0

        # Check if player is also a pitcher (from advanced stats)
        if not adv_df.empty:
            adv_match = adv_df[adv_df["Player"].str.contains(
                player_name, case=False, na=False)]
            if not adv_match.empty:
                era = float(adv_match.iloc[0].get("ERA", 0) or 0)
                wins = float(adv_match.iloc[0].get("W", 0) or 0)
                if era > 0 and era < 3.50:
                    impact = -35.0  # Ace pitcher
                elif era > 0 and era < 4.00:
                    impact = -25.0  # Quality starter

        total_impact += impact

    return total_impact


def show_injury_report(model=None):
    """Display current injury report with impact estimates."""
    from color_helpers import cok, cwarn, cdim, chi, div, cred

    injuries = fetch_injury_report()
    if not injuries:
        print(cwarn("  No injury data available."))
        return

    by_team = {}
    for inj in injuries:
        team = inj.get("team", "Unknown")
        if team not in by_team:
            by_team[team] = []
        by_team[team].append(inj)

    div(70)
    print("  %s  (%d players listed)" % (chi("MLB INJURY REPORT"), len(injuries)))
    div(70)
    for team in sorted(by_team.keys()):
        players = by_team[team]
        out_names = [p["player"] for p in players if p.get("status") in ("Out", "Doubtful", "60-Day IL", "15-Day IL", "10-Day IL")]
        impact = calc_injury_impact(team, out_names) if out_names else 0
        impact_s = cred("%.0f Elo" % impact) if impact < -20 else cwarn("%.0f Elo" % impact) if impact < 0 else ""
        print("  %s %s" % (chi(team), impact_s))
        for p in players:
            status = p.get("status", "?")
            if status in ("Out", "60-Day IL"):
                ss = cred(status)
            elif status in ("Doubtful", "15-Day IL", "10-Day IL"):
                ss = cwarn(status)
            else:
                ss = cdim(status)
            print("    %-25s %s" % (p["player"], ss))
    div(70)
