"""Player stats download, loading, scoring, and display for MLB."""

import os
import logging

import numpy as np
import pandas as pd

from config import (
    PLAYER_STATS_FILE, ADVANCED_STATS_FILE, TEAM_ABBR,
    get_season_label, get_team_abbr, is_cache_stale,
)
from color_helpers import cok, cwarn, cdim, cbold, chi, cyel, cblu, cgrn


def download_player_stats(csv_file=PLAYER_STATS_FILE):
    if not is_cache_stale(csv_file, data_type="players"):
        logging.info("Using cached player stats %s", csv_file)
        return csv_file
    logging.info("Downloading MLB player stats via statsapi...")
    try:
        import statsapi

        season = get_season_label()
        all_players = []

        # Build team ID -> abbreviation lookup
        teams_data = statsapi.get("teams", {"sportId": 1})
        team_id_to_abbr = {}
        for team in teams_data.get("teams", []):
            team_id_to_abbr[team["id"]] = team.get("abbreviation", "")

        # Try current season first; fall back to prior year if data is thin
        seasons_to_try = [season]
        from datetime import datetime as _dt
        if _dt.now().month <= 4:
            seasons_to_try.append(str(int(season) - 1))

        # Fetch top hitters via stats leaders
        try:
            for try_season in seasons_to_try:
                if all_players and len(set(p["Tm"] for p in all_players)) >= 20:
                    break
                all_players = []
                for stat_type in ["battingAverage", "homeRuns", "runsBattedIn"]:
                    leaders = statsapi.get(
                        "stats_leaders",
                        {"leaderCategories": stat_type, "season": try_season,
                         "sportId": 1, "limit": 200}
                    )
                    for cat in leaders.get("leagueLeaders", []):
                        for leader in cat.get("leaders", []):
                            person = leader.get("person", {})
                            team_info = leader.get("team", {})
                            player_name = person.get("fullName", "")
                            team_id = team_info.get("id", 0)
                            team_abbr = team_id_to_abbr.get(team_id, team_info.get("abbreviation", ""))
                            stat_val = leader.get("value", "0")
                            found = False
                            for p in all_players:
                                if p["Player"] == player_name and p["Tm"] == team_abbr:
                                    if stat_type == "battingAverage":
                                        p["AVG"] = float(stat_val)
                                    elif stat_type == "homeRuns":
                                        p["HR"] = float(stat_val)
                                    elif stat_type == "runsBattedIn":
                                        p["RBI"] = float(stat_val)
                                    found = True
                                    break
                            if not found:
                                entry = {"Player": player_name, "Tm": team_abbr,
                                         "G": 0, "AVG": 0.0, "OBP": 0.0, "SLG": 0.0,
                                         "HR": 0, "RBI": 0, "R": 0, "SB": 0, "OPS": 0.0}
                                if stat_type == "battingAverage":
                                    entry["AVG"] = float(stat_val)
                                elif stat_type == "homeRuns":
                                    entry["HR"] = float(stat_val)
                                elif stat_type == "runsBattedIn":
                                    entry["RBI"] = float(stat_val)
                                all_players.append(entry)
                if all_players:
                    logging.info("Using season %s batting data (%d players)", try_season, len(all_players))
        except Exception as e:
            logging.warning("Stats leaders fetch failed: %s", e)

        if not all_players:
            # Minimal fallback using schedule/standings data
            logging.warning("Could not fetch individual player stats")
            return csv_file if os.path.exists(csv_file) else None

        df = pd.DataFrame(all_players)
        # OPS: use OBP+SLG if available, else estimate from AVG+HR
        if "OBP" in df.columns and "SLG" in df.columns:
            obp = pd.to_numeric(df["OBP"], errors="coerce").fillna(0)
            slg = pd.to_numeric(df["SLG"], errors="coerce").fillna(0)
            ops = obp + slg
            # If SLG is mostly 0 (not returned by API), estimate from AVG+HR
            if (slg > 0).sum() < len(slg) * 0.3:
                avg = pd.to_numeric(df["AVG"], errors="coerce").fillna(0.250)
                hr = pd.to_numeric(df["HR"], errors="coerce").fillna(0)
                # Rough OPS estimate: OBP ≈ AVG+0.06, SLG ≈ AVG+0.17*(HR/20)
                est_obp = avg + 0.06
                est_slg = avg + 0.17 * (hr / 20.0).clip(0, 1.5)
                ops = est_obp + est_slg
            df["OPS"] = ops
        else:
            avg = pd.to_numeric(df["AVG"], errors="coerce").fillna(0.250)
            hr = pd.to_numeric(df["HR"], errors="coerce").fillna(0)
            df["OPS"] = (avg + 0.06) + (avg + 0.17 * (hr / 20.0).clip(0, 1.5))
        for c in ["G", "AVG", "OBP", "SLG", "HR", "RBI", "R", "SB", "OPS"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Player", "Tm"])
        df.to_csv(csv_file, index=False)
        logging.info("Downloaded %d players -> %s", len(df), csv_file)
        return csv_file
    except Exception as e:
        logging.warning("MLB player stats failed (%s).", e)
        return csv_file if os.path.exists(csv_file) else None


def download_advanced_stats(csv_file=ADVANCED_STATS_FILE):
    """Download pitching stats as advanced stats for MLB."""
    if not is_cache_stale(csv_file, data_type="players"):
        logging.info("Using cached advanced stats %s", csv_file)
        return csv_file
    logging.info("Downloading MLB pitching stats via statsapi...")
    try:
        import statsapi

        season = get_season_label()
        all_pitchers = []

        # Build team ID -> abbreviation lookup
        teams_data = statsapi.get("teams", {"sportId": 1})
        team_id_to_abbr = {}
        for team in teams_data.get("teams", []):
            team_id_to_abbr[team["id"]] = team.get("abbreviation", "")

        # Try current season first; fall back to prior year if data is thin
        from datetime import datetime as _dt
        seasons_to_try = [season]
        if _dt.now().month <= 4:
            seasons_to_try.append(str(int(season) - 1))

        for try_season in seasons_to_try:
            if all_pitchers and len(set(p["Tm"] for p in all_pitchers)) >= 20:
                break
            all_pitchers = []
            for stat_type in ["earnedRunAverage", "strikeouts", "wins"]:
                try:
                    leaders = statsapi.get(
                        "stats_leaders",
                        {"leaderCategories": stat_type, "season": try_season,
                         "sportId": 1, "limit": 150}
                    )
                    for cat in leaders.get("leagueLeaders", []):
                        for leader in cat.get("leaders", []):
                            person = leader.get("person", {})
                            team_info = leader.get("team", {})
                            player_name = person.get("fullName", "")
                            team_id = team_info.get("id", 0)
                            team_abbr = team_id_to_abbr.get(team_id, team_info.get("abbreviation", ""))
                            stat_val = leader.get("value", "0")
                            found = False
                            for p in all_pitchers:
                                if p["Player"] == player_name and p["Tm"] == team_abbr:
                                    if stat_type == "earnedRunAverage":
                                        p["ERA"] = float(stat_val)
                                    elif stat_type == "strikeouts":
                                        p["K"] = float(stat_val)
                                    elif stat_type == "wins":
                                        p["W"] = float(stat_val)
                                    found = True
                                    break
                            if not found:
                                entry = {"Player": player_name, "Tm": team_abbr,
                                         "G": 0, "ERA": 0.0, "WHIP": 0.0,
                                         "K": 0, "W": 0, "IP": 0.0}
                                if stat_type == "earnedRunAverage":
                                    entry["ERA"] = float(stat_val)
                                elif stat_type == "strikeouts":
                                    entry["K"] = float(stat_val)
                                elif stat_type == "wins":
                                    entry["W"] = float(stat_val)
                                all_pitchers.append(entry)
                except Exception:
                    pass
            if all_pitchers:
                logging.info("Using season %s pitching data (%d pitchers)", try_season, len(all_pitchers))

        if all_pitchers:
            df = pd.DataFrame(all_pitchers)
            for c in ["G", "ERA", "WHIP", "K", "W", "IP"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Player", "Tm"])
            df.to_csv(csv_file, index=False)
            logging.info("Downloaded %d pitchers -> %s", len(df), csv_file)
            return csv_file
        return csv_file if os.path.exists(csv_file) else None
    except Exception as e:
        logging.warning("Advanced stats download failed (%s).", e)
        return csv_file if os.path.exists(csv_file) else None


def load_player_stats(filename=PLAYER_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    for c in ["HR", "RBI", "AVG", "OPS", "R", "SB"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_advanced_stats(filename=ADVANCED_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    for c in ["ERA", "WHIP", "K", "W", "IP"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


FANGRAPHS_PITCHING_FILE = "mlb_fangraphs_pitching.csv"


def load_fangraphs_pitching(filename=FANGRAPHS_PITCHING_FILE):
    """Load FanGraphs pitching data (FIP, xFIP, SIERA) for all seasons."""
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    for c in ["ERA", "FIP", "xFIP", "SIERA", "WAR", "IP", "K/9", "BB/9", "K%", "BB%", "K-BB%"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def calc_team_player_score(player_df, team_full_name, top_n=10):
    """Calculate a composite team strength score from batting stats."""
    if player_df is None or player_df.empty:
        return 0.0
    abbr = get_team_abbr(team_full_name)
    team_df = player_df[player_df["Tm"] == abbr].copy()
    if team_df.empty:
        return 0.0
    for c in ["HR", "RBI", "AVG", "OPS", "R", "SB"]:
        if c not in team_df.columns:
            team_df[c] = 0.0
        team_df[c] = pd.to_numeric(team_df[c], errors="coerce").fillna(0.0)
    # Score top batters by composite (not HR alone)
    team_df["_bat_composite"] = (
        team_df["HR"] * 2.0 + team_df["RBI"] * 1.0 + team_df["AVG"] * 100.0
    )
    top = team_df.nlargest(top_n, "_bat_composite")
    batting_score = (
        top["HR"] * 2.0 + top["RBI"] * 1.0 + top["R"] * 0.5
        + top["SB"] * 0.5 + top["AVG"] * 100.0
    ).sum()
    # Incorporate pitching if available
    adv_df = load_advanced_stats()
    if adv_df.empty:
        return float(batting_score)
    team_pitch = adv_df[adv_df["Tm"] == abbr].copy()
    if team_pitch.empty:
        return float(batting_score)
    # Lower ERA is better, more K is better
    for c in ["ERA", "K", "W"]:
        team_pitch[c] = pd.to_numeric(team_pitch.get(c), errors="coerce").fillna(0.0)
    # Sort pitchers by composite (ERA-inverse + K), take top 8
    team_pitch["_pitch_composite"] = (4.50 - team_pitch["ERA"].clip(0, 10)) * 5.0 + team_pitch["K"] * 0.5
    top_pitch = team_pitch.nlargest(min(8, len(team_pitch)), "_pitch_composite")
    # Invert ERA (lower = better): use 4.50 as league avg
    pitch_score = (
        (4.50 - top_pitch["ERA"].clip(0, 10)) * 10.0
        + top_pitch["K"] * 0.5
        + top_pitch["W"] * 3.0
    ).sum()
    # Scale pitching up so magnitudes are comparable before blending
    if len(top) > 0 and len(top_pitch) > 0:
        bat_per = batting_score / max(len(top), 1)
        pitch_per = pitch_score / max(len(top_pitch), 1)
        if pitch_per > 0:
            scale = bat_per / pitch_per
            pitch_score *= scale
    return float(batting_score * 0.45 + pitch_score * 0.55)


def build_league_player_scores(player_df):
    if player_df is None or player_df.empty:
        return {}
    teams = list(TEAM_ABBR.keys())
    raw = {t: calc_team_player_score(player_df, t) for t in teams}
    values = [v for v in raw.values() if v != 0.0]
    if not values:
        return {}
    mean = float(np.mean(values))
    std = float(np.std(values)) or 1.0
    return {t: (v - mean) / std for t, v in raw.items()}


def show_player_metrics(player_df, team=None, top_n=5):
    if player_df.empty:
        print("   " + cwarn("No player stats loaded."))
        return
    adv_df = load_advanced_stats()
    if team:
        abbr = get_team_abbr(team)
        df_sub = player_df[player_df["Tm"] == abbr].copy()
        if df_sub.empty:
            print("   " + cwarn("No stats found for %s (%s)" % (team, abbr)))
            return
    else:
        df_sub = player_df.copy()

    # Identify pitchers from advanced stats (have ERA > 0)
    pitcher_names = set()
    if not adv_df.empty:
        pitcher_rows = adv_df[adv_df["ERA"].notna() & (adv_df["ERA"].astype(float) > 0)]
        pitcher_names = set(pitcher_rows["Player"].values)

    # Separate batters from pitchers
    batters = df_sub[~df_sub["Player"].isin(pitcher_names)].copy()
    # Also filter out anyone with AVG < 0.100 (likely a pitcher in batter stats)
    if "AVG" in batters.columns:
        batters = batters[batters["AVG"].fillna(0).astype(float) >= 0.050]
    batters = batters.sort_values("HR", ascending=False).head(top_n)

    pitchers = df_sub[df_sub["Player"].isin(pitcher_names)].head(2)

    if team:
        label = "Top %d for %s (%s)" % (top_n, team, abbr)
    else:
        label = "LEAGUE TOP %d SLUGGERS" % top_n

    if batters.empty and pitchers.empty:
        print("   " + cwarn("No stats found for %s (%s)" % (team, abbr) if team else "league"))
        return

    print("\n   %s:" % chi(label))

    # Show batters (HR, RBI, AVG, OPS — no ERA)
    sep = cdim("|")
    for i, (_, row) in enumerate(batters.iterrows(), 1):
        hr_s = cok("%4d" % int(row.get("HR", 0) or 0))
        rbi_s = cyel("%4d" % int(row.get("RBI", 0) or 0))
        avg_s = cblu("%.3f" % float(row.get("AVG", 0) or 0))
        ops_s = cgrn("%.3f" % float(row.get("OPS", 0) or 0))
        name_s = cbold("%-22s" % str(row.get("Player", ""))[:22])
        print("     %s %s %s HR:%s %s RBI:%s %s AVG:%s %s OPS:%s"
              % (cdim("%d." % i), name_s, sep, hr_s, sep, rbi_s, sep, avg_s, sep, ops_s))

    # Show pitchers (ERA, K — no batting stats)
    if not pitchers.empty and not adv_df.empty:
        for _, row in pitchers.iterrows():
            name = str(row.get("Player", ""))[:22]
            adv_row = adv_df[adv_df["Player"] == name]
            if not adv_row.empty:
                era = adv_row.iloc[0].get("ERA", float("nan"))
                k = adv_row.iloc[0].get("SO", adv_row.iloc[0].get("K", float("nan")))
                if not pd.isna(era) and float(era) > 0:
                    k_str = ""
                    if not pd.isna(k):
                        k_str = " %s K:%s" % (sep, cyel("%d" % int(k)))
                    print("     %s %s %s ERA:%s%s"
                          % (cdim("P."), cbold("%-22s" % name), sep, cok("%.2f" % float(era)), k_str))
