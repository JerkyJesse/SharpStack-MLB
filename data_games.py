"""Game data download and validation via MLB Stats API."""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

from config import GAMES_FILE, TEAM_ABBR, get_season_label, is_cache_stale

# Normalize alternate team names to canonical names
_TEAM_NAME_NORMALIZE = {
    "Athletics": "Oakland Athletics",
}

# Set of valid MLB team names (canonical only)
_VALID_MLB_TEAMS = {t for t in TEAM_ABBR if t != "Athletics"}


def validate_games_df(df):
    required = {"date", "home_team", "away_team", "home_score", "away_score", "neutral_site",
                "home_starter", "away_starter"}
    if df is None or df.empty:
        logging.warning("Games dataframe is empty")
        return False
    if not required.issubset(df.columns):
        logging.warning("Games dataframe missing columns: %s", required - set(df.columns))
        return False
    if len(df) < 100:
        logging.warning("Games dataframe looks small: %d rows", len(df))
        return False
    return True


def download_recent_games(csv_file=GAMES_FILE):
    if not is_cache_stale(csv_file):
        logging.info("Using cached %s (%d KB)", csv_file, os.path.getsize(csv_file) // 1024)
        return csv_file

    # --- Incremental download: load existing data and find last date ---
    existing_df = None
    last_date_dt = None
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 500:
        try:
            existing_df = pd.read_csv(csv_file)
            existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
            last_date_dt = existing_df["date"].max()
            if pd.notna(last_date_dt):
                logging.info("Existing data has %d games through %s — fetching only newer",
                             len(existing_df), last_date_dt.strftime("%Y-%m-%d"))
        except Exception as e:
            logging.warning("Could not read existing file for incremental update: %s", e)
            existing_df = None

    logging.info("Downloading recent MLB games via statsapi...")
    try:
        import statsapi
        import time

        end_date = datetime.now()
        # If we have existing data, only fetch from last known date onward
        if last_date_dt is not None:
            start_date = last_date_dt.to_pydatetime() - timedelta(days=1)
        else:
            start_date = end_date - timedelta(days=730)

        all_games = []
        current = start_date.replace(day=1)
        while current <= end_date:
            chunk_start = current.strftime("%Y-%m-%d")
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            chunk_end = min(next_month - timedelta(days=1), end_date).strftime("%Y-%m-%d")
            logging.info("  Fetching %s to %s ...", chunk_start, chunk_end)
            try:
                sched = statsapi.schedule(start_date=chunk_start, end_date=chunk_end)
            except Exception as chunk_err:
                logging.warning("  Chunk %s failed: %s", chunk_start, chunk_err)
                current = next_month
                time.sleep(1)
                continue
            for g in sched:
                # Only include completed games (Final status)
                status = str(g.get("status", "")).strip()
                if "Final" not in status and "Completed" not in status:
                    continue
                home_team = str(g.get("home_name", "")).strip()
                away_team = str(g.get("away_name", "")).strip()
                home_score = g.get("home_score", 0)
                away_score = g.get("away_score", 0)
                game_date = g.get("game_date", "")

                if not home_team or not away_team:
                    continue
                if home_score is None or away_score is None:
                    continue

                # Normalize team names (e.g. "Athletics" -> "Oakland Athletics")
                home_team = _TEAM_NAME_NORMALIZE.get(home_team, home_team)
                away_team = _TEAM_NAME_NORMALIZE.get(away_team, away_team)

                # Skip non-MLB teams (All-Star, exhibitions, minor league, international)
                if home_team not in _VALID_MLB_TEAMS or away_team not in _VALID_MLB_TEAMS:
                    continue

                # Extract starting pitcher names (probable pitchers announced ~24h before game)
                home_pitcher = str(g.get("home_probable_pitcher", "") or "").strip()
                away_pitcher = str(g.get("away_probable_pitcher", "") or "").strip()

                all_games.append({
                    "date": pd.to_datetime(game_date),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": float(home_score),
                    "away_score": float(away_score),
                    "neutral_site": False,
                    "home_starter": home_pitcher,
                    "away_starter": away_pitcher,
                })
            current = next_month
            time.sleep(0.5)

        if not all_games and existing_df is None:
            logging.warning("No games retrieved.")
            return csv_file if os.path.exists(csv_file) else None

        new_df = pd.DataFrame(all_games) if all_games else pd.DataFrame()
        # Merge with existing data
        if existing_df is not None and not new_df.empty:
            new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            n_before = len(existing_df)
        elif existing_df is not None:
            combined = existing_df
            n_before = len(existing_df)
        else:
            combined = new_df
            n_before = 0
        games_df = (combined
                    .drop_duplicates(subset=["date", "home_team", "away_team", "home_score", "away_score"])
                    .sort_values("date").reset_index(drop=True))

        if not validate_games_df(games_df):
            return csv_file if os.path.exists(csv_file) else None

        n_new = len(games_df) - n_before
        games_df.to_csv(csv_file, index=False)
        logging.info("Total %d games (%d new) -> %s", len(games_df), max(0, n_new), csv_file)
        return csv_file
    except Exception as e:
        logging.error("MLB game download failed: %s", e, exc_info=True)
        return csv_file if os.path.exists(csv_file) else None
