"""Odds tracker: fetch live/pre-game odds, track line movement, compute CLV.

Shared module used by all sports. Uses The Odds API (free tier: 500 requests/month).
Set environment variable ODDS_API_KEY to your API key, or enter it when prompted.
"""

import os
import json
import logging
import math
from datetime import datetime, timedelta

import requests

ODDS_CACHE_DIR = "odds_cache"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CLV_HISTORY_FILE = "clv_history.json"
ODDS_CACHE_FILE = "odds_latest.json"
ODDS_CACHE_MAX_MINUTES = 15  # Only re-fetch if cache older than 15 min

# Sport keys for The Odds API
SPORT_KEYS = {
    "nfl": "americanfootball_nfl",
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
}

# Bookmaker priority (sharper books first)
SHARP_BOOKS = ["pinnacle", "betfair_ex_us", "betonlineag", "bovada", "fanduel", "draftkings"]


def _get_api_key():
    """Get API key from env or key file."""
    key = os.environ.get("ODDS_API_KEY", "")
    if key:
        return key
    key_file = os.path.join(os.path.dirname(__file__), ".odds_api_key")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            return f.read().strip()
    return ""


def _save_api_key(key):
    key_file = os.path.join(os.path.dirname(__file__), ".odds_api_key")
    with open(key_file, "w") as f:
        f.write(key)


def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    elif american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100.0)
    return 0.5


def decimal_to_implied_prob(decimal_odds):
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.5
    return 1.0 / decimal_odds


def remove_vig(prob_home, prob_away):
    """Remove vigorish to get fair probabilities."""
    total = prob_home + prob_away
    if total <= 0:
        return 0.5, 0.5
    return prob_home / total, prob_away / total


def _ensure_cache_dir(sport):
    """Create cache directory for a sport."""
    cache_dir = os.path.join(os.path.dirname(__file__), sport + "Claude", ODDS_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def fetch_odds(sport, api_key=None):
    """Fetch current odds from The Odds API.

    Returns list of game dicts with odds from multiple bookmakers.
    """
    if api_key is None:
        api_key = _get_api_key()
    if not api_key:
        logging.warning("No ODDS_API_KEY set. Run with key or set env variable.")
        return []

    sport_key = SPORT_KEYS.get(sport.lower(), "")
    if not sport_key:
        logging.error("Unknown sport: %s", sport)
        return []

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 401:
            logging.error("Invalid API key for The Odds API")
            return []
        if resp.status_code == 429:
            logging.error("Odds API rate limit exceeded (500/month free)")
            return []
        resp.raise_for_status()

        remaining = resp.headers.get("x-requests-remaining", "?")
        logging.info("Odds API requests remaining this month: %s", remaining)

        data = resp.json()
        return data

    except requests.RequestException as e:
        logging.error("Failed to fetch odds: %s", e)
        return []


def parse_game_odds(game_data):
    """Parse a single game's odds into a clean dict.

    Returns dict with:
        home_team, away_team, commence_time,
        bookmaker_odds: [{book, home_ml, away_ml, home_prob, away_prob, spread, total}]
        consensus_home_prob, sharp_home_prob
    """
    home = game_data.get("home_team", "")
    away = game_data.get("away_team", "")
    commence = game_data.get("commence_time", "")

    bookmaker_odds = []
    for bm in game_data.get("bookmakers", []):
        book_name = bm.get("key", "")
        entry = {"book": book_name}

        for market in bm.get("markets", []):
            mk = market.get("key", "")
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}

            if mk == "h2h":
                home_o = outcomes.get(home, {})
                away_o = outcomes.get(away, {})
                entry["home_ml"] = home_o.get("price", 0)
                entry["away_ml"] = away_o.get("price", 0)
                h_prob = american_to_implied_prob(entry["home_ml"])
                a_prob = american_to_implied_prob(entry["away_ml"])
                fair_h, fair_a = remove_vig(h_prob, a_prob)
                entry["home_prob"] = round(fair_h, 4)
                entry["away_prob"] = round(fair_a, 4)

            elif mk == "spreads":
                home_o = outcomes.get(home, {})
                entry["spread"] = home_o.get("point", 0)

            elif mk == "totals":
                over_o = outcomes.get("Over", {})
                entry["total"] = over_o.get("point", 0)

        if "home_ml" in entry:
            bookmaker_odds.append(entry)

    # Consensus: average across all books
    consensus_home = 0.5
    if bookmaker_odds:
        probs = [b["home_prob"] for b in bookmaker_odds if "home_prob" in b]
        if probs:
            consensus_home = sum(probs) / len(probs)

    # Sharp book probability (pinnacle first, then fallback)
    sharp_home = consensus_home
    for sb in SHARP_BOOKS:
        for b in bookmaker_odds:
            if b["book"] == sb and "home_prob" in b:
                sharp_home = b["home_prob"]
                break
        if sharp_home != consensus_home:
            break

    return {
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "bookmaker_odds": bookmaker_odds,
        "consensus_home_prob": round(consensus_home, 4),
        "sharp_home_prob": round(sharp_home, 4),
        "num_books": len(bookmaker_odds),
    }


def cache_odds_snapshot(sport, games_odds):
    """Save current odds snapshot to cache for CLV tracking."""
    cache_dir = _ensure_cache_dir(sport)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_file = os.path.join(cache_dir, f"odds_{timestamp}.json")

    snapshot = {
        "fetched_at": datetime.now().isoformat(),
        "sport": sport,
        "games": games_odds,
    }

    with open(cache_file, "w") as f:
        json.dump(snapshot, f, indent=2)
    return cache_file


def load_latest_cached_odds(sport):
    """Load the most recent cached odds for a sport."""
    cache_dir = _ensure_cache_dir(sport)
    files = sorted([f for f in os.listdir(cache_dir) if f.startswith("odds_") and f.endswith(".json")])
    if not files:
        return None
    latest = os.path.join(cache_dir, files[-1])
    with open(latest, "r") as f:
        return json.load(f)


def compute_clv(model_prob, opening_prob, closing_prob):
    """Compute Closing Line Value.

    CLV = model_prob - closing_prob (positive = model was sharper than market)
    Also returns opening-to-closing movement direction.
    """
    clv = model_prob - closing_prob
    movement = closing_prob - opening_prob  # positive = market moved toward home
    return {
        "clv": round(clv, 4),
        "line_movement": round(movement, 4),
        "model_beat_close": clv > 0,
        "model_prob": round(model_prob, 4),
        "opening_prob": round(opening_prob, 4),
        "closing_prob": round(closing_prob, 4),
    }


def load_clv_history(sport):
    """Load historical CLV tracking data."""
    hist_file = os.path.join(os.path.dirname(__file__), sport + "Claude", CLV_HISTORY_FILE)
    if os.path.exists(hist_file):
        with open(hist_file, "r") as f:
            return json.load(f)
    return {"games": [], "summary": {}}


def save_clv_record(sport, game_id, model_prob, opening_prob, closing_prob, outcome):
    """Record a CLV data point for tracking."""
    hist = load_clv_history(sport)
    record = {
        "game_id": game_id,
        "date": datetime.now().isoformat(),
        "model_prob": model_prob,
        "opening_prob": opening_prob,
        "closing_prob": closing_prob,
        "outcome": outcome,
        "clv": round(model_prob - closing_prob, 4),
    }
    hist["games"].append(record)

    # Update running summary
    games = hist["games"]
    n = len(games)
    avg_clv = sum(g["clv"] for g in games) / n
    beat_rate = sum(1 for g in games if g["clv"] > 0) / n
    hist["summary"] = {
        "n_games": n,
        "avg_clv": round(avg_clv, 4),
        "beat_close_rate": round(beat_rate, 4),
        "updated": datetime.now().isoformat(),
    }

    hist_file = os.path.join(os.path.dirname(__file__), sport + "Claude", CLV_HISTORY_FILE)
    with open(hist_file, "w") as f:
        json.dump(hist, f, indent=2)


def _odds_cache_path(sport):
    """Get the odds cache file path for a sport."""
    cache_dir = os.path.join(os.path.dirname(__file__), ODDS_CACHE_DIR, sport)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, ODDS_CACHE_FILE)


def _load_odds_cache(sport):
    """Load cached odds if fresh enough. Returns (data, is_fresh)."""
    path = _odds_cache_path(sport)
    if not os.path.exists(path):
        return None, False

    try:
        with open(path, "r") as f:
            cached = json.load(f)

        fetched_at = datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))
        age_minutes = (datetime.now() - fetched_at).total_seconds() / 60

        if age_minutes < ODDS_CACHE_MAX_MINUTES:
            return cached.get("games", []), True

        return cached.get("games", []), False
    except Exception:
        return None, False


def _save_odds_cache(sport, parsed_games):
    """Save parsed odds to cache."""
    path = _odds_cache_path(sport)
    data = {
        "fetched_at": datetime.now().isoformat(),
        "sport": sport,
        "n_games": len(parsed_games),
        "games": parsed_games,
    }
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logging.debug("Failed to save odds cache: %s", e)


def get_today_odds(sport, api_key=None, force_refresh=False):
    """Fetch and parse today's odds. Uses 15-minute cache to save API calls.

    The Odds API free tier only allows 500 requests/month.
    This cache ensures we don't waste them on repeated calls.

    Returns list of parsed game dicts.
    """
    # Check cache first (unless forced refresh)
    if not force_refresh:
        cached, is_fresh = _load_odds_cache(sport)
        if is_fresh and cached:
            logging.info("Using cached odds (%s, %d games)", sport, len(cached))
            return cached

    # Fetch fresh data
    raw = fetch_odds(sport, api_key)
    if not raw:
        # If API fails, return stale cache if available
        cached, _ = _load_odds_cache(sport)
        if cached:
            logging.info("API failed, using stale odds cache (%d games)", len(cached))
            return cached
        return []

    parsed = []
    for game in raw:
        p = parse_game_odds(game)
        if p["num_books"] > 0:
            parsed.append(p)

    # Save to both snapshot archive and quick cache
    if parsed:
        cache_odds_snapshot(sport, parsed)
        _save_odds_cache(sport, parsed)

    return parsed


def find_game_odds(games_odds, home_team, away_team):
    """Find odds for a specific matchup from parsed odds list.

    Uses fuzzy matching on team names.
    """
    home_lower = home_team.lower()
    away_lower = away_team.lower()

    for g in games_odds:
        gh = g["home_team"].lower()
        ga = g["away_team"].lower()
        # Check for substring matches (e.g. "Lakers" in "Los Angeles Lakers")
        if (home_lower in gh or gh in home_lower) and \
           (away_lower in ga or ga in away_lower):
            return g
    return None


def show_odds_table(sport, colorize=None):
    """Display today's odds in a formatted table.

    colorize: optional function(text, color) for terminal colors.
    Returns the parsed odds data.
    """
    games = get_today_odds(sport)
    if not games:
        print("  No odds available (check API key or try later)")
        return []

    print(f"\n  {'Game':<45} {'Consensus':>10} {'Sharp':>10} {'Spread':>7} {'Total':>7} {'Books':>5}")
    print("  " + "-" * 87)

    for g in games:
        matchup = f"{g['away_team']} @ {g['home_team']}"
        if len(matchup) > 43:
            matchup = matchup[:43]

        con = f"{g['consensus_home_prob']:.1%}"
        shp = f"{g['sharp_home_prob']:.1%}"

        # Get consensus spread and total
        spread_str = "-"
        total_str = "-"
        if g["bookmaker_odds"]:
            spreads = [b.get("spread", None) for b in g["bookmaker_odds"] if b.get("spread") is not None]
            totals = [b.get("total", None) for b in g["bookmaker_odds"] if b.get("total") is not None]
            if spreads:
                avg_spread = sum(spreads) / len(spreads)
                spread_str = f"{avg_spread:+.1f}"
            if totals:
                avg_total = sum(totals) / len(totals)
                total_str = f"{avg_total:.1f}"

        print(f"  {matchup:<45} {con:>10} {shp:>10} {spread_str:>7} {total_str:>7} {g['num_books']:>5}")

    # Show CLV summary if available
    hist = load_clv_history(sport)
    if hist.get("summary", {}).get("n_games", 0) > 0:
        s = hist["summary"]
        print(f"\n  CLV History: {s['n_games']} games tracked, "
              f"avg CLV: {s['avg_clv']:+.2%}, "
              f"beat close rate: {s['beat_close_rate']:.1%}")

    return games
