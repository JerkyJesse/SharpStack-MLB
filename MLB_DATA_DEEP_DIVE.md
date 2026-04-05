# MLB Data Sources Deep Dive -- Prediction Accuracy Improvement Guide
# For MLBClaude Prediction System
# Last updated: 2026-03-29

This document goes deeper than FREE_DATA_SOURCES.md on MLB-specific data sources,
with concrete implementation paths tied to what the project currently uses.

---

## TABLE OF CONTENTS

1. [Free MLB APIs and Data Sources](#1-free-mlb-apis-and-data-sources)
2. [Weather Data for Game Prediction](#2-weather-data-for-game-prediction)
3. [Park Factors](#3-park-factors)
4. [Advanced Pitcher/Batter Stats](#4-advanced-pitcherbatter-stats)
5. [Injury and Lineup Data](#5-injury-and-lineup-data)
6. [Betting Market Data](#6-betting-market-data)
7. [Implementation Priority and Expected Gains](#7-implementation-priority-and-expected-gains)

---

## 1. FREE MLB APIs AND DATA SOURCES

### 1.1 MLB Stats API (statsapi.mlb.com) -- FULL ENDPOINT CATALOG

**Status**: Already used in data_games.py and data_players.py
**Package**: `MLB-StatsAPI` (already in requirements.txt)
**Auth**: NONE required
**Base URL**: `https://statsapi.mlb.com/api/v1/`

#### Endpoints Currently Used
- `statsapi.schedule()` -- game results with probable pitchers
- `statsapi.get("stats_leaders", ...)` -- batting/pitching leaders

#### Endpoints NOT Yet Used (High Value)

**A) Team Season Stats (aggregated, no need for leaders endpoint)**
```python
import statsapi

# Full team batting stats -- more complete than leaders endpoint
team_stats = statsapi.get("teams_stats", {
    "stats": "season",
    "group": "hitting",
    "season": 2025,
    "sportId": 1,
    "order": "asc",
    "sortStat": "earnedRunAverage"  # or any stat
})

# Team pitching stats
team_pitch = statsapi.get("teams_stats", {
    "stats": "season",
    "group": "pitching",
    "season": 2025,
    "sportId": 1
})

# Team fielding stats
team_field = statsapi.get("teams_stats", {
    "stats": "season",
    "group": "fielding",
    "season": 2025,
    "sportId": 1
})
```

**B) Individual Player Season Stats (full stat line, not just leaders)**
```python
# Get full season stats for a specific player
player_stats = statsapi.player_stat_data(
    personId=660271,  # Shohei Ohtani
    group="hitting",
    type="season"
)
# Returns: AVG, OBP, SLG, OPS, HR, RBI, R, SB, BB, K, etc.

# Pitching stats for a pitcher
pitcher_stats = statsapi.player_stat_data(
    personId=543037,  # Gerrit Cole
    group="pitching",
    type="season"
)
# Returns: ERA, WHIP, W, L, K, BB, IP, SO9, BB9, HR9, etc.

# Game log for a player (per-game breakdown)
game_log = statsapi.player_stat_data(
    personId=660271,
    group="hitting",
    type="gameLog"
)
```

**C) Box Score Data (per-game detail)**
```python
# Full box score for a completed game
box = statsapi.boxscore_data(gamePk=717540)
# Returns: every player's stats for that game, pitch counts, LOB, etc.

# Linescore (inning-by-inning)
linescore = statsapi.linescore(gamePk=717540)

# Play-by-play (every at-bat, every pitch)
pbp = statsapi.get("game_playByPlay", {"gamePk": 717540})
```

**D) Schedule with Hydrations (add weather, venue, lineups)**
```python
import requests

# Schedule with weather data embedded
r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
    "sportId": 1,
    "date": "2025-06-15",
    "hydrate": "probablePitcher,linescore,weather,venue,team"
})
# weather block returns: condition, temp, wind speed, wind direction

# Schedule with lineups (when available, usually ~2hrs before game)
r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
    "sportId": 1,
    "date": "2025-06-15",
    "hydrate": "lineups,probablePitcher"
})
```

**E) Standings with Advanced Records**
```python
# Standings with home/away/last10/division records
standings = statsapi.standings(leagueId="103,104", season=2025)

# Raw API with more detail
r = requests.get("https://statsapi.mlb.com/api/v1/standings", params={
    "leagueId": "103,104",
    "season": 2025,
    "hydrate": "team,division",
    "standingsTypes": "regularSeason"
})
# Returns: wins, losses, winPct, runsScored, runsAllowed,
#          runDifferential, home record, away record, last10, streak,
#          divisionRank, wildCardRank, eliminationNumber
```

**F) Live Game Feed (real-time, for live_scores.py enhancement)**
```python
# Full live feed with pitch-level data
r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live")
# v1.1 returns more detail than v1, including:
# - exitVelocity, launchAngle on batted balls
# - pitchSpeed, spinRate on pitches
# - win probability at each at-bat

# Game context metrics (leverage, win probability)
r = requests.get(f"https://statsapi.mlb.com/api/v1/game/{gamePk}/contextMetrics")
```

**G) Roster and Transactions**
```python
# Active roster
roster = statsapi.roster(teamId=147)  # Yankees = 147

# 40-man roster
roster_40 = statsapi.roster(teamId=147, rosterType="40Man")

# Recent transactions (trades, IL stints, callups)
r = requests.get("https://statsapi.mlb.com/api/v1/transactions", params={
    "startDate": "2025-06-01",
    "endDate": "2025-06-15",
    "sportId": 1
})
```

**H) Team IDs Reference**
```
ARI=109  ATL=144  BAL=110  BOS=111  CHC=112  CWS=145
CIN=113  CLE=114  COL=115  DET=116  HOU=117  KC=118
LAA=108  LAD=119  MIA=146  MIL=158  MIN=142  NYM=121
NYY=147  OAK=133  PHI=143  PIT=134  SD=135   SF=137
SEA=136  STL=138  TB=139   TEX=140  TOR=141  WSH=120
```

---

### 1.2 pybaseball -- THE Most Important Addition

**Status**: NOT in project yet. HIGHEST priority addition.
**Install**: `pip install pybaseball`
**Auth**: NONE
**Wraps**: Statcast (Baseball Savant), FanGraphs, Baseball-Reference, Retrosheet

pybaseball is the single most impactful library to add. It provides access to
advanced stats (FIP, xFIP, SIERA, wRC+, wOBA) that dramatically outperform
the basic stats (ERA, AVG, HR, RBI) currently used in data_players.py.

#### Core Functions for Prediction

```python
from pybaseball import (
    # FanGraphs season leaderboards (THE gold standard for advanced stats)
    pitching_stats,          # FIP, xFIP, SIERA, K%, BB%, WAR, ERA-
    batting_stats,           # wRC+, wOBA, ISO, BABIP, WAR, Off, Def

    # Team-level aggregates
    team_batting,            # Team wRC+, wOBA, ISO, K%, BB%
    team_pitching,           # Team FIP, xFIP, SIERA, K%, BB%
    team_fielding,           # Team UZR, DRS, OAA

    # Statcast (pitch/batted-ball level)
    statcast,                # All pitches in a date range
    statcast_pitcher,        # One pitcher's Statcast data
    statcast_batter,         # One batter's Statcast data
    statcast_running,        # Sprint speed data

    # Park factors
    park_factors,            # Venue adjustments for all 30 parks (HUGE for MLB)

    # Game logs and schedules
    schedule_and_record,     # Team game-by-game results from B-Ref

    # Player ID lookup
    playerid_lookup,         # Name -> MLBAM/FanGraphs/B-Ref IDs
    playerid_reverse_lookup, # ID -> name

    # Retrosheet historical data
    retrosheet,              # Decades of play-by-play
)
```

#### Concrete Usage for This Project

**Replace basic stats with advanced stats in data_players.py:**
```python
# CURRENT (basic -- ERA, K, W for pitchers; AVG, HR, RBI for batters)
# These are noisy, not park-adjusted, and poorly predictive

# BETTER (advanced -- via pybaseball)
from pybaseball import pitching_stats, batting_stats

# Get all qualified pitchers with advanced stats
pitchers = pitching_stats(2025, qual=30)  # 30 IP minimum
# Key columns: Name, Team, W, L, ERA, FIP, xFIP, SIERA,
#   K/9, BB/9, K%, BB%, K-BB%, HR/9, BABIP, LOB%, WAR,
#   GB%, FB%, LD%, IFFB%, Soft%, Med%, Hard%

# Get all qualified batters with advanced stats
batters = batting_stats(2025, qual=50)  # 50 PA minimum
# Key columns: Name, Team, AVG, OBP, SLG, OPS, wOBA, wRC+,
#   ISO, BABIP, K%, BB%, WAR, Off, Def, BsR,
#   Soft%, Med%, Hard%, Pull%, Cent%, Oppo%

# Team-level (most useful for game prediction)
team_bat = team_batting(2025)
# Columns: Team, wRC+, wOBA, ISO, K%, BB%, BABIP, AVG, OBP, SLG, HR, R

team_pitch = team_pitching(2025)
# Columns: Team, ERA, FIP, xFIP, SIERA, K/9, BB/9, K%, BB%, HR/9, WAR
```

**Why these stats are better for prediction:**
- **FIP** (Fielding Independent Pitching): Strips out defense/luck, uses only K/BB/HR/HBP.
  Correlates year-to-year at r=0.70 vs ERA r=0.40. Much better predictor.
- **xFIP**: FIP but normalizes HR/FB rate to league average. Even more stable.
- **SIERA**: Most complex ERA estimator, accounts for batted ball types. Best single
  pitching predictor for future performance.
- **wRC+** (Weighted Runs Created Plus): Park-adjusted, league-adjusted. 100 = average.
  The single best offensive metric. Correlates with team winning at r=0.85+.
- **wOBA** (Weighted On-Base Average): Properly weights each offensive event
  (single=0.69, double=1.00, HR=1.40, etc.). Better than OPS.

---

### 1.3 Baseball Savant / Statcast (via pybaseball)

**URL**: `https://baseballsavant.mlb.com`
**Direct CSV**: `https://baseballsavant.mlb.com/statcast_search/csv`
**Auth**: NONE

Statcast provides pitch-level and batted-ball data with measurements from
Hawk-Eye tracking cameras in every MLB stadium.

#### Key Statcast Metrics for Prediction

```python
from pybaseball import statcast

# All pitches thrown in a week (WARNING: large datasets, ~50K rows/week)
data = statcast(start_dt="2025-06-01", end_dt="2025-06-07")

# Key columns for prediction:
# - launch_speed (exit velocity): Higher = harder contact = more damage
# - launch_angle: 10-30 degrees = line drive zone = highest xBA
# - release_speed: Pitch velocity
# - release_spin_rate: Pitch spin (higher = more movement)
# - estimated_ba_using_speedangle (xBA): Expected batting average
# - estimated_woba_using_speedangle (xwOBA): Expected wOBA from contact quality
# - barrel: 1/0 flag for "barreled" balls (optimal exit velo + launch angle)
```

#### Statcast Leaderboards (easier than pitch-level)

```python
# Direct CSV download from Baseball Savant (no pybaseball needed)
import pandas as pd

# Expected stats leaderboard (batters)
url = ("https://baseballsavant.mlb.com/leaderboard/expected_statistics"
       "?type=batter&year=2025&position=&team=&min=50&csv=true")
xstats = pd.read_csv(url)
# Columns: player_id, pa, xba, xslg, xwoba, xobp, xiso, brl_percent, etc.

# Expected stats leaderboard (pitchers)
url = ("https://baseballsavant.mlb.com/leaderboard/expected_statistics"
       "?type=pitcher&year=2025&position=&team=&min=50&csv=true")
xstats_pitch = pd.read_csv(url)
# Columns: xba, xslg, xwoba (against), xera, brl_percent, etc.

# Pitch arsenal stats
url = ("https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
       "?type=pitcher&pitchType=&year=2025&team=&min=50&csv=true")
arsenal = pd.read_csv(url)
# Columns: pitch types, velocity, spin rate, whiff rate per pitch
```

#### Most Predictive Statcast Features
1. **Team xwOBA differential** (batting xwOBA minus pitching xwOBA-against): r=0.80+ with winning
2. **Barrel rate**: Teams that barrel the ball more score more runs
3. **Hard-hit rate** (exit velo >= 95 mph): Stabilizes faster than actual results
4. **Average exit velocity**: Team-level is very predictive of run scoring
5. **xERA for starting pitcher**: Better than ERA for predicting next-start performance

---

### 1.4 FanGraphs (via pybaseball)

**URL**: `https://www.fangraphs.com`
**CSV export**: Append `&content=csv` to any leaderboard URL
**Python**: Via `pybaseball` functions `pitching_stats()` and `batting_stats()`

#### FanGraphs-Specific Data Not in Statcast

```python
from pybaseball import pitching_stats, batting_stats

# Pitcher stats with FanGraphs-specific metrics
p = pitching_stats(2025, qual=30)
# FanGraphs-only columns:
#   FIP, xFIP, SIERA -- THE best ERA predictors
#   K-BB% -- best simple pitcher quality metric
#   WAR (fWAR) -- total pitcher value
#   LOB% -- strand rate (regresses to ~72%)
#   GB%, FB%, LD% -- batted ball profile
#   Soft%, Med%, Hard% -- contact quality allowed
#   O-Swing%, Z-Contact% -- plate discipline induced
#   CSW% (Called Strike + Whiff %) -- pitch quality metric

# Batter stats with FanGraphs-specific metrics
b = batting_stats(2025, qual=50)
# FanGraphs-only columns:
#   wRC+ -- THE best single offensive metric (park+league adjusted)
#   wOBA -- properly weighted OBP variant
#   ISO (Isolated Power = SLG - AVG) -- pure power
#   BABIP -- batting average on balls in play (regresses to ~.300)
#   Off (offensive runs above average)
#   Def (defensive runs above average)
#   BsR (baserunning runs above average)
#   WAR (fWAR)
#   Pull%, Cent%, Oppo% -- spray chart tendencies
#   Soft%, Med%, Hard% -- contact quality
#   O-Swing%, Z-Contact% -- plate discipline
```

#### FanGraphs CSV Export URLs (no pybaseball needed)

```python
import pandas as pd

# Team batting leaderboard (all teams, current season)
url = ("https://www.fangraphs.com/api/leaders/major-league/data"
       "?age=&pos=all&stats=bat&lg=all&qual=0&season=2025&season1=2025"
       "&startdate=&enddate=&month=0&hand=&team=0,ts&pageitems=50"
       "&pagenum=1&ind=0&rost=0&players=&type=8&postseason=&sortdir=default"
       "&sortstat=wRC%2B")
# Returns JSON; parse with requests

# Team pitching leaderboard
url = ("https://www.fangraphs.com/api/leaders/major-league/data"
       "?age=&pos=all&stats=pit&lg=all&qual=0&season=2025&season1=2025"
       "&startdate=&enddate=&month=0&hand=&team=0,ts&pageitems=50"
       "&pagenum=1&ind=0&rost=0&players=&type=8&postseason=&sortdir=default"
       "&sortstat=WAR")
```

---

### 1.5 Retrosheet -- Historical Play-by-Play

**URL**: `https://www.retrosheet.org`
**Bulk download**: `https://www.retrosheet.org/game.htm`
**Python**: `pybaseball.retrosheet` or download directly

Retrosheet has every MLB game back to 1871, with play-by-play back to 1950s
for many games, and complete play-by-play from ~1974 onward.

```python
# Via pybaseball
from pybaseball import retrosheet

# Get all games for a season
games = retrosheet.season_game_logs(2024)

# Direct download of event files
# https://www.retrosheet.org/events/{year}eve.zip
# Contains .EVA/.EVN files (American/National league event files)
# Parse with Chadwick tools (cwevent, cwgame) or Python parsers
```

**Use case for this project**: Backtest your Elo model against decades of data
to validate parameter stability.

---

### 1.6 Lahman Database

**URL**: `https://www.seanlahman.com/baseball-archive/statistics/`
**Python**: `pip install pybaseball` includes Lahman data
**Also**: Direct CSV from GitHub: `https://github.com/chadwickbureau/baseballdatabank`

```python
from pybaseball import lahman

# All batting data (1871-present)
batting = lahman.batting()

# All pitching data
pitching = lahman.pitching()

# Team-level data
teams = lahman.teams()

# Park data
parks = lahman.parks()

# People (player biographical info)
people = lahman.people()
```

**Use case**: Historical park factors, long-term team strength trends,
and extremely deep backtesting (100+ years of game data).

---

### 1.7 ESPN MLB API (Partial in project)

**Status**: injuries.py uses the injury endpoint
**Additional endpoints for prediction**:

```python
import requests

# Scoreboard with EMBEDDED ODDS (moneyline, spread, O/U)
r = requests.get("https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard")
data = r.json()
for event in data.get("events", []):
    for comp in event.get("competitions", []):
        # Odds data
        odds = comp.get("odds", [{}])
        if odds:
            moneyline_home = odds[0].get("homeTeamOdds", {}).get("moneyLine")
            moneyline_away = odds[0].get("awayTeamOdds", {}).get("moneyLine")
            over_under = odds[0].get("overUnder")
            spread = odds[0].get("spread")

        # ESPN's own win probability prediction
        predictor = comp.get("predictor", {})
        espn_home_prob = predictor.get("homeTeam", {}).get("gameProjection")

# Date-specific scoreboard
r = requests.get("https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
                  params={"dates": "20250615"})

# ESPN BPI / Power Index (when available)
r = requests.get("https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/powerindex")
```

---

## 2. WEATHER DATA FOR GAME PREDICTION

### 2.1 How Much Does Weather Affect MLB Outcomes?

**Research findings on weather effects:**

| Factor | Effect on Runs/Game | Significance |
|--------|-------------------|-------------|
| Temperature +10F | +0.3 runs/game | HIGH -- ball travels ~4ft further per 10F |
| Wind out to CF 10mph | +0.5 to +1.0 runs/game | HIGH at certain parks (Wrigley, Kauffman) |
| Wind in from CF 10mph | -0.5 to -1.0 runs/game | HIGH -- suppresses HR |
| Altitude +1000ft | +0.5 runs/game | HUGE at Coors Field (5280ft) |
| Humidity +20% | +0.05 runs/game | MINIMAL (myth that humid air is heavier) |
| Rain/mist | -0.2 runs/game | MINOR, but affects grip on ball |
| Day game after night game | -0.1 to -0.3 runs for batting team | MODERATE fatigue effect |

**Key insight**: Weather matters most for TOTALS (over/under) predictions and
for games at specific wind-sensitive parks. For moneyline prediction, the
effect is smaller but measurable at ~0.3-0.5% accuracy gain for affected games.

**Wind-sensitive parks (weather matters most here)**:
- Wrigley Field (CHC): Wind blowing out = HR paradise; wind blowing in = pitcher's park
- Kauffman Stadium (KC): Open outfield, wind carries or suppresses
- Citi Field (NYM): Exposed to wind off Flushing Bay
- Oracle Park (SF): Cold ocean wind suppresses offense
- PNC Park (PIT): River winds affect ball flight
- Fenway Park (BOS): Wind to/from the Green Monster matters

**Dome/retractable roof parks (weather irrelevant)**:
- Tropicana Field (TB): Dome
- Globe Life Field (TEX): Retractable, usually closed
- Chase Field (ARI): Retractable, usually closed in summer
- American Family Field (MIL): Retractable
- Minute Maid Park (HOU): Retractable
- Rogers Centre (TOR): Retractable
- T-Mobile Park (SEA): Retractable
- loanDepot park (MIA): Retractable

### 2.2 Open-Meteo -- BEST Free Weather API (No Auth!)

**URL**: `https://open-meteo.com`
**Auth**: NONE -- completely free, no API key needed
**Rate limit**: 10,000 requests/day (more than enough)
**Historical data**: Also free via archive API

```python
import requests

def get_game_weather(lat, lon, date, hour=19):
    """Get weather for a game location and time.

    Args:
        lat, lon: Stadium coordinates
        date: Game date (YYYY-MM-DD)
        hour: Game start hour (local time, 24h format)
    """
    # Forecast (up to 16 days ahead)
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "hourly": ("temperature_2m,relative_humidity_2m,"
                   "wind_speed_10m,wind_direction_10m,"
                   "precipitation,rain,cloud_cover"),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "start_date": date,
        "end_date": date,
    })
    data = r.json()
    hourly = data.get("hourly", {})
    # Find the hour index matching game time
    idx = hour  # hours are 0-23
    return {
        "temp_f": hourly["temperature_2m"][idx],
        "humidity": hourly["relative_humidity_2m"][idx],
        "wind_mph": hourly["wind_speed_10m"][idx],
        "wind_dir": hourly["wind_direction_10m"][idx],
        "precip_mm": hourly["precipitation"][idx],
        "rain_mm": hourly["rain"][idx],
        "cloud_pct": hourly["cloud_cover"][idx],
    }

def get_historical_weather(lat, lon, date, hour=19):
    """Get historical weather for backtesting."""
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat,
        "longitude": lon,
        "hourly": ("temperature_2m,relative_humidity_2m,"
                   "wind_speed_10m,wind_direction_10m,"
                   "precipitation,rain"),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "start_date": date,
        "end_date": date,
    })
    data = r.json()
    hourly = data.get("hourly", {})
    idx = hour
    return {
        "temp_f": hourly["temperature_2m"][idx],
        "humidity": hourly["relative_humidity_2m"][idx],
        "wind_mph": hourly["wind_speed_10m"][idx],
        "wind_dir": hourly["wind_direction_10m"][idx],
        "precip_mm": hourly["precipitation"][idx],
    }
```

### 2.3 MLB Stats API Weather Hydration

**Already available in your existing API** -- just add `hydrate=weather` to schedule requests:

```python
import requests

r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
    "sportId": 1,
    "date": "2025-06-15",
    "hydrate": "weather,venue,probablePitcher"
})
data = r.json()
for date_entry in data.get("dates", []):
    for game in date_entry.get("games", []):
        weather = game.get("weather", {})
        # Returns:
        #   condition: "Sunny", "Cloudy", "Overcast", "Dome", etc.
        #   temp: "78"
        #   wind: "12 mph, Out To CF" or "10 mph, In From LF"
        # Note: This is game-time weather, MLB's own data
```

**Advantage over Open-Meteo**: MLB provides wind direction relative to the field
(e.g., "Out To CF", "In From RF") which is much more useful than compass direction.

**Disadvantage**: Only available for today/upcoming games, not historical.
For backtesting weather features, you need Open-Meteo's archive API.

### 2.4 Other Weather APIs (Require Free API Key)

**OpenWeatherMap**: `https://openweathermap.org/api`
- Free tier: 1,000 calls/day, 60/min
- Requires API key signup
- Current + 5-day forecast free; historical requires paid "One Call 3.0"

**Visual Crossing**: `https://www.visualcrossing.com/weather-api`
- Free tier: 1,000 records/day
- Requires API key signup
- Historical data included in free tier (advantage over OpenWeatherMap)

**Recommendation**: Use Open-Meteo (no auth) as primary. MLB API weather hydration
for day-of accuracy. Visual Crossing as backup for historical if needed.

---

## 3. PARK FACTORS

### 3.1 What Are Park Factors?

Park factors measure how much a stadium inflates or deflates run scoring
compared to league average. A park factor of 110 means 10% more runs are
scored there than average. This is caused by:
- **Altitude**: Thin air = ball travels further (Coors Field)
- **Dimensions**: Short fences = more HRs (Yankee Stadium RF)
- **Wind patterns**: Consistent winds affect ball flight
- **Temperature**: Warmer cities score more
- **Foul territory**: Large foul territory = more outs (old Oakland Coliseum)
- **Wall heights**: Green Monster (37ft) turns HRs into doubles

### 3.2 Getting Park Factors via pybaseball

```python
from pybaseball import park_factors

# Get park factors for all 30 stadiums
pf = park_factors(2025)
# Returns DataFrame with columns:
#   Team, Basic (overall park factor), 1B, 2B, 3B, HR, BB, SO
#   Each is indexed to 100 (100 = neutral)

# Example values (approximate, vary by year):
# COL (Coors Field):     ~115 (15% more runs)
# ARI (Chase Field):     ~105 (roof open hot games)
# CIN (Great American):  ~105 (small park, hot summer)
# NYY (Yankee Stadium):  ~107 (short RF porch for LHB HR)
# TEX (Globe Life):      ~100 (retractable roof normalized it)
# SF  (Oracle Park):     ~92  (cold wind, deep CF, huge RF)
# OAK (Coliseum):        ~94  (huge foul territory)
# SEA (T-Mobile Park):   ~95  (marine layer, deep dimensions)
# MIA (loanDepot):       ~96  (humid but deep, retractable)
# STL (Busch Stadium):   ~97
# SD  (Petco Park):      ~96  (pitcher friendly)
# PIT (PNC Park):        ~98
# LAD (Dodger Stadium):  ~100 (neutral)
```

### 3.3 Hardcoded Park Factor Table

For reliability (no API call needed), maintain a hardcoded table updated annually:

```python
# 2024-2025 approximate MLB park factors (run factor, 100 = neutral)
# Source: FanGraphs, compiled from 3-year rolling average
MLB_PARK_FACTORS = {
    "Arizona Diamondbacks":     105,  # Chase Field (retractable, hot when open)
    "Atlanta Braves":           100,  # Truist Park
    "Baltimore Orioles":       103,  # Camden Yards (short LF)
    "Boston Red Sox":          106,  # Fenway Park (Green Monster, short LF)
    "Chicago Cubs":            102,  # Wrigley Field (varies wildly with wind)
    "Chicago White Sox":       104,  # Guaranteed Rate Field (hitter friendly)
    "Cincinnati Reds":         105,  # Great American Ball Park (small, hot)
    "Cleveland Guardians":      98,  # Progressive Field
    "Colorado Rockies":        115,  # Coors Field (5,280ft altitude -- EXTREME)
    "Detroit Tigers":           98,  # Comerica Park (deep CF)
    "Houston Astros":          101,  # Minute Maid (retractable, Crawford Boxes)
    "Kansas City Royals":      101,  # Kauffman Stadium (wind dependent)
    "Los Angeles Angels":      100,  # Angel Stadium
    "Los Angeles Dodgers":     100,  # Dodger Stadium (neutral)
    "Miami Marlins":            96,  # loanDepot park (retractable, deep)
    "Milwaukee Brewers":       102,  # American Family Field (retractable)
    "Minnesota Twins":         101,  # Target Field
    "New York Mets":            97,  # Citi Field (deep, wind off bay)
    "New York Yankees":        107,  # Yankee Stadium (short RF = LHB HRs)
    "Oakland Athletics":        94,  # Oakland Coliseum (huge foul territory)
    "Philadelphia Phillies":   102,  # Citizens Bank Park
    "Pittsburgh Pirates":       98,  # PNC Park
    "San Diego Padres":         96,  # Petco Park (marine layer)
    "San Francisco Giants":     92,  # Oracle Park (cold, deep, wind)
    "Seattle Mariners":         95,  # T-Mobile Park (marine layer)
    "St. Louis Cardinals":      97,  # Busch Stadium
    "Tampa Bay Rays":           97,  # Tropicana Field (dome, turf)
    "Texas Rangers":           100,  # Globe Life Field (retractable, AC)
    "Toronto Blue Jays":        99,  # Rogers Centre (dome)
    "Washington Nationals":    100,  # Nationals Park
}
```

### 3.4 Stadium Coordinates and Details

```python
# Stadium coordinates for weather API calls and travel distance
MLB_STADIUMS = {
    "Arizona Diamondbacks":     {"lat": 33.445, "lon": -112.067, "alt_ft": 1082, "roof": "retractable", "name": "Chase Field"},
    "Atlanta Braves":           {"lat": 33.891, "lon": -84.468,  "alt_ft": 1050, "roof": "open",        "name": "Truist Park"},
    "Baltimore Orioles":       {"lat": 39.284, "lon": -76.622,  "alt_ft": 30,   "roof": "open",        "name": "Camden Yards"},
    "Boston Red Sox":          {"lat": 42.346, "lon": -71.098,  "alt_ft": 20,   "roof": "open",        "name": "Fenway Park"},
    "Chicago Cubs":            {"lat": 41.948, "lon": -87.656,  "alt_ft": 595,  "roof": "open",        "name": "Wrigley Field"},
    "Chicago White Sox":       {"lat": 41.830, "lon": -87.634,  "alt_ft": 595,  "roof": "open",        "name": "Guaranteed Rate Field"},
    "Cincinnati Reds":         {"lat": 39.097, "lon": -84.508,  "alt_ft": 485,  "roof": "open",        "name": "Great American Ball Park"},
    "Cleveland Guardians":     {"lat": 41.496, "lon": -81.685,  "alt_ft": 620,  "roof": "open",        "name": "Progressive Field"},
    "Colorado Rockies":        {"lat": 39.756, "lon": -104.994, "alt_ft": 5280, "roof": "open",        "name": "Coors Field"},
    "Detroit Tigers":          {"lat": 42.339, "lon": -83.049,  "alt_ft": 600,  "roof": "open",        "name": "Comerica Park"},
    "Houston Astros":          {"lat": 29.757, "lon": -95.356,  "alt_ft": 42,   "roof": "retractable", "name": "Minute Maid Park"},
    "Kansas City Royals":      {"lat": 39.052, "lon": -94.481,  "alt_ft": 750,  "roof": "open",        "name": "Kauffman Stadium"},
    "Los Angeles Angels":      {"lat": 33.800, "lon": -117.883, "alt_ft": 160,  "roof": "open",        "name": "Angel Stadium"},
    "Los Angeles Dodgers":     {"lat": 34.074, "lon": -118.240, "alt_ft": 515,  "roof": "open",        "name": "Dodger Stadium"},
    "Miami Marlins":           {"lat": 25.778, "lon": -80.220,  "alt_ft": 7,    "roof": "retractable", "name": "loanDepot park"},
    "Milwaukee Brewers":       {"lat": 43.028, "lon": -87.971,  "alt_ft": 600,  "roof": "retractable", "name": "American Family Field"},
    "Minnesota Twins":         {"lat": 44.982, "lon": -93.278,  "alt_ft": 840,  "roof": "open",        "name": "Target Field"},
    "New York Mets":           {"lat": 40.757, "lon": -73.846,  "alt_ft": 12,   "roof": "open",        "name": "Citi Field"},
    "New York Yankees":        {"lat": 40.829, "lon": -73.926,  "alt_ft": 55,   "roof": "open",        "name": "Yankee Stadium"},
    "Oakland Athletics":       {"lat": 37.751, "lon": -122.201, "alt_ft": 5,    "roof": "open",        "name": "Oakland Coliseum"},
    "Philadelphia Phillies":   {"lat": 39.906, "lon": -75.167,  "alt_ft": 20,   "roof": "open",        "name": "Citizens Bank Park"},
    "Pittsburgh Pirates":      {"lat": 40.447, "lon": -80.006,  "alt_ft": 730,  "roof": "open",        "name": "PNC Park"},
    "San Diego Padres":        {"lat": 32.707, "lon": -117.157, "alt_ft": 13,   "roof": "open",        "name": "Petco Park"},
    "San Francisco Giants":    {"lat": 37.778, "lon": -122.389, "alt_ft": 5,    "roof": "open",        "name": "Oracle Park"},
    "Seattle Mariners":        {"lat": 47.591, "lon": -122.332, "alt_ft": 17,   "roof": "retractable", "name": "T-Mobile Park"},
    "St. Louis Cardinals":     {"lat": 38.623, "lon": -90.193,  "alt_ft": 455,  "roof": "open",        "name": "Busch Stadium"},
    "Tampa Bay Rays":          {"lat": 27.768, "lon": -82.653,  "alt_ft": 44,   "roof": "dome",        "name": "Tropicana Field"},
    "Texas Rangers":           {"lat": 32.747, "lon": -97.084,  "alt_ft": 545,  "roof": "retractable", "name": "Globe Life Field"},
    "Toronto Blue Jays":       {"lat": 43.641, "lon": -79.389,  "alt_ft": 266,  "roof": "retractable", "name": "Rogers Centre"},
    "Washington Nationals":    {"lat": 38.873, "lon": -77.007,  "alt_ft": 25,   "roof": "open",        "name": "Nationals Park"},
}
```

### 3.5 How to Use Park Factors in Prediction

```python
def park_adjusted_matchup(home_team, away_team, park_factors):
    """Adjust prediction based on park factor."""
    # Home team plays in their park
    pf = park_factors.get(home_team, 100)

    # Convert park factor to run multiplier
    # PF of 115 (Coors) means ~15% more runs
    multiplier = pf / 100.0

    # For moneyline prediction:
    # - High PF benefits the better offense (more variance = more upsets)
    # - Low PF benefits the better pitching team (less variance = favorites win more)

    # For Elo adjustment:
    # - Don't adjust Elo directly (park is already in results)
    # - But DO adjust when comparing stats across parks
    # - A team with 4.5 runs/game at Coors (PF 115) is really ~3.9 runs/game neutral

    # Neutral-site equivalent runs
    # adj_runs = actual_runs / (pf / 100)
    return multiplier

def neutralize_stats(team_rpg, team_rapg, home_pf_runs):
    """Convert raw team stats to park-neutral."""
    divisor = home_pf_runs / 100.0
    # Teams play ~half their games at home
    # So adjustment is roughly (1 + divisor) / 2 of raw stats
    home_fraction = 0.5
    adj_factor = home_fraction * divisor + (1 - home_fraction) * 1.0
    neutral_rpg = team_rpg / adj_factor
    neutral_rapg = team_rapg / adj_factor
    return neutral_rpg, neutral_rapg
```

---

## 4. ADVANCED PITCHER/BATTER STATS

### 4.1 Pitcher Stats Available for Free

#### Tier 1: Best Predictors of Future Performance (via pybaseball/FanGraphs)

| Stat | Source | Correlation to Future ERA | How to Get |
|------|--------|--------------------------|------------|
| **SIERA** | FanGraphs | r=0.65 | `pitching_stats(2025)["SIERA"]` |
| **xFIP** | FanGraphs | r=0.62 | `pitching_stats(2025)["xFIP"]` |
| **FIP** | FanGraphs | r=0.60 | `pitching_stats(2025)["FIP"]` |
| **K-BB%** | FanGraphs | r=0.58 | `pitching_stats(2025)["K-BB%"]` |
| **xERA** | Statcast | r=0.55 | Baseball Savant CSV leaderboard |
| **CSW%** | FanGraphs | r=0.50 | `pitching_stats(2025)["CSW%"]` |

**FIP formula** (can compute yourself without pybaseball):
```
FIP = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + FIP_constant
FIP_constant ~ 3.10 (varies by league-year, = lgERA - lgFIP_raw)
```

**xFIP** = FIP but replaces actual HR with expected HR (using 10.5% HR/FB rate):
```
xFIP = ((13*(FB*0.105) + 3*(BB+HBP) - 2*K) / IP) + FIP_constant
```

#### Tier 2: Useful Supporting Metrics

| Stat | What It Measures | Source |
|------|-----------------|--------|
| K/9, K% | Strikeout ability | MLB Stats API (already have) |
| BB/9, BB% | Walk rate (control) | MLB Stats API |
| HR/9, HR/FB | Home run prevention | FanGraphs |
| GB%, FB%, LD% | Batted ball profile | FanGraphs |
| BABIP | Luck indicator (regresses to ~.295) | FanGraphs |
| LOB% | Strand rate (regresses to ~72%) | FanGraphs |
| Soft%, Hard% | Contact quality allowed | FanGraphs |

#### Tier 3: Statcast Pitch-Level (via pybaseball)

| Stat | What It Measures | Source |
|------|-----------------|--------|
| Avg fastball velocity | Stuff quality | Statcast |
| Spin rate (by pitch type) | Movement quality | Statcast |
| Whiff rate | Swing-and-miss ability | Statcast |
| Chase rate | Ability to get batters to chase | Statcast |
| xwOBA against | Expected quality of contact allowed | Statcast |
| Barrel% against | Hard contact allowed | Statcast |

```python
from pybaseball import statcast_pitcher

# Get all Statcast data for a specific pitcher
cole = statcast_pitcher("2025-04-01", "2025-06-15", player_id=543037)

# Compute pitch-level metrics
avg_velo = cole[cole["pitch_type"] == "FF"]["release_speed"].mean()  # Fastball velocity
whiff_rate = cole["description"].str.contains("swinging_strike").mean()
barrel_rate = cole["barrel"].mean()
chase_rate = cole[cole["zone"] > 9]["description"].str.contains("swing").mean()
```

### 4.2 Batter Stats Available for Free

#### Tier 1: Best Offensive Metrics (via pybaseball/FanGraphs)

| Stat | Source | What It Measures | How to Get |
|------|--------|-----------------|------------|
| **wRC+** | FanGraphs | Total offensive value (park+league adjusted, 100=avg) | `batting_stats(2025)["wRC+"]` |
| **wOBA** | FanGraphs | Weighted on-base (properly weights each event) | `batting_stats(2025)["wOBA"]` |
| **xwOBA** | Statcast | Expected wOBA from contact quality | Savant CSV |
| **OPS+** | B-Ref | OPS park-adjusted (100=avg) | pybaseball or scrape |
| **WAR** | FanGraphs | Total player value | `batting_stats(2025)["WAR"]` |

#### Tier 2: Component Stats

| Stat | Formula / Source | Use Case |
|------|-----------------|----------|
| ISO | SLG - AVG | Pure power (isolates extra bases) |
| BABIP | (H-HR)/(AB-K-HR+SF) | Luck detector (regresses to ~.300) |
| BB% | BB/PA | Plate discipline |
| K% | K/PA | Contact ability |
| HR/FB | HR/Fly Balls | Home run rate (regresses to ~10-12%) |
| Barrel% | From Statcast | Quality of hardest contact |
| Hard hit% | Exit velo >= 95mph | Contact quality |

#### wOBA Calculation (can compute without pybaseball)

```python
# wOBA weights (approximate, vary slightly by year)
# Based on linear weights (run value of each event)
WOBA_WEIGHTS = {
    "uBB": 0.690,   # unintentional walk
    "HBP": 0.722,   # hit by pitch
    "1B":  0.878,   # single
    "2B":  1.242,   # double
    "3B":  1.568,   # triple
    "HR":  2.007,   # home run
}
WOBA_SCALE = 1.157  # scales wOBA to OBP scale

def calc_woba(bb, hbp, singles, doubles, triples, hr, ab, sf):
    """Calculate wOBA from counting stats."""
    numerator = (WOBA_WEIGHTS["uBB"] * bb +
                 WOBA_WEIGHTS["HBP"] * hbp +
                 WOBA_WEIGHTS["1B"] * singles +
                 WOBA_WEIGHTS["2B"] * doubles +
                 WOBA_WEIGHTS["3B"] * triples +
                 WOBA_WEIGHTS["HR"] * hr)
    denominator = ab + bb + sf + hbp
    if denominator == 0:
        return 0.0
    return numerator / denominator

# wRC+ calculation:
# wRC+ = ((wRAA/PA + lgR/PA) + (lgR/PA - PF*lgR/PA)) / (lgwRC/PA) * 100
# Much easier to just pull from FanGraphs via pybaseball
```

### 4.3 Team-Level Advanced Stats (Most Useful for Game Prediction)

For game-to-game prediction, team-level stats are more useful than individual:

```python
from pybaseball import team_batting, team_pitching

# Team offense
tb = team_batting(2025)
# Key columns: wRC+, wOBA, ISO, K%, BB%, HR, R, BABIP

# Team pitching
tp = team_pitching(2025)
# Key columns: FIP, xFIP, SIERA, K/9, BB/9, HR/9, WAR

# Best single-number team strength metrics:
# OFFENSE: wRC+ (park-adjusted, 100 = average)
# PITCHING: FIP or xFIP (fielding-independent, better than ERA)
# COMBINED: Team WAR (off WAR + pitch WAR + field WAR)

# For matchup prediction, the differential is key:
# home_wRC+ - away_pitching_FIP_adj ... etc.
```

---

## 5. INJURY AND LINEUP DATA

### 5.1 Current Injury Sources (Project Already Has ESPN)

The project's `injuries.py` uses ESPN's MLB injury API. Here are additional sources:

#### MLB Stats API Transactions (IL Stints)

```python
import requests

# Recent transactions including IL placements
r = requests.get("https://statsapi.mlb.com/api/v1/transactions", params={
    "startDate": "2025-06-01",
    "endDate": "2025-06-15",
    "sportId": 1,
    "transactionTypes": "Disabled List"  # IL placements
})
# Returns: player name, team, type (10-day IL, 60-day IL), date
```

#### Roster Status via MLB Stats API

```python
import statsapi

# Active roster (who is currently available)
active = statsapi.roster(teamId=147, rosterType="active")

# 40-man roster (includes IL players with notes)
full = statsapi.roster(teamId=147, rosterType="40Man")

# Compare active vs 40-man to find IL players
```

#### CBS Sports Injury API (Unofficial)

```
https://www.cbssports.com/mlb/injuries/
```
Scrapeable, provides status (Out, Day-to-Day, 15-Day IL, 60-Day IL) with return dates.

#### Rotoworld/NBC Sports Injuries

```
https://www.nbcsports.com/fantasy/baseball/injury-report
```
Includes fantasy-relevant injury notes and expected return timelines.

### 5.2 Daily Lineup Sources

Daily lineups are critical for prediction because they tell you:
- Which pitcher is actually starting (not just probable)
- The batting order (which impacts run expectancy)
- Who is resting (load management)
- Platoon matchups (vs LHP/RHP)

#### MLB Stats API Lineups (Official, ~2 hours before game)

```python
import requests

# Schedule with lineups hydrated
r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
    "sportId": 1,
    "date": "2025-06-15",
    "hydrate": "lineups,probablePitcher"
})
# lineups array contains battingOrder for each team

# Live game feed also has lineups once submitted
r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live")
# gameData.players has full roster info
# liveData.boxscore.teams.home.battingOrder has lineup
```

#### Rotowire / Rotogrinders (Fantasy Sources)

```
https://www.rotowire.com/baseball/daily-lineups.php
```
Rotowire posts lineups as soon as they are announced (~2-4 hours before first pitch).
Scrapeable via requests + BeautifulSoup.

#### Baseball Press

```
https://www.baseballpress.com/lineups
```
Another early lineup source. Also includes probable pitchers for multi-day lookahead.

#### Twitter/X MLB Lineups Bot

`@Lineups` on Twitter posts lineups as they are confirmed.
Not API-accessible, but a useful cross-reference.

### 5.3 Starting Pitcher Confirmation

Your project already uses probable pitchers from `statsapi.schedule()`. The gap is:
- Probable pitchers are set ~24-48 hours in advance
- Actual starters may change (scratches happen)
- Confirmation typically comes ~2 hours before game time

```python
# Check for last-minute starter changes
import requests

# Hydrate with probablePitcher gets the latest update
r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
    "sportId": 1,
    "date": "2025-06-15",
    "hydrate": "probablePitcher(note)"
})
# The "note" sub-hydration includes scratch/change notes
```

---

## 6. BETTING MARKET DATA

### 6.1 The Odds API (Best Free Source)

**URL**: `https://the-odds-api.com`
**Auth**: Free API key required (signup at website)
**Free tier**: 500 requests/month (enough for daily pulls)
**Coverage**: All major US sportsbooks

```python
import requests

API_KEY = "your_free_api_key"  # Get from https://the-odds-api.com

# Current MLB odds from all books
r = requests.get(
    "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
    params={
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,totals",  # h2h = moneyline, totals = over/under
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,caesars,pointsbet"
    }
)
odds = r.json()
# Each game has: bookmakers[] -> markets[] -> outcomes[] with price (odds)

# Historical odds (past games)
r = requests.get(
    "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/",
    params={
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "date": "2025-06-15T00:00:00Z",
    }
)

# Check remaining API quota (returned in response headers)
# x-requests-remaining, x-requests-used
```

### 6.2 ESPN Embedded Odds (Free, No Auth)

Already partially documented. Key addition for MLB:

```python
import requests

r = requests.get("https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard")
data = r.json()

for event in data.get("events", []):
    game_name = event.get("name", "")
    for comp in event.get("competitions", []):
        odds_list = comp.get("odds", [])
        if odds_list:
            odds = odds_list[0]  # Primary book (usually Caesars)
            print(f"{game_name}")
            print(f"  Spread: {odds.get('spread')}")
            print(f"  O/U: {odds.get('overUnder')}")
            print(f"  Home ML: {odds.get('homeTeamOdds', {}).get('moneyLine')}")
            print(f"  Away ML: {odds.get('awayTeamOdds', {}).get('moneyLine')}")

            # ALSO: ESPN sometimes includes their win probability model
            predictor = comp.get("predictor", {})
            if predictor:
                home_prob = predictor.get("homeTeam", {}).get("gameProjection")
                print(f"  ESPN Win Prob: {home_prob}%")
```

### 6.3 DraftKings Public API (Unofficial)

DraftKings has public-facing JSON endpoints that power their website:

```python
import requests

# MLB event group
r = requests.get(
    "https://sportsbook-nash-usdc.draftkings.com/sites/US-DC-SB/api/v5/eventgroups/84240",
    params={"format": "json"}
)
# Returns all available MLB games with odds
# WARNING: This endpoint can change without notice and may have legal restrictions
```

### 6.4 Historical Closing Line Data

Historical odds are the most valuable for backtesting and for building a
"market-aware" model. Sources:

**A) Kaggle MLB Odds Datasets**
```
https://www.kaggle.com/datasets/
Search: "MLB odds historical", "MLB moneyline", "MLB betting lines"
```
Several free datasets with years of historical closing lines.

**B) Sports-Statistics.com**
```
https://sports-statistics.com/sports-data/mlb-historical-odds-scores-datasets/
```
Historical MLB odds datasets (some free, some paid).

**C) Odds-API Historical (Limited Free)**
The Odds API historical endpoint works within your monthly quota.

**D) SBR Odds Archive**
```
https://www.sportsbookreview.com/betting-odds/mlb-baseball/money-line/
```
Has historical odds viewable by date. Requires scraping (JavaScript-rendered).

### 6.5 Using Market Data for Prediction

**Why market data helps:**
1. **Closing Line Value (CLV)**: If your model consistently beats closing lines,
   you have a profitable edge. CLV is the best measure of prediction skill.
2. **Market as a feature**: Vegas lines encode all public + sharp information.
   Using the market line as an input feature to your model can improve accuracy.
3. **Calibration benchmark**: Compare your model's log loss / Brier score against
   the implied probabilities from closing lines.
4. **Ensemble**: Blending your Elo/XGBoost prediction with market-implied
   probability (e.g., 60/40 or 70/30) often beats either alone.

**Converting moneyline odds to implied probability:**
```python
def ml_to_prob(ml):
    """Convert American moneyline to implied probability."""
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)

# Example: NYY -150, BOS +130
home_prob = ml_to_prob(-150)  # 0.600 (60%)
away_prob = ml_to_prob(130)   # 0.435 (43.5%)
# Total > 100% because of vig/juice (~4-5% for MLB)
# Remove vig: normalize to sum to 1.0
total = home_prob + away_prob
home_fair = home_prob / total  # ~0.580
away_fair = away_prob / total  # ~0.420
```

---

## 7. IMPLEMENTATION PRIORITY AND EXPECTED GAINS

### What the Project Currently Uses vs. What to Add

| Data | Current Source | Current Quality | Recommended Upgrade |
|------|---------------|----------------|-------------------|
| Game results | MLB Stats API | Good | No change needed |
| Batting stats | Stats leaders (AVG, HR, RBI) | Basic | **pybaseball: wRC+, wOBA** |
| Pitching stats | Stats leaders (ERA, K, W) | Basic | **pybaseball: FIP, xFIP, SIERA** |
| Starting pitchers | Probable pitcher from schedule | Good | Add actual confirmation check |
| Injuries | ESPN API | Good | Add MLB API transactions |
| Park factors | Only Coors altitude | Minimal | **Full 30-park factor table** |
| Weather | None | Missing | **Open-Meteo + MLB API hydrate** |
| Lineups | None | Missing | MLB API lineup hydration |
| Odds/market | None | Missing | **ESPN embedded + The Odds API** |
| Bullpen usage | None | Missing | Box score pitch counts |
| Platoon splits | None | Missing | pybaseball batting_stats splits |

### Prioritized Implementation Roadmap

#### Phase 1: HIGHEST IMPACT (Expected +1-3% accuracy)

1. **Add pybaseball to requirements.txt**
   ```
   pybaseball>=2.3
   ```

2. **Upgrade data_players.py to use FanGraphs stats via pybaseball**
   - Replace `stats_leaders` batting with `batting_stats()` -> wRC+, wOBA
   - Replace `stats_leaders` pitching with `pitching_stats()` -> FIP, xFIP, SIERA
   - Compute team-level aggregates from individual player stats

3. **Add park factors to elo_model.py**
   - Hardcode the 30-park factor table (update annually)
   - Adjust Elo prediction: high-PF parks increase variance (more upsets)
   - Park-neutralize team stats before comparison

#### Phase 2: MEDIUM IMPACT (Expected +0.5-1% accuracy)

4. **Add weather data**
   - Use MLB API `hydrate=weather` for today's games (free, no auth)
   - Use Open-Meteo archive API for historical backtesting
   - Only apply to outdoor parks (skip domes/retractable-closed)
   - Key features: temperature, wind speed, wind direction (in/out)

5. **Add market data as calibration benchmark**
   - Pull ESPN embedded odds for today's games
   - Compare model probability vs market-implied probability
   - Optional: blend model with market (60/40)

6. **Upgrade starting pitcher evaluation**
   - Current: pitcher cumulative Elo rating
   - Add: pitcher's FIP/xFIP from pybaseball
   - Add: recent form (last 3-5 starts) from game logs

#### Phase 3: LOWER IMPACT BUT VALUABLE (Expected +0.2-0.5% accuracy)

7. **Add lineup data for same-day predictions**
   - Pull confirmed lineups ~2 hours before game
   - Compute lineup wRC+ (sum of starter wRC+ values)
   - Detect rest days for key players

8. **Add bullpen usage tracking**
   - Track pitch counts from recent box scores
   - Fatigued bullpen (high recent usage) = vulnerability
   - Key for predicting late-inning collapses

9. **Add platoon splits**
   - Team batting wRC+ vs LHP vs RHP
   - Match against opposing starter handedness
   - LHB-heavy lineups struggle vs elite LHP, etc.

### Python Library Summary

| Library | Install | Auth | What It Provides |
|---------|---------|------|-----------------|
| `MLB-StatsAPI` | `pip install MLB-StatsAPI` | None | Game results, schedule, rosters (ALREADY HAVE) |
| `pybaseball` | `pip install pybaseball` | None | FanGraphs stats, Statcast, park factors, Lahman (ADD THIS) |
| `requests` | `pip install requests` | None | ESPN API, Open-Meteo, Odds API (ALREADY HAVE) |

### Quick Test Script

```python
"""Test all MLB data sources."""

# 1. MLB Stats API (already working)
import statsapi
today = statsapi.schedule()
print(f"[OK] MLB Stats API: {len(today)} games today")

# 2. pybaseball (ADD THIS)
from pybaseball import pitching_stats, batting_stats, team_batting, park_factors
pitchers = pitching_stats(2025, qual=30)
print(f"[OK] pybaseball pitching: {len(pitchers)} pitchers with FIP/xFIP/SIERA")
batters = batting_stats(2025, qual=50)
print(f"[OK] pybaseball batting: {len(batters)} batters with wRC+/wOBA")
tb = team_batting(2025)
print(f"[OK] Team batting: {len(tb)} teams")
pf = park_factors(2025)
print(f"[OK] Park factors: {len(pf)} parks")

# 3. Open-Meteo weather (no auth!)
import requests
r = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": 40.829, "longitude": -73.926,
    "hourly": "temperature_2m,wind_speed_10m",
    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"
})
print(f"[OK] Open-Meteo: {len(r.json()['hourly']['time'])} hours forecast")

# 4. ESPN odds (no auth!)
r = requests.get("https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard")
events = r.json().get("events", [])
odds_count = sum(1 for e in events for c in e.get("competitions", []) if c.get("odds"))
print(f"[OK] ESPN: {len(events)} games, {odds_count} with odds")

# 5. MLB API weather
r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
    "sportId": 1, "hydrate": "weather"
})
games = [g for d in r.json().get("dates", []) for g in d.get("games", [])]
weather_count = sum(1 for g in games if g.get("weather"))
print(f"[OK] MLB weather: {len(games)} games, {weather_count} with weather data")
```

---

## APPENDIX: STAT GLOSSARY

| Stat | Full Name | Scale | Good | Great | Elite |
|------|-----------|-------|------|-------|-------|
| wRC+ | Weighted Runs Created Plus | 100=avg | 110 | 130 | 150+ |
| wOBA | Weighted On-Base Average | .320=avg | .350 | .380 | .400+ |
| FIP | Fielding Independent Pitching | ~3.80=avg | 3.50 | 3.00 | 2.50 |
| xFIP | Expected FIP | ~3.80=avg | 3.50 | 3.00 | 2.50 |
| SIERA | Skill-Interactive ERA | ~3.80=avg | 3.50 | 3.00 | 2.50 |
| ISO | Isolated Power | .150=avg | .180 | .220 | .270+ |
| K-BB% | Strikeout minus Walk Rate | 12%=avg | 18% | 25% | 30%+ |
| Barrel% | Barrels / Batted Ball Events | 7%=avg | 10% | 14% | 18%+ |
| xwOBA | Expected wOBA (Statcast) | .320=avg | .350 | .380 | .400+ |
| WAR | Wins Above Replacement | 0=repl. | 2 | 4 | 6+ |
| CSW% | Called Strike + Whiff % | 28%=avg | 30% | 33% | 36%+ |
