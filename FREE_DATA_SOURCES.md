# Free Public Data Sources for Sports Prediction Systems
# NBA / NFL / NHL / MLB
# Compiled for PredictMarkets project

---

## TABLE OF CONTENTS

1. [MLB Data Sources](#1-mlb-data-sources)
2. [NBA Data Sources](#2-nba-data-sources)
3. [NFL Data Sources](#3-nfl-data-sources)
4. [NHL Data Sources](#4-nhl-data-sources)
5. [Cross-Sport: Betting/Odds Data](#5-cross-sport-bettingodds-data)
6. [Cross-Sport: Weather Data](#6-cross-sport-weather-data)
7. [Cross-Sport: Venue/Environmental Data](#7-cross-sport-venueenvironmental-data)
8. [Cross-Sport: Schedule/Fatigue Data](#8-cross-sport-schedulefatigue-data)
9. [Multi-Sport Python Packages](#9-multi-sport-python-packages)
10. [Reference Websites (Scraping Targets)](#10-reference-websites-scraping-targets)
11. [Data Source Priority Matrix](#11-data-source-priority-matrix)

---

## 1. MLB DATA SOURCES

### 1.1 MLB Stats API (Official, FREE, No Auth)

- **URL**: `https://statsapi.mlb.com/api/v1/`
- **Python Package**: `MLB-StatsAPI` (pip install MLB-StatsAPI)
- **Auth**: NONE required -- completely free and open
- **Rate Limits**: No published limits, but be respectful (1-2 req/sec)
- **Already in your project**: Yes (data_games.py, data_players.py)

**Key Endpoints**:

```python
import statsapi

# Schedule with scores
statsapi.schedule(start_date='2025-04-01', end_date='2025-04-30')

# Box score (detailed game-level stats)
statsapi.boxscore(gamePk=717540)

# Play-by-play
statsapi.get('game_playByPlay', {'gamePk': 717540})

# Linescore (inning-by-inning)
statsapi.linescore(gamePk=717540)

# Team stats
statsapi.team_leaders(teamId=147, leaderCategories='homeRuns', season=2025)

# Player stats
statsapi.player_stat_data(playerid=660271, group='hitting', type='season')

# Standings
statsapi.standings(leagueId=103, season=2025)  # AL=103, NL=104

# Probable pitchers (PRE-GAME -- critical for predictions)
schedule = statsapi.schedule(date='2025-06-15')
# Each game dict includes 'away_probable_pitcher' and 'home_probable_pitcher'

# Game context data
statsapi.get('game_contextMetrics', {'gamePk': 717540})

# Roster
statsapi.roster(teamId=147, rosterType='active')
```

**Raw API endpoints (no package needed)**:
```
GET https://statsapi.mlb.com/api/v1/schedule?date=2025-06-15&sportId=1&hydrate=probablePitcher,linescore,team
GET https://statsapi.mlb.com/api/v1/game/{gamePk}/feed/live
GET https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore
GET https://statsapi.mlb.com/api/v1/game/{gamePk}/playByPlay
GET https://statsapi.mlb.com/api/v1/teams/{teamId}/stats?stats=season&group=hitting&season=2025
GET https://statsapi.mlb.com/api/v1/people/{playerId}/stats?stats=season&group=pitching&season=2025
GET https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live   # v1.1 has more detail
GET https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025
GET https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2025-04-01&endDate=2025-04-07&hydrate=weather
```

**What this adds to your model**:
- Probable pitchers (you already use pitcher Elo, but the schedule endpoint gives them for upcoming games)
- Box score detail: pitch counts, LOB, RISP -- could feed XGBoost features
- Play-by-play: exit velocity, launch angle, sprint speed per at-bat
- Weather data via hydrate parameter on schedule endpoint

---

### 1.2 Baseball Savant / Statcast (FREE, No Auth)

- **URL**: `https://baseballsavant.mlb.com`
- **CSV Export URL**: `https://baseballsavant.mlb.com/statcast_search/csv`
- **Auth**: NONE
- **Python Access**: `pybaseball` package

**What it provides**:
- Exit velocity, launch angle, sprint speed, barrel rate
- Expected stats: xBA, xSLG, xwOBA, xERA
- Pitch tracking: spin rate, velocity, movement
- Fielding: OAA (Outs Above Average), arm strength
- Catcher framing metrics

```python
# pip install pybaseball
from pybaseball import statcast, statcast_pitcher, pitching_stats, batting_stats
from pybaseball import team_batting, team_pitching, team_fielding
from pybaseball import playerid_lookup, statcast_batter

# All Statcast data for a date range (pitch-level!)
data = statcast(start_dt='2025-06-01', end_dt='2025-06-07')
# Returns: pitch_type, release_speed, launch_speed, launch_angle,
#          estimated_ba_using_speedangle, estimated_woba_using_speedangle, etc.

# Season-level advanced stats
pitching = pitching_stats(2025, qual=50)  # FIP, xFIP, SIERA, K%, BB%, WAR
batting = batting_stats(2025, qual=100)   # wRC+, wOBA, ISO, BABIP, WAR

# Team-level
team_bat = team_batting(2025)
team_pitch = team_pitching(2025)

# Specific pitcher's Statcast data
pitcher_data = statcast_pitcher('2025-04-01', '2025-06-15', player_id=543037)
```

**Prediction improvement**:
- xwOBA differential (team batting xwOBA vs opponent pitching xwOBA) is one of the strongest single predictors
- FIP/xFIP for starting pitchers (better than ERA for predicting future performance)
- Barrel rate and hard hit rate for team strength
- Sprint speed for stolen base/manufacturing runs context

---

### 1.3 FanGraphs (FREE, No Auth for basic data)

- **URL**: `https://www.fangraphs.com`
- **API**: Leaderboards exportable via CSV
- **Python Access**: `pybaseball` package wraps FanGraphs leaderboards

**What it provides**:
- WAR (fWAR), wRC+, FIP, xFIP, SIERA, K-BB%
- Park factors (critical for adjusting stats)
- Pitch values (how much value each pitch type adds)
- Defensive metrics (DEF, UZR, DRS)
- Baserunning metrics (BsR)
- Team-level advanced stats

```python
from pybaseball import pitching_stats, batting_stats
# These pull from FanGraphs by default

# FanGraphs park factors
from pybaseball import park_factors
pf = park_factors(2025)  # Park factor for each stadium
# Coors Field ~115, Oracle Park ~95, etc.

# Custom FanGraphs leaderboard URL (CSV export):
# https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=8&season=2025&month=0&season1=2025&ind=0&team=0,ts&rost=0&age=0&filter=&players=0&page=1_50
# Add &content=csv to any leaderboard URL to get CSV
```

**Prediction improvement**:
- Park factors: adjust all stats for venue (HUGE for MLB)
- wRC+ is park-adjusted and league-adjusted -- the best single offensive metric
- SIERA (Skill-Interactive ERA) is the best ERA predictor
- WAR for overall team strength composite

---

### 1.4 Baseball-Reference (Scraping)

- **URL**: `https://www.baseball-reference.com`
- **Auth**: None (but rate-limit conscious; they block aggressive scrapers)
- **Python**: `pybaseball` has some B-Ref endpoints; otherwise `requests` + `BeautifulSoup`

**What it provides**:
- Game logs for every player and team
- Historical splits (home/away, vs L/R, day/night, by month)
- Pythagorean W-L records
- Adjusted stats (OPS+, ERA+)
- Bullpen usage logs
- Complete transaction/injury history

```python
from pybaseball import schedule_and_record
# Team's game-by-game results
team_schedule = schedule_and_record(2025, 'NYY')

# Or scrape directly:
# https://www.baseball-reference.com/teams/NYY/2025-schedule-scores.shtml
# https://www.baseball-reference.com/leagues/majors/2025-standard-pitching.shtml
```

**Prediction improvement**:
- Splits data (vs LHP/RHP) for lineup-based adjustments
- Bullpen rest/workload tracking
- ERA+ is park/league adjusted -- use for pitcher comparison

---

### 1.5 pybaseball (Meta-Package, FREE)

- **Package**: `pip install pybaseball`
- **Wraps**: Statcast, FanGraphs, Baseball-Reference, Retrosheet, Chadwick Bureau
- **Auth**: NONE

**Complete function list relevant to prediction**:
```python
from pybaseball import (
    # Statcast
    statcast,                    # All pitch-level data for date range
    statcast_pitcher,            # Pitcher's Statcast data
    statcast_batter,             # Batter's Statcast data
    statcast_running,            # Sprint speed data
    statcast_catcher,            # Catcher pop time + framing

    # FanGraphs leaderboards
    batting_stats,               # Season batting (wRC+, WAR, wOBA, etc.)
    pitching_stats,              # Season pitching (FIP, xFIP, SIERA, WAR)
    fielding_stats,              # Fielding (UZR, DRS, OAA)

    # Team-level
    team_batting,                # Team batting stats
    team_pitching,               # Team pitching stats
    team_fielding,               # Team fielding stats

    # Game-level
    schedule_and_record,         # Team's game log with results

    # Park factors
    park_factors,                # Venue adjustments

    # Retrosheet (historical)
    retrosheet,                  # Decades of play-by-play data

    # Player lookup
    playerid_lookup,             # Name -> ID mapping
    playerid_reverse_lookup,     # ID -> name mapping
)
```

---

### 1.6 ESPN MLB API (FREE, No Auth)

- **URL**: `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/`
- **Auth**: NONE
- **Already partially in your project**: injuries.py uses this

```python
import requests

# Scoreboard (today's games with odds!)
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard')
games = r.json()['events']

# Team info
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/nyy')

# Team schedule
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/nyy/schedule?season=2025')

# Injuries (you already use this)
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries')

# News
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/news')

# BPI/Power Index (if available)
# https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/powerindex
```

**Prediction improvement**:
- ESPN scoreboard includes betting odds (spread, moneyline, over/under) embedded in the JSON
- Injury data with return dates (you already use this)

---

## 2. NBA DATA SOURCES

### 2.1 nba_api Python Package (Official NBA Stats, FREE, No Auth)

- **Package**: `pip install nba_api`
- **Base URL**: `https://stats.nba.com/stats/`
- **Auth**: NONE, but requires specific headers (User-Agent, Referer)
- **Rate Limits**: Aggressive rate limiting; add 0.6s delay between calls

```python
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
    teamgamelog,
    leaguestandingsv3,
    teamdashboardbygeneralsplits,
    playergamelog,
    commonteamroster,
    scoreboardv2,
    teamestimatedmetrics,
    leaguedashteamstats,
)
from nba_api.stats.static import teams, players

# All NBA teams
nba_teams = teams.get_teams()

# Team game log (full season results)
gamelog = teamgamelog.TeamGameLog(team_id=1610612747, season='2025-26')
df = gamelog.get_data_frames()[0]

# Box score (traditional)
box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id='0022500001')
team_stats = box.team_stats.get_data_frame()

# Box score (advanced -- has OffRtg, DefRtg, Pace, eFG%, etc.)
adv = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id='0022500001')
team_adv = adv.team_stats.get_data_frame()

# League-wide team stats (season-level)
league = leaguedashteamstats.LeagueDashTeamStats(season='2025-26')
all_teams = league.get_data_frames()[0]
# Columns include: W, L, W_PCT, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT,
#   FTM, FTA, FT_PCT, OREB, DREB, REB, AST, TOV, STL, BLK, PF, PTS,
#   PLUS_MINUS, etc.

# Advanced team stats (Net Rating, Pace, Four Factors)
league_adv = leaguedashteamstats.LeagueDashTeamStats(
    season='2025-26',
    measure_type_detailed_defense='Advanced'
)
# Columns: OFF_RATING, DEF_RATING, NET_RATING, PACE, AST_PCT, AST_TO,
#   OREB_PCT, DREB_PCT, REB_PCT, EFG_PCT, TS_PCT, TM_TOV_PCT

# Estimated metrics (NBA's own advanced stats)
est = teamestimatedmetrics.TeamEstimatedMetrics(season='2025-26')
# E_OFF_RATING, E_DEF_RATING, E_NET_RATING, E_PACE

# Scoreboard (today's games)
sb = scoreboardv2.ScoreboardV2(game_date='2025-12-25')

# Player game log
pgl = playergamelog.PlayerGameLog(player_id=2544, season='2025-26')

# IMPORTANT: Custom headers often needed
from nba_api.stats.endpoints import leaguedashteamstats
# The package handles headers internally, but if raw requests:
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Referer': 'https://www.nba.com/',
    'Accept': 'application/json',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
}
```

**Prediction improvement**:
- Net Rating (OffRtg - DefRtg) is the single best team strength metric for NBA
- Four Factors (eFG%, TOV%, OREB%, FT rate) explain ~95% of point differential
- Pace data for over/under and tempo matchup adjustments
- Home/away splits for home court advantage modeling

---

### 2.2 NBA Advanced Stats Endpoints (FREE, via nba_api)

```python
from nba_api.stats.endpoints import (
    teamdashboardbygeneralsplits,  # Home/Away, W/L, monthly splits
    teamdashboardbyshootingsplits, # Shot distance, shot type breakdowns
    teamdashboardbyclutch,         # Clutch performance (last 5 min, within 5 pts)
    teamdashboardbyopponent,       # Stats against each opponent
    teamdashboardbylastngames,     # Last N games rolling performance
    teamvsplayer,                  # Team performance with/without specific player
    playerdashboardbygeneralsplits, # Player home/away, pre/post ASB
    leaguehustlestatsplayer,       # Hustle stats: deflections, loose balls, charges
    leaguehustlestatsteam,         # Team hustle stats
    boxscoreplayertrackv2,         # Player tracking: speed, distance, touches
    boxscoremiscv2,                # Misc: pts off TO, 2nd chance, fast break
    boxscorescoringv2,             # Scoring: %AST, %UAST, %FGM at rim/mid/3pt
)

# Clutch stats (last 5 min, score within 5)
clutch = teamdashboardbyclutch.TeamDashboardByClutch(
    team_id=1610612747, season='2025-26'
)

# Shooting splits (by distance)
shooting = teamdashboardbyshootingsplits.TeamDashboardByShootingSplits(
    team_id=1610612747, season='2025-26'
)

# Last 10 games (rolling form)
recent = teamdashboardbylastngames.TeamDashboardByLastNGames(
    team_id=1610612747, season='2025-26'
)
```

**Prediction improvement**:
- Clutch ratings identify teams that perform differently in close games
- Rolling last-10 form captures hot/cold streaks
- Home/away splits reveal true home court advantage per team
- Opponent-adjusted stats for strength of schedule

---

### 2.3 nba_api Live/Real-Time Endpoints

```python
from nba_api.live.nba.endpoints import scoreboard, boxscore

# Today's live scoreboard
sb = scoreboard.ScoreBoard()
games = sb.get_dict()

# Live box score
bs = boxscore.BoxScore(game_id='0022500001')
```

---

### 2.4 Basketball-Reference (Scraping)

- **URL**: `https://www.basketball-reference.com`
- **Auth**: None (but rate limit: ~20 requests/minute, they use Cloudflare)
- **Python**: `basketball_reference_web_scraper` package or direct scraping

```python
# pip install basketball-reference-web-scraper
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType

# Season schedule with results
schedule = client.season_schedule(season_end_year=2026)

# Player box scores for a date
box = client.player_box_scores(day=25, month=12, year=2025)

# Player season totals
totals = client.players_season_totals(season_end_year=2026)

# Advanced stats (PER, TS%, BPM, VORP, WS, WS/48)
advanced = client.players_advanced_season_totals(season_end_year=2026)
```

**Data available via scraping**:
```
# Team stats page (per game, totals, per 100 poss, advanced)
https://www.basketball-reference.com/leagues/NBA_2026.html

# Team game log
https://www.basketball-reference.com/teams/LAL/2026/gamelog/

# Player game log
https://www.basketball-reference.com/players/j/jamesle01/gamelog/2026/

# Injuries
https://www.basketball-reference.com/friv/injuries.fcgi

# Four Factors
https://www.basketball-reference.com/leagues/NBA_2026.html#all_misc_stats
# Columns: Pace, eFG%, TOV%, ORB%, FT/FGA (the Dean Oliver Four Factors)

# Schedule with rest days visible
https://www.basketball-reference.com/leagues/NBA_2026_games.html
```

**Prediction improvement**:
- BPM (Box Plus/Minus) and VORP for player-level impact
- Four Factors at team level are the best team comparison metrics
- PER and WS/48 for identifying key player injuries' impact

---

### 2.5 ESPN NBA API (FREE, No Auth)

```python
import requests

# Scoreboard (includes odds!)
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard')
data = r.json()

# Team details
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/lal')

# Team schedule
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/lal/schedule?season=2026')

# Injuries
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries')

# NBA BPI (Basketball Power Index)
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/powerindex')

# Predictions (ESPN's own win probabilities)
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20251225')
# Each event has 'competitions[0].odds' with spread, ML, O/U
# Each event has 'competitions[0].predictor' with ESPN's win probability
```

**Prediction improvement**:
- ESPN BPI is a useful benchmark/feature (ensemble of public models often beats individual ones)
- Embedded odds in scoreboard JSON for line shopping and market consensus
- Injury status directly in game data

---

### 2.6 Cleaning the Glass (Paid, but reference)

- **URL**: `https://cleaningtheglass.com`
- **Auth**: Paid subscription ($10/mo)
- **Note**: Mentioned for completeness; the underlying data (play-type frequency, lineup stats) can be derived from nba_api

---

## 3. NFL DATA SOURCES

### 3.1 nfl_data_py (nflverse ecosystem, FREE, No Auth)

- **Package**: `pip install nfl_data_py`
- **Data Source**: nflverse GitHub repositories (CSV/parquet files on GitHub)
- **Auth**: NONE
- **THE premier source for NFL data**

```python
import nfl_data_py as nfl

# Play-by-play data (THE most valuable NFL dataset)
# Contains EPA, WPA, air yards, CPOE, and 300+ columns per play
pbp = nfl.import_pbp_data([2025])
# Key columns: epa, wpa, cpoe, air_yards, yards_after_catch,
#   pass_oe (pass rate over expected), xpass, posteam, defteam,
#   down, ydstogo, yardline_100, score_differential, etc.

# Seasonal stats (player-level)
seasonal = nfl.import_seasonal_data([2025])

# Weekly stats (player-level, per game)
weekly = nfl.import_weekly_data([2025])

# Roster data (position, height, weight, age, draft info)
roster = nfl.import_rosters([2025])

# Team schedules with results
schedules = nfl.import_schedules([2025])
# Includes: result, total, spread_line, total_line, roof, surface, temp, wind

# Win totals (preseason)
win_totals = nfl.import_win_totals([2025])

# Draft data
draft = nfl.import_draft_picks([2025])

# Injuries
injuries = nfl.import_injuries([2025])

# Snap counts
snaps = nfl.import_snap_counts([2025])

# QBR data
qbr = nfl.import_qbr([2025])

# Next Gen Stats (NFL's player tracking)
ngs_passing = nfl.import_ngs_data(stat_type='passing', years=[2025])
ngs_rushing = nfl.import_ngs_data(stat_type='rushing', years=[2025])
ngs_receiving = nfl.import_ngs_data(stat_type='receiving', years=[2025])

# PFR (Pro Football Reference) season-level
pfr_pass = nfl.import_seasonal_pfr(s_type='pass', years=[2025])
pfr_rush = nfl.import_seasonal_pfr(s_type='rush', years=[2025])
pfr_recv = nfl.import_seasonal_pfr(s_type='rec', years=[2025])

# Combine data
combine = nfl.import_combine_data([2020, 2021, 2022, 2023, 2024, 2025])

# Player IDs (cross-reference between sources)
ids = nfl.import_ids()

# Officials data
officials = nfl.import_officials([2025])

# Depth charts
depth = nfl.import_depth_charts([2025])
```

**Prediction improvement -- this is GOLD**:
- **EPA (Expected Points Added)**: The single best play-level efficiency metric for NFL
  - Offensive EPA/play and Defensive EPA/play allowed are the strongest predictors
  - EPA is already adjusted for down, distance, field position, score
- **CPOE (Completion Probability Over Expected)**: QB skill metric
- **Success rate**: % of plays with positive EPA
- **Pass rate over expected (PROE)**: Play-calling tendency
- **Schedule data includes weather, surface, and roof** -- no separate weather API needed for NFL!
- **Snap counts**: Identify key player workload and usage
- **Injuries with game status**: Questionable/Doubtful/Out

---

### 3.2 ESPN NFL API (FREE, No Auth)

```python
import requests

# Scoreboard (includes odds and ESPN FPI predictions!)
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard')
data = r.json()
# Events contain: competitions[0].odds, competitions[0].predictor (ESPN FPI win prob)

# Team stats
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/ne')

# FPI (Football Power Index) -- ESPN's proprietary prediction model
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/powerindex')

# Injuries
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries')

# Schedule
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/ne/schedule?season=2025')

# Standings
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/standings')
```

---

### 3.3 Pro-Football-Reference (Scraping)

- **URL**: `https://www.pro-football-reference.com`
- **Auth**: None (rate limited like all Sports Reference sites)

**Key pages**:
```
# Team stats (offensive/defensive)
https://www.pro-football-reference.com/years/2025/

# Advanced passing
https://www.pro-football-reference.com/years/2025/passing_advanced.htm

# Advanced rushing
https://www.pro-football-reference.com/years/2025/rushing_advanced.htm

# Advanced receiving
https://www.pro-football-reference.com/years/2025/receiving_advanced.htm

# Team game log
https://www.pro-football-reference.com/teams/nwe/2025/gamelog/

# Snap counts
https://www.pro-football-reference.com/years/2025/fantasy.htm

# Team defense
https://www.pro-football-reference.com/years/2025/opp.htm

# Weekly results
https://www.pro-football-reference.com/years/2025/week_1.htm
```

**Prediction improvement**:
- DVOA-equivalent stats (PFR's advanced metrics approach)
- Strength of schedule calculations
- Player Approximate Value (AV) for injury impact estimation

---

### 3.4 NFL Official Stats API

- **URL**: `https://site.api.espn.com` is the best proxy; the official NFL API has been increasingly restricted
- **NFL Next Gen Stats**: Available through nfl_data_py (see section 3.1)

---

## 4. NHL DATA SOURCES

### 4.1 NHL Official API (FREE, No Auth)

- **URL**: `https://api-web.nhle.com/v1/` (new API, replaced old statsapi.web.nhl.com)
- **Auth**: NONE
- **Rate Limits**: Reasonable; no published limits

```python
import requests

BASE = 'https://api-web.nhle.com/v1'

# Today's scoreboard
r = requests.get(f'{BASE}/score/now')

# Schedule for specific date
r = requests.get(f'{BASE}/schedule/2025-12-25')

# Team schedule/season
r = requests.get(f'{BASE}/club-schedule-season/TOR/20252026')

# Standings
r = requests.get(f'{BASE}/standings/now')

# Team stats (season)
r = requests.get(f'{BASE}/club-stats/TOR/20252026/2')  # 2=regular season

# Player stats leaders
r = requests.get(f'{BASE}/skater-stats-leaders/20252026/2?categories=goals&limit=50')
r = requests.get(f'{BASE}/goalie-stats-leaders/20252026/2?categories=wins&limit=50')

# Game landing (box score equivalent)
r = requests.get(f'{BASE}/gamecenter/{gameId}/landing')

# Game box score
r = requests.get(f'{BASE}/gamecenter/{gameId}/boxscore')

# Game play-by-play
r = requests.get(f'{BASE}/gamecenter/{gameId}/play-by-play')

# Player profile/stats
r = requests.get(f'{BASE}/player/8478402/landing')  # Connor McDavid

# Team roster
r = requests.get(f'{BASE}/roster/TOR/20252026')

# Player game log
r = requests.get(f'{BASE}/player/8478402/game-log/20252026/2')

# Draft rankings, prospects
r = requests.get(f'{BASE}/draft/rankings/now')

# Playoff bracket
r = requests.get(f'{BASE}/playoff-bracket/20252026')

# Older API (still works for some endpoints):
OLD_BASE = 'https://statsapi.web.nhl.com/api/v1'
r = requests.get(f'{OLD_BASE}/schedule?date=2025-12-25&expand=schedule.linescore')
```

**Prediction improvement**:
- Starting goalie info (critical -- goalie is ~50% of a team's save ability)
- Game score context (rest, travel) from schedule data
- Player stats for injury impact calculation

---

### 4.2 Natural Stat Trick (FREE, No Auth for basic data)

- **URL**: `https://www.naturalstattrick.com`
- **Auth**: None for basic; premium features are paid
- **Access**: CSV download via URL manipulation

```python
import requests
import pandas as pd

# Team 5v5 stats (the gold standard for NHL analytics)
# sit=5v5, score=all, rate=y (per 60 minutes)
url = ('https://www.naturalstattrick.com/teamtable.php'
       '?fromseason=20252026&thruseason=20252026'
       '&stype=2&sit=5v5&score=all&rate=y&team=all'
       '&loc=B&gpf=410&fd=&td=')
# Add '&csv=1' to get CSV format (but check if still available publicly)

# Key stats available:
# CF%, FF%, xGF%, SCF%, HDCF% (high-danger chances for%)
# GF/60, GA/60, xGF/60, xGA/60
# Sh%, Sv%, PDO (shooting% + save%)
# HDGF/60, HDGA/60 (high-danger goals for/against per 60)

# Individual player stats at 5v5
# https://www.naturalstattrick.com/playerteams.php?...

# Goalie stats
# https://www.naturalstattrick.com/goalie.php?...
```

**Prediction improvement**:
- **xGF% (Expected Goals For %)** at 5v5 is THE best team quality metric in NHL
- **HDCF% (High-Danger Chance For %)** is the most predictive shot quality metric
- **PDO (Sh% + Sv%)** regresses hard to 100 -- teams with extreme PDO regress
- 5v5 data only (strips out power play noise) for cleaner signal

---

### 4.3 Evolving Hockey (FREE tier)

- **URL**: `https://evolving-hockey.com`
- **Auth**: Some free data; full access is paid
- **Provides**: GAR (Goals Above Replacement), xGAR, WAR for NHL players

---

### 4.4 MoneyPuck (FREE, CSV Downloads!)

- **URL**: `https://moneypuck.com`
- **Data download**: `https://moneypuck.com/data.htm`
- **Auth**: NONE -- all CSVs freely downloadable

```python
import pandas as pd

# All shots for a season (includes xG model!)
shots = pd.read_csv('https://peter-tanner.com/moneypuck/downloads/shots_2025.csv')
# Or: https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters.csv

# Skater stats
skaters = pd.read_csv('https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters.csv')

# Goalie stats
goalies = pd.read_csv('https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv')

# Team stats
teams = pd.read_csv('https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/teams.csv')

# Key columns in skater data: xGoals, goals, xGoalsFor, xGoalsAgainst,
#   iceTime, onIce_corsiFor, onIce_corsiAgainst, etc.

# Key columns in goalie data: xGoals (expected goals against), goals (actual),
#   GSAA (Goals Saved Above Average), saves, shots, etc.
```

**Prediction improvement**:
- **GSAA (Goals Saved Above Average)** for goalie quality -- critical for NHL prediction
- **xG model** applied to every shot -- aggregate for team expected goal differential
- Free CSV bulk downloads -- easy to integrate

---

### 4.5 hockey-scraper / hockey_scraper (Python Package)

- **Package**: `pip install hockey-scraper`
- **Auth**: NONE
- **Source**: Scrapes NHL play-by-play from NHL.com and ESPN

```python
import hockey_scraper

# Play-by-play for a date range
pbp = hockey_scraper.scrape_date_range('2025-10-15', '2025-10-20', if_scrape_shifts=True)
# Returns DataFrame with every event: shots, goals, hits, blocks, faceoffs
# Includes x/y coordinates for shot location!

# Specific game
game = hockey_scraper.scrape_games([2025020001], if_scrape_shifts=True)

# Shifts data (player time-on-ice logs)
# Included when if_scrape_shifts=True
```

**Prediction improvement**:
- Shot location data for building your own xG model
- Shift data for fatigue analysis within games
- Event-level data for Corsi/Fenwick calculations

---

### 4.6 Hockey-Reference (Scraping)

- **URL**: `https://www.hockey-reference.com`
- **Auth**: None (rate limited)

```
# Team stats
https://www.hockey-reference.com/leagues/NHL_2026.html

# Team game log
https://www.hockey-reference.com/teams/TOR/2026_gamelog.html

# Advanced stats (Corsi, Fenwick, PDO, xG)
https://www.hockey-reference.com/leagues/NHL_2026.html#all_stats_adv

# Goalie stats
https://www.hockey-reference.com/leagues/NHL_2026_goalies.html

# Player stats
https://www.hockey-reference.com/players/m/mcdavco01.html
```

---

### 4.7 ESPN NHL API (FREE, No Auth)

```python
import requests

# Scoreboard
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard')

# Team
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/tor')

# Injuries
r = requests.get('https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries')
```

---

## 5. CROSS-SPORT: BETTING/ODDS DATA

### 5.1 The Odds API (FREE Tier, Auth Required)

- **URL**: `https://the-odds-api.com`
- **Auth**: Free API key (500 requests/month free tier)
- **Covers**: NBA, NFL, NHL, MLB (and many other sports)

```python
import requests

API_KEY = 'your_free_api_key'

# List available sports
r = requests.get(f'https://api.the-odds-api.com/v4/sports/?apiKey={API_KEY}')

# Get odds for NBA
r = requests.get(
    f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/',
    params={
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',  # moneyline, spread, over/under
        'oddsFormat': 'american',
    }
)
odds = r.json()

# Historical odds (also available)
r = requests.get(
    f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds-history/',
    params={
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'date': '2025-12-25T00:00:00Z',
    }
)

# Sports keys:
# basketball_nba, americanfootball_nfl, icehockey_nhl, baseball_mlb
```

**Prediction improvement**:
- Closing line value (CLV) is the best measure of prediction skill
- Multiple bookmakers for consensus/sharp line identification
- Historical odds for backtest validation against market

---

### 5.2 ESPN Embedded Odds (FREE, No Auth)

Already covered above -- every ESPN scoreboard endpoint includes odds data:
```python
# All four sports scoreboard URLs include odds
sports = [
    'baseball/mlb', 'basketball/nba', 'football/nfl', 'hockey/nhl'
]
for sport in sports:
    r = requests.get(f'https://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard')
    for event in r.json().get('events', []):
        for comp in event.get('competitions', []):
            odds = comp.get('odds', [])
            # Contains: spread, overUnder, moneyline home/away
```

---

### 5.3 Sportsbook Review (SBR) Historical Lines

- **URL**: `https://www.sportsbookreview.com`
- **Auth**: None (scraping)
- **Provides**: Historical opening and closing lines from multiple books

```
# Opening/closing lines, line movement
https://www.sportsbookreview.com/betting-odds/nba-basketball/
https://www.sportsbookreview.com/betting-odds/nfl-football/
https://www.sportsbookreview.com/betting-odds/nhl-hockey/
https://www.sportsbookreview.com/betting-odds/mlb-baseball/

# Historical (date-specific)
https://www.sportsbookreview.com/betting-odds/nba-basketball/20251225/
```

---

### 5.4 Odds Portal

- **URL**: `https://www.oddsportal.com`
- **Auth**: None (heavy JS rendering, needs Selenium)
- **Provides**: Historical odds from dozens of bookmakers

---

### 5.5 Action Network / DraftKings / FanDuel APIs

These are not truly "free APIs" but their public-facing odds pages can be scraped:
```
# DraftKings odds API (public, powers their website)
https://sportsbook-nash-usdc.draftkings.com/sites/US-DC-SB/api/v5/eventgroups/88808/categories/487?format=json

# FanDuel (similar public JSON endpoints power their web UI)
```
Note: These endpoints can change without notice and may have legal restrictions.

---

### 5.6 nfl_data_py Includes Historical Lines

```python
import nfl_data_py as nfl
schedules = nfl.import_schedules([2024, 2025])
# Columns include: spread_line, total_line, result, overtime
# This gives you historical Vegas lines for every NFL game!
```

---

## 6. CROSS-SPORT: WEATHER DATA

### 6.1 OpenWeatherMap (FREE Tier, Auth Required)

- **URL**: `https://openweathermap.org/api`
- **Auth**: Free API key (1,000 calls/day, 60 calls/min)
- **Relevant for**: MLB (outdoor), NFL (outdoor/some domes)

```python
import requests

API_KEY = 'your_free_key'

# Current weather
r = requests.get(
    'https://api.openweathermap.org/data/2.5/weather',
    params={'lat': 40.829, 'lon': -73.926, 'appid': API_KEY, 'units': 'imperial'}
)
weather = r.json()
# temp, humidity, wind speed/direction, rain, snow, clouds

# 5-day forecast (3-hour intervals) -- for upcoming games
r = requests.get(
    'https://api.openweathermap.org/data/2.5/forecast',
    params={'lat': 40.829, 'lon': -73.926, 'appid': API_KEY, 'units': 'imperial'}
)

# Historical weather (requires One Call API 3.0, first 1000 calls/day free)
r = requests.get(
    'https://api.openweathermap.org/data/3.0/onecall/timemachine',
    params={'lat': 40.829, 'lon': -73.926, 'dt': 1703462400, 'appid': API_KEY}
)
```

**Prediction improvement (MLB)**:
- Wind direction + speed at Wrigley Field: massive HR impact
- Temperature: ball carries further in heat
- Humidity: minimal effect despite myth
- Rain probability: lineup changes, shortened games

**Prediction improvement (NFL)**:
- Wind > 15 mph: reduces passing game efficiency, depresses scoring
- Cold < 20F: slightly favors running game
- Rain/snow: reduces passing completion rate
- Note: nfl_data_py already includes temp/wind for NFL games

---

### 6.2 Visual Crossing Weather API (FREE Tier)

- **URL**: `https://www.visualcrossing.com/weather-api`
- **Auth**: Free key (1,000 records/day)
- **Advantage**: Historical weather data for free (OpenWeatherMap charges for historical)

```python
import requests

API_KEY = 'your_free_key'
r = requests.get(
    'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    'New%20York/2025-06-15/2025-06-15',
    params={'unitGroup': 'us', 'key': API_KEY, 'include': 'hours'}
)
# Returns hourly temp, humidity, wind, precip, conditions
```

---

### 6.3 Open-Meteo (FREE, No Auth!)

- **URL**: `https://open-meteo.com`
- **Auth**: NONE -- completely free, no API key needed
- **Rate Limit**: 10,000 requests/day
- **THE BEST free weather API for sports prediction**

```python
import requests

# Current + forecast
r = requests.get(
    'https://api.open-meteo.com/v1/forecast',
    params={
        'latitude': 40.829,  # Yankee Stadium
        'longitude': -73.926,
        'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,rain,snowfall,cloud_cover',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'America/New_York',
    }
)
weather = r.json()

# Historical weather (also free!)
r = requests.get(
    'https://archive-api.open-meteo.com/v1/archive',
    params={
        'latitude': 40.829,
        'longitude': -73.926,
        'start_date': '2025-06-15',
        'end_date': '2025-06-15',
        'hourly': 'temperature_2m,wind_speed_10m,wind_direction_10m,precipitation',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
    }
)
```

**Prediction improvement**:
- Free historical weather means you can backtest weather features
- No auth = no rate limit headaches
- Hourly granularity lets you match weather to game start times

---

### 6.4 MLB Stats API Weather (Already in your pipeline!)

The MLB Stats API schedule endpoint can include weather via hydrate:
```python
import requests
r = requests.get(
    'https://statsapi.mlb.com/api/v1/schedule',
    params={
        'sportId': 1,
        'date': '2025-06-15',
        'hydrate': 'weather',
    }
)
# Returns: condition, temp, wind, wind direction for each game
```

---

## 7. CROSS-SPORT: VENUE/ENVIRONMENTAL DATA

### 7.1 Stadium/Venue Database (Build Your Own)

No single API exists; compile from Wikipedia/team sites. Here are the key fields:

```python
# MLB Venue Data
MLB_VENUES = {
    'Coors Field':     {'team': 'COL', 'lat': 39.756, 'lon': -104.994, 'alt_ft': 5280, 'roof': 'open', 'surface': 'grass', 'capacity': 50144, 'LF': 347, 'CF': 415, 'RF': 350, 'park_factor': 115},
    'Oracle Park':     {'team': 'SF',  'lat': 37.778, 'lon': -122.389, 'alt_ft': 0,    'roof': 'open', 'surface': 'grass', 'capacity': 41915, 'LF': 339, 'CF': 399, 'RF': 309, 'park_factor': 95},
    'Yankee Stadium':  {'team': 'NYY', 'lat': 40.829, 'lon': -73.926,  'alt_ft': 55,   'roof': 'open', 'surface': 'grass', 'capacity': 46537, 'LF': 318, 'CF': 408, 'RF': 314, 'park_factor': 107},
    'Globe Life Field': {'team': 'TEX', 'lat': 32.747, 'lon': -97.084, 'alt_ft': 545,  'roof': 'retractable', 'surface': 'artificial', 'capacity': 40300},
    # ... etc for all 30 stadiums
    # Chase Field (ARI): retractable roof
    # Tropicana Field / new TB stadium: dome
    # Marlins Park (MIA): retractable roof
    # Miller Park / American Family Field (MIL): retractable roof
    # Minute Maid Park (HOU): retractable roof
    # Rogers Centre (TOR): retractable roof
    # T-Mobile Park (SEA): retractable roof
    # Nationals Park, Citizens Bank, Fenway, etc.: open air
}

# NFL Venue Data
NFL_VENUES = {
    'SoFi Stadium':       {'team': ['LAR', 'LAC'], 'roof': 'dome', 'surface': 'turf', 'lat': 33.953, 'lon': -118.339},
    'Lambeau Field':      {'team': 'GB',  'roof': 'open', 'surface': 'grass', 'lat': 44.501, 'lon': -88.062},
    'Caesars Superdome':   {'team': 'NO',  'roof': 'dome', 'surface': 'turf', 'lat': 29.951, 'lon': -90.081},
    'Empower Field':      {'team': 'DEN', 'roof': 'open', 'surface': 'grass', 'lat': 39.744, 'lon': -105.020, 'alt_ft': 5280},
    # Dome teams: ARI, ATL, DAL, DET, HOU, IND, LV, LAR/LAC, MIN, NO
    # Open air cold: BUF, CHI, CIN, CLE, DEN, GB, KC, NE, NYG/NYJ, PIT
    # Open air warm: JAX, MIA, TB, TEN
    # ... etc
}

# NHL and NBA are 100% indoors -- venue data less relevant for weather
# But arena altitude (Denver) and travel patterns still matter
```

**Prediction improvement**:
- Park factors for MLB (Coors Field adds ~15% to run scoring)
- Dome vs. open air for NFL (weather irrelevant in domes)
- Field dimensions affect HR rates in MLB
- Altitude affects ball flight (MLB) and player fatigue (all sports)
- Turf vs. grass affects injury rates and speed of play (NFL)

---

### 7.2 Travel Distance Data

```python
# Precompute city-to-city distances using coordinates
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """Miles between two coordinate pairs."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 3956 * asin(sqrt(a))  # 3956 = Earth radius in miles

# Example: Your Elo model already uses travel_factor
# This can be derived from venue lat/lon data above
```

### 7.3 Time Zones

```python
# Mapping teams to time zones for jet lag calculation
TEAM_TIMEZONES = {
    # MLB
    'NYY': 'America/New_York', 'NYM': 'America/New_York',
    'BOS': 'America/New_York', 'BAL': 'America/New_York',
    'TB':  'America/New_York', 'ATL': 'America/New_York',
    'MIA': 'America/New_York', 'PHI': 'America/New_York',
    'WSH': 'America/New_York', 'PIT': 'America/New_York',
    'CIN': 'America/New_York', 'CLE': 'America/New_York',
    'DET': 'America/New_York', 'TOR': 'America/New_York',
    'CHC': 'America/Chicago', 'CWS': 'America/Chicago',
    'MIL': 'America/Chicago', 'MIN': 'America/Chicago',
    'KC':  'America/Chicago', 'STL': 'America/Chicago',
    'HOU': 'America/Chicago', 'TEX': 'America/Chicago',
    'COL': 'America/Denver',
    'ARI': 'America/Phoenix',
    'LAD': 'America/Los_Angeles', 'LAA': 'America/Los_Angeles',
    'SD':  'America/Los_Angeles', 'SF':  'America/Los_Angeles',
    'SEA': 'America/Los_Angeles', 'OAK': 'America/Los_Angeles',
}
# East->West travel: 3-hour shift
# West->East travel: 3-hour shift (often worse due to early start)
```

---

## 8. CROSS-SPORT: SCHEDULE/FATIGUE DATA

### 8.1 Back-to-Back / Rest Day Analysis

All schedule data is available from the respective APIs (MLB Stats API, nba_api, NHL API, nfl_data_py). The key calculations:

```python
# For NBA (back-to-backs matter HUGELY -- ~2-3 point disadvantage)
# Calculate from nba_api team game log:
# - Is this a back-to-back? (played yesterday)
# - Is this the 3rd game in 4 nights?
# - Days since last game
# - Home/away on back-to-back (worse when away)

# For NHL (back-to-backs matter -- ~0.5-1 goal disadvantage)
# Similar to NBA but also consider:
# - Starting goalie likely different on B2B
# - Afternoon game after previous night game (worst case)

# For MLB (less impactful due to daily play, but still measurable)
# - Day game after night game (worst for batters)
# - Bullpen usage in previous 3 games
# - Travel day (off-day helps more after cross-country travel)

# For NFL (weekly schedule, so fatigue = short week)
# - Thursday Night Football (short week = ~1 pt disadvantage for road team)
# - Bye week (slight advantage in game after bye)
# - Monday night -> Sunday (short week)
# - Already in nfl_data_py schedule data as 'gameday' column
```

---

## 9. MULTI-SPORT PYTHON PACKAGES

### 9.1 sportsipy / sportsreference

- **Package**: `pip install sportsipy`
- **Covers**: NBA, NFL, NHL, MLB (all four!)
- **Source**: Scrapes Sports Reference sites
- **Auth**: None

```python
# NBA
from sportsipy.nba.teams import Teams as NBATeams
from sportsipy.nba.boxscore import Boxscores as NBABoxscores
nba_teams = NBATeams(2026)
for team in nba_teams:
    print(team.name, team.offensive_rating, team.defensive_rating)

# NFL
from sportsipy.nfl.teams import Teams as NFLTeams
nfl_teams = NFLTeams(2025)
for team in nfl_teams:
    print(team.name, team.points_for, team.points_against)

# NHL
from sportsipy.nhl.teams import Teams as NHLTeams
nhl_teams = NHLTeams(2026)
for team in nhl_teams:
    print(team.name, team.goals_for, team.goals_against)

# MLB
from sportsipy.mlb.teams import Teams as MLBTeams
mlb_teams = MLBTeams(2025)
for team in mlb_teams:
    print(team.name, team.runs, team.runs_allowed)
```

**Note**: sportsipy can be flaky if Sports Reference changes their HTML. Check GitHub issues before relying on it.

---

### 9.2 sportsdata.io (Free Trial)

- **URL**: `https://sportsdata.io`
- **Auth**: API key required (free trial available)
- **Covers**: All four major sports + more
- **Note**: Limited free tier, primarily a paid service

---

### 9.3 Sportradar (Paid, but powers many apps)

- **Mentioned for completeness**: Sportradar provides official data to many leagues
- **Auth**: Paid API
- **Not recommended for free use**

---

### 9.4 balldontlie (NBA only, FREE)

- **URL**: `https://www.balldontlie.io`
- **Auth**: Free API key
- **Covers**: NBA only

```python
import requests

headers = {'Authorization': 'your_free_api_key'}

# Games
r = requests.get('https://api.balldontlie.io/v1/games',
                  params={'dates[]': '2025-12-25'}, headers=headers)

# Player season averages
r = requests.get('https://api.balldontlie.io/v1/season_averages',
                  params={'season': 2025, 'player_ids[]': [237]}, headers=headers)

# Box scores
r = requests.get('https://api.balldontlie.io/v1/box_scores',
                  params={'date': '2025-12-25'}, headers=headers)
```

---

### 9.5 Additional Python Packages

| Package | Sport | Install | Notes |
|---------|-------|---------|-------|
| `pybaseball` | MLB | `pip install pybaseball` | Statcast, FanGraphs, B-Ref |
| `nba_api` | NBA | `pip install nba_api` | Official NBA.com stats |
| `nfl_data_py` | NFL | `pip install nfl_data_py` | nflverse ecosystem (best NFL source) |
| `hockey_scraper` | NHL | `pip install hockey-scraper` | NHL play-by-play + shifts |
| `sportsipy` | All 4 | `pip install sportsipy` | Sports Reference wrapper |
| `basketball_reference_web_scraper` | NBA | `pip install basketball-reference-web-scraper` | B-Ref scraper |
| `nhlpy` | NHL | `pip install nhlpy` | NHL API wrapper |
| `MLB-StatsAPI` | MLB | `pip install MLB-StatsAPI` | Official MLB API (you use this) |
| `sportradar` | All | `pip install sportradar` | Paid, not free |
| `sportsbet` | Odds | `pip install sportsbet` | Historical odds (limited) |
| `openskill` | Rating | `pip install openskill` | Alternative to Elo (Weng-Lin) |
| `elote` | Rating | `pip install elote` | Elo/Glicko implementations |

---

## 10. REFERENCE WEBSITES (SCRAPING TARGETS)

### MLB
| Site | URL | Best For | Scrapability |
|------|-----|----------|-------------|
| Baseball-Reference | baseball-reference.com | Historical stats, splits, game logs | Moderate (rate limited) |
| FanGraphs | fangraphs.com | WAR, advanced metrics, park factors | Good (CSV export URLs) |
| Baseball Savant | baseballsavant.mlb.com | Statcast, xStats, pitch tracking | Good (CSV search export) |
| Baseball Prospectus | baseballprospectus.com | DRC+, PECOTA projections | Limited (paywalled) |
| Retrosheet | retrosheet.org | Historical play-by-play (decades) | Excellent (bulk download) |

### NBA
| Site | URL | Best For | Scrapability |
|------|-----|----------|-------------|
| Basketball-Reference | basketball-reference.com | Historical stats, advanced, game logs | Moderate (rate limited) |
| NBA.com/stats | nba.com/stats | Official stats, tracking data | Use nba_api package |
| Cleaning the Glass | cleaningtheglass.com | Lineup data, on/off splits | Paid |
| PBP Stats | pbpstats.com | Play-by-play derived stats | Good |
| Dunks and Threes | dunksandthrees.com | Team ratings, four factors | Good (simple pages) |

### NFL
| Site | URL | Best For | Scrapability |
|------|-----|----------|-------------|
| Pro-Football-Reference | pro-football-reference.com | Historical stats, AV, game logs | Moderate (rate limited) |
| nflverse (GitHub) | github.com/nflverse | Play-by-play, EPA, CPOE | Excellent (use nfl_data_py) |
| Football Outsiders | footballoutsiders.com | DVOA (paid) | Limited |
| rbsdm.com | rbsdm.com | EPA rankings, pass rate | Good |
| The Football Database | footballdb.com | Schedules, results | Good |

### NHL
| Site | URL | Best For | Scrapability |
|------|-----|----------|-------------|
| Hockey-Reference | hockey-reference.com | Historical stats, advanced | Moderate (rate limited) |
| Natural Stat Trick | naturalstattrick.com | 5v5 stats, xG, shot data | Good (CSV parameters) |
| MoneyPuck | moneypuck.com | xG model, shot data, team stats | Excellent (CSV downloads!) |
| Evolving Hockey | evolving-hockey.com | GAR, WAR, player cards | Limited free |
| HockeyViz | hockeyviz.com | Visualizations, shot maps | Reference only |
| Elite Prospects | eliteprospects.com | Prospect data, international | Good |

---

## 11. DATA SOURCE PRIORITY MATRIX

### Highest Impact Per Sport (What to implement first)

#### MLB (Your Current System)
| Priority | Source | Data | Expected Accuracy Gain |
|----------|--------|------|----------------------|
| 1 | pybaseball / Statcast | xwOBA, FIP, xFIP for starters | +1-2% accuracy |
| 2 | pybaseball / FanGraphs | Park factors | +0.5-1% accuracy |
| 3 | Open-Meteo | Wind/temp for outdoor parks | +0.3-0.5% for affected games |
| 4 | MLB Stats API hydrate | Bullpen usage (pitch counts) | +0.3-0.5% accuracy |
| 5 | The Odds API | Closing lines for calibration benchmark | Better calibration |
| 6 | pybaseball / B-Ref | Platoon splits (vs L/R) | +0.2-0.3% accuracy |

#### NBA (New System)
| Priority | Source | Data | Expected Accuracy Gain |
|----------|--------|------|----------------------|
| 1 | nba_api | Net Rating (ORtg-DRtg) | Baseline ~67% accuracy |
| 2 | nba_api | Four Factors (eFG%, TOV%, ORB%, FTR) | Core features |
| 3 | nba_api | Rest days / back-to-back detection | +1-2% on B2B games |
| 4 | ESPN API | Injuries + embedded odds | +0.5-1% accuracy |
| 5 | Basketball-Reference | BPM/VORP for key player absence | +0.5-1% accuracy |
| 6 | nba_api | Clutch stats, home/away splits | +0.3-0.5% |

#### NFL (New System)
| Priority | Source | Data | Expected Accuracy Gain |
|----------|--------|------|----------------------|
| 1 | nfl_data_py | EPA/play (off + def) | Baseline ~65-68% accuracy |
| 2 | nfl_data_py | Schedule (includes lines, weather) | Core features + benchmarks |
| 3 | nfl_data_py | CPOE, success rate, PROE | +1-2% accuracy |
| 4 | nfl_data_py | Injuries + snap counts | +0.5-1% accuracy |
| 5 | nfl_data_py | Next Gen Stats (air yards, separation) | +0.3-0.5% accuracy |
| 6 | ESPN API | FPI for ensemble blending | +0.2-0.5% |

#### NHL (New System)
| Priority | Source | Data | Expected Accuracy Gain |
|----------|--------|------|----------------------|
| 1 | MoneyPuck CSVs | xGF%, team xG differential | Baseline ~58-60% accuracy |
| 2 | NHL API | Starting goalie identification | +2-3% accuracy (HUGE) |
| 3 | MoneyPuck/NST | GSAA for goalies | +1-2% accuracy |
| 4 | Natural Stat Trick | 5v5 HDCF%, PDO regression | +0.5-1% accuracy |
| 5 | NHL API | Schedule for B2B detection | +0.5-1% on B2B games |
| 6 | ESPN API | Injuries | +0.3-0.5% accuracy |

---

## QUICK START: Minimum Viable Package Installs

```bash
# All four sports in one command:
pip install nba_api MLB-StatsAPI nfl_data_py hockey-scraper pybaseball sportsipy nhlpy requests pandas
```

## QUICK START: Test Each Source

```python
# === Test MLB (you already have this) ===
import statsapi
today = statsapi.schedule()
print(f"MLB: {len(today)} games today")

# === Test NBA ===
from nba_api.stats.static import teams
nba_teams = teams.get_teams()
print(f"NBA: {len(nba_teams)} teams loaded")

# === Test NFL ===
import nfl_data_py as nfl
sched = nfl.import_schedules([2025])
print(f"NFL: {len(sched)} games loaded")

# === Test NHL ===
import requests
r = requests.get('https://api-web.nhle.com/v1/standings/now')
print(f"NHL: {len(r.json()['standings'])} teams in standings")

# === Test Weather (no auth!) ===
r = requests.get('https://api.open-meteo.com/v1/forecast',
    params={'latitude': 40.829, 'longitude': -73.926,
            'hourly': 'temperature_2m', 'temperature_unit': 'fahrenheit'})
print(f"Weather: {len(r.json()['hourly']['time'])} hours of forecast")

# === Test ESPN (all sports, no auth!) ===
for sport in ['baseball/mlb', 'basketball/nba', 'football/nfl', 'hockey/nhl']:
    r = requests.get(f'https://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard')
    events = r.json().get('events', [])
    print(f"ESPN {sport}: {len(events)} games on scoreboard")
```
