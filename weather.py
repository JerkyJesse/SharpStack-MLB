"""Weather integration for outdoor sports (NFL, MLB) via Open-Meteo API.

No API key required. Free for non-commercial use.
Fetches temperature, wind speed/direction, precipitation probability, humidity.
"""

import json
import logging
import os
from datetime import datetime, timedelta

import requests

WEATHER_CACHE_FILE = "weather_cache.json"
CACHE_MAX_AGE_HOURS = 2

# NFL stadiums: name -> (lat, lon, is_dome)
NFL_VENUES = {
    "Arizona Cardinals":       (33.5276, -112.2626, True),   # State Farm Stadium (retractable roof)
    "Atlanta Falcons":         (33.7554, -84.4010, True),    # Mercedes-Benz Stadium
    "Baltimore Ravens":        (39.2780, -76.6227, False),   # M&T Bank Stadium
    "Buffalo Bills":           (42.7738, -78.7870, False),   # Highmark Stadium
    "Carolina Panthers":       (35.2258, -80.8528, False),   # Bank of America Stadium
    "Chicago Bears":           (41.8623, -87.6167, False),   # Soldier Field
    "Cincinnati Bengals":      (39.0955, -84.5160, False),   # Paycor Stadium
    "Cleveland Browns":        (41.5061, -81.6996, False),   # Cleveland Browns Stadium
    "Dallas Cowboys":          (32.7473, -97.0945, True),    # AT&T Stadium
    "Denver Broncos":          (39.7439, -105.0201, False),  # Empower Field
    "Detroit Lions":           (42.3400, -83.0456, True),    # Ford Field
    "Green Bay Packers":       (44.5013, -88.0622, False),   # Lambeau Field
    "Houston Texans":          (29.6847, -95.4107, True),    # NRG Stadium (retractable)
    "Indianapolis Colts":      (39.7601, -86.1638, True),    # Lucas Oil Stadium
    "Jacksonville Jaguars":    (30.3239, -81.6373, False),   # EverBank Stadium
    "Kansas City Chiefs":      (39.0489, -94.4839, False),   # Arrowhead Stadium
    "Las Vegas Raiders":       (36.0908, -115.1833, True),   # Allegiant Stadium
    "Los Angeles Chargers":    (33.9534, -118.3387, True),   # SoFi Stadium
    "Los Angeles Rams":        (33.9534, -118.3387, True),   # SoFi Stadium
    "Miami Dolphins":          (25.9580, -80.2389, False),   # Hard Rock Stadium
    "Minnesota Vikings":       (44.9736, -93.2575, True),    # U.S. Bank Stadium
    "New England Patriots":    (42.0909, -71.2643, False),   # Gillette Stadium
    "New Orleans Saints":      (29.9511, -90.0812, True),    # Caesars Superdome
    "New York Giants":         (40.8128, -74.0742, False),   # MetLife Stadium
    "New York Jets":           (40.8128, -74.0742, False),   # MetLife Stadium
    "Philadelphia Eagles":     (39.9008, -75.1675, False),   # Lincoln Financial Field
    "Pittsburgh Steelers":     (40.4468, -80.0158, False),   # Acrisure Stadium
    "San Francisco 49ers":     (37.4033, -121.9694, False),  # Levi's Stadium
    "Seattle Seahawks":        (47.5952, -122.3316, False),  # Lumen Field
    "Tampa Bay Buccaneers":    (27.9759, -82.5033, False),   # Raymond James Stadium
    "Tennessee Titans":        (36.1665, -86.7713, False),   # Nissan Stadium
    "Washington Commanders":   (38.9076, -76.8645, False),   # Northwest Stadium
}

# MLB stadiums: name -> (lat, lon, is_dome)
MLB_VENUES = {
    "Arizona Diamondbacks":    (33.4453, -112.0667, True),   # Chase Field (retractable)
    "Atlanta Braves":          (33.8907, -84.4677, False),   # Truist Park
    "Baltimore Orioles":       (39.2838, -76.6216, False),   # Camden Yards
    "Boston Red Sox":          (42.3467, -71.0972, False),   # Fenway Park
    "Chicago Cubs":            (41.9484, -87.6553, False),   # Wrigley Field
    "Chicago White Sox":       (41.8300, -87.6339, False),   # Guaranteed Rate Field
    "Cincinnati Reds":         (39.0974, -84.5082, False),   # Great American Ball Park
    "Cleveland Guardians":     (41.4962, -81.6852, False),   # Progressive Field
    "Colorado Rockies":        (39.7559, -104.9942, False),  # Coors Field (5,280 ft)
    "Detroit Tigers":          (42.3390, -83.0485, False),   # Comerica Park
    "Houston Astros":          (29.7572, -95.3554, True),    # Minute Maid Park (retractable)
    "Kansas City Royals":      (39.0517, -94.4803, False),   # Kauffman Stadium
    "Los Angeles Angels":      (33.8003, -117.8827, False),  # Angel Stadium
    "Los Angeles Dodgers":     (34.0739, -118.2400, False),  # Dodger Stadium
    "Miami Marlins":           (25.7781, -80.2196, True),    # loanDepot Park (retractable)
    "Milwaukee Brewers":       (43.0280, -87.9712, True),    # American Family Field (retractable)
    "Minnesota Twins":         (44.9817, -93.2776, False),   # Target Field
    "New York Mets":           (40.7571, -73.8458, False),   # Citi Field
    "New York Yankees":        (40.8296, -73.9262, False),   # Yankee Stadium
    "Oakland Athletics":       (37.7516, -122.2005, False),  # Oakland Coliseum
    "Philadelphia Phillies":   (39.9061, -75.1665, False),   # Citizens Bank Park
    "Pittsburgh Pirates":      (40.4468, -80.0057, False),   # PNC Park
    "San Diego Padres":        (32.7076, -117.1570, False),  # Petco Park
    "San Francisco Giants":    (37.7786, -122.3893, False),  # Oracle Park
    "Seattle Mariners":        (47.5914, -122.3323, True),   # T-Mobile Park (retractable)
    "St. Louis Cardinals":     (38.6226, -90.1928, False),   # Busch Stadium
    "Tampa Bay Rays":          (27.7682, -82.6534, True),    # Tropicana Field
    "Texas Rangers":           (32.7512, -97.0832, True),    # Globe Life Field (retractable)
    "Toronto Blue Jays":       (43.6414, -79.3894, True),    # Rogers Centre (retractable)
    "Washington Nationals":    (38.8730, -77.0074, False),   # Nationals Park
}


def _get_venue_info(team, sport):
    """Look up venue coordinates and dome status for a team."""
    sport_l = sport.lower()
    if sport_l == "nfl":
        return NFL_VENUES.get(team)
    elif sport_l == "mlb":
        return MLB_VENUES.get(team)
    return None


def _load_cache():
    cache_file = os.path.join(os.path.dirname(__file__), WEATHER_CACHE_FILE)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_cache(cache):
    cache_file = os.path.join(os.path.dirname(__file__), WEATHER_CACHE_FILE)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def _is_cache_fresh(cache_entry):
    fetched = cache_entry.get("fetched_at", "")
    if not fetched:
        return False
    try:
        dt = datetime.fromisoformat(fetched)
        return (datetime.now() - dt).total_seconds() < CACHE_MAX_AGE_HOURS * 3600
    except (ValueError, TypeError):
        return False


def fetch_weather(lat, lon, game_datetime=None):
    """Fetch weather from Open-Meteo API.

    Uses the forecast endpoint for recent/future dates and the
    archive endpoint for historical dates (>5 days ago).

    Args:
        lat, lon: venue coordinates
        game_datetime: datetime of game start (default: now)

    Returns dict with temperature_f, wind_speed_mph, wind_direction,
        precipitation_probability, humidity, weather_code, description.
    """
    if game_datetime is None:
        game_datetime = datetime.now()

    date_str = game_datetime.strftime("%Y-%m-%d")
    hour = game_datetime.hour

    # Open-Meteo forecast API only covers ~3 months back;
    # use the archive API for anything older than 5 days.
    days_ago = (datetime.now() - game_datetime).days
    is_historical = days_ago > 5

    if is_historical:
        url = "https://archive-api.open-meteo.com/v1/archive"
        # Archive API doesn't have precipitation_probability; use precipitation instead
        hourly_vars = ("temperature_2m,relative_humidity_2m,precipitation,"
                       "weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m")
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        hourly_vars = ("temperature_2m,relative_humidity_2m,precipitation_probability,"
                       "weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        weather_code = hourly.get("weather_code", [])
        wind_speed = hourly.get("wind_speed_10m", [])
        wind_dir = hourly.get("wind_direction_10m", [])
        wind_gust = hourly.get("wind_gusts_10m", [])

        # Archive has precipitation (mm), forecast has precipitation_probability (%)
        if is_historical:
            precip_mm = hourly.get("precipitation", [])
        else:
            precip_mm = []
        precip_prob = hourly.get("precipitation_probability", [])

        # Get the hour closest to game time
        idx = min(hour, len(temps) - 1) if temps else 0

        # For historical data, convert precipitation mm to an approximate probability
        if is_historical and idx < len(precip_mm):
            mm = precip_mm[idx] or 0
            # 0mm=0%, 0.5mm=40%, 1mm=60%, 2mm=80%, 5mm+=100%
            precip_pct = min(100, mm * 50) if mm > 0 else 0
        elif idx < len(precip_prob):
            precip_pct = precip_prob[idx]
        else:
            precip_pct = None

        result = {
            "temperature_f": temps[idx] if idx < len(temps) else None,
            "humidity": humidity[idx] if idx < len(humidity) else None,
            "precipitation_probability": precip_pct,
            "weather_code": weather_code[idx] if idx < len(weather_code) else None,
            "wind_speed_mph": wind_speed[idx] if idx < len(wind_speed) else None,
            "wind_direction": wind_dir[idx] if idx < len(wind_dir) else None,
            "wind_gust_mph": wind_gust[idx] if idx < len(wind_gust) else None,
            "description": _weather_code_to_desc(weather_code[idx] if idx < len(weather_code) else 0),
            "fetched_at": datetime.now().isoformat(),
        }
        return result

    except requests.RequestException as e:
        logging.error("Weather API error: %s", e)
        return None


def _weather_code_to_desc(code):
    """Convert WMO weather code to human-readable description."""
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return codes.get(code, f"Code {code}")


def get_game_weather(home_team, sport, game_datetime=None):
    """Get weather for a game at the home team's venue.

    Returns None for dome stadiums (weather irrelevant).
    Returns weather dict for outdoor stadiums.
    """
    venue = _get_venue_info(home_team, sport)
    if venue is None:
        logging.warning("No venue data for %s (%s)", home_team, sport)
        return None

    lat, lon, is_dome = venue
    if is_dome:
        return {"is_dome": True, "description": "Dome stadium - weather N/A"}

    # Check cache
    cache = _load_cache()
    cache_key = f"{home_team}_{sport}_{game_datetime.strftime('%Y%m%d_%H') if game_datetime else 'now'}"
    if cache_key in cache and _is_cache_fresh(cache[cache_key]):
        return cache[cache_key]

    weather = fetch_weather(lat, lon, game_datetime)
    if weather:
        weather["is_dome"] = False
        weather["venue_team"] = home_team
        cache[cache_key] = weather
        _save_cache(cache)

    return weather


def compute_weather_impact(weather, sport):
    """Compute weather impact adjustments for predictions.

    Returns dict with adjustment factors:
        passing_impact: NFL passing efficiency modifier (-1.0 to 0)
        scoring_impact: expected scoring modifier (-1.0 to 0)
        home_advantage_mod: home field advantage modifier
        unpredictability: how much weather increases variance (0 to 1)
    """
    if weather is None or weather.get("is_dome"):
        return {"passing_impact": 0, "scoring_impact": 0,
                "home_advantage_mod": 0, "unpredictability": 0}

    wind = weather.get("wind_speed_mph", 0) or 0
    temp = weather.get("temperature_f", 70) or 70
    precip = weather.get("precipitation_probability", 0) or 0
    gust = weather.get("wind_gust_mph", 0) or 0

    sport_l = sport.lower()
    impact = {"passing_impact": 0, "scoring_impact": 0,
              "home_advantage_mod": 0, "unpredictability": 0}

    if sport_l == "nfl":
        # Wind impact on passing (significant above 15 mph)
        if wind > 15:
            impact["passing_impact"] = -0.02 * (wind - 15)  # ~2% per mph over 15
        if gust > 25:
            impact["passing_impact"] -= 0.01 * (gust - 25)

        # Cold weather impact (below 32F)
        if temp < 32:
            impact["scoring_impact"] = -0.01 * (32 - temp)  # ~1% per degree below freezing

        # Rain/snow impact
        if precip > 50:
            impact["scoring_impact"] -= 0.05  # Rain reduces scoring ~5%
            impact["passing_impact"] -= 0.03
        if precip > 80:
            impact["scoring_impact"] -= 0.05  # Heavy precip doubles the penalty

        # Extreme weather increases home advantage (familiarity)
        extreme = (wind > 20) or (temp < 20) or (precip > 60)
        if extreme:
            impact["home_advantage_mod"] = 0.03  # +3% home edge in bad weather

        # Unpredictability
        impact["unpredictability"] = min(1.0, (wind / 40) + (precip / 200) + max(0, (32 - temp) / 60))

    elif sport_l == "mlb":
        # Wind impact on home runs
        if wind > 15:
            impact["scoring_impact"] = -0.015 * (wind - 15)

        # Hot weather = more HRs, cold = fewer
        if temp > 85:
            impact["scoring_impact"] += 0.02 * (temp - 85) / 10  # Slight increase
        elif temp < 55:
            impact["scoring_impact"] -= 0.01 * (55 - temp) / 10

        # Rain impact
        if precip > 40:
            impact["scoring_impact"] -= 0.03
            impact["unpredictability"] += 0.15

        if precip > 70:
            impact["scoring_impact"] -= 0.05
            impact["unpredictability"] += 0.15

        impact["unpredictability"] = min(1.0, impact["unpredictability"] + wind / 50)

    # Cap impacts
    impact["passing_impact"] = max(-0.30, impact["passing_impact"])
    impact["scoring_impact"] = max(-0.25, min(0.10, impact["scoring_impact"]))

    return impact


def show_weather_report(home_team, sport, game_datetime=None):
    """Print weather report for a game venue."""
    weather = get_game_weather(home_team, sport, game_datetime)
    if weather is None:
        print(f"  No weather data available for {home_team}")
        return None

    if weather.get("is_dome"):
        print(f"  {home_team}: Dome stadium - weather not a factor")
        return weather

    temp = weather.get("temperature_f", "?")
    wind = weather.get("wind_speed_mph", "?")
    gust = weather.get("wind_gust_mph", "?")
    precip = weather.get("precipitation_probability", "?")
    humidity = weather.get("humidity", "?")
    desc = weather.get("description", "Unknown")

    print(f"\n  Weather at {home_team} venue:")
    print(f"    Conditions : {desc}")
    print(f"    Temperature: {temp}°F")
    print(f"    Wind       : {wind} mph (gusts {gust} mph)")
    print(f"    Precip prob: {precip}%")
    print(f"    Humidity   : {humidity}%")

    impact = compute_weather_impact(weather, sport)
    if abs(impact["scoring_impact"]) > 0.01 or abs(impact["passing_impact"]) > 0.01:
        print(f"    --- Impact ---")
        if impact["passing_impact"] != 0:
            print(f"    Passing    : {impact['passing_impact']:+.1%}")
        if impact["scoring_impact"] != 0:
            print(f"    Scoring    : {impact['scoring_impact']:+.1%}")
        if impact["home_advantage_mod"] != 0:
            print(f"    Home edge  : {impact['home_advantage_mod']:+.1%}")
        if impact["unpredictability"] > 0.2:
            print(f"    Variance   : {'HIGH' if impact['unpredictability'] > 0.5 else 'MODERATE'}")

    return weather
