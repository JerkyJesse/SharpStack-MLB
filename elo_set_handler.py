"""Shared handler for 'set param=value' commands across all sports.

Maps every Elo parameter to aliases and applies the value to the model.
Also handles boolean and string settings (use_mov, autoresolve, kelly).
"""

# Complete parameter registry: canonical_name -> (aliases, type, description)
ELO_PARAMS = {
    # Core
    "k":                          (["k_factor"], "float", "Elo K-factor (learning rate per game)"),
    "base_rating":                (["base", "rating"], "float", "Starting Elo rating"),
    "home_adv":                   (["home", "hca", "home_advantage"], "float", "Home court/field advantage (Elo points)"),
    "use_mov":                    (["mov", "margin"], "bool", "Use margin of victory adjustment"),

    # Player / pitcher
    "player_boost":               (["boost", "player"], "float", "Team-level player strength boost"),
    "starter_boost":              (["starter", "pitcher_boost", "sp_boost"], "float", "Starting pitcher quality adjustment"),
    "bullpen_factor":             (["bullpen", "bp_factor", "reliever"], "float", "Bullpen/reliever quality factor"),
    "opp_pitcher_factor":         (["opp_pitcher", "opp_sp"], "float", "Opponent pitcher adjustment factor"),

    # Margin of victory
    "mov_base":                   (["mov_mult", "mov_constant"], "float", "MOV multiplier constant (log curve shift)"),
    "mov_cap":                    (["movcap", "margin_cap"], "float", "Maximum MOV adjustment cap"),

    # Rest / schedule
    "rest_factor":                (["rest"], "float", "Rest days advantage factor"),
    "rest_advantage_cap":         (["restcap", "rest_cap"], "float", "Maximum rest advantage multiplier"),
    "b2b_penalty":                (["b2b", "back_to_back"], "float", "Back-to-back game penalty"),
    "road_trip_factor":           (["roadtrip", "road_trip"], "float", "Extended road trip penalty"),
    "homestand_factor":           (["homestand"], "float", "Extended homestand bonus"),

    # Travel / venue
    "travel_factor":              (["travel"], "float", "Elo penalty per timezone crossed"),
    "east_travel_penalty":        (["east_travel", "eastbound"], "float", "Extra penalty for eastbound travel"),
    "altitude_factor":            (["altitude", "alt"], "float", "Altitude bonus multiplier (Denver/Colorado)"),
    "park_factor_weight":         (["parkfactor", "park_factor", "park"], "float", "Park factor weight"),

    # Form / momentum
    "form_weight":                (["form"], "float", "Recent form weight"),
    "win_streak_factor":          (["streak", "win_streak"], "float", "Win/loss streak momentum factor"),
    "mean_reversion":             (["reversion", "regress"], "float", "Mean reversion after extreme results"),
    "season_regress":             (["season_regression", "regress_pct"], "float", "Season boundary regression fraction"),

    # Matchup adjustments
    "sos_factor":                 (["sos", "strength_of_schedule"], "float", "Strength of schedule weight"),
    "division_factor":            (["division", "div"], "float", "Divisional game confidence reducer"),
    "interleague_factor":         (["interleague", "il_factor"], "float", "Interleague game adjustment"),
    "series_adaptation":          (["series", "adaptation"], "float", "Series adaptation factor (rematches)"),

    # Scoring model
    "pace_factor":                (["pace", "tempo"], "float", "Run/scoring environment mismatch"),
    "pyth_factor":                (["pyth", "pythagorean"], "float", "Pythagorean expected W% adjustment"),
    "scoring_consistency_factor": (["consistency", "scoring_consistency"], "float", "Penalty for volatile scoring"),
    "home_road_factor":           (["home_road", "split"], "float", "Team-specific home/road split bonus"),

    # Season / phase
    "playoff_hca_factor":         (["playoff", "playoff_hca", "postseason"], "float", "Playoff home advantage multiplier"),
    "season_phase_factor":        (["phase", "season_phase"], "float", "Early-season dampener"),

    # K-factor variants
    "k_decay":                    (["kdecay", "k_reduction"], "float", "K-factor decay over season"),
    "surprise_k":                 (["surprise", "upset_k"], "float", "Extra K for surprise results"),

    # NFL-specific
    "bye_week_factor":            (["bye", "bye_week"], "float", "Bye week advantage factor (NFL)"),

    # Account / trading
    "kelly_fraction":             (["kelly"], "special", "Kelly criterion fraction (quarter/half/0.25/0.50)"),
    "starting_balance":           (["balance", "bankroll"], "float", "Starting account balance"),
    "autoresolve_enabled":        (["autoresolve", "auto_resolve"], "bool", "Auto-resolve finished trades"),
    "auto_kalshi":                (["autokalshi", "kalshi"], "bool", "Auto-fetch Kalshi odds for Kelly sizing"),
}

# Build reverse alias map
_ALIAS_MAP = {}
for canonical, (aliases, ptype, desc) in ELO_PARAMS.items():
    _ALIAS_MAP[canonical] = canonical
    for alias in aliases:
        _ALIAS_MAP[alias] = canonical


def handle_elo_set(cmd_args, model, load_settings_fn, save_settings_fn):
    """Handle 'set param=value' for any Elo parameter.

    Args:
        cmd_args: string after 'set ', e.g. 'k=10' or 'home_adv=48'
        model: the sport's Elo model instance
        load_settings_fn: function to load settings dict
        save_settings_fn: function to save settings dict

    Returns (success: bool, message: str)
    """
    parts = cmd_args.split("=", 1)
    if len(parts) != 2:
        return False, "Usage: set <param>=<value>"

    raw_name = parts[0].strip().lower()
    raw_value = parts[1].strip()

    # Resolve alias
    canonical = _ALIAS_MAP.get(raw_name)
    if canonical is None:
        # Show available params grouped by category
        return False, _unknown_param_msg(raw_name)

    aliases, ptype, desc = ELO_PARAMS[canonical]

    # ── Special handling for kelly ──
    if canonical == "kelly_fraction":
        kelly_map = {
            "quarter": 0.25, "half": 0.50, "full": 1.0,
            "0.25": 0.25, "0.50": 0.50, "0.5": 0.50, "1.0": 1.0,
            "25": 0.25, "50": 0.50, "100": 1.0,
        }
        val_str = raw_value.lower()
        if val_str in kelly_map:
            settings = load_settings_fn()
            settings["kelly_fraction"] = kelly_map[val_str]
            save_settings_fn(settings)
            return True, "Kelly fraction set to %.0f%% (%s Kelly)" % (kelly_map[val_str] * 100, val_str)
        else:
            try:
                val = float(raw_value)
                if 0 < val <= 1:
                    settings = load_settings_fn()
                    settings["kelly_fraction"] = val
                    save_settings_fn(settings)
                    return True, "Kelly fraction set to %.0f%%" % (val * 100)
            except ValueError:
                pass
            return False, "Usage: set kelly=quarter / half / 0.25 / 0.50"

    # ── Boolean params ──
    if ptype == "bool":
        val_str = raw_value.lower()
        if val_str in ("true", "1", "yes", "on"):
            value = True
        elif val_str in ("false", "0", "no", "off"):
            value = False
        else:
            return False, "Invalid boolean: %s (use true/false/on/off)" % raw_value

        # Set on model if it has the attribute
        if hasattr(model, canonical):
            setattr(model, canonical, value)

        # Also save to settings
        settings = load_settings_fn()
        settings[canonical] = value
        save_settings_fn(settings)
        return True, "Set %s = %s  (%s)" % (canonical, value, desc)

    # ── Numeric params ──
    try:
        val = float(raw_value)
    except ValueError:
        return False, "Invalid number: %s" % raw_value

    if ptype == "int":
        val = int(val)

    # Set on model
    if hasattr(model, canonical):
        setattr(model, canonical, val)
    else:
        # Some params are in settings but not direct model attributes
        pass

    # Save to settings
    save_settings_fn(model.settings_dict())

    return True, "Set %s = %s  (%s)" % (canonical, val, desc)


def _unknown_param_msg(name):
    """Build helpful error message with available params."""
    lines = ["Unknown parameter: %s" % name, "", "Available parameters:"]

    # Group by category
    categories = {
        "Core":     ["k", "base_rating", "home_adv", "use_mov"],
        "Player":   ["player_boost", "starter_boost", "bullpen_factor", "opp_pitcher_factor"],
        "MOV":      ["mov_base", "mov_cap"],
        "Rest":     ["rest_factor", "rest_advantage_cap", "b2b_penalty"],
        "Travel":   ["travel_factor", "east_travel_penalty", "road_trip_factor",
                      "homestand_factor", "altitude_factor", "park_factor_weight"],
        "Form":     ["form_weight", "win_streak_factor", "mean_reversion", "season_regress"],
        "Matchup":  ["sos_factor", "division_factor", "interleague_factor", "series_adaptation"],
        "Scoring":  ["pace_factor", "pyth_factor", "scoring_consistency_factor", "home_road_factor"],
        "Season":   ["playoff_hca_factor", "season_phase_factor"],
        "K-factor": ["k_decay", "surprise_k"],
        "Account":  ["kelly_fraction", "starting_balance", "autoresolve_enabled", "auto_kalshi"],
    }

    for cat, params in categories.items():
        param_list = []
        for p in params:
            aliases_str = ", ".join(ELO_PARAMS[p][0][:1])
            param_list.append("%s (%s)" % (p, aliases_str))
        lines.append("  %s: %s" % (cat, ", ".join(param_list)))

    return "\n".join(lines)


def print_elo_settings(model, load_settings_fn):
    """Print all Elo settings with current values."""
    settings = load_settings_fn()

    print("\n  %-30s %12s  %s" % ("Parameter", "Value", "Description"))
    print("  " + "-" * 75)

    for canonical, (aliases, ptype, desc) in ELO_PARAMS.items():
        # Get value from model if available, else from settings
        if hasattr(model, canonical):
            val = getattr(model, canonical)
        else:
            val = settings.get(canonical, "(default)")

        # Format value
        if isinstance(val, float):
            if val == int(val) and abs(val) < 10000:
                val_str = "%g" % val
            else:
                val_str = "%.4f" % val if abs(val) < 1 else "%.2f" % val
        elif isinstance(val, bool):
            val_str = "true" if val else "false"
        else:
            val_str = str(val)

        alias_str = aliases[0] if aliases else ""
        print("  %-30s %12s  %s" % (canonical, val_str, desc))
