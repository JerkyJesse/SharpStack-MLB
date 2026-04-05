"""Mega-Ensemble Configuration: per-model on/off switches + settings.

Controls which models are active in the mega-ensemble.
Supports individual model enable/disable and single-model optimization.

Settings file: {sport}_mega_settings.json in each sport directory.
"""

import os
import json
import logging

# ── Master model registry ──────────────────────────────────────────
# Every model in the mega-ensemble, with default on/off and description.

MODEL_REGISTRY = {
    # Tier 0: Core (always on by default)
    "elo":              {"default": True,  "tier": 0, "desc": "Elo ratings (24+ adjusters)"},
    "xgboost":          {"default": True,  "tier": 0, "desc": "XGBoost ensemble (31 rolling features)"},

    # Tier 1: Proven models
    "hmm":              {"default": True,  "tier": 1, "desc": "Hidden Markov Model (hot/cold states)"},
    "kalman":           {"default": True,  "tier": 1, "desc": "Kalman Filter (strength estimation)"},
    "pagerank":         {"default": True,  "tier": 1, "desc": "PageRank + HITS (network analysis)"},
    "lightgbm":         {"default": True,  "tier": 1, "desc": "LightGBM (leaf-wise boosting)"},
    "catboost":         {"default": True,  "tier": 1, "desc": "CatBoost (ordered boosting)"},
    "mlp":              {"default": True,  "tier": 1, "desc": "MLP neural network (deep features)"},
    "lstm":             {"default": True,  "tier": 1, "desc": "LSTM (sequential patterns)"},

    # Tier 2: Exotic / physics-inspired
    "garch":            {"default": True,  "tier": 2, "desc": "GARCH volatility (time-varying)"},
    "fourier":          {"default": True,  "tier": 2, "desc": "Fourier / wavelet (cycle detection)"},
    "survival":         {"default": True,  "tier": 2, "desc": "Survival analysis (streak hazards)"},
    "copula":           {"default": True,  "tier": 2, "desc": "Copula (off/def joint dependency)"},

    # Tier 3: Information & physics
    "info_theory":      {"default": True,  "tier": 3, "desc": "Shannon entropy + KL divergence"},
    "momentum":         {"default": True,  "tier": 3, "desc": "Newtonian momentum / inertia"},
    "markov":           {"default": True,  "tier": 3, "desc": "Markov chain (transition matrices)"},
    "clustering":       {"default": True,  "tier": 3, "desc": "k-Means team archetypes"},
    "game_theory":      {"default": True,  "tier": 3, "desc": "Nash equilibrium + style matchups"},

    # Tier 4: Classical rating systems
    "poisson":          {"default": True,  "tier": 4, "desc": "Poisson / Dixon-Coles (score dist)"},
    "glicko":           {"default": True,  "tier": 4, "desc": "Glicko-2 (uncertainty-aware ratings)"},
    "bradley_terry":    {"default": True,  "tier": 4, "desc": "Bradley-Terry MLE (paired comparison)"},
    "monte_carlo":      {"default": True,  "tier": 4, "desc": "Monte Carlo simulation (3000 sims)"},
    "random_forest":    {"default": True,  "tier": 4, "desc": "Random Forest (bagging diversity)"},

    # Tier 5: Classical baseball/sports models
    "srs":              {"default": True,  "tier": 5, "desc": "Simple Rating System (margin + SOS)"},
    "colley":           {"default": True,  "tier": 5, "desc": "Colley Matrix (bias-free ranking)"},
    "log5":             {"default": True,  "tier": 5, "desc": "Log5 Bill James (h2h formula)"},
    "pythagenpat":      {"default": True,  "tier": 5, "desc": "PythagenPat (dynamic exponent)"},
    "exp_smoothing":    {"default": True,  "tier": 5, "desc": "Exponential smoothing (trend tracking)"},
    "mean_reversion":   {"default": True,  "tier": 5, "desc": "Mean reversion (Bollinger bands)"},

    # Tier 1 addition: ML classifier
    "svm":              {"default": True,  "tier": 1, "desc": "SVM classifier (RBF kernel + Platt scaling)"},

    # Tier 2 additions: Exotic / physics-inspired
    "fibonacci":        {"default": True,  "tier": 2, "desc": "Fibonacci retracement (support/resistance)"},
    "evt":              {"default": True,  "tier": 2, "desc": "Extreme Value Theory (tail risk)"},

    # Tier 3 addition: Information & physics
    "benford":          {"default": True,  "tier": 3, "desc": "Benford's Law (scoring pattern anomaly)"},

    # Data enrichment models
    "weather":          {"default": True,  "tier": 6, "desc": "Weather impact (temperature, wind)"},
    "odds":             {"default": False, "tier": 6, "desc": "Market odds / CLV tracking"},
}

# All model names in order
ALL_MODELS = list(MODEL_REGISTRY.keys())


def get_default_switches():
    """Get default on/off switches for all models."""
    return {name: spec["default"] for name, spec in MODEL_REGISTRY.items()}


def load_model_switches(sport, sport_dir):
    """Load model on/off switches from settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    switches = get_default_switches()

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            saved = data.get("model_switches", {})
            switches.update(saved)
        except Exception as e:
            logging.debug("Failed to load model switches: %s", e)

    return switches


def save_model_switches(sport, sport_dir, switches):
    """Save model on/off switches to settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")

    # Load existing settings
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.debug("Failed to load existing settings for switch save: %s", e)

    data["model_switches"] = switches

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def is_model_enabled(model_name, switches):
    """Check if a model is enabled."""
    return switches.get(model_name, MODEL_REGISTRY.get(model_name, {}).get("default", False))


def print_model_status(switches):
    """Print all models with their on/off status."""
    print("\n  %-20s %-5s  %-4s  %s" % ("Model", "Status", "Tier", "Description"))
    print("  " + "-" * 70)

    current_tier = -1
    for name, spec in MODEL_REGISTRY.items():
        if spec["tier"] != current_tier:
            current_tier = spec["tier"]
            tier_names = {0: "Core", 1: "Proven", 2: "Exotic",
                          3: "Info/Physics", 4: "Classical", 5: "Sports-specific",
                          6: "Data Enrichment"}
            print("  -- %s --" % tier_names.get(current_tier, f"Tier {current_tier}"))

        enabled = is_model_enabled(name, switches)
        status = " ON " if enabled else " OFF"
        marker = "[*]" if enabled else "[ ]"
        print("  %s %-17s %s   T%d    %s" % (marker, name, status, spec["tier"], spec["desc"]))

    n_on = sum(1 for n in ALL_MODELS if is_model_enabled(n, switches))
    print("\n  %d/%d models enabled" % (n_on, len(ALL_MODELS)))


# ── Mega parameter set/get ─────────────────────────────────────────

# All settable mega-ensemble parameters with aliases and descriptions
MEGA_PARAMS = {
    # Critical
    "max_adj":                 {"aliases": ["maxadj", "adj", "adjustment"], "type": "float",
                                "desc": "Max meta-learner adjustment (+/- probability)"},
    "meta_model":              {"aliases": ["meta", "metalearner", "stacker"], "type": "str",
                                "desc": "Meta-learner type (ridge, logistic, xgboost, bma, confidence_voting)"},
    "retrain_every":           {"aliases": ["retrain", "retrain_interval"], "type": "int",
                                "desc": "Retrain meta-learner every N games"},
    "min_train":               {"aliases": ["mintrain", "min_games", "warmup"], "type": "int",
                                "desc": "Games before meta-learner starts predicting"},
    # Kalman
    "kalman_process_noise":    {"aliases": ["kalman_pn", "process_noise", "pn"], "type": "float",
                                "desc": "Kalman filter process noise"},
    "kalman_measurement_noise":{"aliases": ["kalman_mn", "measurement_noise", "mn"], "type": "float",
                                "desc": "Kalman filter measurement noise"},
    # HMM
    "hmm_states":              {"aliases": ["hmm_n", "n_states", "states"], "type": "int",
                                "desc": "Number of HMM hidden states"},
    # Network
    "network_decay":           {"aliases": ["net_decay", "pagerank_decay", "decay"], "type": "float",
                                "desc": "PageRank temporal decay (0-1)"},
    # Momentum
    "momentum_friction":       {"aliases": ["friction", "mom_friction"], "type": "float",
                                "desc": "Momentum friction coefficient"},
    # Clustering
    "n_clusters":              {"aliases": ["clusters", "k_clusters", "nclusters"], "type": "int",
                                "desc": "Number of team archetype clusters"},
    # Glicko
    "glicko_initial_rd":       {"aliases": ["glicko_rd", "initial_rd", "rd"], "type": "float",
                                "desc": "Glicko-2 initial rating deviation"},
    # Bradley-Terry
    "bt_decay":                {"aliases": ["bt_recency", "bradley_decay"], "type": "float",
                                "desc": "Bradley-Terry recency decay (0-1)"},
    # Monte Carlo
    "mc_simulations":          {"aliases": ["mc_sims", "simulations", "n_sims", "sims"], "type": "int",
                                "desc": "Monte Carlo simulations per game"},
    # Window
    "window":                  {"aliases": ["rolling_window", "feat_window"], "type": "int",
                                "desc": "Rolling feature window size (games)"},
}

# ── Per-model hyperparameter registry ─────────────────────────────
# Every tunable hyperparameter per model, with default, type, and search values.
# Used by the mega optimizer for per-model tuning (Phase 1).

MODEL_HYPERPARAMS = {
    "meta_xgb": {
        "meta_xgb_max_depth":      {"type": "int",   "default": 4,    "values": [3, 4, 5, 6]},
        "meta_xgb_eta":            {"type": "float", "default": 0.05, "values": [0.01, 0.03, 0.05, 0.08, 0.12]},
        "meta_xgb_subsample":      {"type": "float", "default": 0.8,  "values": [0.6, 0.7, 0.8, 0.9]},
        "meta_xgb_min_child_weight":{"type": "int",  "default": 5,    "values": [3, 5, 8, 12]},
        "meta_xgb_alpha":          {"type": "float", "default": 0.5,  "values": [0.1, 0.3, 0.5, 1.0, 2.0]},
        "meta_xgb_lambda":         {"type": "float", "default": 1.0,  "values": [0.5, 1.0, 2.0, 5.0]},
        "meta_xgb_num_boost_round":{"type": "int",   "default": 200,  "values": [100, 150, 200, 300, 500]},
    },
    "meta_ridge": {
        "meta_ridge_alpha_scale":  {"type": "float", "default": 0.5,  "values": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]},
    },
    "meta_logistic": {
        "meta_logistic_l2":        {"type": "float", "default": 0.01, "values": [0.001, 0.005, 0.01, 0.05, 0.1]},
    },
    "lightgbm": {
        "lgbm_num_leaves":         {"type": "int",   "default": 31,   "values": [8, 15, 31, 63]},
        "lgbm_learning_rate":      {"type": "float", "default": 0.03, "values": [0.01, 0.03, 0.05, 0.1]},
        "lgbm_n_rounds":           {"type": "int",   "default": 300,  "values": [100, 200, 300, 500]},
        "lgbm_lambda_l1":          {"type": "float", "default": 0.1,  "values": [0.0, 0.1, 0.5, 1.0, 2.0]},
        "lgbm_lambda_l2":          {"type": "float", "default": 0.1,  "values": [0.0, 0.1, 0.5, 1.0, 2.0]},
    },
    "catboost": {
        "cb_iterations":           {"type": "int",   "default": 300,  "values": [100, 200, 300, 500]},
        "cb_learning_rate":        {"type": "float", "default": 0.05, "values": [0.01, 0.03, 0.05, 0.1]},
        "cb_depth":                {"type": "int",   "default": 6,    "values": [3, 4, 6, 8]},
        "cb_l2_leaf_reg":          {"type": "float", "default": 3.0,  "values": [1.0, 3.0, 5.0, 10.0]},
    },
    "mlp": {
        "mlp_hidden":              {"type": "str",   "default": "64,32", "values": ["32,16", "64,32", "128,64", "128,64,32"]},
        "mlp_lr":                  {"type": "float", "default": 0.001,"values": [0.0005, 0.001, 0.003, 0.01]},
        "mlp_epochs":              {"type": "int",   "default": 100,  "values": [50, 100, 150, 200]},
        "mlp_dropout":             {"type": "float", "default": 0.3,  "values": [0.1, 0.2, 0.3, 0.4, 0.5]},
    },
    "lstm": {
        "lstm_hidden_dim":         {"type": "int",   "default": 64,   "values": [32, 64, 128]},
        "lstm_n_layers":           {"type": "int",   "default": 2,    "values": [1, 2, 3]},
        "lstm_lr":                 {"type": "float", "default": 0.001,"values": [0.0005, 0.001, 0.003]},
        "lstm_epochs":             {"type": "int",   "default": 80,   "values": [40, 80, 120]},
        "lstm_dropout":            {"type": "float", "default": 0.3,  "values": [0.1, 0.2, 0.3, 0.5]},
    },
    "random_forest": {
        "rf_n_trees":              {"type": "int",   "default": 100,  "values": [50, 100, 200, 300]},
        "rf_max_depth":            {"type": "int",   "default": 5,    "values": [3, 5, 7, 10]},
        "rf_min_samples_leaf":     {"type": "int",   "default": 10,   "values": [5, 10, 15, 20]},
    },
    "hmm": {
        "hmm_states":              {"type": "int",   "default": 3,    "values": [2, 3, 4, 5]},
        "hmm_covariance_type":     {"type": "str",   "default": "diag", "values": ["diag", "full", "spherical"]},
        "hmm_n_iter":              {"type": "int",   "default": 100,  "values": [50, 100, 200]},
    },
    "kalman": {
        "kalman_process_noise":    {"type": "float", "default": 0.3,  "values": [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]},
        "kalman_measurement_noise":{"type": "float", "default": 2.5,  "values": [1.5, 2.0, 2.5, 3.0, 3.5, 5.0]},
    },
    "pagerank": {
        "network_decay":           {"type": "float", "default": 0.95, "values": [0.90, 0.93, 0.95, 0.97, 0.99]},
        "pagerank_damping":        {"type": "float", "default": 0.85, "values": [0.75, 0.80, 0.85, 0.90, 0.95]},
    },
    "garch": {
        "garch_alpha":             {"type": "float", "default": 0.10, "values": [0.05, 0.10, 0.15, 0.20, 0.30]},
        "garch_beta":              {"type": "float", "default": 0.80, "values": [0.60, 0.70, 0.80, 0.85, 0.90]},
    },
    "poisson": {
        "poisson_home_adv":        {"type": "float", "default": 1.2,  "values": [1.0, 1.1, 1.2, 1.3, 1.5]},
        "poisson_decay":           {"type": "float", "default": 0.98, "values": [0.95, 0.97, 0.98, 0.99, 1.0]},
    },
    "glicko": {
        "glicko_initial_rd":       {"type": "float", "default": 200,  "values": [100, 150, 200, 300, 400]},
        "glicko_initial_vol":      {"type": "float", "default": 0.06, "values": [0.03, 0.06, 0.09, 0.12]},
    },
    "bradley_terry": {
        "bt_decay":                {"type": "float", "default": 0.99, "values": [0.95, 0.97, 0.99, 1.0]},
        "bt_max_iterations":       {"type": "int",   "default": 100,  "values": [50, 100, 200]},
    },
    "monte_carlo": {
        "mc_simulations":          {"type": "int",   "default": 2000, "values": [1000, 2000, 3000, 5000]},
        "mc_kde_bandwidth":        {"type": "float", "default": 0.3,  "values": [0.15, 0.2, 0.3, 0.4, 0.5]},
    },
    "momentum": {
        "momentum_friction":       {"type": "float", "default": 0.05, "values": [0.02, 0.05, 0.08, 0.10, 0.15]},
        "momentum_velocity_window":{"type": "int",   "default": 5,    "values": [3, 5, 7, 10]},
        "momentum_impulse_window": {"type": "int",   "default": 3,    "values": [2, 3, 5]},
    },
    "clustering": {
        "n_clusters":              {"type": "int",   "default": 4,    "values": [3, 4, 5, 6, 8]},
    },
    "markov": {
        "markov_n_states":         {"type": "int",   "default": 4,    "values": [3, 4, 5, 6]},
    },
    "game_theory": {
        "gt_ema_alpha":            {"type": "float", "default": 0.05, "values": [0.02, 0.05, 0.10, 0.15]},
    },
    "info_theory": {
        "it_n_bins":               {"type": "int",   "default": 5,    "values": [3, 5, 7, 10]},
    },
    "svm": {
        "svm_C":                   {"type": "float", "default": 1.0,  "values": [0.1, 0.5, 1.0, 5.0, 10.0]},
        "svm_kernel":              {"type": "str",   "default": "rbf", "values": ["rbf", "linear", "poly"]},
        "svm_gamma":               {"type": "str",   "default": "scale", "values": ["scale", "auto", "0.01", "0.1"]},
    },
    "fibonacci": {
        "fib_ema_alpha":           {"type": "float", "default": 0.15, "values": [0.05, 0.10, 0.15, 0.20, 0.30]},
        "fib_swing_window":        {"type": "int",   "default": 10,   "values": [5, 8, 10, 15, 20]},
    },
    "benford": {
        "benford_window":          {"type": "int",   "default": 20,   "values": [10, 15, 20, 30]},
    },
    "evt": {
        "evt_threshold_quantile":  {"type": "float", "default": 0.90, "values": [0.80, 0.85, 0.90, 0.95]},
        "evt_min_exceedances":     {"type": "int",   "default": 5,    "values": [3, 5, 8, 10]},
    },
    "meta_bma": {
        "bma_decay":               {"type": "float", "default": 0.99, "values": [0.95, 0.97, 0.99, 1.0]},
    },
    "meta_cv": {
        "cv_min_confidence":       {"type": "float", "default": 0.05, "values": [0.0, 0.02, 0.05, 0.10]},
        "cv_agreement_bonus":      {"type": "float", "default": 1.5,  "values": [1.0, 1.2, 1.5, 2.0]},
    },
}


def get_model_hyperparams(model_name):
    """Get tunable hyperparameters for a specific model."""
    return MODEL_HYPERPARAMS.get(model_name, {})


def get_all_tunable_params():
    """Get all tunable params flattened: {param_name: {model, type, default, values}}."""
    result = {}
    for model_name, params in MODEL_HYPERPARAMS.items():
        for param_name, spec in params.items():
            result[param_name] = {"model": model_name, **spec}
    return result


def param_to_model(param_name):
    """Reverse lookup: which model owns a parameter. Returns None if unknown."""
    for model_name, params in MODEL_HYPERPARAMS.items():
        if param_name in params:
            return model_name
    return None


def load_model_params(sport, sport_dir):
    """Load per-model hyperparameters from settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("model_params", {})
        except Exception as e:
            logging.debug("Failed to load model params: %s", e)
    return {}


def save_model_params(sport, sport_dir, model_params):
    """Save per-model hyperparameters to settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.debug("Failed to load existing settings for param save: %s", e)
    data["model_params"] = model_params
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Build reverse lookup: alias -> canonical name
_ALIAS_MAP = {}
for canonical, spec in MEGA_PARAMS.items():
    _ALIAS_MAP[canonical] = canonical
    for alias in spec["aliases"]:
        _ALIAS_MAP[alias] = canonical

# Also register all model hyperparams so 'mega set' can handle them
for _model_name, _params in MODEL_HYPERPARAMS.items():
    for _param_name in _params:
        if _param_name not in _ALIAS_MAP:
            _ALIAS_MAP[_param_name] = _param_name


def resolve_mega_param(name):
    """Resolve a parameter name/alias to its canonical name. Returns None if unknown."""
    return _ALIAS_MAP.get(name.lower().strip())


def load_mega_params(sport, sport_dir):
    """Load all mega params from settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.debug("Failed to load mega params: %s", e)
    return {}


def save_mega_params(sport, sport_dir, params):
    """Save mega params to settings file (merges with existing)."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception as e:
            logging.debug("Failed to load existing mega params for save: %s", e)
    existing.update(params)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def handle_mega_set(cmd_args, sport, sport_dir):
    """Handle 'mega set param=value' command. Returns (success, message).

    Usage: mega set max_adj=0.10
           mega set kalman_pn=1.5
           mega set meta=ridge
           mega set mc_sims=3000
    """
    parts = cmd_args.split("=", 1)
    if len(parts) != 2:
        return False, "Usage: mega set <param>=<value>"

    raw_name = parts[0].strip()
    raw_value = parts[1].strip()

    canonical = resolve_mega_param(raw_name)
    if canonical is None:
        avail = ", ".join(sorted(MEGA_PARAMS.keys()))
        return False, "Unknown mega param: %s\n  Available: %s" % (raw_name, avail)

    # Check if this is a per-model hyperparameter
    owner_model = param_to_model(canonical)
    if owner_model is not None:
        hp_spec = MODEL_HYPERPARAMS[owner_model][canonical]
        # Parse value
        if hp_spec["type"] == "str":
            value = raw_value
        elif hp_spec["type"] == "int":
            try:
                value = int(float(raw_value))
            except ValueError:
                return False, "Invalid integer: %s" % raw_value
        elif hp_spec["type"] == "float":
            try:
                value = round(float(raw_value), 6)
            except ValueError:
                return False, "Invalid number: %s" % raw_value
        else:
            value = raw_value

        # Save to model_params section
        model_params = load_model_params(sport, sport_dir)
        model_params[canonical] = value
        save_model_params(sport, sport_dir, model_params)

        return True, "Set %s = %s  (model: %s)" % (canonical, value, owner_model)

    spec = MEGA_PARAMS[canonical]

    # Parse value
    if spec["type"] == "str":
        value = raw_value
    elif spec["type"] == "int":
        try:
            value = int(float(raw_value))
        except ValueError:
            return False, "Invalid integer: %s" % raw_value
    elif spec["type"] == "float":
        try:
            value = round(float(raw_value), 6)
        except ValueError:
            return False, "Invalid number: %s" % raw_value
    else:
        value = raw_value

    # Save
    params = load_mega_params(sport, sport_dir)
    params[canonical] = value
    save_mega_params(sport, sport_dir, params)

    return True, "Set %s = %s  (%s)" % (canonical, value, spec["desc"])


def print_mega_settings(sport, sport_dir):
    """Print all mega-ensemble settings with current values."""
    params = load_mega_params(sport, sport_dir)

    print("\n  %-30s %-12s  %s" % ("Parameter", "Value", "Description"))
    print("  " + "-" * 75)

    for canonical, spec in MEGA_PARAMS.items():
        value = params.get(canonical, "(default)")
        aliases = ", ".join(spec["aliases"][:2])
        print("  %-30s %-12s  %s" % (canonical, value, spec["desc"]))
        if aliases:
            print("  %-30s              aliases: %s" % ("", aliases))

    # Also show model switches
    switches = load_model_switches(sport, sport_dir)
    n_on = sum(1 for m in ALL_MODELS if is_model_enabled(m, switches))
    print("\n  Models: %d/%d enabled (use 'mega models' to see full list)" % (n_on, len(ALL_MODELS)))
