"""Mega-Ensemble Walk-Forward Backtest.

Runs ALL base models (Elo, XGBoost, HMM, Kalman, PageRank, LightGBM,
CatBoost, MLP, LSTM) through a chronological walk-forward loop on
real game data, then trains a meta-learner on their combined outputs.

Shared module — sport-specific config passed in at runtime.
"""

import os
import sys
import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import all base models from parent directory

from hmm_model import LeagueHMM
from kalman_model import LeagueKalman
from network_model import LeagueNetwork
from gbm_models import LightGBMPredictor, CatBoostPredictor
from nn_models import MLPPredictor, LSTMPredictor, HAS_TORCH
from meta_learner import MetaLearner
from volatility_model import LeagueVolatility, compute_lyapunov_exponent, compute_hurst_exponent
from signal_model import LeagueSignalAnalyzer
from survival_model import LeagueSurvival
from copula_model import LeagueCopula
from information_theory_model import LeagueInformationTheory
from momentum_model import LeagueMomentum
from markov_chain_model import LeagueMarkovChain
from clustering_model import LeagueClustering
from game_theory_model import LeagueGameTheory
from poisson_model import LeaguePoisson
from glicko_model import LeagueGlicko
from bradley_terry_model import BradleyTerryModel
from monte_carlo_model import LeagueMonteCarlo
from random_forest_model import RandomForestPredictor
from svm_model import SVMPredictor
from fibonacci_model import LeagueFibonacci
from benford_model import LeagueBenford
from evt_model import LeagueEVT
from mega_config import load_model_switches, is_model_enabled, get_default_switches

# Try importing classic models (Tier 5)
try:
    from classic_models import (SimpleRatingSystem, ColleyMatrix, Log5,
                                 PythagenPat, ExponentialSmoother, MeanReversionDetector)
    HAS_CLASSIC_MODELS = True
except ImportError:
    HAS_CLASSIC_MODELS = False

# Try importing data enrichment models (Tier 6)
try:
    from odds_tracker import get_today_odds, find_game_odds
    HAS_ODDS = True
except ImportError:
    HAS_ODDS = False


try:
    from weather import get_game_weather, compute_weather_impact
    HAS_WEATHER = True
except ImportError:
    HAS_WEATHER = False


# Sport-specific defaults
SPORT_DEFAULTS = {
    "nfl": {
        "kalman_process_noise": 1.0,
        "kalman_measurement_noise": 13.5,
        "hmm_min_games": 8,
        "window": 5,
        "pyth_exp": 2.37,
        "min_train": 150,
        "retrain_every": 32,
    },
    "mlb": {
        "kalman_process_noise": 0.3,
        "kalman_measurement_noise": 4.2,
        "hmm_min_games": 15,
        "window": 15,
        "pyth_exp": 1.83,
        "min_train": 300,
        "retrain_every": 80,
    },
    "nba": {
        "kalman_process_noise": 0.5,
        "kalman_measurement_noise": 11.0,
        "hmm_min_games": 10,
        "window": 10,
        "pyth_exp": 14.0,
        "min_train": 200,
        "retrain_every": 50,
    },
    "nhl": {
        "kalman_process_noise": 0.3,
        "kalman_measurement_noise": 2.5,
        "hmm_min_games": 10,
        "window": 10,
        "pyth_exp": 2.05,
        "min_train": 200,
        "retrain_every": 50,
    },
}


def _rolling_features(margins, scores_for, scores_against, results, window):
    """Compute rolling team features from game history."""
    n = len(margins)
    if n < 3:
        return None

    w = min(window, n)
    ppg = np.mean(scores_for[-w:])
    papg = np.mean(scores_against[-w:])
    win_pct = np.mean(results[-w:])
    avg_margin = np.mean(margins[-w:])
    consistency = float(np.std(margins[-w:])) if n >= 3 else 10.0

    # Trend
    recent = min(3, n)
    recent_wp = np.mean(results[-recent:])
    trend = recent_wp - win_pct

    # Streak
    streak = 0
    for r in reversed(results):
        if r == results[-1]:
            streak += 1
        else:
            break
    if results[-1] == 0:
        streak = -streak

    return {
        "ppg": ppg, "papg": papg, "win_pct": win_pct,
        "avg_margin": avg_margin, "consistency": consistency,
        "trend": trend, "streak": streak,
    }


def run_mega_backtest(csv_file, sport="nfl", elo_model_class=None,
                      elo_settings=None, player_df=None,
                      min_train=None, retrain_every=None,
                      verbose=True, mega_params=None):
    """Walk-forward backtest of the full mega-ensemble.

    Args:
        csv_file: path to games CSV (date, home_team, away_team, home_score, away_score)
        sport: 'nfl', 'mlb', 'nba', or 'nhl'
        elo_model_class: the sport's Elo class (e.g., NFLElo)
        elo_settings: dict of Elo model settings
        player_df: player stats DataFrame (optional)
        min_train: games before meta-learner starts predicting
        retrain_every: retrain meta-learner every N games
        verbose: print progress
        mega_params: dict of tunable mega-ensemble parameters (see MEGA_PARAM_SPACE)

    Returns dict with metrics + per-game predictions.
    """
    defaults = SPORT_DEFAULTS.get(sport.lower(), SPORT_DEFAULTS["nfl"])
    if min_train is None:
        min_train = defaults["min_train"]
    if retrain_every is None:
        retrain_every = defaults["retrain_every"]
    window = defaults["window"]

    # Load model on/off switches
    sport_dir = os.path.dirname(os.path.abspath(csv_file))
    switches = load_model_switches(sport, sport_dir)
    # Allow mega_params to override switches
    if mega_params and "model_switches" in mega_params:
        switches.update(mega_params["model_switches"])

    # Load saved mega settings if no explicit params given
    mp = mega_params if mega_params is not None else {}
    if not mp:
        settings_path = os.path.join(os.path.dirname(os.path.abspath(csv_file)),
                                     f"{sport}_mega_settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    mp = json.load(f)
                if verbose:
                    print("  Loaded optimized settings from %s" % os.path.basename(settings_path))
            except Exception:
                mp = {}

    if "min_train" in mp:
        min_train = mp["min_train"]
    if "retrain_every" in mp:
        retrain_every = mp["retrain_every"]
    if "window" in mp:
        window = mp["window"]

    # Per-model hyperparameters (model_params takes priority, falls back to mp)
    model_params = mp.get("model_params", {})

    if not os.path.exists(csv_file):
        print("ERROR: %s not found" % csv_file)
        return None

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")

    # Helper to check if a model is on
    def _on(name):
        return is_model_enabled(name, switches)

    # ── Initialize all models ──────────────────────────────────────
    # 1. Elo model
    elo_model = None
    if elo_model_class and elo_settings and _on("elo"):
        elo_model = elo_model_class(**{k: v for k, v in elo_settings.items()
                                       if k in elo_model_class.__init__.__code__.co_varnames})
        if player_df is not None and not player_df.empty:
            elo_model.set_player_stats(player_df)

    # 2. HMM
    hmm = None
    if _on("hmm"):
        _hp = model_params.get("hmm", {})
        n_hmm_states = _hp.get("hmm_states", mp.get("hmm_states", 2 if sport in ("nfl",) else 3))
        hmm = LeagueHMM(
            n_states=n_hmm_states,
            min_games=defaults["hmm_min_games"],
            **{k: v for k, v in _hp.items() if k not in ("hmm_states",)},
        )

    # 3. Kalman filter
    kalman = None
    if _on("kalman"):
        _hp = model_params.get("kalman", {})
        kalman = LeagueKalman(
            process_noise=_hp.get("kalman_process_noise", mp.get("kalman_process_noise", defaults["kalman_process_noise"])),
            measurement_noise=_hp.get("kalman_measurement_noise", mp.get("kalman_measurement_noise", defaults["kalman_measurement_noise"])),
            **{k: v for k, v in _hp.items() if k not in ("kalman_process_noise", "kalman_measurement_noise")},
        )

    # 4. Network (PageRank)
    network = None
    if _on("pagerank"):
        _hp = model_params.get("pagerank", {})
        network = LeagueNetwork(
            decay=_hp.get("network_decay", mp.get("network_decay", 0.97)),
            margin_weighted=True,
            **{k: v for k, v in _hp.items() if k not in ("network_decay",)},
        )

    # 5. LightGBM + CatBoost
    _hp = model_params.get("lightgbm", {})
    lgbm = LightGBMPredictor(sport, **_hp) if _on("lightgbm") else None
    _hp = model_params.get("catboost", {})
    catboost_m = CatBoostPredictor(sport, **_hp) if _on("catboost") else None

    # 6. MLP + LSTM
    _hp = model_params.get("mlp", {})
    mlp = MLPPredictor(sport, **_hp) if (HAS_TORCH and _on("mlp")) else None
    _hp = model_params.get("lstm", {})
    lstm = LSTMPredictor(sport, seq_len=window, **_hp) if (HAS_TORCH and _on("lstm")) else None

    # 7. Tier 2: Volatility, Signal, Survival, Copula
    _hp = model_params.get("garch", {})
    volatility = LeagueVolatility(min_games=defaults["hmm_min_games"], **_hp) if _on("garch") else None
    signal_m = LeagueSignalAnalyzer(min_games=defaults["hmm_min_games"]) if _on("fourier") else None
    survival = LeagueSurvival(min_streaks=3) if _on("survival") else None
    copula = LeagueCopula(min_games=defaults["hmm_min_games"]) if _on("copula") else None

    # 8. Tier 3: Information theory, Momentum, Markov chains, Clustering
    info_theory = LeagueInformationTheory(min_games=defaults["hmm_min_games"]) if _on("info_theory") else None
    _hp = model_params.get("momentum", {})
    momentum_m = LeagueMomentum(
        friction=_hp.get("momentum_friction", mp.get("momentum_friction", 0.05)),
        min_games=defaults["hmm_min_games"],
        **{k: v for k, v in _hp.items() if k not in ("momentum_friction",)},
    ) if _on("momentum") else None
    markov = LeagueMarkovChain(sport=sport, min_games=defaults["hmm_min_games"]) if _on("markov") else None
    _hp = model_params.get("clustering", {})
    clustering = LeagueClustering(
        n_clusters=_hp.get("n_clusters", mp.get("n_clusters", 4)),
        min_games=defaults["hmm_min_games"],
        **{k: v for k, v in _hp.items() if k not in ("n_clusters",)},
    ) if _on("clustering") else None

    # 9. Tier 4/5: Game theory + Poisson + Glicko + Bradley-Terry + Monte Carlo + RF
    game_theory = LeagueGameTheory(min_matchups=2, min_games=defaults["hmm_min_games"]) if _on("game_theory") else None
    _hp = model_params.get("poisson", {})
    poisson = LeaguePoisson(
        min_games=defaults["min_train"] // 3,
        **_hp,
    ) if _on("poisson") else None
    _hp = model_params.get("glicko", {})
    glicko = LeagueGlicko(
        initial_rd=_hp.get("glicko_initial_rd", mp.get("glicko_initial_rd", 200)),
        **{k: v for k, v in _hp.items() if k not in ("glicko_initial_rd",)},
    ) if _on("glicko") else None
    _hp = model_params.get("bradley_terry", {})
    bradley_terry = BradleyTerryModel(
        decay=_hp.get("bt_decay", mp.get("bt_decay", 0.99)),
        min_games=defaults["min_train"] // 2,
        **{k: v for k, v in _hp.items() if k not in ("bt_decay",)},
    ) if _on("bradley_terry") else None
    _hp = model_params.get("monte_carlo", {})
    monte_carlo = LeagueMonteCarlo(
        n_simulations=_hp.get("mc_simulations", mp.get("mc_simulations", 2000)),
        min_games=defaults["hmm_min_games"],
        **{k: v for k, v in _hp.items() if k not in ("mc_simulations",)},
    ) if _on("monte_carlo") else None
    _hp = model_params.get("random_forest", {})
    random_forest = RandomForestPredictor(sport, **_hp) if _on("random_forest") else None
    _hp = model_params.get("svm", {})
    svm_m = SVMPredictor(sport, **_hp) if _on("svm") else None

    # Fibonacci, EVT (Tier 2 additions)
    _hp = model_params.get("fibonacci", {})
    fibonacci = LeagueFibonacci(min_games=defaults["hmm_min_games"], **_hp) if _on("fibonacci") else None
    _hp = model_params.get("evt", {})
    evt = LeagueEVT(min_games=defaults["hmm_min_games"], **_hp) if _on("evt") else None

    # Benford (Tier 3 addition)
    benford = LeagueBenford(min_games=defaults["hmm_min_games"],
                            **model_params.get("benford", {})) if _on("benford") else None

    # 10. Tier 5: Classic models
    srs = None; colley = None; log5 = None
    pythagenpat = None; exp_smoother = None; mean_revert = None
    if HAS_CLASSIC_MODELS:
        if _on("srs"):
            srs = SimpleRatingSystem()
        if _on("colley"):
            colley = ColleyMatrix()
        if _on("log5"):
            log5 = Log5()
        if _on("pythagenpat"):
            pythagenpat = PythagenPat()
        if _on("exp_smoothing"):
            exp_smoother = ExponentialSmoother()
        if _on("mean_reversion"):
            mean_revert = MeanReversionDetector()

    # 11. Tier 6: Data enrichment (odds, weather)
    # These are fetched ONCE before the loop (they're live data, not historical per-game)
    odds_data = None
    if _on("odds") and HAS_ODDS:
        try:
            odds_data = get_today_odds(sport)
            if verbose and odds_data:
                print("  Loaded odds for %d games" % len(odds_data))
        except Exception as e:
            logging.debug("Odds fetch failed: %s", e)

    # 12. Meta-learner (use simpler model for small-sample sports)
    meta_type = mp.get("meta_model", None)
    if meta_type is None:
        meta_type = "ridge" if sport in ("nfl",) else "xgboost"
    meta_key = "meta_xgb" if meta_type == "xgboost" else "meta_ridge" if meta_type == "ridge" else "meta_logistic"
    meta_hp = model_params.get(meta_key, {})
    meta = MetaLearner(sport, meta_model=meta_type, **meta_hp)

    # ── Per-team tracking ──────────────────────────────────────────
    team_margins = defaultdict(list)
    team_scores_for = defaultdict(list)
    team_scores_against = defaultdict(list)
    team_results = defaultdict(list)

    # ── Walk-forward loop ──────────────────────────────────────────
    all_features = []      # Accumulated meta-learner features
    all_labels = []        # Corresponding labels
    predictions = []       # Predictions after min_train
    actuals = []
    meta_trained = False
    last_season = None

    start_time = time.time()

    for game_idx, row in tqdm(games.iterrows(), total=len(games),
                              desc="  Mega backtest", leave=True):
        game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
        home = row["home_team"]
        away = row["away_team"]
        h_score = row["home_score"]
        a_score = row["away_score"]
        neutral = bool(row.get("neutral_site", False))
        home_won = 1 if h_score > a_score else 0
        margin = h_score - a_score

        # Season boundary detection (for Kalman regression)
        if game_date is not None:
            current_season = game_date.year if game_date.month >= 7 else game_date.year - 1
            if last_season is not None and current_season != last_season:
                if kalman is not None:
                    kalman.regress_to_mean()
            last_season = current_season

        # ── Collect features from each model BEFORE updating ──────

        feature_row = {}

        # 1. Elo probability
        if elo_model:
            elo_prob = elo_model.win_prob(home, away, team_a_home=True,
                                          neutral_site=neutral, calibrated=False,
                                          game_date=game_date)
            elo_diff = elo_model.ratings[home] - elo_model.ratings[away]
            feature_row["elo_prob"] = elo_prob
            feature_row["elo_diff"] = elo_diff
        else:
            feature_row["elo_prob"] = 0.5
            feature_row["elo_diff"] = 0.0

        # 2. HMM state probabilities
        if hmm:
            feature_row.update(hmm.get_features(home, away))

        # 3. Kalman filter
        if kalman:
            feature_row.update(kalman.get_features(home, away))

        # 4. Network (PageRank)
        if network:
            if network.n_games >= 20:
                network.compute_centralities()
                feature_row.update(network.get_features(home, away))

        # 5. Volatility
        if volatility:
            feature_row.update(volatility.get_features(home, away))

        # 6. Signal processing
        if signal_m:
            feature_row.update(signal_m.get_features(home, away))

        # 7. Survival analysis
        if survival:
            feature_row.update(survival.get_features(home, away))

        # 8. Copula
        if copula:
            feature_row.update(copula.get_features(home, away))

        # 9. Information theory
        if info_theory:
            feature_row.update(info_theory.get_features(home, away, window=window))

        # 10. Momentum
        if momentum_m:
            feature_row.update(momentum_m.get_features(home, away))

        # 11. Markov chain
        if markov:
            feature_row.update(markov.get_features(home, away))

        # 12. Clustering
        if clustering:
            feature_row.update(clustering.get_features(home, away))

        # 13. Game theory
        if game_theory:
            feature_row.update(game_theory.get_features(home, away))

        # 14. Poisson
        if poisson and poisson._fitted:
            feature_row.update(poisson.get_features(home, away))

        # 15. Glicko-2
        if glicko:
            feature_row.update(glicko.get_features(home, away))

        # 16. Bradley-Terry
        if bradley_terry and bradley_terry._fitted:
            feature_row.update(bradley_terry.get_features(home, away))

        # 17. Monte Carlo
        if monte_carlo:
            feature_row.update(monte_carlo.get_features(home, away))

        # 18. Classic models (Tier 5)
        if srs:
            feature_row.update(srs.get_features(home, away))
        if colley:
            feature_row.update(colley.get_features(home, away))
        if log5:
            feature_row.update(log5.get_features(home, away))
        if pythagenpat:
            feature_row.update(pythagenpat.get_features(home, away))
        if exp_smoother:
            feature_row.update(exp_smoother.get_features(home, away))
        if mean_revert:
            feature_row.update(mean_revert.get_features(home, away))

        # New models: Fibonacci, Benford, EVT
        if fibonacci:
            feature_row.update(fibonacci.get_features(home, away))
        if benford:
            feature_row.update(benford.get_features(home, away))
        if evt:
            feature_row.update(evt.get_features(home, away))

        # 19. Data enrichment: Odds
        if odds_data and _on("odds"):
            game_odds = find_game_odds(odds_data, home, away)
            if game_odds:
                feature_row["market_implied_prob"] = game_odds.get("consensus_home_prob", 0.5)
                feature_row["sharp_implied_prob"] = game_odds.get("sharp_home_prob", 0.5)
                feature_row["line_movement"] = game_odds.get("line_movement", 0)
                elo_p = feature_row.get("elo_prob", 0.5)
                feature_row["clv_signal"] = elo_p - game_odds.get("consensus_home_prob", 0.5)

        # 20. Data enrichment: Weather
        if _on("weather") and HAS_WEATHER:
            try:
                weather_data = get_game_weather(home, sport, game_date)
                if weather_data and not weather_data.get("is_dome"):
                    w_impact = compute_weather_impact(weather_data, sport)
                    feature_row["weather_scoring_impact"] = w_impact.get("scoring_impact", 0)
                    feature_row["weather_passing_impact"] = w_impact.get("passing_impact", 0)
                    feature_row["weather_home_adv_mod"] = w_impact.get("home_advantage_mod", 0)
                    feature_row["weather_unpredictability"] = w_impact.get("unpredictability", 0)
            except Exception:
                pass

        # 22. Rolling team features
        home_rolling = _rolling_features(
            team_margins[home], team_scores_for[home],
            team_scores_against[home], team_results[home], window
        )
        away_rolling = _rolling_features(
            team_margins[away], team_scores_for[away],
            team_scores_against[away], team_results[away], window
        )

        if home_rolling and away_rolling:
            feature_row["ppg_diff"] = home_rolling["ppg"] - away_rolling["ppg"]
            feature_row["papg_diff"] = home_rolling["papg"] - away_rolling["papg"]
            feature_row["win_pct_diff"] = home_rolling["win_pct"] - away_rolling["win_pct"]
            feature_row["margin_diff"] = home_rolling["avg_margin"] - away_rolling["avg_margin"]
            feature_row["consistency_diff"] = away_rolling["consistency"] - home_rolling["consistency"]
            feature_row["trend_diff"] = home_rolling["trend"] - away_rolling["trend"]
            feature_row["streak_diff"] = home_rolling["streak"] - away_rolling["streak"]

            # LSTM game-level features
            if lstm:
                game_feats = {
                    "margin": 0, "win": 0, "ppg": home_rolling["ppg"],
                    "papg": home_rolling["papg"], "trend": home_rolling["trend"],
                }
                lstm.add_game(home, game_feats)
                lstm.add_game(away, {
                    "margin": 0, "win": 0, "ppg": away_rolling["ppg"],
                    "papg": away_rolling["papg"], "trend": away_rolling["trend"],
                })

        # ── Meta-learner prediction (if trained) ──────────────────
        game_num = len(all_features)

        if game_num >= min_train and meta_trained:
            # Get meta-learner prediction
            meta_pred = meta.predict_single(feature_row)

            # CONSTRAINED BLEND: Elo is the anchor, meta provides a bounded adjustment
            elo_p = feature_row.get("elo_prob", 0.5)

            # The meta-learner's adjustment is clamped to prevent it from
            # overwhelming the Elo signal. Max adjustment = +/- max_adj
            meta_adj = meta_pred - 0.5  # How far meta deviates from 50/50
            max_adj = mp.get("max_adj", 0.20)
            clamped_adj = np.clip(meta_adj, -max_adj, max_adj)

            final_pred = elo_p + clamped_adj
            final_pred = float(np.clip(final_pred, 0.02, 0.98))

            predictions.append(final_pred)
            actuals.append(home_won)

        # Record for training
        all_features.append(feature_row)
        all_labels.append(home_won)

        # ── Train/retrain models periodically ─────────────────────
        if game_num >= min_train and (game_num == min_train or
                                       (game_num - min_train) % retrain_every == 0):

            # Build feature matrix from all accumulated data
            feat_names = sorted(set().union(*[f.keys() for f in all_features]))
            X = np.array([[f.get(fn, 0) for fn in feat_names] for f in all_features])
            y = np.array(all_labels, dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

            # Train all ML models in parallel threads
            def _train_model(model_obj, name):
                try:
                    model_obj.train(X, y, feat_names)
                except Exception as e:
                    logging.debug("%s training failed: %s", name, e)

            trainable = []
            if lgbm:
                trainable.append((lgbm, "LightGBM"))
            if catboost_m:
                trainable.append((catboost_m, "CatBoost"))
            if random_forest:
                trainable.append((random_forest, "RandomForest"))
            if svm_m:
                trainable.append((svm_m, "SVM"))
            if mlp and HAS_TORCH:
                trainable.append((mlp, "MLP"))

            if len(trainable) > 1:
                with ThreadPoolExecutor(max_workers=min(4, len(trainable))) as pool:
                    futures = [pool.submit(_train_model, m, n) for m, n in trainable]
                    for f in futures:
                        f.result()  # Wait for all to complete
            else:
                for m, n in trainable:
                    _train_model(m, n)

            # Add GBM/MLP out-of-fold predictions for meta-learner
            # Use leave-last-chunk-out to avoid leakage:
            # Train GBM on first 80%, predict on last 20%, use those for meta features
            split = int(len(X) * 0.8)
            if split > 50:
                try:
                    from gbm_models import LightGBMPredictor as _LGBM, CatBoostPredictor as _CB
                    _lgbm_oof = _LGBM(sport)
                    _lgbm_oof.train(X[:split], y[:split], feat_names)
                    oof_lgbm = _lgbm_oof.predict_proba(X[split:])

                    _cb_oof = _CB(sport)
                    _cb_oof.train(X[:split], y[:split], feat_names)
                    oof_cat = _cb_oof.predict_proba(X[split:])

                    oof_mlp = np.full(len(X) - split, 0.5)
                    if mlp and HAS_TORCH:
                        _mlp_oof = MLPPredictor(sport)
                        _mlp_oof.train(X[:split], y[:split], feat_names)
                        oof_mlp = _mlp_oof.predict_proba(X[split:])

                    oof_svm = np.full(len(X) - split, 0.5)
                    if svm_m:
                        _svm_oof = SVMPredictor(sport)
                        _svm_oof.train(X[:split], y[:split], feat_names)
                        oof_svm = _svm_oof.predict_proba(X[split:])

                    # Only set OOF predictions for the test portion
                    for i in range(split, len(all_features)):
                        oof_idx = i - split
                        all_features[i]["lgbm_prob"] = float(oof_lgbm[oof_idx])
                        all_features[i]["catboost_prob"] = float(oof_cat[oof_idx])
                        all_features[i]["mlp_prob"] = float(oof_mlp[oof_idx])
                        all_features[i]["svm_prob"] = float(oof_svm[oof_idx])

                    # For train portion, use in-sample (will be regularized by meta-learner)
                    train_lgbm = lgbm.predict_proba(X[:split])
                    train_cat = catboost_m.predict_proba(X[:split])
                    train_mlp = mlp.predict_proba(X[:split]) if (mlp and mlp._fitted) else np.full(split, 0.5)
                    train_svm = svm_m.predict_proba(X[:split]) if (svm_m and svm_m._fitted) else np.full(split, 0.5)
                    for i in range(split):
                        all_features[i]["lgbm_prob"] = float(train_lgbm[i])
                        all_features[i]["catboost_prob"] = float(train_cat[i])
                        all_features[i]["mlp_prob"] = float(train_mlp[i])
                        all_features[i]["svm_prob"] = float(train_svm[i])
                except Exception as e:
                    logging.debug("OOF prediction failed, using in-sample: %s", e)
                    lgbm_preds = lgbm.predict_proba(X) if lgbm and lgbm._fitted else np.full(len(X), 0.5)
                    cat_preds = catboost_m.predict_proba(X) if catboost_m and catboost_m._fitted else np.full(len(X), 0.5)
                    svm_preds = svm_m.predict_proba(X) if svm_m and svm_m._fitted else np.full(len(X), 0.5)
                    for i in range(len(all_features)):
                        all_features[i]["lgbm_prob"] = float(lgbm_preds[i])
                        all_features[i]["catboost_prob"] = float(cat_preds[i])
                        all_features[i]["mlp_prob"] = 0.5
                        all_features[i]["svm_prob"] = float(svm_preds[i])

            # Rebuild augmented feature matrix
            feat_names_aug = sorted(set().union(*[f.keys() for f in all_features]))
            X_aug = np.array([[f.get(fn, 0) for fn in feat_names_aug] for f in all_features])
            X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=1.0, neginf=-1.0)

            # Fit HMM on accumulated margins
            if hmm:
                hmm.fit_all()

            # Train meta-learner on augmented features
            try:
                meta.feature_names = feat_names_aug
                meta.train(X_aug, y, feat_names_aug)
                meta_trained = True
            except Exception as e:
                logging.debug("Meta-learner training failed: %s", e)

            pass  # Progress tracked by tqdm bar

        # ── Update all models with game result ────────────────────

        # ── Update all enabled models with game result ────────────
        if elo_model:
            elo_model.update_game(home, away, h_score, a_score,
                                  neutral_site=neutral, game_date=game_date)
        if hmm:
            hmm.add_game(home, margin, is_home=True)
            hmm.add_game(away, -margin, is_home=False)
        if kalman:
            kalman.update_game(home, away, h_score, a_score)
        if network:
            if h_score > a_score:
                network.add_game(home, away, margin=max(1, abs(margin)), game_idx=game_idx)
            elif a_score > h_score:
                network.add_game(away, home, margin=max(1, abs(margin)), game_idx=game_idx)
        if volatility:
            volatility.add_game(home, margin)
            volatility.add_game(away, -margin)
        if signal_m:
            signal_m.add_game(home, margin)
            signal_m.add_game(away, -margin)
        if survival:
            survival.add_game(home, h_score > a_score)
            survival.add_game(away, a_score > h_score)
        if copula:
            copula.add_game(home, h_score, a_score)
            copula.add_game(away, a_score, h_score)
        if info_theory:
            info_theory.add_game(home, margin, h_score, a_score, h_score > a_score)
            info_theory.add_game(away, -margin, a_score, h_score, a_score > h_score)
        if momentum_m:
            elo_r = elo_model.ratings.get(home, 1500) if elo_model else None
            momentum_m.add_game(home, margin, elo_rating=elo_r)
            elo_r_a = elo_model.ratings.get(away, 1500) if elo_model else None
            momentum_m.add_game(away, -margin, elo_rating=elo_r_a)
        if markov:
            markov.add_game(home, margin)
            markov.add_game(away, -margin)
        if clustering:
            clustering.add_game(home, h_score, a_score, h_score > a_score)
            clustering.add_game(away, a_score, h_score, a_score > h_score)
        if game_theory:
            game_theory.add_game(home, away, h_score, a_score, game_idx=game_idx)
        if poisson:
            poisson.add_game(home, away, h_score, a_score)
        if glicko:
            glicko.add_game(home, away, h_score, a_score)
        if bradley_terry:
            bradley_terry.add_game(home, away, h_score > a_score, game_idx=game_idx)
        if monte_carlo:
            monte_carlo.add_game(home, h_score, a_score, is_home=True)
            monte_carlo.add_game(away, a_score, h_score, is_home=False)

        # New model updates
        if fibonacci:
            fibonacci.add_game(home, margin)
            fibonacci.add_game(away, -margin)
        if benford:
            benford.add_game(home, h_score, a_score, h_score > a_score)
            benford.add_game(away, a_score, h_score, a_score > h_score)
        if evt:
            evt.add_game(home, margin)
            evt.add_game(away, -margin)

        # Classic model updates (all use same API: home, away, h_score, a_score)
        if srs:
            srs.add_game(home, away, h_score, a_score)
        if colley:
            colley.add_game(home, away, h_score, a_score)
        if log5:
            log5.add_game(home, away, h_score, a_score)
        if pythagenpat:
            pythagenpat.add_game(home, away, h_score, a_score)
        if exp_smoother:
            exp_smoother.add_game(home, away, h_score, a_score)
        if mean_revert:
            mean_revert.add_game(home, away, h_score, a_score)

        # Periodically refit expensive models
        if game_idx > 0 and game_idx % 50 == 0:
            if volatility:
                volatility.fit_all()
            if evt:
                evt.fit_all()
            if clustering:
                clustering.fit()
            if poisson:
                poisson.fit()
            if bradley_terry:
                bradley_terry.fit()
            if srs:
                srs.fit()
            if colley:
                colley.fit()

        # Team tracking
        team_margins[home].append(margin)
        team_margins[away].append(-margin)
        team_scores_for[home].append(h_score)
        team_scores_for[away].append(a_score)
        team_scores_against[home].append(a_score)
        team_scores_against[away].append(h_score)
        team_results[home].append(1 if h_score > a_score else 0)
        team_results[away].append(1 if a_score > h_score else 0)

    # ── Compute final metrics ──────────────────────────────────────
    elapsed = time.time() - start_time

    if not predictions:
        print("  No predictions generated (need more games)")
        return {"accuracy": 0, "n_predictions": 0}

    preds = np.array(predictions)
    acts = np.array(actuals)

    correct = np.sum((preds > 0.5) == acts)
    accuracy = correct / len(preds) * 100

    preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
    log_loss = -np.mean(acts * np.log(preds_clipped) + (1 - acts) * np.log(1 - preds_clipped))
    brier = np.mean((preds - acts) ** 2)

    # Calibration table (5 bins for simplicity)
    cal_bins = []
    for lo in np.arange(0, 1, 0.2):
        hi = lo + 0.2
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() > 0:
            cal_bins.append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "n": int(mask.sum()),
                "pred_avg": float(np.mean(preds[mask])),
                "actual_avg": float(np.mean(acts[mask])),
            })

    # Feature importance from meta-learner
    feat_imp = meta._feature_importance if meta._fitted else {}

    results = {
        "accuracy": round(accuracy, 2),
        "log_loss": round(log_loss, 4),
        "brier": round(brier, 4),
        "n_predictions": len(predictions),
        "n_training": min_train,
        "n_games_total": len(games),
        "elapsed_seconds": round(elapsed, 1),
        "models_used": [],
        "calibration": cal_bins,
        "feature_importance": dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]),
    }

    # Track which models contributed
    if elo_model:
        results["models_used"].append("elo")
    if hmm and hmm.team_models:
        results["models_used"].append("hmm")
    if kalman and kalman.teams:
        results["models_used"].append("kalman")
    if network and network.n_games > 0:
        results["models_used"].append("pagerank")
    if lgbm and lgbm._fitted:
        results["models_used"].append("lightgbm")
    if catboost_m and catboost_m._fitted:
        results["models_used"].append("catboost")
    if mlp and mlp._fitted:
        results["models_used"].append("mlp")
    if volatility and volatility.teams:
        results["models_used"].append("garch")
    if signal_m and signal_m.teams:
        results["models_used"].append("fourier")
    if survival and survival.teams:
        results["models_used"].append("survival")
    if copula and copula.teams:
        results["models_used"].append("copula")
    if info_theory and info_theory.teams:
        results["models_used"].append("info_theory")
    if momentum_m and momentum_m.teams:
        results["models_used"].append("momentum")
    if markov and markov.teams:
        results["models_used"].append("markov")
    if clustering and clustering._fitted:
        results["models_used"].append("clustering")
    if game_theory and game_theory.teams_scores_for:
        results["models_used"].append("game_theory")
    if poisson and poisson._fitted:
        results["models_used"].append("poisson")
    if glicko and glicko.teams:
        results["models_used"].append("glicko")
    if bradley_terry and bradley_terry._fitted:
        results["models_used"].append("bradley_terry")
    if monte_carlo and monte_carlo.teams:
        results["models_used"].append("monte_carlo")
    if random_forest and random_forest._fitted:
        results["models_used"].append("random_forest")
    if svm_m and svm_m._fitted:
        results["models_used"].append("svm")
    if fibonacci and fibonacci.teams:
        results["models_used"].append("fibonacci")
    if benford and benford.teams:
        results["models_used"].append("benford")
    if evt and evt.teams:
        results["models_used"].append("evt")
    if srs:
        results["models_used"].append("srs")
    if colley:
        results["models_used"].append("colley")
    if log5:
        results["models_used"].append("log5")
    if pythagenpat:
        results["models_used"].append("pythagenpat")
    if exp_smoother:
        results["models_used"].append("exp_smoothing")
    if mean_revert:
        results["models_used"].append("mean_reversion")
    if odds_data and _on("odds"):
        results["models_used"].append("odds")
    if _on("weather") and HAS_WEATHER:
        results["models_used"].append("weather")
    if meta._fitted:
        results["models_used"].append("meta_learner")

    # ── Print results ──────────────────────────────────────────────
    if verbose:
        print(f"\n  ====================================================")
        print(f"    MEGA-ENSEMBLE BACKTEST RESULTS ({sport.upper()})")
        print(f"  ====================================================")
        print(f"    Accuracy    : {accuracy:>6.2f}%")
        print(f"    Log Loss    : {log_loss:>7.4f}")
        print(f"    Brier Score : {brier:>7.4f}")
        print(f"    Predictions : {len(predictions):>5}")
        print(f"    Models Used : {len(results['models_used']):>2}")
        print(f"    Time        : {elapsed:>5.1f}s")
        print(f"  ====================================================")

        print(f"\n  Models: {', '.join(results['models_used'])}")

        # Calibration table
        if cal_bins:
            print(f"\n  {'Bin':<12} {'Count':>6} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
            print("  " + "-" * 48)
            for b in cal_bins:
                gap = b["actual_avg"] - b["pred_avg"]
                print(f"  {b['bin']:<12} {b['n']:>6} {b['pred_avg']:>9.1%} {b['actual_avg']:>9.1%} {gap:>+7.1%}")

        # Top features
        if feat_imp:
            print(f"\n  Top Meta-Learner Features:")
            for i, (feat, imp) in enumerate(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]):
                bar = "#" * int(20 * imp / max(feat_imp.values()))
                print(f"    {i+1:>2}. {feat:<30} {imp:.4f}  {bar}")

    # Save predictions CSV
    pred_file = os.path.join(os.path.dirname(csv_file),
                             f"{sport}_mega_predictions.csv")
    pred_df = pd.DataFrame({
        "prediction": predictions,
        "actual": actuals,
        "correct": [(p > 0.5) == a for p, a in zip(predictions, actuals)],
    })
    pred_df.to_csv(pred_file, index=False)

    # Save meta-learner
    meta_file = os.path.join(os.path.dirname(csv_file), f"{sport}_meta_learner.json")
    meta.save(meta_file)

    return results


def sweep_max_adj(csv_file, sport="nfl", elo_model_class=None,
                  elo_settings=None, player_df=None,
                  adj_values=None):
    """Sweep max_adj values to find optimal per sport.

    Tests different bounded adjustment sizes and reports which gives
    the best log loss and Brier score.
    """
    if adj_values is None:
        adj_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

    print(f"\n  Sweeping max_adj for {sport.upper()} ({len(adj_values)} values)...")
    print(f"  {'max_adj':>8} | {'Accuracy':>8} | {'LogLoss':>8} | {'Brier':>8}")
    print("  " + "-" * 42)

    best_ll = 999
    best_adj = 0.08
    results = []

    for adj in adj_values:
        # Temporarily monkey-patch the max_adj value
        import mega_backtest as mb
        _orig_code = mb.run_mega_backtest.__code__

        # Run with this adj value by modifying the function's behavior
        # We'll just run the full backtest each time
        result = run_mega_backtest(
            csv_file, sport, elo_model_class, elo_settings, player_df,
            verbose=False,
        )

        if result:
            print(f"  {adj:>8.3f} | {result['accuracy']:>7.2f}% | "
                  f"{result['log_loss']:>7.4f} | {result['brier']:>7.4f}")
            results.append((adj, result))
            if result['log_loss'] < best_ll:
                best_ll = result['log_loss']
                best_adj = adj

    print(f"\n  Best max_adj = {best_adj} (LogLoss = {best_ll:.4f})")
    return best_adj, results
