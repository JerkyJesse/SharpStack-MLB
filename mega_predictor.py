"""Mega-Ensemble Live Predictor.

Initializes all 35 base models, replays game history through them,
loads the trained meta-learner, and provides live predictions that
use the full mega-ensemble for bounded adjustment on top of Elo.

This module bridges the gap between mega_backtest (training) and
elo_model (live predictions) so that today/predict commands use
ALL 35 models, not just Elo + XGBoost.
"""

import os
import sys
import json
import logging
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure parent directory is on path for imports
_parent = os.path.dirname(__file__)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from hmm_model import LeagueHMM
from kalman_model import LeagueKalman
from network_model import LeagueNetwork
from gbm_models import LightGBMPredictor, CatBoostPredictor
from nn_models import MLPPredictor, LSTMPredictor, HAS_TORCH
from meta_learner import MetaLearner
from volatility_model import LeagueVolatility
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
from mega_config import load_model_switches, is_model_enabled
from mega_backtest import SPORT_DEFAULTS, _rolling_features

try:
    from classic_models import (SimpleRatingSystem, ColleyMatrix, Log5,
                                 PythagenPat, ExponentialSmoother, MeanReversionDetector)
    HAS_CLASSIC_MODELS = True
except ImportError:
    HAS_CLASSIC_MODELS = False


class MegaPredictor:
    """Live prediction engine using all 35 mega-ensemble models.

    Initializes all base models, replays game history to bring them
    to current state, loads the trained meta-learner, and provides
    bounded probability adjustments for live predictions.
    """

    def __init__(self, sport, csv_file, elo_model_class=None,
                 elo_settings=None, player_df=None):
        self.sport = sport
        self._available = False
        self._models = {}
        self._meta = None
        self._max_adj = 0.08
        self._n_models = 0

        sport_dir = os.path.dirname(os.path.abspath(csv_file))

        # Load mega settings
        settings_path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
        mp = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    mp = json.load(f)
            except Exception as e:
                logging.debug("Mega settings load failed: %s", e)

        self._max_adj = mp.get("max_adj", 0.08)
        defaults = SPORT_DEFAULTS.get(sport, SPORT_DEFAULTS["nfl"])
        window = mp.get("window", defaults["window"])
        min_train = mp.get("min_train", defaults["min_train"])
        retrain_every = mp.get("retrain_every", defaults["retrain_every"])
        self._window = window

        # Load model switches
        switches = load_model_switches(sport, sport_dir)
        if mp.get("model_switches"):
            switches.update(mp["model_switches"])

        def _on(name):
            return is_model_enabled(name, switches)

        # Load meta-learner
        meta_type = mp.get("meta_model", "ridge" if sport == "nfl" else "xgboost")
        meta_file = os.path.join(sport_dir, f"{sport}_meta_learner.json")
        self._meta = MetaLearner(sport, meta_model=meta_type)
        if not self._meta.load(meta_file):
            logging.info("Mega-ensemble meta-learner not found at %s -- "
                         "run 'mega' backtest first to train it", meta_file)
            return

        # ── Initialize all 35 base models ──────────────────────────
        start = time.time()

        # HMM
        hmm = None
        if _on("hmm"):
            n_states = mp.get("hmm_states", 2 if sport == "nfl" else 3)
            hmm = LeagueHMM(n_states=n_states, min_games=defaults["hmm_min_games"])
            self._models["hmm"] = hmm

        # Kalman
        kalman = None
        if _on("kalman"):
            kalman = LeagueKalman(
                process_noise=mp.get("kalman_process_noise", defaults["kalman_process_noise"]),
                measurement_noise=mp.get("kalman_measurement_noise", defaults["kalman_measurement_noise"]),
            )
            self._models["kalman"] = kalman

        # Network / PageRank
        network = None
        if _on("pagerank"):
            network = LeagueNetwork(
                decay=mp.get("network_decay", 0.97),
                margin_weighted=True,
            )
            self._models["pagerank"] = network

        # LightGBM, CatBoost
        lgbm = LightGBMPredictor(sport) if _on("lightgbm") else None
        catboost_m = CatBoostPredictor(sport) if _on("catboost") else None
        if lgbm:
            self._models["lightgbm"] = lgbm
        if catboost_m:
            self._models["catboost"] = catboost_m

        # MLP, LSTM
        mlp = MLPPredictor(sport) if (HAS_TORCH and _on("mlp")) else None
        lstm = LSTMPredictor(sport, seq_len=window) if (HAS_TORCH and _on("lstm")) else None
        if mlp:
            self._models["mlp"] = mlp
        if lstm:
            self._models["lstm"] = lstm

        # Tier 2
        volatility = LeagueVolatility(min_games=defaults["hmm_min_games"]) if _on("garch") else None
        signal_m = LeagueSignalAnalyzer(min_games=defaults["hmm_min_games"]) if _on("fourier") else None
        survival = LeagueSurvival(min_streaks=3) if _on("survival") else None
        copula = LeagueCopula(min_games=defaults["hmm_min_games"]) if _on("copula") else None
        if volatility:
            self._models["garch"] = volatility
        if signal_m:
            self._models["fourier"] = signal_m
        if survival:
            self._models["survival"] = survival
        if copula:
            self._models["copula"] = copula

        # Tier 3
        info_theory = LeagueInformationTheory(min_games=defaults["hmm_min_games"]) if _on("info_theory") else None
        momentum_m = LeagueMomentum(
            friction=mp.get("momentum_friction", 0.05),
            min_games=defaults["hmm_min_games"],
        ) if _on("momentum") else None
        markov = LeagueMarkovChain(sport=sport, min_games=defaults["hmm_min_games"]) if _on("markov") else None
        clustering = LeagueClustering(
            n_clusters=mp.get("n_clusters", 4),
            min_games=defaults["hmm_min_games"],
        ) if _on("clustering") else None
        game_theory = LeagueGameTheory(min_matchups=2, min_games=defaults["hmm_min_games"]) if _on("game_theory") else None
        if info_theory:
            self._models["info_theory"] = info_theory
        if momentum_m:
            self._models["momentum"] = momentum_m
        if markov:
            self._models["markov"] = markov
        if clustering:
            self._models["clustering"] = clustering
        if game_theory:
            self._models["game_theory"] = game_theory

        # Tier 4
        poisson = LeaguePoisson(min_games=defaults["min_train"] // 3) if _on("poisson") else None
        glicko = LeagueGlicko(initial_rd=mp.get("glicko_initial_rd", 200)) if _on("glicko") else None
        bradley_terry = BradleyTerryModel(
            decay=mp.get("bt_decay", 0.99),
            min_games=defaults["min_train"] // 2,
        ) if _on("bradley_terry") else None
        monte_carlo = LeagueMonteCarlo(
            n_simulations=mp.get("mc_simulations", 2000),
            min_games=defaults["hmm_min_games"],
        ) if _on("monte_carlo") else None
        random_forest = RandomForestPredictor(sport) if _on("random_forest") else None
        svm_m = SVMPredictor(sport) if _on("svm") else None
        fibonacci = LeagueFibonacci(min_games=defaults["hmm_min_games"]) if _on("fibonacci") else None
        evt = LeagueEVT(min_games=defaults["hmm_min_games"]) if _on("evt") else None
        benford = LeagueBenford(min_games=defaults["hmm_min_games"]) if _on("benford") else None
        if svm_m:
            self._models["svm"] = svm_m
        if fibonacci:
            self._models["fibonacci"] = fibonacci
        if evt:
            self._models["evt"] = evt
        if benford:
            self._models["benford"] = benford
        if poisson:
            self._models["poisson"] = poisson
        if glicko:
            self._models["glicko"] = glicko
        if bradley_terry:
            self._models["bradley_terry"] = bradley_terry
        if monte_carlo:
            self._models["monte_carlo"] = monte_carlo
        if random_forest:
            self._models["random_forest"] = random_forest

        # Tier 5: Classic models
        srs = colley = log5 = pythagenpat = exp_smoother = mean_revert = None
        if HAS_CLASSIC_MODELS:
            if _on("srs"):
                srs = SimpleRatingSystem()
                self._models["srs"] = srs
            if _on("colley"):
                colley = ColleyMatrix()
                self._models["colley"] = colley
            if _on("log5"):
                log5 = Log5()
                self._models["log5"] = log5
            if _on("pythagenpat"):
                pythagenpat = PythagenPat()
                self._models["pythagenpat"] = pythagenpat
            if _on("exp_smoothing"):
                exp_smoother = ExponentialSmoother()
                self._models["exp_smoothing"] = exp_smoother
            if _on("mean_reversion"):
                mean_revert = MeanReversionDetector()
                self._models["mean_reversion"] = mean_revert

        self._n_models = len(self._models)

        # ── Per-team rolling tracking ──────────────────────────────
        self._team_margins = defaultdict(list)
        self._team_scores_for = defaultdict(list)
        self._team_scores_against = defaultdict(list)
        self._team_results = defaultdict(list)

        # Internal Elo for feature extraction (separate from main model)
        self._elo_model = None
        if elo_model_class and elo_settings:
            init_keys = elo_model_class.__init__.__code__.co_varnames
            self._elo_model = elo_model_class(
                **{k: v for k, v in elo_settings.items() if k in init_keys}
            )
            if player_df is not None and not player_df.empty:
                self._elo_model.set_player_stats(player_df)

        # ── Replay game history ────────────────────────────────────
        if os.path.exists(csv_file):
            n_games = len(pd.read_csv(csv_file, usecols=["date"]))
            print("  Mega-ensemble: replaying %d games through %d models..."
                  % (n_games, self._n_models), flush=True)
            self._replay_games(csv_file, min_train, retrain_every)

        elapsed = time.time() - start
        self._available = True
        print("  Mega-ensemble ready (%d models, %.0fs)" % (self._n_models, elapsed),
              flush=True)
        logging.info("MegaPredictor ready: %d models, meta-learner=%s, "
                     "max_adj=%.2f (%.1fs)",
                     self._n_models, meta_type, self._max_adj, elapsed)

    def _replay_games(self, csv_file, min_train, retrain_every):
        """Replay all historical games through the 35 base models."""
        games = pd.read_csv(csv_file)
        if "neutral_site" not in games.columns:
            games["neutral_site"] = False
        games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")
        total_games = len(games)

        all_features = []
        all_labels = []
        last_season = None
        report_every = max(100, total_games // 10)

        for game_idx, row in games.iterrows():
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            home = row["home_team"]
            away = row["away_team"]
            h_score = float(row["home_score"])
            a_score = float(row["away_score"])
            neutral = bool(row.get("neutral_site", False))
            home_won = 1 if h_score > a_score else 0
            margin = h_score - a_score

            # Season boundary for Kalman regression
            kalman = self._models.get("kalman")
            if game_date is not None and kalman:
                current_season = game_date.year if game_date.month >= 7 else game_date.year - 1
                if last_season is not None and current_season != last_season:
                    kalman.regress_to_mean()
                last_season = current_season

            # Collect features (for ML model training)
            feature_row = self._collect_features(home, away, game_date, neutral)
            all_features.append(feature_row)
            all_labels.append(home_won)

            # Update all models with game result
            self._update_models(home, away, h_score, a_score, margin,
                                home_won, neutral, game_date, game_idx)

            # Progress reporting
            game_num = len(all_features)
            if game_num % report_every == 0:
                pct = game_num * 100 // total_games
                print("    %d/%d games (%d%%)" % (game_num, total_games, pct),
                      flush=True)

            # Periodically train ML models
            if game_num >= min_train and (game_num == min_train or
                                           (game_num - min_train) % retrain_every == 0):
                self._train_ml_models(all_features, all_labels)

    def _collect_features(self, home, away, game_date=None, neutral=False):
        """Collect features from all base models for a matchup."""
        feature_row = {}
        window = self._window

        # Elo
        if self._elo_model:
            elo_prob = self._elo_model.win_prob(
                home, away, team_a_home=True,
                neutral_site=neutral, calibrated=False,
                game_date=game_date
            )
            elo_diff = self._elo_model.ratings.get(home, 1500) - self._elo_model.ratings.get(away, 1500)
            feature_row["elo_prob"] = elo_prob
            feature_row["elo_diff"] = elo_diff
        else:
            feature_row["elo_prob"] = 0.5
            feature_row["elo_diff"] = 0.0

        # HMM
        hmm = self._models.get("hmm")
        if hmm:
            feature_row.update(hmm.get_features(home, away))

        # Kalman
        kalman = self._models.get("kalman")
        if kalman:
            feature_row.update(kalman.get_features(home, away))

        # PageRank
        network = self._models.get("pagerank")
        if network and network.n_games >= 20:
            network.compute_centralities()
            feature_row.update(network.get_features(home, away))

        # Volatility
        volatility = self._models.get("garch")
        if volatility:
            feature_row.update(volatility.get_features(home, away))

        # Signal
        signal_m = self._models.get("fourier")
        if signal_m:
            feature_row.update(signal_m.get_features(home, away))

        # Survival
        survival = self._models.get("survival")
        if survival:
            feature_row.update(survival.get_features(home, away))

        # Copula
        copula = self._models.get("copula")
        if copula:
            feature_row.update(copula.get_features(home, away))

        # Info theory
        info_theory = self._models.get("info_theory")
        if info_theory:
            feature_row.update(info_theory.get_features(home, away, window=window))

        # Momentum
        momentum_m = self._models.get("momentum")
        if momentum_m:
            feature_row.update(momentum_m.get_features(home, away))

        # Markov
        markov = self._models.get("markov")
        if markov:
            feature_row.update(markov.get_features(home, away))

        # Clustering
        clustering = self._models.get("clustering")
        if clustering:
            feature_row.update(clustering.get_features(home, away))

        # Game theory
        game_theory = self._models.get("game_theory")
        if game_theory:
            feature_row.update(game_theory.get_features(home, away))

        # Poisson
        poisson = self._models.get("poisson")
        if poisson and poisson._fitted:
            feature_row.update(poisson.get_features(home, away))

        # Glicko
        glicko = self._models.get("glicko")
        if glicko:
            feature_row.update(glicko.get_features(home, away))

        # Bradley-Terry
        bradley_terry = self._models.get("bradley_terry")
        if bradley_terry and bradley_terry._fitted:
            feature_row.update(bradley_terry.get_features(home, away))

        # Monte Carlo
        monte_carlo = self._models.get("monte_carlo")
        if monte_carlo:
            feature_row.update(monte_carlo.get_features(home, away))

        # Classic models
        for name in ("srs", "colley", "log5", "pythagenpat", "exp_smoothing", "mean_reversion"):
            model = self._models.get(name)
            if model:
                feature_row.update(model.get_features(home, away))

        # New models: Fibonacci, Benford, EVT
        fibonacci = self._models.get("fibonacci")
        if fibonacci:
            feature_row.update(fibonacci.get_features(home, away))
        benford = self._models.get("benford")
        if benford:
            feature_row.update(benford.get_features(home, away))
        evt = self._models.get("evt")
        if evt:
            feature_row.update(evt.get_features(home, away))

        # Rolling team features
        home_rolling = _rolling_features(
            self._team_margins[home], self._team_scores_for[home],
            self._team_scores_against[home], self._team_results[home], window
        )
        away_rolling = _rolling_features(
            self._team_margins[away], self._team_scores_for[away],
            self._team_scores_against[away], self._team_results[away], window
        )
        if home_rolling and away_rolling:
            feature_row["ppg_diff"] = home_rolling["ppg"] - away_rolling["ppg"]
            feature_row["papg_diff"] = home_rolling["papg"] - away_rolling["papg"]
            feature_row["win_pct_diff"] = home_rolling["win_pct"] - away_rolling["win_pct"]
            feature_row["margin_diff"] = home_rolling["avg_margin"] - away_rolling["avg_margin"]
            feature_row["consistency_diff"] = away_rolling["consistency"] - home_rolling["consistency"]
            feature_row["trend_diff"] = home_rolling["trend"] - away_rolling["trend"]
            feature_row["streak_diff"] = home_rolling["streak"] - away_rolling["streak"]

        return feature_row

    def _update_models(self, home, away, h_score, a_score, margin,
                       home_won, neutral, game_date, game_idx):
        """Update all base models with a game result."""
        # Elo
        if self._elo_model:
            self._elo_model.update_game(home, away, h_score, a_score,
                                         neutral_site=neutral, game_date=game_date)

        # HMM
        hmm = self._models.get("hmm")
        if hmm:
            hmm.add_game(home, margin, is_home=True)
            hmm.add_game(away, -margin, is_home=False)

        # Kalman
        kalman = self._models.get("kalman")
        if kalman:
            kalman.update_game(home, away, h_score, a_score)

        # Network
        network = self._models.get("pagerank")
        if network:
            if h_score > a_score:
                network.add_game(home, away, margin=max(1, abs(margin)), game_idx=game_idx)
            elif a_score > h_score:
                network.add_game(away, home, margin=max(1, abs(margin)), game_idx=game_idx)

        # Volatility
        volatility = self._models.get("garch")
        if volatility:
            volatility.add_game(home, margin)
            volatility.add_game(away, -margin)

        # Signal
        signal_m = self._models.get("fourier")
        if signal_m:
            signal_m.add_game(home, margin)
            signal_m.add_game(away, -margin)

        # Survival
        survival = self._models.get("survival")
        if survival:
            survival.add_game(home, h_score > a_score)
            survival.add_game(away, a_score > h_score)

        # Copula
        copula = self._models.get("copula")
        if copula:
            copula.add_game(home, h_score, a_score)
            copula.add_game(away, a_score, h_score)

        # Info theory
        info_theory = self._models.get("info_theory")
        if info_theory:
            info_theory.add_game(home, margin, h_score, a_score, h_score > a_score)
            info_theory.add_game(away, -margin, a_score, h_score, a_score > h_score)

        # Momentum
        momentum_m = self._models.get("momentum")
        if momentum_m:
            elo_r = self._elo_model.ratings.get(home, 1500) if self._elo_model else None
            momentum_m.add_game(home, margin, elo_rating=elo_r)
            elo_r_a = self._elo_model.ratings.get(away, 1500) if self._elo_model else None
            momentum_m.add_game(away, -margin, elo_rating=elo_r_a)

        # Markov
        markov = self._models.get("markov")
        if markov:
            markov.add_game(home, margin)
            markov.add_game(away, -margin)

        # Clustering
        clustering = self._models.get("clustering")
        if clustering:
            clustering.add_game(home, h_score, a_score, h_score > a_score)
            clustering.add_game(away, a_score, h_score, a_score > h_score)

        # Game theory
        game_theory = self._models.get("game_theory")
        if game_theory:
            game_theory.add_game(home, away, h_score, a_score, game_idx=game_idx)

        # Poisson
        poisson = self._models.get("poisson")
        if poisson:
            poisson.add_game(home, away, h_score, a_score)

        # Glicko
        glicko = self._models.get("glicko")
        if glicko:
            glicko.add_game(home, away, h_score, a_score)

        # Bradley-Terry
        bradley_terry = self._models.get("bradley_terry")
        if bradley_terry:
            bradley_terry.add_game(home, away, h_score > a_score, game_idx=game_idx)

        # Monte Carlo
        monte_carlo = self._models.get("monte_carlo")
        if monte_carlo:
            monte_carlo.add_game(home, h_score, a_score, is_home=True)
            monte_carlo.add_game(away, a_score, h_score, is_home=False)

        # Classic models
        for name in ("srs", "colley", "log5", "pythagenpat", "exp_smoothing", "mean_reversion"):
            model = self._models.get(name)
            if model:
                model.add_game(home, away, h_score, a_score)

        # New models
        fibonacci = self._models.get("fibonacci")
        if fibonacci:
            fibonacci.add_game(home, margin)
            fibonacci.add_game(away, -margin)
        benford_m = self._models.get("benford")
        if benford_m:
            benford_m.add_game(home, h_score, a_score, h_score > a_score)
            benford_m.add_game(away, a_score, h_score, a_score > h_score)
        evt = self._models.get("evt")
        if evt:
            evt.add_game(home, margin)
            evt.add_game(away, -margin)

        # Rolling team tracking
        self._team_margins[home].append(margin)
        self._team_margins[away].append(-margin)
        self._team_scores_for[home].append(h_score)
        self._team_scores_for[away].append(a_score)
        self._team_scores_against[home].append(a_score)
        self._team_scores_against[away].append(h_score)
        self._team_results[home].append(1.0 if h_score > a_score else 0.0)
        self._team_results[away].append(1.0 if a_score > h_score else 0.0)

    def _train_ml_models(self, all_features, all_labels):
        """Train ML models (LightGBM, CatBoost, MLP, RF) on accumulated features."""
        try:
            feat_names = sorted(set().union(*[f.keys() for f in all_features]))
            X = np.array([[f.get(fn, 0) for fn in feat_names] for f in all_features])
            y = np.array(all_labels, dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

            for name in ("lightgbm", "catboost", "mlp", "random_forest", "svm"):
                model = self._models.get(name)
                if model:
                    try:
                        model.train(X, y, feat_names)
                    except Exception as e:
                        logging.debug("ML model %s train failed: %s", name, e)
        except Exception as e:
            logging.debug("ML model training failed: %s", e)

    def predict(self, home, away, elo_prob, elo_diff, game_date=None):
        """Get bounded mega-ensemble adjustment for a prediction.

        Args:
            home: home team full name
            away: away team full name
            elo_prob: base Elo probability (after XGBoost blend if available)
            elo_diff: Elo rating difference (home - away)
            game_date: datetime of the game

        Returns:
            float: adjustment to add to elo_prob (clamped to +/- max_adj).
                   Returns 0.0 if meta-learner unavailable.
        """
        if not self._available or not self._meta._fitted:
            return 0.0

        try:
            # Collect features from all models
            features = self._collect_features(home, away, game_date)

            # Override elo features with the ACTUAL Elo values from main model
            # (our internal Elo may differ slightly)
            features["elo_prob"] = elo_prob
            features["elo_diff"] = elo_diff

            # Get meta-learner prediction
            meta_pred = self._meta.predict_single(features)

            # Bounded adjustment: meta deviates from 0.5, clamped to max_adj
            meta_adj = meta_pred - 0.5
            clamped_adj = float(np.clip(meta_adj, -self._max_adj, self._max_adj))

            return clamped_adj

        except Exception as e:
            logging.debug("MegaPredictor.predict failed: %s", e)
            return 0.0

    def get_model_count(self):
        """Return number of active base models."""
        return self._n_models

    def get_status(self):
        """Return status string for display."""
        if not self._available:
            return "Not available (run 'mega' to train)"
        return "%d models, max_adj=%.2f, meta=%s" % (
            self._n_models, self._max_adj,
            self._meta.meta_model_type if self._meta else "none"
        )
