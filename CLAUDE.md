# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MLB game prediction system combining Elo ratings with a 35-model mega-ensemble, with Predicts $1 contract trading ledger. Interactive CLI app -- no web server, no test framework, no build system.

## Running

```bash
pip install -r requirements.txt
python main.py
```

First run auto-downloads game data (MLB Stats API), player stats, pitching stats, and injury reports (ESPN). Then runs a backtest to fit the Platt calibration scaler and enters the interactive CLI loop.

## Architecture

**Three-stage prediction pipeline:**
1. **Elo model** (`elo_model.py` -> `MLBElo` class) -- base team ratings adjusted for home field, park factors, altitude, player strength, starting pitcher quality (K_PITCHER=6), rest days, travel fatigue, series adaptation, interleague factor, bullpen factor, opponent pitcher factor, injuries, and strength of schedule
2. **XGBoost ensemble** (`enhanced_model.py`) -- 80% Elo / 20% XGBoost (default `elo_weight=0.8`) using 96 rolling features per team (15-game window via `TeamTracker`, includes Pythagorean win expectation, streaks, consistency, and trend)
3. **Mega-ensemble** (`mega_predictor.py` + `mega_backtest.py`) -- 35 base models stacked via a meta-learner (XGBoost, ridge, or logistic). Produces a bounded adjustment (+/- max_adj, default 0.08) on top of the Elo+XGBoost probability. Models span 7 tiers: Core (Elo, XGBoost), Proven (HMM, Kalman, PageRank, LightGBM, CatBoost, MLP, LSTM), Exotic (GARCH, Fourier/Wavelet, Survival, Copula), Info/Physics (Shannon Entropy, Momentum, Markov Chain, Clustering, Game Theory), Classical Ratings (Poisson/Dixon-Coles, Glicko-2, Bradley-Terry, Monte Carlo, Random Forest), Sports-Specific (SRS, Colley Matrix, Log5, PythagenPat, Exponential Smoothing, Mean Reversion), Additional (SVM, Fibonacci, EVT, Benford), and Data Enrichment (Weather, Odds).
4. **Platt calibration** (`platt.py`) -- logistic regression on raw probabilities for well-calibrated outputs

**Data flow:**
- `data_games.py` / `data_players.py` -> download from MLB Stats API with 6-hour cache (`config.is_cache_stale`)
- `injuries.py` -> ESPN JSON API with 4-hour cache
- `weather.py` -> Open-Meteo API (free, no key) with 2-hour cache
- `odds_tracker.py` -> The Odds API (free tier, 500 req/month) for moneyline odds + CLV
- `kalshi.py` -> Kalshi public API for live contract prices
- `advanced_stats.py` -> Statcast/FanGraphs via pybaseball (xwOBA, barrel rate, xERA)
- `build_model.py` -> constructs `MLBElo` from settings + game CSV, applies season regression, sets player/pitcher scores
- `backtest.py` -> walk-forward backtest, also houses grid search, genetic (`scipy.optimize.differential_evolution`), and Bayesian (GP + EI) optimizers
- `mega_backtest.py` -> walk-forward backtest for all 35 models with meta-learner training
- `mega_optimizer.py` -> 7-phase per-model optimization (solo tuning, tournament, ablation, DE fine-tuning, validation)
- `mega_predictor.py` -> live predictions using all 35 models (replays history, loads trained meta-learner)
- `main.py` -> CLI entry point, `dispatch()` routes all commands, team name input triggers prediction flow

**State files (all gitignored, generated at runtime):**
- `mlb_elo_settings.json` -- tunable Elo parameters (39 params: K-factor, home advantage, pitcher factors, etc.)
- `mlb_elo_ratings.json` -- current team Elo ratings
- `mlb_platt_scaler.json` / `mlb_isotonic_scaler.json` / `mlb_beta_scaler.json` -- calibration scalers
- `mlb_enhanced_model.json` / `mlb_xgb_model.json` -- saved XGBoost model weights
- `mlb_mega_settings.json` -- mega-ensemble model switches, per-model hyperparameters, meta-learner config
- `mlb_meta_xgb.json` / `mlb_meta_learner.json` -- trained meta-learner weights
- `mlb_enhanced_features.npz` -- saved enhanced feature matrix for SHAP analysis
- `weather_cache.json` -- cached weather API responses (2-hour TTL)
- `predicts_lots.csv` -- trade ledger (positions, P&L)

**Key patterns:**
- All settings load/save through `config.py` (`load_elo_settings` / `save_elo_settings`)
- Color output uses wrapper functions in `color_helpers.py` (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), not raw colorama
- The `MLBElo` class stores per-team state in dicts/defaultdicts on the instance (ratings, last game dates, recent results, player scores, pitcher Elo, etc.)
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- this weighting is intentional
- `backtest_model(..., fit_platt=True)` should only be called for user-facing runs, never inside optimizer loops (leakage + speed)
- Mega-ensemble config managed through `mega_config.py` (model registry, switches, per-model hyperparams)
- Smart caching via `cache_utils.py` -- season-aware staleness checks (different refresh rates for games, players, odds, weather)

### Core Model Pipeline

1. **Data Download** (`data_games.py`, `data_players.py`)
   - Games: MLB Stats API (`statsapi.schedule()`) -> 2 years of completed games
   - Batting: Stats leaders (battingAverage, homeRuns, runsBattedIn) -> top 200
   - Pitching: Stats leaders (earnedRunAverage, strikeouts, wins) -> top 150
   - Cache: 6-hour staleness check on all CSV files (season-aware via `cache_utils.py`)

2. **Elo Model** (`elo_model.py`)
   - `MLBElo` class with 30 MLB teams
   - Base rating 1500, configurable K-factor (default 2.62 for 162-game season)
   - Home field advantage (default 37.63 Elo, ~55.3% implied home win rate)
   - Margin of victory adjustment (logarithmic, capped at mov_cap=19.9)
   - Player strength boost (z-scored team batting+pitching composite)
   - Starting pitcher quality (per-pitcher cumulative Elo ratings, 700+ tracked, K_PITCHER=6, 50% season regression)
   - Park factors (FanGraphs 5-year aggregate, Colorado 1.27 at top, Miami 0.93 at bottom)
   - Rest days, travel distance, form, SOS, run-environment (pace), altitude adjustments
   - Series adaptation, interleague factor, bullpen factor, opponent pitcher factor
   - East travel penalty, b2b penalty, road trip / homestand factors
   - Playoff detection (October) with reduced HCA factor (0.934)
   - Injury-aware predictions via ESPN API
   - K-decay and surprise-K for adaptive learning rate

3. **XGBoost Ensemble** (`enhanced_model.py`)
   - 96 rolling features per game (TeamTracker class, includes Pythagorean, streaks, consistency, trend)
   - Walk-forward training (no leakage): Elo-only for first 200 games
   - Default blend: 80% Elo / 20% XGBoost
   - Optional time-decay: transitions from 95% Elo early to 70% Elo late
   - SHAP feature importance via XGBoost native `pred_contribs`

4. **Mega-Ensemble** (`mega_backtest.py`, `mega_predictor.py`, `mega_config.py`, `mega_optimizer.py`, `meta_learner.py`)
   - 35 base models across 7 tiers (see MODEL_REGISTRY in `mega_config.py`)
   - Walk-forward backtest: first 300 games Elo-only, retrain every 80 games (MLB defaults)
   - Meta-learner stacking: XGBoost (default), ridge, or logistic regression
   - Bounded probability adjustment: meta prediction clamped to +/- max_adj (default 0.08)
   - Per-model on/off switches saved to `mlb_mega_settings.json`
   - Per-model hyperparameter tuning via 7-phase optimizer
   - Live predictions via `MegaPredictor` class (replays all history through 35 models at startup)

5. **Calibration** (`platt.py`)
   - Platt scaling: logistic regression on logit(raw_prob)
   - Isotonic regression: PAV algorithm (pure numpy, no sklearn)
   - Beta calibration: 3-parameter asymmetric (a, b, c)
   - Season regression: 33% pull toward mean at year boundaries

6. **Backtesting & Optimization** (`backtest.py`)
   - Walk-forward backtest with Platt fitting
   - Grid search, genetic (scipy DE), Bayesian (GP + EI) optimization
   - `autoopt`: automatic grid->genetic->bayesian pipeline
   - `superopt`: exhaustive 7-phase optimization (9 params, hours)
   - `singleopt`: coordinate descent (one param at a time, fine-grained)
   - Purged CV, CPCV, PBO, Monte Carlo permutation test
   - Kelly criterion, sliding window, convergence analysis
   - Conformal prediction, beta calibration comparison
   - Deflated Sharpe Ratio for multiple-testing adjustment

### Trading Ledger (`predict_ledger.py`)
- CSV-based lot tracking with entry/exit fees (2%)
- Add, sell (partial), resolve, mark, invert positions
- Monthly P&L chart generation
- Live score matching for open positions

### Live Features
- `live_scores.py`: MLB Stats API live scores, 60s refresh loop
- `auto_resolve.py`: Auto-settle finished games against open trades
- `injuries.py`: ESPN MLB injury API with Elo impact scoring
- `html_generator.py`: Blogger-ready HTML prediction tables

## File Map

### Core System
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, `dispatch()` routes all commands |
| `config.py` | Constants, 30 MLB teams, divisions, settings I/O |
| `elo_model.py` | `MLBElo` class (ratings, predictions, 24+ adjusters, park factors) |
| `build_model.py` | Model training with season regression |
| `data_games.py` | Game data download via MLB Stats API |
| `data_players.py` | Player stats download + team scoring |
| `backtest.py` | All backtesting & optimization (~2100 lines) |
| `enhanced_model.py` | XGBoost ensemble + SHAP + TeamTracker |
| `platt.py` | Calibration (Platt, isotonic, beta, regression) |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal |
| `elo_set_handler.py` | Shared handler for `set param=value` commands (39 Elo params) |
| `single_param_opt.py` | Coordinate descent optimizer (one param at a time) |
| `cache_utils.py` | Season-aware smart caching for all API data |

### Mega-Ensemble (35 models)
| File | Purpose |
|------|---------|
| `mega_predictor.py` | `MegaPredictor` class: live predictions using all 35 models |
| `mega_backtest.py` | Walk-forward backtest for full mega-ensemble + meta-learner training |
| `mega_optimizer.py` | 7-phase per-model optimizer (solo, tournament, ablation, DE, validation) |
| `mega_config.py` | Model registry, on/off switches, per-model hyperparams, `mega set` handler |
| `meta_learner.py` | `MetaLearner` class: XGBoost/ridge/logistic stacking combiner |

### Individual Base Models
| File | Purpose |
|------|---------|
| `hmm_model.py` | Hidden Markov Model (hot/cold team states) |
| `kalman_model.py` | Kalman filter (latent strength estimation) |
| `network_model.py` | PageRank + HITS (network/graph analysis) |
| `gbm_models.py` | LightGBM + CatBoost gradient boosting |
| `nn_models.py` | MLP + LSTM neural networks (requires PyTorch) |
| `random_forest_model.py` | Random Forest (bagging diversity) |
| `volatility_model.py` | GARCH volatility + Lyapunov exponent + Hurst exponent |
| `signal_model.py` | Fourier / wavelet (cycle detection) |
| `survival_model.py` | Survival analysis (streak hazard rates) |
| `copula_model.py` | Copula (offense/defense joint dependency) |
| `information_theory_model.py` | Shannon entropy + KL divergence |
| `momentum_model.py` | Newtonian momentum / inertia model |
| `markov_chain_model.py` | Markov chain (transition matrices) |
| `clustering_model.py` | k-Means team archetypes |
| `game_theory_model.py` | Nash equilibrium + style matchups |
| `poisson_model.py` | Poisson / Dixon-Coles (score distributions) |
| `glicko_model.py` | Glicko-2 (uncertainty-aware ratings) |
| `bradley_terry_model.py` | Bradley-Terry MLE (paired comparison) |
| `monte_carlo_model.py` | Monte Carlo simulation (2000 sims default) |
| `classic_models.py` | SRS, Colley Matrix, Log5, PythagenPat, Exponential Smoothing, Mean Reversion |
| `svm_model.py` | SVM classifier (RBF kernel + Platt scaling) |
| `fibonacci_model.py` | Fibonacci retracement (EMA-smoothed support/resistance levels) |
| `evt_model.py` | Extreme Value Theory (Generalized Pareto tail risk) |
| `benford_model.py` | Benford's Law (chi-squared scoring anomaly detection) |

### Data Enrichment
| File | Purpose |
|------|---------|
| `odds_tracker.py` | Live odds via The Odds API (free tier), CLV tracking |
| `weather.py` | Weather impact via Open-Meteo API (free, no key) |
| `kalshi.py` | Kalshi public API for live contract prices + auto-Kelly |
| `advanced_stats.py` | Statcast / FanGraphs via pybaseball (xwOBA, barrel rate, xERA) |
| `injuries.py` | ESPN injury report + Elo impact scoring |

### Trading & Display
| File | Purpose |
|------|---------|
| `predict_ledger.py` | Contract ledger management (add, sell, resolve, mark, invert) |
| `live_scores.py` | Live MLB scores via MLB Stats API, 60s refresh loop |
| `auto_resolve.py` | Auto-settle finished trades against live final scores |
| `html_generator.py` | Blogger-ready HTML prediction tables |
| `help_system.py` | Help text for all commands |
| `accuracy_test.py` | Quick walk-forward accuracy test |
| `color_helpers.py` | Colorama terminal formatting (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`) |

### Utility Scripts (standalone)
| File | Purpose |
|------|---------|
| `run_optimize.py` | Standalone optimization runner |
| `accuracy_optimize.py` | Accuracy-focused optimization |
| `run_enhanced_all.py` | Run enhanced backtest standalone |
| `quick_optimizer.py` | Quick parameter sweep |
| `sweep_enhanced.py` | Enhanced parameter sweep |

## Data Files (generated at runtime, all gitignored)

### API Data Caches
- `mlb_recent_games.csv` -- Game history (2 years, 6-hour cache)
- `mlb_player_stats.csv` -- Batting leaders (6-hour cache)
- `mlb_advanced_stats.csv` -- Pitching leaders (6-hour cache)
- `mlb_statcast_cache.csv` -- Statcast/FanGraphs advanced stats
- `weather_cache.json` -- Open-Meteo weather API responses (2-hour TTL)

### Model State
- `mlb_elo_ratings.json` -- Current team Elo ratings (30 teams)
- `mlb_elo_settings.json` -- Tuned Elo parameters (39 params)
- `mlb_platt_scaler.json` -- Platt calibration coefficients
- `mlb_isotonic_scaler.json` -- Isotonic calibration mappings
- `mlb_beta_scaler.json` -- Beta calibration (a, b, c params)
- `mlb_enhanced_model.json` -- XGBoost ensemble metadata
- `mlb_xgb_model.json` -- XGBoost model weights
- `mlb_enhanced_features.npz` -- Saved feature matrix for SHAP analysis

### Mega-Ensemble State
- `mlb_mega_settings.json` -- Model switches, per-model hyperparams, meta-learner config
- `mlb_meta_xgb.json` -- Trained meta-learner XGBoost weights
- `mlb_meta_learner.json` -- Meta-learner state (feature names, importance)

### Optimization Results
- `mlb_grid_search.csv` -- Grid search scored parameter combos
- `mlb_bayesian_results.csv` -- Bayesian optimization trial history
- `mlb_genetic_results.csv` -- Genetic optimizer results
- `mlb_optimization_log.txt` -- Optimization progress log
- `mlb_super_grid.csv` / `mlb_super_fine_grid.csv` -- Super-optimizer grids
- `mlb_backtest_predictions.csv` -- Per-game backtest predictions
- `mlb_calibration.csv` -- 10-bin calibration table

### Trading & Display
- `predicts_lots.csv` -- Trading ledger (positions, P&L)
- `today_mlb_predictions.html` -- Blogger-ready HTML predictions
- `today_mlb_predictions.txt` -- Plain-text predictions

## Key MLB-Specific Design Choices

### Elo Parameters (defaults from `config.load_elo_settings`)
- **K-factor = 2.62**: Optimized for 162-game season (higher than original 1.0 -- allows faster adaptation)
- **K_PITCHER = 6**: Per-pitcher Elo update speed (class constant on `MLBElo`), with 50% season regression
- **k_decay = 2.07**: Adaptive K reduction as game count increases
- **Home advantage = 37.63 Elo**: Optimized MLB home field advantage (~55.3% implied home win rate)
- **starter_boost = 88.08**: Starting pitcher quality boost (extremely impactful -- pitching dominates MLB)
- **player_boost = 19.53**: Team player strength boost from composite scoring
- **pace_factor = 41.87**: Run environment / pace adjustment (significant in MLB)
- **mean_reversion = 34.23**: Regression after extreme results (strong signal in baseball)
- **travel_factor = 13.71**: Cross-country travel fatigue
- **form_weight = 7.08**: Recent form (last 15 games) adjustment
- **division_factor = 6.64**: Divisional familiarity adjustment
- **rest_factor = 2.80**: Rest days impact (minimal in daily-game MLB)
- **sos_factor = 2.45**: Strength of schedule adjustment
- **b2b_penalty = 26.51**: Back-to-back/doubleheader fatigue penalty
- **altitude_factor = 12.48**: Coors Field altitude bonus (only Colorado)
- **season_phase_factor = 9.35**: Early/late season adjustment
- **playoff_hca_factor = 0.82**: Reduced home advantage in October playoffs
- **interleague_factor = 2.04**: Elo adjustment for AL vs NL matchups
- **series_adaptation = 3.92**: Adjustment for multi-game series familiarity
- **bullpen_factor = 6.43**: Bullpen quality impact on win probability
- **opp_pitcher_factor = 18.0**: Opponent starting pitcher quality adjustment
- **mov_cap = 19.9**: Maximum margin of victory cap
- **mov_base = 0.3**: Margin of victory logarithmic base
- **pyth_factor = 16.0**: Pythagorean expectation weighting
- **elo_scale = 400.0**: Elo logistic scale factor
- **season_regress = 0.33**: 33% pull toward mean at season boundaries

### Mega-Ensemble MLB Defaults (from `mega_backtest.SPORT_DEFAULTS`)
- **window = 15**: Rolling feature window (wider than NBA's 10 due to higher variance in baseball)
- **min_train = 300**: Games before meta-learner starts predicting
- **retrain_every = 80**: Retrain ML models every 80 games
- **kalman_process_noise = 0.3**: Kalman filter process noise
- **kalman_measurement_noise = 4.2**: Kalman filter measurement noise
- **hmm_min_games = 15**: Minimum games before HMM predictions
- **pyth_exp = 1.83**: Pythagorean exponent for run-based win estimation

### Other MLB-Specific Design
- **Season = calendar year**: MLB runs April-October within one year (unlike NBA cross-year)
- **Park factors**: 30 team-specific run-scoring multipliers (1.0 = average), sourced from FanGraphs 5-year aggregate
- **Player scoring**: Top 10 batters by composite (HR*2 + RBI + AVG*100), scored as HR*2 + RBI + R*0.5 + SB*0.5 + AVG*100. Batting 45% / pitching 55% (pitching-heavy, reflects MLB reality)
- **Pitching score**: Top 8 pitchers by composite ((4.50-ERA)*5 + K*0.5), scored as (4.50-ERA)*10 + K*0.5 + W*3. Scaled to match batting magnitude before blending
- **Division awareness**: `config.same_division()` and `config.get_league()` for division/interleague logic
- **Timezone tracking**: Per-team UTC offsets for travel fatigue calculations

## Backtesting & Optimization

### Walk-Forward Backtest (`backtest` command)

`backtest_model()` in `backtest.py` iterates through every game in `mlb_recent_games.csv` chronologically. For each game it:
1. Calls `model.win_prob()` with `calibrated=False` and `use_injuries=False` (raw Elo only -- no future leakage)
2. Records the prediction, then calls `model.update_game()` to update ratings *after* predicting
3. Outputs `mlb_backtest_predictions.csv` (per-game predictions) and `mlb_calibration.csv` (10-bin calibration table)

When `fit_platt=True` (user-facing runs only), it fits a Platt scaler on the full run's raw probabilities and saves to `mlb_platt_scaler.json`. This flag must **never** be used inside optimizer objective functions -- it causes data leakage and slows iteration.

Metrics reported: accuracy (%), log loss, Brier score. Calibration table bins predictions into 10 probability buckets and compares predicted vs actual win rates.

### XGBoost Enhanced Backtest (`enhanced` command)

`run_enhanced_backtest()` in `enhanced_model.py` is a separate walk-forward loop that builds an 80/20 Elo+XGBoost ensemble:
- First `min_train` games (default 200): Elo-only predictions while accumulating training features
- After that: XGBoost is trained on accumulated features and retrained every `retrain_every` games (default 50)
- `TeamTracker` maintains rolling 15-game windows (runs scored/allowed, win%, margins, rest days) per team
- Feature vector has 96 columns (`FEATURE_COLS` in `enhanced_model.py`): elo_prob, elo_diff, player_diff, per-team rolling stats, differentials, rest, Pythagorean win expectation, streaks, consistency, and trend
- After the walk-forward, fits both Platt and isotonic calibrators on ensemble probabilities
- Saves the trained XGBoost booster to `mlb_xgb_model.json` and metadata to `mlb_enhanced_model.json`

### Mega-Ensemble Backtest (`mega` command)

`run_mega_backtest()` in `mega_backtest.py` runs all 35 base models through a walk-forward loop:
- Initializes all enabled models (controlled by `mega_config.py` switches)
- First 300 games (MLB default): accumulates features from all models without meta-learner
- After min_train: trains the meta-learner (XGBoost/ridge/logistic) on accumulated features
- Retrains ML models (LightGBM, CatBoost, MLP, Random Forest) every 80 games
- Meta-learner produces bounded adjustment clamped to +/- max_adj (default 0.08)
- Saves trained meta-learner to `mlb_meta_xgb.json` / `mlb_meta_learner.json`

### Elo Optimization Commands

- **Grid search** (`grid`): Cartesian product over 7 Elo parameters with configurable ranges
- **Genetic** (`genetic`): `scipy.optimize.differential_evolution` over same 7 params
- **Bayesian** (`bayesian`): GP surrogate + Expected Improvement acquisition (~50-100 evaluations)
- **Auto-optimize** (`autoopt`): automatic grid->genetic->bayesian pipeline
- **Super-optimize** (`superopt`): exhaustive 7-phase optimization (9 params, takes hours)
- **Coordinate descent** (`singleopt`): one-param-at-a-time sweep with fine-grained refinement

All use the same objective: `score = -(LogLoss * 8 + Brier * 40)`. After completion, `_apply_best_settings()` saves the winning params and refits the Platt scaler.

### Mega-Ensemble Optimization (`mega optimize` command)

`mega_optimizer.py` runs a 7-phase per-model optimization:
- **Phase 0**: Elo-only baseline -- establish the floor
- **Phase 1**: Per-model solo optimization -- Elo + one model at a time, coordinate descent
- **Phase 2**: Head-to-head tournament -- pairs, triples, top-N combos
- **Phase 3**: Meta-learner + global tuning -- max_adj, meta_model, retrain_every, min_train, window
- **Phase 4**: Combined DE fine-tuning -- differential evolution over top numeric params
- **Phase 5**: Final ablation -- test each model's contribution with fully tuned params
- **Phase 6**: Validation -- run best config 5 times, report stability

### Calibration Methods (`platt.py`)

Three calibrators are available, all implemented without sklearn:
- **Platt scaling**: logistic regression on `logit(raw_prob)` via `scipy.optimize.minimize` (L-BFGS-B). Corrects overconfidence at 0.9+ and underconfidence at 0.1-0.2.
- **Isotonic regression**: Pool Adjacent Violators algorithm on binned probabilities (50 bins default), with linear interpolation for continuous mapping.
- **Beta calibration**: 3-parameter `logit(p_cal) = c + a*log(p) - b*log(1-p)`. If a != b, miscalibration is asymmetric and beta is better than Platt.

Season regression (`regress_ratings_to_mean`): pulls all team ratings 33% toward the league mean at season boundaries, applied during model building.

## Advanced Backtesting Methods

All 16 advanced backtesting techniques are implemented across `backtest.py`, `enhanced_model.py`, `platt.py`, and `metrics.py`. No additional dependencies beyond the existing stack (numpy, scipy, pandas, xgboost).

### Metrics (auto-reported by `backtest` command)

- **ECE / MCE** (`metrics.py: ece_score, mce_score`) -- Expected and Maximum Calibration Error. ECE < 0.03 is excellent; > 0.08 needs work.
- **BSS** (`metrics.py: brier_skill_score`) -- Brier Skill Score vs 50% baseline and vs home-win-rate baseline. BSS > 0 means model beats the reference.
- **DSR** (`backtest.py: deflated_sharpe_ratio`) -- Deflated Sharpe Ratio. Adjusts optimization "best" for multiple-testing bias. DSR < 1.96 means best is not significant at 95%.

### Cross-Validation (`purgedcv`, `cpcv` commands)

- **Purged Walk-Forward CV** (`purgedcv`) -- k chronological folds with embargo gap. Reports mean/std across folds. Accuracy variance >3% = fragile model.
- **CPCV** (`cpcv`) -- All C(k, k_test) train/test combinations. With k=5, k_test=2: 10 paths.

### Overfitting Detection (`pbo`, `montecarlo` commands)

- **PBO** (`pbo`) -- Symmetric cross-validation on grid search trial population. PBO > 0.5 = optimization is likely overfit.
- **Monte Carlo Permutation** (`montecarlo`) -- Shuffles scores randomly, re-runs backtest N times (default 500). p > 0.05 = model not statistically significant.

### Calibration (`rollingcal`, `betacal`, `conformal` commands)

- **Rolling Origin Recalibration** (`rollingcal`) -- Expanding-window Platt fitting every 50 games.
- **Beta Calibration** (`betacal`) -- 3-parameter asymmetric calibration.
- **Conformal Prediction** (`conformal`) -- Distribution-free prediction sets at 90%, 95%, 80% coverage.

### Ensemble Analysis (`enhanced decay`, `shap` commands)

- **Time-Decayed Weighting** (`enhanced decay`) -- Elo weight transitions 95% -> 70% over the season.
- **SHAP Feature Importance** (`shap`) -- XGBoost native `pred_contribs`.

### Structural Analysis (`sliding`, `convergence` commands)

- **Sliding Window** (`sliding`) -- Heavy regression (67%) every N games vs standard expanding window.
- **Convergence Analysis** (`convergence`) -- Chunks predictions by N-game windows, finds burn-in period.

### Position Sizing (`kelly` command)

**Kelly Criterion Backtest** -- Simulates fractional Kelly position sizing. Default quarter-Kelly. Reports final bankroll, return %, max drawdown, annualized Sharpe.

## CLI Commands

Enter a team name to start a prediction. Core commands:

**Data & Display**: `all`, `refresh`, `players`, `settings`, `injuries`, `today`/`html`, `tomorrow`

**Data Enrichment**: `odds`, `kalshi`, `weather`, `advstats`/`statcast`

**Backtesting**: `backtest`, `enhanced`, `enhanced decay`

**Elo Optimization**: `grid`, `genetic`, `bayesian`, `autoopt`, `superopt`, `singleopt`, `results`

**Validation**: `purgedcv`, `cpcv`, `pbo`, `montecarlo`, `convergence`, `sliding`

**Calibration**: `rollingcal`, `betacal`, `conformal`

**Analysis**: `shap`, `kelly`

**Mega-Ensemble (35 models)**:
- `mega` -- Run mega-ensemble backtest (all enabled models)
- `mega optimize` -- 7-phase per-model optimization (all phases, takes hours)
- `mega quick` -- Quick mega optimization (Phases 0-1 only)
- `mega tune` -- Per-model solo optimization (same as mega quick)
- `mega tournament` -- Head-to-head model tournament (Phase 2)
- `mega ablation` -- Test each model's individual contribution, auto-prune weak ones
- `mega models` -- Show all 35 models with ON/OFF status by tier
- `mega on <model>` / `mega off <model>` -- Enable/disable individual models (or `mega on all`)
- `mega settings` -- Show all mega parameter values
- `mega set <param>=<value>` -- Set mega parameter (e.g., `mega set max_adj=0.10`, `mega set meta=ridge`, `mega set mc_sims=3000`)

**Trading**: `predicts`, `balance`, `resolve`, `sell`, `mark`, `invert`, `chart`, `live`, `autoresolve`, `autoresolve on/off`

**Elo Settings** (39 params, type `set` to see all):
- `set k=2.62`, `set home=38`, `set starter=88`, `set rest=3`, `set b2b=27`
- `set travel=14`, `set pace=42`, `set parkfactor=0`, `set interleague=2`
- `set bullpen=6`, `set opp_pitcher=18`, `set series=4`, `set east_travel=0`
- `set kelly=quarter`, `set balance=1000`, `set autoresolve=true`

`help` for overview, `help <command>` for details, `help advanced` for all validation commands, `quit` to exit.

## Comprehensive Workflow

### Daily Prediction Workflow

1. **Launch**: `python main.py` -- auto-downloads fresh data (games, players, pitching, injuries), builds Elo model, runs baseline backtest, fits Platt scaler
2. **Check model**: `settings` to verify parameters, `all` to see team rankings, `injuries` to review injury impact
3. **Make predictions**: type a team name (e.g. `Yankees`) -> enter opponent -> specify home/away (`a`/`b`/`n`) -> see calibrated win probability with injury impact and key player stats
4. **Log position**: type `y` -> enter contract count, price, optional notes -> position saved to `predicts_lots.csv`
5. **Publish**: `today` or `tomorrow` generates Blogger HTML + plain text for all scheduled games
6. **Monitor**: `live` shows real-time scores for open positions; `mark` updates current market prices
7. **Settle**: `resolve` to manually settle (win/loss prompt), `autoresolve` to auto-settle from live final scores, `sell` to exit early at a price
8. **Review**: `predicts` shows full P&L ledger with realized/unrealized/marked totals, win rate, ROI; `chart` saves monthly P&L bar chart

### Model Tuning Workflow

1. **Baseline**: run `backtest` -- note accuracy, log loss, Brier, ECE, MCE, BSS (all auto-reported)
2. **Diagnostics**: run `convergence` to find burn-in period, `sliding` to check if old data helps
3. **Coarse search**: run `grid` with wide ranges and large steps to identify promising parameter regions
4. **Validate search**: run `pbo` to check overfitting, `results` to check DSR significance
5. **Fine-tune**: run `genetic` or `bayesian` with tighter bounds around the grid search sweet spot
6. **Cross-validate**: run `purgedcv` for fold stability, `cpcv` for combinatorial robustness
7. **Significance test**: run `montecarlo` -- if p > 0.05, model has no proven edge
8. **Train ensemble**: run `enhanced` (SHAP auto-runs), then `enhanced decay` to compare
9. **Calibration**: run `rollingcal` for OOS calibration, `betacal` for asymmetry, `conformal` for coverage
10. **P&L simulation**: run `kelly` to connect model quality to bankroll trajectory
11. **Mega-ensemble**: run `mega` to train 35-model stack, then `mega optimize` for full tuning
12. **Iterate**: adjust individual params with `set k=2.62`, `set home=38`, etc. -- always rerun `backtest` after to refit the Platt scaler

### Position Management

Predicts contracts are $1 binary options. The ledger tracks:
- **Entry**: contracts count, price per contract, 2% entry fee on potential payout ($1 per contract)
- **Mark-to-market**: `mark` updates current price; `predicts` shows unrealized P&L based on marks
- **Exit**: `sell` for partial/full exit at a price (with 2% exit fee); `resolve` for binary win/loss settlement
- **Invert**: `invert` flips a position's direction (swaps predicted winner, inverts probability) without changing cost basis
- **Auto-resolve**: matches open positions against today's final scores via MLB Stats API; toggle with `autoresolve on/off` (runs on startup when enabled)

### Data Refresh

- `refresh` deletes cached game/player/ratings/scaler files and re-downloads everything, then rebuilds the model
- Data caches expire after 6 hours for games/players, 4 hours for injuries, 2 hours for weather, 60 seconds for live scores
- Cache staleness checked by `config.is_cache_stale()` with sport-aware logic from `cache_utils.py`
- Files under 500 bytes are always considered stale (empty/corrupt stubs)

## Conventions

- Python 3.8+ compatibility (no walrus operators, no `match` statements)
- No test framework -- validation is via `backtest` command and `accuracy_test.py` (run manually: `python accuracy_test.py [label]`)
- All data files use `mlb_` prefix; temp files use `temp_` prefix (both gitignored)
- Settings are tuned via `set param=value` CLI command or optimizer; changes require `backtest` rerun to refit Platt scaler
- Color output uses `color_helpers.py` wrappers (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), never raw colorama
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- this weighting is intentional and should not be changed casually
- Team lookup is fuzzy: `MLBElo.find_team()` accepts full names, abbreviations, partial matches, and close matches via `difflib.get_close_matches`
- Mega-ensemble model switches are per-sport and saved to `{sport}_mega_settings.json`
- All data sources must be completely free (no paid APIs)

## Dependencies
- pandas, numpy, scipy, colorama, tqdm, xgboost, requests, MLB-StatsAPI, matplotlib
- Optional: torch (for MLP/LSTM), pybaseball (for Statcast/FanGraphs)
