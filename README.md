# MLB Moneyball -- 31-Model Mega-Ensemble

A production-grade MLB game prediction system that fuses 31 independent models -- spanning Elo ratings, gradient boosting, Hidden Markov Models, Kalman filters, PageRank, neural networks, survival analysis, information theory, game theory, and classical baseball sabermetrics -- into a single calibrated probability through a walk-forward meta-learner. Every model trains on real MLB data pulled from completely free APIs (MLB Stats API, ESPN injuries, Open-Meteo weather). The system includes a full Predicts $1 binary contract trading ledger with Kelly criterion position sizing, live score tracking, auto-settlement, and monthly P&L charting. All 162-game-season parameters are tuned through a 7-phase exhaustive optimizer with multithreaded backtesting and optional GPU acceleration.

**30 MLB teams** | **31 models** | **86+ tunable parameters** (32 Elo + 54 per-model) | **7-phase per-model optimizer** | **No paid APIs**

---

## Quick Start

```bash
# 1. Clone and enter directory
cd MLBClaude

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install PyTorch for neural network models (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Launch
python main.py
```

**First-time workflow:**

```
1. System auto-downloads 2 years of MLB game data via MLB Stats API
2. System auto-downloads batting leaders, pitching leaders, injury reports
3. Baseline backtest runs automatically (fits Platt calibration scaler)
4. Enter starting balance when prompted (for contract tracking)
5. Type a team name (e.g. "Yankees") to make your first prediction
6. Run 'mega' for the full 31-model ensemble backtest
7. Run 'mega tune' to solo-test each model's optimal settings
8. Run 'mega optimize' for full 7-phase per-model optimization
```

On startup, the system downloads and caches all required data, builds the Elo model with season regression, runs a baseline backtest with Platt calibration, and drops you into the interactive command loop. No API keys are needed for core functionality -- the MLB Stats API, ESPN injuries, and Open-Meteo weather are all completely free.

---

## The 31 Models

Every model runs independently on the same game-by-game walk-forward loop. Their raw outputs feed into the meta-learner, which produces a single calibrated adjustment bounded by `max_adj`.

### Tier 0 -- Core (Always On)

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 1 | **Elo** | 1960 | Paired comparison rating | 24+ adjusters: home advantage, MOV, pitcher Elo, rest, travel, altitude, park factors, form, SOS, interleague, playoff detection. 30 teams, 700+ pitchers tracked individually. K=1.0 for 162-game season with logarithmic MOV. |
| 2 | **XGBoost** | 2016 | Gradient boosted trees | 31 rolling features per game (win%, Pythagorean, streaks, consistency, scoring trend, rest, travel). Walk-forward training with 80/20 Elo/XGBoost blend. SHAP feature importance built in. |

### Tier 1 -- Proven Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 3 | **HMM** | 1966 | Hidden Markov Model | Detects latent hot/cold team states from win/loss sequences. Forward-backward algorithm estimates state probabilities. Captures momentum shifts invisible to pure ratings. |
| 4 | **Kalman** | 1960 | Kalman Filter | Treats true team strength as a hidden state with process noise. Bayesian updates after each game. Provides uncertainty estimates alongside point predictions. |
| 5 | **PageRank** | 1998 | Network analysis | Builds directed win graph, runs PageRank + HITS authority scores. Teams that beat strong teams get more credit. Temporal decay weights recent results. |
| 6 | **LightGBM** | 2017 | Leaf-wise gradient boosting | Microsoft's fast GBM with leaf-wise splits. Handles categorical features natively. Lower memory than XGBoost with comparable accuracy. |
| 7 | **CatBoost** | 2017 | Ordered gradient boosting | Yandex's ordered boosting prevents target leakage during training. Handles categorical features with target statistics. Robust to overfitting. |
| 8 | **MLP** | 1986 | Multi-layer perceptron | PyTorch feedforward neural network with batch normalization and dropout. Learns nonlinear feature interactions that tree models miss. |
| 9 | **LSTM** | 1997 | Long Short-Term Memory | Recurrent neural network that models sequential game patterns. Captures long-range dependencies in team performance trajectories. |

### Tier 2 -- Exotic / Physics-Inspired

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 10 | **GARCH** | 1986 | Volatility modeling | Generalized AutoRegressive Conditional Heteroskedasticity. Models time-varying volatility in run scoring. High-variance teams are harder to predict. |
| 11 | **Fourier** | 1822 | Cycle detection | Fourier transforms + wavelet analysis on scoring time series. Detects periodic patterns (weekly, monthly) and seasonal rhythms in team performance. |
| 12 | **Survival** | 1958 | Hazard modeling | Cox proportional hazards applied to win/loss streaks. Models the probability that a streak ends given its length and covariates. |
| 13 | **Copula** | 1959 | Joint dependency | Models the dependency structure between offensive and defensive run production using copula functions. Captures teams where offense/pitching move together vs independently. |

### Tier 3 -- Information & Physics

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 14 | **Info Theory** | 1948 | Shannon entropy + KL divergence | Measures predictability of each team's run-scoring distribution. High-entropy teams are chaotic; low-entropy teams are predictable. KL divergence quantifies matchup asymmetry. |
| 15 | **Momentum** | 1687 | Newtonian mechanics analogy | Treats team strength as a physical object with mass (games played) and velocity (recent trend). Friction coefficient controls decay. Captures inertia in form. |
| 16 | **Markov Chain** | 1906 | Transition matrices | Models sequences of outcomes (W/L/close-W/blowout-W) as Markov transitions. Stationary distribution gives long-run expected state probabilities. |
| 17 | **Clustering** | 1957 | k-Means archetypes | Groups teams into archetypes (e.g., high-offense/low-pitching, balanced, pitching-dominant). Matchup predictions based on how archetype pairs historically perform. |
| 18 | **Game Theory** | 1950 | Nash equilibrium | Models strategic matchups: power vs finesse, contact vs strikeout-heavy. Computes Nash equilibrium strategies and style-based advantages. |

### Tier 4 -- Classical Rating Systems

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 19 | **Poisson** | 1898 | Dixon-Coles run distribution | Models run scoring as Poisson-distributed. Dixon-Coles correction for low-scoring games. Produces full run probability matrix for each matchup. |
| 20 | **Glicko-2** | 2001 | Uncertainty-aware ratings | Extends Elo with rating deviation (confidence interval) and volatility. Teams with fewer recent games have wider uncertainty. More principled than fixed-K Elo. |
| 21 | **Bradley-Terry** | 1952 | Maximum likelihood paired comparison | MLE estimation of team strengths from pairwise outcomes. Recency-weighted decay ensures recent games matter more. Clean probabilistic framework. |
| 22 | **Monte Carlo** | 1940s | Stochastic simulation | Runs 2,000+ game simulations per matchup using historical run-scoring distributions. Produces win probability from simulation outcomes. |
| 23 | **Random Forest** | 2001 | Bagged decision trees | Ensemble of decorrelated decision trees. Provides diversity to the meta-learner -- different inductive bias from boosted methods. |

### Tier 5 -- Classical Baseball / Sabermetric Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 24 | **SRS** | ~1980s | Simple Rating System | Average run margin adjusted for strength of schedule. Iterative convergence. The backbone of many newspaper power rankings. |
| 25 | **Colley** | 2001 | Colley Matrix | Bias-free ranking using only wins and losses. Solves a linear system -- no preseason assumptions, no margin of victory. Used by the BCS. |
| 26 | **Log5** | 1981 | Bill James formula | The original sabermetric head-to-head formula: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB). Elegant and theoretically grounded. |
| 27 | **PythagenPat** | 2005 | Dynamic Pythagorean exponent | Extends the Pythagorean expected win% formula with a dynamic exponent based on the run environment (RPG). Better than fixed-exponent Pythagorean for MLB. |
| 28 | **Exp Smoothing** | 1957 | Exponential smoothing | Holt-Winters style smoothing on team performance metrics. Captures level, trend, and seasonality in run scoring. Simple but effective trend tracker. |
| 29 | **Mean Reversion** | ~1990s | Bollinger band analog | Identifies teams performing above/below their "true" level using a z-score band approach. Teams far from the mean are expected to regress. |

### Tier 6 -- Data Enrichment

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 30 | **Weather** | -- | Environmental impact | Temperature, wind speed, humidity, precipitation probability from Open-Meteo API (free, no key). Adjusts predictions for extreme weather at outdoor ballparks. |
| 31 | **Odds** | -- | Market consensus | Ingests moneyline odds from The Odds API. Closing Line Value (CLV) tracking. Markets are efficient -- odds provide a strong independent signal. Off by default (requires free API key). |

---

## Architecture

```
                           MLB STATS API          ESPN INJURIES API
                          (games, scores)         (IL, DTD, Out)
                               |                         |
                     OPEN-METEO WEATHER            PYBASEBALL/STATCAST
                     (temp, wind, precip)          (xwOBA, xERA, barrel%)
                               |                         |
                     +---------+---------+---------------+
                     |                                   |
                     v                                   v
              +------+-------+                  +--------+--------+
              |  DATA LAYER  |                  | ADVANCED STATS  |
              | data_games   |                  | advanced_stats  |
              | data_players |                  | (Statcast/FG)   |
              +--------------+                  +-----------------+
                     |                                   |
                     +-----------------------------------+
                     |
                     v
    +=====================================+
    |         ELO ENGINE (Tier 0)         |
    |  MLBElo class -- 30 teams           |
    |  700+ pitcher sub-ratings           |
    |  24 adjustment factors              |
    |  30 ballpark factor table           |
    |  Season regression (33%)            |
    |  Platt/Isotonic/Beta calibration    |
    +=====================================+
                     |
                     | Elo probability (anchor)
                     |
    +=====================================+
    |    31 BASE MODEL PREDICTIONS        |
    |                                     |
    |  [Tier 0] Elo, XGBoost             |
    |  [Tier 1] HMM, Kalman, PageRank,   |
    |           LightGBM, CatBoost,       |
    |           MLP, LSTM                 |
    |  [Tier 2] GARCH, Fourier,          |
    |           Survival, Copula          |
    |  [Tier 3] InfoTheory, Momentum,    |
    |           Markov, Clustering,       |
    |           GameTheory                |
    |  [Tier 4] Poisson, Glicko, B-T,    |
    |           MonteCarlo, RandomForest  |
    |  [Tier 5] SRS, Colley, Log5,       |
    |           PythagenPat, ExpSmooth,   |
    |           MeanReversion             |
    |  [Tier 6] Weather, Odds            |
    |                                     |
    |  All models run in PARALLEL via     |
    |  ThreadPoolExecutor                 |
    +=====================================+
                     |
                     | Vector of 31 probabilities
                     v
    +=====================================+
    |       META-LEARNER (Stacker)        |
    |                                     |
    |  Ridge / Logistic / XGBoost         |
    |  Walk-forward retrain every N games |
    |  min_train warmup period            |
    |  Trains on base model outputs only  |
    +=====================================+
                     |
                     | Raw adjustment delta
                     v
    +=====================================+
    |   ELO-ANCHORED BOUNDED ADJUSTMENT   |
    |                                     |
    |  final = elo_prob + clamp(          |
    |    meta_adjustment, -max_adj,       |
    |    +max_adj)                        |
    |                                     |
    |  Elo is ALWAYS the anchor.          |
    |  Meta-learner can only nudge the    |
    |  probability within +/- max_adj     |
    |  (default 0.10 = 10 percentage      |
    |   points).                          |
    +=====================================+
                     |
                     v
           FINAL CALIBRATED PROBABILITY
                     |
                     v
         +---------------------+
         |  PREDICTION OUTPUT  |
         |  + Trading Ledger   |
         |  + HTML Export      |
         |  + Live Scores      |
         +---------------------+
```

### Elo-Anchored Bounded Adjustment

The Elo model serves as the anchor probability. The meta-learner (trained on all 31 base model outputs) produces an adjustment that is **clamped** to `+/- max_adj` (default 0.10). This means even if all exotic models disagree with Elo, the final probability can shift at most 10 percentage points. This design prevents catastrophic predictions from untested models while allowing proven signal to improve accuracy.

### Multithreaded Training

All 31 base models run inside a `ThreadPoolExecutor`. On a typical 8-core machine, the mega-ensemble backtest completes 3-5x faster than sequential execution. Each model receives the same game-by-game data and produces an independent probability estimate.

### GPU Acceleration

XGBoost, LightGBM, and CatBoost automatically detect CUDA-capable GPUs. If available, tree construction runs on GPU (`tree_method='gpu_hist'` for XGBoost, `device='gpu'` for LightGBM/CatBoost). PyTorch models (MLP, LSTM) also move to GPU when `torch.cuda.is_available()`. CPU fallback is always automatic and silent.

---

## MLB-Specific Design Choices

Every parameter in this system was chosen with the specific structure of Major League Baseball in mind. Here is why each default is what it is:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| **K-factor** | 1.0 | MLB plays 162 games per season -- far more than NFL (17) or NBA (82). A lower K means each individual game moves ratings less, preventing wild swings from single-game randomness. Baseball has the highest game-to-game variance of the four major sports. |
| **Home advantage** | 23.47 Elo (~54%) | MLB home teams historically win about 54% of games. 23.47 Elo points in the standard Elo formula yields approximately 53.5% expected win rate, matching observed data. This is lower than NBA (~60%) because baseball home advantage is more subtle (last at-bat, familiar park). |
| **Player scoring weight** | 45% batting / 55% pitching | Pitching dominates baseball outcomes more than hitting. A dominant starter can single-handedly suppress a lineup. The 55/45 split reflects the asymmetry where pitching controls a game's tempo and ceiling. |
| **Starting pitcher tracking** | Per-pitcher Elo (K=6, 50% regression) | MLB is unique: the starting pitcher identity changes every game and has massive impact. The system tracks 700+ individual pitcher Elo ratings with K_PITCHER=6 (higher than team K because pitcher sample sizes are smaller). 50% season regression prevents staleness. |
| **Rolling window** | 15 games | Wider than NBA's 10-game window because baseball has higher game-to-game variance. A 15-game window smooths out noise while still capturing meaningful form changes over ~2 weeks of play. |
| **Altitude factor** | Colorado-only | Only the Colorado Rockies play at significant altitude (Coors Field, 5,280 ft). The thin air increases home run rates and run scoring dramatically. No other MLB park has meaningful altitude effects. |
| **Park factor weight** | Configurable | Each MLB stadium has a unique run environment. Coors Field inflates scoring by 20-30%, while Oracle Park suppresses it. Park factors from FanGraphs 5-year aggregates adjust expected scoring for each venue. All 30 ballparks have individual multipliers (0.93 to 1.27). |
| **Interleague factor** | 2.04 | AL vs NL teams have different roster construction philosophies. Interleague games introduce unfamiliarity. The interleague factor accounts for this systematic difference. |
| **Bullpen factor** | 6.43 | Baseball games are won or lost in the bullpen. Cumulative bullpen Elo tracks reliever quality per team. The bullpen factor weights this signal into the prediction. |
| **Opponent pitcher factor** | 18.0 | The opposing starting pitcher has enormous impact on any given game. This factor weights the opponent's pitcher Elo into the prediction, allowing the system to distinguish between facing an ace versus a fifth starter. |
| **Playoff HCA factor** | 0.934 | October games have slightly reduced home advantage compared to regular season. The 0.934 multiplier reduces the 23.47-point HCA to ~22 points in playoffs. Genetic optimization confirmed the reduction is modest. |
| **Season regression** | 33% | At the start of each new season, all ratings regress 33% toward 1500. This accounts for roster turnover, free agency, and the reality that last year's team is not this year's team. |
| **MOV formula** | log(max(1.0, abs(rd))+1.0), capped at 19.9 | Run margins in baseball follow a roughly logarithmic value curve -- the difference between a 1-run win and a 2-run win is much more informative than between an 8-run win and a 9-run win. The cap prevents blowouts from having outsized influence. |
| **Season calendar** | April-October (single year) | Unlike NBA/NHL which cross calendar year boundaries, the MLB season runs entirely within one calendar year. This simplifies season detection and regression timing. |
| **B2B penalty** | 26.51 | Doubleheaders and consecutive-day games cause fatigue, especially for bullpens. The back-to-back penalty reflects reduced pitching depth after heavy workload. |
| **Series adaptation** | 3.92 | In multi-game series (standard in MLB), the visiting team adapts to the home park. Game 2 and 3 of a series see reduced home advantage as hitters adjust to the mound, lighting, and environment. |
| **PythagenPat exponent** | Dynamic (pyth_factor=16.0) | Bill James' Pythagorean theorem for baseball uses an exponent based on the run environment. PythagenPat computes the exponent dynamically from runs per game rather than using a fixed value, improving accuracy across different scoring eras. |
| **Batting composite** | HR*2 + RBI + AVG*100 | Top 10 batters per team are scored by this composite, then the team score blends batting (45%) and pitching (55%). Pitching composite: (4.50-ERA)*5 + K*0.5 for ranking, (4.50-ERA)*10 + K*0.5 + W*3 for scoring. |

---

## Complete Command Reference

### Predictions & Data

| Command | Description | Time |
|---------|-------------|------|
| `<team name>` | Start prediction for any team (fuzzy match: `Yankees`, `lad`, `BOS`, `mets`) | ~2s |
| `today` / `html` / `blogger` | Generate HTML prediction table for today's games | ~5s |
| `tomorrow` | Generate HTML prediction table for tomorrow's games | ~5s |
| `all` | Show current Elo ratings for all 30 teams, sorted by rating | ~1s |
| `players` | Show top batters by composite score (league-wide or per team) | ~1s |
| `injuries` | Show current MLB injury report from ESPN with Elo impact | ~3s |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT for a team | instant |
| `refresh` | Force redownload of all game + player + injury data | ~30s |
| `odds` | Show today's moneyline odds from The Odds API | ~3s |
| `kalshi` | Show Kalshi prediction market odds | ~3s |
| `weather` | Show weather forecast for a home team's stadium | ~2s |
| `advstats` / `statcast` | Show Statcast/FanGraphs team rankings (xwOBA, xERA, barrel rate) | ~15s |

### Backtesting

| Command | Description | Time |
|---------|-------------|------|
| `backtest` | Walk-forward backtest + fit Platt calibration scaler | ~15s |
| `enhanced` | XGBoost ensemble backtest (80/20 Elo/XGB blend, 31 features) | ~30s |
| `enhanced decay` | Time-decayed ensemble (95% Elo early -> 70% Elo late season) | ~30s |
| `shap` | SHAP feature importance analysis for XGBoost features | ~10s |
| `sliding` | Sliding vs expanding window comparison | ~30s |
| `convergence` | Elo rating convergence / burn-in analysis | ~10s |
| `platt` / `calibrate` | Show Platt calibration scaler status and coefficients | instant |

### Elo Optimization

| Command | Description | Time |
|---------|-------------|------|
| `grid` | Grid search over K, HomeAdv, PlayerBoost (Brier-optimized) | ~5-10m |
| `genetic` | Genetic algorithm optimization (scipy differential evolution) | ~10-20m |
| `bayesian` | Bayesian optimization with GP surrogate + Expected Improvement | ~10-15m |
| `autoopt` | Automatic pipeline: grid -> genetic -> bayesian, apply best | ~30-45m |
| `superopt` | Exhaustive 7-phase optimization, all 9 params (hours) | ~2-4h |
| `singleopt` | Coordinate descent, one param at a time (accuracy-focused) | ~15-30m |
| `results` | Show best parameters found across all optimizers + DSR | instant |

### Validation & Statistical Testing

| Command | Description | Time |
|---------|-------------|------|
| `purgedcv` | Purged walk-forward cross-validation (k-fold with embargo gap) | ~2m |
| `cpcv` | Combinatorial purged CV (all C(k, k_test) train/test paths) | ~5m |
| `pbo` | Probability of backtest overfitting (requires `grid` first) | ~1m |
| `montecarlo` | Monte Carlo permutation test (500 shuffles, p-value) | ~8m |
| `rollingcal` | Rolling origin Platt recalibration (expanding OOS window) | ~2m |
| `conformal` | Conformal prediction intervals (coverage at 80/90/95%) | ~1m |
| `betacal` | Beta calibration (3-param asymmetric, compare to Platt) | ~1m |
| `kelly` | Kelly criterion position sizing backtest (bankroll sim) | ~1m |

### Mega-Ensemble

| Command | Description | Time |
|---------|-------------|------|
| `mega` | Run full mega-ensemble backtest (all enabled models) | ~3-10m |
| `mega optimize` | 7-phase per-model exhaustive mega optimization (54 hyperparameters) | ~1-3h |
| `mega tune` | Per-model solo optimization (Phase 1 only) | ~15-30m |
| `mega tournament` | Head-to-head model tournament (Phase 2 only) | ~15-30m |
| `mega quick` | Quick grid search only (Phase 1) | ~20-40m |
| `mega ablation` | Ablation study: test each model's individual contribution | ~30-60m |
| `mega models` | Show all 31 models with ON/OFF status and tier | instant |
| `mega on <model>` | Enable a specific model (e.g., `mega on lstm`) | instant |
| `mega off <model>` | Disable a specific model (e.g., `mega off weather`) | instant |
| `mega on all` | Enable all 31 models | instant |
| `mega settings` | Show all mega parameter current values | instant |
| `mega set <param>=<value>` | Set a mega parameter (e.g., `mega set adj=0.10`) | instant |

### Trading Ledger

| Command | Description | Time |
|---------|-------------|------|
| `predicts` / `summary` | Show full contract ledger with P&L summary | instant |
| `balance` | Show current account balance | instant |
| `resolve` | Settle a finished contract (win/loss outcome) | instant |
| `sell` | Sell partial or full open position at market price | instant |
| `mark` | Update current market price on open lots | instant |
| `invert` | Flip side of an open position (e.g., bought YES -> now SHORT NO) | instant |
| `chart` | Generate monthly realized P&L bar chart (matplotlib) | ~2s |
| `live` | Live score tracker + open trade status (60s auto-refresh) | ongoing |
| `autoresolve` | Manually run auto-settle on today's finished games | ~5s |
| `autoresolve on` / `off` | Toggle automatic resolution during `live` tracking | instant |

### Settings

| Command | Description | Time |
|---------|-------------|------|
| `settings` | Show all current Elo parameters + Platt scaler status | instant |
| `set <param>=<value>` | Change any Elo parameter (see table below) | instant |
| `set` (no args) | List all available parameters with current values | instant |

### Utility

| Command | Description | Time |
|---------|-------------|------|
| `help` | Show full command list | instant |
| `help <command>` | Detailed help for a specific command | instant |
| `quit` | Save state and exit | instant |

---

## All Settable Parameters

### Elo Parameters (32 tunable via optimizer)

Type `set <param>=<value>` or `set <alias>=<value>`. Example: `set k=4.0`, `set home=24`, `set starter=30`.

#### Core

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k` | `k_factor` | float | 1.0 | Elo K-factor (learning rate per game). Range 0.5-10 for 162-game season. |
| `base_rating` | `base`, `rating` | float | 1500.0 | Starting Elo rating for all 30 teams |
| `home_adv` | `home`, `hca`, `home_advantage` | float | 23.47 | Home field advantage in Elo points (~54% implied win rate) |
| `use_mov` | `mov`, `margin` | bool | true | Use margin of victory (run differential) adjustment |

#### Player / Pitcher

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `player_boost` | `boost`, `player` | float | 2.54 | Team-level player strength boost from batting/pitching composite |
| `starter_boost` | `starter`, `pitcher_boost`, `sp_boost` | float | 9.61 | Starting pitcher quality adjustment (per-pitcher Elo) |
| `bullpen_factor` | `bullpen`, `bp_factor`, `reliever` | float | 6.43 | Bullpen/reliever quality factor (cumulative bullpen Elo) |
| `opp_pitcher_factor` | `opp_pitcher`, `opp_sp` | float | 18.0 | Opponent starting pitcher adjustment factor |

#### Margin of Victory

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `mov_base` | `mov_mult`, `mov_constant` | float | 0.3 | MOV multiplier constant (shifts logarithmic curve) |
| `mov_cap` | `movcap`, `margin_cap` | float | 19.9 | Maximum MOV adjustment cap (prevents blowout overweight) |

#### Rest / Schedule

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `rest_factor` | `rest` | float | 0.0 | Rest days advantage factor |
| `rest_advantage_cap` | `restcap`, `rest_cap` | float | 4.14 | Maximum rest advantage multiplier |
| `b2b_penalty` | `b2b`, `back_to_back` | float | 26.51 | Back-to-back / doubleheader game penalty |
| `road_trip_factor` | `roadtrip`, `road_trip` | float | 0.0 | Extended road trip penalty |
| `homestand_factor` | `homestand` | float | 1.41 | Extended homestand bonus |

#### Travel / Venue

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `travel_factor` | `travel` | float | 0.0 | Elo penalty per timezone crossed |
| `east_travel_penalty` | `east_travel`, `eastbound` | float | 0.0 | Extra penalty for eastbound travel (jet lag asymmetry) |
| `altitude_factor` | `altitude`, `alt` | float | 12.48 | Altitude bonus (Colorado Rockies / Coors Field only, 5,280 ft) |
| `park_factor_weight` | `parkfactor`, `park_factor`, `park` | float | 0.0 | Park factor weight for stadium run environment (FanGraphs 5-year) |

#### Form / Momentum

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `form_weight` | `form` | float | 0.0 | Recent form weight (last 15-game rolling window) |
| `win_streak_factor` | `streak`, `win_streak` | float | 0.0 | Win/loss streak momentum factor |
| `mean_reversion` | `reversion`, `regress` | float | 10.0 | Mean reversion after extreme results |
| `season_regress` | `season_regression`, `regress_pct` | float | 0.33 | Season boundary regression fraction toward 1500 |

#### Matchup Adjustments

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `sos_factor` | `sos`, `strength_of_schedule` | float | 0.0 | Strength of schedule weight |
| `division_factor` | `division`, `div` | float | 0.0 | Divisional game confidence reducer |
| `interleague_factor` | `interleague`, `il_factor` | float | 2.04 | Interleague (AL vs NL) game adjustment |
| `series_adaptation` | `series`, `adaptation` | float | 3.92 | Series adaptation factor (visiting team adjusts in games 2-3) |

#### Scoring Model

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `pace_factor` | `pace`, `tempo` | float | 5.0 | Run environment mismatch adjustment |
| `pyth_factor` | `pyth`, `pythagorean` | float | 16.0 | Pythagorean expected W% adjustment (PythagenPat dynamic exponent) |
| `scoring_consistency_factor` | `consistency`, `scoring_consistency` | float | 0.0 | Penalty for volatile run-scoring patterns |
| `home_road_factor` | `home_road`, `split` | float | 4.04 | Team-specific home/road split bonus |

#### Season / Phase

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `playoff_hca_factor` | `playoff`, `playoff_hca`, `postseason` | float | 0.934 | Playoff (October) home advantage multiplier on home_adv |
| `season_phase_factor` | `phase`, `season_phase` | float | 9.35 | Early-season dampener (reduces confidence in April ratings) |

#### K-Factor Variants

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k_decay` | `kdecay`, `k_reduction` | float | 2.07 | K-factor decay over the season (K shrinks as games played increases) |
| `surprise_k` | `surprise`, `upset_k` | float | 0.0 | Extra K for surprise/upset results (autocorrelation) |

#### Elo Scale

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `elo_scale` | -- | float | 400.0 | Probability scaling divisor (400 = standard Elo, lower = more extreme) |

#### Account / Trading

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `kelly_fraction` | `kelly` | special | 0.50 | Kelly criterion fraction (`quarter`/`half`/`full` or 0.25/0.50/1.0) |
| `starting_balance` | `balance`, `bankroll` | float | 0.0 | Starting account balance for contract tracking |
| `autoresolve_enabled` | `autoresolve`, `auto_resolve` | bool | false | Auto-resolve finished trades during live tracking |

### Mega-Ensemble Parameters (14 total)

Type `mega set <param>=<value>` or `mega set <alias>=<value>`. Example: `mega set adj=0.10`, `mega set meta=ridge`.

| Parameter | Aliases | Type | Description |
|-----------|---------|------|-------------|
| `max_adj` | `maxadj`, `adj`, `adjustment` | float | Max meta-learner adjustment (+/- probability, default ~0.10) |
| `meta_model` | `meta`, `metalearner`, `stacker` | str | Meta-learner type: `ridge`, `logistic`, or `xgboost` |
| `retrain_every` | `retrain`, `retrain_interval` | int | Retrain meta-learner every N games |
| `min_train` | `mintrain`, `min_games`, `warmup` | int | Games before meta-learner starts predicting |
| `kalman_process_noise` | `kalman_pn`, `process_noise`, `pn` | float | Kalman filter process noise |
| `kalman_measurement_noise` | `kalman_mn`, `measurement_noise`, `mn` | float | Kalman filter measurement noise |
| `hmm_states` | `hmm_n`, `n_states`, `states` | int | Number of HMM hidden states |
| `network_decay` | `net_decay`, `pagerank_decay`, `decay` | float | PageRank temporal decay (0-1) |
| `momentum_friction` | `friction`, `mom_friction` | float | Momentum friction coefficient |
| `n_clusters` | `clusters`, `k_clusters`, `nclusters` | int | Number of team archetype clusters |
| `glicko_initial_rd` | `glicko_rd`, `initial_rd`, `rd` | float | Glicko-2 initial rating deviation |
| `bt_decay` | `bt_recency`, `bradley_decay` | float | Bradley-Terry recency decay (0-1) |
| `mc_simulations` | `mc_sims`, `simulations`, `n_sims`, `sims` | int | Monte Carlo simulations per game |
| `window` | `rolling_window`, `feat_window` | int | Rolling feature window size (games, default 15) |

### Per-Model Hyperparameters (54 tunable via mega optimizer)

These are tuned automatically by `mega optimize` and can be set manually with `mega set <param>=<value>`.

| Model | Parameter | Type | Default | Search Values |
|-------|-----------|------|---------|---------------|
| **Meta XGBoost** | `meta_xgb_max_depth` | int | 4 | 3, 4, 5, 6 |
| | `meta_xgb_eta` | float | 0.05 | 0.01, 0.03, 0.05, 0.08, 0.12 |
| | `meta_xgb_subsample` | float | 0.8 | 0.6, 0.7, 0.8, 0.9 |
| | `meta_xgb_min_child_weight` | int | 5 | 3, 5, 8, 12 |
| | `meta_xgb_alpha` | float | 0.5 | 0.1, 0.3, 0.5, 1.0, 2.0 |
| | `meta_xgb_lambda` | float | 1.0 | 0.5, 1.0, 2.0, 5.0 |
| | `meta_xgb_num_boost_round` | int | 200 | 100, 150, 200, 300, 500 |
| **LightGBM** | `lgbm_num_leaves` | int | 31 | 8, 15, 31, 63 |
| | `lgbm_learning_rate` | float | 0.03 | 0.01, 0.03, 0.05, 0.1 |
| | `lgbm_n_rounds` | int | 300 | 100, 200, 300, 500 |
| | `lgbm_lambda_l1` | float | 0.1 | 0.0, 0.1, 0.5, 1.0, 2.0 |
| | `lgbm_lambda_l2` | float | 0.1 | 0.0, 0.1, 0.5, 1.0, 2.0 |
| **CatBoost** | `cb_iterations` | int | 300 | 100, 200, 300, 500 |
| | `cb_learning_rate` | float | 0.05 | 0.01, 0.03, 0.05, 0.1 |
| | `cb_depth` | int | 6 | 3, 4, 6, 8 |
| | `cb_l2_leaf_reg` | float | 3.0 | 1.0, 3.0, 5.0, 10.0 |
| **MLP** | `mlp_hidden` | str | 64,32 | 32,16 / 64,32 / 128,64 / 128,64,32 |
| | `mlp_lr` | float | 0.001 | 0.0005, 0.001, 0.003, 0.01 |
| | `mlp_epochs` | int | 100 | 50, 100, 150, 200 |
| | `mlp_dropout` | float | 0.3 | 0.1, 0.2, 0.3, 0.4, 0.5 |
| **LSTM** | `lstm_hidden_dim` | int | 64 | 32, 64, 128 |
| | `lstm_n_layers` | int | 2 | 1, 2, 3 |
| | `lstm_lr` | float | 0.001 | 0.0005, 0.001, 0.003 |
| | `lstm_epochs` | int | 80 | 40, 80, 120 |
| | `lstm_dropout` | float | 0.3 | 0.1, 0.2, 0.3, 0.5 |
| **Random Forest** | `rf_n_trees` | int | 100 | 50, 100, 200, 300 |
| | `rf_max_depth` | int | 5 | 3, 5, 7, 10 |
| | `rf_min_samples_leaf` | int | 10 | 5, 10, 15, 20 |
| **HMM** | `hmm_states` | int | 3 | 2, 3, 4, 5 |
| | `hmm_covariance_type` | str | diag | diag, full, spherical |
| | `hmm_n_iter` | int | 100 | 50, 100, 200 |
| **Kalman** | `kalman_process_noise` | float | 0.3 | 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0 |
| | `kalman_measurement_noise` | float | 2.5 | 1.5, 2.0, 2.5, 3.0, 3.5, 5.0 |
| **PageRank** | `network_decay` | float | 0.95 | 0.90, 0.93, 0.95, 0.97, 0.99 |
| | `pagerank_damping` | float | 0.85 | 0.75, 0.80, 0.85, 0.90, 0.95 |
| **GARCH** | `garch_alpha` | float | 0.10 | 0.05, 0.10, 0.15, 0.20, 0.30 |
| | `garch_beta` | float | 0.80 | 0.60, 0.70, 0.80, 0.85, 0.90 |
| **Poisson** | `poisson_home_adv` | float | 1.2 | 1.0, 1.1, 1.2, 1.3, 1.5 |
| | `poisson_decay` | float | 0.98 | 0.95, 0.97, 0.98, 0.99, 1.0 |
| **Glicko** | `glicko_initial_rd` | float | 200 | 100, 150, 200, 300, 400 |
| | `glicko_initial_vol` | float | 0.06 | 0.03, 0.06, 0.09, 0.12 |
| **Bradley-Terry** | `bt_decay` | float | 0.99 | 0.95, 0.97, 0.99, 1.0 |
| | `bt_max_iterations` | int | 100 | 50, 100, 200 |
| **Monte Carlo** | `mc_simulations` | int | 2000 | 1000, 2000, 3000, 5000 |
| | `mc_kde_bandwidth` | float | 0.3 | 0.15, 0.2, 0.3, 0.4, 0.5 |
| **Momentum** | `momentum_friction` | float | 0.05 | 0.02, 0.05, 0.08, 0.10, 0.15 |
| | `momentum_velocity_window` | int | 5 | 3, 5, 7, 10 |
| | `momentum_impulse_window` | int | 3 | 2, 3, 5 |
| **Clustering** | `n_clusters` | int | 4 | 3, 4, 5, 6, 8 |
| **Markov** | `markov_n_states` | int | 4 | 3, 4, 5, 6 |
| **Game Theory** | `gt_ema_alpha` | float | 0.05 | 0.02, 0.05, 0.10, 0.15 |
| **Info Theory** | `it_n_bins` | int | 5 | 3, 5, 7, 10 |
| **Meta Ridge** | `meta_ridge_alpha_scale` | float | 0.5 | 0.1, 0.3, 0.5, 1.0, 2.0, 5.0 |
| **Meta Logistic** | `meta_logistic_l2` | float | 0.01 | 0.001, 0.005, 0.01, 0.05, 0.1 |

---

## Data Sources

All data sources are **completely free**. No paid APIs required for core functionality.

| Source | Package / URL | Data Provided | API Key? | Rate Limit |
|--------|--------------|---------------|----------|------------|
| **MLB Stats API** | `MLB-StatsAPI` (pip) | Game scores, schedules, rosters, live scores, pitcher matchups | None needed | Unlimited |
| **ESPN Injuries** | ESPN public API | Injury reports, IL status (15-day, 60-day), DTD, Out designations | None needed | Unlimited |
| **Open-Meteo Weather** | `open-meteo.com` REST API | Temperature, wind speed/direction, humidity, precipitation | None needed | 10,000/day |
| **The Odds API** | `the-odds-api.com` | Moneyline odds from major sportsbooks (optional) | Free key (500 req/month) | 500/month |
| **Statcast / pybaseball** | `pybaseball` (pip) | xwOBA, xERA, barrel rate, sprint speed (optional) | None needed | Unlimited |

### The Odds API Setup (Optional)

The Odds API provides real-time moneyline odds. Free for up to 500 requests per month.

1. Go to [https://the-odds-api.com](https://the-odds-api.com)
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Set environment variable: `set ODDS_API_KEY=your_key_here` (Windows) or `export ODDS_API_KEY=your_key_here` (Linux/Mac)
5. Run `odds` in the CLI to see today's lines

500 requests per month is plenty for daily use -- each `odds` call uses 1 request.

### Smart Caching

The caching system adapts to the MLB season calendar (April-October) and whether today is a game day. This minimizes unnecessary API calls while keeping data fresh when it matters.

| Data Type | Offseason | Game Day (In-Season) | Non-Game Day (In-Season) |
|-----------|-----------|---------------------|--------------------------|
| **Games** (scores, results) | 7 days (168h) | 4 hours | 12 hours |
| **Players** (batting/pitching leaders) | 30 days (720h) | 48 hours | 48 hours |
| **Injuries** (ESPN IL/DTD/Out) | 30 days (720h) | 2 hours | 6 hours |
| **Odds** (moneyline) | Never fetched | 15 minutes | 4 hours |
| **Weather** (Open-Meteo forecast) | Never fetched | 2 hours | 12 hours |
| **Advanced Stats** (Statcast/FanGraphs) | 30 days (720h) | 24 hours | 24 hours |

### MLB Season Calendar

```
Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
[----OFFSEASON----][--SPRING--][------REGULAR SEASON------][OFF]
                   ^                                       ^
                   Season starts                    Postseason ends
                   (games every day)                (reduced schedule)
```

MLB plays games every single day during the regular season (April-September), with reduced schedules in March (spring training) and October (postseason). The caching system treats every day as a potential game day during the active season.

---

## File Structure

```
MLBClaude/
|
|-- main.py                     # CLI entry point, command dispatch loop
|-- config.py                   # Constants, 30 MLB teams, 6 divisions, settings I/O
|-- elo_model.py                # MLBElo class (ratings, predictions, 24+ adjusters, park factors)
|-- build_model.py              # Model training pipeline with season regression
|
|-- data_games.py               # Game data download via MLB Stats API (statsapi.schedule())
|-- data_players.py             # Batting + pitching leaders download + team composite scoring
|-- advanced_stats.py           # Statcast/FanGraphs xwOBA, xERA, barrel rate
|
|-- backtest.py                 # All backtesting & optimization (~2100 lines)
|-- enhanced_model.py           # XGBoost ensemble (31 features) + SHAP
|-- single_param_opt.py         # Coordinate descent optimizer
|
|-- platt.py                    # Calibration (Platt, isotonic, beta, season regression)
|-- metrics.py                  # LogLoss, Brier, ECE, MCE, BSS, conformal
|
|-- predict_ledger.py           # Predicts $1 contract ledger management
|-- live_scores.py              # Live MLB scores + open trade display (60s refresh)
|-- auto_resolve.py             # Auto-settle finished trades from live scores
|
|-- injuries.py                 # ESPN injury report + Elo impact calculation
|-- html_generator.py           # Blogger HTML prediction table generation
|-- help_system.py              # Help text for all commands
|-- color_helpers.py            # Colorama terminal formatting utilities
|-- cache_utils.py              # Smart season-aware API caching
|-- elo_set_handler.py          # 'set param=value' command handler (32+ params)
|
|-- hmm_model.py                # Hidden Markov Model (hot/cold team states)
|-- kalman_model.py             # Kalman Filter (team strength estimation)
|-- network_model.py            # PageRank + HITS (network analysis)
|-- gbm_models.py               # LightGBM + CatBoost gradient boosting
|-- nn_models.py                # MLP + LSTM neural networks (PyTorch)
|-- volatility_model.py         # GARCH volatility + Lyapunov/Hurst
|-- signal_model.py             # Fourier + wavelet (cycle detection)
|-- survival_model.py           # Survival analysis (streak hazards)
|-- copula_model.py             # Copula (offense/defense joint dependency)
|-- information_theory_model.py # Shannon entropy + KL divergence
|-- momentum_model.py           # Newtonian momentum / inertia
|-- markov_chain_model.py       # Markov chain transition matrices
|-- clustering_model.py         # k-Means team archetypes
|-- game_theory_model.py        # Nash equilibrium + style matchups
|-- poisson_model.py            # Poisson / Dixon-Coles run distribution
|-- glicko_model.py             # Glicko-2 uncertainty-aware ratings
|-- bradley_terry_model.py      # Bradley-Terry MLE paired comparison
|-- monte_carlo_model.py        # Monte Carlo simulation (2000+ sims)
|-- random_forest_model.py      # Random Forest (bagging diversity)
|-- classic_models.py           # SRS, Colley, Log5, PythagenPat, ExpSmooth, MeanReversion
|
|-- odds_tracker.py             # The Odds API integration + CLV tracking
|-- weather.py                  # Open-Meteo weather impact calculation
|-- kalshi.py                   # Kalshi prediction market odds
|
|-- meta_learner.py             # Ridge/Logistic/XGBoost meta-learner stacker
|-- mega_backtest.py            # Mega-ensemble walk-forward backtest engine
|-- mega_optimizer.py           # 7-phase mega-ensemble optimization
|-- mega_predictor.py           # MegaPredictor class (31-model runner)
|-- mega_config.py              # Per-model on/off switches + mega params + hyperparams
|
|-- run_optimize.py             # Optimization runner script
|-- quick_optimizer.py          # Quick optimization utilities
|-- run_enhanced_all.py         # Batch enhanced model runner
|-- sweep_enhanced.py           # Enhanced parameter sweep
|-- quick_sweep_k.py            # Quick K-factor sweep
|-- quick_sweep_new.py          # Quick parameter sweep (new params)
|-- accuracy_optimize.py        # Accuracy-focused optimization utilities
|-- accuracy_test.py            # Quick walk-forward accuracy test
|-- test_all_improvements.py    # Integration test suite
|
|-- requirements.txt            # Python dependencies
|-- CLAUDE.md                   # Claude Code context file
|-- README.md                   # This file
|
|-- mlb_elo_settings.json       # [generated] Tuned Elo parameters
|-- mlb_mega_settings.json      # [generated] Mega-ensemble settings + model switches
|-- mlb_recent_games.csv        # [generated] 2 years of game history
|-- mlb_player_stats.csv        # [generated] Batting leaders
|-- mlb_advanced_stats.csv      # [generated] Pitching leaders
|-- mlb_elo_ratings.json        # [generated] Saved Elo ratings for all 30 teams
|-- mlb_platt_scaler.json       # [generated] Platt calibration coefficients
|-- mlb_isotonic_scaler.json    # [generated] Isotonic calibration
|-- mlb_beta_scaler.json        # [generated] Beta calibration
|-- mlb_enhanced_model.json     # [generated] XGBoost metadata
|-- mlb_xgb_model.json          # [generated] XGBoost model weights
|-- mlb_meta_xgb.json           # [generated] Meta-learner XGBoost model
|-- mlb_enhanced_features.npz   # [generated] Cached enhanced feature arrays
|-- predicts_lots.csv           # [generated] Trading ledger
|-- weather_cache.json          # [generated] Weather forecast cache
```

---

## Optimization System

### Elo Optimization (6 methods)

| Phase | Command | Method | Params | Time |
|-------|---------|--------|--------|------|
| 1 | `grid` | Exhaustive grid search | K, HomeAdv, PlayerBoost (768+ combos) | ~5-10m |
| 2 | `genetic` | Differential evolution (scipy) | 7 params, 50 gen x 25 pop | ~10-20m |
| 3 | `bayesian` | Gaussian Process + Expected Improvement | 7 params, 15 initial + 40 iter | ~10-15m |
| 4 | `autoopt` | Automatic pipeline (grid -> genetic -> bayesian) | 7 params, best of all three | ~30-45m |
| 5 | `superopt` | Exhaustive 7-phase multi-round optimization | 9 params, hours of search | ~2-4h |
| 6 | `singleopt` | Coordinate descent (one param at a time) | All params, accuracy-focused | ~15-30m |

**`superopt` 7-phase detail:**

1. **Phase 1 -- Broad Grid Search**: 9 params, ~6,000+ combinations. Establishes the promising region of parameter space.
2. **Phase 2 -- Genetic Round 1**: Wide bounds, 100 generations x 50 population. Differential evolution explores the full space.
3. **Phase 3 -- Bayesian Round 1**: Wide bounds, 30 initial points + 80 iterations. GP surrogate models the objective surface.
4. **Phase 4 -- Genetic Round 2**: Tightened bounds around best-so-far, 80 generations x 40 population. Intensifies search in the best region.
5. **Phase 5 -- Bayesian Round 2**: Tightened bounds, 20 initial + 60 iterations. Fine-grained exploitation of the GP model.
6. **Phase 6 -- Fine Grid**: Tiny step sizes around the absolute best parameters found. Ensures no nearby optimum was missed.
7. **Phase 7 -- Validation**: Runs purgedcv + PBO + Monte Carlo on the winning parameters to confirm they are not overfit.

### Mega-Ensemble Optimization (7 phases, 54 per-model hyperparameters)

| Phase | Method | Description |
|-------|--------|-------------|
| 1 | Solo test | Test each model individually to find per-model optimal settings. |
| 2 | Tournament | Top configs compete head-to-head on held-out data. |
| 3 | Meta-learner tuning | Optimize `max_adj`, `meta_model`, `retrain_every`, `min_train`. |
| 4 | Differential Evolution (DE) | Fine-tune all continuous params with genetic optimization. |
| 5 | Ablation | Prune models that hurt ensemble accuracy. |
| 6 | Validation | Purged CV + stability test with multiple random seeds. |
| 7 | Apply best | Save winning parameters. |

Use `mega tune` for per-model solo optimization (Phase 1 only) and `mega tournament` for head-to-head model comparison (Phase 2 only).

**Mega ablation** (`mega ablation`): Disables each model one at a time and measures the accuracy change. Models that hurt overall accuracy are automatically flagged for pruning. This identifies which of the 31 models are contributing positive signal and which are adding noise.

### Recommended Optimization Workflow

```
Step 1:  python main.py                     # Baseline backtest runs automatically (32 Elo params)
Step 2:  grid                               # Find promising region (~5-10m)
Step 3:  genetic                            # Refine with evolution (~10-20m)
Step 4:  bayesian                           # Fine-tune with GP (~10-15m)
Step 5:  results                            # Compare all optimizer outputs
Step 6:  backtest                           # Refit Platt with best params
Step 7:  mega                               # Run mega-ensemble with Elo base
Step 8:  mega tune                          # Per-model solo optimization
Step 9:  mega tournament                    # Head-to-head model comparison
Step 10: mega optimize                      # Full 7-phase optimization (54 per-model hyperparams, ~1-3h)
Step 11: mega ablation                      # Prune bad models (~30-60m)
Step 12: purgedcv -> pbo -> montecarlo      # Validate (not overfit)
Step 13: kelly                              # Size positions optimally
```

Or use the fully automated shortcut:
```
autoopt                                     # Steps 2-4 automated (~30-45m)
superopt                                    # Steps 2-6 + validation (~2-4h)
```

---

## Complete Model Validation Workflow

A rigorous 6-phase workflow ensures your model is genuinely predictive and not overfit to historical data.

### Phase 1: Baseline & Diagnostics

```
backtest          # Walk-forward accuracy, LogLoss, Brier, ECE
convergence       # How many games before Elo ratings stabilize?
sliding           # Does old data help or hurt? (sliding vs expanding window)
```

Establishes baseline metrics. The `convergence` command identifies the burn-in period (typically 200-400 games for MLB). `sliding` determines whether the model benefits from full history or performs better with a shorter memory.

### Phase 2: Parameter Optimization

```
grid              # Broad search over K, HomeAdv, PlayerBoost
pbo               # Is the grid search overfit? (PBO > 0.5 = overfit)
results           # Compare all optimizer outputs + Deflated Sharpe Ratio
genetic           # Refine with differential evolution
```

The Probability of Backtest Overfitting (PBO) test is critical here. If PBO > 0.5, your grid search likely found parameters that are overfit to this specific data window. The Deflated Sharpe Ratio (DSR) adjusts for multiple comparisons.

### Phase 3: Cross-Validation

```
purgedcv          # k-fold with embargo gap (prevents Elo momentum leakage)
cpcv              # All C(k, k_test) paths for tighter confidence intervals
montecarlo        # 500 permutation shuffles -> p-value for significance
```

Purged CV adds an embargo gap between train and test folds to prevent Elo momentum from leaking across boundaries. CPCV produces many more backtest paths for tighter confidence. Monte Carlo gives a p-value: if < 0.05, the model's edge is statistically significant.

### Phase 4: Ensemble & Features

```
enhanced          # XGBoost ensemble (31 features, 80/20 blend)
shap              # Which features are driving XGBoost predictions?
enhanced decay    # Time-decayed weighting (XGB gets more weight over season)
```

SHAP analysis reveals whether XGBoost is adding genuine signal beyond Elo or just echoing it. If the top SHAP features are all Elo-derived, the ensemble may not be adding value.

### Phase 5: Calibration

```
rollingcal        # Expanding-window Platt recalibration (truly OOS)
betacal           # 3-parameter beta calibration (handles asymmetry)
conformal         # Distribution-free prediction intervals with coverage
```

Rolling calibration gives truly out-of-sample calibrated metrics. Beta calibration fixes asymmetric miscalibration (e.g., overconfident on favorites but well-calibrated on underdogs). Conformal prediction provides coverage guarantees without distributional assumptions.

### Phase 6: P&L Simulation

```
kelly             # Kelly criterion bankroll simulation on backtest
```

Simulates optimal position sizing over the backtest period. Reports final bankroll, maximum drawdown, Sharpe ratio, and win rate. Uses fractional Kelly (default 50%) for practical sizing.

---

## Decision Framework

| Metric | Good | Marginal | Bad |
|--------|------|----------|-----|
| **Accuracy** | > 57% | 55-57% | < 55% |
| **ECE** (calibration error) | < 0.02 | 0.02-0.05 | > 0.05 |
| **BSS** (Brier Skill Score vs 50%) | > 0.02 | 0.00-0.02 | < 0.00 |
| **PBO** (Prob Backtest Overfit) | < 0.30 | 0.30-0.50 | > 0.50 |
| **DSR** (Deflated Sharpe Ratio) | > 2.0 | 1.0-2.0 | < 1.0 |
| **Monte Carlo p-value** | < 0.05 | 0.05-0.10 | > 0.10 |
| **Purged CV std** | < 2% | 2-4% | > 4% |
| **CPCV paths > 55%** | > 90% | 70-90% | < 70% |
| **Kelly Sharpe** | > 1.0 | 0.5-1.0 | < 0.5 |
| **Kelly max drawdown** | < 20% | 20-40% | > 40% |

**Interpretation**: If most metrics are "Good", the model has genuine predictive power. If PBO is "Bad" or Monte Carlo p > 0.10, the model's apparent edge is likely noise. Do not trade a model with "Bad" validation metrics.

---

## Trading Ledger

The Predicts $1 contract tracking system models binary outcome contracts (similar to prediction market contracts) where each contract settles at $1.00 (win) or $0.00 (loss).

### How It Works

1. **Entry**: Buy a contract at the model's implied probability (e.g., buy NYY at $0.62)
2. **Entry fee**: 2% of entry price deducted at purchase
3. **Settlement**: Contract resolves to $1.00 (team wins) or $0.00 (team loses)
4. **Profit/Loss**: Settlement value minus entry price minus fees
5. **Exit fee**: 2% deducted if you sell before settlement

### Commands

- `predicts` / `summary` -- Show all lots with entry price, current mark, P&L
- `balance` -- Current account balance
- `resolve` -- Settle a contract (enter W or L outcome)
- `sell` -- Exit a position early at current market price
- `mark` -- Update the current market price of open positions
- `invert` -- Flip the side of a position (e.g., bought YES -> now SHORT NO)
- `chart` -- Monthly realized P&L bar chart
- `autoresolve` -- Auto-settle using today's game results from the MLB Stats API
- `live` -- Watch live scores with real-time P&L on open trades (60s refresh loop)

### Kelly Criterion Position Sizing

The system recommends position sizes using the Kelly criterion formula. Default is half-Kelly (50% of optimal) for practical risk management. Configurable via `set kelly=quarter` (25%), `set kelly=half` (50%), or `set kelly=full` (100%).

### Auto-Resolve

When `autoresolve on` is active, the system automatically settles contracts when final game scores are detected during `live` tracking. You can also manually trigger `autoresolve` to batch-settle all finished games.

### Mark-to-Market

Open positions can be marked to current market prices at any time using `mark`. This updates the unrealized P&L without closing the position. The `predicts` summary shows both unrealized (mark-to-market) and realized (settled) P&L.

---

## Daily Prediction Workflow

```
1. LAUNCH
   python main.py
   -> Auto-downloads latest games, players, injuries
   -> Runs baseline backtest (fits Platt scaler)
   -> Shows baseline accuracy

2. CHECK MODEL STATUS
   settings            # Verify parameters are tuned
   platt               # Confirm calibration scaler is fitted
   mega models         # Check which models are enabled

3. MAKE PREDICTIONS
   today               # Generate HTML table for all today's games
   yankees             # Individual matchup prediction (fuzzy search)
   -> Enter opponent, home team, see calibrated probability
   -> 'y' to log as Predicts contract

4. LOG POSITIONS
   balance             # Check account balance
   -> Enter contracts through prediction flow
   predicts            # Review all open positions

5. PUBLISH (optional)
   today               # Generates today_mlb_predictions.html
   blogger             # Same as today -- copy HTML to Blogger

6. MONITOR
   live                # Live score tracker with open trade status
                       # Auto-refreshes every 60 seconds
   odds                # Check latest odds for CLV comparison

7. SETTLE
   autoresolve         # Auto-settle finished games
   resolve             # Manually settle a specific contract
   sell                # Exit early at market price

8. REVIEW
   predicts            # Full P&L summary
   chart               # Monthly P&L bar chart
   kelly               # Was sizing optimal?
```

---

## Performance

### Multithreading

The mega-ensemble uses `concurrent.futures.ThreadPoolExecutor` to run all 31 base models in parallel. On typical hardware:

- **4-core machine**: ~2-3x speedup over sequential
- **8-core machine**: ~3-5x speedup over sequential
- **16-core machine**: ~5-8x speedup over sequential

The GIL is not a bottleneck because most models spend time in C extensions (numpy, scipy, xgboost, lightgbm, catboost) which release the GIL.

### GPU Acceleration

| Library | GPU Backend | Speedup | Detection |
|---------|------------|---------|-----------|
| XGBoost | CUDA (`gpu_hist`) | 3-10x on tree construction | Automatic if CUDA available |
| LightGBM | CUDA (`device='gpu'`) | 2-5x on tree construction | Automatic if CUDA available |
| CatBoost | CUDA (`task_type='GPU'`) | 3-8x on tree construction | Automatic if CUDA available |
| PyTorch (MLP/LSTM) | CUDA | 5-20x on neural network training | `torch.cuda.is_available()` |

GPU is entirely optional. All models fall back to CPU silently. No configuration needed.

### Typical Backtest Results

Results vary by season and parameter tuning. Typical ranges:

- **Baseline accuracy**: ~56.78%
- **Platt-calibrated accuracy**: ~56.78%
- **Full mega-ensemble**: 55-58%
- **LogLoss**: ~0.6792
- **Brier score**: ~0.2431
- **ECE** (calibration error): ~0.0037
- **BSS vs 50%**: ~0.0274
- **Games tested**: ~5,918
- **Pitchers tracked**: ~648

Baseball is inherently more random than other sports (best teams win ~60% of games, worst teams win ~40%), so accuracy above 57% on moneyline picks represents strong performance.

---

## Requirements

### System Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: 4 GB minimum, 8 GB recommended (mega-ensemble holds all models in memory)
- **Disk**: ~500 MB for cached data + model files
- **Internet**: Required for API data downloads (can run offline with cached data)
- **GPU**: Optional (CUDA-capable NVIDIA GPU for XGBoost/LightGBM/CatBoost/PyTorch acceleration)

### Python Dependencies

#### Core (Required)

```
pandas>=1.5
numpy>=1.24
scipy>=1.10
colorama>=0.4
xgboost>=2.0
requests>=2.28
matplotlib>=3.7
```

#### MLB Data APIs

```
MLB-StatsAPI>=1.7         # MLB game scores, schedules, live data (free, no key)
```

#### Prediction Models

```
hmmlearn>=0.3             # Hidden Markov Models
filterpy>=1.4             # Kalman filters
lightgbm>=4.0             # LightGBM gradient boosting
catboost>=1.2             # CatBoost gradient boosting
networkx>=3.0             # PageRank / HITS graph analysis
```

#### Neural Networks (CPU or GPU)

```
torch>=2.0                # MLP + LSTM (PyTorch)
```

CPU-only install (smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

GPU install (requires CUDA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Optional (Enhanced Features)

```
pybaseball>=2.3           # Statcast data (xwOBA, xERA, barrel rate) from FanGraphs/Baseball Savant
nolds                     # Lyapunov exponents, Hurst exponent (chaos theory metrics)
PyWavelets                # Wavelet transforms (signal processing)
openmeteo-requests        # Weather data helper (not strictly required, plain requests works)
```

### Quick Install

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Sports prediction models are inherently uncertain -- even the best models are wrong 40-45% of the time for MLB moneyline picks. No model can guarantee profits. Past backtest performance does not predict future results. Always gamble responsibly and never risk money you cannot afford to lose.

The prediction probabilities produced by this system are statistical estimates, not certainties. The Predicts $1 contract ledger is a paper-trading simulation tool, not a connection to any real prediction market or sportsbook.

---

## Recent Changes

- **Fixed `_rolling()` rest_days bug**: Away team was incorrectly using home team's rest days in XGBoost features. Each team now correctly computes its own rest days from its own schedule.
- **Fixed broad exception handling in `backtest.py`**: Replaced `except Exception` with `except OSError` to avoid silently swallowing unexpected errors during optimization I/O.
- **Added NaN guard for momentum autocorrelation in `enhanced_model.py`**: Prevents NaN propagation when autocorrelation returns undefined values for short or constant series.
- **Updated Athletics park factor to 1.00 (neutral)**: Reflects the move to Sacramento's temporary venue, which has no established park factor history.
- **Optimized Elo parameters from Bayesian optimization results**: Updated 16 default parameters based on Bayesian optimization best-found values.
- **Made `season_regress` configurable via settings**: Season regression fraction (default 0.33) can now be tuned through the `set season_regress=<value>` command and is included in optimizer parameter sweeps.
- **Re-optimized all Elo parameters via genetic optimization**: Current baseline at 56.78% acc, 0.6792 LL, 0.2431 Brier, 0.0037 ECE, 0.0274 BSS (5918 games). Updated all 30+ defaults including k=1.0, home_adv=23.47, starter_boost=9.61, pyth_factor=16.0, opp_pitcher_factor=18.0, altitude_factor=12.48, series_adaptation=3.92, interleague_factor=2.04, k_decay=2.07. Several factors zeroed out (rest_factor, form_weight, travel_factor, sos_factor, division_factor, win_streak_factor, road_trip_factor, park_factor_weight).

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
