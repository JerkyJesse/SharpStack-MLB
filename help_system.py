"""Help display system."""

from color_helpers import chi, cdim, div, hdr


def show_help(topic=""):
    topic = topic.strip().lower()
    HELP_TOPICS = {
        "predict","all","refresh","backtest","grid","genetic","results",
        "players","settings","set","predicts","summary","resolve","sell",
        "mark","invert","chart","live","autoresolve","today","html",
        "blog","blogger","quit","help","platt","calibrate",
        "bayesian","purgedcv","cpcv","pbo","montecarlo","rollingcal",
        "kelly","sliding","convergence","conformal","betacal","shap",
        "enhanced","advanced","autoopt","superopt","singleopt",
    }
    if topic and topic not in HELP_TOPICS:
        from color_helpers import cwarn
        print(cwarn("\n  Unknown help topic: '%s'. Type 'help' for full list.\n" % topic))
        return

    W = 100

    def section(cmd, usage, desc, steps=None, notes=None, examples=None):
        div(W)
        print("  %s  Usage: %s" % (chi(cmd.upper()), usage))
        print("  Summary: %s" % desc)
        if steps:
            print("  Steps:")
            for i, s in enumerate(steps, 1):
                print("    %d. %s" % (i, s))
        if notes:
            print("  Notes:")
            for n in notes:
                print("    * %s" % n)
        if examples:
            print("  Examples: %s" % "  ".join(chi(e) for e in examples))

    if not topic:
        hdr("MLB SHARPSTACK - PREDICTS $1 TRACKER  .  HELP")
        print("""
  QUICK WORKFLOW SUMMARY
  1. Run 'backtest' FIRST -> fits Platt calibration scaler
  2. Type a team name     -> start prediction (uses calibrated probs)
  3. Answer 'y' to log contract -> saved to predicts_lots.csv
  4. Use 'live' to watch games in real time
  5. Use 'resolve' / 'sell' / 'mark' to manage positions
  6. Use 'predicts' / 'chart' for overview & visualization
  7. Use 'today' / 'blogger' to export today's games table for Blogger
""")
        print("  MAIN COMMANDS  (type 'help <command>' for detailed help)")
        div(W)
        cmds = [
            ("<team name>",        "Start prediction for any team (fuzzy match)"),
            ("backtest",           "Run backtest + fit Platt calibration scaler"),
            ("platt",              "Show Platt scaler status / refit info"),
            ("live",               "Live score tracker + open trades status"),
            ("today/html/blogger", "Generate HTML table for today's games"),
            ("all",                "Show current Elo ratings for all teams"),
            ("players",            "Show top sluggers (league or per team)"),
            ("refresh",            "Force redownload of games + player stats"),
            ("grid",               "Grid search best K / HomeAdv / PlayerBoost (Brier-optimized)"),
            ("genetic",            "Genetic algorithm optimization (Brier-optimized)"),
            ("autoopt",            "Auto grid+genetic+bayesian, apply best (no prompts)"),
            ("superopt",           "Exhaustive 7-phase optimization, all 9 params (hours)"),
            ("singleopt",          "Coordinate descent, one param at a time (accuracy-focused)"),
            ("results",            "Show best found parameters from all optimizers"),
            ("predicts/summary",   "Show full contract ledger + P&L"),
            ("chart",              "Generate monthly realized P&L bar chart"),
            ("resolve",            "Settle finished contract (win/loss)"),
            ("sell",               "Sell partial or full open position"),
            ("mark",               "Update current market price on open lots"),
            ("invert",             "Flip side of open position (no accounting change)"),
            ("autoresolve",        "Manually run auto-settle on finished games"),
            ("autoresolve on/off", "Toggle automatic resolve during 'live'"),
            ("settings",           "Show current Elo parameters + Platt scaler status"),
            ("set k=32",           "Change settings (k, home_adv, player_boost, etc.)"),
            ("help [topic]",       "This help screen (or detailed help on one command)"),
            ("quit",               "Save & exit"),
        ]
        for cmd, desc in cmds:
            print("  %-22s %s" % (chi(cmd), desc))
        print()
        print("  ADVANCED BACKTESTING & VALIDATION  (type 'help advanced' for details)")
        div(W)
        adv_cmds = [
            ("bayesian",           "Bayesian hyperparameter optimization (GP surrogate + EI)"),
            ("purgedcv",           "Purged walk-forward cross-validation (k-fold with embargo)"),
            ("cpcv",               "Combinatorial purged CV (all C(k,k_test) paths)"),
            ("pbo",                "Probability of backtest overfitting"),
            ("montecarlo",         "Monte Carlo permutation test (null distribution)"),
            ("rollingcal",         "Rolling origin recalibration (expanding Platt window)"),
            ("kelly",              "Kelly criterion position sizing backtest"),
            ("sliding",            "Sliding vs expanding window comparison"),
            ("convergence",        "Elo rating convergence / burn-in analysis"),
            ("conformal",          "Conformal prediction intervals (coverage guarantees)"),
            ("betacal",            "Beta calibration (asymmetric, 3-param)"),
            ("shap",               "SHAP feature importance for XGBoost ensemble"),
            ("enhanced decay",     "Enhanced backtest with time-decayed Elo/XGB weighting"),
            ("autoopt",            "Auto-optimize: grid -> genetic -> bayesian, apply best"),
            ("superopt",           "Exhaustive multi-round optimization (all methods, 9 params)"),
            ("singleopt",          "Coordinate descent optimizer (one param at a time)"),
        ]
        for cmd, desc in adv_cmds:
            print("  %-22s %s" % (chi(cmd), desc))
        print()
        return

    if topic == "advanced":
        section("advanced", "help advanced",
                "All advanced backtesting and validation methods.",
                notes=[
                    "bayesian     - GP surrogate + Expected Improvement optimizer (~50-100 evals)",
                    "purgedcv     - k-fold CV with embargo gap to prevent Elo leakage",
                    "cpcv         - Combinatorial purged CV for tighter confidence intervals",
                    "pbo          - Probability of backtest overfitting (post-grid analysis)",
                    "montecarlo   - Null distribution test (shuffle outcomes, compute p-value)",
                    "rollingcal   - Expanding-window Platt recalibration (OOS calibrated metrics)",
                    "kelly        - Kelly criterion bankroll simulation on backtest predictions",
                    "sliding      - Sliding vs expanding window comparison",
                    "convergence  - Elo burn-in period analysis (chunk accuracy)",
                    "conformal    - Distribution-free prediction intervals with coverage guarantees",
                    "betacal      - 3-parameter beta calibration for asymmetric miscalibration",
                    "shap         - XGBoost native SHAP feature importance analysis",
                    "enhanced decay - Time-decayed ensemble (Elo 95%->70%, XGB 5%->30%)",
                    "singleopt    - Coordinate descent (one param at a time, accuracy-focused)",
                    "DSR (Deflated Sharpe Ratio) shown automatically in 'results' command",
                    "ECE/MCE/BSS shown automatically in 'backtest' command output",
                ])
    elif topic == "bayesian":
        section("bayesian", "bayesian",
                "Bayesian hyperparameter optimization with GP surrogate.",
                notes=["More sample-efficient than grid search (finds optima in ~50-100 evals)",
                       "Uses Gaussian Process surrogate + Expected Improvement acquisition",
                       "Same 7 parameters and objective as grid/genetic",
                       "Auto-saves best settings and refits Platt on completion"])
    elif topic == "purgedcv":
        section("purgedcv", "purgedcv",
                "Purged walk-forward cross-validation.",
                notes=["Splits games into k chronological folds with embargo gap",
                       "Embargo prevents Elo momentum leakage between train/test",
                       "Reports mean/std of metrics across folds",
                       "If accuracy varies >3%%, model is fragile to training data"])
    elif topic == "cpcv":
        section("cpcv", "cpcv",
                "Combinatorial purged CV - all C(k, k_test) train/test combinations.",
                notes=["Produces many backtest paths with overlapping training data",
                       "Gives tighter confidence intervals than standard k-fold",
                       "If 90%% of paths show >55%% accuracy, model is robust"])
    elif topic == "pbo":
        section("pbo", "pbo",
                "Probability of backtest overfitting.",
                notes=["Requires grid search results (run 'grid' first)",
                       "PBO > 0.5 means best params are likely overfit",
                       "Uses symmetric cross-validation on trial population"])
    elif topic == "montecarlo":
        section("montecarlo", "montecarlo",
                "Monte Carlo permutation test for statistical significance.",
                notes=["Shuffles win/loss outcomes, re-runs backtest 500+ times",
                       "Computes p-value: fraction of shuffled runs that beat real model",
                       "p > 0.05 means model's edge isn't statistically significant",
                       "Takes ~8 minutes for 500 permutations"])
    elif topic == "rollingcal":
        section("rollingcal", "rollingcal",
                "Rolling origin Platt recalibration.",
                notes=["Fits Platt on expanding window of past games only",
                       "Gives truly out-of-sample calibrated metrics",
                       "Detects if calibration drifts over the season"])
    elif topic == "kelly":
        section("kelly", "kelly",
                "Kelly criterion position sizing backtest.",
                notes=["Simulates optimal position sizing on historical predictions",
                       "Uses fractional Kelly (default 25%%) for practical sizing",
                       "Reports: final bankroll, max drawdown, Sharpe ratio",
                       "Requires 'backtest' to have been run first"])
    elif topic == "sliding":
        section("sliding", "sliding",
                "Sliding vs expanding window backtest comparison.",
                notes=["Sliding: heavy regression every N games (model 'forgets')",
                       "Expanding: standard Elo (full history, current behavior)",
                       "Determines if old data (>3 months) helps or hurts predictions"])
    elif topic == "convergence":
        section("convergence", "convergence",
                "Elo rating convergence / burn-in analysis.",
                notes=["Chunks predictions and computes per-chunk accuracy",
                       "Identifies how many games before ratings stabilize",
                       "Suggested burn-in period for honest accuracy figures",
                       "Requires 'backtest' to have been run first"])
    elif topic == "conformal":
        section("conformal", "conformal",
                "Conformal prediction intervals with coverage guarantees.",
                notes=["Distribution-free: 'set contains correct outcome with X%% probability'",
                       "Reports coverage at 90%%, 95%%, and 80%% significance levels",
                       "Singleton = model confident, Both = uncertain, Empty = overconfident",
                       "Requires 'backtest' to have been run first"])
    elif topic == "betacal":
        section("betacal", "betacal",
                "Beta calibration (3-parameter, handles asymmetric miscalibration).",
                notes=["More flexible than Platt: logit(p_cal) = c + a*log(p) - b*log(1-p)",
                       "Fixes overconfident-on-favorites but well-calibrated-on-underdogs",
                       "If a != b, calibration is asymmetric (beta is better)",
                       "Requires 'backtest' to have been run first"])
    elif topic == "shap":
        section("shap", "shap",
                "SHAP feature importance for XGBoost ensemble.",
                notes=["Uses XGBoost native pred_contribs (no extra deps)",
                       "Shows which of the 20 features drive predictions",
                       "Reveals if XGBoost is just echoing Elo or adding signal",
                       "Auto-runs at end of 'enhanced', also standalone"])
    elif topic == "enhanced":
        section("enhanced / enhanced decay", "enhanced  OR  enhanced decay",
                "XGBoost ensemble backtest, optionally with time-decayed weighting.",
                notes=["enhanced: fixed 80/20 Elo/XGBoost blend",
                       "enhanced decay: transitions from 95%% Elo early to 70%% Elo late",
                       "Time decay acknowledges XGBoost features are noisy early-season",
                       "SHAP analysis runs automatically at end"])
    elif topic == "superopt":
        section("superopt", "superopt",
                "Exhaustive multi-round optimization - finds absolute best settings.",
                steps=["Phase 1: Broad grid search (9 params, ~6000+ combos)",
                       "Phase 2: Genetic round 1 (wide bounds, 100 gen x 50 pop)",
                       "Phase 3: Bayesian round 1 (wide bounds, 30 initial + 80 iter)",
                       "Phase 4: Genetic round 2 (tightened bounds, 80 gen x 40 pop)",
                       "Phase 5: Bayesian round 2 (tightened, 20 initial + 60 iter)",
                       "Phase 6: Fine grid (tiny steps around absolute best)",
                       "Phase 7: Validation (purgedcv + PBO + Monte Carlo)"],
                notes=["Searches ALL 9 tunable params (K, HA, PB, Rest, Travel, Pace, PHCA, SOS, Form)",
                       "No prompts - runs fully unattended for hours if needed",
                       "Each round narrows the search space based on prior results",
                       "Settings auto-saved to mlb_elo_settings.json (loads on every start)",
                       "Platt scaler auto-refitted with winning parameters",
                       "Takes 2-4 hours depending on data size"])
    elif topic == "autoopt":
        section("autoopt", "autoopt",
                "Fully automatic optimization: grid -> genetic -> bayesian.",
                steps=["Phase 1: Coarse grid search (768 combos) to find promising region",
                       "Phase 2: Genetic optimization (50 gen, pop 25) with tightened bounds",
                       "Phase 3: Bayesian optimization (15 initial + 40 iterations)",
                       "Phase 4: Compare all three winners, apply the absolute best"],
                notes=["No interactive prompts — runs fully unattended",
                       "Takes ~15-30 minutes depending on data size",
                       "Auto-saves best settings and refits Platt on completion",
                       "Equivalent to running grid -> genetic -> bayesian manually"])
    elif topic == "singleopt":
        section("singleopt", "singleopt",
                "Coordinate descent: optimize one parameter at a time for accuracy.",
                steps=["Load current settings as starting point",
                       "For each parameter: coarse sweep across full range",
                       "Fine-grained sweep around best coarse value",
                       "Accept new value only if accuracy improves",
                       "Repeat passes until no parameter improves (max 5 passes)"],
                notes=["Optimizes for accuracy (%) not LogLoss/Brier",
                       "Conservative local search - won't miss nearby optima",
                       "Good for fine-tuning after grid/genetic narrows the region",
                       "Aliases: singleopt, single-opt, coorddescent, coord",
                       "Auto-saves best settings and rebuilds model on completion"])
    elif topic in ("platt", "calibrate"):
        section("platt / calibrate", "platt",
                "Show Platt calibration scaler status.",
                notes=[
                    "Fixes overconfidence at 0.9+ and underconfidence at 0.1-0.2 (from your calibration data)",
                    "Fitted automatically when you run 'backtest'",
                    "Saved to mlb_platt_scaler.json, loaded automatically on startup",
                    "Refit after any major refresh or param change",
                ],
                examples=["platt", "backtest"])
    elif topic == "backtest":
        section("backtest", "backtest",
                "Run historical accuracy test and fit Platt calibration scaler.",
                notes=[
                    "Fits scaler from raw Elo probabilities (no leakage)",
                    "Reports both raw and calibrated Brier scores",
                    "All subsequent predictions use calibrated output",
                ])
    elif topic in ("predict", "team"):
        section("<team name>", "yankees  OR  dodgers  OR  any team name fragment",
                "Start prediction for a game.",
                steps=["Type part of a team name", "Enter opponent",
                       "Answer home team question (a / b / n)",
                       "See calibrated win probability + top players",
                       "Optionally log as contract"],
                examples=["yankees", "dodgers", "cubs"])
    elif topic in ("today","html","blog","blogger"):
        section("today / html / blog / blogger", "today  OR  html  OR  blogger",
                "Generate ready-to-paste HTML table for today's MLB games.",
                steps=["Type 'today'", "Fetches today's schedule + live scores",
                       "Saves today_mlb_predictions.html",
                       "Copy all -> paste into Blogger HTML view -> publish"])
    elif topic in ("settings","set"):
        section("settings / set", "settings  OR  set k=4  OR  set home=24",
                "View or change Elo parameters.",
                notes=["Available: k, home, boost, rest, travel, sos, pace, playoff, form",
                       "After changing params, run 'backtest' to refit Platt scaler"],
                examples=["set k=4.0", "set home=24.0", "set boost=20", "set sos=10",
                           "set form=5", "settings"])
    elif topic in ("backtest","grid","genetic","results"):
        section("backtest / grid / genetic / results",
                "backtest  OR  grid  OR  genetic  OR  results",
                "Evaluate & tune model. Grid/genetic now optimize for Brier (calibration quality).",
                notes=["backtest -> fits Platt scaler, reports calibrated Brier",
                       "grid/genetic -> objective is -(logloss*8 + brier*40)",
                       "results -> show best parameters found so far"])
    print()
