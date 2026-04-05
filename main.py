#!/usr/bin/env python3
"""
MLB MONEYBALL + PREDICTS $1 CONTRACT TRACKER
v3: Elo + XGBoost ensemble, rest days, injury awareness
"""

import os
import logging

from colorama import Fore, Back, Style

from color_helpers import cok, cerr, cwarn, chi, cdim, cbold, div
from config import (
    GAMES_FILE, PLAYER_STATS_FILE, RATINGS_FILE, PLATT_SCALER_FILE, SETTINGS_FILE,
    load_elo_settings, save_elo_settings,
)
from platt import load_platt_scaler
from data_games import download_recent_games
from data_players import load_player_stats, download_player_stats, download_advanced_stats, show_player_metrics
from build_model import build_model
from backtest import (backtest_model, grid_search_optimization, genetic_optimization,
                      show_optimization_results, bayesian_optimization,
                      purged_walk_forward_cv, combinatorial_purged_cv,
                      probability_of_backtest_overfitting, monte_carlo_permutation_test,
                      rolling_origin_recalibration, kelly_criterion_backtest,
                      sliding_window_backtest, elo_convergence_analysis,
                      conformal_analysis, run_beta_calibration, auto_optimize,
                      super_optimize)
from predict_ledger import (
    load_predict_lots, save_predict_lots, add_predict_contract,
    show_open_lots, mark_pending_positions, invert_open_trade,
    sell_predict_contract, resolve_predict_contracts,
    summarize_predict_lots, plot_pnl_chart,
    prompt_balance, show_balance, show_kelly_recommendation,
)
from live_scores import show_live_scores_for_open_trades
from auto_resolve import auto_resolve_finished_trades
from html_generator import generate_today_predictions_html, generate_tomorrow_predictions_html
from help_system import show_help
from enhanced_model import (run_enhanced_backtest, save_enhanced_model,
                            load_enhanced_model, shap_feature_importance)
from injuries import show_injury_report, get_team_injuries, calc_injury_impact, manual_set_injuries, fetch_injury_report
from single_param_opt import run_coordinate_descent

# Phase 1: Mega-ensemble data sources
import sys
try:
    from odds_tracker import show_odds_table, get_today_odds, find_game_odds
    from weather import show_weather_report, get_game_weather, compute_weather_impact
    pass  # sentiment module removed
    HAS_MEGA_DATA = True
except ImportError as _e:
    HAS_MEGA_DATA = False
    logging.debug("Mega-ensemble data modules not available: %s", _e)

try:
    from kalshi import find_kalshi_odds, show_kalshi_odds
    HAS_KALSHI = True
except ImportError:
    HAS_KALSHI = False

try:
    from advanced_stats import (load_or_download_advanced, show_team_rankings as show_adv_rankings,
                                get_advanced_features as get_mlb_advanced_features)
    HAS_ADVANCED_STATS = True
except ImportError as _e:
    HAS_ADVANCED_STATS = False
    logging.debug("Advanced stats module not available: %s", _e)

try:
    from mega_backtest import run_mega_backtest
    HAS_MEGA_BACKTEST = True
except ImportError as _e:
    HAS_MEGA_BACKTEST = False
    logging.debug("Mega backtest not available: %s", _e)

try:
    from mega_optimizer import (run_mega_optimize, run_quick_optimize,
                                 run_deep_optimize, run_single_model_optimize)
    HAS_MEGA_OPTIMIZER = True
except ImportError as _e:
    HAS_MEGA_OPTIMIZER = False
    logging.debug("Mega optimizer not available: %s", _e)

try:
    from mega_config import (load_model_switches, save_model_switches,
                              is_model_enabled, print_model_status, ALL_MODELS,
                              handle_mega_set, print_mega_settings)
    HAS_MEGA_CONFIG = True
except ImportError:
    HAS_MEGA_CONFIG = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def dispatch(cmd, model, csv_file):
    if cmd == "all":
        model.show_all_teams()
    elif cmd == "refresh":
        for f in [GAMES_FILE, PLAYER_STATS_FILE, RATINGS_FILE, PLATT_SCALER_FILE]:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logging.info("Deleted %s to force refresh", f)
            except Exception as e:
                logging.warning("Could not delete %s: %s", f, e)
        new_file = download_recent_games()
        download_player_stats()
        if new_file:
            csv_file = new_file
            model    = build_model(csv_file)
            print(cok("Refresh complete -> games + players updated, model rebuilt"))
            print(cwarn("Run 'backtest' to refit the Platt calibration scaler."))
        else:
            print(cerr("Refresh failed - could not download recent games"))
    elif cmd == "backtest":
        _, metrics = backtest_model(csv_file, fit_platt=True)
        model._platt_scaler = load_platt_scaler()
        acc_s = cok("%.2f%%" % metrics.get("accuracy", 0))
        print("  Backtest accuracy: %s | LogLoss: %.4f | Brier: %.4f"
              % (acc_s, metrics.get("log_loss", 0), metrics.get("brier", 0)))
    elif cmd in ("platt", "calibrate"):
        if model._platt_scaler:
            ps = model._platt_scaler
            print("\n  Platt scaler: %s" % cok("ACTIVE"))
            print("  Fitted   : %s" % cdim(ps.get("fitted_at","?")))
            print("  Samples  : %s" % cok(ps.get("n_samples","?")))
            print("  Coef     : %.4f  Intercept: %.4f" % (ps.get("coef",0), ps.get("intercept",0)))
            print("  To refit : run %s" % chi("backtest"))
        else:
            print(cwarn("  No Platt scaler fitted yet. Run 'backtest' to fit one."))
    elif cmd == "grid":
        grid_search_optimization(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "genetic":
        genetic_optimization(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "results":
        show_optimization_results()
    elif cmd == "players":
        show_player_metrics(load_player_stats(), None, 10)
    elif cmd == "settings":
        model.show_settings()
    elif cmd == "live":
        show_live_scores_for_open_trades(model)
    elif cmd == "balance":
        show_balance()
    elif cmd in ("predicts","summary"):
        summarize_predict_lots()
    elif cmd == "chart":
        plot_pnl_chart()
    elif cmd == "resolve":
        resolve_predict_contracts()
    elif cmd == "sell":
        sell_predict_contract()
    elif cmd == "mark":
        mark_pending_positions()
    elif cmd == "invert":
        invert_open_trade()
    elif cmd in ("today","html","blog","blogger"):
        generate_today_predictions_html(model)
    elif cmd == "tomorrow":
        generate_tomorrow_predictions_html(model)
    elif cmd in ("autoresolve on","autoresolve off"):
        enabled  = cmd.endswith("on")
        settings = load_elo_settings()
        settings["autoresolve_enabled"] = enabled
        save_elo_settings(settings)
        print(cok("  Auto-resolve is now %s. Saved to %s." % ("ENABLED" if enabled else "DISABLED", SETTINGS_FILE)))
    elif cmd == "autoresolve":
        count = auto_resolve_finished_trades(model, verbose=True)
        if count == 0:
            print(cwarn("  No lots auto-resolved (no matching final games found in today's feed)."))
        else:
            print(cok("  Auto-resolved %d lot(s)." % count))
    elif cmd == "enhanced":
        print(chi("  Running enhanced backtest (Elo + XGBoost ensemble)..."))
        result = run_enhanced_backtest(csv_file, elo_weight=0.8, label="enhanced")
        if result and result.get("xgb_model"):
            save_enhanced_model(result)
            print(cok("  Enhanced model saved."))
    elif cmd == "enhanced decay":
        print(chi("  Running enhanced backtest with TIME-DECAYED weighting..."))
        result = run_enhanced_backtest(csv_file, elo_weight=0.8, label="enhanced-decay", time_decay=True)
        if result and result.get("xgb_model"):
            save_enhanced_model(result)
            print(cok("  Enhanced model saved (time-decayed)."))
    elif cmd == "shap":
        shap_feature_importance()
    elif cmd == "bayesian":
        bayesian_optimization(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "purgedcv":
        purged_walk_forward_cv(csv_file)
    elif cmd == "cpcv":
        combinatorial_purged_cv(csv_file)
    elif cmd == "pbo":
        probability_of_backtest_overfitting()
    elif cmd == "montecarlo":
        monte_carlo_permutation_test(csv_file)
    elif cmd == "rollingcal":
        rolling_origin_recalibration(csv_file)
    elif cmd == "kelly":
        kelly_criterion_backtest()
    elif cmd == "sliding":
        sliding_window_backtest(csv_file)
    elif cmd == "convergence":
        elo_convergence_analysis(csv_file)
    elif cmd == "conformal":
        conformal_analysis(csv_file)
    elif cmd == "betacal":
        run_beta_calibration(csv_file)
    elif cmd in ("autoopt", "auto-optimize", "auto optimize"):
        auto_optimize(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd in ("superopt", "super-optimize", "super optimize", "super"):
        super_optimize(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd in ("singleopt", "single-opt", "single opt", "coorddescent", "coord"):
        run_coordinate_descent(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "injuries":
        show_injury_report(model)
    elif cmd.startswith("injuries set "):
        # injuries set <team> <player1>, <player2>
        rest = cmd[len("injuries set "):].strip()
        parts = rest.split(None, 1)
        if len(parts) >= 2:
            team = model.find_team(parts[0])
            if team:
                players = [p.strip() for p in parts[1].split(",")]
                manual_set_injuries(team, players)
                print(cok("  Marked %d player(s) OUT for %s: %s" % (len(players), team, ", ".join(players))))
            else:
                print(cerr("  Team not found: %s" % parts[0]))
        else:
            print(cwarn("  Usage: injuries set <team> <player1>, <player2>"))
    elif cmd.startswith("set "):
        try:
            from elo_set_handler import handle_elo_set, print_elo_settings
            args = cmd[4:].strip()
            if not args or args in ("help", "?", "list"):
                print_elo_settings(model, load_elo_settings)
            else:
                ok, msg = handle_elo_set(args, model, load_elo_settings,
                                         lambda s: save_elo_settings(s))
                if ok:
                    print(cok("  %s" % msg))
                    print(cwarn("  Tip: run 'backtest' to refit Platt scaler with new params."))
                else:
                    print(cerr("  %s" % msg))
        except ImportError:
            print(cerr("  Set handler not available. Check elo_set_handler.py"))
    # ── Mega-ensemble commands ─────────────────────────────────────────
    elif cmd in ("mega", "megabacktest", "mega backtest"):
        if HAS_MEGA_BACKTEST:
            from elo_model import MLBElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Running mega-ensemble backtest (12+ models)..."))
            print(cdim("  This may take a few minutes."))
            result = run_mega_backtest(
                csv_file, sport="mlb",
                elo_model_class=MLBElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
            if result:
                print(cok("\n  Mega-ensemble: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (result["accuracy"], result["log_loss"], result["brier"])))
        else:
            print(cerr("Mega backtest not available. Check imports."))
    elif cmd in ("mega optimize", "megaopt", "mega opt", "megaoptimize"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import MLBElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Starting mega-ensemble optimization (all phases)..."))
            print(cdim("  This will take a LONG time. No shortcuts."))
            print()
            result = run_deep_optimize(
                csv_file, sport="mlb",
                elo_model_class=MLBElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
            if result and result.get("best_results"):
                br = result["best_results"]
                print(cok("\n  Optimized: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (br["accuracy"], br["log_loss"], br["brier"])))
                print(cok("  Settings saved. Run 'mega' to use them."))
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd in ("mega quick", "megaquick"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import MLBElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Quick mega optimization (grid search only)..."))
            result = run_quick_optimize(
                csv_file, sport="mlb",
                elo_model_class=MLBElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
            if result and result.get("best_results"):
                br = result["best_results"]
                print(cok("\n  Optimized: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (br["accuracy"], br["log_loss"], br["brier"])))
                print(cok("  Settings saved. Run 'mega' to use them."))
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd in ("mega ablation", "mega ablate", "mega single", "megasingle"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import MLBElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Running single-model ablation study..."))
            print(cdim("  Tests each model's individual contribution."))
            print()
            run_single_model_optimize(
                csv_file, sport="mlb",
                elo_model_class=MLBElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd.startswith("mega tune"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import MLBElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Per-model solo optimization (Phase 0 + 1)..."))
            result = run_quick_optimize(
                csv_file, sport="mlb",
                elo_model_class=MLBElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
            if result and result.get("best_results"):
                br = result["best_results"]
                print(cok("\n  Tuned: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (br["accuracy"], br["log_loss"], br["brier"])))
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd in ("mega tournament", "mega tourney"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import MLBElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Head-to-head model tournament (Phase 2)..."))
            result = run_mega_optimize(
                csv_file, sport="mlb",
                elo_model_class=MLBElo,
                elo_settings=settings,
                player_df=player_df_fresh,
                phases=[2],
            )
            if result and result.get("best_results"):
                br = result["best_results"]
                print(cok("\n  Tournament winner: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (br["accuracy"], br["log_loss"], br["brier"])))
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd in ("mega models", "mega status", "models"):
        if HAS_MEGA_CONFIG:
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            switches = load_model_switches("mlb", sport_dir)
            print_model_status(switches)
        else:
            print(cerr("Mega config not available."))
    elif cmd.startswith("mega on ") or cmd.startswith("mega enable "):
        if HAS_MEGA_CONFIG:
            model_name = cmd.split()[-1].strip().lower()
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            switches = load_model_switches("mlb", sport_dir)
            if model_name in ALL_MODELS:
                switches[model_name] = True
                save_model_switches("mlb", sport_dir, switches)
                print(cok("  Enabled: %s" % model_name))
            elif model_name == "all":
                for m in ALL_MODELS:
                    switches[m] = True
                save_model_switches("mlb", sport_dir, switches)
                print(cok("  All models enabled"))
            else:
                print(cerr("  Unknown model: %s" % model_name))
                print(cdim("  Available: %s" % ", ".join(ALL_MODELS)))
        else:
            print(cerr("Mega config not available."))
    elif cmd.startswith("mega set "):
        if HAS_MEGA_CONFIG:
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            args = cmd[9:]  # everything after "mega set "
            ok, msg = handle_mega_set(args, "mlb", sport_dir)
            print(("  " + cok(msg)) if ok else ("  " + cerr(msg)))
            if ok:
                print(cwarn("  Tip: run 'mega' to use new settings."))
        else:
            print(cerr("Mega config not available."))
    elif cmd in ("mega settings", "mega params", "mega config"):
        if HAS_MEGA_CONFIG:
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            print_mega_settings("mlb", sport_dir)
        else:
            print(cerr("Mega config not available."))
    elif cmd.startswith("mega off ") or cmd.startswith("mega disable "):
        if HAS_MEGA_CONFIG:
            model_name = cmd.split()[-1].strip().lower()
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            switches = load_model_switches("mlb", sport_dir)
            if model_name in ALL_MODELS:
                switches[model_name] = False
                save_model_switches("mlb", sport_dir, switches)
                print(cwarn("  Disabled: %s" % model_name))
            else:
                print(cerr("  Unknown model: %s" % model_name))
        else:
            print(cerr("Mega config not available."))
    # ── Phase 1: Mega-ensemble data commands ──────────────────────────
    elif cmd == "odds":
        if HAS_MEGA_DATA:
            show_odds_table("mlb")
        else:
            print(cerr("Odds module not available. pip install requests"))
    elif cmd == "kalshi":
        if HAS_KALSHI:
            show_kalshi_odds("mlb")
        else:
            print(cerr("Kalshi module not available. Check kalshi.py"))
    elif cmd == "weather":
        if HAS_MEGA_DATA:
            team = input(chi("  Home team: ")).strip()
            if team:
                found = model.find_team(team)
                if found:
                    show_weather_report(found, "mlb")
                else:
                    print(cerr("Team not found: %s" % team))
        else:
            print(cerr("Weather module not available."))
    elif cmd in ("advstats", "statcast"):
        if HAS_ADVANCED_STATS:
            print(cdim("  Downloading MLB advanced stats from FanGraphs (may take a minute)..."))
            team_stats, pitcher_df = load_or_download_advanced()
            show_adv_rankings(team_stats)
        else:
            print(cerr("Advanced stats module not available. pip install pybaseball"))
    elif cmd == "sentiment":
        print(cerr("Sentiment module has been removed."))
    elif cmd.startswith("help"):
        parts = cmd.split(None, 1)
        show_help(parts[1] if len(parts) > 1 else "")
    else:
        print(cwarn("Unknown command: '%s'" % cmd))
    return model, csv_file


def main():
    print(Back.GREEN + Fore.BLACK + Style.BRIGHT
          + "  MLB MONEYBALL  [v4 - 31-Model Mega-Ensemble]  "
          + Style.RESET_ALL)
    div(80)
    print("""
WORKFLOW:
1. Run %s FIRST -> fits calibration scaler (better probabilities)
2. Type team name (e.g. Yankees) -> opponent -> home? (a/b/n)?
3. See calibrated prediction + key player metrics
4. Type 'y' to log moneyline contract  |  'resolve' after game
5. Run %s for 31-model ensemble  |  %s to find best settings

PREDICTIONS: <teamname> | today | tomorrow | all | players | injuries | settings

DATA:        refresh | odds | kalshi | weather | advstats | statcast

BACKTEST:    backtest | enhanced | enhanced decay | shap

ELO OPT:    grid | genetic | bayesian | autoopt | superopt | singleopt | results

VALIDATE:    purgedcv | cpcv | pbo | montecarlo | convergence | sliding
             rollingcal | conformal | betacal | kelly

MEGA (31 models):
  mega                Run mega-ensemble backtest (26+ models)
  mega optimize       7-phase per-model optimization (best results at all cost)
  mega quick          Baseline + per-model solo tuning (Phases 0-1)
  mega tune           Per-model solo optimization (same as mega quick)
  mega tournament     Head-to-head model tournament (Phase 2)
  mega ablation       Test each model's contribution, auto-prune bad ones
  mega models         Show all 31 models with ON/OFF status
  mega on/off <model> Enable/disable individual models
  mega settings       Show all mega parameter values
  mega set adj=0.10   Set mega parameter (adj, meta, retrain, pn, mn, etc.)

SETTINGS (39 Elo params, type 'set' to see all):
  set k=4 | set home=24 | set starter=30 | set rest=10 | set b2b=75
  set travel=8 | set pace=19 | set parkfactor=0 | set interleague=0.5
  set kelly=quarter | set balance=1000 | set autoresolve=true

TRADING:     predicts | balance | resolve | sell | mark | invert | chart
             live | autoresolve | autoresolve on/off

             help [command] | quit
""" % (chi("backtest"), chi("mega"), chi("mega optimize")))

    csv_file = download_recent_games()
    download_player_stats()
    download_advanced_stats()
    injuries = fetch_injury_report()
    if injuries:
        out_count = sum(1 for i in injuries if i.get("status") in ("Out", "Doubtful", "60-Day IL", "15-Day IL"))
        print(cdim("  Injuries loaded: %d players (%d Out/IL)" % (len(injuries), out_count)))
    if not csv_file:
        logging.error("Could not obtain game data. Exiting.")
        return

    model = build_model(csv_file)

    print("\n" + chi("BASELINE PERFORMANCE:"))
    _, baseline = backtest_model(csv_file, model=model, fit_platt=True)
    model._platt_scaler = load_platt_scaler()
    acc_s = cok("%.2f%%" % baseline.get("accuracy", 0))
    print("  Baseline accuracy: %s | LogLoss: %.4f | Brier: %.4f"
          % (acc_s, baseline.get("log_loss", 0), baseline.get("brier", 0)))

    if load_elo_settings().get("autoresolve_enabled", False):
        auto_resolve_finished_trades(model, verbose=True)

    settings = load_elo_settings()
    if float(settings.get("starting_balance", 0)) <= 0:
        prompt_balance()

    KNOWN_CMDS = {
        "all","refresh","backtest","enhanced","enhanced decay","grid","genetic",
        "bayesian","results","players","settings","predicts","summary","chart",
        "resolve","sell","mark","invert","live","help","autoresolve","balance",
        "autoresolve on","autoresolve off","today","html","blog","blogger",
        "tomorrow","platt","calibrate","injuries","shap","purgedcv","cpcv",
        "pbo","montecarlo","rollingcal","kelly","sliding","convergence",
        "conformal","betacal","autoopt","auto-optimize","auto optimize",
        "superopt","super-optimize","super optimize","super",
        "singleopt","single-opt","single opt","coorddescent","coord",
        "kalshi",
    }

    while True:
        try:
            cmd = input("\n%s " % cok(">")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("quit","q"):
            model.metadata["settings"] = model.settings_dict()
            model.save()
            save_elo_settings(model.settings_dict())
            break

        if cmd in ("today","html","blog","blogger"):
            print("\n" + chi("Generating today's MLB predictions HTML..."))
            generate_today_predictions_html(model)
            continue

        if cmd == "tomorrow":
            print("\n" + chi("Generating tomorrow's MLB predictions HTML..."))
            generate_tomorrow_predictions_html(model)
            continue

        if cmd not in KNOWN_CMDS and not cmd.startswith("set ") and cmd.strip():
            team_a = model.find_team(cmd)
            if team_a:
                team_b_input = input("Opponent team: ").strip()
                team_b       = model.find_team(team_b_input)
                if not team_b:
                    print(cerr("Team not found: '%s'" % team_b_input))
                    continue
                home_input = input(
                    "Home team? (a = first team home, b = second, n = neutral): "
                ).strip().lower()
                if home_input == "a":
                    team_a_home, neutral = True, False
                elif home_input == "b":
                    team_a_home, neutral = False, False
                else:
                    team_a_home, neutral = None, True
                winner, prob = model.pick_winner(
                    team_a, team_b, team_a_home=team_a_home, neutral_site=neutral,
                )
                cal_label = cdim(" (calibrated)") if model._platt_scaler else ""
                prob_s    = cok("%.1f%%" % (prob * 100))
                print("\n   %s - %s win probability%s" % (cok(winner), prob_s, cal_label))
                print("    %s Elo: %s  |  %s Elo: %s"
                      % (chi(team_a), cok("%.0f" % model.ratings[team_a]),
                         chi(team_b), cok("%.0f" % model.ratings[team_b])))
                if neutral:
                    site_s = "Neutral"
                elif team_a_home:
                    site_s = "%s home" % chi(team_a)
                else:
                    site_s = "%s home" % chi(team_b)
                print("    Site: %s" % site_s)

                # Show injury impact if any
                for t in (team_a, team_b):
                    out = get_team_injuries(t)
                    if out:
                        impact = calc_injury_impact(t, out)
                        print("    %s injuries: %s %s" % (
                            cwarn(t), ", ".join(out),
                            cwarn("(%.0f Elo)" % impact)))

                player_df = load_player_stats()
                if not player_df.empty:
                    print("\n  " + cbold("KEY PLAYERS (season stats):"))
                    show_player_metrics(player_df, team_a, 3)
                    show_player_metrics(player_df, team_b, 3)
                kelly_contracts = 0
                market_cents = None
                # Auto-fetch from Kalshi if enabled
                settings_now = load_elo_settings()
                if HAS_KALSHI and settings_now.get("auto_kalshi"):
                    h_team = team_a if team_a_home else team_b
                    a_team = team_b if team_a_home else team_a
                    kalshi = find_kalshi_odds(h_team, a_team, "mlb")
                    if kalshi and kalshi.get("midpoint"):
                        bid = kalshi.get("home_yes_bid")
                        ask = kalshi.get("home_yes_ask")
                        mid = kalshi["midpoint"]
                        # If our predicted winner is the AWAY team, flip to away price
                        if winner == a_team:
                            bid = (100 - ask) if ask else None
                            ask = (100 - (kalshi.get("home_yes_bid") or 0)) if kalshi.get("home_yes_bid") else None
                            mid = (100 - kalshi["midpoint"])
                        bid_s = "%d\u00a2" % bid if bid else "?"
                        ask_s = "%d\u00a2" % ask if ask else "?"
                        print("\n   %s  Kalshi: %s bid / %s ask  (mid %d\u00a2)"
                              % (cok("KALSHI"), chi(bid_s), chi(ask_s), mid))
                        market_cents = float(mid)
                    else:
                        print(cdim("\n   Kalshi: no matching market found, enter manually"))
                if market_cents is None:
                    odds_input = input(
                        "\nActual trade odds in cents (e.g. 62 for $0.62, or Enter to skip): "
                    ).strip()
                    if odds_input:
                        try:
                            market_cents = float(odds_input)
                            if not (1 <= market_cents <= 99):
                                print(cwarn("  Odds must be between 1-99 cents."))
                                market_cents = None
                        except ValueError:
                            print(cwarn("  Couldn't parse odds input."))
                if market_cents and 1 <= market_cents <= 99:
                    kelly_contracts = show_kelly_recommendation(prob, market_cents)
                log_choice = input(
                    "\nLog this moneyline pick as a Predicts position? (y/n): "
                ).strip().lower()
                if log_choice == "y":
                    h = team_a if team_a_home else team_b
                    a = team_b if team_a_home else team_a
                    add_predict_contract(h, a, winner, prob, kelly_contracts)
                    try:
                        df = load_predict_lots()
                        if not df.empty:
                            last_idx   = df.index[-1]
                            clean_note = "Model %.1f%% - %s" % (prob * 100, winner)
                            if model._platt_scaler:
                                clean_note += " (calibrated)"
                            df.loc[last_idx, "notes"] = clean_note
                            save_predict_lots(df)
                            print(cdim("  Note updated: %s" % clean_note))
                    except Exception as e:
                        print(cwarn("  Could not update note: %s" % e))
                continue

        model, csv_file = dispatch(cmd, model, csv_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR: %s" % e)
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        pass
