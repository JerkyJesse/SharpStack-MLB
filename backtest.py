"""Backtest, grid search, genetic optimization, and advanced validation methods."""

import os
import logging
from itertools import product, combinations
from collections import defaultdict
from math import comb as math_comb

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm as norm_dist

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from color_helpers import cok, cwarn, cdim, chi, div
from elo_model import MLBElo
from data_players import load_player_stats, build_league_player_scores
from metrics import (log_loss_binary, brier_score_binary, calibration_table,
                     ece_score, mce_score, brier_skill_score,
                     conformal_prediction_set)
from platt import (fit_platt_scaler, save_platt_scaler, load_platt_scaler,
                   apply_platt, regress_ratings_to_mean)
from build_model import _calc_altitude_bonus

# All Elo constructor keys
_ELO_KEYS = {"base_rating", "k", "home_adv", "use_mov", "player_boost",
             "starter_boost", "rest_factor", "form_weight", "travel_factor",
             "sos_factor", "playoff_hca_factor", "pace_factor",
             "division_factor", "mean_reversion",
             "pyth_factor", "home_road_factor", "mov_base",
             "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
             "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
             "rest_advantage_cap", "park_factor_weight",
             "mov_cap", "east_travel_penalty", "series_adaptation",
             "interleague_factor", "bullpen_factor", "opp_pitcher_factor",
             "k_decay", "surprise_k", "elo_scale"}


def backtest_model(csv_file=GAMES_FILE, output_csv="mlb_backtest_predictions.csv",
                   calibration_csv="mlb_calibration.csv", k=None, home_adv=None,
                   model=None, fit_platt=False):
    """
    fit_platt=True: fit and save Platt scaler from this run's raw probs.
    Only pass fit_platt=True on the main/user-facing backtest call,
    not inside optimizer loops (leakage + speed).
    """
    if not os.path.exists(csv_file):
        return False, {}
    if model is None:
        settings = load_elo_settings()
        model = MLBElo(**{k_: v for k_, v in settings.items() if k_ in _ELO_KEYS})
        if k is not None:
            model.k = k
        if home_adv is not None:
            model.home_adv = home_adv
        model._altitude_bonus = _calc_altitude_bonus(csv_file)
        player_df = load_player_stats()
        if not player_df.empty:
            model.set_player_stats(player_df)
    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    # Parse dates for rest-day calculations
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")
    # Detect season boundaries for pitcher rating regression
    prev_season = None
    predictions, probs, actuals = [], [], []
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            # Regress pitcher ratings at season boundaries
            if game_date is not None:
                row_season = game_date.year
                if prev_season is not None and row_season != prev_season:
                    model.regress_pitcher_ratings(factor=0.5)
                prev_season = row_season
            home_starter = str(row.get("home_starter", "") or "").strip()
            away_starter = str(row.get("away_starter", "") or "").strip()
            home_win_prob = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True, neutral_site=bool(row["neutral_site"]),
                calibrated=False, game_date=game_date, use_injuries=False,
                home_starter=home_starter, away_starter=away_starter,
            )
            pred_winner   = row["home_team"] if home_win_prob >= 0.5 else row["away_team"]
            home_actual   = 1 if row["home_score"] > row["away_score"] else 0
            actual_winner = row["home_team"] if home_actual == 1 else row["away_team"]
            predictions.append({
                "date": row["date"], "home_team": row["home_team"],
                "away_team": row["away_team"], "home_score": row["home_score"],
                "away_score": row["away_score"], "neutral_site": bool(row["neutral_site"]),
                "home_win_prob": home_win_prob, "pred_winner": pred_winner,
                "actual_winner": actual_winner, "correct": int(pred_winner == actual_winner),
            })
            probs.append(home_win_prob)
            actuals.append(home_actual)
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]),
                game_date=game_date,
                home_starter=home_starter, away_starter=away_starter,
            )
        except Exception as e:
            logging.warning("Backtest row error: %s", e)
    if not predictions:
        return False, {}
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_csv, index=False)
    calibration_table(probs, actuals).to_csv(calibration_csv, index=False)

    if fit_platt and len(probs) >= 100:
        scaler     = fit_platt_scaler(probs, actuals)
        save_platt_scaler(scaler)
        model._platt_scaler = scaler
        cal_probs  = [apply_platt(p, scaler) for p in probs]
        brier_raw  = brier_score_binary(actuals, probs)
        brier_cal  = brier_score_binary(actuals, cal_probs)
        ll_cal     = log_loss_binary(actuals, cal_probs)
        raw_s      = cwarn("%.4f" % brier_raw)
        cal_s      = cok("%.4f" % brier_cal)
        print("\n  %s" % chi("Platt scaler fitted and saved."))
        print("  Brier (raw):        %s" % raw_s)
        print("  Brier (calibrated): %s  <- use this going forward" % cal_s)
        print("  LogLoss (cal):      %.4f" % ll_cal)
        # ECE, MCE, BSS
        ece = ece_score(cal_probs, actuals)
        mce = mce_score(cal_probs, actuals)
        home_rate = float(np.mean(actuals))
        bss_50 = brier_skill_score(cal_probs, actuals)
        bss_home = brier_skill_score(cal_probs, actuals, [home_rate] * len(actuals))
        print("  ECE:                %.4f%s" % (ece, "  (excellent)" if ece < 0.03 else "  (needs work)" if ece > 0.08 else ""))
        print("  MCE:                %.4f" % mce)
        print("  BSS vs 50%%:         %.4f  |  BSS vs home-rate (%.0f%%): %.4f"
              % (bss_50, home_rate * 100, bss_home))
    else:
        brier_cal = brier_score_binary(actuals, probs)
        ll_cal    = log_loss_binary(actuals, probs)

    metrics = {
        "accuracy": float(pred_df["correct"].mean() * 100.0),
        "log_loss": ll_cal,
        "brier":    brier_cal,
        "n_games":  int(len(pred_df)),
    }
    return True, metrics


_OPT_KEYS = ("k", "home_adv", "player_boost", "starter_boost", "rest_factor",
             "form_weight", "travel_factor", "sos_factor", "playoff_hca_factor",
             "pace_factor", "division_factor", "mean_reversion",
             "pyth_factor", "home_road_factor", "mov_base",
             "b2b_penalty", "road_trip_factor", "homestand_factor",
             "win_streak_factor", "altitude_factor", "season_phase_factor",
             "scoring_consistency_factor", "rest_advantage_cap",
             "park_factor_weight", "mov_cap", "east_travel_penalty",
             "series_adaptation", "interleague_factor",
             "bullpen_factor", "opp_pitcher_factor",
             "k_decay", "surprise_k")

def _apply_best_settings(best_params, csv_file):
    """Save best params to settings, rebuild model, refit Platt."""
    settings = load_elo_settings()
    for key in _OPT_KEYS:
        if key in best_params:
            settings[key] = best_params[key]
    save_elo_settings(settings)

    # Refit Platt scaler with the winning params
    model = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()
    if not player_df.empty:
        model.set_player_stats(player_df)
    _, metrics = backtest_model(csv_file, model=model, fit_platt=True)

    print("\n  %s" % cok("Best settings saved to mlb_elo_settings.json"))
    core = "  K=%.1f  HomeAdv=%.1f  PBoost=%.1f  SB=%.1f  Rest=%.1f  FW=%.1f  Travel=%.1f  SOS=%.1f  PHCA=%.2f  Pace=%.1f" % (
        settings.get("k", 0), settings.get("home_adv", 0),
        settings.get("player_boost", 0), settings.get("starter_boost", 0),
        settings.get("rest_factor", 0), settings.get("form_weight", 0),
        settings.get("travel_factor", 0), settings.get("sos_factor", 0),
        settings.get("playoff_hca_factor", 1.0), settings.get("pace_factor", 0))
    print(core)
    extras = []
    for key in _OPT_KEYS[10:]:
        val = settings.get(key, 0)
        if isinstance(val, (int, float)) and abs(val) > 0.001:
            extras.append("%s=%.1f" % (key[:8], val))
    if extras:
        print("  + " + "  ".join(extras))
    return settings


def grid_search_optimization(csv_file=GAMES_FILE, output_file="mlb_grid_search.csv"):
    logging.info("Running GRID SEARCH...")
    settings = load_elo_settings()
    base     = settings.get("base_rating", 1500.0)
    use_mov  = settings.get("use_mov", True)

    print("\nGRID SEARCH RANGES  (press Enter to use defaults)")
    print(cdim("  Objective: minimize Brier + LogLoss  (calibration-focused)"))
    div(48)

    def _prompt_range(label, default_lo, default_hi, default_step):
        try:
            lo   = input("  %s min   [%s]: " % (label, default_lo)).strip()
            hi   = input("  %s max   [%s]: " % (label, default_hi)).strip()
            step = input("  %s step  [%s]: " % (label, default_step)).strip()
            lo   = float(lo)   if lo   else default_lo
            hi   = float(hi)   if hi   else default_hi
            step = float(step) if step else default_step
        except ValueError:
            print("  Invalid input - using defaults for %s" % label)
            lo, hi, step = default_lo, default_hi, default_step
        return np.arange(lo, hi + step * 0.001, step)

    k_values            = _prompt_range("K",           10, 50,  4)
    print()
    home_adv_values     = _prompt_range("HomeAdv",     20, 80,  5)
    print()
    player_boost_values = _prompt_range("PlayerBoost",  0, 30,  5)
    print()
    rest_factor_values  = _prompt_range("RestFactor",   0, 30,  5)
    print()
    travel_values       = _prompt_range("TravelFactor", 0, 50, 10)
    print()
    pace_values         = _prompt_range("PaceFactor",   0, 50, 10)
    print()
    playoff_hca_values  = _prompt_range("PlayoffHCA",   0.4, 1.0, 0.2)
    print()

    player_df   = load_player_stats()
    has_players = not player_df.empty
    _prebuilt_scores = build_league_player_scores(player_df) if has_players else {}
    _alt_bonus = _calc_altitude_bonus(csv_file)

    results    = []
    best_score = -1e9
    best_params = None
    total = (len(k_values) * len(home_adv_values) * len(player_boost_values)
             * len(rest_factor_values) * len(travel_values) * len(pace_values)
             * len(playoff_hca_values))

    div(120)
    print("  GRID SEARCH - %d combos   |   Objective: minimize Brier + LogLoss" % total)
    div(120)

    for i, (k, home_adv, player_boost, rest_factor, travel, pace, phca) in enumerate(
        product(k_values, home_adv_values, player_boost_values, rest_factor_values,
                travel_values, pace_values, playoff_hca_values), 1
    ):
        fresh_model = MLBElo(
            base_rating=base, k=float(k), home_adv=float(home_adv),
            use_mov=use_mov, player_boost=float(player_boost),
            rest_factor=float(rest_factor), travel_factor=float(travel),
            pace_factor=float(pace), playoff_hca_factor=float(phca),
        )
        fresh_model._altitude_bonus = _alt_bonus
        if has_players:
            fresh_model._player_scores = _prebuilt_scores
        success, metrics = backtest_model(
            csv_file, "temp_backtest.csv", "temp_cal.csv", model=fresh_model,
        )
        if not success:
            continue
        score = -(metrics["log_loss"] * 8.0 + metrics["brier"] * 40.0)
        row   = {
            "k": float(k), "home_adv": float(home_adv),
            "player_boost": float(player_boost), "rest_factor": float(rest_factor),
            "travel_factor": float(travel), "pace_factor": float(pace),
            "playoff_hca_factor": float(phca),
            "accuracy": metrics["accuracy"], "log_loss": metrics["log_loss"],
            "brier": metrics["brier"], "score": score,
        }
        results.append(row)
        is_best = score > best_score
        if is_best:
            best_score  = score
            best_params = row
        flag = (" " + cok("<- BEST")) if is_best else ""
        if is_best or i % 50 == 0 or i == total:
            print("  %5d/%d  K=%.0f HA=%.0f PB=%.0f R=%.0f T=%.0f P=%.0f PHCA=%.1f  "
                  "Acc=%.2f%% LL=%.4f Br=%.4f%s"
                  % (i, total, k, home_adv, player_boost, rest_factor,
                     travel, pace, phca,
                     metrics["accuracy"], metrics["log_loss"], metrics["brier"], flag),
                  flush=True)

    div(120)
    for tmp in ["temp_backtest.csv", "temp_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
    if best_params:
        acc_s = cok("%.2f%%" % best_params["accuracy"])
        print("\n%s K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.1f"
              % (cok("* GRID BEST:"), best_params["k"], best_params["home_adv"],
                 best_params["player_boost"], best_params["rest_factor"],
                 best_params["travel_factor"], best_params["pace_factor"],
                 best_params["playoff_hca_factor"]))
        print("  Acc=%s  LogLoss=%.4f  Brier=%.4f"
              % (acc_s, best_params["log_loss"], best_params["brier"]))
        _apply_best_settings(best_params, csv_file)
    return best_params


def genetic_optimization(csv_file=GAMES_FILE, output_file="mlb_genetic_results.csv"):
    logging.info("Running GENETIC ALGORITHM...")
    settings    = load_elo_settings()
    base        = settings["base_rating"]
    use_mov     = settings.get("use_mov", True)
    player_df   = load_player_stats()
    has_players = not player_df.empty
    _prebuilt_scores = build_league_player_scores(player_df) if has_players else {}

    print("\nGENETIC OPTIMIZER SETTINGS  (press Enter to use defaults)")
    print(cdim("  Objective: minimize Brier + LogLoss  (calibration-focused)"))
    div(52)

    def _prompt_float(label, default):
        try:
            raw = input("  %s [%s]: " % (label, default)).strip()
            return float(raw) if raw else default
        except ValueError:
            return default

    def _prompt_bound(label, lo_default, hi_default):
        try:
            lo_raw = input("  %s min [%s]: " % (label, lo_default)).strip()
            hi_raw = input("  %s max [%s]: " % (label, hi_default)).strip()
            lo = float(lo_raw) if lo_raw else lo_default
            hi = float(hi_raw) if hi_raw else hi_default
        except ValueError:
            lo, hi = lo_default, hi_default
        return (lo, hi)

    k_bounds    = _prompt_bound("K",           5,  80)
    ha_bounds   = _prompt_bound("HomeAdv",     0, 150)
    pb_bounds   = _prompt_bound("PlayerBoost", 0,  50)
    rf_bounds   = _prompt_bound("RestFactor",  0,  50)
    tf_bounds   = _prompt_bound("TravelFactor",0,  60)
    pf_bounds   = _prompt_bound("PaceFactor",  0,  60)
    phca_bounds = _prompt_bound("PlayoffHCA",  0.0, 1.0)
    maxiter     = int(_prompt_float("Max generations (maxiter)", 80))
    popsize     = int(_prompt_float("Population size (popsize)", 40))
    print()

    bounds        = [k_bounds, ha_bounds, pb_bounds, rf_bounds,
                     tf_bounds, pf_bounds, phca_bounds]
    _gen_counter  = [0]
    _best_score   = [1e9]
    _best_params  = [None]
    _eval_counter = [0]
    _alt_bonus    = _calc_altitude_bonus(csv_file)

    div(120)
    print("  GENETIC OPTIMIZER - maxiter=%d  popsize=%d  (7 params)" % (maxiter, popsize))
    print("  K:%s  HA:%s  PB:%s  Rest:%s  Travel:%s  Pace:%s  PHCA:%s"
          % (k_bounds, ha_bounds, pb_bounds, rf_bounds,
             tf_bounds, pf_bounds, phca_bounds))
    div(120)
    print("  %4s  %6s  %10s  %5s %5s %4s %4s %4s %4s %4s"
          % (chi("Gen"), chi("Evals"), chi("BestScore"),
             chi("K"), chi("HA"), chi("PB"), chi("R"),
             chi("T"), chi("P"), chi("PHCA")))
    div(90)

    def _progress_callback(xk, convergence):
        _gen_counter[0] += 1
        if _best_params[0] is not None:
            p = _best_params[0]
            best_s = cok("%10.4f" % (-_best_score[0]))
            print("  %4d  %6d  %s  %5.1f %5.1f %4.1f %4.1f %4.1f %4.1f %4.1f"
                  % (_gen_counter[0], _eval_counter[0], best_s,
                     p[0], p[1], p[2], p[3], p[4], p[5], p[6]),
                  flush=True)

    def objective(params):
        k, home_adv, player_boost, rest_factor, travel, pace, phca = params
        fresh_model = MLBElo(
            base_rating=base, k=float(k), home_adv=float(home_adv),
            use_mov=use_mov, player_boost=float(player_boost),
            rest_factor=float(rest_factor), travel_factor=float(travel),
            pace_factor=float(pace), playoff_hca_factor=float(phca),
        )
        fresh_model._altitude_bonus = _alt_bonus
        if has_players:
            fresh_model._player_scores = _prebuilt_scores
        success, metrics = backtest_model(
            csv_file, "temp_genetic.csv", "temp_genetic_cal.csv", model=fresh_model,
        )
        if not success:
            return 1e9
        score = metrics["log_loss"] * 8.0 + metrics["brier"] * 40.0
        _eval_counter[0] += 1
        if score < _best_score[0]:
            _best_score[0]  = score
            _best_params[0] = tuple(float(x) for x in params)
        return score

    result = differential_evolution(
        objective, bounds=bounds, maxiter=maxiter, popsize=popsize,
        workers=1, polish=True, callback=_progress_callback,
    )
    best_k, best_home, best_pb, best_rf, best_tf, best_pf, best_phca = result.x
    div(90)
    div(120)
    print("%s  K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f"
          % (cok("* GENETIC BEST:"), best_k, best_home, best_pb, best_rf,
             best_tf, best_pf, best_phca))
    print("  Convergence: %s  |  Total evaluations: %d" % (result.message, _eval_counter[0]))
    div(120)

    # Final backtest with best params
    fresh = MLBElo(
        base_rating=base, k=float(best_k), home_adv=float(best_home),
        use_mov=use_mov, player_boost=float(best_pb),
        rest_factor=float(best_rf), travel_factor=float(best_tf),
        pace_factor=float(best_pf), playoff_hca_factor=float(best_phca),
    )
    fresh._altitude_bonus = _alt_bonus
    if has_players:
        fresh._player_scores = _prebuilt_scores
    success, metrics = backtest_model(
        csv_file, "temp_genetic.csv", "temp_genetic_cal.csv", model=fresh,
    )
    row = {
        "k": float(best_k), "home_adv": float(best_home),
        "player_boost": float(best_pb), "rest_factor": float(best_rf),
        "travel_factor": float(best_tf), "pace_factor": float(best_pf),
        "playoff_hca_factor": float(best_phca),
        "accuracy": metrics.get("accuracy", np.nan),
        "log_loss": metrics.get("log_loss",  np.nan),
        "brier":    metrics.get("brier",     np.nan),
        "score":    -(metrics.get("log_loss",0)*8.0 + metrics.get("brier",0)*40.0),
    }
    pd.DataFrame([row]).to_csv(output_file, index=False)
    for tmp in ["temp_genetic.csv", "temp_genetic_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass
    acc_s = cok("%.2f%%" % row["accuracy"])
    print("  FINAL METRICS:  Acc=%s  LogLoss=%.4f  Brier=%.4f  Score=%.4f"
          % (acc_s, row["log_loss"], row["brier"], row["score"]))

    # Auto-save best settings and refit Platt
    _apply_best_settings(row, csv_file)
    return row


def show_optimization_results():
    result_files = [
        "mlb_grid_search.csv", "mlb_genetic_results.csv", "mlb_bayesian_results.csv",
        "mlb_super_grid.csv", "mlb_super_fine_grid.csv",
    ]
    found_any = False
    for f in result_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            if not df.empty and "score" in df.columns:
                found_any = True
                best  = df.loc[df["score"].idxmax()]
                extras = ""
                for key, label in [("player_boost","PB"), ("rest_factor","R"),
                                   ("travel_factor","T"), ("pace_factor","P"),
                                   ("playoff_hca_factor","PHCA"),
                                   ("sos_factor","SOS"), ("form_weight","FW")]:
                    if key in best and pd.notna(best[key]):
                        extras += ", %s=%.1f" % (label, best[key])
                acc_s = cok("%.2f%%" % best["accuracy"]) if "accuracy" in best else "?"
                ll_s = "%.4f" % best["log_loss"] if "log_loss" in best else "?"
                br_s = "%.4f" % best["brier"] if "brier" in best else "?"
                print("  %s: BEST K=%.1f, HA=%.1f%s, Acc=%s, LL=%s, Br=%s"
                      % (chi(f), best.get("k"), best.get("home_adv"), extras,
                         acc_s, ll_s, br_s))
                print("    %d total trials" % len(df))
                # Deflated Sharpe Ratio
                dsr = deflated_sharpe_ratio(df["score"].values)
                if dsr is not None:
                    if dsr >= 1.96:
                        print("    DSR=%.2f %s (significant at 95%%)" % (dsr, cok("*")))
                    else:
                        print("    DSR=%.2f %s (not significant — best may be noise)"
                              % (dsr, cwarn("!")))
    if not found_any:
        print(cwarn("  No optimization results found. Run 'grid', 'genetic', 'bayesian', 'autoopt', or 'superopt' first."))


# -- Deflated Sharpe Ratio ------------------------------------------------

def deflated_sharpe_ratio(scores):
    """Adjust best optimization score for multiple-testing bias.
    DSR < 1.96 -> best result not significant at 95% confidence."""
    if scores is None or len(scores) < 5:
        return None
    scores = np.array(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n < 5:
        return None
    mu = np.mean(scores)
    sigma = np.std(scores, ddof=1)
    if sigma < 1e-12:
        return None
    sr = mu / sigma
    z = (scores - mu) / sigma
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0  # excess kurtosis
    # Expected max SR under null (EVT approximation)
    log_n = np.log(max(n, 2))
    sr_expected = np.sqrt(2 * log_n) - (np.log(np.pi) + np.log(log_n)) / (2 * np.sqrt(2 * log_n))
    # SE of SR (Lo 2002)
    se = np.sqrt((1 - skew * sr + ((kurt + 2) / 4) * sr ** 2) / max(n - 1, 1))
    if se < 1e-12:
        return None
    return float((sr - sr_expected) / se)


# -- Bayesian Hyperparameter Optimization ---------------------------------

def bayesian_optimization(csv_file=GAMES_FILE, output_file="mlb_bayesian_results.csv"):
    """GP surrogate + Expected Improvement — more sample-efficient than grid search."""
    from scipy.stats.qmc import LatinHypercube

    settings = load_elo_settings()
    base = settings.get("base_rating", 1500.0)
    use_mov = settings.get("use_mov", True)
    player_df = load_player_stats()
    has_players = not player_df.empty
    _prebuilt_scores = build_league_player_scores(player_df) if has_players else {}
    _alt_bonus = _calc_altitude_bonus(csv_file)

    print("\nBAYESIAN OPTIMIZER SETTINGS  (press Enter to use defaults)")
    print(cdim("  GP surrogate + Expected Improvement acquisition"))
    div(52)

    def _pf(label, default):
        try:
            raw = input("  %s [%s]: " % (label, default)).strip()
            return float(raw) if raw else default
        except ValueError:
            return default

    def _pb(label, lo_d, hi_d):
        try:
            lo = input("  %s min [%s]: " % (label, lo_d)).strip()
            hi = input("  %s max [%s]: " % (label, hi_d)).strip()
            return (float(lo) if lo else lo_d, float(hi) if hi else hi_d)
        except ValueError:
            return (lo_d, hi_d)

    bounds = [_pb("K", 5, 80), _pb("HomeAdv", 0, 150), _pb("PlayerBoost", 0, 50),
              _pb("RestFactor", 0, 50), _pb("TravelFactor", 0, 60),
              _pb("PaceFactor", 0, 60), _pb("PlayoffHCA", 0.0, 1.0)]
    n_initial = int(_pf("Initial samples", 20))
    n_iter = int(_pf("Optimization iterations", 60))
    print()

    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    ranges = highs - lows
    ranges[ranges < 1e-12] = 1.0  # prevent div by zero

    def objective(params):
        k, ha, pb, rf, tf, pf, phca = params
        m = MLBElo(base_rating=base, k=float(k), home_adv=float(ha), use_mov=use_mov,
                   player_boost=float(pb), rest_factor=float(rf), travel_factor=float(tf),
                   pace_factor=float(pf), playoff_hca_factor=float(phca))
        m._altitude_bonus = _alt_bonus
        if has_players:
            m._player_scores = _prebuilt_scores
        ok, met = backtest_model(csv_file, "temp_bayes.csv", "temp_bayes_cal.csv", model=m)
        if not ok:
            return 1e9
        return met["log_loss"] * 8.0 + met["brier"] * 40.0

    # Minimal GP surrogate
    class _GP:
        def __init__(self, ls=1.0, noise=1e-4):
            self.ls, self.noise = ls, noise
        def _kern(self, A, B):
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-0.5 * d / self.ls ** 2)
        def fit(self, X, y):
            self.X, self.y = X, y
            K = self._kern(X, X) + self.noise * np.eye(len(X))
            try:
                self.L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(len(X))
                self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        def predict(self, Xn):
            k = self._kern(Xn, self.X)
            mu = k @ self.alpha
            v = np.linalg.solve(self.L, k.T)
            var = 1.0 + self.noise - np.sum(v ** 2, axis=0)
            return mu, np.sqrt(np.maximum(var, 1e-12))

    # Latin Hypercube initial sampling
    sampler = LatinHypercube(d=7)
    X_init = lows + sampler.random(n=n_initial) * ranges

    div(120)
    print("  BAYESIAN OPTIMIZER - %d initial + %d iterations (7 params)" % (n_initial, n_iter))
    div(120)

    X_all, y_all = [], []
    best_score, best_params = 1e9, None

    print("  Evaluating %d initial samples..." % n_initial)
    for i, x in enumerate(X_init):
        s = objective(x)
        X_all.append(x)
        y_all.append(s)
        if s < best_score:
            best_score, best_params = s, x
            print("  [%d/%d] Score=%.4f %s" % (i + 1, n_initial, s, cok("<- BEST")), flush=True)
        elif (i + 1) % 5 == 0:
            print("  [%d/%d] evaluated" % (i + 1, n_initial), flush=True)

    gp = _GP()
    print("\n  Starting Bayesian optimization loop...")
    for it in range(n_iter):
        Xa = np.array(X_all)
        ya = np.array(y_all)
        X_norm = (Xa - lows) / ranges
        ym, ys = ya.mean(), max(ya.std(), 1e-8)
        y_norm = (ya - ym) / ys
        gp.fit(X_norm, y_norm)

        cands = np.random.rand(1000, 7)
        mu, sigma = gp.predict(cands)
        best_n = (best_score - ym) / ys
        z = (best_n - mu) / sigma
        ei = sigma * (z * norm_dist.cdf(z) + norm_dist.pdf(z))
        ei[sigma < 1e-10] = 0.0

        x_next = np.clip(lows + cands[np.argmax(ei)] * ranges, lows, highs)
        s = objective(x_next)
        X_all.append(x_next)
        y_all.append(s)
        is_best = s < best_score
        if is_best:
            best_score, best_params = s, x_next
        if is_best or (it + 1) % 10 == 0:
            flag = cok(" <- BEST") if is_best else ""
            print("  Iter %3d/%d  Score=%.4f  Best=%.4f%s"
                  % (it + 1, n_iter, s, best_score, flag), flush=True)

    for tmp in ["temp_bayes.csv", "temp_bayes_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    # Save results
    rows = []
    for x, s in zip(X_all, y_all):
        if s < 1e8:
            rows.append({"k": float(x[0]), "home_adv": float(x[1]),
                         "player_boost": float(x[2]), "rest_factor": float(x[3]),
                         "travel_factor": float(x[4]), "pace_factor": float(x[5]),
                         "playoff_hca_factor": float(x[6]), "score": -s})
    if rows:
        pd.DataFrame(rows).to_csv(output_file, index=False)

    div(120)
    if best_params is not None:
        bp = best_params
        m = MLBElo(base_rating=base, k=float(bp[0]), home_adv=float(bp[1]),
                   use_mov=use_mov, player_boost=float(bp[2]),
                   rest_factor=float(bp[3]), travel_factor=float(bp[4]),
                   pace_factor=float(bp[5]), playoff_hca_factor=float(bp[6]))
        m._altitude_bonus = _alt_bonus
        if has_players:
            m._player_scores = _prebuilt_scores
        _, met = backtest_model(csv_file, model=m)
        print("\n%s K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f"
              % (cok("* BAYESIAN BEST:"), bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6]))
        acc_s = cok("%.2f%%" % met.get("accuracy", 0))
        print("  Acc=%s  LogLoss=%.4f  Brier=%.4f  (%d evaluations)"
              % (acc_s, met.get("log_loss", 0), met.get("brier", 0), len(y_all)))
        _apply_best_settings({"k": float(bp[0]), "home_adv": float(bp[1]),
                              "player_boost": float(bp[2]), "rest_factor": float(bp[3]),
                              "travel_factor": float(bp[4]), "pace_factor": float(bp[5]),
                              "playoff_hca_factor": float(bp[6])}, csv_file)
    return best_params


# -- Purged Walk-Forward Cross-Validation ---------------------------------

def purged_walk_forward_cv(csv_file=GAMES_FILE, k_folds=5, embargo_games=5):
    """k-fold CV with embargo gap to prevent Elo momentum leakage between folds."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    games = pd.read_csv(csv_file)
    n = len(games)
    fold_size = n // k_folds

    print("\nPURGED WALK-FORWARD CROSS-VALIDATION")
    print(cdim("  %d folds, ~%d games/fold, %d embargo games" % (k_folds, fold_size, embargo_games)))
    div(80)

    player_df = load_player_stats()
    _alt_bonus = _calc_altitude_bonus(csv_file)
    fold_metrics = []

    for fold in range(k_folds):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        train_end = max(0, test_start - embargo_games)
        if train_end < 100:
            print("  Fold %d: skipping (insufficient training data)" % (fold + 1))
            continue

        train_games = games.iloc[:train_end]
        train_file = "temp_purged_train.csv"
        train_games.to_csv(train_file, index=False)

        model = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
        model._altitude_bonus = _alt_bonus
        if not player_df.empty:
            model.set_player_stats(player_df)

        # Train: replay training games to build ratings
        backtest_model(train_file, "temp_purged_p.csv", "temp_purged_c.csv", model=model)

        # Predict test fold
        test_df = games.iloc[test_start:test_end].copy()
        if "neutral_site" not in test_df.columns:
            test_df["neutral_site"] = False
        test_df["_dp"] = pd.to_datetime(test_df["date"], errors="coerce")
        probs, actuals = [], []
        for _, row in test_df.iterrows():
            try:
                gd = row["_dp"] if pd.notna(row["_dp"]) else None
                p = model.win_prob(row["home_team"], row["away_team"],
                                   team_a_home=True, neutral_site=bool(row.get("neutral_site", False)),
                                   calibrated=False, game_date=gd, use_injuries=False)
                a = 1 if row["home_score"] > row["away_score"] else 0
                probs.append(p)
                actuals.append(a)
                model.update_game(row["home_team"], row["away_team"],
                                  row["home_score"], row["away_score"],
                                  neutral_site=bool(row.get("neutral_site", False)), game_date=gd)
            except Exception as e:
                logging.debug("Purged CV prediction error: %s", e)

        if probs:
            acc = sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == (a == 1)) / len(probs) * 100
            ll = log_loss_binary(actuals, probs)
            br = brier_score_binary(actuals, probs)
            fold_metrics.append({"fold": fold + 1, "accuracy": acc, "log_loss": ll, "brier": br, "n": len(probs)})
            print("  Fold %d: Acc=%.2f%%  LL=%.4f  Brier=%.4f  (%d games)"
                  % (fold + 1, acc, ll, br, len(probs)))

    for tmp in ["temp_purged_train.csv", "temp_purged_p.csv", "temp_purged_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    if fold_metrics:
        accs = [m["accuracy"] for m in fold_metrics]
        lls = [m["log_loss"] for m in fold_metrics]
        brs = [m["brier"] for m in fold_metrics]
        div(80)
        print("  MEAN:  Acc=%.2f%% (+/-%.2f)  LL=%.4f (+/-%.4f)  Brier=%.4f (+/-%.4f)"
              % (np.mean(accs), np.std(accs), np.mean(lls), np.std(lls), np.mean(brs), np.std(brs)))
        if np.std(accs) > 3.0:
            print("  %s Accuracy varies >3%% across folds - model may be fragile" % cwarn("WARNING:"))
        else:
            print("  %s Accuracy variance within normal range" % cok("OK:"))
    return fold_metrics


# -- Combinatorial Purged Cross-Validation (CPCV) ------------------------

def combinatorial_purged_cv(csv_file=GAMES_FILE, k_blocks=5, k_test=2):
    """All C(k, k_test) train/test combinations — tighter confidence intervals."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    games = pd.read_csv(csv_file)
    n = len(games)
    block_size = n // k_blocks
    n_combos = math_comb(k_blocks, k_test)

    print("\nCOMBINATORIAL PURGED CROSS-VALIDATION")
    print(cdim("  %d blocks, %d test, %d combinations" % (k_blocks, k_test, n_combos)))
    div(80)

    player_df = load_player_stats()
    _alt_bonus = _calc_altitude_bonus(csv_file)
    all_accs = []

    for ci, test_blocks in enumerate(combinations(range(k_blocks), k_test), 1):
        train_blocks = [b for b in range(k_blocks) if b not in test_blocks]
        train_idx = []
        for b in train_blocks:
            train_idx.extend(range(b * block_size, min((b + 1) * block_size, n)))
        test_idx = []
        for b in test_blocks:
            test_idx.extend(range(b * block_size, min((b + 1) * block_size, n)))
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue
        train_idx.sort()

        train_file = "temp_cpcv.csv"
        games.iloc[train_idx].to_csv(train_file, index=False)
        model = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
        model._altitude_bonus = _alt_bonus
        if not player_df.empty:
            model.set_player_stats(player_df)
        backtest_model(train_file, "temp_cpcv_p.csv", "temp_cpcv_c.csv", model=model)

        test_df = games.iloc[test_idx].copy()
        if "neutral_site" not in test_df.columns:
            test_df["neutral_site"] = False
        test_df["_dp"] = pd.to_datetime(test_df["date"], errors="coerce")
        correct, total = 0, 0
        for _, row in test_df.iterrows():
            try:
                gd = row["_dp"] if pd.notna(row["_dp"]) else None
                p = model.win_prob(row["home_team"], row["away_team"],
                                   team_a_home=True, neutral_site=bool(row.get("neutral_site", False)),
                                   calibrated=False, game_date=gd, use_injuries=False)
                a = 1 if row["home_score"] > row["away_score"] else 0
                if (p >= 0.5) == (a == 1):
                    correct += 1
                total += 1
            except Exception as e:
                logging.debug("CPCV prediction error: %s", e)
        if total > 0:
            acc = correct / total * 100
            all_accs.append(acc)
            if ci <= 10 or ci % 5 == 0 or ci == n_combos:
                print("  Combo %d/%d  test=%s  Acc=%.2f%%  (%d games)"
                      % (ci, n_combos, test_blocks, acc, total), flush=True)

    for tmp in ["temp_cpcv.csv", "temp_cpcv_p.csv", "temp_cpcv_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    if all_accs:
        div(80)
        print("  CPCV RESULTS (%d paths):" % len(all_accs))
        print("  Mean=%.2f%%  Std=%.2f%%  Min=%.2f%%  Max=%.2f%%"
              % (np.mean(all_accs), np.std(all_accs), np.min(all_accs), np.max(all_accs)))
        above65 = sum(1 for a in all_accs if a > 65) / len(all_accs) * 100
        below55 = sum(1 for a in all_accs if a < 55) / len(all_accs) * 100
        print("  Paths >65%%: %.0f%%  |  Paths <55%%: %.0f%%" % (above65, below55))
        if below55 > 10:
            print("  %s Some paths <55%% - path-dependent overfitting risk" % cwarn("WARNING:"))
        else:
            print("  %s Model robust across combinatorial paths" % cok("OK:"))
    return all_accs


# -- Probability of Backtest Overfitting (PBO) ----------------------------

def probability_of_backtest_overfitting(grid_file="mlb_grid_search.csv"):
    """Check if grid search 'best' is likely overfit via symmetric cross-validation."""
    if not os.path.exists(grid_file):
        print(cwarn("  No grid search results found. Run 'grid' first."))
        return None
    df = pd.read_csv(grid_file)
    if "score" not in df.columns or len(df) < 10:
        print(cwarn("  Insufficient grid search data for PBO analysis."))
        return None

    scores = df["score"].values
    n = len(scores)
    half = n // 2

    print("\nPROBABILITY OF BACKTEST OVERFITTING")
    print(cdim("  Symmetric CV on %d grid search trials" % n))
    div(80)

    n_splits = min(1000, n * 10)
    overfit_count = 0
    logit_lambdas = []

    for _ in range(n_splits):
        perm = np.random.permutation(n)
        s1_idx, s2_idx = perm[:half], perm[half:2 * half]
        s1_scores = scores[s1_idx]
        # Best in s1 (by index into s1)
        best_s1_local = np.argmax(s1_scores)
        best_s1_global = s1_idx[best_s1_local]
        # Rank this trial's score among s2
        s2_sorted = np.sort(scores[s2_idx])
        rank = np.searchsorted(s2_sorted, scores[best_s1_global])
        pct = rank / len(s2_idx) if len(s2_idx) > 0 else 0.5
        # logit(pct) for lambda distribution
        pct_c = np.clip(pct, 0.01, 0.99)
        logit_lambdas.append(np.log(pct_c / (1 - pct_c)))
        if pct < 0.5:
            overfit_count += 1

    pbo = overfit_count / n_splits
    print("  PBO = %.3f  (%d/%d splits where IS-best ranks below OOS median)"
          % (pbo, overfit_count, n_splits))
    if pbo > 0.5:
        print("  %s PBO > 0.5 - optimization is likely overfit" % cwarn("WARNING:"))
    else:
        print("  %s PBO <= 0.5 - optimization results appear genuine" % cok("OK:"))
    return pbo


# -- Monte Carlo Permutation Testing --------------------------------------

def monte_carlo_permutation_test(csv_file=GAMES_FILE, n_permutations=500):
    """Null distribution test: is the model's edge statistically significant?"""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    player_df = load_player_stats()
    _alt_bonus = _calc_altitude_bonus(csv_file)

    # Real model performance
    model = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)
    _, real_met = backtest_model(csv_file, "temp_mc_real.csv", "temp_mc_real_cal.csv", model=model)
    real_acc = real_met["accuracy"]
    real_brier = real_met["brier"]

    try:
        raw = input("\n  Permutations [%d]: " % n_permutations).strip()
        if raw:
            n_permutations = int(raw)
    except ValueError:
        pass

    print("\nMONTE CARLO PERMUTATION TEST")
    print(cdim("  Real: Acc=%.2f%% Brier=%.4f  |  %d permutations" % (real_acc, real_brier, n_permutations)))
    div(80)

    games = pd.read_csv(csv_file)
    perm_accs, perm_briers = [], []

    for i in range(n_permutations):
        shuffled = games.copy()
        swap = np.random.random(len(shuffled)) < 0.5
        hs = shuffled["home_score"].values.copy()
        aws = shuffled["away_score"].values.copy()
        shuffled.loc[swap, "home_score"] = aws[swap]
        shuffled.loc[swap, "away_score"] = hs[swap]
        shuffled.to_csv("temp_mc_shuf.csv", index=False)

        pm = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
        pm._altitude_bonus = _alt_bonus
        if not player_df.empty:
            pm.set_player_stats(player_df)
        _, met = backtest_model("temp_mc_shuf.csv", "temp_mc_p.csv", "temp_mc_c.csv", model=pm)
        if met:
            perm_accs.append(met["accuracy"])
            perm_briers.append(met["brier"])
        if (i + 1) % 50 == 0:
            print("  %d/%d permutations..." % (i + 1, n_permutations), flush=True)

    for tmp in ["temp_mc_real.csv", "temp_mc_real_cal.csv", "temp_mc_shuf.csv", "temp_mc_p.csv", "temp_mc_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    if perm_accs:
        p_acc = sum(1 for a in perm_accs if a >= real_acc) / len(perm_accs)
        p_brier = sum(1 for b in perm_briers if b <= real_brier) / len(perm_briers)
        div(80)
        print("  Real Acc: %.2f%%  |  Null: %.2f%% (std %.2f%%)"
              % (real_acc, np.mean(perm_accs), np.std(perm_accs)))
        print("  Real Brier: %.4f  |  Null: %.4f (std %.4f)"
              % (real_brier, np.mean(perm_briers), np.std(perm_briers)))
        print("  p-value (acc): %.4f  |  p-value (Brier): %.4f" % (p_acc, p_brier))
        if p_acc < 0.05:
            print("  %s Model is statistically significant (p < 0.05)" % cok("SIGNIFICANT:"))
        else:
            print("  %s Model NOT significant (p >= 0.05)" % cwarn("NOT SIGNIFICANT:"))
        return {"p_acc": p_acc, "p_brier": p_brier}
    return None


# -- Rolling Origin Recalibration -----------------------------------------

def rolling_origin_recalibration(csv_file=GAMES_FILE, chunk_size=50):
    """Expanding-window Platt refitting for truly out-of-sample calibrated metrics."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    model = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()
    if not player_df.empty:
        model.set_player_stats(player_df)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_dp"] = pd.to_datetime(games["date"], errors="coerce")

    print("\nROLLING ORIGIN RECALIBRATION")
    print(cdim("  Expanding-window Platt every %d games" % chunk_size))
    div(80)

    raw_probs, actuals, cal_probs = [], [], []
    scaler = None

    for _, row in games.iterrows():
        try:
            gd = row["_dp"] if pd.notna(row["_dp"]) else None
            p = model.win_prob(row["home_team"], row["away_team"],
                               team_a_home=True, neutral_site=bool(row["neutral_site"]),
                               calibrated=False, game_date=gd, use_injuries=False)
            a = 1 if row["home_score"] > row["away_score"] else 0
            raw_probs.append(p)
            actuals.append(a)
            cal_probs.append(apply_platt(p, scaler) if scaler else p)

            if len(raw_probs) >= 200 and len(raw_probs) % chunk_size == 0:
                scaler = fit_platt_scaler(raw_probs, actuals)
                cs = max(0, len(raw_probs) - chunk_size)
                cb = brier_score_binary(actuals[cs:], cal_probs[cs:])
                print("  Games 1-%d: refit Platt  |  chunk Brier=%.4f" % (len(raw_probs), cb))

            model.update_game(row["home_team"], row["away_team"],
                              row["home_score"], row["away_score"],
                              neutral_site=bool(row["neutral_site"]), game_date=gd)
        except Exception:
            pass

    if cal_probs:
        acc = sum(1 for p, a in zip(cal_probs, actuals) if (p >= 0.5) == (a == 1)) / len(actuals) * 100
        ll = log_loss_binary(actuals, cal_probs)
        br = brier_score_binary(actuals, cal_probs)
        ece = ece_score(cal_probs, actuals)
        # Compare with single-pass
        ss = fit_platt_scaler(raw_probs, actuals)
        sc = [apply_platt(p, ss) for p in raw_probs]
        br_single = brier_score_binary(actuals, sc)
        div(80)
        print("  ROLLING:     Acc=%.2f%%  LL=%.4f  Brier=%.4f  ECE=%.4f" % (acc, ll, br, ece))
        print("  SINGLE-PASS: Brier=%.4f (for comparison)" % br_single)
        if br < br_single:
            print("  %s Rolling is better by %.4f Brier" % (cok("BETTER:"), br_single - br))
        else:
            print("  %s Single-pass is better by %.4f Brier" % (cwarn("NOTE:"), br - br_single))
    return {"rolling_brier": br, "single_brier": br_single} if cal_probs else None


# -- Kelly Criterion Position Sizing Backtest ------------------------------

def kelly_criterion_backtest(predictions_csv="mlb_backtest_predictions.csv",
                             kelly_fraction=0.25, initial_bankroll=1000.0):
    """Simulate Kelly-optimal position sizing on backtest predictions."""
    if not os.path.exists(predictions_csv):
        print(cwarn("  No backtest predictions found. Run 'backtest' first."))
        return None
    scaler = load_platt_scaler()
    df = pd.read_csv(predictions_csv)

    try:
        raw = input("\n  Kelly fraction (0.25 = quarter-Kelly) [%.2f]: " % kelly_fraction).strip()
        if raw:
            kelly_fraction = float(raw)
        raw = input("  Initial bankroll [%.0f]: " % initial_bankroll).strip()
        if raw:
            initial_bankroll = float(raw)
    except ValueError:
        pass

    print("\nKELLY CRITERION BACKTEST")
    print(cdim("  %.0f%%-Kelly  |  $%.0f starting bankroll" % (kelly_fraction * 100, initial_bankroll)))
    div(80)

    bankroll = initial_bankroll
    peak = bankroll
    max_dd = 0.0
    trades, wins = 0, 0
    history = [bankroll]

    for _, row in df.iterrows():
        raw_prob = row["home_win_prob"]
        prob = apply_platt(raw_prob, scaler) if scaler else raw_prob
        # Our edge: model prob vs fair-odds entry (0.50)
        entry_price = 0.50
        if prob >= 0.5:
            edge_prob = prob
            won = int(row.get("correct", 0)) == 1
        else:
            edge_prob = 1 - prob
            won = int(row.get("correct", 0)) == 1
        # Kelly: f = (p * b - q) / b where b = payout odds, q = 1-p
        payout = 1.0 - entry_price
        kelly_f = max(0, (edge_prob * payout - (1 - edge_prob) * entry_price) / payout) * kelly_fraction
        if kelly_f <= 0 or bankroll <= 0:
            history.append(bankroll)
            continue
        position = kelly_f * bankroll
        contracts = position / entry_price
        if won:
            profit = contracts * payout - contracts * 0.02
            wins += 1
        else:
            profit = -contracts * entry_price
        bankroll += profit
        bankroll = max(bankroll, 0)
        trades += 1
        history.append(bankroll)
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if trades == 0:
        print(cwarn("  No trades taken (Kelly fraction too conservative)"))
        return None

    ret = (bankroll - initial_bankroll) / initial_bankroll * 100
    wr = wins / trades * 100
    returns = np.diff(history) / np.maximum(np.array(history[:-1]), 1e-8)
    nz = returns[returns != 0]
    sharpe = float(np.mean(nz) / max(np.std(nz), 1e-8) * np.sqrt(252)) if len(nz) > 1 else 0

    div(80)
    print("  KELLY RESULTS (%d trades, %d wins)" % (trades, wins))
    print("  Final bankroll:    $%.2f (%.1f%% return)" % (bankroll, ret))
    print("  Win rate:          %.1f%%" % wr)
    print("  Max drawdown:      %.1f%%" % (max_dd * 100))
    print("  Annualized Sharpe: %.2f" % sharpe)
    return {"final": bankroll, "return_pct": ret, "max_dd": max_dd, "sharpe": sharpe,
            "trades": trades, "win_rate": wr}


# -- Sliding Window vs Expanding Window Backtest ---------------------------

def sliding_window_backtest(csv_file=GAMES_FILE, window_size=300):
    """Compare sliding (reset every N games) vs expanding window."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    _alt_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()

    try:
        raw = input("\n  Window size (games before heavy regression) [%d]: " % window_size).strip()
        if raw:
            window_size = int(raw)
    except ValueError:
        pass

    print("\nSLIDING WINDOW BACKTEST")
    print(cdim("  Window: %d games (heavy regression at boundary)" % window_size))
    div(80)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_dp"] = pd.to_datetime(games["date"], errors="coerce")

    model = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    probs, actuals = [], []
    games_since_reset = 0

    for _, row in games.iterrows():
        try:
            gd = row["_dp"] if pd.notna(row["_dp"]) else None
            if games_since_reset >= window_size:
                model.ratings = defaultdict(
                    lambda: model.base_rating,
                    regress_ratings_to_mean(dict(model.ratings), factor=0.67))
                games_since_reset = 0

            p = model.win_prob(row["home_team"], row["away_team"],
                               team_a_home=True, neutral_site=bool(row["neutral_site"]),
                               calibrated=False, game_date=gd, use_injuries=False)
            a = 1 if row["home_score"] > row["away_score"] else 0
            probs.append(p)
            actuals.append(a)
            model.update_game(row["home_team"], row["away_team"],
                              row["home_score"], row["away_score"],
                              neutral_site=bool(row["neutral_site"]), game_date=gd)
            games_since_reset += 1
        except Exception:
            pass

    if not probs:
        return None
    acc_s = sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == (a == 1)) / len(probs) * 100
    ll_s = log_loss_binary(actuals, probs)
    br_s = brier_score_binary(actuals, probs)

    # Expanding window comparison
    model2 = MLBElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model2._altitude_bonus = _alt_bonus
    if not player_df.empty:
        model2.set_player_stats(player_df)
    _, em = backtest_model(csv_file, "temp_sw.csv", "temp_sw_c.csv", model=model2)
    for tmp in ["temp_sw.csv", "temp_sw_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    div(80)
    print("  SLIDING (reset/%d): Acc=%.2f%%  LL=%.4f  Brier=%.4f" % (window_size, acc_s, ll_s, br_s))
    print("  EXPANDING (std):    Acc=%.2f%%  LL=%.4f  Brier=%.4f"
          % (em["accuracy"], em["log_loss"], em["brier"]))
    if br_s < em["brier"]:
        print("  %s Sliding window is better - old data may hurt" % cok("RESULT:"))
    else:
        print("  %s Expanding window is better - full history helps" % cok("RESULT:"))
    return {"sliding_brier": br_s, "expanding_brier": em["brier"]}


# -- Elo Rating Convergence Analysis --------------------------------------

def elo_convergence_analysis(csv_file=GAMES_FILE, chunk_size=100):
    """Identify burn-in period by chunking predictions into accuracy segments."""
    pred_file = "mlb_backtest_predictions.csv"
    if not os.path.exists(pred_file):
        print(cwarn("  Run 'backtest' first to generate predictions."))
        return None

    try:
        raw = input("\n  Chunk size [%d]: " % chunk_size).strip()
        if raw:
            chunk_size = int(raw)
    except ValueError:
        pass

    df = pd.read_csv(pred_file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("\nELO RATING CONVERGENCE ANALYSIS")
    print(cdim("  Chunk size: %d games" % chunk_size))
    div(80)

    chunks = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        if len(chunk) < 20:
            continue
        acc = float(chunk["correct"].mean() * 100)
        dr = ""
        dates = chunk["date"].dropna()
        if len(dates) > 0:
            dr = " (%s to %s)" % (str(dates.iloc[0])[:10], str(dates.iloc[-1])[:10])
        chunks.append({"start": start, "n": len(chunk), "accuracy": acc, "dr": dr})
        print("  Games %4d-%4d: Acc=%.2f%% (%d games)%s"
              % (start + 1, start + len(chunk), acc, len(chunk), dr))

    if len(chunks) >= 2:
        accs = [c["accuracy"] for c in chunks]
        converged_at = 0
        for i in range(len(chunks)):
            if np.std(accs[i:]) < 2.0 and len(accs[i:]) >= 2:
                converged_at = i
                break
        burn_in = chunks[converged_at]["start"]
        steady = float(np.mean(accs[converged_at:]))
        div(80)
        if converged_at > 0:
            print("  Burn-in: first %d games (avg Acc=%.2f%%)"
                  % (burn_in, np.mean(accs[:converged_at])))
        print("  Steady state after game %d: avg Acc=%.2f%%" % (burn_in, steady))
        print("  Suggested burn-in exclusion: %d games" % burn_in)
    return chunks


# -- Conformal Prediction Analysis ----------------------------------------

def conformal_analysis(csv_file=GAMES_FILE):
    """Run conformal prediction set analysis on backtest predictions."""
    pred_file = "mlb_backtest_predictions.csv"
    if not os.path.exists(pred_file):
        print(cwarn("  Run 'backtest' first."))
        return None

    df = pd.read_csv(pred_file)
    probs = df["home_win_prob"].values
    actuals = (df["home_score"] > df["away_score"]).astype(int).values

    scaler = load_platt_scaler()
    if scaler:
        probs = np.array([apply_platt(p, scaler) for p in probs])

    print("\nCONFORMAL PREDICTION ANALYSIS")
    div(80)

    for alpha in [0.05, 0.10, 0.20]:
        result = conformal_prediction_set(probs, actuals, alpha=alpha)
        if result:
            cov = result["actual_coverage"] * 100
            tgt = result["target_coverage"] * 100
            ok = cok("*") if abs(cov - tgt) < 3 else cwarn("!")
            print("  alpha=%.2f  Target=%.0f%%  Actual=%.1f%% %s  Avg set=%.2f  "
                  "Singleton=%.0f%%  Empty=%.0f%%  Both=%.0f%%"
                  % (alpha, tgt, cov, ok, result["avg_set_size"],
                     result["singleton_pct"], result["empty_pct"], result["both_pct"]))

    print(cdim("\n  Singleton = model is confident (only one outcome in set)"))
    print(cdim("  Both = model is uncertain (both outcomes in set)"))
    print(cdim("  Empty = model is overconfident (set excludes true outcome)"))
    return True


# -- Beta Calibration Command ----------------------------------------------

def run_beta_calibration(csv_file=GAMES_FILE):
    """Fit beta calibration on backtest predictions and compare with Platt."""
    from platt import fit_beta_scaler, apply_beta, save_beta_scaler
    pred_file = "mlb_backtest_predictions.csv"
    if not os.path.exists(pred_file):
        print(cwarn("  Run 'backtest' first."))
        return None

    df = pd.read_csv(pred_file)
    probs = df["home_win_prob"].values
    actuals = (df["home_score"] > df["away_score"]).astype(int).values

    print("\nBETA CALIBRATION")
    div(80)

    beta_sc = fit_beta_scaler(probs, actuals)
    save_beta_scaler(beta_sc)
    beta_cal = [apply_beta(p, beta_sc) for p in probs]
    br_beta = brier_score_binary(actuals, beta_cal)
    ll_beta = log_loss_binary(actuals, beta_cal)
    ece_beta = ece_score(beta_cal, actuals)

    # Compare with Platt
    platt_sc = fit_platt_scaler(probs, actuals)
    platt_cal = [apply_platt(p, platt_sc) for p in probs]
    br_platt = brier_score_binary(actuals, platt_cal)
    ll_platt = log_loss_binary(actuals, platt_cal)
    ece_platt = ece_score(platt_cal, actuals)

    print("  BETA (a=%.3f b=%.3f c=%.3f):  Brier=%.4f  LL=%.4f  ECE=%.4f"
          % (beta_sc["a"], beta_sc["b"], beta_sc["c"], br_beta, ll_beta, ece_beta))
    print("  PLATT (coef=%.3f int=%.3f):     Brier=%.4f  LL=%.4f  ECE=%.4f"
          % (platt_sc["coef"], platt_sc["intercept"], br_platt, ll_platt, ece_platt))

    if abs(beta_sc["a"] - beta_sc["b"]) > 0.3:
        print("  %s a != b — asymmetric miscalibration detected (beta is more appropriate)"
              % chi("NOTE:"))
    else:
        print("  %s a ~ b — calibration is symmetric (Platt is sufficient)" % cdim("NOTE:"))

    if br_beta < br_platt:
        print("  %s Beta calibration wins by %.4f Brier" % (cok("BETTER:"), br_platt - br_beta))
    else:
        print("  %s Platt calibration is equal or better" % cok("OK:"))
    return beta_sc


# -- Auto-Optimize --------------------------------------------------------

def auto_optimize(csv_file=GAMES_FILE):
    """Run grid -> genetic -> bayesian automatically, compare all, apply the best.

    No interactive prompts. Uses sensible defaults for each phase:
      Phase 1: Coarse grid (10 core params, manageable combos) to find the neighborhood
      Phase 2: Genetic with ALL params, tightened bounds (50 gen x 25 pop) to refine
      Phase 3: Bayesian with ALL params, tightened bounds (15 initial + 40 iter) to polish
      Phase 4: Compare all three winners, apply the absolute best
    """
    import time
    from scipy.stats.qmc import LatinHypercube

    if not os.path.exists(csv_file):
        print(cwarn("  Game data not found: %s" % csv_file))
        return None

    settings = load_elo_settings()
    base = settings.get("base_rating", 1500.0)
    use_mov = settings.get("use_mov", True)
    player_df = load_player_stats()
    has_players = not player_df.empty
    _prebuilt = build_league_player_scores(player_df) if has_players else {}
    _alt = _calc_altitude_bonus(csv_file)

    eval_count = [0]

    # ALL optimizable parameters with (name, grid_vals_or_None, (lo, hi), tight_hw)
    # grid_vals=None means skip this param in grid search (only genetic/bayesian)
    PARAMS = [
        # --- Core (in grid) ---
        ("k",                      [5, 15, 30, 50],    (1, 80),     10),
        ("home_adv",               [15, 30, 50, 75],   (0, 150),    15),
        ("player_boost",           [0, 10, 25],         (0, 60),     10),
        ("starter_boost",          [0, 20, 40],          (0, 80),     15),
        ("rest_factor",            [0, 15],              (0, 50),     10),
        ("form_weight",            [0, 10],              (0, 40),     10),
        ("travel_factor",          [0, 20],              (0, 60),     15),
        ("sos_factor",             [0, 10],              (0, 40),     10),
        ("playoff_hca_factor",     [0.5, 0.8],           (0.0, 1.5),  0.3),
        ("pace_factor",            [0, 20],              (0, 60),     15),
        # --- Secondary (genetic/bayesian only) ---
        ("division_factor",        None,                 (0, 40),     10),
        ("mean_reversion",         None,                 (0, 20),     5),
        ("pyth_factor",            None,                 (0, 30),     8),
        ("home_road_factor",       None,                 (0, 20),     5),
        ("mov_base",               None,                 (0.3, 2.0),  0.3),
        ("b2b_penalty",            None,                 (0, 150),    30),
        ("road_trip_factor",       None,                 (0, 20),     5),
        ("homestand_factor",       None,                 (0, 20),     5),
        ("win_streak_factor",      None,                 (0, 20),     5),
        ("altitude_factor",        None,                 (0, 30),     8),
        ("season_phase_factor",    None,                 (0, 20),     5),
        ("scoring_consistency_factor", None,             (0, 15),     4),
        ("rest_advantage_cap",     None,                 (0, 10),     3),
        ("park_factor_weight",     None,                 (0, 30),     8),
        ("mov_cap",                None,                 (0, 20),     5),
        ("east_travel_penalty",    None,                 (0, 15),     4),
        ("series_adaptation",      None,                 (0, 15),     4),
        ("interleague_factor",     None,                 (0, 10),     3),
        ("bullpen_factor",         None,                 (0, 30),     8),
        ("opp_pitcher_factor",     None,                 (0, 30),     8),
        ("k_decay",                None,                 (0, 5),      1.5),
        ("surprise_k",             None,                 (0, 10),     3),
    ]

    param_names = [p[0] for p in PARAMS]
    n_params = len(PARAMS)
    grid_indices = [i for i, p in enumerate(PARAMS) if p[1] is not None]
    grid_value_lists = [PARAMS[i][1] for i in grid_indices]
    full_bounds = [(p[2][0], p[2][1]) for p in PARAMS]

    # Default values for non-grid params (from current settings)
    defaults = {p[0]: settings.get(p[0], 0.0) for p in PARAMS}
    # Special default for mov_base
    if defaults.get("mov_base", 0.0) < 0.01:
        defaults["mov_base"] = 1.0

    def _eval(params):
        """Shared objective: minimize LogLoss*8 + Brier*40."""
        kw = {param_names[i]: float(params[i]) for i in range(n_params)}
        m = MLBElo(base_rating=base, use_mov=use_mov, **kw)
        m._altitude_bonus = _alt
        if has_players:
            m._player_scores = _prebuilt
        ok, met = backtest_model(csv_file, "temp_auto.csv", "temp_auto_cal.csv", model=m)
        eval_count[0] += 1
        if not ok:
            return 1e9, {}
        return met["log_loss"] * 8.0 + met["brier"] * 40.0, met

    def _params_dict(p):
        return {param_names[i]: float(p[i]) for i in range(n_params)}

    def _fmt(p):
        core = "K=%.1f HA=%.1f PB=%.1f SB=%.1f R=%.1f FW=%.1f T=%.1f SOS=%.1f PHCA=%.2f P=%.1f" % tuple(p[:10])
        nonzero = []
        for i in range(10, n_params):
            if abs(float(p[i])) > 0.001:
                nonzero.append("%s=%.1f" % (param_names[i][:4], float(p[i])))
        if nonzero:
            core += " +" + ",".join(nonzero[:5])
        return core

    winners = []  # [(score, params_array, source_name, metrics)]

    print("\n" + "=" * 80)
    print(chi("  AUTO-OPTIMIZE: Grid -> Genetic -> Bayesian (%d total params)" % n_params))
    print("=" * 80)
    t_start = time.time()

    # -- Phase 1: Coarse Grid Search --------------------------------------
    print(chi("\n  PHASE 1: Coarse Grid Search"))
    div(80)

    total = 1
    for v in grid_value_lists:
        total *= len(v)
    print("  %d combinations (%d grid params, %d total params)" % (total, len(grid_indices), n_params))

    grid_best_score, grid_best_params = 1e9, None
    grid_best_met = {}
    grid_rows = []
    t1 = time.time()

    for i, combo in enumerate(product(*grid_value_lists), 1):
        full_params = [defaults.get(param_names[j], 0.0) for j in range(n_params)]
        for gi, val in zip(grid_indices, combo):
            full_params[gi] = val
        score, met = _eval(full_params)
        if score < 1e8:
            row = {param_names[j]: full_params[j] for j in range(n_params)}
            row.update({"accuracy": met.get("accuracy", 0),
                        "log_loss": met.get("log_loss", 0),
                        "brier": met.get("brier", 0), "score": -score})
            grid_rows.append(row)
        is_best = score < grid_best_score
        if is_best:
            grid_best_score = score
            grid_best_params = np.array(full_params, dtype=float)
            grid_best_met = met
        if is_best or i % 100 == 0 or i == total:
            flag = cok(" <- BEST") if is_best else ""
            print("  %4d/%d  Acc=%.2f%% LL=%.4f Br=%.4f  %s%s"
                  % (i, total, met.get("accuracy", 0), met.get("log_loss", 0),
                     met.get("brier", 0), _fmt(full_params), flag),
                  flush=True)

    if grid_rows:
        pd.DataFrame(grid_rows).to_csv("mlb_grid_search.csv", index=False)

    t1_elapsed = time.time() - t1
    if grid_best_params is not None:
        winners.append((grid_best_score, grid_best_params, "Grid", grid_best_met))
        print("  Grid done: %.0fs  |  Best score=%.4f  Acc=%.2f%%"
              % (t1_elapsed, -grid_best_score, grid_best_met.get("accuracy", 0)))
    else:
        print(cwarn("  Grid search produced no valid results"))
        return None

    # -- Tighten bounds around grid best -----------------------------------
    def _tight(val, lo, hi, hw):
        return (max(lo, val - hw), min(hi, val + hw))

    bp = grid_best_params
    bounds = [_tight(bp[i], PARAMS[i][2][0], PARAMS[i][2][1], PARAMS[i][3]) for i in range(n_params)]
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    ranges = highs - lows
    ranges[ranges < 1e-12] = 1.0

    print(cdim("  Tightened bounds (%d params): %s" %
               (n_params, ["(%.1f-%.1f)" % b for b in bounds[:10]])))
    print(cdim("    + %d secondary params with tightened bounds" % (n_params - 10)))

    # -- Phase 2: Genetic Optimization ------------------------------------
    print(chi("\n  PHASE 2: Genetic Algorithm (ALL %d params, tightened bounds)" % n_params))
    div(80)
    t2 = time.time()

    gen_best_score = [1e9]
    gen_best_params = [None]
    gen_counter = [0]

    def _gen_cb(xk, convergence):
        gen_counter[0] += 1
        if gen_counter[0] % 10 == 0 and gen_best_params[0] is not None:
            print("  Gen %d  best=%.4f" % (gen_counter[0], -gen_best_score[0]), flush=True)

    def _gen_obj(params):
        score, met = _eval(params)
        if score < gen_best_score[0]:
            gen_best_score[0] = score
            gen_best_params[0] = np.array(params)
        return score

    de_result = differential_evolution(
        _gen_obj, bounds=bounds, maxiter=50, popsize=25,
        workers=1, polish=True, callback=_gen_cb)

    gen_p = de_result.x
    gen_score, gen_met = _eval(gen_p)
    winners.append((gen_score, np.array(gen_p), "Genetic", gen_met))
    pd.DataFrame([{**_params_dict(gen_p), "score": -gen_score,
                    "accuracy": gen_met.get("accuracy", 0),
                    "log_loss": gen_met.get("log_loss", 0),
                    "brier": gen_met.get("brier", 0)}]).to_csv(
        "mlb_genetic_results.csv", index=False)

    t2_elapsed = time.time() - t2
    print("  Genetic done: %.0fs  |  Best score=%.4f  Acc=%.2f%%"
          % (t2_elapsed, -gen_score, gen_met.get("accuracy", 0)))

    # -- Phase 3: Bayesian Optimization -----------------------------------
    print(chi("\n  PHASE 3: Bayesian Optimization (ALL %d params, GP surrogate)" % n_params))
    div(80)
    t3 = time.time()

    class _GP:
        def __init__(self, ls=1.0, noise=1e-4):
            self.ls, self.noise = ls, noise
        def _kern(self, A, B):
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-0.5 * d / self.ls ** 2)
        def fit(self, X, y):
            self.X, self.y = X, y
            K = self._kern(X, X) + self.noise * np.eye(len(X))
            try:
                self.L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(len(X))
                self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        def predict(self, Xn):
            k = self._kern(Xn, self.X)
            mu = k @ self.alpha
            v = np.linalg.solve(self.L, k.T)
            var = 1.0 + self.noise - np.sum(v ** 2, axis=0)
            return mu, np.sqrt(np.maximum(var, 1e-12))

    n_initial, n_iter = 15, 40
    sampler = LatinHypercube(d=n_params)
    X_init = lows + sampler.random(n=n_initial) * ranges

    X_all, y_all = [], []
    bay_best_score, bay_best_params = 1e9, None
    bay_best_met = {}

    print("  Evaluating %d initial samples..." % n_initial)
    for x in X_init:
        s, met = _eval(x)
        X_all.append(x)
        y_all.append(s)
        if s < bay_best_score:
            bay_best_score, bay_best_params, bay_best_met = s, x, met

    gp = _GP()
    for it in range(n_iter):
        Xa, ya = np.array(X_all), np.array(y_all)
        Xn = (Xa - lows) / ranges
        ym, ys = ya.mean(), max(ya.std(), 1e-8)
        yn = (ya - ym) / ys
        gp.fit(Xn, yn)

        cands = np.random.rand(1000, n_params)
        mu, sigma = gp.predict(cands)
        best_n = (bay_best_score - ym) / ys
        z = (best_n - mu) / sigma
        ei = sigma * (z * norm_dist.cdf(z) + norm_dist.pdf(z))
        ei[sigma < 1e-10] = 0.0

        x_next = np.clip(lows + cands[np.argmax(ei)] * ranges, lows, highs)
        s, met = _eval(x_next)
        X_all.append(x_next)
        y_all.append(s)
        is_best = s < bay_best_score
        if is_best:
            bay_best_score, bay_best_params, bay_best_met = s, x_next, met
        if is_best or (it + 1) % 10 == 0:
            flag = cok(" <- BEST") if is_best else ""
            print("  Iter %2d/%d  score=%.4f  best=%.4f%s"
                  % (it + 1, n_iter, -s if s < 1e8 else 0, -bay_best_score, flag), flush=True)

    winners.append((bay_best_score, np.array(bay_best_params), "Bayesian", bay_best_met))
    bay_rows = []
    for x, s in zip(X_all, y_all):
        if s < 1e8:
            bay_rows.append({**_params_dict(x), "score": -s})
    if bay_rows:
        pd.DataFrame(bay_rows).to_csv("mlb_bayesian_results.csv", index=False)

    t3_elapsed = time.time() - t3
    print("  Bayesian done: %.0fs  |  Best score=%.4f  Acc=%.2f%%"
          % (t3_elapsed, -bay_best_score, bay_best_met.get("accuracy", 0)))

    # -- Cleanup temp files -----------------------------------------------
    for tmp in ["temp_auto.csv", "temp_auto_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    # -- Phase 4: Compare and apply best ----------------------------------
    total_time = time.time() - t_start
    total_evals = eval_count[0]

    print("\n" + "=" * 80)
    print(chi("  AUTO-OPTIMIZE RESULTS (%d params)" % n_params))
    print("=" * 80)
    print("  %-10s  %-8s  %-8s  %-8s  %-8s  %s"
          % (chi("Source"), chi("Score"), chi("Acc%"), chi("LogLoss"), chi("Brier"), chi("Parameters")))
    div(80)

    overall_best_score = 1e9
    overall_best = None
    for score, params, source, met in sorted(winners, key=lambda x: x[0]):
        is_winner = score < overall_best_score
        if is_winner:
            overall_best_score = score
            overall_best = (score, params, source, met)
        flag = cok(" ** WINNER") if is_winner and score == sorted(winners, key=lambda x: x[0])[0][0] else ""
        print("  %-10s  %8.4f  %7.2f%%  %8.4f  %8.4f  %s%s"
              % (source, -score, met.get("accuracy", 0), met.get("log_loss", 0),
                 met.get("brier", 0), _fmt(params), flag))

    div(80)
    print("  Total time: %.0f seconds  |  Total evaluations: %d" % (total_time, total_evals))

    # Apply the winner
    if overall_best:
        _, best_p, best_src, best_met = overall_best
        print("\n  %s  %s  %s" % (cok("WINNER:"), chi(best_src), _fmt(best_p)))
        print("  Acc=%s  LogLoss=%.4f  Brier=%.4f"
              % (cok("%.2f%%" % best_met.get("accuracy", 0)),
                 best_met.get("log_loss", 0), best_met.get("brier", 0)))
        _apply_best_settings(_params_dict(best_p), csv_file)

    print("=" * 80)
    return overall_best


# -- Super-Optimize -------------------------------------------------------

def super_optimize(csv_file=GAMES_FILE):
    """Exhaustive multi-round optimization across all 9 tunable parameters.

    Runs every optimization method with aggressive settings, validates the
    winner, and saves to mlb_elo_settings.json (auto-loaded on every start).

    Phase 1: Broad grid search (all 9 params, ~6000+ combos)
    Phase 2: Genetic round 1 - wide bounds (50 gen x 15 pop)
    Phase 3: Bayesian round 1 - wide bounds, 30 initial + 80 iterations
    Phase 4: Genetic round 2 - tightened around overall best (30 gen x 10 pop)
    Phase 5: Bayesian round 2 - tightened, 20 initial + 60 iterations
    Phase 6: Fine grid - tiny steps around absolute best
    Phase 7: Validation - purgedcv + PBO + montecarlo on the winner
    """
    import time
    from scipy.stats.qmc import LatinHypercube

    if not os.path.exists(csv_file):
        print(cwarn("  Game data not found: %s" % csv_file))
        return None

    settings = load_elo_settings()
    base = settings.get("base_rating", 1500.0)
    use_mov = settings.get("use_mov", True)
    player_df = load_player_stats()
    has_players = not player_df.empty
    _prebuilt = build_league_player_scores(player_df) if has_players else {}
    _alt = _calc_altitude_bonus(csv_file)

    eval_count = [0]

    # 9-param evaluation: K, HomeAdv, PlayerBoost, RestFactor, TravelFactor,
    #                      PaceFactor, PlayoffHCA, SOSFactor, FormWeight
    _PARAM_NAMES = ["K", "HA", "PB", "Rest", "Travel", "Pace", "PHCA", "SOS", "Form"]

    def _eval(params):
        k, ha, pb, rf, tf, pf, phca, sos, fw = params
        m = MLBElo(base_rating=base, k=float(k), home_adv=float(ha), use_mov=use_mov,
                   player_boost=float(pb), rest_factor=float(rf),
                   travel_factor=float(tf), pace_factor=float(pf),
                   playoff_hca_factor=float(phca), sos_factor=float(sos),
                   form_weight=float(fw))
        m._altitude_bonus = _alt
        if has_players:
            m._player_scores = _prebuilt
        ok, met = backtest_model(csv_file, "temp_super.csv", "temp_super_cal.csv", model=m)
        eval_count[0] += 1
        if not ok:
            return 1e9, {}
        # Accuracy tiebreaker: when Brier is on a plateau (differs by <0.001),
        # accuracy breaks the tie.  0.1 weight means 1% accuracy ~ 0.001 Brier.
        return met["log_loss"] * 8.0 + met["brier"] * 40.0 - met.get("accuracy", 0) * 0.1, met

    def _params_dict(p):
        return {"k": float(p[0]), "home_adv": float(p[1]), "player_boost": float(p[2]),
                "rest_factor": float(p[3]), "travel_factor": float(p[4]),
                "pace_factor": float(p[5]), "playoff_hca_factor": float(p[6]),
                "sos_factor": float(p[7]), "form_weight": float(p[8])}

    def _fmt(p):
        return ("K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f SOS=%.1f FW=%.1f"
                % tuple(p[:9]))

    def _fmt_short(p):
        return ("K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f"
                % tuple(p[:6]))

    winners = []  # [(score, params_array, source_name, metrics)]
    phase_times = []

    print("\n" + "=" * 100)
    print(chi("  SUPER-OPTIMIZE: Exhaustive multi-round optimization (9 params, all methods)"))
    print(chi("  No prompts. No shortcuts. Finds the absolute best settings."))
    print("=" * 100)
    t_start = time.time()

    # -- Phase 1: Broad Grid Search (9 params) ----------------------------
    print(chi("\n  PHASE 1/7: Broad Grid Search (9 parameters)"))
    div(100)

    k_vals    = [8, 13, 18, 25, 35, 50]
    ha_vals   = [15, 30, 45, 60, 80]
    pb_vals   = [0, 8, 18, 30]
    rf_vals   = [0, 5, 15, 30]
    tf_vals   = [0, 15, 35]
    pf_vals   = [0, 15, 35]
    phca_vals = [0.4, 0.7, 1.0]
    sos_vals  = [0, 10, 25]
    fw_vals   = [0, 5]

    total = (len(k_vals) * len(ha_vals) * len(pb_vals) * len(rf_vals) *
             len(tf_vals) * len(pf_vals) * len(phca_vals) * len(sos_vals) * len(fw_vals))
    print("  %d combinations across 9 dimensions" % total)

    grid_best_score, grid_best_params, grid_best_met = 1e9, None, {}
    grid_rows = []
    t1 = time.time()

    for i, (k, ha, pb, rf, tf, pf, phca, sos, fw) in enumerate(
        product(k_vals, ha_vals, pb_vals, rf_vals, tf_vals, pf_vals,
                phca_vals, sos_vals, fw_vals), 1
    ):
        score, met = _eval([k, ha, pb, rf, tf, pf, phca, sos, fw])
        if score < 1e8:
            grid_rows.append({"k": k, "home_adv": ha, "player_boost": pb,
                              "rest_factor": rf, "travel_factor": tf,
                              "pace_factor": pf, "playoff_hca_factor": phca,
                              "sos_factor": sos, "form_weight": fw,
                              "accuracy": met.get("accuracy", 0),
                              "log_loss": met.get("log_loss", 0),
                              "brier": met.get("brier", 0), "score": -score})
        is_best = score < grid_best_score
        if is_best:
            grid_best_score = score
            grid_best_params = np.array([k, ha, pb, rf, tf, pf, phca, sos, fw], dtype=float)
            grid_best_met = met
        if is_best or i % 200 == 0 or i == total:
            flag = cok(" <- BEST") if is_best else ""
            print("  %5d/%d  Acc=%.2f%% Br=%.4f  %s%s"
                  % (i, total, met.get("accuracy", 0), met.get("brier", 0),
                     _fmt_short([k, ha, pb, rf, tf, pf]), flag), flush=True)

    if grid_rows:
        pd.DataFrame(grid_rows).to_csv("mlb_super_grid.csv", index=False)

    t1_elapsed = time.time() - t1
    phase_times.append(("Grid (broad)", t1_elapsed))
    if grid_best_params is not None:
        winners.append((grid_best_score, grid_best_params, "Grid-Broad", grid_best_met))
        print("  Grid done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
              % (t1_elapsed, -grid_best_score, grid_best_met.get("accuracy", 0)))
    else:
        print(cwarn("  Grid search failed"))
        return None

    # Track the overall best across all phases
    overall_best_score = grid_best_score
    overall_best_params = grid_best_params.copy()

    # -- Phase 2: Genetic Round 1 (wide bounds, big pop) ------------------
    print(chi("\n  PHASE 2/7: Genetic Algorithm Round 1 (wide bounds, aggressive)"))
    div(100)
    t2 = time.time()

    wide_bounds = [
        (3, 80),      # K
        (0, 150),     # HomeAdv
        (0, 50),      # PlayerBoost
        (0, 50),      # RestFactor
        (0, 60),      # TravelFactor
        (0, 60),      # PaceFactor
        (0.0, 1.0),   # PlayoffHCA
        (0, 40),      # SOSFactor
        (0, 20),      # FormWeight
    ]

    gen1_best = [1e9]
    gen1_params = [None]
    gen1_met = [{}]
    gen1_counter = [0]

    def _gen1_cb(xk, convergence):
        gen1_counter[0] += 1
        if gen1_counter[0] % 20 == 0 and gen1_params[0] is not None:
            print("  Gen %d  best=%.4f" % (gen1_counter[0], -gen1_best[0]), flush=True)

    def _gen1_obj(params):
        score, met = _eval(params)
        if score < gen1_best[0]:
            gen1_best[0] = score
            gen1_params[0] = np.array(params)
            gen1_met[0] = met
        return score

    print("  50 generations, population 15, 9 parameters")
    de1 = differential_evolution(
        _gen1_obj, bounds=wide_bounds, maxiter=50, popsize=15,
        workers=1, polish=True, callback=_gen1_cb)

    gen1_p = de1.x
    gen1_s, gen1_m = _eval(gen1_p)
    winners.append((gen1_s, np.array(gen1_p), "Genetic-R1", gen1_m))

    t2_elapsed = time.time() - t2
    phase_times.append(("Genetic R1", t2_elapsed))
    print("  Genetic R1 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t2_elapsed, -gen1_s, gen1_m.get("accuracy", 0)))
    if gen1_s < overall_best_score:
        overall_best_score = gen1_s
        overall_best_params = np.array(gen1_p)

    # -- Phase 3: Bayesian Round 1 (wide bounds) -------------------------
    print(chi("\n  PHASE 3/7: Bayesian Optimization Round 1 (wide bounds)"))
    div(100)
    t3 = time.time()

    class _GP:
        def __init__(self, ls=1.0, noise=1e-4):
            self.ls, self.noise = ls, noise
        def _kern(self, A, B):
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-0.5 * d / self.ls ** 2)
        def fit(self, X, y):
            self.X, self.y = X, y
            K = self._kern(X, X) + self.noise * np.eye(len(X))
            try:
                self.L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(len(X))
                self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        def predict(self, Xn):
            k = self._kern(Xn, self.X)
            mu = k @ self.alpha
            v = np.linalg.solve(self.L, k.T)
            var = 1.0 + self.noise - np.sum(v ** 2, axis=0)
            return mu, np.sqrt(np.maximum(var, 1e-12))

    def _run_bayesian(bounds_list, n_initial, n_iter, label):
        lows = np.array([b[0] for b in bounds_list])
        highs = np.array([b[1] for b in bounds_list])
        ranges = highs - lows
        ranges[ranges < 1e-12] = 1.0

        sampler = LatinHypercube(d=9)
        X_init = lows + sampler.random(n=n_initial) * ranges
        X_all, y_all = [], []
        b_best_score, b_best_params, b_best_met = 1e9, None, {}

        print("  %s: %d initial + %d iterations" % (label, n_initial, n_iter))
        for idx, x in enumerate(X_init):
            s, met = _eval(x)
            X_all.append(x)
            y_all.append(s)
            if s < b_best_score:
                b_best_score, b_best_params, b_best_met = s, x, met
                if (idx + 1) % 5 == 0 or idx == 0:
                    print("  [%d/%d] score=%.4f %s" % (idx + 1, n_initial, -s, cok("BEST")), flush=True)
            elif (idx + 1) % 10 == 0:
                print("  [%d/%d] evaluated" % (idx + 1, n_initial), flush=True)

        gp = _GP()
        for it in range(n_iter):
            Xa, ya = np.array(X_all), np.array(y_all)
            Xn = (Xa - lows) / ranges
            ym, ys = ya.mean(), max(ya.std(), 1e-8)
            yn = (ya - ym) / ys
            gp.fit(Xn, yn)

            cands = np.random.rand(2000, 9)
            mu, sigma = gp.predict(cands)
            best_n = (b_best_score - ym) / ys
            z = (best_n - mu) / sigma
            ei = sigma * (z * norm_dist.cdf(z) + norm_dist.pdf(z))
            ei[sigma < 1e-10] = 0.0

            x_next = np.clip(lows + cands[np.argmax(ei)] * ranges, lows, highs)
            s, met = _eval(x_next)
            X_all.append(x_next)
            y_all.append(s)
            is_best = s < b_best_score
            if is_best:
                b_best_score, b_best_params, b_best_met = s, x_next, met
            if is_best or (it + 1) % 15 == 0:
                flag = cok(" <- BEST") if is_best else ""
                print("  Iter %3d/%d  best=%.4f%s" % (it + 1, n_iter, -b_best_score, flag), flush=True)

        return b_best_score, b_best_params, b_best_met, X_all, y_all

    bay1_s, bay1_p, bay1_m, bay1_X, bay1_y = _run_bayesian(wide_bounds, 30, 80, "Bayesian-R1")
    winners.append((bay1_s, np.array(bay1_p), "Bayesian-R1", bay1_m))

    t3_elapsed = time.time() - t3
    phase_times.append(("Bayesian R1", t3_elapsed))
    print("  Bayesian R1 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t3_elapsed, -bay1_s, bay1_m.get("accuracy", 0)))
    if bay1_s < overall_best_score:
        overall_best_score = bay1_s
        overall_best_params = np.array(bay1_p)

    # -- Tighten bounds around overall best --------------------------------
    def _tight(val, lo, hi, hw):
        return (max(lo, val - hw), min(hi, val + hw))

    bp = overall_best_params
    tight_bounds = [
        _tight(bp[0], 3, 80, 6),       # K
        _tight(bp[1], 0, 150, 10),      # HomeAdv
        _tight(bp[2], 0, 50, 6),        # PlayerBoost
        _tight(bp[3], 0, 50, 6),        # RestFactor
        _tight(bp[4], 0, 60, 10),       # TravelFactor
        _tight(bp[5], 0, 60, 10),       # PaceFactor
        _tight(bp[6], 0.0, 1.0, 0.15),  # PlayoffHCA
        _tight(bp[7], 0, 40, 8),        # SOSFactor
        _tight(bp[8], 0, 20, 5),        # FormWeight
    ]
    print(cdim("\n  Tightened bounds around best-so-far: %s" % _fmt(bp)))

    # -- Phase 4: Genetic Round 2 (tight bounds) --------------------------
    print(chi("\n  PHASE 4/7: Genetic Algorithm Round 2 (tightened bounds)"))
    div(100)
    t4 = time.time()

    gen2_best = [1e9]
    gen2_params = [None]
    gen2_met = [{}]
    gen2_counter = [0]

    def _gen2_cb(xk, convergence):
        gen2_counter[0] += 1
        if gen2_counter[0] % 15 == 0 and gen2_params[0] is not None:
            print("  Gen %d  best=%.4f" % (gen2_counter[0], -gen2_best[0]), flush=True)

    def _gen2_obj(params):
        score, met = _eval(params)
        if score < gen2_best[0]:
            gen2_best[0] = score
            gen2_params[0] = np.array(params)
            gen2_met[0] = met
        return score

    print("  30 generations, population 10, 9 parameters")
    de2 = differential_evolution(
        _gen2_obj, bounds=tight_bounds, maxiter=30, popsize=10,
        workers=1, polish=True, callback=_gen2_cb)

    gen2_p = de2.x
    gen2_s, gen2_m = _eval(gen2_p)
    winners.append((gen2_s, np.array(gen2_p), "Genetic-R2", gen2_m))

    t4_elapsed = time.time() - t4
    phase_times.append(("Genetic R2", t4_elapsed))
    print("  Genetic R2 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t4_elapsed, -gen2_s, gen2_m.get("accuracy", 0)))
    if gen2_s < overall_best_score:
        overall_best_score = gen2_s
        overall_best_params = np.array(gen2_p)

    # -- Phase 5: Bayesian Round 2 (tight bounds) ------------------------
    print(chi("\n  PHASE 5/7: Bayesian Optimization Round 2 (tightened bounds)"))
    div(100)
    t5 = time.time()

    bay2_s, bay2_p, bay2_m, _, _ = _run_bayesian(tight_bounds, 20, 60, "Bayesian-R2")
    winners.append((bay2_s, np.array(bay2_p), "Bayesian-R2", bay2_m))

    t5_elapsed = time.time() - t5
    phase_times.append(("Bayesian R2", t5_elapsed))
    print("  Bayesian R2 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t5_elapsed, -bay2_s, bay2_m.get("accuracy", 0)))
    if bay2_s < overall_best_score:
        overall_best_score = bay2_s
        overall_best_params = np.array(bay2_p)

    # -- Phase 6: Fine Grid around absolute best --------------------------
    print(chi("\n  PHASE 6/7: Fine Grid Search (tiny steps around best)"))
    div(100)
    t6 = time.time()

    bp = overall_best_params
    # Generate fine values: 3 points centered on best for each param (3^9 = 19683 max)
    def _fine_range(val, lo, hi, step):
        pts = [val - step, val, val + step]
        return sorted(set(max(lo, min(hi, round(v, 4))) for v in pts))

    fine_k    = _fine_range(bp[0], 3, 80, 0.8)
    fine_ha   = _fine_range(bp[1], 0, 150, 1.5)
    fine_pb   = _fine_range(bp[2], 0, 50, 1.0)
    fine_rf   = _fine_range(bp[3], 0, 50, 1.0)
    fine_tf   = _fine_range(bp[4], 0, 60, 1.5)
    fine_pf   = _fine_range(bp[5], 0, 60, 1.5)
    fine_phca = _fine_range(bp[6], 0.0, 1.0, 0.03)
    fine_sos  = _fine_range(bp[7], 0, 40, 1.5)
    fine_fw   = _fine_range(bp[8], 0, 20, 1.0)

    fine_total = (len(fine_k) * len(fine_ha) * len(fine_pb) * len(fine_rf) *
                  len(fine_tf) * len(fine_pf) * len(fine_phca) * len(fine_sos) * len(fine_fw))
    print("  %d fine combos (3^9 max, deduped)" % fine_total)

    fine_best_score, fine_best_params, fine_best_met = 1e9, None, {}
    fine_rows = []

    for i, (k, ha, pb, rf, tf, pf, phca, sos, fw) in enumerate(
        product(fine_k, fine_ha, fine_pb, fine_rf, fine_tf, fine_pf,
                fine_phca, fine_sos, fine_fw), 1
    ):
        score, met = _eval([k, ha, pb, rf, tf, pf, phca, sos, fw])
        if score < 1e8:
            fine_rows.append({"k": k, "home_adv": ha, "player_boost": pb,
                              "rest_factor": rf, "travel_factor": tf,
                              "pace_factor": pf, "playoff_hca_factor": phca,
                              "sos_factor": sos, "form_weight": fw,
                              "accuracy": met.get("accuracy", 0),
                              "log_loss": met.get("log_loss", 0),
                              "brier": met.get("brier", 0), "score": -score})
        is_best = score < fine_best_score
        if is_best:
            fine_best_score = score
            fine_best_params = np.array([k, ha, pb, rf, tf, pf, phca, sos, fw], dtype=float)
            fine_best_met = met
        if is_best or i % 500 == 0 or i == fine_total:
            flag = cok(" <- BEST") if is_best else ""
            print("  %5d/%d  Acc=%.2f%% Br=%.4f%s"
                  % (i, fine_total, met.get("accuracy", 0), met.get("brier", 0), flag), flush=True)

    if fine_rows:
        pd.DataFrame(fine_rows).to_csv("mlb_super_fine_grid.csv", index=False)

    t6_elapsed = time.time() - t6
    phase_times.append(("Fine Grid", t6_elapsed))
    if fine_best_params is not None:
        winners.append((fine_best_score, fine_best_params, "Fine-Grid", fine_best_met))
        print("  Fine grid done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
              % (t6_elapsed, -fine_best_score, fine_best_met.get("accuracy", 0)))
        if fine_best_score < overall_best_score:
            overall_best_score = fine_best_score
            overall_best_params = fine_best_params.copy()

    # -- Cleanup temp files -----------------------------------------------
    for tmp in ["temp_super.csv", "temp_super_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    # -- Select the overall winner ----------------------------------------
    winners.sort(key=lambda x: x[0])
    champion_score, champion_params, champion_src, champion_met = winners[0]

    # -- Apply winner BEFORE validation (so validation uses winning params)
    print("\n" + "=" * 100)
    print(chi("  SUPER-OPTIMIZE: Applying best settings before validation"))
    print("=" * 100)
    print("  %s from %s: %s" % (cok("CHAMPION"), chi(champion_src), _fmt(champion_params)))
    print("  Acc=%s  LogLoss=%.4f  Brier=%.4f"
          % (cok("%.2f%%" % champion_met.get("accuracy", 0)),
             champion_met.get("log_loss", 0), champion_met.get("brier", 0)))
    _apply_best_settings(_params_dict(champion_params), csv_file)

    # -- Phase 7: Validation ----------------------------------------------
    print(chi("\n  PHASE 7/7: Validation (Purged CV + PBO + Monte Carlo)"))
    div(100)
    t7 = time.time()

    # 7a: Purged walk-forward CV
    print(chi("\n  7a. Purged Walk-Forward Cross-Validation"))
    try:
        purged_walk_forward_cv(csv_file, k_folds=5, embargo_games=5)
    except Exception as e:
        print(cwarn("  Purged CV error: %s" % e))

    # 7b: PBO (needs grid search CSV)
    print(chi("\n  7b. Probability of Backtest Overfitting"))
    try:
        pbo_file = "mlb_super_grid.csv"
        if os.path.exists(pbo_file):
            probability_of_backtest_overfitting(pbo_file)
        elif os.path.exists("mlb_grid_search.csv"):
            probability_of_backtest_overfitting()
        else:
            print(cdim("  No grid CSV for PBO — skipped"))
    except Exception as e:
        print(cwarn("  PBO error: %s" % e))

    # 7c: Monte Carlo permutation test (reduced to 200 for speed)
    print(chi("\n  7c. Monte Carlo Permutation Test (200 permutations)"))
    try:
        monte_carlo_permutation_test(csv_file, n_permutations=200)
    except Exception as e:
        print(cwarn("  Monte Carlo error: %s" % e))

    t7_elapsed = time.time() - t7
    phase_times.append(("Validation", t7_elapsed))

    # -- Final Summary ----------------------------------------------------
    total_time = time.time() - t_start
    total_evals = eval_count[0]

    print("\n" + "=" * 100)
    print(chi("  SUPER-OPTIMIZE COMPLETE"))
    print("=" * 100)

    # Results table
    print("\n  %-14s  %-8s  %-8s  %-8s  %-8s  %s"
          % (chi("Phase"), chi("Score"), chi("Acc%"), chi("LogLoss"), chi("Brier"), chi("Parameters")))
    div(100)

    for score, params, source, met in winners:
        is_champ = (score == champion_score and source == champion_src)
        flag = cok(" ** WINNER") if is_champ else ""
        print("  %-14s  %8.4f  %7.2f%%  %8.4f  %8.4f  %s%s"
              % (source, -score, met.get("accuracy", 0), met.get("log_loss", 0),
                 met.get("brier", 0), _fmt(params), flag))

    div(100)
    print("\n  Phase Timing:")
    for name, elapsed in phase_times:
        print("    %-20s %6.0fs  (%d min)" % (name, elapsed, elapsed / 60))
    print("    %-20s %6.0fs  (%d min)" % ("TOTAL", total_time, total_time / 60))
    print("  Total evaluations: %d" % total_evals)

    # Confirm settings are saved
    print("\n  %s" % cok("Settings saved to mlb_elo_settings.json (auto-loads on every start)"))
    print("  %s" % cok("Platt scaler refitted and saved to mlb_platt_scaler.json"))
    print("\n  Winning parameters (%s):" % champion_src)
    print("    %s" % _fmt(champion_params))
    print("    Accuracy: %s  LogLoss: %.4f  Brier: %.4f"
          % (cok("%.2f%%" % champion_met.get("accuracy", 0)),
             champion_met.get("log_loss", 0), champion_met.get("brier", 0)))

    # DSR on the fine grid results
    if fine_rows:
        scores_arr = np.array([r["score"] for r in fine_rows])
        dsr = deflated_sharpe_ratio(scores_arr)
        if dsr is not None:
            if dsr >= 1.96:
                print("    DSR: %.2f %s" % (dsr, cok("(significant at 95%)")))
            else:
                print("    DSR: %.2f %s" % (dsr, cwarn("(not significant)")))

    print("=" * 100)
    return (champion_score, champion_params, champion_src, champion_met)
