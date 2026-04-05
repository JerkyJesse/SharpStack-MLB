"""Coordinate descent (single-parameter-at-a-time) optimizer for MLB Elo model.

Sweeps each parameter individually while keeping all others fixed,
then refines with a fine-grained pass around the best value found.
Repeats until no parameter improves or max passes reached.
"""

import sys
import time
import copy
import logging

import numpy as np
import pandas as pd

from collections import defaultdict

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from elo_model import MLBElo
from data_players import load_player_stats, load_fangraphs_pitching
from build_model import _calc_altitude_bonus
from platt import regress_ratings_to_mean


# ── Elo constructor keys ────────────────────────────────────────────
_ELO_KEYS = {
    "base_rating", "k", "home_adv", "use_mov", "player_boost",
    "starter_boost", "rest_factor", "form_weight", "travel_factor",
    "sos_factor", "pace_factor", "playoff_hca_factor",
    "division_factor", "mean_reversion",
    "pyth_factor", "home_road_factor", "mov_base",
    "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
    "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
    "rest_advantage_cap", "park_factor_weight",
    "mov_cap", "east_travel_penalty", "series_adaptation",
    "interleague_factor", "bullpen_factor", "opp_pitcher_factor",
    "k_decay", "surprise_k",
}

# ── Parameter search ranges: (min, max, coarse_step) ────────────────
PARAM_RANGES = {
    "k":                  (0.1, 100.0,  2.0),
    "home_adv":           (0.0, 200.0,  5.0),
    "player_boost":       (0.0, 200.0,  5.0),
    "starter_boost":      (0.0, 200.0,  5.0),
    "rest_factor":        (0.0, 200.0,  5.0),
    "form_weight":        (0.0, 200.0,  5.0),
    "travel_factor":      (0.0, 200.0,  5.0),
    "sos_factor":         (0.0, 200.0,  5.0),
    "pace_factor":        (0.0, 200.0,  5.0),
    "playoff_hca_factor": (0.0,   2.0,  0.1),
    "division_factor":    (0.0, 200.0,  5.0),
    "mean_reversion":     (0.0, 200.0,  5.0),
    "season_regress":     (0.0,   1.0,  0.05),
    "pyth_factor":        (0.0, 200.0,  5.0),
    "home_road_factor":   (0.0, 200.0,  5.0),
    "b2b_penalty":        (0.0, 200.0,  5.0),
    "road_trip_factor":   (0.0, 200.0,  5.0),
    "homestand_factor":   (0.0, 200.0,  5.0),
    "win_streak_factor":  (0.0, 200.0,  5.0),
    "altitude_factor":    (0.0,  20.0,  1.0),
    "season_phase_factor": (0.0, 200.0, 5.0),
    "scoring_consistency_factor": (0.0, 200.0, 5.0),
    "rest_advantage_cap": (0.0,  30.0,  1.0),
    "park_factor_weight": (0.0,  20.0,  1.0),
    "mov_cap":            (0.0,  20.0,  1.0),
    "east_travel_penalty": (0.0, 200.0, 5.0),
    "series_adaptation":  (0.0, 200.0,  5.0),
    "interleague_factor": (0.0, 200.0,  5.0),
    "bullpen_factor":     (0.0, 200.0,  5.0),
    "opp_pitcher_factor": (0.0, 200.0,  5.0),
    "k_decay":            (0.0,   5.0,  0.2),
    "surprise_k":         (0.0,  10.0,  0.5),
}

MAX_PASSES = 5


# ── Pre-loaded data (set once in main) ──────────────────────────────
_games_df = None        # type: pd.DataFrame
_player_df = None       # type: pd.DataFrame
_adv_df = None          # type: pd.DataFrame
_altitude_bonus = None  # type: dict
_fg_pitching = None     # type: pd.DataFrame  (FanGraphs FIP/xFIP data)


def _preload_data(csv_file=GAMES_FILE):
    """Load games CSV and player stats once."""
    global _games_df, _player_df, _adv_df, _altitude_bonus, _fg_pitching
    print("Loading game data from", csv_file, "...")
    _games_df = pd.read_csv(csv_file)
    if "neutral_site" not in _games_df.columns:
        _games_df["neutral_site"] = False
    _games_df["_date_parsed"] = pd.to_datetime(_games_df["date"], errors="coerce")
    print("  {} games loaded.".format(len(_games_df)))

    print("Loading player stats ...")
    _player_df = load_player_stats()
    if _player_df is not None and not _player_df.empty:
        print("  Player stats loaded ({} rows).".format(len(_player_df)))
    else:
        print("  No player stats available.")

    print("Loading pitcher stats ...")
    from data_players import load_advanced_stats
    _adv_df = load_advanced_stats()
    if _adv_df is not None and not _adv_df.empty:
        print("  Pitcher stats loaded ({} rows).".format(len(_adv_df)))
    else:
        print("  No pitcher stats available.")

    print("Loading FanGraphs pitching data (FIP/xFIP) ...")
    _fg_pitching = load_fangraphs_pitching()
    if _fg_pitching is not None and not _fg_pitching.empty:
        seasons = sorted(_fg_pitching["season"].unique()) if "season" in _fg_pitching.columns else []
        print("  FanGraphs pitching loaded ({} pitcher-seasons, seasons: {}).".format(
            len(_fg_pitching), seasons))
    else:
        print("  No FanGraphs pitching data available.")

    print("Calculating altitude bonus ...")
    _altitude_bonus = _calc_altitude_bonus(csv_file)
    if _altitude_bonus:
        print("  Altitude bonuses:", _altitude_bonus)
    else:
        print("  No altitude bonus applicable.")


def evaluate(settings):
    """Run a full walk-forward backtest and return accuracy percentage."""
    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = MLBElo(**elo_kwargs)
    season_regress = settings.get("season_regress", 0.33)

    if _altitude_bonus:
        model._altitude_bonus = _altitude_bonus

    # Initialize league membership for interleague detection
    from config import build_league_map
    model._league_membership = build_league_map()

    if _player_df is not None and not _player_df.empty:
        model.set_player_stats(_player_df)

    correct = 0
    total = 0
    prev_season = None

    for _, row in _games_df.iterrows():
        game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None

        # Regress ratings at season boundaries (team + pitcher)
        if game_date is not None:
            row_season = game_date.year
            if prev_season is not None and row_season != prev_season:
                if season_regress > 0:
                    model.ratings = defaultdict(
                        lambda: model.base_rating,
                        regress_ratings_to_mean(dict(model.ratings), factor=season_regress)
                    )
                model.regress_pitcher_ratings(factor=0.5)
                # Apply FIP-based priors for new pitchers from previous season
                if _fg_pitching is not None and not _fg_pitching.empty:
                    model.set_fip_priors(_fg_pitching, season=prev_season, only_new=True)
            prev_season = row_season

        home_starter = str(row.get("home_starter", "") or "").strip()
        away_starter = str(row.get("away_starter", "") or "").strip()

        home_win_prob = model.win_prob(
            row["home_team"], row["away_team"],
            team_a_home=True,
            neutral_site=bool(row["neutral_site"]),
            calibrated=False,
            game_date=game_date,
            use_injuries=False,
            home_starter=home_starter, away_starter=away_starter,
        )

        pred_winner = row["home_team"] if home_win_prob >= 0.5 else row["away_team"]
        home_actual = 1 if row["home_score"] > row["away_score"] else 0
        actual_winner = row["home_team"] if home_actual == 1 else row["away_team"]

        if pred_winner == actual_winner:
            correct += 1
        total += 1

        model.update_game(
            row["home_team"], row["away_team"],
            row["home_score"], row["away_score"],
            neutral_site=bool(row["neutral_site"]),
            game_date=game_date,
            home_starter=home_starter, away_starter=away_starter,
        )

    if total == 0:
        return 0.0
    return 100.0 * correct / total


def _coarse_sweep(param, settings, current_best_acc):
    """Sweep the full range for *param* at coarse resolution.

    Returns (best_value, best_accuracy).
    """
    lo, hi, step = PARAM_RANGES[param]
    original_val = settings[param]
    best_val = original_val
    best_acc = current_best_acc

    candidates = np.arange(lo, hi + step * 0.01, step)
    n = len(candidates)

    for idx, val in enumerate(candidates):
        val = round(float(val), 4)
        if abs(val - original_val) < 1e-9:
            # Already evaluated as baseline
            continue
        settings[param] = val
        acc = evaluate(settings)
        tag = " ** NEW BEST" if acc > best_acc else ""
        print("    [{}/{}] {} = {:<8.4f}  acc = {:.4f}%{}".format(
            idx + 1, n, param, val, acc, tag))
        if acc > best_acc:
            best_acc = acc
            best_val = val

    # Restore best into settings
    settings[param] = best_val
    return best_val, best_acc


def _fine_sweep(param, settings, coarse_best_val, current_best_acc):
    """Refine around coarse_best_val with step/5 resolution, +/- 2*step."""
    lo, hi, step = PARAM_RANGES[param]
    fine_step = step / 5.0
    sweep_lo = max(lo, coarse_best_val - 2 * step)
    sweep_hi = min(hi, coarse_best_val + 2 * step)

    best_val = coarse_best_val
    best_acc = current_best_acc

    candidates = np.arange(sweep_lo, sweep_hi + fine_step * 0.01, fine_step)
    n = len(candidates)

    print("    Fine sweep: {} in [{:.4f}, {:.4f}] step {:.4f}  ({} pts)".format(
        param, sweep_lo, sweep_hi, fine_step, n))

    for idx, val in enumerate(candidates):
        val = round(float(val), 4)
        if abs(val - coarse_best_val) < 1e-9:
            continue
        settings[param] = val
        acc = evaluate(settings)
        tag = " ** NEW BEST" if acc > best_acc else ""
        print("    [{}/{}] {} = {:<8.4f}  acc = {:.4f}%{}".format(
            idx + 1, n, param, val, acc, tag))
        if acc > best_acc:
            best_acc = acc
            best_val = val

    settings[param] = best_val
    return best_val, best_acc


def run_coordinate_descent(csv_file=GAMES_FILE):
    """Run coordinate descent optimization."""
    print("=" * 65)
    print("  MLB Elo -- Coordinate Descent Parameter Optimizer")
    print("=" * 65)
    print()

    # Pre-load all data
    _preload_data(csv_file)
    print()

    # Load current settings
    settings = load_elo_settings()
    param_names = list(PARAM_RANGES.keys())

    # Baseline accuracy
    print("Evaluating baseline accuracy ...")
    baseline_acc = evaluate(settings)
    print("Baseline accuracy: {:.4f}%".format(baseline_acc))
    print("Current parameter values:")
    for p in param_names:
        print("  {:22s} = {}".format(p, settings[p]))
    print()

    global_best_acc = baseline_acc

    for pass_num in range(1, MAX_PASSES + 1):
        pass_start_acc = global_best_acc
        print("-" * 65)
        print("PASS {}/{}  (current best accuracy: {:.4f}%)".format(
            pass_num, MAX_PASSES, global_best_acc))
        print("-" * 65)

        improved_any = False

        for param in param_names:
            print()
            print("  Sweeping '{}'  (current value: {})".format(
                param, settings[param]))
            t0 = time.time()

            # Coarse sweep
            coarse_val, coarse_acc = _coarse_sweep(
                param, settings, global_best_acc)

            # Fine sweep around best coarse value
            fine_val, fine_acc = _fine_sweep(
                param, settings, coarse_val, coarse_acc)

            elapsed = time.time() - t0

            if fine_acc > global_best_acc:
                print("  >> '{}' improved: {} -> {}  (acc {:.4f}% -> {:.4f}%)  [{:.1f}s]".format(
                    param, settings.get(param), fine_val,
                    global_best_acc, fine_acc, elapsed))
                settings[param] = fine_val
                global_best_acc = fine_acc
                improved_any = True
            else:
                print("  >> '{}' unchanged at {}  [{:.1f}s]".format(
                    param, settings[param], elapsed))

        pass_improvement = global_best_acc - pass_start_acc
        print()
        print("Pass {} complete.  Accuracy: {:.4f}%  (improvement this pass: {:+.4f}%)".format(
            pass_num, global_best_acc, pass_improvement))

        if not improved_any:
            print("No parameter improved -- stopping early.")
            break

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 65)
    total_improvement = global_best_acc - baseline_acc
    print("Baseline accuracy:  {:.4f}%".format(baseline_acc))
    print("Final accuracy:     {:.4f}%".format(global_best_acc))
    print("Total improvement:  {:+.4f}%".format(total_improvement))
    print()
    print("Optimized parameters:")
    for p in param_names:
        print("  {:22s} = {}".format(p, settings[p]))
    print()

    # Save
    print("Saving best settings ...")
    save_elo_settings(settings)
    print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    optimize()
