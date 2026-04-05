"""Quick sweep of only the new/zero params to find their optimal values fast."""

import sys, time, logging
import numpy as np
import pandas as pd
from collections import defaultdict

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from elo_model import MLBElo
from data_players import load_player_stats, load_fangraphs_pitching
from build_model import _calc_altitude_bonus
from platt import regress_ratings_to_mean

# Only sweep these params (all currently zero or needing re-check)
TARGET_PARAMS = {
    "mov_cap":            (0.0,  20.0,  1.0),
    "east_travel_penalty": (0.0, 50.0,  2.0),
    "series_adaptation":  (0.0, 50.0,  2.0),
    "interleague_factor": (0.0, 50.0,  2.0),
    "bullpen_factor":     (0.0, 50.0,  2.0),
    "opp_pitcher_factor": (0.0, 50.0,  2.0),
    "altitude_factor":    (0.0, 10.0,  0.5),
    "home_road_factor":   (0.0, 50.0,  2.0),
    "homestand_factor":   (0.0, 50.0,  2.0),
    "road_trip_factor":   (0.0, 50.0,  2.0),
    "win_streak_factor":  (0.0, 50.0,  2.0),
    "season_phase_factor": (0.0, 50.0, 2.0),
    "scoring_consistency_factor": (0.0, 50.0, 2.0),
    "mean_reversion":     (0.0, 50.0,  2.0),
    "player_boost":       (0.0, 50.0,  2.0),
    "park_factor_weight": (0.0, 10.0,  0.5),
    "rest_advantage_cap": (0.0, 20.0,  1.0),
}

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
}

_games_df = None
_player_df = None
_adv_df = None
_altitude_bonus = None
_fg_pitching = None


def _preload_data():
    global _games_df, _player_df, _adv_df, _altitude_bonus, _fg_pitching
    print("Loading game data ...")
    _games_df = pd.read_csv(GAMES_FILE)
    if "neutral_site" not in _games_df.columns:
        _games_df["neutral_site"] = False
    _games_df["_date_parsed"] = pd.to_datetime(_games_df["date"], errors="coerce")
    print(f"  {len(_games_df)} games loaded.")
    _player_df = load_player_stats()
    from data_players import load_advanced_stats
    _adv_df = load_advanced_stats()
    _fg_pitching = load_fangraphs_pitching()
    _altitude_bonus = _calc_altitude_bonus(GAMES_FILE)


def evaluate(settings):
    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = MLBElo(**elo_kwargs)
    season_regress = settings.get("season_regress", 0.33)
    if _altitude_bonus:
        model._altitude_bonus = _altitude_bonus
    from config import build_league_map
    model._league_membership = build_league_map()
    if _player_df is not None and not _player_df.empty:
        model.set_player_stats(_player_df)

    correct = 0; total = 0; prev_season = None
    for _, row in _games_df.iterrows():
        game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
        if game_date is not None:
            row_season = game_date.year
            if prev_season is not None and row_season != prev_season:
                if season_regress > 0:
                    model.ratings = defaultdict(
                        lambda: model.base_rating,
                        regress_ratings_to_mean(dict(model.ratings), factor=season_regress)
                    )
                model.regress_pitcher_ratings(factor=0.5)
                if _fg_pitching is not None and not _fg_pitching.empty:
                    model.set_fip_priors(_fg_pitching, season=prev_season, only_new=True)
            prev_season = row_season

        home_starter = str(row.get("home_starter", "") or "").strip()
        away_starter = str(row.get("away_starter", "") or "").strip()
        home_win_prob = model.win_prob(
            row["home_team"], row["away_team"], team_a_home=True,
            neutral_site=bool(row["neutral_site"]), calibrated=False,
            game_date=game_date, use_injuries=False,
            home_starter=home_starter, away_starter=away_starter,
        )
        pred_winner = row["home_team"] if home_win_prob >= 0.5 else row["away_team"]
        actual_winner = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
        if pred_winner == actual_winner:
            correct += 1
        total += 1
        model.update_game(
            row["home_team"], row["away_team"],
            row["home_score"], row["away_score"],
            neutral_site=bool(row["neutral_site"]), game_date=game_date,
            home_starter=home_starter, away_starter=away_starter,
        )
    return 100.0 * correct / total if total else 0.0


def sweep_param(param, settings, best_acc):
    lo, hi, step = TARGET_PARAMS[param]
    orig = settings[param]
    best_val = orig
    # Coarse
    candidates = np.arange(lo, hi + step * 0.01, step)
    print(f"    Coarse sweep {param}: {lo} to {hi} step {step} ({len(candidates)} pts)")
    for val in candidates:
        val = round(float(val), 4)
        if abs(val - orig) < 1e-9:
            continue
        settings[param] = val
        acc = evaluate(settings)
        tag = " ** BEST" if acc > best_acc else ""
        print(f"    {param}={val:.4f}  acc={acc:.4f}%{tag}")
        if acc > best_acc:
            best_acc = acc
            best_val = val
    settings[param] = best_val

    # Fine sweep
    fine_step = step / 4.0
    sweep_lo = max(lo, best_val - 2*step)
    sweep_hi = min(hi, best_val + 2*step)
    candidates2 = np.arange(sweep_lo, sweep_hi + fine_step*0.01, fine_step)
    print(f"    Fine sweep {param}: {sweep_lo:.2f} to {sweep_hi:.2f} step {fine_step:.4f} ({len(candidates2)} pts)")
    for val in candidates2:
        val = round(float(val), 4)
        if abs(val - best_val) < 1e-9:
            continue
        settings[param] = val
        acc = evaluate(settings)
        tag = " ** BEST" if acc > best_acc else ""
        print(f"    {param}={val:.4f}  acc={acc:.4f}%{tag}")
        if acc > best_acc:
            best_acc = acc
            best_val = val
    settings[param] = best_val
    return best_val, best_acc


def main():
    logging.basicConfig(level=logging.WARNING)
    _preload_data()
    settings = load_elo_settings()
    print("\nBaseline eval ...")
    baseline = evaluate(settings)
    print(f"Baseline: {baseline:.4f}%\n")

    best_acc = baseline
    improved = {}

    for param in TARGET_PARAMS:
        print(f"\n{'='*50}")
        print(f"Sweeping: {param}  (current={settings[param]})")
        t0 = time.time()
        val, acc = sweep_param(param, settings, best_acc)
        elapsed = time.time() - t0
        if acc > best_acc:
            print(f"  >> IMPROVED {param}: {settings[param]} -> {val}  ({best_acc:.4f}% -> {acc:.4f}%)  [{elapsed:.1f}s]")
            improved[param] = (settings[param], val, acc - best_acc)
            best_acc = acc
        else:
            print(f"  >> No improvement for {param}  [{elapsed:.1f}s]")

    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print(f"Baseline: {baseline:.4f}%  ->  Final: {best_acc:.4f}%  ({best_acc - baseline:+.4f}%)")
    print("\nImproved params:")
    for p, (old, new, gain) in improved.items():
        print(f"  {p:25s}: {old} -> {new}  (+{gain:.4f}%)")
    print("\nAll param values:")
    for p in TARGET_PARAMS:
        print(f"  {p:25s} = {settings[p]}")

    print("\nSaving ...")
    save_elo_settings(settings)
    print("Done.")


if __name__ == "__main__":
    main()
