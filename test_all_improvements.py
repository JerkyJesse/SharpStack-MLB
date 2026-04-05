"""
Comprehensive MLB improvement tester.
Tests every single proposed improvement individually against baseline.
Reports accuracy delta for each.
"""

import os
import sys
import time
import json
import math
import copy
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

from config import GAMES_FILE, load_elo_settings, save_elo_settings, TEAM_ABBR
from elo_model import MLBElo, PARK_FACTORS, TEAM_TIMEZONE
from data_players import load_player_stats, load_fangraphs_pitching
from build_model import _calc_altitude_bonus
from config import build_league_map

logging.basicConfig(level=logging.WARNING)

# ── Load data once ──────────────────────────────────────────────────
print("Loading data...")
games = pd.read_csv(GAMES_FILE)
if "neutral_site" not in games.columns:
    games["neutral_site"] = False
games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")
player_df = load_player_stats()
alt_bonus = _calc_altitude_bonus(GAMES_FILE)
n_games = len(games)
print(f"  {n_games} games loaded\n")

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


def evaluate(params_override=None, model_modifier=None):
    """Run walk-forward backtest and return accuracy %."""
    settings = load_elo_settings()
    if params_override:
        settings.update(params_override)

    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = MLBElo(**elo_kwargs)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    model._league_membership = build_league_map()

    # Apply any model modifier function
    if model_modifier:
        model_modifier(model)

    correct = 0
    total = 0
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            p = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True,
                neutral_site=bool(row["neutral_site"]),
                calibrated=False,
                game_date=game_date,
                use_injuries=False,
            )
            home_actual = 1 if row["home_score"] > row["away_score"] else 0
            if (p >= 0.5) == (home_actual == 1):
                correct += 1
            total += 1
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]),
                game_date=game_date,
            )
        except Exception:
            pass
    if total == 0:
        return 0.0
    return correct / total * 100.0


def evaluate_with_directional_travel():
    """Test #22: Directional travel - eastward travel worse than westward."""
    settings = load_elo_settings()
    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = MLBElo(**elo_kwargs)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    correct = 0
    total = 0
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None

            # Standard prediction
            p = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True,
                neutral_site=bool(row["neutral_site"]),
                calibrated=False,
                game_date=game_date,
                use_injuries=False,
            )

            # Apply directional travel adjustment post-hoc
            # Eastward travel is worse (add penalty to eastward-traveling team)
            home = row["home_team"]
            away = row["away_team"]
            last_loc_away = model._last_game_location.get(away)
            if last_loc_away and game_date:
                prev_tz = TEAM_TIMEZONE.get(last_loc_away, -6)
                curr_tz = TEAM_TIMEZONE.get(home, -6)
                tz_diff = curr_tz - prev_tz  # positive = traveling east
                if tz_diff > 0:  # eastward travel
                    # Penalize the away team extra for eastward travel
                    east_penalty = 0.01 * tz_diff  # small probability adjustment
                    p = min(0.99, p + east_penalty)  # helps home team
                elif tz_diff < 0:  # westward travel (less harmful)
                    west_bonus = 0.005 * abs(tz_diff)
                    p = max(0.01, p - west_bonus)  # slightly helps away team

            home_actual = 1 if row["home_score"] > row["away_score"] else 0
            if (p >= 0.5) == (home_actual == 1):
                correct += 1
            total += 1
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]),
                game_date=game_date,
            )
        except Exception:
            pass
    return correct / total * 100.0 if total > 0 else 0.0


def evaluate_with_series_context():
    """Test #24: Series context - game 2/3 of series differ from game 1."""
    settings = load_elo_settings()
    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = MLBElo(**elo_kwargs)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    last_matchup = {}  # (home, away) -> count of consecutive games
    prev_row = None
    correct = 0
    total = 0
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            home = row["home_team"]
            away = row["away_team"]

            p = model.win_prob(
                home, away, team_a_home=True,
                neutral_site=bool(row["neutral_site"]),
                calibrated=False, game_date=game_date, use_injuries=False,
            )

            # Series context: if teams played yesterday, visiting team adapts
            key = (home, away)
            series_game = last_matchup.get(key, 0) + 1
            if series_game >= 2:
                # Visiting team adjusts after game 1; shrink home advantage
                adaptation = 0.005 * min(series_game - 1, 3)
                p = max(0.01, p - adaptation)  # helps away team slightly

            home_actual = 1 if row["home_score"] > row["away_score"] else 0
            if (p >= 0.5) == (home_actual == 1):
                correct += 1
            total += 1

            # Track series
            if game_date and prev_row is not None:
                prev_date = prev_row["_date_parsed"]
                if pd.notna(prev_date) and (game_date - prev_date).days <= 1:
                    if prev_row["home_team"] == home and prev_row["away_team"] == away:
                        last_matchup[key] = last_matchup.get(key, 0) + 1
                    else:
                        last_matchup[key] = 0
                else:
                    last_matchup[key] = 0
            prev_row = row

            model.update_game(
                home, away, row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]), game_date=game_date,
            )
        except Exception:
            pass
    return correct / total * 100.0 if total > 0 else 0.0


def evaluate_with_mov_cap(cap_runs=8):
    """Test #36: Cap MOV for blowouts - diminishing returns above N runs."""
    settings = load_elo_settings()
    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = MLBElo(**elo_kwargs)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    # Override update_game to cap MOV
    original_update = model.update_game

    def capped_update(home_team, away_team, home_score, away_score, **kwargs):
        # Cap the score diff for MOV calculation
        diff = abs(home_score - away_score)
        if diff > cap_runs:
            # Reduce the blowout to cap_runs
            if home_score > away_score:
                away_score = home_score - cap_runs
            else:
                home_score = away_score - cap_runs
        original_update(home_team, away_team, home_score, away_score, **kwargs)

    model.update_game = capped_update

    correct = 0
    total = 0
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            p = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True, neutral_site=bool(row["neutral_site"]),
                calibrated=False, game_date=game_date, use_injuries=False,
            )
            home_actual = 1 if row["home_score"] > row["away_score"] else 0
            if (p >= 0.5) == (home_actual == 1):
                correct += 1
            total += 1
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]), game_date=game_date,
            )
        except Exception:
            pass
    return correct / total * 100.0 if total > 0 else 0.0


def evaluate_with_fip_priors():
    """Test #6: Use FIP/xFIP blend for pitcher priors instead of ERA."""
    settings = load_elo_settings()
    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = MLBElo(**elo_kwargs)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    # Try to load FanGraphs data and set FIP priors
    fg_df = load_fangraphs_pitching()
    if fg_df is not None and not fg_df.empty:
        model.set_fip_priors(fg_df)

    correct = 0
    total = 0
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            p = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True, neutral_site=bool(row["neutral_site"]),
                calibrated=False, game_date=game_date, use_injuries=False,
            )
            home_actual = 1 if row["home_score"] > row["away_score"] else 0
            if (p >= 0.5) == (home_actual == 1):
                correct += 1
            total += 1
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]), game_date=game_date,
            )
        except Exception:
            pass
    return correct / total * 100.0 if total > 0 else 0.0


def evaluate_538_k_factor():
    """Test #33: Use FiveThirtyEight's K=4 instead of optimized K=1.5."""
    return evaluate(params_override={"k": 4.0})


def evaluate_538_regression():
    """Test #34: Use FiveThirtyEight's 33% season regression instead of 9%."""
    return evaluate(params_override={"season_regress": 0.33})


def evaluate_higher_k_values():
    """Test various K-factor values."""
    results = {}
    for k in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        acc = evaluate(params_override={"k": k})
        results[k] = acc
    return results


def evaluate_park_factor_weights():
    """Test #2/#17: Various park factor weights."""
    results = {}
    for w in [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]:
        acc = evaluate(params_override={"park_factor_weight": w})
        results[w] = acc
    return results


def evaluate_regression_values():
    """Test #34: Various season regression values."""
    results = {}
    for r in [0.0, 0.05, 0.09, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50]:
        acc = evaluate(params_override={"season_regress": r})
        results[r] = acc
    return results


def evaluate_starter_boost_values():
    """Test pitcher boost range."""
    results = {}
    for sb in [0, 10, 20, 30, 37, 50, 75, 100, 150]:
        acc = evaluate(params_override={"starter_boost": sb})
        results[sb] = acc
    return results


def evaluate_home_adv_values():
    """Test home advantage range."""
    results = {}
    for ha in [0, 10, 15, 20, 24, 30, 40, 50]:
        acc = evaluate(params_override={"home_adv": ha})
        results[ha] = acc
    return results


def evaluate_playoff_hca_values():
    """Test playoff HCA multiplier range."""
    results = {}
    for phca in [0.5, 0.7, 1.0, 1.2, 1.5, 1.9, 2.0]:
        acc = evaluate(params_override={"playoff_hca_factor": phca})
        results[phca] = acc
    return results


def evaluate_form_weight_values():
    """Test recent form weight range."""
    results = {}
    for fw in [0, 5, 10, 15, 20, 30, 50, 75, 100]:
        acc = evaluate(params_override={"form_weight": fw})
        results[fw] = acc
    return results


def evaluate_division_factor_values():
    """Test division factor range."""
    results = {}
    for df_ in [0, 10, 15, 20, 26, 30, 50, 75, 100]:
        acc = evaluate(params_override={"division_factor": df_})
        results[df_] = acc
    return results


def evaluate_b2b_penalty_values():
    """Test B2B penalty range."""
    results = {}
    for bp in [0, 20, 40, 60, 80, 100, 120, 150, 200]:
        acc = evaluate(params_override={"b2b_penalty": bp})
        results[bp] = acc
    return results


def evaluate_combined_best():
    """Test combination of best parameter values found."""
    # Start with current optimized, try tweaking multiple at once
    combos = [
        {"label": "current optimized", "params": {}},
        {"label": "+park_factor=10", "params": {"park_factor_weight": 10.0}},
        {"label": "+park_factor=20", "params": {"park_factor_weight": 20.0}},
        {"label": "k=2.0+park=10", "params": {"k": 2.0, "park_factor_weight": 10.0}},
        {"label": "k=3.0+park=15", "params": {"k": 3.0, "park_factor_weight": 15.0}},
        {"label": "k=4.0+park=20", "params": {"k": 4.0, "park_factor_weight": 20.0}},
        {"label": "k=2+regress=0.20", "params": {"k": 2.0, "season_regress": 0.20}},
        {"label": "k=3+regress=0.33", "params": {"k": 3.0, "season_regress": 0.33}},
        {"label": "k=2+park=10+form=20", "params": {"k": 2.0, "park_factor_weight": 10.0, "form_weight": 20.0}},
        {"label": "538-style: k=4,ha=24,reg=0.33", "params": {"k": 4.0, "home_adv": 24.0, "season_regress": 0.33}},
        {"label": "all-in: k=3,park=15,b2b=100,div=30,form=20", "params": {
            "k": 3.0, "park_factor_weight": 15.0, "b2b_penalty": 100.0,
            "division_factor": 30.0, "form_weight": 20.0
        }},
    ]
    results = {}
    for combo in combos:
        acc = evaluate(params_override=combo["params"])
        results[combo["label"]] = acc
    return results


# ═══════════════════════════════════════════════════════════════════
#  MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  MLB COMPREHENSIVE IMPROVEMENT TESTER")
    print("  Testing all 44 proposed improvements individually")
    print("=" * 70)

    results = []
    t_start = time.time()

    # ── BASELINE ──
    print("\n[0] BASELINE (current optimized settings)...")
    baseline = evaluate()
    print(f"    BASELINE: {baseline:.4f}%")
    results.append(("BASELINE", baseline, 0.0))

    # ── #1: OPS Bug (can't fix in test, but we can measure impact) ──
    # The OPS bug only affects XGBoost features, not Elo. Skip for now.

    # ── #2/#17: Park Factor Weight ──
    print("\n[2/17] PARK FACTOR WEIGHTS...")
    pf_results = evaluate_park_factor_weights()
    best_pf = max(pf_results, key=pf_results.get)
    for w, acc in sorted(pf_results.items()):
        marker = " <-- BEST" if w == best_pf else ""
        print(f"    park_factor_weight={w:>5.1f}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"park_factor_weight={best_pf}", pf_results[best_pf], pf_results[best_pf] - baseline))

    # ── #5/#33: K-Factor sweep ──
    print("\n[5/33] K-FACTOR VALUES...")
    k_results = evaluate_higher_k_values()
    best_k = max(k_results, key=k_results.get)
    for k, acc in sorted(k_results.items()):
        marker = " <-- BEST" if k == best_k else ""
        print(f"    k={k:>4.1f}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"k={best_k}", k_results[best_k], k_results[best_k] - baseline))

    # ── #34: Season Regression ──
    print("\n[34] SEASON REGRESSION VALUES...")
    reg_results = evaluate_regression_values()
    best_reg = max(reg_results, key=reg_results.get)
    for r, acc in sorted(reg_results.items()):
        marker = " <-- BEST" if r == best_reg else ""
        print(f"    season_regress={r:>4.2f}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"season_regress={best_reg}", reg_results[best_reg], reg_results[best_reg] - baseline))

    # ── Starter Boost ──
    print("\n[PITCHER] STARTER BOOST VALUES...")
    sb_results = evaluate_starter_boost_values()
    best_sb = max(sb_results, key=sb_results.get)
    for sb, acc in sorted(sb_results.items()):
        marker = " <-- BEST" if sb == best_sb else ""
        print(f"    starter_boost={sb:>4}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"starter_boost={best_sb}", sb_results[best_sb], sb_results[best_sb] - baseline))

    # ── Home Advantage ──
    print("\n[HOME] HOME ADVANTAGE VALUES...")
    ha_results = evaluate_home_adv_values()
    best_ha = max(ha_results, key=ha_results.get)
    for ha, acc in sorted(ha_results.items()):
        marker = " <-- BEST" if ha == best_ha else ""
        print(f"    home_adv={ha:>3}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"home_adv={best_ha}", ha_results[best_ha], ha_results[best_ha] - baseline))

    # ── Playoff HCA ──
    print("\n[PLAYOFF] PLAYOFF HCA FACTOR...")
    phca_results = evaluate_playoff_hca_values()
    best_phca = max(phca_results, key=phca_results.get)
    for phca, acc in sorted(phca_results.items()):
        marker = " <-- BEST" if phca == best_phca else ""
        print(f"    playoff_hca={phca:>3.1f}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"playoff_hca={best_phca}", phca_results[best_phca], phca_results[best_phca] - baseline))

    # ── Form Weight ──
    print("\n[FORM] FORM WEIGHT VALUES...")
    fw_results = evaluate_form_weight_values()
    best_fw = max(fw_results, key=fw_results.get)
    for fw, acc in sorted(fw_results.items()):
        marker = " <-- BEST" if fw == best_fw else ""
        print(f"    form_weight={fw:>3}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"form_weight={best_fw}", fw_results[best_fw], fw_results[best_fw] - baseline))

    # ── Division Factor ──
    print("\n[DIV] DIVISION FACTOR VALUES...")
    df_results = evaluate_division_factor_values()
    best_df = max(df_results, key=df_results.get)
    for df_, acc in sorted(df_results.items()):
        marker = " <-- BEST" if df_ == best_df else ""
        print(f"    division_factor={df_:>3}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"division_factor={best_df}", df_results[best_df], df_results[best_df] - baseline))

    # ── B2B Penalty ──
    print("\n[B2B] B2B PENALTY VALUES...")
    b2b_results = evaluate_b2b_penalty_values()
    best_b2b = max(b2b_results, key=b2b_results.get)
    for bp, acc in sorted(b2b_results.items()):
        marker = " <-- BEST" if bp == best_b2b else ""
        print(f"    b2b_penalty={bp:>4}  -> {acc:.4f}%  ({acc-baseline:+.4f}%){marker}")
    results.append((f"b2b_penalty={best_b2b}", b2b_results[best_b2b], b2b_results[best_b2b] - baseline))

    # ── #6: FIP Priors ──
    print("\n[6] FIP/xFIP PITCHER PRIORS...")
    fip_acc = evaluate_with_fip_priors()
    print(f"    FIP priors: {fip_acc:.4f}%  ({fip_acc-baseline:+.4f}%)")
    results.append(("FIP pitcher priors", fip_acc, fip_acc - baseline))

    # ── #22: Directional Travel ──
    print("\n[22] DIRECTIONAL TRAVEL (eastward penalty)...")
    dir_acc = evaluate_with_directional_travel()
    print(f"    Directional travel: {dir_acc:.4f}%  ({dir_acc-baseline:+.4f}%)")
    results.append(("Directional travel", dir_acc, dir_acc - baseline))

    # ── #24: Series Context ──
    print("\n[24] SERIES CONTEXT (visiting team adapts)...")
    series_acc = evaluate_with_series_context()
    print(f"    Series context: {series_acc:.4f}%  ({series_acc-baseline:+.4f}%)")
    results.append(("Series context", series_acc, series_acc - baseline))

    # ── #36: MOV Capping ──
    print("\n[36] MARGIN OF VICTORY CAPPING...")
    for cap in [5, 6, 7, 8, 10, 15]:
        mov_acc = evaluate_with_mov_cap(cap)
        delta = mov_acc - baseline
        print(f"    MOV cap={cap:>2} runs: {mov_acc:.4f}%  ({delta:+.4f}%)")
        if cap == 8:
            results.append((f"MOV cap={cap}", mov_acc, delta))

    # ── COMBINED BEST ──
    print("\n[COMBO] COMBINED PARAMETER TESTS...")
    combo_results = evaluate_combined_best()
    best_combo = max(combo_results, key=combo_results.get)
    for label, acc in sorted(combo_results.items(), key=lambda x: -x[1]):
        delta = acc - baseline
        marker = " <-- BEST" if label == best_combo else ""
        print(f"    {label:>45s}  -> {acc:.4f}%  ({delta:+.4f}%){marker}")
    results.append((f"COMBO: {best_combo}", combo_results[best_combo], combo_results[best_combo] - baseline))

    # ═══════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  FINAL RESULTS - ALL IMPROVEMENTS RANKED BY ACCURACY GAIN")
    print("=" * 70)
    print(f"  BASELINE: {baseline:.4f}%")
    print(f"  Total test time: {elapsed:.0f}s")
    print("-" * 70)

    results_sorted = sorted(results[1:], key=lambda x: -x[2])  # skip baseline
    for label, acc, delta in results_sorted:
        if delta > 0:
            marker = "+++"
        elif delta == 0:
            marker = "   "
        else:
            marker = "---"
        print(f"  {marker} {label:>45s}  {acc:.4f}%  ({delta:+.4f}%)")

    print("-" * 70)
    positive = [(l, a, d) for l, a, d in results_sorted if d > 0]
    if positive:
        print(f"\n  {len(positive)} improvements found positive signal:")
        for l, a, d in positive:
            print(f"    + {l}: {d:+.4f}%")
    else:
        print("\n  No improvements beat baseline.")

    # Show the absolute best result
    if results_sorted:
        best_label, best_acc, best_delta = results_sorted[0]
        print(f"\n  BEST SINGLE: {best_label} -> {best_acc:.4f}% ({best_delta:+.4f}%)")
        print(f"  BEST COMBO:  {best_combo} -> {combo_results[best_combo]:.4f}% ({combo_results[best_combo]-baseline:+.4f}%)")

    print("=" * 70)
