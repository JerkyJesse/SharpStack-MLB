#!/usr/bin/env python3
"""
Accuracy-focused optimizer for MLB.
Maximizes binary prediction accuracy (not calibration).

Usage: python accuracy_optimize.py
"""
import sys, os, time
import numpy as np
from scipy.optimize import differential_evolution

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from data_players import load_player_stats, build_league_player_scores
from build_model import _calc_altitude_bonus
from elo_model import MLBElo as EloClass
from backtest import backtest_model

sport_prefix = "MLB"

# Load prerequisites
settings = load_elo_settings()
base = settings.get("base_rating", 1500.0)
use_mov = settings.get("use_mov", True)
player_df = load_player_stats()
has_players = not player_df.empty
prebuilt_scores = build_league_player_scores(player_df) if has_players else {}
alt_bonus = _calc_altitude_bonus(GAMES_FILE)

_ELO_KEYS = {"base_rating", "k", "home_adv", "use_mov", "player_boost",
             "rest_factor", "form_weight", "travel_factor", "sos_factor",
             "playoff_hca_factor", "pace_factor"}

bounds = [
    (2, 12),     # K
    (10, 45),    # HomeAdv
    (5, 35),     # PlayerBoost
    (2, 25),     # RestFactor
    (0, 30),     # TravelFactor
    (0, 35),     # PaceFactor
    (0.3, 0.9),  # PlayoffHCA
    (0, 25),     # SOSFactor
    (0, 25),     # FormWeight
]
PARAM_NAMES = ["k", "home_adv", "player_boost", "rest_factor",
               "travel_factor", "pace_factor", "playoff_hca_factor",
               "sos_factor", "form_weight"]

eval_count = [0]
best_acc = [0.0]
best_params = [None]
t_start = time.time()

def objective(params):
    k, ha, pb, rf, tf, pf, phca, sos, fw = params
    m = EloClass(base_rating=base, k=float(k), home_adv=float(ha), use_mov=use_mov,
                 player_boost=float(pb), rest_factor=float(rf),
                 travel_factor=float(tf), pace_factor=float(pf),
                 playoff_hca_factor=float(phca), sos_factor=float(sos),
                 form_weight=float(fw))
    m._altitude_bonus = alt_bonus
    if has_players:
        m._player_scores = prebuilt_scores
    ok, met = backtest_model(GAMES_FILE, "temp_acc_opt.csv", "temp_acc_cal.csv", model=m)
    eval_count[0] += 1
    if not ok:
        return 1.0

    acc = met["accuracy"]
    # Primary: maximize accuracy. Secondary: minimize calibration error as tiebreaker.
    # Score = (100 - accuracy) + small calibration penalty
    score = (100.0 - acc) + met["brier"] * 5.0

    if acc > best_acc[0]:
        best_acc[0] = acc
        best_params[0] = params.copy()
        elapsed = time.time() - t_start
        print("  [%4d] NEW BEST: %.2f%%  Brier=%.4f  LL=%.4f  (%.0fs)  K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f SOS=%.1f FW=%.1f"
              % (eval_count[0], acc, met["brier"], met["log_loss"], elapsed,
                 k, ha, pb, rf, tf, pf, phca, sos, fw))
    elif eval_count[0] % 100 == 0:
        elapsed = time.time() - t_start
        print("  [%4d] Acc=%.2f%%  best=%.2f%%  (%.0fs)" % (eval_count[0], acc, best_acc[0], elapsed))

    return score

print("=" * 80)
print("  %s ACCURACY-FOCUSED OPTIMIZER (9 params, differential_evolution)" % sport_prefix)
print("  Objective: maximize accuracy with calibration tiebreaker")
print("  Bounds: %s" % [(PARAM_NAMES[i], bounds[i]) for i in range(len(bounds))])
print("=" * 80)

# Phase 1: Broad search
print("\n  PHASE 1: Broad genetic search (popsize=30, maxiter=60)")
result = differential_evolution(
    objective, bounds=bounds,
    maxiter=60, popsize=30,
    tol=1e-7, seed=42,
    mutation=(0.5, 1.5), recombination=0.9,
    polish=True, workers=1,
)

phase1_best = best_params[0].copy()
phase1_acc = best_acc[0]
print("  Phase 1 done: %.2f%% accuracy" % phase1_acc)

# Phase 2: Tightened search around the best
print("\n  PHASE 2: Tightened search around best")
def tighten(val, lo, hi, hw):
    return (max(lo, val - hw), min(hi, val + hw))

bp = phase1_best
tight_bounds = [
    tighten(bp[0], bounds[0][0], bounds[0][1], 3),     # K
    tighten(bp[1], bounds[1][0], bounds[1][1], 8),     # HomeAdv
    tighten(bp[2], bounds[2][0], bounds[2][1], 6),     # PlayerBoost
    tighten(bp[3], bounds[3][0], bounds[3][1], 8),     # RestFactor
    tighten(bp[4], bounds[4][0], bounds[4][1], 8),     # TravelFactor
    tighten(bp[5], bounds[5][0], bounds[5][1], 8),     # PaceFactor
    tighten(bp[6], bounds[6][0], bounds[6][1], 0.15),  # PlayoffHCA
    tighten(bp[7], bounds[7][0], bounds[7][1], 8),     # SOSFactor
    tighten(bp[8], bounds[8][0], bounds[8][1], 8),     # FormWeight
]

result2 = differential_evolution(
    objective, bounds=tight_bounds,
    maxiter=40, popsize=25,
    tol=1e-7, seed=123,
    mutation=(0.5, 1.5), recombination=0.9,
    polish=True, workers=1,
)

final_params = best_params[0]
final_acc = best_acc[0]

# Save best settings
best_dict = dict(zip(PARAM_NAMES, final_params))
settings = load_elo_settings()
settings.update(best_dict)
save_elo_settings(settings)

# Final backtest with Platt fitting
from backtest import _apply_best_settings
_apply_best_settings(best_dict, GAMES_FILE)

# Cleanup
for tmp in ["temp_acc_opt.csv", "temp_acc_cal.csv"]:
    try:
        os.remove(tmp)
    except Exception:
        pass

elapsed = time.time() - t_start
print("\n" + "=" * 80)
print("  %s ACCURACY OPTIMIZATION COMPLETE" % sport_prefix)
print("  Best accuracy: %.2f%%" % final_acc)
print("  Total evaluations: %d" % eval_count[0])
print("  Total time: %.0f seconds" % elapsed)
print("  Parameters: %s" % {k: round(v, 2) for k, v in best_dict.items()})
print("=" * 80)
