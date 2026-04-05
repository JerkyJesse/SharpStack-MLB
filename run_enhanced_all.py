#!/usr/bin/env python3
"""
Run enhanced (XGBoost) backtest with multiple elo_weights.
Finds the best elo_weight for MLB.

Usage: python run_enhanced_all.py
"""
import sys, os, time

from config import GAMES_FILE
from enhanced_model import run_enhanced_backtest

sport_prefix = "MLB"

# Test multiple elo_weights to find the best for this sport
weights_to_test = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

print("=" * 80)
print("  %s ENHANCED MODEL SWEEP (XGBoost ensemble)" % sport_prefix)
print("  Testing elo_weights: %s" % weights_to_test)
print("=" * 80)

best_acc = 0.0
best_weight = 0.5
results = []

for ew in weights_to_test:
    t0 = time.time()
    print("\n  Testing elo_weight=%.1f (%d%% Elo / %d%% XGBoost)..."
          % (ew, int(ew * 100), int((1 - ew) * 100)))

    result = run_enhanced_backtest(
        csv_file=GAMES_FILE,
        min_train=200,
        retrain_every=50,
        elo_weight=ew,
        label="ew%.1f" % ew,
        time_decay=False,
    )

    elapsed = time.time() - t0
    if result is not None:
        acc = result.get("acc_cal", result.get("accuracy", 0))
        brier = result.get("brier_cal", result.get("brier", 0))
        ll = result.get("ll_cal", result.get("log_loss", 0))
        results.append((ew, acc, brier, ll))
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            best_weight = ew
        flag = " ** BEST" if is_best else ""
        print("  elo_weight=%.1f  Acc=%.2f%%  Brier=%.4f  LL=%.4f  (%.0fs)%s"
              % (ew, acc, brier, ll, elapsed, flag))

# Also test time-decay mode with the best weight
print("\n  Testing time-decay mode with elo_weight=%.1f..." % best_weight)
t0 = time.time()
result = run_enhanced_backtest(
    csv_file=GAMES_FILE,
    min_train=200,
    retrain_every=50,
    elo_weight=best_weight,
    label="decay",
    time_decay=True,
)
if result is not None:
    acc = result.get("acc_cal", result.get("accuracy", 0))
    results.append(("decay", acc, result.get("brier_cal", result.get("brier", 0)), result.get("ll_cal", result.get("log_loss", 0))))
    if acc > best_acc:
        best_acc = acc
        best_weight = "decay"
    print("  Time-decay  Acc=%.2f%%  Brier=%.4f  (%.0fs)"
          % (acc, result.get("brier_cal", result.get("brier", 0)), time.time() - t0))

print("\n" + "=" * 80)
print("  %s ENHANCED MODEL RESULTS" % sport_prefix)
print("  %-12s  %8s  %8s  %8s" % ("Weight", "Acc%", "Brier", "LogLoss"))
print("-" * 50)
for ew, acc, brier, ll in results:
    flag = " **" if (isinstance(ew, float) and ew == best_weight) or ew == best_weight else ""
    print("  %-12s  %7.2f%%  %8.4f  %8.4f%s" % (str(ew), acc, brier, ll, flag))
print("=" * 80)
print("  Best: elo_weight=%s -> %.2f%% accuracy" % (best_weight, best_acc))
