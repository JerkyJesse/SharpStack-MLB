#!/usr/bin/env python3
"""Non-interactive auto-optimization script. Run directly: python run_optimize.py"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import GAMES_FILE, load_elo_settings
from backtest import auto_optimize, backtest_model
from build_model import build_model

print("=" * 60)
print("  MLB AUTO-OPTIMIZE (non-interactive)")
print("=" * 60)

settings = load_elo_settings()
if float(settings.get("starting_balance", 0)) <= 0:
    settings["starting_balance"] = 1000.0
    from config import save_elo_settings
    save_elo_settings(settings)

result = auto_optimize(GAMES_FILE)

if result:
    _, params, source, met = result
    print("\n" + "=" * 60)
    print("  MLB FINAL: %s winner" % source)
    print("  Accuracy: %.2f%%" % met.get("accuracy", 0))
    print("  LogLoss:  %.4f" % met.get("log_loss", 0))
    print("  Brier:    %.4f" % met.get("brier", 0))
    print("=" * 60)
else:
    print("  Optimization failed!")
