#!/usr/bin/env python3
"""Master optimization script -- runs ALL optimization methods sequentially.

Works in any sport directory (NHL, NFL, MLB, NBA). Each sport has identical
file structure; the script auto-detects sport from the directory name and
imports the correct modules.

Usage:
    cd NHL   (or NFL, MLB, NBA)
    python master_optimize.py            # run all phases
    python master_optimize.py --skip 6   # skip phase 6 (mega backtest)
    python master_optimize.py --only 1 2 # run only phases 1 and 2

Phases:
     1  Quick Optimizer (12-param DE, accuracy focus)
     2  Accuracy Optimize (two-phase DE, 9 params)
     3  Auto Optimize (grid -> genetic -> bayesian pipeline)
     4  Super Optimize (exhaustive multi-round)
     5  Coordinate Descent (single-param-at-a-time sweeps)
     6  Mega Backtest (all 35 models, walk-forward)
     7  Mega Quick Optimize (Phase 0 + 1)
     8  Mega Full Optimize (all 7 phases)
     9  Mega Ablation (test each model's contribution)
    10  Final backtest with Platt fitting
    11  Summary
"""

import os
import sys
import time
import argparse
import traceback
import subprocess
import json
import logging

# Ensure we run from the script's own directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Detect sport from directory name ────────────────────────────
_SPORT_MAP = {"nhl": "nhl", "nfl": "nfl", "mlb": "mlb", "nba": "nba"}
_dir_name = os.path.basename(_SCRIPT_DIR).lower()
SPORT = _SPORT_MAP.get(_dir_name, "nhl")
SPORT_UPPER = SPORT.upper()

# ── Imports (sport-generic) ─────────────────────────────────────
from config import GAMES_FILE, load_elo_settings, save_elo_settings
from data_players import load_player_stats

# Import Elo class dynamically based on sport
import importlib as _importlib
_elo_mod = _importlib.import_module("elo_model")
_elo_class_name = "%sElo" % SPORT_UPPER
EloClass = getattr(_elo_mod, _elo_class_name)


# ── Formatting helpers ──────────────────────────────────────────
try:
    from color_helpers import cok, cerr, cwarn, chi, cdim, cbold, div
except ImportError:
    def cok(t): return str(t)
    def cerr(t): return str(t)
    def cwarn(t): return str(t)
    def chi(t): return str(t)
    def cdim(t): return str(t)
    def cbold(t): return str(t)
    def div(n=80): print("-" * n)


def banner(text, width=72):
    """Print a prominent phase banner."""
    print()
    print("=" * width)
    print("  %s" % text)
    print("=" * width)


def sub_banner(text, width=60):
    """Print a sub-section banner."""
    print()
    print("-" * width)
    print("  %s" % text)
    print("-" * width)


def elapsed_str(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return "%.1fs" % seconds
    elif seconds < 3600:
        return "%dm %ds" % (seconds // 60, seconds % 60)
    else:
        return "%dh %dm %ds" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


# ── Result tracking ─────────────────────────────────────────────
_results = {}  # phase_num -> {status, elapsed, metrics, error}


def record_result(phase, status, elapsed, metrics=None, error=None):
    """Record the result of a phase."""
    _results[phase] = {
        "status": status,
        "elapsed": elapsed,
        "metrics": metrics or {},
        "error": error,
    }


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Quick Optimizer (12-param DE)
# ══════════════════════════════════════════════════════════════════

def phase_1_quick_optimizer():
    """Run quick_optimizer.py: 12-param differential evolution."""
    banner("PHASE 1: Quick Optimizer (12-param DE, accuracy focus)")
    print("  Focus: accuracy | maxiter: 30 | popsize: 20")

    from quick_optimizer import run_optimizer

    t0 = time.time()
    best_params = run_optimizer(
        csv_file=GAMES_FILE,
        maxiter=30,
        popsize=20,
        focus="accuracy",
    )
    elapsed = time.time() - t0

    if best_params:
        print(cok("\n  Phase 1 complete: settings saved"))
        record_result(1, "OK", elapsed, {"params_optimized": len(best_params)})
    else:
        print(cwarn("\n  Phase 1: no improvement found"))
        record_result(1, "NO_IMPROVEMENT", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Accuracy Optimize (two-phase DE, 9 params)
# ══════════════════════════════════════════════════════════════════

def phase_2_accuracy_optimize():
    """Run accuracy_optimize.py via subprocess (module-level code)."""
    banner("PHASE 2: Accuracy Optimize (two-phase DE, 9 params)")
    print("  Phase 2a: Broad search (popsize=30, maxiter=60)")
    print("  Phase 2b: Tightened search around best")
    print("  (Running as subprocess -- module has top-level code)")

    t0 = time.time()
    script_path = os.path.join(_SCRIPT_DIR, "accuracy_optimize.py")
    if not os.path.exists(script_path):
        print(cwarn("  accuracy_optimize.py not found -- skipping"))
        record_result(2, "SKIPPED", 0, error="File not found")
        return

    proc = subprocess.run(
        [sys.executable, script_path],
        cwd=_SCRIPT_DIR,
        timeout=7200,  # 2 hour timeout
        capture_output=False,
    )
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(cok("\n  Phase 2 complete: settings saved"))
        record_result(2, "OK", elapsed)
    else:
        print(cerr("\n  Phase 2 failed (exit code %d)" % proc.returncode))
        record_result(2, "FAILED", elapsed, error="exit code %d" % proc.returncode)


# ══════════════════════════════════════════════════════════════════
# PHASE 3: Auto Optimize (grid -> genetic -> bayesian)
# ══════════════════════════════════════════════════════════════════

def phase_3_auto_optimize():
    """Run auto_optimize(): grid -> genetic -> bayesian pipeline."""
    banner("PHASE 3: Auto Optimize (grid -> genetic -> bayesian)")
    print("  Optimizes ALL 21 prediction-relevant Elo parameters")

    from backtest import auto_optimize

    t0 = time.time()
    result = auto_optimize(GAMES_FILE)
    elapsed = time.time() - t0

    if result:
        _, params, source, met = result
        print(cok("\n  Phase 3 complete: %s winner" % source))
        print("  Accuracy: %.2f%% | LogLoss: %.4f | Brier: %.4f" % (
            met.get("accuracy", 0), met.get("log_loss", 0), met.get("brier", 0)))
        record_result(3, "OK", elapsed, {
            "accuracy": met.get("accuracy", 0),
            "log_loss": met.get("log_loss", 0),
            "brier": met.get("brier", 0),
            "source": source,
        })
    else:
        print(cwarn("\n  Phase 3: optimization returned no result"))
        record_result(3, "NO_RESULT", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 4: Super Optimize (exhaustive multi-round)
# ══════════════════════════════════════════════════════════════════

def phase_4_super_optimize():
    """Run super_optimize(): exhaustive multi-round optimization."""
    banner("PHASE 4: Super Optimize (exhaustive multi-round)")
    print("  Searches ALL 21 params across multiple rounds")
    print("  This is the longest-running Elo optimizer.")

    from backtest import super_optimize

    t0 = time.time()
    result = super_optimize(GAMES_FILE)
    elapsed = time.time() - t0

    if result:
        print(cok("\n  Phase 4 complete"))
        record_result(4, "OK", elapsed, result if isinstance(result, dict) else {})
    else:
        print(cwarn("\n  Phase 4: optimization returned no result"))
        record_result(4, "NO_RESULT", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 5: Coordinate Descent (single-param sweeps)
# ══════════════════════════════════════════════════════════════════

def phase_5_coordinate_descent():
    """Run coordinate descent across all parameters."""
    banner("PHASE 5: Coordinate Descent (single-param sweeps)")
    print("  Sweeps each parameter individually, up to %d passes" % 5)

    from single_param_opt import run_coordinate_descent

    t0 = time.time()
    run_coordinate_descent(csv_file=GAMES_FILE)
    elapsed = time.time() - t0

    print(cok("\n  Phase 5 complete: settings saved"))
    record_result(5, "OK", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 6: Mega Backtest (all 35 models)
# ══════════════════════════════════════════════════════════════════

def phase_6_mega_backtest():
    """Run mega-ensemble backtest with all enabled models."""
    banner("PHASE 6: Mega Backtest (all 35 models)")
    print("  Walk-forward backtest of the full mega-ensemble")

    from mega_backtest import run_mega_backtest

    settings = load_elo_settings()
    player_df = load_player_stats()

    t0 = time.time()
    result = run_mega_backtest(
        GAMES_FILE,
        sport=SPORT,
        elo_model_class=EloClass,
        elo_settings=settings,
        player_df=player_df,
        verbose=True,
    )
    elapsed = time.time() - t0

    if result:
        print(cok("\n  Phase 6 complete"))
        print("  Accuracy: %.2f%% | LogLoss: %.4f | Brier: %.4f" % (
            result.get("accuracy", 0), result.get("log_loss", 0), result.get("brier", 0)))
        record_result(6, "OK", elapsed, {
            "accuracy": result.get("accuracy", 0),
            "log_loss": result.get("log_loss", 0),
            "brier": result.get("brier", 0),
            "n_predictions": result.get("n_predictions", 0),
        })
    else:
        print(cwarn("\n  Phase 6: mega backtest returned no result"))
        record_result(6, "NO_RESULT", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 7: Mega Quick Optimize (Phases 0 + 1)
# ══════════════════════════════════════════════════════════════════

def phase_7_mega_quick_optimize():
    """Run mega optimizer phases 0 + 1 (baseline + per-model solo)."""
    banner("PHASE 7: Mega Quick Optimize (Phase 0 + 1)")
    print("  Phase 0: Elo-only baseline")
    print("  Phase 1: Per-model solo optimization with coordinate descent")

    from mega_optimizer import run_quick_optimize

    settings = load_elo_settings()
    player_df = load_player_stats()

    t0 = time.time()
    result = run_quick_optimize(
        GAMES_FILE,
        sport=SPORT,
        elo_model_class=EloClass,
        elo_settings=settings,
        player_df=player_df,
    )
    elapsed = time.time() - t0

    if result and result.get("best_results"):
        br = result["best_results"]
        print(cok("\n  Phase 7 complete"))
        print("  Accuracy: %.2f%% | LogLoss: %.4f | Brier: %.4f" % (
            br.get("accuracy", 0), br.get("log_loss", 0), br.get("brier", 0)))
        record_result(7, "OK", elapsed, {
            "accuracy": br.get("accuracy", 0),
            "log_loss": br.get("log_loss", 0),
            "brier": br.get("brier", 0),
            "n_evaluations": result.get("n_evaluations", 0),
        })
    else:
        print(cwarn("\n  Phase 7: mega quick optimize returned no result"))
        record_result(7, "NO_RESULT", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 8: Mega Full Optimize (all 7 phases)
# ══════════════════════════════════════════════════════════════════

def phase_8_mega_full_optimize():
    """Run mega optimizer all 7 phases (0-6)."""
    banner("PHASE 8: Mega Full Optimize (all 7 phases)")
    print("  Phase 0: Elo baseline")
    print("  Phase 1: Per-model solo optimization")
    print("  Phase 2: Head-to-head tournament")
    print("  Phase 3: Meta-learner + global tuning")
    print("  Phase 4: Combined DE fine-tuning")
    print("  Phase 5: Final ablation")
    print("  Phase 6: Validation")
    print("  WARNING: This will take a VERY long time.")

    from mega_optimizer import run_mega_optimize

    settings = load_elo_settings()
    player_df = load_player_stats()

    t0 = time.time()
    result = run_mega_optimize(
        GAMES_FILE,
        sport=SPORT,
        elo_model_class=EloClass,
        elo_settings=settings,
        player_df=player_df,
        phases=[0, 1, 2, 3, 4, 5, 6],
        verbose=True,
    )
    elapsed = time.time() - t0

    if result and result.get("best_results"):
        br = result["best_results"]
        print(cok("\n  Phase 8 complete"))
        print("  Accuracy: %.2f%% | LogLoss: %.4f | Brier: %.4f" % (
            br.get("accuracy", 0), br.get("log_loss", 0), br.get("brier", 0)))
        print("  Total evaluations: %d" % result.get("n_evaluations", 0))
        record_result(8, "OK", elapsed, {
            "accuracy": br.get("accuracy", 0),
            "log_loss": br.get("log_loss", 0),
            "brier": br.get("brier", 0),
            "n_evaluations": result.get("n_evaluations", 0),
        })
    else:
        print(cwarn("\n  Phase 8: mega full optimize returned no result"))
        record_result(8, "NO_RESULT", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 9: Mega Ablation
# ══════════════════════════════════════════════════════════════════

def phase_9_mega_ablation():
    """Run ablation study: test each model's contribution."""
    banner("PHASE 9: Mega Ablation Study")
    print("  Tests each model's individual contribution")
    print("  Auto-prunes models that hurt overall performance")

    from mega_optimizer import run_single_model_optimize

    settings = load_elo_settings()
    player_df = load_player_stats()

    t0 = time.time()
    result = run_single_model_optimize(
        GAMES_FILE,
        sport=SPORT,
        elo_model_class=EloClass,
        elo_settings=settings,
        player_df=player_df,
        verbose=True,
    )
    elapsed = time.time() - t0

    if result:
        n_hurting = result.get("n_hurting", 0)
        print(cok("\n  Phase 9 complete"))
        if n_hurting > 0:
            print(cwarn("  %d model(s) auto-disabled (hurting performance)" % n_hurting))
        else:
            print("  All enabled models are contributing positively")
        record_result(9, "OK", elapsed, {
            "n_hurting": n_hurting,
            "baseline_obj": result.get("baseline_obj", 0),
        })
    else:
        print(cwarn("\n  Phase 9: ablation returned no result"))
        record_result(9, "NO_RESULT", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 10: Final Backtest with Platt Fitting
# ══════════════════════════════════════════════════════════════════

def phase_10_final_backtest():
    """Run final backtest with Platt fitting to get calibrated metrics."""
    banner("PHASE 10: Final Backtest with Platt Fitting")
    print("  Runs walk-forward backtest with fit_platt=True")
    print("  This produces the definitive calibrated metrics.")

    from backtest import backtest_model

    t0 = time.time()
    ok, metrics = backtest_model(GAMES_FILE, fit_platt=True)
    elapsed = time.time() - t0

    if ok and metrics:
        print(cok("\n  Phase 10 complete: Platt scaler fitted and saved"))
        print("  Accuracy: %.2f%%" % metrics.get("accuracy", 0))
        print("  LogLoss:  %.4f" % metrics.get("log_loss", 0))
        print("  Brier:    %.4f" % metrics.get("brier", 0))
        if "ece" in metrics:
            print("  ECE:      %.4f" % metrics.get("ece", 0))
        if "bss" in metrics:
            print("  BSS:      %.4f" % metrics.get("bss", 0))
        record_result(10, "OK", elapsed, {
            "accuracy": metrics.get("accuracy", 0),
            "log_loss": metrics.get("log_loss", 0),
            "brier": metrics.get("brier", 0),
            "ece": metrics.get("ece", 0),
            "bss": metrics.get("bss", 0),
            "n_games": metrics.get("n_games", 0),
        })
    else:
        print(cerr("\n  Phase 10 failed: backtest returned no metrics"))
        record_result(10, "FAILED", elapsed)


# ══════════════════════════════════════════════════════════════════
# PHASE 11: Summary
# ══════════════════════════════════════════════════════════════════

def phase_11_summary(total_elapsed):
    """Print summary of all optimization results."""
    banner("MASTER OPTIMIZATION SUMMARY (%s)" % SPORT_UPPER, width=72)

    phase_names = {
        1:  "Quick Optimizer (12-param DE)",
        2:  "Accuracy Optimize (two-phase DE)",
        3:  "Auto Optimize (grid/genetic/bayesian)",
        4:  "Super Optimize (exhaustive)",
        5:  "Coordinate Descent",
        6:  "Mega Backtest (all models)",
        7:  "Mega Quick Optimize (Phase 0+1)",
        8:  "Mega Full Optimize (all phases)",
        9:  "Mega Ablation",
        10: "Final Backtest + Platt Fit",
    }

    print()
    print("  %-4s  %-40s  %-10s  %s" % ("#", "Phase", "Status", "Time"))
    print("  " + "-" * 68)

    for phase_num in sorted(phase_names.keys()):
        name = phase_names[phase_num]
        if phase_num in _results:
            r = _results[phase_num]
            status = r["status"]
            etime = elapsed_str(r["elapsed"])
        else:
            status = "SKIPPED"
            etime = "-"
        print("  %-4d  %-40s  %-10s  %s" % (phase_num, name, status, etime))

    # Print final metrics from phase 10 if available
    if 10 in _results and _results[10]["status"] == "OK":
        met = _results[10]["metrics"]
        print()
        print("  " + "=" * 50)
        print("  FINAL CALIBRATED METRICS")
        print("  " + "-" * 50)
        print("  Accuracy: %.2f%%" % met.get("accuracy", 0))
        print("  LogLoss:  %.4f" % met.get("log_loss", 0))
        print("  Brier:    %.4f" % met.get("brier", 0))
        if met.get("ece"):
            print("  ECE:      %.4f" % met["ece"])
        if met.get("bss"):
            print("  BSS:      %.4f" % met["bss"])
        print("  Games:    %d" % met.get("n_games", 0))
        print("  " + "=" * 50)

    # Print current settings
    settings = load_elo_settings()
    print()
    print("  Current Elo Settings:")
    print("  " + "-" * 50)
    key_params = [
        "k", "home_adv", "player_boost", "starter_boost", "rest_factor",
        "form_weight", "travel_factor", "sos_factor", "pace_factor",
        "playoff_hca_factor", "division_factor", "mean_reversion",
        "b2b_penalty", "road_trip_factor", "homestand_factor",
        "win_streak_factor", "altitude_factor", "season_phase_factor",
        "scoring_consistency_factor", "rest_advantage_cap", "overtime_factor",
    ]
    for k in key_params:
        if k in settings:
            print("    %-30s = %s" % (k, settings[k]))

    print()
    print("  Total wall time: %s" % elapsed_str(total_elapsed))

    # Count successes/failures
    ok_count = sum(1 for r in _results.values() if r["status"] == "OK")
    fail_count = sum(1 for r in _results.values() if r["status"] in ("FAILED", "ERROR"))
    skip_count = 10 - len(_results)  # 10 runnable phases (not counting summary)

    print("  Phases: %d OK, %d failed, %d skipped" % (ok_count, fail_count, skip_count))
    print()

    # Save summary to JSON
    summary_file = os.path.join(_SCRIPT_DIR, "%s_master_opt_summary.json" % SPORT)
    summary = {
        "sport": SPORT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_elapsed,
        "phases": {},
    }
    for phase_num, r in _results.items():
        summary["phases"][str(phase_num)] = {
            "status": r["status"],
            "elapsed_seconds": r["elapsed"],
            "metrics": r["metrics"],
            "error": r["error"],
        }
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print("  Summary saved to %s" % os.path.basename(summary_file))
    except Exception as e:
        print(cwarn("  Could not save summary: %s" % e))

    print("=" * 72)


# ══════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════

PHASE_MAP = {
    1:  phase_1_quick_optimizer,
    2:  phase_2_accuracy_optimize,
    3:  phase_3_auto_optimize,
    4:  phase_4_super_optimize,
    5:  phase_5_coordinate_descent,
    6:  phase_6_mega_backtest,
    7:  phase_7_mega_quick_optimize,
    8:  phase_8_mega_full_optimize,
    9:  phase_9_mega_ablation,
    10: phase_10_final_backtest,
}

ALL_PHASES = list(range(1, 11))


def main():
    parser = argparse.ArgumentParser(
        description="Master %s optimization -- run all optimizers sequentially." % SPORT_UPPER,
    )
    parser.add_argument(
        "--skip", type=int, nargs="*", default=[],
        help="Phase numbers to skip (e.g., --skip 6 8)",
    )
    parser.add_argument(
        "--only", type=int, nargs="*", default=None,
        help="Run ONLY these phases (e.g., --only 1 2 10)",
    )
    args = parser.parse_args()

    if args.only is not None:
        phases_to_run = [p for p in args.only if p in PHASE_MAP]
    else:
        phases_to_run = [p for p in ALL_PHASES if p not in (args.skip or [])]

    # Verify game data exists
    if not os.path.exists(GAMES_FILE):
        print(cerr("ERROR: Game data file not found: %s" % GAMES_FILE))
        print(cerr("Run main.py first to download game data."))
        sys.exit(1)

    banner("MASTER %s OPTIMIZATION" % SPORT_UPPER, width=72)
    print("  Sport:     %s" % SPORT_UPPER)
    print("  Data:      %s" % GAMES_FILE)
    print("  Directory: %s" % _SCRIPT_DIR)
    print("  Phases:    %s" % phases_to_run)
    print("  Start:     %s" % time.strftime("%Y-%m-%d %H:%M:%S"))

    total_start = time.time()

    for phase_num in phases_to_run:
        func = PHASE_MAP.get(phase_num)
        if func is None:
            print(cwarn("\n  Unknown phase %d -- skipping" % phase_num))
            continue

        try:
            func()
        except KeyboardInterrupt:
            print(cerr("\n\n  Keyboard interrupt -- stopping master optimization."))
            record_result(phase_num, "INTERRUPTED", time.time() - total_start)
            break
        except Exception as e:
            elapsed_so_far = time.time() - total_start
            error_msg = "%s: %s" % (type(e).__name__, e)
            print(cerr("\n  Phase %d FAILED with exception:" % phase_num))
            print(cerr("  %s" % error_msg))
            traceback.print_exc()
            record_result(phase_num, "ERROR", 0, error=error_msg)
            print(cwarn("  Continuing to next phase..."))

    total_elapsed = time.time() - total_start
    phase_11_summary(total_elapsed)


if __name__ == "__main__":
    main()
