"""Mega-Ensemble 7-Phase Per-Model Optimizer.

Phase 0: Elo-only Baseline -- establish the floor
Phase 1: Per-Model Solo Optimization -- Elo + one model at a time, coordinate descent
Phase 2: Head-to-Head Tournament -- pairs, triples, top-N combos
Phase 3: Meta-Learner + Global Tuning -- max_adj, meta_model, retrain_every, min_train, window
Phase 4: Combined DE Fine-Tuning -- differential evolution over top numeric params
Phase 5: Final Ablation -- test each model's contribution with fully tuned params
Phase 6: Validation -- run best config 5 times, report stability

Objective: minimize 8*LogLoss + 40*Brier (lower is better)
"""

import os
import sys
import json
import time
import logging
from itertools import product

import numpy as np
from tqdm import tqdm

_parent = os.path.dirname(__file__)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from mega_backtest import run_mega_backtest
from mega_config import (MODEL_HYPERPARAMS, get_model_hyperparams, get_all_tunable_params,
                         param_to_model, load_model_params, save_model_params,
                         load_model_switches, save_model_switches, is_model_enabled,
                         load_mega_params, save_mega_params, ALL_MODELS,
                         MODEL_REGISTRY)

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


# ── Settings file ────────────────────────────────────────────────
SETTINGS_FILE_TPL = "{sport}_mega_settings.json"


def _load_settings(sport, sport_dir):
    """Load saved mega-ensemble settings."""
    path = os.path.join(sport_dir, SETTINGS_FILE_TPL.format(sport=sport))
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_settings(sport, sport_dir, params):
    """Save mega-ensemble settings."""
    path = os.path.join(sport_dir, SETTINGS_FILE_TPL.format(sport=sport))
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


def _save_all(sport, sport_dir, params, switches):
    """Save params, model_params, and switches together."""
    path = _save_settings(sport, sport_dir, params)
    save_model_switches(sport, sport_dir, switches)
    return path


# ── Core evaluation function ─────────────────────────────────────

_eval_count = 0
_eval_start = None


def _eval(csv_file, sport, sport_dir, params, switches,
          elo_model_class=None, elo_settings=None, player_df=None,
          label=""):
    """Run mega backtest with given params, return (obj, metrics).

    Args:
        csv_file: path to games CSV
        sport: sport code
        sport_dir: directory for settings
        params: dict of mega params (may include model_params)
        switches: dict of model on/off switches
        elo_model_class: sport Elo class
        elo_settings: Elo settings dict
        player_df: player stats DataFrame
        label: label for logging

    Returns:
        (objective_value, metrics_dict) -- objective = LogLoss*8 + Brier*40
    """
    global _eval_count
    _eval_count += 1

    mp = dict(params)
    mp["model_switches"] = switches

    t0 = time.time()
    result = run_mega_backtest(
        csv_file, sport, elo_model_class, elo_settings, player_df,
        verbose=False, mega_params=mp,
    )
    dt = time.time() - t0

    if result is None or result.get("n_predictions", 0) == 0:
        return 1e9, {}

    obj = result["log_loss"] * 8.0 + result["brier"] * 40.0

    return obj, result


# ── Sport-specific defaults ──────────────────────────────────────

_RETRAIN_MAP = {
    "nfl": [16, 24, 32, 48, 64],
    "mlb": [40, 60, 80, 100, 120],
    "nba": [25, 40, 50, 75, 100],
    "nhl": [25, 40, 50, 75, 100],
}

_MIN_TRAIN_MAP = {
    "nfl": [80, 100, 120, 150, 180, 200],
    "mlb": [150, 200, 250, 300, 400, 500],
    "nba": [100, 150, 200, 250, 300],
    "nhl": [100, 150, 200, 250, 300],
}

_WINDOW_MAP = {
    "nfl": [3, 5, 7, 10],
    "mlb": [5, 7, 10, 15, 20],
    "nba": [5, 7, 10, 15],
    "nhl": [5, 7, 10, 15],
}


# ══════════════════════════════════════════════════════════════════
# Phase 0: Elo-only Baseline
# ══════════════════════════════════════════════════════════════════

def _phase0_baseline(csv_file, sport, sport_dir, base_params,
                     elo_model_class, elo_settings, player_df):
    """Run Elo-only baseline (all other models OFF)."""
    print()
    print(cbold("  Phase 0: Elo-only Baseline"))
    div(60)

    elo_only_switches = {m: False for m in ALL_MODELS}
    elo_only_switches["elo"] = True

    obj, met = _eval(csv_file, sport, sport_dir, base_params, elo_only_switches,
                     elo_model_class, elo_settings, player_df, "elo_baseline")

    if obj >= 1e9:
        print(cerr("  Elo baseline failed! Cannot proceed."))
        return obj, met

    print("  Elo-only baseline: Acc=%.2f%% LL=%.4f Brier=%.4f obj=%.4f" % (
        met.get("accuracy", 0), met.get("log_loss", 1),
        met.get("brier", 0.5), obj))

    return obj, met


# ══════════════════════════════════════════════════════════════════
# Phase 1: Per-Model Solo Optimization (THE BIG ONE)
# ══════════════════════════════════════════════════════════════════

def _phase1_per_model_solo(csv_file, sport, sport_dir, base_params,
                           elo_model_class, elo_settings, player_df,
                           elo_baseline_obj):
    """Optimize each model individually with Elo + that model only.

    For each testable model:
    1. Enable ONLY Elo + this model
    2. Get solo baseline (default hyperparams)
    3. Coordinate descent on this model's hyperparams (up to 3 passes)
    4. Also tune max_adj for this solo config
    5. Also tune meta-learner params for current meta_model
    6. Record final results and delta vs Elo baseline

    Returns:
        (model_results, all_model_params)
        model_results: {model_name: {obj, accuracy, log_loss, brier, delta, n_evals, skipped}}
        all_model_params: {model_name: {param: value, ...}}
    """
    print()
    print(cbold("  Phase 1: Per-Model Solo Optimization"))
    div(60)
    print("  Testing each model individually (Elo + model) with hyperparameter tuning")
    print("  Elo baseline obj: %.4f" % elo_baseline_obj)
    print()

    # Models that can be individually tested (exclude elo, weather, odds)
    testable_models = [m for m in ALL_MODELS
                       if m not in ("elo", "weather", "odds") and m in MODEL_REGISTRY]

    model_results = {}   # model_name -> {obj, accuracy, log_loss, brier, delta_vs_elo, ...}
    all_model_params = {}  # model_name -> optimized hyperparams dict

    total_models = len(testable_models)

    pbar_models = tqdm(enumerate(testable_models), total=total_models,
                        desc="  Phase 1: Solo opt", leave=True)
    for model_idx, model_name in pbar_models:
        pbar_models.set_postfix_str(model_name.upper())
        tqdm.write("\n  [%d/%d] Optimizing: %s" % (model_idx + 1, total_models, model_name.upper()))
        tqdm.write("  " + "-" * 50)

        # Enable ONLY elo + this model
        solo_switches = {m: False for m in ALL_MODELS}
        solo_switches["elo"] = True
        solo_switches[model_name] = True

        # Get this model's hyperparameters
        hp_specs = get_model_hyperparams(model_name)

        # Get solo baseline (before tuning)
        test_params = dict(base_params)
        solo_obj, solo_met = _eval(csv_file, sport, sport_dir, test_params, solo_switches,
                                   elo_model_class, elo_settings, player_df,
                                   "%s_baseline" % model_name)

        if solo_met.get("n_predictions", 0) == 0:
            print("    %s: No predictions generated, skipping" % model_name)
            model_results[model_name] = {"obj": 1e9, "accuracy": 0, "skipped": True}
            continue

        print("    Solo baseline: Acc=%.2f%% LL=%.4f Br=%.4f obj=%.4f" % (
            solo_met.get("accuracy", 0), solo_met.get("log_loss", 1),
            solo_met.get("brier", 0.5), solo_obj))

        # ── Coordinate descent on this model's hyperparams (up to 3 passes) ──
        current_obj = solo_obj
        current_model_hp = {}
        eval_count = 0

        if hp_specs:
            for pass_num in range(1, 4):
                improved = False
                for param_name, spec in hp_specs.items():
                    current_val = current_model_hp.get(param_name, spec["default"])
                    best_val = current_val
                    best_obj_p = current_obj

                    for test_val in spec["values"]:
                        if test_val == current_val:
                            continue
                        test_hp = dict(current_model_hp)
                        test_hp[param_name] = test_val
                        test_params = dict(base_params)
                        test_params["model_params"] = {model_name: test_hp}

                        obj, met = _eval(csv_file, sport, sport_dir, test_params,
                                         solo_switches, elo_model_class, elo_settings,
                                         player_df)
                        eval_count += 1

                        if obj < best_obj_p:
                            best_obj_p = obj
                            best_val = test_val

                    if best_val != current_val:
                        current_model_hp[param_name] = best_val
                        current_obj = best_obj_p
                        improved = True
                        print("    %s=%s -> obj=%.4f %s" % (
                            param_name, best_val, current_obj, cok("(improved)")))

                if not improved:
                    if pass_num > 1:
                        print(cdim("    Pass %d: converged" % pass_num))
                    break

        # ── Also tune meta-learner params for this solo config ───────
        meta_type = base_params.get("meta_model", "ridge")
        meta_key = {"ridge": "meta_ridge", "logistic": "meta_logistic",
                    "xgboost": "meta_xgb"}.get(meta_type)

        if meta_key and meta_key in MODEL_HYPERPARAMS:
            meta_hp_specs = MODEL_HYPERPARAMS[meta_key]
            for param_name, spec in meta_hp_specs.items():
                current_val = current_model_hp.get(param_name, spec["default"])
                best_val = current_val
                best_obj_p = current_obj

                for test_val in spec["values"]:
                    if test_val == current_val:
                        continue
                    test_hp = dict(current_model_hp)
                    test_hp[param_name] = test_val
                    test_params = dict(base_params)
                    test_params["model_params"] = {model_name: test_hp}

                    obj, met = _eval(csv_file, sport, sport_dir, test_params,
                                     solo_switches, elo_model_class, elo_settings,
                                     player_df)
                    eval_count += 1

                    if obj < best_obj_p:
                        best_obj_p = obj
                        best_val = test_val

                if best_val != current_val:
                    current_model_hp[param_name] = best_val
                    current_obj = best_obj_p
                    print("    %s=%s -> obj=%.4f %s" % (
                        param_name, best_val, current_obj, cok("(meta improved)")))

        # ── Also tune max_adj for this solo config ───────────────────
        best_adj = base_params.get("max_adj", 0.10)
        for adj in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            test_params = dict(base_params)
            test_params["max_adj"] = adj
            test_params["model_params"] = {model_name: current_model_hp}
            obj, met = _eval(csv_file, sport, sport_dir, test_params, solo_switches,
                             elo_model_class, elo_settings, player_df)
            eval_count += 1
            if obj < current_obj:
                current_obj = obj
                best_adj = adj

        # ── Record final results ─────────────────────────────────────
        final_params = dict(base_params)
        final_params["max_adj"] = best_adj
        final_params["model_params"] = {model_name: current_model_hp}
        final_obj, final_met = _eval(csv_file, sport, sport_dir, final_params,
                                     solo_switches, elo_model_class, elo_settings,
                                     player_df, "%s_final" % model_name)

        delta = elo_baseline_obj - final_obj  # positive = model improves over elo

        model_results[model_name] = {
            "obj": final_obj,
            "accuracy": final_met.get("accuracy", 0),
            "log_loss": final_met.get("log_loss", 1),
            "brier": final_met.get("brier", 0.5),
            "delta": delta,
            "n_evals": eval_count,
            "best_adj": best_adj,
        }
        all_model_params[model_name] = current_model_hp

        print("    FINAL: Acc=%.2f%% LL=%.4f Br=%.4f delta=%+.4f (%d evals)" % (
            final_met.get("accuracy", 0), final_met.get("log_loss", 1),
            final_met.get("brier", 0.5), delta, eval_count))

    # ── Print leaderboard sorted by delta (most helpful first) ───
    print()
    print("=" * 80)
    print("  MODEL LEADERBOARD (sorted by improvement over Elo-only)")
    print("  %-20s %8s %8s %8s %10s" % ("Model", "Acc%", "LogLoss", "Brier", "Delta"))
    print("  " + "-" * 60)

    sorted_models = sorted(model_results.items(),
                           key=lambda x: x[1].get("delta", -999), reverse=True)

    for name, res in sorted_models:
        if res.get("skipped"):
            print("  %-20s %8s %8s %8s %10s  SKIPPED" % (name, "-", "-", "-", "-"))
            continue
        tag = "HELPS" if res["delta"] > 0.001 else ("HURTS" if res["delta"] < -0.001 else "neutral")
        if tag == "HELPS":
            tag_colored = cok(tag)
        elif tag == "HURTS":
            tag_colored = cerr(tag)
        else:
            tag_colored = cdim(tag)
        print("  %-20s %7.2f%% %8.4f %8.4f %+10.4f  %s" % (
            name, res["accuracy"], res["log_loss"], res["brier"],
            res["delta"], tag_colored))

    print("=" * 80)

    return model_results, all_model_params


# ══════════════════════════════════════════════════════════════════
# Phase 2: Head-to-Head Tournament
# ══════════════════════════════════════════════════════════════════

def _phase2_tournament(csv_file, sport, sport_dir, base_params,
                       elo_model_class, elo_settings, player_df,
                       model_results, all_model_params, elo_baseline_obj):
    """Head-to-head tournament: pairs, triples, top-N combinations.

    Uses results from Phase 1 to build increasingly large model combos.
    1. Each solo result (from Phase 1)
    2. Top 5 in pairs (10 combos)
    3. Top 3 together, top 5, top 10, ALL together
    4. Find the best combination

    Returns:
        (best_obj, best_params, best_switches, best_combo_name)
    """
    print()
    print(cbold("  Phase 2: Head-to-Head Tournament"))
    div(60)

    # Sort models by delta (best first), skip those that were skipped or hurt badly
    ranked = [(name, res) for name, res in model_results.items()
              if not res.get("skipped") and res.get("delta", -999) > -1.0]
    ranked.sort(key=lambda x: x[1]["delta"], reverse=True)

    if not ranked:
        print(cerr("  No viable models found from Phase 1. Returning Elo-only."))
        switches = {m: False for m in ALL_MODELS}
        switches["elo"] = True
        obj, met = _eval(csv_file, sport, sport_dir, base_params, switches,
                         elo_model_class, elo_settings, player_df, "P2_elo_only")
        return obj, base_params, switches, "elo_only"

    model_names = [name for name, _ in ranked]

    print("  Models ranked by Phase 1 delta:")
    for i, (name, res) in enumerate(ranked):
        print("    %2d. %-20s delta=%+.4f" % (i + 1, name, res["delta"]))

    # Track all combos: (combo_name, obj, switches)
    combo_results = []

    # ── Step 2a: Solo results from Phase 1 ───────────────────────
    print()
    print(chi("  Step 2a: Solo results (from Phase 1)"))
    for name, res in ranked:
        if not res.get("skipped"):
            combo_results.append(("solo_%s" % name, res["obj"],
                                  {"elo": True, name: True}))

    # ── Step 2b: Top 5 models in pairs (10 combos) ──────────────
    top5 = model_names[:min(5, len(model_names))]
    print()
    print(chi("  Step 2b: Pairwise combinations (top %d models)" % len(top5)))

    pair_count = 0
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            m1, m2 = top5[i], top5[j]
            pair_switches = {m: False for m in ALL_MODELS}
            pair_switches["elo"] = True
            pair_switches[m1] = True
            pair_switches[m2] = True

            # Use combined model params
            test_params = dict(base_params)
            combined_mp = {}
            if m1 in all_model_params:
                combined_mp.update(all_model_params[m1])
            if m2 in all_model_params:
                combined_mp.update(all_model_params[m2])
            test_params["model_params"] = combined_mp

            obj, met = _eval(csv_file, sport, sport_dir, test_params, pair_switches,
                             elo_model_class, elo_settings, player_df,
                             "pair_%s_%s" % (m1, m2))
            pair_count += 1

            delta_vs_elo = elo_baseline_obj - obj
            tag = cok("HELPS") if delta_vs_elo > 0.001 else cdim("neutral")
            print("    %s + %s: obj=%.4f delta=%+.4f %s" % (
                m1, m2, obj, delta_vs_elo, tag))

            combo_results.append(("pair_%s_%s" % (m1, m2), obj,
                                  dict(pair_switches)))

    # ── Step 2c: Group combinations ──────────────────────────────
    print()
    print(chi("  Step 2c: Group combinations"))

    group_configs = []

    # Top 3
    if len(model_names) >= 3:
        group_configs.append(("top3", model_names[:3]))
    # Top 5
    if len(model_names) >= 5:
        group_configs.append(("top5", model_names[:5]))
    # Top 10
    if len(model_names) >= 10:
        group_configs.append(("top10", model_names[:10]))
    # Top 15
    if len(model_names) >= 15:
        group_configs.append(("top15", model_names[:15]))
    # All positive-delta models
    positive_models = [n for n, r in ranked if r.get("delta", 0) > 0.001]
    if positive_models and len(positive_models) != len(model_names):
        group_configs.append(("all_positive", positive_models))
    # All models
    group_configs.append(("all_models", model_names))

    for group_name, group_models in group_configs:
        group_switches = {m: False for m in ALL_MODELS}
        group_switches["elo"] = True
        for m in group_models:
            group_switches[m] = True

        # Merge all model params
        test_params = dict(base_params)
        combined_mp = {}
        for m in group_models:
            if m in all_model_params:
                combined_mp.update(all_model_params[m])
        test_params["model_params"] = combined_mp

        obj, met = _eval(csv_file, sport, sport_dir, test_params, group_switches,
                         elo_model_class, elo_settings, player_df,
                         "group_%s" % group_name)

        delta_vs_elo = elo_baseline_obj - obj
        n_models = len(group_models)
        print("    %-15s (%2d models): obj=%.4f delta=%+.4f acc=%.2f%%" % (
            group_name, n_models, obj, delta_vs_elo, met.get("accuracy", 0)))

        combo_results.append((group_name, obj, dict(group_switches)))

    # ── Find the best combination ────────────────────────────────
    combo_results.sort(key=lambda x: x[1])
    best_combo_name, best_obj, best_switch_map = combo_results[0]

    # Build full switches dict
    best_switches = {m: False for m in ALL_MODELS}
    best_switches["elo"] = True
    for m, v in best_switch_map.items():
        if v:
            best_switches[m] = True

    # Build combined model params for winning combo
    best_params = dict(base_params)
    combined_mp = {}
    for m in ALL_MODELS:
        if best_switches.get(m) and m in all_model_params:
            combined_mp.update(all_model_params[m])
    best_params["model_params"] = combined_mp

    # Print top 5 combos
    print()
    print("  Top 5 combinations:")
    for rank, (cname, cobj, _) in enumerate(combo_results[:5]):
        delta = elo_baseline_obj - cobj
        marker = cok(" <- WINNER") if rank == 0 else ""
        print("    %d. %-25s obj=%.4f delta=%+.4f%s" % (
            rank + 1, cname, cobj, delta, marker))

    n_enabled = sum(1 for m in ALL_MODELS if best_switches.get(m))
    print()
    print(cok("  Phase 2 complete: best combo = %s (%d models, obj=%.4f)" % (
        best_combo_name, n_enabled, best_obj)))

    return best_obj, best_params, best_switches, best_combo_name


# ══════════════════════════════════════════════════════════════════
# Phase 3: Meta-Learner + Global Tuning
# ══════════════════════════════════════════════════════════════════

def _phase3_meta_global_tuning(csv_file, sport, sport_dir, base_params, switches,
                               elo_model_class, elo_settings, player_df):
    """Fine-tune meta-learner and global params using winning combo from Phase 2.

    1. Grid search max_adj (25 values: 0.02 to 0.50 step 0.02)
    2. Test all 3 meta-learners x retrain_every values
    3. Fine-tune min_train
    4. Fine-tune window

    Returns:
        (best_obj, best_params, switches)
    """
    print()
    print(cbold("  Phase 3: Meta-Learner + Global Tuning"))
    div(60)

    current_params = dict(base_params)
    current_obj, current_met = _eval(csv_file, sport, sport_dir, current_params,
                                     switches, elo_model_class, elo_settings,
                                     player_df, "P3_baseline")

    if current_obj >= 1e9:
        print(cerr("    Baseline failed!"))
        return current_obj, current_params, switches

    print("    Baseline: obj=%.4f acc=%.2f%% ll=%.4f br=%.4f" % (
        current_obj, current_met.get("accuracy", 0),
        current_met.get("log_loss", 1), current_met.get("brier", 0.5)))

    # ── Step 3a: Fine grid on max_adj (25 values) ────────────────
    print()
    print(chi("  Step 3a: max_adj grid search (25 values)"))

    adj_values = [round(0.02 + i * 0.02, 2) for i in range(25)]  # 0.02 to 0.50
    best_adj = current_params.get("max_adj", 0.10)

    for adj in adj_values:
        test_params = dict(current_params)
        test_params["max_adj"] = adj
        obj, met = _eval(csv_file, sport, sport_dir, test_params, switches,
                         elo_model_class, elo_settings, player_df)
        if obj < current_obj:
            current_obj = obj
            best_adj = adj
            current_params["max_adj"] = adj
            print("    max_adj=%.2f -> obj=%.4f acc=%.2f%% %s" % (
                adj, obj, met.get("accuracy", 0), cok("<- BEST")))

    print("    Best max_adj: %.2f" % best_adj)

    # ── Step 3b: Meta-learner x retrain_every grid ───────────────
    print()
    print(chi("  Step 3b: meta_model x retrain_every grid"))

    meta_values = ["ridge", "logistic", "xgboost"]
    retrain_values = _RETRAIN_MAP.get(sport, [25, 50, 75, 100])

    best_meta = current_params.get("meta_model", "ridge")
    best_retrain = current_params.get("retrain_every", retrain_values[len(retrain_values) // 2])
    total_grid = len(meta_values) * len(retrain_values)

    print("    Grid: %d meta x %d retrain = %d combos" % (
        len(meta_values), len(retrain_values), total_grid))

    for i, (meta, retrain) in enumerate(product(meta_values, retrain_values)):
        test_params = dict(current_params)
        test_params["meta_model"] = meta
        test_params["retrain_every"] = retrain
        obj, met = _eval(csv_file, sport, sport_dir, test_params, switches,
                         elo_model_class, elo_settings, player_df)

        if obj < current_obj:
            current_obj = obj
            best_meta = meta
            best_retrain = retrain
            current_params["meta_model"] = meta
            current_params["retrain_every"] = retrain
            print("    meta=%s retrain=%d -> obj=%.4f acc=%.2f%% %s" % (
                meta, retrain, obj, met.get("accuracy", 0), cok("<- BEST")))
        elif (i + 1) % 5 == 0 or (i + 1) == total_grid:
            print(cdim("    #%d/%d tested (best obj=%.4f)" % (i + 1, total_grid, current_obj)))

    print("    Best: meta=%s retrain=%d" % (best_meta, best_retrain))

    # ── Step 3b-ii: Tune meta-learner hyperparams for winning meta type ──
    print()
    print(chi("  Step 3b-ii: Tune %s hyperparams" % best_meta))

    meta_key = {"ridge": "meta_ridge", "logistic": "meta_logistic",
                "xgboost": "meta_xgb"}.get(best_meta)

    if meta_key and meta_key in MODEL_HYPERPARAMS:
        model_params = dict(current_params.get("model_params", {}))
        meta_hp_specs = MODEL_HYPERPARAMS[meta_key]

        for pass_num in range(1, 4):
            improved = False
            for param_name, spec in meta_hp_specs.items():
                current_val = model_params.get(param_name, spec["default"])
                best_val = current_val
                best_obj_p = current_obj

                for test_val in spec["values"]:
                    if test_val == current_val:
                        continue
                    test_mp = dict(model_params)
                    test_mp[param_name] = test_val
                    test_params = dict(current_params)
                    test_params["model_params"] = test_mp

                    obj, met = _eval(csv_file, sport, sport_dir, test_params,
                                     switches, elo_model_class, elo_settings,
                                     player_df)

                    if obj < best_obj_p:
                        best_obj_p = obj
                        best_val = test_val

                if best_val != current_val:
                    model_params[param_name] = best_val
                    current_obj = best_obj_p
                    current_params["model_params"] = dict(model_params)
                    improved = True
                    print("    %s=%s -> obj=%.4f %s" % (
                        param_name, best_val, current_obj, cok("(improved)")))

            if not improved:
                break

    # ── Step 3c: Fine-tune min_train ─────────────────────────────
    print()
    print(chi("  Step 3c: Fine-tune min_train"))

    min_train_values = _MIN_TRAIN_MAP.get(sport, [100, 200, 300])
    for mt in min_train_values:
        test_params = dict(current_params)
        test_params["min_train"] = mt
        obj, met = _eval(csv_file, sport, sport_dir, test_params, switches,
                         elo_model_class, elo_settings, player_df)
        if obj < current_obj:
            current_obj = obj
            current_params["min_train"] = mt
            print("    min_train=%d -> obj=%.4f acc=%.2f%% %s" % (
                mt, obj, met.get("accuracy", 0), cok("<- BEST")))

    # ── Step 3d: Fine-tune window ────────────────────────────────
    print()
    print(chi("  Step 3d: Fine-tune window"))

    window_values = _WINDOW_MAP.get(sport, [5, 7, 10, 15])
    for w in window_values:
        test_params = dict(current_params)
        test_params["window"] = w
        obj, met = _eval(csv_file, sport, sport_dir, test_params, switches,
                         elo_model_class, elo_settings, player_df)
        if obj < current_obj:
            current_obj = obj
            current_params["window"] = w
            print("    window=%d -> obj=%.4f acc=%.2f%% %s" % (
                w, obj, met.get("accuracy", 0), cok("<- BEST")))

    print()
    print(cok("  Phase 3 complete: obj=%.4f" % current_obj))
    print("    max_adj=%.2f  meta=%s  retrain=%s  min_train=%s  window=%s" % (
        current_params.get("max_adj", 0.10),
        current_params.get("meta_model", "ridge"),
        current_params.get("retrain_every", "default"),
        current_params.get("min_train", "default"),
        current_params.get("window", "default")))

    return current_obj, current_params, switches


# ══════════════════════════════════════════════════════════════════
# Phase 4: Combined Differential Evolution
# ══════════════════════════════════════════════════════════════════

def _phase4_combined_de(csv_file, sport, sport_dir, base_params, switches,
                        elo_model_class, elo_settings, player_df):
    """Differential evolution over top ~20 numeric params + max_adj.

    Uses scipy.optimize.differential_evolution for continuous optimization
    over all numeric model hyperparams of enabled models.

    Returns:
        (best_obj, best_params, switches)
    """
    print()
    print(cbold("  Phase 4: Combined Differential Evolution"))
    div(60)

    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        print(cwarn("    scipy not available, skipping DE phase"))
        obj, met = _eval(csv_file, sport, sport_dir, base_params, switches,
                         elo_model_class, elo_settings, player_df)
        return obj, base_params, switches

    current_params = dict(base_params)
    model_params = dict(current_params.get("model_params", {}))

    # Build list of continuous params to optimize with DE
    # (name, model_or_None, lo, hi, current_val, is_int)
    de_params = []
    de_params.append(("max_adj", None, 0.02, 0.50,
                      current_params.get("max_adj", 0.10), False))

    # Add per-model params (only floats/ints with numeric ranges)
    for model_name, hp_specs in MODEL_HYPERPARAMS.items():
        if not is_model_enabled(model_name, switches):
            continue
        # Skip meta params that don't match current meta model
        meta_type = current_params.get("meta_model", "ridge")
        if model_name.startswith("meta_"):
            if model_name == "meta_xgb" and meta_type != "xgboost":
                continue
            if model_name == "meta_ridge" and meta_type != "ridge":
                continue
            if model_name == "meta_logistic" and meta_type != "logistic":
                continue

        for param_name, spec in hp_specs.items():
            if spec["type"] not in ("float", "int"):
                continue
            if len(spec["values"]) < 3:
                continue
            vals = [float(v) for v in spec["values"]]
            lo, hi = min(vals), max(vals)
            cur = float(model_params.get(param_name, spec["default"]))
            is_int = (spec["type"] == "int")
            de_params.append((param_name, model_name, lo, hi, cur, is_int))

    if len(de_params) < 2:
        print(cwarn("  Not enough continuous params for DE, skipping"))
        obj, met = _eval(csv_file, sport, sport_dir, current_params, switches,
                         elo_model_class, elo_settings, player_df)
        return obj, current_params, switches

    # Cap at ~25 params for tractability
    if len(de_params) > 25:
        print(cdim("  Limiting DE to first 25 of %d params" % len(de_params)))
        de_params = de_params[:25]

    bounds = [(p[2], p[3]) for p in de_params]

    print("  Optimizing %d params with DE (maxiter=15, popsize=8):" % len(de_params))
    for name, model, lo, hi, cur, is_int in de_params:
        owner = "(%s)" % model if model else "(global)"
        print("    %-30s  [%.4g, %.4g]  cur=%.4g  %s" % (name, lo, hi, cur, owner))

    best_result = [1e9, dict(current_params)]
    de_eval_count = [0]

    def de_objective(x):
        test_params = dict(current_params)
        test_mp = dict(model_params)

        for i, (name, model, lo, hi, cur, is_int) in enumerate(de_params):
            val = float(x[i])
            if is_int:
                val = int(round(val))
            else:
                val = round(val, 6)

            if model is None:
                test_params[name] = val
            else:
                test_mp[name] = val

        test_params["model_params"] = test_mp
        obj, met = _eval(csv_file, sport, sport_dir, test_params, switches,
                         elo_model_class, elo_settings, player_df)
        de_eval_count[0] += 1

        if obj < best_result[0]:
            best_result[0] = obj
            best_result[1] = dict(test_params)
            best_result[1]["model_params"] = dict(test_mp)
            print("    DE #%d obj=%.4f acc=%.2f%% %s" % (
                de_eval_count[0], obj, met.get("accuracy", 0), cok("<- NEW BEST")))
        elif de_eval_count[0] % 20 == 0:
            print(cdim("    DE #%d obj=%.4f (best=%.4f)" % (
                de_eval_count[0], obj, best_result[0])))

        return obj

    try:
        result = differential_evolution(
            de_objective, bounds=bounds,
            maxiter=15, popsize=8,
            workers=1, seed=42, tol=0.001,
        )
        print()
        print(cok("  Phase 4 complete: DE converged at obj=%.4f (%d evals)" % (
            result.fun, de_eval_count[0])))
    except Exception as e:
        print(cerr("  Phase 4: DE error: %s" % e))

    return best_result[0], best_result[1], switches


# ══════════════════════════════════════════════════════════════════
# Phase 5: Final Ablation
# ══════════════════════════════════════════════════════════════════

def _phase5_final_ablation(csv_file, sport, sport_dir, base_params, switches,
                           elo_model_class, elo_settings, player_df):
    """Final ablation: test each model's contribution with fully tuned params.

    Only NOW, after everything is optimized, do we test whether each model
    still helps. Disable any that hurt with the fully-tuned configuration.

    Returns:
        (best_obj, best_params, switches)
    """
    print()
    print(cbold("  Phase 5: Final Ablation"))
    div(60)

    baseline_obj, baseline_met = _eval(
        csv_file, sport, sport_dir, base_params, switches,
        elo_model_class, elo_settings, player_df, "P5_baseline")

    if baseline_obj >= 1e9:
        print(cerr("    Baseline failed!"))
        return baseline_obj, base_params, switches

    print("    Baseline: obj=%.4f acc=%.2f%% ll=%.4f br=%.4f" % (
        baseline_obj, baseline_met.get("accuracy", 0),
        baseline_met.get("log_loss", 1), baseline_met.get("brier", 0.5)))

    # Test each enabled model (except elo)
    enabled_models = [m for m in ALL_MODELS
                      if is_model_enabled(m, switches) and m != "elo"]

    if not enabled_models:
        print(cdim("  No models to ablate (only Elo enabled)."))
        return baseline_obj, base_params, switches

    print()
    print("  %-20s %8s %8s %8s  %s" % ("Model", "Obj", "Delta", "Acc%", "Verdict"))
    print("  " + "-" * 65)

    contributions = []
    for model_name in enabled_models:
        test_switches = dict(switches)
        test_switches[model_name] = False
        obj, met = _eval(csv_file, sport, sport_dir, base_params, test_switches,
                         elo_model_class, elo_settings, player_df,
                         "ablate_%s" % model_name)

        if not met:
            continue

        delta = obj - baseline_obj  # positive = model helps (removing it hurts)

        if delta > 0.001:
            verdict = "HELPS"
            v_colored = cok("HELPS")
        elif delta < -0.001:
            verdict = "HURTS"
            v_colored = cerr("HURTS")
        else:
            verdict = "neutral"
            v_colored = cdim("neutral")

        contributions.append({
            "model": model_name,
            "delta": delta,
            "obj_without": obj,
            "accuracy_without": met.get("accuracy", 0),
            "verdict": verdict,
        })

        print("  %-20s %8.4f %+7.4f %7.2f%%  %s" % (
            model_name, obj, delta, met.get("accuracy", 0), v_colored))

    # Sort by contribution
    contributions.sort(key=lambda x: x["delta"], reverse=True)

    # Auto-disable models that hurt
    hurting = [c for c in contributions if c["verdict"] == "HURTS"]
    if hurting:
        print()
        print(cwarn("  Auto-disabling %d model(s) that hurt performance:" % len(hurting)))
        for c in hurting:
            switches[c["model"]] = False
            print("    Disabled: %s (delta=%+.4f)" % (c["model"], c["delta"]))

        # Verify improvement
        verify_obj, verify_met = _eval(
            csv_file, sport, sport_dir, base_params, switches,
            elo_model_class, elo_settings, player_df, "P5_verify")

        if verify_obj < baseline_obj:
            print(cok("    After pruning: obj=%.4f (improved from %.4f)" % (
                verify_obj, baseline_obj)))
            baseline_obj = verify_obj
        else:
            # Revert
            for c in hurting:
                switches[c["model"]] = True
            print(cwarn("    No improvement from pruning -- reverted"))
    else:
        print()
        print(cok("  All models contributing. No pruning needed."))

    # Summary
    print()
    n_enabled = sum(1 for m in ALL_MODELS if is_model_enabled(m, switches))
    n_helping = len([c for c in contributions if c["verdict"] == "HELPS"])
    n_neutral = len([c for c in contributions if c["verdict"] == "neutral"])
    print("  Phase 5 summary: %d helping, %d neutral, %d hurting, %d/%d enabled" % (
        n_helping, n_neutral, len(hurting), n_enabled, len(ALL_MODELS)))

    return baseline_obj, base_params, switches


# ══════════════════════════════════════════════════════════════════
# Phase 6: Validation
# ══════════════════════════════════════════════════════════════════

def _phase6_validation(csv_file, sport, sport_dir, base_params, switches,
                       elo_model_class, elo_settings, player_df,
                       n_runs=5):
    """Run best config multiple times, compute stability.

    Returns:
        (mean_obj, base_params, switches)
    """
    print()
    print(cbold("  Phase 6: Validation + Stability"))
    div(60)

    objs = []
    accs = []
    lls = []
    briers = []

    for run in range(1, n_runs + 1):
        obj, met = _eval(csv_file, sport, sport_dir, base_params, switches,
                         elo_model_class, elo_settings, player_df,
                         "P6_run%d" % run)
        objs.append(obj)
        accs.append(met.get("accuracy", 0))
        lls.append(met.get("log_loss", 1))
        briers.append(met.get("brier", 0.5))
        print("    Run %d/%d: obj=%.4f acc=%.2f%% ll=%.4f br=%.4f" % (
            run, n_runs, obj, met.get("accuracy", 0),
            met.get("log_loss", 1), met.get("brier", 0.5)))

    mean_obj = float(np.mean(objs))
    std_obj = float(np.std(objs))
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    print()
    print("  Stability results (%d runs):" % n_runs)
    print("    Objective : %.4f +/- %.4f (%.1f%% variation)" % (
        mean_obj, std_obj,
        std_obj / mean_obj * 100 if mean_obj > 0 else 0))
    print("    Accuracy  : %.2f%% +/- %.2f%%" % (mean_acc, std_acc))
    print("    Log Loss  : %.4f +/- %.4f" % (np.mean(lls), np.std(lls)))
    print("    Brier     : %.4f +/- %.4f" % (np.mean(briers), np.std(briers)))

    pct_var = std_obj / mean_obj * 100 if mean_obj > 0 else 0
    if pct_var < 1.0:
        print(cok("    Result is STABLE (< 1%% variation)"))
    elif pct_var < 3.0:
        print(cwarn("    Result is MODERATE stability (1-3%% variation)"))
    else:
        print(cerr("    Result is UNSTABLE (> 3%% variation)"))

    return mean_obj, base_params, switches


# ══════════════════════════════════════════════════════════════════
# Main entry points
# ══════════════════════════════════════════════════════════════════

def run_mega_optimize(csv_file, sport="nfl", elo_model_class=None,
                      elo_settings=None, player_df=None,
                      phases=None, verbose=True):
    """Run full multi-phase mega-ensemble optimization.

    Args:
        csv_file: path to games CSV
        sport: 'nfl', 'mlb', 'nba', 'nhl'
        elo_model_class: the sport's Elo class
        elo_settings: Elo model settings dict
        player_df: player stats DataFrame
        phases: list of phases to run (default [0,1,2,3,4,5,6] = all)
        verbose: print progress

    Returns dict with best_params, best_objective, best_results,
        baseline_objective, n_evaluations, elapsed_seconds.
    """
    global _eval_count, _eval_start

    if phases is None:
        phases = [0, 1, 2, 3, 4, 5, 6]

    sport_dir = os.path.dirname(os.path.abspath(csv_file))
    start_time = time.time()
    _eval_count = 0
    _eval_start = start_time

    # Load existing settings
    best_params = _load_settings(sport, sport_dir)
    switches = load_model_switches(sport, sport_dir)
    model_params = load_model_params(sport, sport_dir)
    if model_params and "model_params" not in best_params:
        best_params["model_params"] = model_params

    # ── Header ────────────────────────────────────────────────────
    if verbose:
        print()
        print(cbold("  ===================================================="))
        print(cbold("    MEGA-ENSEMBLE 7-PHASE PER-MODEL OPTIMIZER (%s)" % sport.upper()))
        print(cbold("  ===================================================="))
        print("    Objective = LogLoss*8 + Brier*40  (lower is better)")
        print("    Phases: %s" % phases)
        n_enabled = sum(1 for m in ALL_MODELS if is_model_enabled(m, switches))
        print("    Models: %d/%d enabled" % (n_enabled, len(ALL_MODELS)))
        print()

    # State that carries between phases
    elo_baseline_obj = None
    elo_baseline_met = None
    model_results = None
    all_model_params = None
    best_obj = 1e9
    best_result = None
    baseline_obj = 1e9

    # ── Phase 0: Elo-only Baseline ───────────────────────────────
    if 0 in phases:
        elo_baseline_obj, elo_baseline_met = _phase0_baseline(
            csv_file, sport, sport_dir, best_params,
            elo_model_class, elo_settings, player_df)
        baseline_obj = elo_baseline_obj
        best_obj = elo_baseline_obj
        best_result = elo_baseline_met
        _save_all(sport, sport_dir, best_params, switches)
        if verbose:
            print(cdim("  [Saved after Phase 0]"))

    # If we didn't run Phase 0 but need elo baseline for later phases
    if elo_baseline_obj is None and any(p in phases for p in [1, 2]):
        elo_only_sw = {m: False for m in ALL_MODELS}
        elo_only_sw["elo"] = True
        elo_baseline_obj, elo_baseline_met = _eval(
            csv_file, sport, sport_dir, best_params, elo_only_sw,
            elo_model_class, elo_settings, player_df, "elo_baseline_implicit")
        if baseline_obj >= 1e9:
            baseline_obj = elo_baseline_obj

    # ── Phase 1: Per-Model Solo Optimization ─────────────────────
    if 1 in phases:
        if elo_baseline_obj is None or elo_baseline_obj >= 1e9:
            print(cerr("  Cannot run Phase 1 without Elo baseline. Run Phase 0 first."))
        else:
            model_results, all_model_params = _phase1_per_model_solo(
                csv_file, sport, sport_dir, best_params,
                elo_model_class, elo_settings, player_df, elo_baseline_obj)

            # Update model_params in best_params with all tuned values
            merged_mp = dict(best_params.get("model_params", {}))
            for m_name, m_hp in all_model_params.items():
                merged_mp.update(m_hp)
            best_params["model_params"] = merged_mp

            _save_all(sport, sport_dir, best_params, switches)
            if verbose:
                print(cdim("  [Saved after Phase 1]"))

    # ── Phase 2: Head-to-Head Tournament ─────────────────────────
    if 2 in phases:
        if model_results is None:
            print(cwarn("  Phase 2 requires Phase 1 results. Running Phase 1 first..."))
            if elo_baseline_obj is None or elo_baseline_obj >= 1e9:
                elo_only_sw = {m: False for m in ALL_MODELS}
                elo_only_sw["elo"] = True
                elo_baseline_obj, _ = _eval(
                    csv_file, sport, sport_dir, best_params, elo_only_sw,
                    elo_model_class, elo_settings, player_df, "elo_baseline_auto")
            model_results, all_model_params = _phase1_per_model_solo(
                csv_file, sport, sport_dir, best_params,
                elo_model_class, elo_settings, player_df, elo_baseline_obj)
            merged_mp = dict(best_params.get("model_params", {}))
            for m_name, m_hp in all_model_params.items():
                merged_mp.update(m_hp)
            best_params["model_params"] = merged_mp

        obj, params, new_switches, combo_name = _phase2_tournament(
            csv_file, sport, sport_dir, best_params,
            elo_model_class, elo_settings, player_df,
            model_results, all_model_params, elo_baseline_obj)

        if obj < best_obj:
            best_obj = obj
            best_params = params
            switches = new_switches

        _save_all(sport, sport_dir, best_params, switches)
        if verbose:
            print(cdim("  [Saved after Phase 2]"))

    # ── Phase 3: Meta-Learner + Global Tuning ────────────────────
    if 3 in phases:
        obj, params, switches = _phase3_meta_global_tuning(
            csv_file, sport, sport_dir, best_params, switches,
            elo_model_class, elo_settings, player_df)
        if obj < best_obj:
            best_obj = obj
            best_params = params
        _save_all(sport, sport_dir, best_params, switches)
        if verbose:
            print(cdim("  [Saved after Phase 3]"))

    # ── Phase 4: Combined DE Fine-Tuning ─────────────────────────
    if 4 in phases:
        obj, params, switches = _phase4_combined_de(
            csv_file, sport, sport_dir, best_params, switches,
            elo_model_class, elo_settings, player_df)
        if obj < best_obj:
            best_obj = obj
            best_params = params
        _save_all(sport, sport_dir, best_params, switches)
        if verbose:
            print(cdim("  [Saved after Phase 4]"))

    # ── Phase 5: Final Ablation ──────────────────────────────────
    if 5 in phases:
        obj, params, switches = _phase5_final_ablation(
            csv_file, sport, sport_dir, best_params, switches,
            elo_model_class, elo_settings, player_df)
        if obj < best_obj:
            best_obj = obj
            best_params = params
        _save_all(sport, sport_dir, best_params, switches)
        if verbose:
            print(cdim("  [Saved after Phase 5]"))

    # ── Phase 6: Validation ──────────────────────────────────────
    if 6 in phases:
        obj, params, switches = _phase6_validation(
            csv_file, sport, sport_dir, best_params, switches,
            elo_model_class, elo_settings, player_df)
        # Validation doesn't change params, just reports stability

    # ── Final run to get best result metrics ─────────────────────
    final_obj, final_met = _eval(
        csv_file, sport, sport_dir, best_params, switches,
        elo_model_class, elo_settings, player_df, "FINAL")
    if final_met:
        best_result = final_met
    if final_obj < best_obj:
        best_obj = final_obj

    # ── Final save ───────────────────────────────────────────────
    settings_path = _save_all(sport, sport_dir, best_params, switches)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - start_time

    if verbose:
        print()
        print(cbold("  ===================================================="))
        print(cbold("    OPTIMIZATION COMPLETE (%s)" % sport.upper()))
        print(cbold("  ===================================================="))
        print("    Total evaluations : %d" % _eval_count)
        print("    Total time        : %.1f minutes" % (elapsed / 60))
        print("    Elo baseline obj  : %.4f" % (baseline_obj if baseline_obj < 1e9 else 0))
        print("    Best obj          : %.4f" % best_obj)

        if baseline_obj > 0 and baseline_obj < 1e9:
            improvement = (baseline_obj - best_obj) / baseline_obj * 100
            print("    Improvement       : %.2f%%" % improvement)

        print()
        if best_result:
            print("    Best accuracy     : %.2f%%" % best_result.get("accuracy", 0))
            print("    Best log loss     : %.4f" % best_result.get("log_loss", 1))
            print("    Best brier        : %.4f" % best_result.get("brier", 0.5))

        print()
        print("    Key parameters:")
        for k in ["max_adj", "meta_model", "retrain_every", "min_train", "window"]:
            if k in best_params:
                print("      %-25s = %s" % (k, best_params[k]))

        mp = best_params.get("model_params", {})
        if mp:
            print()
            print("    Model hyperparameters:")
            for k, v in sorted(mp.items()):
                print("      %-30s = %s" % (k, v))

        print()
        n_enabled = sum(1 for m in ALL_MODELS if is_model_enabled(m, switches))
        print("    Models enabled    : %d / %d" % (n_enabled, len(ALL_MODELS)))
        print("    Settings saved to : %s" % settings_path)
        print(cbold("  ===================================================="))

    return {
        "best_params": best_params,
        "best_objective": best_obj,
        "best_results": best_result if best_result else None,
        "baseline_objective": baseline_obj if baseline_obj < 1e9 else None,
        "baseline_results": elo_baseline_met if elo_baseline_met else None,
        "n_evaluations": _eval_count,
        "elapsed_seconds": elapsed,
        "settings_path": settings_path,
    }


def run_quick_optimize(csv_file, sport="nfl", elo_model_class=None,
                       elo_settings=None, player_df=None):
    """Quick optimization: Phases 0 + 1 only (Elo baseline + per-model solo)."""
    return run_mega_optimize(csv_file, sport, elo_model_class, elo_settings,
                             player_df, phases=[0, 1])


def run_deep_optimize(csv_file, sport="nfl", elo_model_class=None,
                      elo_settings=None, player_df=None):
    """Deep optimization: All 7 phases, no shortcuts."""
    return run_mega_optimize(csv_file, sport, elo_model_class, elo_settings,
                             player_df, phases=[0, 1, 2, 3, 4, 5, 6])


def run_single_model_optimize(csv_file, sport="nfl", elo_model_class=None,
                               elo_settings=None, player_df=None,
                               verbose=True):
    """Run Phase 5 ablation only -- test each model's contribution.

    For every enabled model:
    1. Run baseline (all enabled models ON)
    2. Turn OFF one model, run backtest
    3. Difference = this model's contribution

    Rank models by contribution and auto-disable models that hurt.
    """
    sport_dir = os.path.dirname(os.path.abspath(csv_file))
    switches = load_model_switches(sport, sport_dir)
    settings = _load_settings(sport, sport_dir)
    start_time = time.time()

    global _eval_count
    _eval_count = 0

    if verbose:
        print()
        print(cbold("  ===================================================="))
        print(cbold("    SINGLE-MODEL ABLATION STUDY (%s)" % sport.upper()))
        print(cbold("  ===================================================="))
        print("    Testing each model's individual contribution...")
        print()

    # 1. Run baseline with ALL currently-enabled models
    if verbose:
        print("  Step 1: Baseline (all enabled models)")

    baseline_obj, baseline_met = _eval(
        csv_file, sport, sport_dir, settings, switches,
        elo_model_class, elo_settings, player_df, "ablation_baseline")

    if baseline_obj >= 1e9:
        print(cerr("    Baseline failed!"))
        return {"contributions": [], "baseline_obj": 1e9,
                "n_hurting": 0, "elapsed_seconds": 0}

    if verbose:
        print("    Baseline: obj=%.4f acc=%.2f%% ll=%.4f br=%.4f" % (
            baseline_obj, baseline_met.get("accuracy", 0),
            baseline_met.get("log_loss", 1), baseline_met.get("brier", 0.5)))
        print()

    # 2. Ablation: turn off each model one at a time
    if verbose:
        print("  Step 2: Ablation (turn off one model at a time)")
        print("  %-20s %8s %8s %8s  %s" % ("Model", "Obj", "Delta", "Acc%", "Verdict"))
        print("  " + "-" * 65)

    contributions = []
    enabled_models = [m for m in ALL_MODELS if is_model_enabled(m, switches)
                      and m not in ("elo",)]  # Never ablate Elo

    for model_name in enabled_models:
        test_switches = dict(switches)
        test_switches[model_name] = False

        obj, met = _eval(csv_file, sport, sport_dir, settings, test_switches,
                         elo_model_class, elo_settings, player_df,
                         "without_%s" % model_name)

        if not met:
            continue

        delta = obj - baseline_obj  # Positive = worse without = model helps

        if delta > 0.001:
            verdict = "HELPS"
            v_colored = cok("HELPS")
        elif delta < -0.001:
            verdict = "HURTS"
            v_colored = cerr("HURTS")
        else:
            verdict = "neutral"
            v_colored = cdim("neutral")

        contributions.append({
            "model": model_name,
            "obj_without": obj,
            "delta": delta,
            "accuracy_without": met.get("accuracy", 0),
            "verdict": verdict,
        })

        if verbose:
            print("  %-20s %8.4f %+7.4f %7.2f%%  %s" % (
                model_name, obj, delta, met.get("accuracy", 0), v_colored))

    # 3. Sort by contribution (highest delta = most helpful)
    contributions.sort(key=lambda x: x["delta"], reverse=True)

    if verbose:
        print()
        print("  Step 3: Rankings (most helpful -> least)")
        print("  %-4s %-20s %+8s  %s" % ("Rank", "Model", "Impact", "Verdict"))
        print("  " + "-" * 45)
        for i, c in enumerate(contributions):
            v_colored = (cok(c["verdict"]) if c["verdict"] == "HELPS"
                         else cerr(c["verdict"]) if c["verdict"] == "HURTS"
                         else cdim(c["verdict"]))
            print("  %3d. %-20s %+7.4f  %s" % (
                i + 1, c["model"], c["delta"], v_colored))

    # 4. Count models that hurt
    hurting = [c for c in contributions if c["verdict"] == "HURTS"]

    if verbose and hurting:
        print()
        print(cwarn("  Models that HURT performance (consider disabling):"))
        for c in hurting:
            print("    - %s (obj %+.4f when removed)" % (c["model"], c["delta"]))

    # 5. Auto-disable models that hurt (if any)
    if hurting:
        if verbose:
            print()
            print("  Auto-disabling %d model(s) that hurt performance..." % len(hurting))

        for c in hurting:
            switches[c["model"]] = False
            if verbose:
                print("    Disabled: %s" % c["model"])

        save_model_switches(sport, sport_dir, switches)

        # Verify improvement
        verify_obj, verify_met = _eval(
            csv_file, sport, sport_dir, settings, switches,
            elo_model_class, elo_settings, player_df, "ablation_verify")

        if verbose:
            print()
            print("    After disabling: obj=%.4f (was %.4f)" % (verify_obj, baseline_obj))
            if verify_obj < baseline_obj:
                print(cok("    Improvement: %+.4f" % (baseline_obj - verify_obj)))
                settings["model_switches"] = switches
                _save_settings(sport, sport_dir, settings)
                print(cok("    Settings saved with pruned model set."))
            else:
                # Revert
                for c in hurting:
                    switches[c["model"]] = True
                save_model_switches(sport, sport_dir, switches)
                print(cwarn("    No improvement -- reverted all changes"))
    else:
        if verbose:
            print()
            print(cok("  All enabled models are helping. No changes needed."))

    elapsed = time.time() - start_time

    if verbose:
        print()
        print(cbold("  ===================================================="))
        print(cbold("    ABLATION COMPLETE (%s)" % sport.upper()))
        print(cbold("  ===================================================="))
        n_enabled = sum(1 for m in ALL_MODELS if is_model_enabled(m, switches))
        print("    Models tested  : %d" % len(enabled_models))
        print("    Models helping : %d" % len([c for c in contributions if c["verdict"] == "HELPS"]))
        print("    Models neutral : %d" % len([c for c in contributions if c["verdict"] == "neutral"]))
        print("    Models hurting : %d" % len(hurting))
        print("    Models enabled : %d / %d" % (n_enabled, len(ALL_MODELS)))
        print("    Time           : %.1f minutes" % (elapsed / 60))
        print(cbold("  ===================================================="))

    return {
        "contributions": contributions,
        "baseline_obj": baseline_obj,
        "n_hurting": len(hurting),
        "elapsed_seconds": elapsed,
    }
