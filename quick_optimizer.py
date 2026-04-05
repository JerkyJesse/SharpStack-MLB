"""Quick 10-parameter optimizer using differential_evolution."""
import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from config import GAMES_FILE, save_elo_settings, load_elo_settings
from elo_model import MLBElo
from data_players import load_player_stats, build_league_player_scores
from build_model import _calc_altitude_bonus
from metrics import log_loss_binary, brier_score_binary

_OPT_KEYS = ("k", "home_adv", "player_boost", "starter_boost", "rest_factor",
              "form_weight", "travel_factor", "sos_factor", "pace_factor",
              "playoff_hca_factor", "division_factor", "mean_reversion")

def run_optimizer(csv_file=GAMES_FILE, maxiter=60, popsize=30, focus="accuracy"):
    if not os.path.exists(csv_file):
        print("ERROR: %s not found" % csv_file)
        return
    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")

    player_df = load_player_stats()
    alt_bonus = _calc_altitude_bonus(csv_file)

    # Pre-extract starters
    starters = []
    for _, row in games.iterrows():
        starters.append((
            str(row.get("home_starter", "") or "").strip(),
            str(row.get("away_starter", "") or "").strip(),
        ))

    bounds = [
        (2, 20),     # k (MLB needs lower K for 162-game season)
        (5, 50),     # home_adv
        (0, 40),     # player_boost
        (0, 150),    # starter_boost
        (0, 50),     # rest_factor
        (0, 20),     # form_weight
        (0, 40),     # travel_factor
        (0, 30),     # sos_factor
        (0, 50),     # pace_factor
        (0.0, 1.0),  # playoff_hca_factor
        (0, 50),     # division_factor
        (0, 50),     # mean_reversion
    ]

    best_acc = [0.0]
    eval_count = [0]

    def objective(params):
        settings = dict(zip(_OPT_KEYS, params))
        model = MLBElo(**settings)
        model._altitude_bonus = alt_bonus
        if not player_df.empty:
            model.set_player_stats(player_df)

        correct = 0
        probs, actuals = [], []
        prev_season = None

        for idx, (_, row) in enumerate(games.iterrows()):
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            home_starter, away_starter = starters[idx]

            # Season boundary -> regress pitcher ratings
            if game_date is not None:
                row_season = game_date.year
                if prev_season is not None and row_season != prev_season:
                    model.regress_pitcher_ratings(factor=0.5)
                prev_season = row_season

            try:
                p = model.win_prob(row["home_team"], row["away_team"],
                                   team_a_home=True,
                                   neutral_site=bool(row["neutral_site"]),
                                   calibrated=False, game_date=game_date,
                                   use_injuries=False,
                                   home_starter=home_starter,
                                   away_starter=away_starter)
                home_actual = 1 if row["home_score"] > row["away_score"] else 0
                probs.append(p)
                actuals.append(home_actual)
                if (p >= 0.5) == (home_actual == 1):
                    correct += 1
                model.update_game(row["home_team"], row["away_team"],
                                  row["home_score"], row["away_score"],
                                  neutral_site=bool(row["neutral_site"]),
                                  game_date=game_date,
                                  home_starter=home_starter,
                                  away_starter=away_starter)
            except Exception:
                pass

        n = len(probs)
        if n == 0:
            return 999.0
        acc = correct / n * 100
        ll = log_loss_binary(actuals, probs)
        brier = brier_score_binary(actuals, probs)

        eval_count[0] += 1
        if acc > best_acc[0]:
            best_acc[0] = acc
            print("  [%d] NEW BEST: Acc=%.2f%% LL=%.4f Brier=%.4f | %s" % (
                eval_count[0], acc, ll, brier,
                " ".join("%.2f" % p for p in params)))

        if focus == "accuracy":
            return (100 - acc) + brier * 5 + ll * 2
        else:
            return ll * 8 + brier * 40

    print("=" * 60)
    print("  MLB 12-PARAM OPTIMIZER (focus=%s)" % focus)
    print("  maxiter=%d popsize=%d" % (maxiter, popsize))
    print("=" * 60)

    result = differential_evolution(
        objective, bounds=bounds, maxiter=maxiter, popsize=popsize,
        seed=42, tol=1e-6, disp=False, workers=1,
    )

    best_params = dict(zip(_OPT_KEYS, result.x))
    print("\n  OPTIMIZER COMPLETE")
    print("  Best objective: %.6f" % result.fun)
    for k, v in best_params.items():
        print("    %-20s = %.4f" % (k, v))

    # Save
    settings = load_elo_settings()
    settings.update(best_params)
    save_elo_settings(settings)
    print("  Settings saved to mlb_elo_settings.json")
    return best_params


if __name__ == "__main__":
    focus = sys.argv[1] if len(sys.argv) > 1 else "accuracy"
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    run_optimizer(focus=focus, maxiter=maxiter)
