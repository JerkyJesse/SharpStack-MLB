"""Sweep elo_weight and XGBoost hyperparameters to find best MLB ensemble accuracy."""
import os, sys, json, math, numpy as np, pandas as pd, xgboost as xgb
from collections import defaultdict
from datetime import timedelta
from config import GAMES_FILE, load_elo_settings
from elo_model import MLBElo
from data_players import load_player_stats

WINDOW = 15
PYTH_EXP = 1.83

class TeamTracker:
    def __init__(self):
        self.points_scored = defaultdict(list)
        self.points_allowed = defaultdict(list)
        self.results = defaultdict(list)
        self.margins = defaultdict(list)
        self.last_date = {}
        self.streak = defaultdict(int)
        self.home_results = defaultdict(list)
        self.away_results = defaultdict(list)
        self.game_dates = defaultdict(list)
        self.close_results = defaultdict(list)
        self.blowout_results = defaultdict(list)

    def get_features(self, team, game_date=None, is_home=True):
        scored = self.points_scored.get(team, [])
        allowed = self.points_allowed.get(team, [])
        results = self.results.get(team, [])
        margins = self.margins.get(team, [])
        n = len(scored)
        if n < 3: return None
        rpg = np.mean(scored[-WINDOW:])
        rapg = np.mean(allowed[-WINDOW:])
        win_pct = np.mean(results[-WINDOW:])
        avg_margin = np.mean(margins[-WINDOW:])
        rs = max(rpg, 0.1) ** PYTH_EXP
        ra = max(rapg, 0.1) ** PYTH_EXP
        pyth = rs / (rs + ra)
        consistency = float(np.std(margins[-WINDOW:])) if len(margins) >= 3 else 3.0
        recent_5 = np.mean(results[-5:]) if len(results) >= 5 else win_pct
        trend = recent_5 - win_pct
        rest = None
        if game_date is not None and team in self.last_date:
            rest = max(0, (game_date - self.last_date[team]).days)
        h_res = self.home_results.get(team, [])
        a_res = self.away_results.get(team, [])
        hwp = np.mean(h_res[-WINDOW:]) if len(h_res) >= 3 else 0.5
        awp = np.mean(a_res[-WINDOW:]) if len(a_res) >= 3 else 0.5
        vwp = hwp if is_home else awp
        fatigue = 0
        if game_date is not None:
            cutoff = game_date - timedelta(days=7)
            fatigue = sum(1 for d in self.game_dates.get(team, []) if d >= cutoff)
        cr = self.close_results.get(team, [])
        clutch = np.mean(cr[-10:]) if len(cr) >= 3 else 0.5
        sv = float(np.std(scored[-WINDOW:])) if len(scored) >= 5 else 2.5
        br = self.blowout_results.get(team, [])
        dom = np.mean(br[-10:]) if len(br) >= 3 else 0.5
        return {"ppg": rpg, "papg": rapg, "win_pct": win_pct, "avg_margin": avg_margin,
                "off_rating": rpg, "def_rating": rapg, "rest_days": rest, "games_played": n,
                "pyth": pyth, "streak": float(self.streak.get(team, 0)),
                "consistency": consistency, "trend": trend, "venue_win_pct": vwp,
                "fatigue": float(fatigue), "clutch": clutch, "score_vol": sv, "dominance": dom}

    def update(self, team, ps, pa, won, gd=None, is_home=True):
        self.points_scored[team].append(ps)
        self.points_allowed[team].append(pa)
        self.results[team].append(1.0 if won else 0.0)
        self.margins[team].append(ps - pa)
        if won: self.streak[team] = max(0, self.streak.get(team, 0)) + 1
        else: self.streak[team] = min(0, self.streak.get(team, 0)) - 1
        for s in (self.points_scored, self.points_allowed, self.results, self.margins):
            if len(s[team]) > 25: s[team] = s[team][-25:]
        if is_home:
            self.home_results[team].append(1.0 if won else 0.0)
            if len(self.home_results[team]) > 25: self.home_results[team] = self.home_results[team][-25:]
        else:
            self.away_results[team].append(1.0 if won else 0.0)
            if len(self.away_results[team]) > 25: self.away_results[team] = self.away_results[team][-25:]
        if gd is not None:
            self.game_dates[team].append(gd)
            if len(self.game_dates[team]) > 20: self.game_dates[team] = self.game_dates[team][-20:]
        margin = abs(ps - pa)
        if margin <= 1:
            self.close_results[team].append(1.0 if won else 0.0)
            if len(self.close_results[team]) > 15: self.close_results[team] = self.close_results[team][-15:]
        if margin >= 5:
            self.blowout_results[team].append(1.0 if won else 0.0)
            if len(self.blowout_results[team]) > 15: self.blowout_results[team] = self.blowout_results[team][-15:]
        if gd is not None: self.last_date[team] = gd

FEATURE_COLS = [
    "elo_prob", "elo_diff", "player_diff",
    "pitcher_diff", "h_pitcher", "a_pitcher",
    "h_ppg", "h_papg", "h_win_pct", "h_margin",
    "a_ppg", "a_papg", "a_win_pct", "a_margin",
    "ppg_diff", "papg_diff", "win_pct_diff", "margin_diff", "off_diff", "def_diff",
    "h_rest", "a_rest", "rest_diff",
    "h_pyth", "a_pyth", "pyth_diff",
    "h_streak", "a_streak", "streak_diff", "h_consistency", "a_consistency",
    "h_trend", "a_trend", "trend_diff",
    "h_venue_wp", "a_venue_wp", "venue_wp_diff",
    "h_fatigue", "a_fatigue", "fatigue_diff",
    "h_clutch", "a_clutch", "clutch_diff",
    "h_score_vol", "a_score_vol", "dominance_diff",
    "day_of_week", "month",
]

def build_feats(hf, af, ep, ed, pd_=0.0, pitd=0.0, hp=0.0, ap=0.0, dow=2, mon=7):
    if hf is None or af is None: return None
    return {
        "elo_prob": ep, "elo_diff": ed, "player_diff": pd_,
        "pitcher_diff": pitd, "h_pitcher": hp, "a_pitcher": ap,
        "h_ppg": hf["ppg"], "h_papg": hf["papg"], "h_win_pct": hf["win_pct"], "h_margin": hf["avg_margin"],
        "a_ppg": af["ppg"], "a_papg": af["papg"], "a_win_pct": af["win_pct"], "a_margin": af["avg_margin"],
        "ppg_diff": hf["ppg"]-af["ppg"], "papg_diff": hf["papg"]-af["papg"],
        "win_pct_diff": hf["win_pct"]-af["win_pct"], "margin_diff": hf["avg_margin"]-af["avg_margin"],
        "off_diff": hf["off_rating"]-af["off_rating"], "def_diff": hf["def_rating"]-af["def_rating"],
        "h_rest": hf["rest_days"] if hf["rest_days"] is not None else 1.0,
        "a_rest": af["rest_days"] if af["rest_days"] is not None else 1.0,
        "rest_diff": (hf["rest_days"] or 1) - (af["rest_days"] or 1),
        "h_pyth": hf["pyth"], "a_pyth": af["pyth"], "pyth_diff": hf["pyth"]-af["pyth"],
        "h_streak": hf["streak"], "a_streak": af["streak"], "streak_diff": hf["streak"]-af["streak"],
        "h_consistency": hf["consistency"], "a_consistency": af["consistency"],
        "h_trend": hf["trend"], "a_trend": af["trend"], "trend_diff": hf["trend"]-af["trend"],
        "h_venue_wp": hf.get("venue_win_pct",0.5), "a_venue_wp": af.get("venue_win_pct",0.5),
        "venue_wp_diff": hf.get("venue_win_pct",0.5)-af.get("venue_win_pct",0.5),
        "h_fatigue": hf.get("fatigue",0.0), "a_fatigue": af.get("fatigue",0.0),
        "fatigue_diff": hf.get("fatigue",0.0)-af.get("fatigue",0.0),
        "h_clutch": hf.get("clutch",0.5), "a_clutch": af.get("clutch",0.5),
        "clutch_diff": hf.get("clutch",0.5)-af.get("clutch",0.5),
        "h_score_vol": hf.get("score_vol",2.5), "a_score_vol": af.get("score_vol",2.5),
        "dominance_diff": hf.get("dominance",0.5)-af.get("dominance",0.5),
        "day_of_week": float(dow), "month": float(mon),
    }

def sweep():
    settings = load_elo_settings()
    _elo_keys = {"base_rating","k","home_adv","use_mov","player_boost",
                 "starter_boost","rest_factor","form_weight","travel_factor",
                 "sos_factor","pace_factor","playoff_hca_factor"}
    games = pd.read_csv(GAMES_FILE)
    if "neutral_site" not in games.columns: games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")
    player_df = load_player_stats()

    model = MLBElo(**{k: v for k, v in settings.items() if k in _elo_keys})
    if not player_df.empty: model.set_player_stats(player_df)
    tracker = TeamTracker()
    elo_probs, actuals, feature_list = [], [], []
    prev_season = None

    for idx, row in games.iterrows():
        gd = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
        home, away = row["home_team"], row["away_team"]
        neutral = bool(row["neutral_site"])
        ha = 1 if row["home_score"] > row["away_score"] else 0
        hs = str(row.get("home_starter", "") or "").strip()
        as_ = str(row.get("away_starter", "") or "").strip()
        if gd is not None:
            row_season = gd.year
            if prev_season is not None and row_season != prev_season:
                model.regress_pitcher_ratings(factor=0.5)
            prev_season = row_season
        ep = model.win_prob(home, away, team_a_home=True, neutral_site=neutral, calibrated=False,
                            game_date=gd, use_injuries=False, home_starter=hs, away_starter=as_)
        ed = model.ratings[home] - model.ratings[away]
        pd_ = 0.0
        if model._player_scores and model.player_boost > 0:
            pd_ = model._player_scores.get(home, 0.0) - model._player_scores.get(away, 0.0)
        hpr = model._pitcher_ratings.get(hs, 0.0)
        apr = model._pitcher_ratings.get(as_, 0.0)
        hf = tracker.get_features(home, gd, is_home=True)
        af = tracker.get_features(away, gd, is_home=False)
        dow = gd.dayofweek if gd is not None else 2
        mon = gd.month if gd is not None else 7
        gf = build_feats(hf, af, ep, ed, pd_, hpr-apr, hpr, apr, dow, mon)
        elo_probs.append(ep); actuals.append(ha)
        feature_list.append([gf[c] for c in FEATURE_COLS] if gf else None)
        model.update_game(home, away, row["home_score"], row["away_score"],
                          neutral_site=neutral, game_date=gd, home_starter=hs, away_starter=as_)
        hw = row["home_score"] > row["away_score"]
        tracker.update(home, row["home_score"], row["away_score"], hw, gd, is_home=True)
        tracker.update(away, row["away_score"], row["home_score"], not hw, gd, is_home=False)

    n = len(actuals)
    elo_correct = sum(1 for i in range(n) if (elo_probs[i] >= 0.5) == (actuals[i] == 1))
    print(f"MLB SWEEP: {n} games, Pure Elo: {elo_correct/n*100:.2f}%")

    xgb_configs = [
        {"max_depth": 2, "eta": 0.08, "subsample": 0.75, "colsample_bytree": 0.5, "min_child_weight": 20, "lambda": 15.0, "alpha": 3.0, "rounds": 150, "label": "stumps"},
        {"max_depth": 3, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 10, "lambda": 5.0, "alpha": 1.0, "rounds": 200, "label": "conservative"},
        {"max_depth": 3, "eta": 0.1, "subsample": 0.7, "colsample_bytree": 0.5, "min_child_weight": 15, "lambda": 10.0, "alpha": 2.0, "rounds": 100, "label": "ultra_conservative"},
        {"max_depth": 4, "eta": 0.03, "subsample": 0.85, "colsample_bytree": 0.7, "min_child_weight": 5, "lambda": 3.0, "alpha": 0.5, "rounds": 300, "label": "moderate"},
        {"max_depth": 3, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 10, "lambda": 5.0, "alpha": 1.0, "rounds": 200, "label": "mlb_tuned"},
    ]
    elo_weights = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    best_acc, best_config = 0, ""

    for xc in xgb_configs:
        xgb_model, train_X, train_y = None, [], []
        xgb_preds = [None] * n
        for i in range(n):
            if feature_list[i] is not None:
                if xgb_model is not None:
                    fvec = np.array([feature_list[i]])
                    xgb_preds[i] = float(xgb_model.predict(xgb.DMatrix(fvec, feature_names=FEATURE_COLS))[0])
                train_X.append(feature_list[i]); train_y.append(actuals[i])
                if len(train_X) >= 200 and len(train_X) % 50 == 0:
                    dtrain = xgb.DMatrix(np.array(train_X), label=np.array(train_y), feature_names=FEATURE_COLS)
                    params = {"objective": "binary:logistic", "eval_metric": "logloss",
                              "max_depth": xc["max_depth"], "eta": xc["eta"], "subsample": xc["subsample"],
                              "colsample_bytree": xc["colsample_bytree"], "min_child_weight": xc["min_child_weight"],
                              "lambda": xc["lambda"], "alpha": xc["alpha"], "verbosity": 0}
                    xgb_model = xgb.train(params, dtrain, num_boost_round=xc["rounds"], verbose_eval=False)
        for ew in elo_weights:
            correct = sum(1 for i in range(n) if ((ew * elo_probs[i] + (1-ew) * xgb_preds[i] if xgb_preds[i] is not None else elo_probs[i]) >= 0.5) == (actuals[i] == 1))
            acc = correct / n * 100
            if acc > best_acc:
                best_acc = acc; best_config = f"xgb={xc['label']} ew={ew}"
                print(f"  NEW BEST: {best_config} -> {acc:.2f}%")

    print(f"\nBEST MLB ENSEMBLE: {best_config} -> {best_acc:.2f}%")
    print(f"Pure Elo baseline: {elo_correct/n*100:.2f}%")
    print(f"Improvement: {best_acc - elo_correct/n*100:+.2f}%")

if __name__ == "__main__":
    sweep()
