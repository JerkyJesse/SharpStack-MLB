"""Auto-resolve finished trades using live score data."""

from config import get_team_abbr
from color_helpers import cok, cerr, cdim, chi
from live_scores import fetch_live_mlb_scores, match_abbr_to_full
from predict_ledger import load_predict_lots, save_predict_lots


def auto_resolve_finished_trades(model, verbose=True):
    df      = load_predict_lots()
    open_df = df[df["contracts_open"].fillna(0).astype(float) > 0].copy()
    if open_df.empty:
        if verbose:
            print(cdim("  [auto-resolve] No open lots to check."))
        return 0
    live        = fetch_live_mlb_scores()
    final_games = [g for g in live if "final" in str(g.get("status","")).lower()]
    if not final_games:
        if verbose:
            print(cdim("  [auto-resolve] No completed (final) games found in today's feed."))
        return 0
    resolved_count = 0
    for real_idx, row in open_df.iterrows():
        home_full = str(row["home_team"])
        away_full = str(row["away_team"])
        predicted = str(row["predicted_winner"])
        lot_abbrs = {get_team_abbr(home_full), get_team_abbr(away_full)}
        matched   = None
        for g in final_games:
            if {g["home_abbr"], g["away_abbr"]} == lot_abbrs:
                matched = g
                break
            live_home = (match_abbr_to_full(g["home_abbr"], model) or g["home_team_city"]).lower()
            live_away = (match_abbr_to_full(g["away_abbr"], model) or g["away_team_city"]).lower()
            lot_names = {home_full.lower(), away_full.lower()}
            if sum(1 for n in lot_names for ln in (live_home, live_away) if n in ln or ln in n) >= 2:
                matched = g
                break
        if matched is None:
            continue
        home_score = int(matched.get("home_score", 0))
        away_score = int(matched.get("away_score", 0))
        if home_score > away_score:
            winning_abbr = matched["home_abbr"]
        elif away_score > home_score:
            winning_abbr = matched["away_abbr"]
        else:
            winning_abbr = None
        pred_abbr = get_team_abbr(predicted)
        correct   = winning_abbr is not None and pred_abbr == winning_abbr
        contracts_open     = int(row["contracts_open"])
        avg_entry          = float(row["avg_entry_price"])
        entry_fee_total    = float(row.get("entry_fee_total", 0))
        original_contracts = int(row.get("contracts_original", contracts_open))
        allocated_fee      = round(entry_fee_total * contracts_open / max(original_contracts, 1), 4)
        cost_basis         = round(contracts_open * avg_entry + allocated_fee, 4)
        if correct:
            winnings = float(contracts_open) * 1.0
            profit   = round(winnings - cost_basis, 4)
            df.loc[real_idx, "realized_cash"]   = round(float(df.loc[real_idx, "realized_cash"]   or 0) + winnings, 4)
            df.loc[real_idx, "realized_profit"] = round(float(df.loc[real_idx, "realized_profit"] or 0) + profit,   4)
            result_str = "%s WON" % predicted
        else:
            profit     = round(-cost_basis, 4)
            df.loc[real_idx, "realized_profit"] = round(float(df.loc[real_idx, "realized_profit"] or 0) + profit, 4)
            result_str = "%s LOST" % predicted
        df.loc[real_idx, "contracts_open"] = 0
        df.loc[real_idx, "status"]         = "settled"
        note = str(df.loc[real_idx, "notes"] or "")
        df.loc[real_idx, "notes"] = "%s | auto-resolved: %s (final score)" % (note, result_str)
        resolved_count += 1
        if verbose:
            outcome = "WIN" if correct else "LOSS"
            out_s   = cok(outcome) if correct else cerr(outcome)
            pnl_s   = cok("+$%.4f" % profit) if correct else cerr("$%.4f" % profit)
            print("  %s Lot #%d | %s | %s | P&L: %s"
                  % (chi("[auto-resolve]"), int(row["lot_id"]), out_s, result_str, pnl_s))
    if resolved_count > 0:
        save_predict_lots(df)
        if verbose:
            print(cok("  [auto-resolve] Successfully settled %d lot(s)." % resolved_count))
    elif verbose:
        print(cdim("  [auto-resolve] No open lots matched any finished game today."))
    return resolved_count
