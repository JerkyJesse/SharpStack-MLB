"""Blogger HTML and plain-text table generation for today's games."""

import os
from datetime import datetime, timedelta

import pandas as pd

from config import get_team_abbr
from color_helpers import cok, cerr, cwarn, cdim, chi
from data_players import load_player_stats
from live_scores import fetch_live_mlb_scores, fetch_today_mlb_games, fetch_tomorrow_mlb_games, match_abbr_to_full


def _top_players_html(player_df, team_full):
    abbr = get_team_abbr(team_full)
    sub  = player_df[player_df["Tm"] == abbr] if not player_df.empty else pd.DataFrame()
    if sub.empty:
        return "<span style='color:#336633;font-size:12px;'>No data</span>"
    top   = sub.nlargest(3, "HR")
    lines = []
    for _, p in top.iterrows():
        parts   = str(p.get("Player","-")).strip().split()
        display = ("%s. %s" % (parts[0][0], " ".join(parts[1:]))) if len(parts) >= 2 else str(p.get("Player","-"))
        hr = float(p.get("HR", 0) or 0)
        rbi = float(p.get("RBI", 0) or 0)
        avg = float(p.get("AVG", 0) or 0)
        lines.append(
            "<div style='margin:2px 0;font-size:12px;color:#00ff00;'>"
            "<b>%s</b> &nbsp;%dHR / %dRBI / %.3fAVG</div>" % (display, hr, rbi, avg)
        )
    return "".join(lines)


def generate_today_predictions_html(model, output_file="today_mlb_predictions.html"):
    games = fetch_today_mlb_games()
    if not games:
        print(cerr("No games data available today."))
        return
    player_df = load_player_stats()
    game_date = datetime.now()
    date_str  = game_date.strftime("%B %d, %Y")
    platt_note = " | Platt-calibrated" if model._platt_scaler else ""

    html  = "<p style='text-align:center;font-size:13px;color:#00ff00;margin-bottom:16px;"
    html += "font-family:Arial,sans-serif;'>\n"
    html += "  Elo model predictions &nbsp;|&nbsp; K=%.1f &nbsp;|&nbsp; " % model.k
    html += "Home Adv=%.1f%s &nbsp;|&nbsp; %s\n</p>\n\n" % (model.home_adv, platt_note, date_str)
    html += "<table style='width:100%;border-collapse:collapse;font-family:Arial,sans-serif;"
    html += "font-size:13px;background:#000;color:#00ff00;'>\n"
    html += "  <thead>\n    <tr style='background:#003300;color:#00ff00;'>\n"
    for col in ["Matchup","Winner","Win%","Elo Edge","Top Away Players","Top Home Players"]:
        align = "left" if col in ("Matchup","Top Away Players","Top Home Players") else "center"
        html += "      <th style='padding:9px 6px;border:1px solid #005500;text-align:%s;'>%s</th>\n" % (align, col)
    html += "    </tr>\n  </thead>\n  <tbody>\n"

    row_bg = ["#000000", "#001a00"]
    for i, g in enumerate(games):
        away_full = match_abbr_to_full(g.get("away_abbr",""), model) or g.get("away_team_city","")
        home_full = match_abbr_to_full(g.get("home_abbr",""), model) or g.get("home_team_city","")
        if not away_full or not home_full:
            continue
        home_win_p = model.win_prob(home_full, away_full, team_a_home=True, neutral_site=False, game_date=game_date)
        away_win_p = 1.0 - home_win_p
        elo_h      = model.ratings.get(home_full, 1500)
        elo_a      = model.ratings.get(away_full, 1500)
        elo_diff   = elo_h - elo_a
        if home_win_p >= away_win_p:
            predicted_winner = home_full
            win_pct          = home_win_p
            edge_label       = "+%d" % elo_diff
        else:
            predicted_winner = away_full
            win_pct          = away_win_p
            edge_label       = "%d" % (-elo_diff)
        matchup      = "%s @ %s" % (g.get("away_team_city", away_full), g.get("home_team_city", home_full))
        confidence   = "&#x1F525;" if win_pct >= 0.70 else ("&#x2705;" if win_pct >= 0.60 else "")
        winner_short = predicted_winner.split()[-1]
        bg = row_bg[i % 2]
        html += "    <tr style='background:%s;'>\n" % bg
        html += "      <td style='padding:8px 6px;border:1px solid #005500;font-weight:bold;color:#00ff00;'>%s</td>\n" % matchup
        html += "      <td style='padding:8px 6px;border:1px solid #005500;text-align:center;color:#00ff00;'>%s %s</td>\n" % (winner_short, confidence)
        html += "      <td style='padding:8px 6px;border:1px solid #005500;text-align:center;font-size:16px;font-weight:bold;color:#00ff00;'>%d%%</td>\n" % int(win_pct*100)
        html += "      <td style='padding:8px 6px;border:1px solid #005500;text-align:center;font-size:12px;color:#33cc33;'>%s pts</td>\n" % edge_label
        html += "      <td style='padding:8px 6px;border:1px solid #005500;'>%s</td>\n" % _top_players_html(player_df, away_full)
        html += "      <td style='padding:8px 6px;border:1px solid #005500;'>%s</td>\n" % _top_players_html(player_df, home_full)
        html += "    </tr>\n"

    html += "  </tbody>\n</table>\n\n"
    html += "<p style='text-align:center;font-size:11px;color:#33cc33;margin-top:10px;"
    html += "font-family:Arial,sans-serif;'>\n"
    html += "  &#x1F525; = 70%+ confidence &nbsp;|&nbsp; &#x2705; = 60-69% &nbsp;|&nbsp; "
    html += "Stats: season stats (HR / RBI / AVG)<br>\n"
    html += "  For entertainment &amp; analysis purposes only. Not financial advice.\n</p>\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(cok("Blogger HTML snippet saved -> %s" % os.path.abspath(output_file)))
    print(cdim("   Open the file, copy ALL, paste into Blogger's HTML view.\n"))
    generate_today_predictions_txt(model)


def generate_today_predictions_txt(model, output_file="today_mlb_predictions.txt"):
    games = fetch_today_mlb_games()
    if not games:
        print(cwarn("No games data available today."))
        return
    player_df = load_player_stats()
    game_date = datetime.now()
    date_str  = game_date.strftime("%B %d, %Y")
    platt_note = " [Platt-calibrated]" if model._platt_scaler else ""

    def abbrev_name(full):
        parts = str(full).strip().split()
        return ("%s.%s" % (parts[0][0], " ".join(parts[1:]))) if len(parts) >= 2 else full[:14]

    COL     = {"matchup":18,"winner":16,"pct":6,"edge":8,"players":36}
    total_w = sum(COL.values()) + len(COL)*3 + 1
    lines   = [
        "MLB PREDICTIONS - %s%s" % (date_str, platt_note),
        "Model: K=%.1f  HomeAdv=%.1f" % (model.k, model.home_adv),
        "=" * total_w,
        "%-18s | %-16s | %6s | %8s | %-36s | TOP HOME PLAYERS"
        % ("MATCHUP","PREDICTED WINNER","WIN%","ELO EDGE","TOP AWAY PLAYERS"),
        "-" * total_w,
    ]
    for g in games:
        away_full = match_abbr_to_full(g.get("away_abbr",""), model) or g.get("away_team_city","")
        home_full = match_abbr_to_full(g.get("home_abbr",""), model) or g.get("home_team_city","")
        if not away_full or not home_full:
            continue
        home_win_p = model.win_prob(home_full, away_full, team_a_home=True, neutral_site=False, game_date=game_date)
        away_win_p = 1.0 - home_win_p
        elo_diff   = model.ratings.get(home_full,1500) - model.ratings.get(away_full,1500)
        if home_win_p >= away_win_p:
            predicted_winner = home_full.split()[-1]
            win_pct          = home_win_p
            edge             = "+%d" % elo_diff
        else:
            predicted_winner = away_full.split()[-1]
            win_pct          = away_win_p
            edge             = "%d" % (-elo_diff)

        def top3(abbr):
            if player_df.empty:
                return "-"
            sub = player_df[player_df["Tm"]==abbr].nlargest(3,"HR")
            if sub.empty:
                return "-"
            return "  ".join("%s %dHR" % (abbrev_name(str(r["Player"])), int(r.get("HR",0)))
                             for _, r in sub.iterrows())

        confidence = " F" if win_pct >= 0.70 else (" V" if win_pct >= 0.60 else "")
        matchup    = "%s @ %s" % (g.get("away_abbr","?"), g.get("home_abbr","?"))
        lines.append("%-18s | %-16s | %6.0f%% | %8s | %-36s | %s"
                     % (matchup, predicted_winner + confidence, win_pct*100, edge,
                        top3(get_team_abbr(away_full)), top3(get_team_abbr(home_full))))

    lines += [
        "=" * total_w,
        "F = 70%+ confidence  |  V = 60-69%  |  Stats: HR season totals",
        "For entertainment & analysis purposes only.",
    ]
    content = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(cok("Plain text saved -> %s" % os.path.abspath(output_file)))
    print()
    print(content)


def generate_tomorrow_predictions_html(model, output_file="tomorrow_mlb_predictions.html"):
    """Generate HTML prediction table for tomorrow's scheduled MLB games."""
    games = fetch_tomorrow_mlb_games()
    if not games:
        print(cerr("No games scheduled for tomorrow."))
        return
    player_df  = load_player_stats()
    game_date  = datetime.now() + timedelta(days=1)
    date_str   = game_date.strftime("%B %d, %Y")
    platt_note = " | Platt-calibrated" if model._platt_scaler else ""

    html  = "<p style='text-align:center;font-size:13px;color:#00ff00;margin-bottom:16px;"
    html += "font-family:Arial,sans-serif;'>\n"
    html += "  Elo model predictions &nbsp;|&nbsp; K=%.1f &nbsp;|&nbsp; " % model.k
    html += "Home Adv=%.1f%s &nbsp;|&nbsp; %s\n</p>\n\n" % (model.home_adv, platt_note, date_str)
    html += "<table style='width:100%;border-collapse:collapse;font-family:Arial,sans-serif;"
    html += "font-size:13px;background:#000;color:#00ff00;'>\n"
    html += "  <thead>\n    <tr style='background:#003300;color:#00ff00;'>\n"
    for col in ["Matchup","Winner","Win%","Elo Edge","Top Away Players","Top Home Players"]:
        align = "left" if col in ("Matchup","Top Away Players","Top Home Players") else "center"
        html += "      <th style='padding:9px 6px;border:1px solid #005500;text-align:%s;'>%s</th>\n" % (align, col)
    html += "    </tr>\n  </thead>\n  <tbody>\n"

    row_bg = ["#000000", "#001a00"]
    for i, g in enumerate(games):
        away_full = match_abbr_to_full(g.get("away_abbr",""), model) or g.get("away_team_city","")
        home_full = match_abbr_to_full(g.get("home_abbr",""), model) or g.get("home_team_city","")
        if not away_full or not home_full:
            continue
        home_win_p = model.win_prob(home_full, away_full, team_a_home=True, neutral_site=False, game_date=game_date)
        away_win_p = 1.0 - home_win_p
        elo_h      = model.ratings.get(home_full, 1500)
        elo_a      = model.ratings.get(away_full, 1500)
        elo_diff   = elo_h - elo_a
        if home_win_p >= away_win_p:
            predicted_winner = home_full
            win_pct          = home_win_p
            edge_label       = "+%d" % elo_diff
        else:
            predicted_winner = away_full
            win_pct          = away_win_p
            edge_label       = "%d" % (-elo_diff)
        matchup      = "%s @ %s" % (g.get("away_team_city", away_full), g.get("home_team_city", home_full))
        confidence   = "&#x1F525;" if win_pct >= 0.70 else ("&#x2705;" if win_pct >= 0.60 else "")
        winner_short = predicted_winner.split()[-1]
        bg = row_bg[i % 2]
        html += "    <tr style='background:%s;'>\n" % bg
        html += "      <td style='padding:8px 6px;border:1px solid #005500;font-weight:bold;color:#00ff00;'>%s</td>\n" % matchup
        html += "      <td style='padding:8px 6px;border:1px solid #005500;text-align:center;color:#00ff00;'>%s %s</td>\n" % (winner_short, confidence)
        html += "      <td style='padding:8px 6px;border:1px solid #005500;text-align:center;font-size:16px;font-weight:bold;color:#00ff00;'>%d%%</td>\n" % int(win_pct*100)
        html += "      <td style='padding:8px 6px;border:1px solid #005500;text-align:center;font-size:12px;color:#33cc33;'>%s pts</td>\n" % edge_label
        html += "      <td style='padding:8px 6px;border:1px solid #005500;'>%s</td>\n" % _top_players_html(player_df, away_full)
        html += "      <td style='padding:8px 6px;border:1px solid #005500;'>%s</td>\n" % _top_players_html(player_df, home_full)
        html += "    </tr>\n"

    html += "  </tbody>\n</table>\n\n"
    html += "<p style='text-align:center;font-size:11px;color:#33cc33;margin-top:10px;"
    html += "font-family:Arial,sans-serif;'>\n"
    html += "  &#x1F525; = 70%+ confidence &nbsp;|&nbsp; &#x2705; = 60-69% &nbsp;|&nbsp; "
    html += "Stats: season stats (HR / RBI / AVG)<br>\n"
    html += "  For entertainment &amp; analysis purposes only. Not financial advice.\n</p>\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(cok("Tomorrow's Blogger HTML saved -> %s" % os.path.abspath(output_file)))
    print(cdim("   Open the file, copy ALL, paste into Blogger's HTML view.\n"))
