#!/usr/bin/env python3
"""Weekly projections board on real lines — fantasy-style, updates as results come in.

For each NFL week it projects every game on its REAL closing line using the
Monte Carlo with walk-forward ratings (ratings known before that week only), then
grades the picks once the week's results are in and grows a season-to-date record.

    python3 weekly.py                 # latest historical season, all weeks graded
    python3 weekly.py --season 2022   # pick a season (2019-2023 in your data)
    python3 weekly.py --through 6     # reveal weeks 1-6 graded, week 7 as upcoming
    python3 weekly.py --live          # in season: project THIS week's live lines

Writes data/weekly.json, which the dashboard's "Weekly" tab renders.
Needs numpy (for the simulator).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

from nflvalue import config, montecarlo as mc

ABBR_OFF = None  # filled from ratings if needed


def _conf(edge_pts):
    e = abs(edge_pts)
    return "high" if e >= 3 else ("med" if e >= 1.5 else "low")


def _project_game(g, priors, sims, ratings=None):
    """Simulate one game from its pre-game ratings; return a board row."""
    if "off_home" in g:                       # historical game (ratings embedded)
        home = {"off": g["off_home"], "def": g["def_home"]}
        away = {"off": g["off_away"], "def": g["def_away"]}
    else:                                     # live game (look up current ratings)
        home = ratings.get(g["home"]) or {"off": 0, "def": 0}
        away = ratings.get(g["away"]) or {"off": 0, "def": 0}

    sp, tot = g["spread_line"], g["total_line"]
    r = mc.simulate(home, away, priors, spread_line=sp, total_line=tot, n=sims)
    home_point = -sp                          # market home spread (e.g. -6.5)

    ats_edge = r["margin_mean"] - sp          # >0 => model likes home vs the line
    ats_side = "home" if ats_edge > 0 else "away"
    tot_edge = r["total_mean"] - tot          # >0 => model likes the over
    tot_side = "over" if tot_edge > 0 else "under"
    su_side = "home" if r["p_home_win"] >= 0.5 else "away"

    row = {
        "home": g["home"], "away": g["away"],
        "market_spread_home": round(home_point, 1), "market_total": tot,
        "proj_home": r["exp_home"], "proj_away": r["exp_away"],
        "proj_margin": r["margin_mean"], "proj_total": r["total_mean"],
        "p_home_win": r["p_home_win"], "p_away_win": r["p_away_win"],
        "fair_home_ml": mc.fair_moneyline_odds(r["p_home_win"]),
        "ats_pick": {"side": ats_side,
                     "team": g["home"] if ats_side == "home" else g["away"],
                     "line": round(home_point if ats_side == "home" else -home_point, 1),
                     "edge": round(ats_edge if ats_side == "home" else -ats_edge, 1),
                     "conf": _conf(ats_edge)},
        "total_pick": {"side": tot_side, "line": tot,
                       "edge": round(abs(tot_edge), 1), "conf": _conf(tot_edge)},
        "su_pick": g["home"] if su_side == "home" else g["away"],
        "settled": False,
    }

    if g.get("home_score") is not None and g.get("away_score") is not None and g.get("_played", True):
        hs, as_ = g["home_score"], g["away_score"]
        margin, total = hs - as_, hs + as_
        row.update({"settled": True, "home_score": hs, "away_score": as_})
        row["su_correct"] = (margin > 0) == (su_side == "home") if margin != 0 else None
        # ATS result for the picked side
        if margin == sp:
            row["ats_result"] = "P"
        elif ats_side == "home":
            row["ats_result"] = "W" if margin > sp else "L"
        else:
            row["ats_result"] = "W" if margin < sp else "L"
        # total result
        if total == tot:
            row["total_result"] = "P"
        elif tot_side == "over":
            row["total_result"] = "W" if total > tot else "L"
        else:
            row["total_result"] = "W" if total < tot else "L"
        row["margin_err"] = round(abs(r["margin_mean"] - margin), 1)
        row["total_err"] = round(abs(r["total_mean"] - total), 1)
    return row


def _blank_record():
    return {"su_correct": 0, "su_total": 0, "ats_w": 0, "ats_l": 0, "ats_p": 0,
            "tot_w": 0, "tot_l": 0, "tot_p": 0, "margin_err": 0.0, "total_err": 0.0,
            "graded": 0}


def _apply(rec, row):
    if not row["settled"]:
        return
    if row.get("su_correct") is not None:
        rec["su_total"] += 1
        rec["su_correct"] += 1 if row["su_correct"] else 0
    rec["ats_w"] += row["ats_result"] == "W"
    rec["ats_l"] += row["ats_result"] == "L"
    rec["ats_p"] += row["ats_result"] == "P"
    rec["tot_w"] += row["total_result"] == "W"
    rec["tot_l"] += row["total_result"] == "L"
    rec["tot_p"] += row["total_result"] == "P"
    rec["margin_err"] += row["margin_err"]
    rec["total_err"] += row["total_err"]
    rec["graded"] += 1


def _summary(rec):
    g = rec["graded"] or 1
    ats_n = rec["ats_w"] + rec["ats_l"] or 1
    tot_n = rec["tot_w"] + rec["tot_l"] or 1
    return {
        "graded": rec["graded"],
        "su_pct": round(rec["su_correct"] / (rec["su_total"] or 1), 3),
        "su": f"{rec['su_correct']}-{rec['su_total'] - rec['su_correct']}",
        "ats": f"{rec['ats_w']}-{rec['ats_l']}" + (f"-{rec['ats_p']}" if rec["ats_p"] else ""),
        "ats_pct": round(rec["ats_w"] / ats_n, 3),
        "ats_roi": round((rec["ats_w"] * 0.909 - rec["ats_l"]) / ats_n, 3),
        "totals": f"{rec['tot_w']}-{rec['tot_l']}" + (f"-{rec['tot_p']}" if rec["tot_p"] else ""),
        "totals_pct": round(rec["tot_w"] / tot_n, 3),
        "avg_margin_err": round(rec["margin_err"] / g, 1),
        "avg_total_err": round(rec["total_err"] / g, 1),
    }


def build_historical(season=None, sims=5000, through=None):
    priors = config.load_json(os.path.join(config.DATA_DIR, "league_priors.json"), None)
    games = config.load_json(os.path.join(config.DATA_DIR, "backtest_games.json"), None)
    if not priors or not games:
        print("Missing data. Run:  python3 build_ratings.py")
        return None
    seasons = sorted({g["season"] for g in games})
    season = season or seasons[-1]
    sgames = [g for g in games if g["season"] == season]
    weeks = defaultdict(list)
    for g in sgames:
        weeks[g["week"]].append(g)

    if not weeks:
        print(f"No games for season {season}.")
        return None

    rec = _blank_record()
    out_weeks = []
    for wk in sorted(weeks):
        reveal = through is None or wk <= through
        rows = []
        for g in weeks[wk]:
            gg = dict(g)
            gg["_played"] = reveal       # hide results for "upcoming" weeks
            rows.append(_project_game(gg, priors, sims))
            _apply(rec, rows[-1])
        out_weeks.append({"week": wk, "label": _week_label(wk),
                          "games": rows, "record_to_date": _summary(rec)})
    return {
        "mode": "historical", "season": season, "seasons_available": seasons,
        "sims": sims, "weeks": out_weeks, "season_record": _summary(rec),
        "default_week": (through if through else max(weeks)),
    }


def build_live(cfg, sims=8000):
    """Project the current live slate (in season) on real odds + current ratings."""
    from nflvalue import pipeline
    from nflvalue.sources import live as livesrc
    priors = config.load_json(os.path.join(config.DATA_DIR, "league_priors.json"), None)
    ratings = config.load_json(os.path.join(config.DATA_DIR, "ratings_current.json"), None)
    if not priors or not ratings:
        return None
    games = livesrc.build_live_slate(cfg)
    rows = []
    for g in games:
        hp, tp = pipeline._market_lines(g)
        if hp is None or tp is None:
            continue
        gg = {"home": g["home_team"], "away": g["away_team"],
              "spread_line": -hp, "total_line": tp, "home_score": None, "away_score": None}
        rows.append(_project_game(gg, priors, sims, ratings))
    if not rows:
        return None
    return {"mode": "live", "season": "current", "seasons_available": [],
            "sims": sims, "weeks": [{"week": 0, "label": "This week",
                                     "games": rows, "record_to_date": _summary(_blank_record())}],
            "season_record": _summary(_blank_record()), "default_week": 0}


def _week_label(wk):
    if wk <= 18:
        return f"Week {wk}"
    return {19: "Wild Card", 20: "Divisional", 21: "Conf Champ", 22: "Super Bowl"}.get(wk, f"Week {wk}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--through", type=int, default=None, help="reveal results only through this week")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    cfg = config.load_config()

    if args.live and cfg.get("odds_api_key"):
        data = build_live(cfg, sims=args.sims)
        if not data:
            print("No live games (offseason?). Falling back to historical.")
            data = build_historical(args.season, args.sims, args.through)
    else:
        data = build_historical(args.season, args.sims, args.through)
    if not data:
        return
    config.save_json(os.path.join(config.DATA_DIR, "weekly.json"), data)

    sr = data["season_record"]
    print(f"  {data['mode'].upper()}  season {data['season']}  weeks {len(data['weeks'])}")
    print(f"  Season-to-date: SU {sr['su']} ({sr['su_pct']*100:.0f}%) | "
          f"ATS {sr['ats']} ({sr['ats_pct']*100:.0f}%) | totals {sr['totals']} | "
          f"avg margin error {sr['avg_margin_err']} pts")
    print("  Saved: data/weekly.json")


if __name__ == "__main__":
    main()
