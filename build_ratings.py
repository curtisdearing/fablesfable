#!/usr/bin/env python3
"""Build team power ratings + league priors from historical NFL data.

Inputs:
    historical_lines.parquet         schedules 2019-2023 (repo root, frozen base)
    historical/lines_extra.parquet   schedules 2024->now (ingested seasons)
    historical/historical_pbp.parquet  play-by-play 2019-2023 (EPA, drive ids)
    historical/pbp_{2024,2025}.parquet per-season play-by-play (EPA, drive ids)
  Drive-outcome rates are preserved from data/league_priors.json (the reduced
  pbp parquets no longer carry fixed_drive_result); everything else is remeasured
  across all available seasons.

Outputs (in ./data/):
    league_priors.json     drive-outcome rates, drives/game, HFA, score sds
    ratings_current.json   latest rating per team (for live sims)
    backtest_games.json    every game with the ratings known BEFORE kickoff
                           (walk-forward, no lookahead) + the closing line

Method
------
* Ratings live in points/game: off[t] = points above an average offense,
  def[t] = points fewer than average allowed (higher = better defense).
* Walk-forward: we snapshot each team's rating *before* a game, then update it
  from the result (opponent-adjusted). No future information leaks in.
* Each new season the rating regresses toward the mean and is blended with an
  EPA-per-play prior from the play-by-play (EPA predicts next year better than
  points do).

Needs: pandas, pyarrow, numpy   (pip install pandas pyarrow numpy)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(ROOT, "historical")
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

K = 0.08          # learning rate for the weekly rating update
CARRY = 0.60      # how much of last season's rating carries over
EPA_PRIOR_W = 0.45  # weight on the EPA prior at season start

ABBR = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "JAC": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders",
    "OAK": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "SD": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "STL": "Los Angeles Rams",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders", "WSH": "Washington Commanders",
}


# Long-run drive-outcome rates (2019-2023 pbp). The reduced pbp parquets no
# longer carry ``fixed_drive_result``, so these can't be recomputed from disk;
# callers pass the previously measured values (data/league_priors.json) and we
# fall back to these constants only if none are available.
DEFAULT_DRIVE_OUTCOMES = {"td": 0.221, "fg": 0.145, "def_td": 0.023, "safety": 0.0026}


def league_priors(pbp: pd.DataFrame, sched: pd.DataFrame,
                  drive_outcomes: dict = None) -> dict:
    # Drives per team-game from the drive id alone (outcome column is gone).
    plays = pbp.dropna(subset=["posteam", "fixed_drive"])
    dpg = plays.groupby(["game_id", "posteam"])["fixed_drive"].nunique()
    s = sched.dropna(subset=["result", "total", "home_score", "away_score"])
    league_ppg = (s["home_score"].sum() + s["away_score"].sum()) / (2 * len(s))
    return {
        "drive_outcomes": drive_outcomes or DEFAULT_DRIVE_OUTCOMES,
        "drives_mean": float(dpg.mean()),
        "drives_sd": float(dpg.std()),
        "league_ppg": float(league_ppg),
        "hfa_points": float(s["result"].mean()),
        "margin_sd": float(s["result"].std()),
        "total_mean": float(s["total"].mean()),
        "total_sd": float(s["total"].std()),
        "seasons": sorted(int(x) for x in s["season"].unique()),
        "n_games": int(len(s)),
    }


SCHED_COLS = ["season", "week", "gameday", "gametime", "home_team", "away_team",
              "home_score", "away_score", "spread_line", "total_line",
              "home_moneyline", "away_moneyline", "result", "total"]
PBP_COLS = ["game_id", "season", "posteam", "defteam", "fixed_drive", "epa"]
# schedules: frozen 2019-2023 base (repo root) + every season ingested since
# (historical/lines_extra.parquet, 2024->now); pbp: base + one file per season.
SCHED_FILES = [os.path.join(ROOT, "historical_lines.parquet"),
               os.path.join(HIST, "lines_extra.parquet")]
PBP_FILES = [os.path.join(HIST, "historical_pbp.parquet"),
             os.path.join(HIST, "pbp_2024.parquet"),
             os.path.join(HIST, "pbp_2025.parquet")]


def build():
    print("Loading data...")
    sched = pd.concat([pd.read_parquet(p, columns=SCHED_COLS)
                       for p in SCHED_FILES if os.path.exists(p)], ignore_index=True)
    sched = sched.dropna(subset=["home_score", "away_score", "spread_line", "total_line"])
    sched = sched.sort_values(["season", "week", "gameday", "gametime"]).reset_index(drop=True)
    pbp = pd.concat([pd.read_parquet(p, columns=PBP_COLS)
                     for p in PBP_FILES if os.path.exists(p)], ignore_index=True)

    # Preserve the previously measured drive-outcome rates (see league_priors).
    prev_lp = os.path.join(DATA, "league_priors.json")
    prev_outcomes = None
    if os.path.exists(prev_lp):
        prev_outcomes = json.load(open(prev_lp)).get("drive_outcomes")
    priors = league_priors(pbp, sched, drive_outcomes=prev_outcomes)
    league_ppg = priors["league_ppg"]
    hfa = priors["hfa_points"]
    print(f"  league PPG {league_ppg:.2f}  HFA {hfa:.2f}  games {priors['n_games']}")

    # ---- EPA per team-season, mapped to a points scale -------------------- #
    ep = pbp.dropna(subset=["epa", "posteam", "defteam"])
    epa_off = ep.groupby(["season", "posteam"])["epa"].mean()
    epa_def = ep.groupby(["season", "defteam"])["epa"].mean()
    # points/game scored & allowed per team-season from schedule
    long = pd.concat([
        sched.rename(columns={"home_team": "team", "home_score": "pf", "away_score": "pa"})[["season", "team", "pf", "pa"]],
        sched.rename(columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})[["season", "team", "pf", "pa"]],
    ])
    ppg = long.groupby(["season", "team"]).agg(pf=("pf", "mean"), pa=("pa", "mean"))
    # fit slope points ~ epa for offense and defense
    jo = ppg.join(epa_off.rename("epa_off"), on=["season", "team"]).dropna()
    bo = np.polyfit(jo["epa_off"], jo["pf"], 1)[0]
    jd = ppg.join(epa_def.rename("epa_def"), on=["season", "team"]).dropna()
    bd = np.polyfit(jd["epa_def"], jd["pa"], 1)[0]
    mean_epa_off, mean_epa_def = epa_off.mean(), epa_def.mean()

    def epa_off_pts(season, team):
        v = epa_off.get((season, team))
        return 0.0 if v is None or pd.isna(v) else bo * (v - mean_epa_off)

    def epa_def_pts(season, team):
        v = epa_def.get((season, team))   # points ABOVE avg allowed; good D => negative
        return 0.0 if v is None or pd.isna(v) else -bd * (v - mean_epa_def)

    # ---- Walk-forward rolling ratings ------------------------------------- #
    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    off = {t: 0.0 for t in teams}
    deff = {t: 0.0 for t in teams}
    games_played = {t: 0 for t in teams}
    backtest = []
    cur_season = None

    for _, g in sched.iterrows():
        season = int(g["season"])
        h, a = g["home_team"], g["away_team"]
        if season != cur_season:
            # new season: regress to mean and blend EPA prior from PRIOR season
            for t in teams:
                base_off = CARRY * off[t]
                base_def = CARRY * deff[t]
                if cur_season is not None:
                    off[t] = (1 - EPA_PRIOR_W) * base_off + EPA_PRIOR_W * epa_off_pts(cur_season, t)
                    deff[t] = (1 - EPA_PRIOR_W) * base_def + EPA_PRIOR_W * epa_def_pts(cur_season, t)
                else:
                    off[t], deff[t] = base_off, base_def
            games_played = {t: 0 for t in teams}
            cur_season = season

        ready = (season > priors["seasons"][0]) or (games_played[h] >= 3 and games_played[a] >= 3)
        backtest.append({
            "season": season, "week": int(g["week"]), "gameday": str(g["gameday"]),
            "home": h, "away": a,
            "off_home": round(off[h], 3), "def_home": round(deff[h], 3),
            "off_away": round(off[a], 3), "def_away": round(deff[a], 3),
            "spread_line": float(g["spread_line"]), "total_line": float(g["total_line"]),
            "home_ml": _f(g.get("home_moneyline")), "away_ml": _f(g.get("away_moneyline")),
            "home_score": float(g["home_score"]), "away_score": float(g["away_score"]),
            "ready": bool(ready),
        })

        # update from the observed result (opponent-adjusted)
        pred_h = league_ppg + off[h] - deff[a] + hfa / 2
        pred_a = league_ppg + off[a] - deff[h] - hfa / 2
        eh = g["home_score"] - pred_h
        ea = g["away_score"] - pred_a
        off[h] += K * eh
        deff[a] += -K * eh
        off[a] += K * ea
        deff[h] += -K * ea
        games_played[h] += 1
        games_played[a] += 1

    # ---- current ratings (latest state), keyed by full team name ---------- #
    current = {}
    for t in teams:
        name = ABBR.get(t, t)
        current[name] = {
            "abbr": t, "off": round(off[t], 3), "def": round(deff[t], 3),
            "net": round(off[t] + deff[t], 3), "season": cur_season,
        }

    with open(os.path.join(DATA, "league_priors.json"), "w") as f:
        json.dump(priors, f, indent=2)
    with open(os.path.join(DATA, "ratings_current.json"), "w") as f:
        json.dump(current, f, indent=2)
    with open(os.path.join(DATA, "backtest_games.json"), "w") as f:
        json.dump(backtest, f)

    print(f"  wrote ratings for {len(current)} teams; {len(backtest)} backtest games")
    top = sorted(current.items(), key=lambda kv: -kv[1]["net"])[:6]
    print("  strongest (end of data):")
    for name, r in top:
        print(f"    {name:24} net {r['net']:+5.1f}  (off {r['off']:+.1f}, def {r['def']:+.1f})")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    build()
