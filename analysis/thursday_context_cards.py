"""Stadium / opponent / primetime / stadium x primetime context cards for one game's players.

Descriptive history only: every card goes through ``factor_evidence.split_context_record``
(always ``context_only``; fewer than ``MIN_SPLIT_GAMES`` games reads "Insufficient
evidence") and the resulting panel is built by the same ``build_panel`` consumer the site
uses, so the printed statuses are what the consumer produced, not what this script claims.
Nothing here feeds a projection, a selection score or a stake.

Inputs
------
* nflverse play-by-play parquet files (player-game volume; regular season only).
* nflverse ``games.csv`` for the schedule: ``gameday``, ``gametime`` (US/Eastern),
  ``stadium_id`` (the PHYSICAL stadium), ``location``.  Its betting columns are never read.

A game counts for a player when the player has at least one attempt/carry/target in it
(involvement, not snaps).  Local kickoff hour uses the stadium's time zone: the home
team's zone, or a named neutral-site zone; a neutral site not in the table is left
unclassified (counted, never guessed).  Team changes and season-level role shares are
reported alongside, because a split pooled across teams/roles is not one player's history.

Run:  python -m analysis.thursday_context_cards --game 2026_03_ATL_GB \
          --players 00-0038542:rush_attempts,... --games-csv games.csv \
          --pbp a.parquet b.parquet --as-of 2026-09-23T22:40:00Z --out cards.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd

from nflvalue import factor_evidence as fe

ET = ZoneInfo("America/New_York")
TEAM_TZ = {t: "America/New_York" for t in (
    "ATL BAL BUF CAR CIN CLE DET IND JAX MIA NE NYG NYJ PHI PIT TB WAS").split()}
TEAM_TZ.update({t: "America/Chicago" for t in "CHI DAL GB HOU KC MIN NO TEN STL".split()})
TEAM_TZ.update({t: "America/Los_Angeles" for t in "LV LAC LA SEA SF OAK SD".split()})
TEAM_TZ.update({"ARI": "America/Phoenix", "DEN": "America/Denver"})
NEUTRAL_TZ = {"london": "Europe/London", "tottenham": "Europe/London", "wembley": "Europe/London",
              "allianz": "Europe/Berlin", "deutsche bank": "Europe/Berlin",
              "frankfurt": "Europe/Berlin", "olympiastadion": "Europe/Berlin",
              "azteca": "America/Mexico_City", "corinthians": "America/Sao_Paulo",
              "croke": "Europe/Dublin", "bernab": "Europe/Madrid"}
STAT = {"rush_attempts": "rush", "pass_attempts": "pass", "passing_yards": "pass",
        "receiving_yards": "recv", "receptions": "recv"}
PBP_COLS = ["season", "week", "game_id", "season_type", "posteam", "defteam", "pass_attempt",
            "rush_attempt", "complete_pass", "passing_yards", "sack", "receiver_player_id",
            "rusher_player_id", "passer_player_id"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_kickoff(row) -> Dict:
    if pd.isna(row.gametime) or pd.isna(row.gameday):
        return {"kickoff_utc": None, "local_hour": None, "tz": None}
    et = dt.datetime.fromisoformat(f"{row.gameday}T{row.gametime}").replace(tzinfo=ET)
    if row.location == "Neutral":
        tz = next((z for k, z in NEUTRAL_TZ.items() if k in str(row.stadium).lower()), None)
    else:
        tz = TEAM_TZ.get(row.home_team)
    if tz is None:
        return {"kickoff_utc": et.astimezone(dt.timezone.utc).isoformat(), "local_hour": None, "tz": None}
    loc = et.astimezone(ZoneInfo(tz))
    return {"kickoff_utc": et.astimezone(dt.timezone.utc).isoformat(), "local_hour": loc.hour, "tz": tz}


def player_games(pbp: pd.DataFrame, pid: str, market: str) -> pd.DataFrame:
    kind = STAT[market]
    if kind == "rush":
        p = pbp[(pbp.rusher_player_id == pid) & (pbp.rush_attempt == 1)]
        agg = p.groupby(["game_id", "posteam", "defteam"]).size().rename("value")
    elif kind == "pass":
        p = pbp[(pbp.passer_player_id == pid) & (pbp.pass_attempt == 1) & (pbp.sack != 1)]
        g = p.groupby(["game_id", "posteam", "defteam"])
        agg = (g.size() if market == "pass_attempts" else g.passing_yards.sum()).rename("value")
    else:
        p = pbp[(pbp.receiver_player_id == pid) & (pbp.pass_attempt == 1)]
        g = p.groupby(["game_id", "posteam", "defteam"])
        agg = (g.complete_pass.sum() if market == "receptions"
               else p[p.complete_pass == 1].groupby(["game_id", "posteam", "defteam"]).passing_yards.sum()
               .reindex(g.size().index, fill_value=0.0)).rename("value")
    return agg.reset_index().rename(columns={"posteam": "team", "defteam": "opp"})


def role_by_season(pbp: pd.DataFrame, pid: str, market: str) -> List[Dict]:
    kind, out = STAT[market], []
    col = {"rush": "rusher_player_id", "pass": "passer_player_id", "recv": "receiver_player_id"}[kind]
    flag = "rush_attempt" if kind == "rush" else "pass_attempt"
    base = pbp[(pbp[flag] == 1) & ((pbp.sack != 1) if kind == "pass" else True)]
    mine = base[base[col] == pid]
    for (season, team), m in mine.groupby(["season", "posteam"]):
        team_n = len(base[(base.season == season) & (base.posteam == team)
                          & base.game_id.isin(m.game_id.unique())])
        out.append({"season": int(season), "team": team, "games": int(m.game_id.nunique()),
                    "share_of_team_" + ("carries" if kind == "rush" else "pass_attempts" if kind == "pass"
                                        else "targets"): round(len(m) / team_n, 3) if team_n else None})
    return out


def cards_for(pbp, sched, pid, name, market, target, as_of) -> Dict:
    games = player_games(pbp, pid, market).merge(sched, on="game_id", how="left")
    games = games[games.gameday.notna() & (games.gameday < target.gameday)].sort_values("gameday")
    team_now = target.home_team if target.home_team in set(games.team.tail(3)) else target.away_team
    opp = target.away_team if team_now == target.home_team else target.home_team
    games["primetime"] = games.local_hour.map(lambda h: None if pd.isna(h) else h >= fe.PRIMETIME_LOCAL_HOUR)
    cutoff = str(games.gameday.max())[:10] if len(games) else as_of[:10]
    base_mean, base_n = (float(games.value.mean()) if len(games) else None), len(games)
    splits = {
        "venue": games[games.stadium_id == target.stadium_id],
        "opponent": games[games.opp == opp],
        "primetime": games[games.primetime == True],  # noqa: E712 -- None stays out
        "venue_x_primetime": games[(games.stadium_id == target.stadium_id) & (games.primetime == True)],  # noqa: E712
    }
    labels = {"venue": f"at {target.stadium} ({target.stadium_id}, physical stadium)",
              "opponent": f"vs {opp}", "primetime": "primetime (local kickoff >= 18:00)",
              "venue_x_primetime": f"primetime at {target.stadium}"}
    recs, detail = [], {}
    for kind, sub in splits.items():
        gl = [{"game_id": r.game_id, "gameday": r.gameday} for r in sub.itertuples()]
        rec = fe.split_context_record(
            entity_id=pid, entity_type="player", game_id=target.game_id, as_of=as_of,
            split_kind=labels[kind], stat=market, split_mean=float(sub.value.mean()) if len(sub) else None,
            split_n=len(sub), baseline_mean=base_mean, baseline_n=base_n, cutoff=cutoff, games=gl,
            source_id="nflverse_pbp+games.csv", team=team_now)
        recs.append(rec)
        detail[kind] = {"n": len(sub), "mean": rec["value"], "baseline_mean": base_mean, "baseline_n": base_n,
                        "sufficient": len(sub) >= fe.MIN_SPLIT_GAMES,
                        "teams_in_split": sorted(set(sub.team)),
                        "games": [{"game_id": r.game_id, "gameday": r.gameday, "team": r.team, "opp": r.opp,
                                   "local_hour": None if pd.isna(r.local_hour) else int(r.local_hour),
                                   "value": float(r.value)} for r in sub.itertuples()]}
    return {"player_id": pid, "player": name, "market": market, "team_now": team_now, "opponent": opp,
            "baseline": {"n": base_n, "mean": base_mean, "first": str(games.gameday.min())[:10] if base_n else None,
                         "last": cutoff, "teams": sorted(set(games.team))},
            "unclassified_local_hour": int(games.local_hour.isna().sum()),
            "role_by_season": role_by_season(pbp, pid, market), "splits": detail, "records": recs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--players", required=True, help="pid:name:market,...")
    ap.add_argument("--games-csv", required=True)
    ap.add_argument("--pbp", nargs="+", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sched = pd.read_csv(a.games_csv, usecols=["game_id", "season", "game_type", "week", "gameday", "gametime",
                                              "home_team", "away_team", "location", "stadium_id", "stadium"])
    target = next(sched[sched.game_id == a.game].itertuples())
    sched = sched[(sched.game_type == "REG") & (sched.season >= 2019)].copy()
    sched = pd.concat([sched, sched.apply(local_kickoff, axis=1, result_type="expand")], axis=1)
    pbp = pd.concat([pd.read_parquet(p, columns=PBP_COLS) for p in a.pbp], ignore_index=True)
    pbp = pbp[pbp.season_type == "REG"]
    out = {"schema": "thursday-context-cards-v1", "game_id": a.game, "as_of": a.as_of,
           "target": {"gameday": target.gameday, "gametime_et": target.gametime, "stadium_id": target.stadium_id,
                      "stadium": target.stadium, **local_kickoff(target)},
           "inputs": {"games_csv_sha256": sha256(a.games_csv),
                      "pbp": {p: sha256(p) for p in a.pbp}, "pbp_seasons": sorted(map(int, pbp.season.unique()))},
           "min_split_games": fe.MIN_SPLIT_GAMES, "primetime_local_hour": fe.PRIMETIME_LOCAL_HOUR, "cards": []}
    all_recs = []
    for spec in a.players.split(","):
        pid, name, market = spec.split(":")
        c = cards_for(pbp, sched, pid, name, market, target, a.as_of)
        all_recs += c["records"]
        out["cards"].append(c)
    panel = fe.build_panel(all_recs, a.as_of)
    out["consumer_panel_status_counts"] = pd.Series([r["status"] for r in all_recs]).value_counts().to_dict()
    out["consumer_panel"] = panel
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    for c in out["cards"]:
        print(c["player"], c["market"], "baseline", c["baseline"]["n"], round(c["baseline"]["mean"] or 0, 1),
              c["baseline"]["teams"], {k: (v["n"], None if v["mean"] is None else round(v["mean"], 1),
                                           "ok" if v["sufficient"] else "INSUFFICIENT")
                                       for k, v in c["splits"].items()})
    print("consumer statuses:", out["consumer_panel_status_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
