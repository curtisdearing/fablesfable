"""Phase 6.6: the historical significance study for every situational tag.

Population = the graded 2019-2025 candidate frame (data/ml_frame.parquet),
the same population the birthday/revenge verdicts were measured on. Verdict
machinery = context_study's exact bars (n>=100, BH-q<0.05, two-sided exact
binomial vs the pooled baseline), applied via study_frame().

Nothing here changes behavior. A tag that clears prints PROMOTABLE and still
needs a human to list it in config context_learning.enabled_tags.

Run:  python3 scripts/study_situations.py [--frame data/ml_frame.parquet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import context_study  # noqa: E402
from nflvalue import situations as sit  # noqa: E402

WEATHER_WIND = 15.0


def build_tag_table(frame: pd.DataFrame) -> pd.DataFrame:
    """(season, week, player_id, market, tag) rows for every tag firing."""
    from nflvalue.candidates import load_schedules
    from nflvalue.sources import rosters as rostersmod
    from nflvalue.weather_study import build_game_weather

    sched = load_schedules()
    gt = sit.game_tags(sched)

    tags = []

    # -- game-level situational tags, joined by (game_id, team) -------------- #
    f = frame.merge(gt.drop(columns=["season", "week", "home"]),
                    on=["game_id", "team"], how="left")
    for tag in ("primetime", "short_week", "long_travel_2tz",
                "west_east_early", "division_game"):
        hit = f[f[tag] == 1]
        tags.append(hit.assign(tag=tag)[["season", "week", "player_id", "market", "tag"]])

    # -- weather-conditioned split (outdoor wind >= 15) ----------------------- #
    gwx = build_game_weather()
    windy = set(gwx[(gwx["effective_outdoor"]) & (gwx["wind_mph"] >= WEATHER_WIND)]["game_id"])
    wf = frame[frame["game_id"].isin(windy)]
    tags.append(wf.assign(tag="wind15_game")[["season", "week", "player_id", "market", "tag"]])

    # -- birthday / revenge stratified ---------------------------------------- #
    if "is_birthday_week" in frame.columns:
        b = frame[frame["is_birthday_week"] == 1]
        tags.append(b[b["home"] == 1].assign(tag="birthday_home")[
            ["season", "week", "player_id", "market", "tag"]])
        tags.append(b[b["home"] == 0].assign(tag="birthday_away")[
            ["season", "week", "player_id", "market", "tag"]])
    if "revenge_game" in frame.columns and "opp_epa_factor" in frame.columns:
        r = frame[frame["revenge_game"] == 1].copy()
        tags.append(r[r["home"] == 1].assign(tag="revenge_home")[
            ["season", "week", "player_id", "market", "tag"]])
        tags.append(r[r["home"] == 0].assign(tag="revenge_away")[
            ["season", "week", "player_id", "market", "tag"]])
        terc = frame["opp_epa_factor"].quantile([1 / 3, 2 / 3]).tolist()
        r["oppq"] = np.where(r["opp_epa_factor"] <= terc[0], "tough",
                             np.where(r["opp_epa_factor"] >= terc[1], "soft", "mid"))
        for q in ("tough", "mid", "soft"):
            tags.append(r[r["oppq"] == q].assign(tag=f"revenge_vs_{q}")[
                ["season", "week", "player_id", "market", "tag"]])

    # -- revenge subtypes (trade / cut / fa) ----------------------------------- #
    seasons = sorted(frame["season"].unique().tolist())
    rosters = rostersmod.fetch_rosters_weekly([int(s) for s in seasons])
    stints: dict = {}
    rs = rosters.sort_values(["player_id", "season", "week"])
    for pid, grp in rs.groupby("player_id"):
        stints[pid] = list(zip(grp["season"].astype(int), grp["week"].astype(int), grp["team"]))
    names = rosters.drop_duplicates("player_id").set_index("player_id")["full_name"].to_dict()
    trades = sit.load_trade_moves()
    expiries = sit.contract_expiry_lookup()
    rev = frame[frame.get("revenge_game", 0) == 1]
    sub_rows = []
    for r in rev.itertuples(index=False):
        sub = sit.revenge_subtype(r.player_id, names.get(r.player_id, ""),
                                  int(r.season), int(r.week), r.team, r.defteam,
                                  stints, trades, expiries)
        if sub:
            sub_rows.append({"season": r.season, "week": r.week, "player_id": r.player_id,
                             "market": r.market, "tag": sub})
    if sub_rows:
        tags.append(pd.DataFrame(sub_rows))

    return pd.concat(tags, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", default=os.path.join(ROOT, "data", "ml_frame.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "situation_study.json"))
    args = ap.parse_args()

    frame = pd.read_parquet(args.frame)
    if "team" not in frame.columns:  # ml_frame keeps ids, not teams: join from pw
        from nflvalue.ingest import load_all_pbp
        from nflvalue.features import build_player_week
        pw = build_player_week(load_all_pbp())
        frame = frame.merge(pw[["season", "week", "player_id", "team", "defteam"]],
                            on=["season", "week", "player_id"], how="left")

    tag_table = build_tag_table(frame)
    res = context_study.study_frame(tag_table, frame)
    print(f"population n={len(frame):,}, baseline hit {res['baseline_hit_rate']:.4f}")
    print(f"{'tag':>20} {'n':>7} {'hit':>7} {'p':>8} {'q':>8}  verdict")
    for tag, r in sorted(res["tags"].items()):
        print(f"{tag:>20} {r['n']:>7,} {r['hit_rate']:>7.4f} "
              f"{(r['p_value'] if r['p_value'] is not None else float('nan')):>8.4f} "
              f"{(r.get('q_value') if r.get('q_value') is not None else float('nan')):>8.4f}  {r['verdict']}")
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
