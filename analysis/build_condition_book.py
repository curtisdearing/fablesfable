#!/usr/bin/env python3
"""Per-player condition book: how every player performs under every condition.

For each player (2020-2025) x condition: games, over-rate on their primary
stat (WR/TE: rec_yards, RB: rush_yards, QB: pass_yards) + receptions (pass
catchers) + TD rate, EB-shrunk toward the player's own unconditional rate
(k=25). 'edge' = shrunk delta vs own base; |edge|>=4pp & n>=6 flagged.

Rerun any week (in-season safe: trailing means are strictly-prior).
Output: player_condition_book.parquet / .csv + stadium_splits.csv
"""
import os
import numpy as np, pandas as pd

pw = pd.read_parquet("data/analysis_cache/pw_ctx2.parquet")

first_meet, rem = {}, []
for r in pw[["season", "week", "team", "opp"]].drop_duplicates().itertuples():
    key = (r.season, tuple(sorted((r.team, r.opp))))
    rem.append((key in first_meet and first_meet[key] < r.week, r.season, r.week, r.team, r.opp))
    if key not in first_meet: first_meet[key] = r.week
rm = pd.DataFrame(rem, columns=["is_rematch", "season", "week", "team", "opp"])
pw = pw.merge(rm, on=["season", "week", "team", "opp"], how="left")

CONDS = {
    "dome": pw["dome"], "outdoor": ~pw["dome"].fillna(False),
    "turf": pw["turf"], "grass": ~pw["turf"].fillna(False),
    "wind15+": pw["windy"], "cold<=32": pw["cold"],
    "primetime": pw["primetime"], "day_game": ~pw["primetime"].fillna(False),
    "home": pw["is_home"].fillna(False), "road": ~pw["is_home"].fillna(True),
    "short_week": pw["short_week"], "post_bye": pw["post_bye"],
    "big_favorite": pw["big_fav"], "big_underdog": pw["big_dog"],
    "revenge": pw["revenge"], "backup_qb": pw["backup_qb"],
    "division_rematch": pw["is_rematch"], "late_season": pw["late_season"],
    "body_clock_1pmET_road": pw["body_clock"], "after_OT": pw["ot_last"],
    "heavy_workload_lastwk": pw["heavy_last"], "after_100yd_game": pw["rec100_last"],
    "vs_high_blitz": pw["hi_blitz"], "fast_ref_crew": pw["fast_ref"], "slow_ref_crew": pw["slow_ref"],
}
PRIMARY = {"WR": "rec_yards", "TE": "rec_yards", "RB": "rush_yards", "QB": "pass_yards"}
K = 25

def over_col(d, stat):
    ok = d[f"tm_{stat}"].notna() & d[stat].notna() & (d[f"ng_{stat}"] >= 3)
    return ok, (d[stat] > d[f"tm_{stat}"])

rows = []
pw["primary"] = pw["role"].map(PRIMARY)
pw = pw[pw["primary"].notna()].copy()
for stat_kind in ["primary", "receptions", "td_any"]:
    if stat_kind == "primary":
        stats = pw["primary"]
    elif stat_kind == "receptions":
        stats = pd.Series(np.where(pw["role"].isin(["WR","TE","RB"]), "receptions", None), index=pw.index)
    else:
        stats = pd.Series(np.where(pw["role"].isin(["WR","TE","RB"]), "td_any", None), index=pw.index)
    for stat in [s for s in pd.unique(stats.dropna())]:
        sub = pw[stats == stat]
        ok, ov = over_col(sub, stat)
        sub = sub[ok]; ov = ov[ok]
        base = ov.groupby(sub["player_id"]).transform("mean")
        pbase = ov.groupby(sub["player_id"]).mean()
        ngames = ov.groupby(sub["player_id"]).size()
        for cname, cmask in CONDS.items():
            m = cmask.reindex(sub.index).fillna(False).astype(bool)
            if not m.any(): continue
            g = pd.DataFrame({"pid": sub.loc[m, "player_id"], "ov": ov[m].astype(int)})
            agg = g.groupby("pid")["ov"].agg(["size", "sum"])
            agg = agg[agg["size"] >= 3]
            if agg.empty: continue
            pb = pbase.reindex(agg.index)
            shrunk = (agg["sum"] + pb * K) / (agg["size"] + K)
            for pid, r in agg.iterrows():
                rows.append({"player_id": pid, "condition": cname, "stat": stat,
                             "n": int(r["size"]), "raw_over": round(r["sum"] / r["size"], 3),
                             "own_base": round(float(pb[pid]), 3),
                             "shrunk_over": round(float(shrunk[pid]), 3),
                             "edge_pp": round(float((shrunk[pid] - pb[pid]) * 100), 1),
                             "career_games": int(ngames.get(pid, 0))})

book = pd.DataFrame(rows)
names = pw.sort_values(["season","week"]).groupby("player_id").agg(
    name=("player_name","last"), role=("role","last"), team=("team","last"),
    last_season=("season","last"))
book = book.merge(names, left_on="player_id", right_index=True, how="left")
book["flag"] = (book["edge_pp"].abs() >= 4) & (book["n"] >= 6)
book = book.sort_values(["flag","edge_pp"], ascending=[False, False])
book.to_parquet("book/player_condition_book.parquet", index=False)
book.to_csv("book/player_condition_book.csv", index=False)
print("book rows:", len(book), "players:", book['player_id'].nunique(), "flagged:", int(book['flag'].sum()))

# stadium splits (away venues incl., n>=4) on primary stat
srows = []
for stat in ["rec_yards", "rush_yards", "pass_yards"]:
    sub = pw[pw["primary"] == stat]
    ok, ov = over_col(sub, stat)
    sub = sub[ok]; ov = ov[ok]
    pbase = ov.groupby(sub["player_id"]).mean()
    g = pd.DataFrame({"pid": sub["player_id"], "stad": sub["stadium"], "ov": ov.astype(int)})
    agg = g.groupby(["pid", "stad"])["ov"].agg(["size", "sum"])
    agg = agg[agg["size"] >= 4].reset_index()
    agg["own_base"] = agg["pid"].map(pbase)
    agg["shrunk"] = (agg["sum"] + agg["own_base"] * 20) / (agg["size"] + 20)
    agg["edge_pp"] = ((agg["shrunk"] - agg["own_base"]) * 100).round(1)
    agg["stat"] = stat
    srows.append(agg)
stad = pd.concat(srows, ignore_index=True).merge(names, left_on="pid", right_index=True, how="left")
stad = stad.sort_values("edge_pp", ascending=False)
stad.to_csv("book/stadium_splits.csv", index=False)
print("stadium split rows:", len(stad))

# top curiosities for the report
active = book[(book["last_season"] >= 2024) & book["flag"]]
print("\nTOP ACTIVE-PLAYER CONDITION EDGES (2024-25 rosters, n>=6):")
for _, r in pd.concat([active.head(18), active.tail(12)]).iterrows():
    print(f"  {r['name']:22s} {r['role']:2s} {r['team']:3s} | {r['condition']:22s} {r['stat']:11s} "
          f"n={r['n']:2d} raw {r['raw_over']:.2f} vs base {r['own_base']:.2f} -> edge {r['edge_pp']:+.1f}pp")
print("\nTOP STADIUM QUIRKS (active players):")
sa = stad[(stad["last_season"] >= 2024) & (stad["edge_pp"].abs() >= 5)]
for _, r in pd.concat([sa.head(12), sa.tail(8)]).iterrows():
    print(f"  {r['name']:22s} {r['role']:2s} @ {str(r['stad'])[:28]:28s} {r['stat']:10s} n={int(r['size']):2d} "
          f"raw {r['sum']/r['size']:.2f} vs base {r['own_base']:.2f} -> {r['edge_pp']:+.1f}pp")
