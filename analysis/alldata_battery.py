#!/usr/bin/env python3
"""Cascade + headline patterns on ALL available data (2019-2025).

Absence = usage-based (roster leader by trailing CUMULATIVE usage, absent =
no positive-usage row that week; team played) UNION injury-report Out —
the same convention as data/absence_matrix.json. This is the honest,
maximum-n version; small-n injury-report-only numbers shown for comparison.
"""
import json, os
import numpy as np, pandas as pd

import pathlib; ROOT = str(pathlib.Path(__file__).resolve().parents[1])
HIST = os.path.join(ROOT, "historical")

pw = pd.read_parquet("data/analysis_cache/pw_cache.parquet")          # 2019-2025, all rows
pw = pw.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

# trailing means (strictly prior, within season)
pw["td_any"] = ((pw["rec_tds"].fillna(0) + pw["rush_tds"].fillna(0)) >= 1).astype(float)
g = pw.groupby(["player_id", "season"])
for s in ["targets", "receptions", "rec_yards", "carries", "rush_yards",
          "pass_attempts", "pass_yards", "td_any"]:
    pw[f"tm_{s}"] = g[s].transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    pw[f"ng_{s}"] = g[s].transform(lambda x: x.shift(1).notna().cumsum())
pw["cum_carries"] = g["carries"].transform(lambda x: x.shift(1).cumsum())
pw["cum_targets"] = g["targets"].transform(lambda x: x.shift(1).cumsum())

# ---- roster leaders by trailing CUM usage (absence_matrix convention, >=30/20)
week_played = pw.groupby(["season", "week", "team"]).size().rename("team_played")
present = set(zip(pw.season, pw.week, pw.team, pw.player_id))

lead_frames = []
for (t, s), d in pw[["player_id","season","week","team","role","cum_carries","cum_targets"]].groupby(["team","season"]):
    cum = {}
    for w in sorted(d["week"].unique()):
        cur = d[d["week"] == w]
        for r in cur.itertuples():   # cum values are strictly-prior already
            cum[r.player_id] = (r.role, r.cum_carries or 0, r.cum_targets or 0)
        rbs = sorted(((p,v) for p,v in cum.items() if v[0]=="RB" and v[1]>=30), key=lambda kv:-kv[1][1])
        tes = sorted(((p,v) for p,v in cum.items() if v[0]=="TE" and v[2]>=20), key=lambda kv:-kv[1][2])
        wrs = sorted(((p,v) for p,v in cum.items() if v[0]=="WR" and v[2]>=20), key=lambda kv:-kv[1][2])
        lead_frames.append((t,s,w, rbs[0][0] if rbs else None, rbs[1][0] if len(rbs)>1 else None,
                            tes[0][0] if tes else None, wrs[0][0] if wrs else None,
                            wrs[1][0] if len(wrs)>1 else None))
lead = pd.DataFrame(lead_frames, columns=["team","season","week","rb1","rb2","te1","wr1","wr2"])

inj = pd.read_parquet(os.path.join(HIST, "injuries.parquet"))
outset = set(zip(*inj[inj["report_status"].str.lower().eq("out")][["season","week","gsis_id"]].to_numpy().T))

def absent(col):
    """leader defined & team played & (no usage row this week OR injury-Out)"""
    ids = lead[col]
    no_row = [pd.notna(i) and (s,w,t,i) not in present
              for s,w,t,i in zip(lead.season, lead.week, lead.team, ids)]
    inj_out = [pd.notna(i) and (s,w,i) in outset
               for s,w,i in zip(lead.season, lead.week, ids)]
    return (np.array(no_row) | np.array(inj_out)) & ids.notna().to_numpy()

for c in ["rb1","rb2","te1","wr1","wr2"]:
    lead[f"{c}_absent"] = absent(c)
pw = pw.merge(lead, on=["team","season","week"], how="left")

# extra singles context (full 2019-2025)
gp = pw.groupby(["player_id","season"])
pw["heavy_last"] = gp["carries"].shift(1) >= 22
pw["rec100_last"] = gp["rec_yards"].shift(1) >= 100
pw["spike_last"] = (gp["targets"].shift(1) - pw["tm_targets"]) >= 5
sc = pd.concat([pd.read_parquet(os.path.join(ROOT,"historical_lines.parquet")),
                pd.read_parquet(os.path.join(HIST,"lines_extra.parquet"))], ignore_index=True)
sc = sc[sc["game_type"]=="REG"]
ot = pd.concat([sc[["season","week","home_team","overtime"]].rename(columns={"home_team":"team"}),
                sc[["season","week","away_team","overtime"]].rename(columns={"away_team":"team"})])
ot = ot.sort_values(["team","season","week"])
ot["ot_last"] = ot.groupby("team")["overtime"].shift(1).fillna(0).astype(bool)
pw = pw.merge(ot[["season","week","team","ot_last"]], on=["season","week","team"], how="left")

# ---------------- battery
def over_outcome(d, stat):
    ok = d[f"tm_{stat}"].notna() & d[stat].notna() & (d[f"ng_{stat}"] >= 3)
    return d[ok], (d.loc[ok, stat] > d.loc[ok, f"tm_{stat}"]).astype(int)

R = []
def run(name, mask, stat, roles=None, k=60):
    m = mask.fillna(False) if hasattr(mask, "fillna") else pd.Series(mask, index=pw.index).fillna(False)
    rm = pw["role"].isin(roles) if roles else pd.Series(True, index=pw.index)
    de, ye = over_outcome(pw[m & rm], stat)
    dc, yc = over_outcome(pw[(~m) & rm], stat)
    n, h = len(ye), int(ye.sum())
    if n < 25:
        R.append({"pattern": name, "n": n, "note": "still <25"}); return
    p0 = yc.mean(); a, b = p0*k, (1-p0)*k
    post = np.random.default_rng(0).beta(a+h, b+n-h, 50000)
    R.append({"pattern": name, "n": n, "raw": round(h/n,4), "baseline": round(float(p0),4),
              "lift_pp": round(float((post.mean()-p0)*100),2),
              "P_gt0": round(float((post>p0).mean()),3)})

IS = lambda col: pw["player_id"] == pw[col]
run("TE1 absent -> RB1 anytime TD",        pw["te1_absent"] & IS("rb1"), "td_any", ["RB"])
run("TE1 absent -> RB2 anytime TD",        pw["te1_absent"] & IS("rb2"), "td_any", ["RB"])
run("WR1 absent -> TE1 anytime TD",        pw["wr1_absent"] & IS("te1"), "td_any", ["TE"])
run("WR1 absent -> RB1 anytime TD",        pw["wr1_absent"] & IS("rb1"), "td_any", ["RB"])
run("RB2 absent -> RB1 anytime TD",        pw["rb2_absent"] & IS("rb1"), "td_any", ["RB"])
run("RB1 absent -> RB2 anytime TD",        pw["rb1_absent"] & IS("rb2"), "td_any", ["RB"])
run("RB1 absent -> RB2 carries OVER",      pw["rb1_absent"] & IS("rb2"), "carries", ["RB"])
run("RB1 absent -> non-RB1 RB carries OVER", pw["rb1_absent"] & ~IS("rb1"), "carries", ["RB"])
run("RB2 absent -> RB1 carries OVER",      pw["rb2_absent"] & IS("rb1"), "carries", ["RB"])
run("WR1 absent -> WR2 targets OVER",      pw["wr1_absent"] & IS("wr2"), "targets", ["WR"])
run("WR2 absent -> WR1 rec yards OVER",    pw["wr2_absent"] & IS("wr1"), "rec_yards", ["WR"])
run("TE1 absent -> QB pass yards OVER",    pw["te1_absent"], "pass_yards", ["QB"])
run("WR 100+ yds last wk -> rec yds OVER", pw["rec100_last"], "rec_yards", ["WR"])
run("target spike last wk -> receptions OVER", pw["spike_last"], "receptions", ["WR","TE"])
run("RB 22+ carries last wk -> rush yds OVER", pw["heavy_last"], "rush_yards", ["RB"])
run("OT last wk -> RB rush yds OVER",      pw["ot_last"], "rush_yards", ["RB"])

json.dump(R, open("book/patterns_alldata.json","w"), indent=1)
print(f"{'pattern (ALL DATA 2019-25, usage+inj absence)':52s}{'n':>6s}{'raw':>7s}{'base':>7s}{'lift':>7s}{'P>0':>6s}")
for r in R:
    if "raw" in r:
        print(f"{r['pattern']:52s}{r['n']:6d}{r['raw']:7.4f}{r['baseline']:7.4f}{r['lift_pp']:+6.1f}pp{r['P_gt0']:6.3f}")
    else:
        print(f"{r['pattern']:52s}{r['n']:6d}  (insufficient)")
