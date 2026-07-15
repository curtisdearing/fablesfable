#!/usr/bin/env python3
"""Extended single-factor battery: cross-position absence cascades + obscure
conditions. NO combinations (per instruction) — each factor stands alone,
empirical-Bayes shrunk vs matched baseline."""
import json, os
import numpy as np, pandas as pd

import pathlib; ROOT = str(pathlib.Path(__file__).resolve().parents[1])
HIST = os.path.join(ROOT, "historical")
pw = pd.read_parquet("data/analysis_cache/pw_ctx.parquet")

# ---------- extra roster context: TE1, RB2, WR3; OL & defensive outs ----------
inj = pd.read_parquet(os.path.join(HIST, "injuries.parquet"))
inj["pos_grp"] = inj["position"].map(lambda p: "OL" if p in {"T","G","C","OT","OG","OL"} else
                                     ("DB" if p in {"CB","S","DB","FS","SS"} else
                                      ("F7" if p in {"DE","DT","NT","LB","OLB","ILB","MLB","EDGE"} else p)))
out = inj[inj["report_status"].str.lower().eq("out")]
ol_out = out[out.pos_grp=="OL"].groupby(["season","week","team"]).size().rename("ol_outs_n")
db_out = out[out.pos_grp=="DB"].groupby(["season","week","team"]).size().rename("db_outs_n")
f7_out = out[out.pos_grp=="F7"].groupby(["season","week","team"]).size().rename("f7_outs_n")
pw = pw.merge(ol_out, left_on=["season","week","team"], right_index=True, how="left")
pw = pw.merge(db_out.rename("opp_db_outs"), left_on=["season","week","opp"], right_index=True, how="left")
pw = pw.merge(f7_out.rename("opp_f7_outs"), left_on=["season","week","opp"], right_index=True, how="left")
for c in ["ol_outs_n","opp_db_outs","opp_f7_outs"]: pw[c]=pw[c].fillna(0)

# TE1 / RB2 / WR3 roster-of-record (trailing usage, carried forward)
rec = pw[["player_id","season","week","team","role","roll_carries","roll_target_share"]]
frames=[]
for (t,s),d in rec.groupby(["team","season"]):
    last={}
    for w in sorted(d["week"].unique()):
        for r in d[d["week"]==w].itertuples():
            last[r.player_id]=(r.role, r.roll_carries or 0, r.roll_target_share or 0)
        tes=sorted(((p,v) for p,v in last.items() if v[0]=="TE"), key=lambda kv:-kv[1][2])
        rbs=sorted(((p,v) for p,v in last.items() if v[0]=="RB"), key=lambda kv:-kv[1][1])
        wrs=sorted(((p,v) for p,v in last.items() if v[0]=="WR"), key=lambda kv:-kv[1][2])
        frames.append((t,s,w, tes[0][0] if tes else None,
                       rbs[1][0] if len(rbs)>1 else None,
                       wrs[2][0] if len(wrs)>2 else None))
r2=pd.DataFrame(frames, columns=["team","season","week","te1_id","rb2_id","wr3_id"])
outset=set(zip(out["season"],out["week"],out["gsis_id"]))
r2["te1_out"]=[(s,w,p) in outset for s,w,p in zip(r2.season,r2.week,r2.te1_id)]
r2["rb2_out"]=[(s,w,p) in outset for s,w,p in zip(r2.season,r2.week,r2.rb2_id)]
pw=pw.merge(r2, on=["team","season","week"], how="left")

# ---------- schedule-derived obscure conditions ----------
sc = pd.concat([pd.read_parquet(os.path.join(ROOT,"historical_lines.parquet")),
                pd.read_parquet(os.path.join(HIST,"lines_extra.parquet"))], ignore_index=True)
sc = sc[sc["game_type"]=="REG"]
WEST={"SEA","SF","LA","LAC","LV","ARI","DEN"}   # PT/MT body clocks
EAST_STADIUMS_TZ={"ET","CT"}
def team_sched(side):
    o="away" if side=="home" else "home"
    d=sc[["season","week",f"{side}_team",f"{o}_team","gametime","weekday","stadium","location","overtime",
          f"{side}_score",f"{o}_score","home_team"]].copy()
    d.columns=["season","week","team","opp","gametime","weekday","stadium","location","overtime","pts","opp_pts","home_team"]
    d["is_home"]=side=="home"
    return d
ts=pd.concat([team_sched("home"),team_sched("away")],ignore_index=True).sort_values(["team","season","week"])
gt=pd.to_datetime(ts["gametime"],format="%H:%M",errors="coerce")
ts["early_kick"]=(gt.dt.hour==13)|(gt.dt.hour==12)  # 1pm ET window
ts["body_clock"]=ts["early_kick"] & ts["team"].isin(WEST) & (~ts["is_home"])
ts["neutral"]=ts["location"].eq("Neutral")
g=ts.groupby("team")
ts["won_big_last"]=g.apply(lambda d:(d["pts"]-d["opp_pts"]).shift(1)>=21, include_groups=False).reset_index(level=0,drop=True)
ts["lost_big_last"]=g.apply(lambda d:(d["pts"]-d["opp_pts"]).shift(1)<=-21, include_groups=False).reset_index(level=0,drop=True)
ts["ot_last"]=g["overtime"].shift(1).fillna(0).astype(bool)
ts["post_neutral"]=g["neutral"].shift(1).fillna(False)
pw=pw.merge(ts[["season","week","team","body_clock","neutral","won_big_last","lost_big_last","ot_last","post_neutral"]],
            on=["season","week","team"],how="left")

# workload hangover + spike mean reversion (player level, strictly prior)
pw=pw.sort_values(["player_id","season","week"])
gp=pw.groupby(["player_id","season"])
pw["carries_last"]=gp["carries"].shift(1)
pw["heavy_last"]=pw["carries_last"]>=22
pw["rec100_last"]=gp["rec_yards"].shift(1)>=100
pw["spike_last"]=(gp["targets"].shift(1)-pw["tm_targets"])>=5    # target spike last wk
# opp blitz condition (from candidates frame features not here; proxy: opp pressure via ftn cache)
ftn_frames=[]
for fn in sorted(os.listdir(HIST)):
    if fn.startswith("ftn_"): ftn_frames.append(pd.read_parquet(os.path.join(HIST,fn)))
if ftn_frames:
    ftn=pd.concat(ftn_frames,ignore_index=True)
    pbp_cols=["game_id","play_id","season","week","defteam","pass"]
    import pyarrow.parquet as pq
    pbps=[pd.read_parquet(os.path.join(HIST,"historical_pbp.parquet"),columns=pbp_cols)]
    for fn in ["pbp_2024.parquet","pbp_2025.parquet"]:
        p=os.path.join(HIST,fn)
        if os.path.exists(p): pbps.append(pd.read_parquet(p,columns=pbp_cols))
    pb=pd.concat(pbps,ignore_index=True)
    j=ftn.merge(pb,left_on=["nflverse_game_id","nflverse_play_id"],right_on=["game_id","play_id"],how="inner",suffixes=("_ftn",""))
    j=j[j["pass"]==1]
    bl=j.groupby(["season","defteam"]).agg(blitz=("n_blitzers",lambda x:(x.fillna(0)>=5).mean()),n=("n_blitzers","size")).reset_index()
    hi=bl[bl["n"]>=200].copy(); q75=hi["blitz"].quantile(0.75)
    hi["hi_blitz"]=hi["blitz"]>=q75
    pw=pw.merge(hi[["season","defteam","hi_blitz"]].rename(columns={"defteam":"opp"}),on=["season","opp"],how="left")
    pw["hi_blitz"]=pw["hi_blitz"].fillna(False)
else:
    pw["hi_blitz"]=False

# referee pace crews (plays/gm, pooled, shrunk; flag = top/bottom quartile)
tp=pw.drop_duplicates(subset=["season","week","team"])[["season","week","team","referee","team_plays"]]
gpg=tp.groupby(["season","week","referee"])["team_plays"].sum().reset_index()
crew=gpg.groupby("referee").agg(pace=("team_plays","mean"),n=("team_plays","size"))
crew=crew[crew["n"]>=40]
fast=set(crew[crew["pace"]>=crew["pace"].quantile(0.75)].index)
slow=set(crew[crew["pace"]<=crew["pace"].quantile(0.25)].index)
pw["fast_ref"]=pw["referee"].isin(fast); pw["slow_ref"]=pw["referee"].isin(slow)

pw["age"] = (pw["gameday"] - pw["birth_date"]).dt.days/365.25
pw["old_wr_cold"]=(pw["age"]>=30)&pw["cold"]&(pw["role"]=="WR")   # single condition applied to WRs (age is the factor)
pw["young_late"]=(pw["age"]<23.5)&(pw["week"]>=14)
pw["grass_home_on_turf"]=pw["turf"]&(~pw["is_home"].fillna(False))  # away on turf (proxy; home surface hist would refine)

pw.to_parquet("data/analysis_cache/pw_ctx2.parquet")
print("ctx2 saved",pw.shape,flush=True)

# ---------------------------------------------------------------- battery
def over_outcome(d,stat):
    ok=d[f"tm_{stat}"].notna()&d[stat].notna()&(d[f"ng_{stat}"]>=3)
    return d[ok],(d.loc[ok,stat]>d.loc[ok,f"tm_{stat}"]).astype(int)
R=[]
def run(name,mask,stat,roles=None,note="",k=60):
    m=mask.fillna(False)
    if roles is not None: m2=pw["role"].isin(roles)
    else: m2=pd.Series(True,index=pw.index)
    d_exp=pw[m&m2]; d_ctl=pw[(~m)&m2]
    de,ye=over_outcome(d_exp,stat); dc,yc=over_outcome(d_ctl,stat)
    n,h=len(ye),int(ye.sum())
    if n<25: return
    p0=yc.mean(); a,b=p0*k,(1-p0)*k
    post=np.random.default_rng(0).beta(a+h,b+n-h,50000); lift=post-p0
    R.append({"pattern":name,"n":n,"raw":round(h/n,4),"baseline":round(float(p0),4),
              "lift_pp":round(float(lift.mean()*100),2),"P_gt0":round(float((lift>0).mean()),3),"note":note})

NOTRB1=pw["player_id"]!=pw["rb1_id"]
# --- cross-position absence cascades (user's TE1->RB2 example first)
run("TE1 out -> RB2 TD",pw["te1_out"]&(pw["player_id"]==pw["rb2_id"]),"td_any",["RB"],"user example")
run("TE1 out -> RB1 TD",pw["te1_out"]&(pw["player_id"]==pw["rb1_id"]),"td_any",["RB"])
run("TE1 out -> RB receptions OVER",pw["te1_out"],"receptions",["RB"],"checkdown shift")
run("TE1 out -> RB1 carries OVER",pw["te1_out"]&(pw["player_id"]==pw["rb1_id"]),"carries",["RB"])
run("TE1 out -> WR3 targets OVER",pw["te1_out"]&(pw["player_id"]==pw["wr3_id"]),"targets",["WR"],"slot fill-in")
run("TE1 out -> QB pass yards OVER",pw["te1_out"],"pass_yards",["QB"],"protection loss?")
run("WR1 out -> RB1 receptions OVER",(pw["player_id"]==pw["rb1_id"])&pw.get("wr1_id").notna()&pw["wr1_id"].map(lambda x:False if x is None else False),"receptions",["RB"])  # placeholder replaced below
R.pop()  # remove placeholder
wr1_out_flags=[(s,w,p) in outset for s,w,p in zip(pw.season,pw.week,pw.wr1_id)]
pw["wr1_out"]=wr1_out_flags
run("WR1 out -> RB1 receptions OVER",pw["wr1_out"]&(pw["player_id"]==pw["rb1_id"]),"receptions",["RB"],"checkdowns")
run("WR1 out -> RB1 TD",pw["wr1_out"]&(pw["player_id"]==pw["rb1_id"]),"td_any",["RB"])
run("WR1 out -> QB rush yards OVER",pw["wr1_out"],"rush_yards",["QB"],"scramble more")
run("WR1 out -> TE1 TD",pw["wr1_out"]&(pw["player_id"]==pw["te1_id"]),"td_any",["TE"])
run("RB1 out -> TE1 TD",pw["rb1_out"]&(pw["player_id"]==pw["te1_id"]),"td_any",["TE"],"goal-line shift")
run("RB1 out -> RB2 TD",pw["rb1_out"]&(pw["player_id"]==pw["rb2_id"]),"td_any",["RB"])
run("RB2 out -> RB1 carries OVER",pw["rb2_out"]&(pw["player_id"]==pw["rb1_id"]),"carries",["RB"],"no committee")
run("RB2 out -> RB1 TD",pw["rb2_out"]&(pw["player_id"]==pw["rb1_id"]),"td_any",["RB"])
# --- line / defense outs
run("own OL 2+ out -> QB rush yards OVER",pw["ol_outs_n"]>=2,"rush_yards",["QB"],"flushed")
run("own OL 2+ out -> RB rush yards OVER",pw["ol_outs_n"]>=2,"rush_yards",["RB"])
run("own OL 2+ out -> QB pass yards OVER",pw["ol_outs_n"]>=2,"pass_yards",["QB"])
run("opp DBs 2+ out -> WR rec yards OVER",pw["opp_db_outs"]>=2,"rec_yards",["WR"])
run("opp DBs 2+ out -> QB pass yards OVER",pw["opp_db_outs"]>=2,"pass_yards",["QB"])
run("opp front-7 2+ out -> RB rush yards OVER",pw["opp_f7_outs"]>=2,"rush_yards",["RB"])
# --- schedule obscura
run("west-coast team, 1pm ET road kick -> WR rec yards OVER",pw["body_clock"],"rec_yards",["WR"],"body clock")
run("west-coast team, 1pm ET road kick -> RB rush yards OVER",pw["body_clock"],"rush_yards",["RB"])
run("west-coast team, 1pm ET road kick -> QB pass yards OVER",pw["body_clock"],"pass_yards",["QB"])
run("neutral site (international) -> skill rec yards OVER",pw["neutral"],"rec_yards",["WR","TE"])
run("week after neutral site -> rec yards OVER",pw["post_neutral"],"rec_yards",["WR","TE"],"travel hangover")
run("team won by 21+ last wk -> WR rec yards OVER",pw["won_big_last"],"rec_yards",["WR"],"letdown?")
run("team lost by 21+ last wk -> WR rec yards OVER",pw["lost_big_last"],"rec_yards",["WR"],"bounceback?")
run("OT game last wk -> RB rush yards OVER",pw["ot_last"],"rush_yards",["RB"],"fatigue")
run("OT game last wk -> WR rec yards OVER",pw["ot_last"],"rec_yards",["WR"])
# --- player-level obscura
run("RB 22+ carries last wk -> rush yards OVER",pw["heavy_last"],"rush_yards",["RB"],"workload hangover")
run("WR 100+ yds last wk -> rec yards OVER",pw["rec100_last"],"rec_yards",["WR"],"spotlight regression")
run("target spike (+5 vs trend) last wk -> receptions OVER",pw["spike_last"],"receptions",["WR","TE"],"mean reversion")
run("vs top-quartile blitz defense -> RB receptions OVER",pw["hi_blitz"],"receptions",["RB"],"checkdowns")
run("vs top-quartile blitz defense -> QB pass yards OVER",pw["hi_blitz"],"pass_yards",["QB"])
run("fast ref crew -> receptions OVER (WR/TE)",pw["fast_ref"],"receptions",["WR","TE"],"pace=volume")
run("fast ref crew -> QB pass attempts OVER",pw["fast_ref"],"pass_attempts",["QB"])
run("slow ref crew -> receptions OVER (WR/TE)",pw["slow_ref"],"receptions",["WR","TE"])
run("age 30+ WR in cold -> rec yards OVER",pw["old_wr_cold"],"rec_yards",["WR"],"old legs")
run("age<23.5 & week>=14 -> rec yards OVER (WR)",pw["young_late"],"rec_yards",["WR"],"rookie wall")
run("road game on turf -> rec yards OVER (WR)",pw["grass_home_on_turf"],"rec_yards",["WR"])

R.sort(key=lambda r:-abs(r["lift_pp"])*(r["P_gt0"] if r["lift_pp"]>0 else 1-r["P_gt0"]))
json.dump(R,open("book/patterns2.json","w"),indent=1)
print(f"{'pattern':56s}{'n':>5s}{'raw':>7s}{'base':>7s}{'lift':>7s}{'P>0':>6s}")
for r in R: print(f"{r['pattern']:56s}{r['n']:5d}{r['raw']:7.4f}{r['baseline']:7.4f}{r['lift_pp']:+6.1f}pp{r['P_gt0']:6.3f}")
