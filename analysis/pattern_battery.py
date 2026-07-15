#!/usr/bin/env python3
"""Empirical-Bayes pattern battery: creative conditional effects, shrunk.

Player-week outcomes are graded vs the player's own trailing mean (same
convention as the repo's synthetic lines). Every pattern gets a Beta-binomial
posterior against a matched baseline; we report shrunk lift and P(lift>0).
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")
rng = np.random.default_rng(7)

pw = pd.read_parquet("data/analysis_cache/pw_cache.parquet")
pw = pw[pw["season"] >= 2020].copy()

# schedules: base + extra
sc = pd.concat([pd.read_parquet(os.path.join(ROOT, "historical_lines.parquet")),
                pd.read_parquet(os.path.join(HIST, "lines_extra.parquet"))], ignore_index=True)
sc = sc[sc["game_type"] == "REG"].copy()

# long form: one row per team-game
def team_rows(sc, side):
    o = "away" if side == "home" else "home"
    d = sc[["season", "week", "gameday", "weekday", "gametime", "stadium", "roof", "surface",
            "temp", "wind", "div_game", "referee", "spread_line", "total_line",
            f"{side}_team", f"{o}_team", f"{side}_qb_id", f"{side}_rest", f"{side}_score", f"{o}_score"]].copy()
    d.columns = ["season", "week", "gameday", "weekday", "gametime", "stadium", "roof", "surface",
                 "temp", "wind", "div_game", "referee", "spread_line", "total_line",
                 "team", "opp", "qb_id", "rest", "pts_for", "pts_against"]
    d["is_home"] = side == "home"
    d["fav_margin"] = d["spread_line"] * (1 if side == "home" else -1)  # expected margin for THIS team
    return d
tg = pd.concat([team_rows(sc, "home"), team_rows(sc, "away")], ignore_index=True)
tg["gameday"] = pd.to_datetime(tg["gameday"])

# fix team codes pw uses (LA etc. match nflverse) -- assume aligned
pw = pw.merge(tg, on=["season", "week", "team"], how="left")
print("joined:", pw.shape, "missing sched:", pw["stadium"].isna().mean().round(4), flush=True)

# trailing means within player-season (strictly prior, min 2 games)
pw = pw.sort_values(["player_id", "season", "week"])
pw["td_any"] = ((pw["rec_tds"].fillna(0) + pw["rush_tds"].fillna(0)) >= 1).astype(float)
STATS = ["targets", "receptions", "rec_yards", "carries", "rush_yards", "pass_attempts", "pass_yards", "td_any"]
g = pw.groupby(["player_id", "season"])
for s in STATS:
    pw[f"tm_{s}"] = g[s].transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    pw[f"ng_{s}"] = g[s].transform(lambda x: x.shift(1).notna().cumsum())

# modal QB per team (trailing 6 games) & backup-start flag
tq = tg.sort_values(["team", "season", "week"]).copy()
def modal_qb(x):
    out = []
    hist = []
    for q in x:
        m = pd.Series([h for h in hist[-6:] if pd.notna(h)]).mode()
        out.append(m.iloc[0] if len(m) else None)
        if pd.notna(q): hist.append(q)
    return pd.Series(out, index=x.index)
tq["modal_qb"] = tq.groupby("team")["qb_id"].transform(modal_qb)
tq["backup_qb"] = (tq["qb_id"] != tq["modal_qb"]) & tq["modal_qb"].notna()
pw = pw.merge(tq[["season", "week", "team", "modal_qb", "backup_qb"]], on=["season", "week", "team"], how="left")

# injuries: Out list per (season, week, team)
inj = pd.read_parquet(os.path.join(HIST, "injuries.parquet"))
inj_out = inj[inj["report_status"].str.lower().eq("out")][["season", "week", "team", "gsis_id", "position"]]

# roster-of-record: latest trailing usage per (team, season) forward through weeks
rec = pw[["player_id", "season", "week", "team", "role", "roll_carries", "roll_target_share"]].copy()
frames = []
for (t, s), d in rec.groupby(["team", "season"]):
    weeks = sorted(d["week"].unique())
    last = {}
    for w in weeks:
        cur = d[d["week"] == w]
        for r in cur.itertuples():
            last[r.player_id] = (r.role, r.roll_carries or 0, r.roll_target_share or 0)
        rb1 = max(((p, v) for p, v in last.items() if v[0] == "RB"), key=lambda kv: kv[1][1], default=(None,))[0]
        wrs = sorted(((p, v) for p, v in last.items() if v[0] == "WR"), key=lambda kv: -kv[1][2])
        wr1 = wrs[0][0] if wrs else None
        wr2 = wrs[1][0] if len(wrs) > 1 else None
        frames.append((t, s, w, rb1, wr1, wr2))
roster = pd.DataFrame(frames, columns=["team", "season", "week", "rb1_id", "wr1_id", "wr2_id"])
outset = set(zip(inj_out["season"], inj_out["week"], inj_out["gsis_id"]))
roster["rb1_out"] = [ (s, w, r) in outset for s, w, r in zip(roster.season, roster.week, roster.rb1_id) ]
roster["wr2_out"] = [ (s, w, r) in outset for s, w, r in zip(roster.season, roster.week, roster.wr2_id) ]
pw = pw.merge(roster, on=["team", "season", "week"], how="left")

# birthdays
meta = pd.read_parquet(os.path.join(HIST, "players_meta.parquet"))
meta["birth_date"] = pd.to_datetime(meta["birth_date"], errors="coerce")
pw = pw.merge(meta, on="player_id", how="left")
bd = pw["birth_date"]
gd = pw["gameday"]
bday_this = pd.to_datetime(dict(year=gd.dt.year, month=bd.dt.month.fillna(1), day=bd.dt.day.clip(upper=28).fillna(1)), errors="coerce")
delta = (gd - bday_this).dt.days.abs()
pw["birthday_week"] = (delta <= 5) | (delta >= 360)
pw.loc[bd.isna(), "birthday_week"] = False

# player revenge: faced a team they played for in a prior season (2019+)
ros = pd.read_parquet(os.path.join(HIST, "rosters_weekly.parquet"))
pt = ros.groupby(["player_id", "season"])["team"].agg(lambda x: x.mode().iloc[0]).reset_index()
hist_teams = pt.groupby("player_id").apply(lambda d: {s: t for s, t in zip(d["season"], d["team"])}, include_groups=False).to_dict()
def is_revenge(r):
    h = hist_teams.get(r.player_id, {})
    return any(s < r.season and t == r.opp for s, t in h.items())
pw["revenge"] = [is_revenge(r) for r in pw.itertuples()]

# context flags
pw["dome"] = pw["roof"].isin(["dome", "closed"])
pw["windy"] = pw["wind"].fillna(0) >= 15
pw["cold"] = pw["temp"].notna() & (pw["temp"] <= 32) & (~pw["dome"])
pw["turf"] = pw["surface"].fillna("").str.contains("turf|fieldturf|matrixturf|sportturf|astroturf", case=False, regex=True)
gt = pd.to_datetime(pw["gametime"], format="%H:%M", errors="coerce")
pw["primetime"] = pw["weekday"].isin(["Monday", "Thursday"]) | (gt.dt.hour >= 20)
pw["short_week"] = pw["rest"] <= 5
pw["post_bye"] = pw["rest"] >= 12
pw["big_fav"] = pw["fav_margin"] >= 6.5
pw["big_dog"] = pw["fav_margin"] <= -6.5
pw["late_season"] = pw["week"] >= 14
pw["denver_away"] = (pw["stadium"].str.contains("Empower|Mile High", case=False, na=False)) & (~pw["is_home"].fillna(False))
pw["established_duo"] = pw["modal_qb"].notna() & (pw["player_id"] == pw["wr1_id"]) & (pw["roll_target_share"] > 0.22) & (pw["roll_games"] >= 6)

pw.to_parquet("data/analysis_cache/pw_ctx.parquet")
print("context table saved", pw.shape, flush=True)

# ---------------------------------------------------------------- battery
def over_outcome(d, stat):
    ok = d[f"tm_{stat}"].notna() & d[stat].notna() & (d[f"ng_{stat}"] >= 3)
    return d[ok], (d.loc[ok, stat] > d.loc[ok, f"tm_{stat}"]).astype(int)

def eb_test(name, d_exp, y_exp, d_ctl, y_ctl, note="", k=60):
    n, h = len(y_exp), int(y_exp.sum())
    if n < 25: return None
    p0 = y_ctl.mean()
    a, b = p0 * k, (1 - p0) * k
    post = np.random.default_rng(0).beta(a + h, b + n - h, 50000)
    lift = post - p0
    return {"pattern": name, "n": n, "raw": round(h / n, 4), "baseline": round(float(p0), 4),
            "shrunk": round(float(post.mean()), 4), "lift_pp": round(float(lift.mean() * 100), 2),
            "P_gt0": round(float((lift > 0).mean()), 3),
            "RR": round(float(post.mean() / p0), 3), "note": note}

R = []
POS = pw["role"]

def seg(mask, roles=None):
    m = mask.fillna(False)
    if roles: m &= POS.isin(roles)
    return pw[m], pw[~m.reindex(pw.index, fill_value=False)]

def run(name, mask, stat, roles=None, note=""):
    d_exp, d_ctl = seg(mask, roles)
    if roles is not None: d_ctl = d_ctl[d_ctl["role"].isin(roles)]
    de, ye = over_outcome(d_exp, stat)
    dc, yc = over_outcome(d_ctl, stat)
    r = eb_test(name, de, ye, dc, yc, note)
    if r: R.append(r)

# --- user hypotheses
run("birthday_week -> TD (WR/TE/RB)", pw["birthday_week"], "td_any", ["WR", "TE", "RB"], "user: +40% TD?")
run("RB1 out -> QB pass attempts OVER", pw["rb1_out"] & (POS == "QB") & (pw["player_id"] != pw["rb1_id"]), "pass_attempts", ["QB"], "user: 80%?")
run("RB1 out -> other RB carries OVER", pw["rb1_out"] & (pw["player_id"] != pw["rb1_id"]), "carries", ["RB"])
run("RB1 out -> WR/TE targets OVER", pw["rb1_out"], "targets", ["WR", "TE"])
run("WR2 out -> WR1 receptions OVER", pw["wr2_out"] & (pw["player_id"] == pw["wr1_id"]), "receptions", ["WR"], "user chemistry hypo")
run("WR2 out -> WR1 rec yards OVER", pw["wr2_out"] & (pw["player_id"] == pw["wr1_id"]), "rec_yards", ["WR"])
run("WR2 out + established duo -> WR1 rec yds OVER", pw["wr2_out"] & pw["established_duo"], "rec_yards", ["WR"], "duo+wr2out")
run("WR2 out -> TE targets OVER", pw["wr2_out"] & (POS == "TE"), "targets", ["TE"])

# --- revenge / situational
run("revenge game -> rec yards OVER (WR/TE)", pw["revenge"], "rec_yards", ["WR", "TE"])
run("revenge game -> TD (all skill)", pw["revenge"], "td_any", ["WR", "TE", "RB"])
run("revenge game -> rush yards OVER (RB)", pw["revenge"], "rush_yards", ["RB"])

# --- environment
run("dome -> pass yards OVER (QB)", pw["dome"], "pass_yards", ["QB"])
run("dome -> rec yards OVER (WR)", pw["dome"], "rec_yards", ["WR"])
run("wind>=15 -> pass attempts OVER (QB)", pw["windy"], "pass_attempts", ["QB"])
run("wind>=15 -> pass yards OVER (QB)", pw["windy"], "pass_yards", ["QB"])
run("wind>=15 -> carries OVER (RB)", pw["windy"], "carries", ["RB"])
run("cold<=32F -> rec yards OVER (WR)", pw["cold"], "rec_yards", ["WR"])
run("cold<=32F -> rush yards OVER (RB)", pw["cold"], "rush_yards", ["RB"])
run("turf -> rec yards OVER (WR)", pw["turf"], "rec_yards", ["WR"])
run("denver away -> rec yards OVER (WR)", pw["denver_away"], "rec_yards", ["WR"])

# --- schedule spots
run("primetime -> rec yards OVER (WR)", pw["primetime"], "rec_yards", ["WR"])
run("primetime -> rush yards OVER (RB)", pw["primetime"], "rush_yards", ["RB"])
run("primetime -> TD (skill)", pw["primetime"], "td_any", ["WR", "TE", "RB"])
run("short week -> rec yards OVER (WR/TE)", pw["short_week"], "rec_yards", ["WR", "TE"])
run("short week -> rush yards OVER (RB)", pw["short_week"], "rush_yards", ["RB"])
run("post-bye -> rec yards OVER (WR/TE)", pw["post_bye"], "rec_yards", ["WR", "TE"])
run("post-bye -> rush yards OVER (RB)", pw["post_bye"], "rush_yards", ["RB"])
run("late season (wk>=14) -> pass yards OVER (QB)", pw["late_season"], "pass_yards", ["QB"])

# --- game script
run("big favorite -> carries OVER (RB)", pw["big_fav"], "carries", ["RB"])
run("big underdog -> pass attempts OVER (QB)", pw["big_dog"], "pass_attempts", ["QB"])
run("big underdog -> targets OVER (WR)", pw["big_dog"], "targets", ["WR"])

# --- QB change
run("backup QB -> TE targets OVER", pw["backup_qb"] & (POS == "TE"), "targets", ["TE"])
run("backup QB -> WR1 rec yards OVER", pw["backup_qb"] & (pw["player_id"] == pw["wr1_id"]), "rec_yards", ["WR"])
run("backup QB -> team WR rec yards OVER (all WR)", pw["backup_qb"], "rec_yards", ["WR"])
run("backup QB start -> QB rush yards OVER", pw["backup_qb"] & (POS == "QB"), "rush_yards", ["QB"])

# --- rematch familiarity
first_meet = {}
rematch = []
for r in pw[["season", "week", "team", "opp"]].drop_duplicates().itertuples():
    key = (r.season, tuple(sorted((r.team, r.opp))))
    rematch.append(((key in first_meet) and first_meet[key] < r.week, r.season, r.week, r.team, r.opp))
    if key not in first_meet: first_meet[key] = r.week
rm = pd.DataFrame(rematch, columns=["is_rematch", "season", "week", "team", "opp"])
pw = pw.merge(rm, on=["season", "week", "team", "opp"], how="left")
run("division rematch -> WR1 rec yards OVER", pw["is_rematch"] & (pw["player_id"] == pw["wr1_id"]), "rec_yards", ["WR"], "familiarity")
run("division rematch -> QB pass yards OVER", pw["is_rematch"], "pass_yards", ["QB"])

R = [r for r in R if r]
R.sort(key=lambda r: -abs(r["lift_pp"]) * (r["P_gt0"] if r["lift_pp"] > 0 else 1 - r["P_gt0"]))
json.dump(R, open("book/patterns.json", "w"), indent=1)
print(f"\n{'pattern':52s} {'n':>5s} {'raw':>6s} {'base':>6s} {'lift':>6s} {'P>0':>5s}")
for r in R:
    print(f"{r['pattern']:52s} {r['n']:5d} {r['raw']:.4f} {r['baseline']:.4f} {r['lift_pp']:+5.1f}pp {r['P_gt0']:.3f}")

# --- referee pace (exploratory, game level)
gg = tg.dropna(subset=["referee", "total_line"]).copy()
gg["tot_over"] = (gg["pts_for"] + gg["pts_against"]) > gg["total_line"]
gg = gg[gg["season"] >= 2020].drop_duplicates(subset=["season", "week", "team"])
gme = gg.groupby(["season", "week", "stadium", "referee"]).first().reset_index()
refs = gme.groupby("referee").agg(n=("tot_over", "size"), over=("tot_over", "mean"))
refs = refs[refs["n"] >= 40]
p0 = gme["tot_over"].mean()
k = 80
refs["shrunk"] = (refs["over"] * refs["n"] + p0 * k) / (refs["n"] + k)
refs = refs.sort_values("shrunk")
print("\nreferee totals-over rates (shrunk, n>=40): base", round(p0, 3))
print(refs.round(3).to_string())
json.dump({"base": float(p0), "refs": refs.reset_index().to_dict("records")}, open("book/refs.json", "w"), indent=1)
