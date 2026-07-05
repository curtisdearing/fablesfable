"""Fit Phase 6.5 injury constants -- same standard as the absence matrix:
pooled, walk-forward-safe (injury reports precede kickoff), controls for the
offense's own trailing production.

  OPPONENT-SIDE ABSENCE   team-game pass/rush yards ~ trailing + opponent
                          defensive outs by position group (DB / front / LB),
                          2019-2023. Cleared groups become the composite's
                          opp_absence matchup dimension.
  O-LINE OUTS (own)       team-game sack rate + pass yards + QB scrambles ~
                          trailing + own OL outs tiers.

Run:  python3 scripts/fit_absence_opp.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIT_SEASONS = (2019, 2023)
DB_POS = {"CB", "S", "SS", "FS", "DB"}
FRONT_POS = {"DE", "DT", "NT", "EDGE", "DL"}
LB_POS = {"LB", "ILB", "OLB", "MLB"}
OL_POS = {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}


def _ols(X, y, labels):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.sum(resid ** 2) / dof * np.linalg.inv(X.T @ X).diagonal())
    for l, b, s in zip(labels, beta, se):
        print(f"    {l:>16}: {b:+9.3f}  (t={b/s:+5.1f})")
    return beta, se


def main() -> None:
    from nflvalue.context_features import OUT_STATUSES, load_injury_history
    from nflvalue.features import load_pbp

    inj = load_injury_history(list(range(FIT_SEASONS[0], FIT_SEASONS[1] + 1)))
    inj = inj[inj["report_status"].isin(OUT_STATUSES)]
    outs = {}
    for name, poss in (("db", DB_POS), ("front", FRONT_POS), ("lb", LB_POS), ("ol", OL_POS)):
        c = (inj[inj["position"].isin(poss)]
             .groupby(["season", "week", "team"]).size().rename(f"{name}_out"))
        outs[name] = c
    outs_df = pd.concat(outs.values(), axis=1).reset_index()
    outs_df = outs_df.fillna(0.0)

    pbp = load_pbp()
    g = pbp.groupby(["season", "week", "posteam", "defteam", "game_id"])
    tg = g.agg(pass_yds=("passing_yards", lambda s: np.nansum(s.to_numpy())),
               rush_yds=("rushing_yards", lambda s: np.nansum(s.to_numpy())),
               pass_att=("pass_attempt", "sum"),
               scrambles=("qb_scramble", "sum") if "qb_scramble" in pbp.columns else ("pass_attempt", "size"),
               ).reset_index().rename(columns={"posteam": "team"})
    # sacks need the ext column; pull from the P6 base directly
    sk = pd.read_parquet(os.path.join(ROOT, "historical", "historical_pbp.parquet"),
                         columns=["game_id", "posteam", "sack", "qb_dropback", "qb_scramble", "season_type"])
    sk = sk[sk["season_type"] == "REG"]
    sks = (sk.groupby(["game_id", "posteam"])
           .agg(sacks=("sack", "sum"), dropbacks=("qb_dropback", "sum"),
                scr=("qb_scramble", "sum")).reset_index().rename(columns={"posteam": "team"}))
    tg = tg.merge(sks, on=["game_id", "team"], how="left")
    tg = tg.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for c, t in (("pass_yds", "trail_pass"), ("rush_yds", "trail_rush"),
                 ("sacks", "trail_sacks"), ("scr", "trail_scr")):
        tg[t] = tg.groupby("team")[c].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    tg = tg[tg["season"].between(*FIT_SEASONS)]

    # ---- opponent-side: my production vs THEIR defensive outs -------------- #
    opp_outs = outs_df.rename(columns={"team": "defteam"})
    d = tg.merge(opp_outs, on=["season", "week", "defteam"], how="left")
    for c in ("db_out", "front_out", "lb_out", "ol_out"):
        d[c] = d[c].fillna(0.0)
    d["db2"] = (d["db_out"] >= 2).astype(float)
    d["front2"] = (d["front_out"] >= 2).astype(float)
    d["lb2"] = (d["lb_out"] >= 2).astype(float)

    m = d.dropna(subset=["trail_pass"])
    print(f"[opp-absence] n={len(m):,} team-games; opp 2+DB out {m['db2'].mean():.1%}, "
          f"2+front {m['front2'].mean():.1%}, 2+LB {m['lb2'].mean():.1%}")
    print("  pass_yds ~ trail + opp outs (per-out counts + 2+ tier dummies):")
    _ols(np.column_stack([np.ones(len(m)), m["trail_pass"], m["db_out"], m["db2"],
                          m["front_out"], m["lb_out"]]),
         m["pass_yds"].to_numpy(),
         ["const", "trail_pass", "opp_db_out(each)", "opp_db_out>=2", "opp_front_out", "opp_lb_out"])
    m2 = d.dropna(subset=["trail_rush"])
    print("  rush_yds ~ trail + opp outs:")
    _ols(np.column_stack([np.ones(len(m2)), m2["trail_rush"], m2["front_out"], m2["front2"],
                          m2["lb_out"], m2["lb2"], m2["db_out"]]),
         m2["rush_yds"].to_numpy(),
         ["const", "trail_rush", "opp_front_out", "opp_front>=2", "opp_lb_out", "opp_lb>=2", "opp_db_out"])
    print(f"  league means: pass {m['pass_yds'].mean():.1f}, rush {m2['rush_yds'].mean():.1f}")

    # ---- own O-line outs ---------------------------------------------------- #
    own = tg.merge(outs_df[["season", "week", "team", "ol_out"]],
                   on=["season", "week", "team"], how="left")
    own["ol_out"] = own["ol_out"].fillna(0.0)
    own["ol2"] = (own["ol_out"] >= 2).astype(float)
    o = own.dropna(subset=["trail_sacks", "sacks", "dropbacks"])
    o = o[o["dropbacks"] > 10]
    print(f"\n[own OL outs] n={len(o):,} team-games; 1+ OL out {(o['ol_out']>=1).mean():.1%}, "
          f"2+ {(o['ol2']).mean():.1%}")
    print("  sacks_taken ~ trail_sacks + OL outs:")
    _ols(np.column_stack([np.ones(len(o)), o["trail_sacks"], o["ol_out"], o["ol2"]]),
         o["sacks"].to_numpy(), ["const", "trail_sacks", "ol_out(each)", "ol_out>=2"])
    op = own.dropna(subset=["trail_pass"])
    print("  pass_yds ~ trail_pass + OL outs:")
    _ols(np.column_stack([np.ones(len(op)), op["trail_pass"], op["ol_out"], op["ol2"]]),
         op["pass_yds"].to_numpy(), ["const", "trail_pass", "ol_out(each)", "ol_out>=2"])
    osc = own.dropna(subset=["trail_scr", "scr"])
    print("  qb_scrambles ~ trail_scr + OL outs:")
    _ols(np.column_stack([np.ones(len(osc)), osc["trail_scr"], osc["ol_out"], osc["ol2"]]),
         osc["scr"].to_numpy(), ["const", "trail_scr", "ol_out(each)", "ol_out>=2"])
    print(f"  league means: sacks {o['sacks'].mean():.2f}, scrambles {osc['scr'].mean():.2f}")


if __name__ == "__main__":
    main()
