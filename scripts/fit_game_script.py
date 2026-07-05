"""Fit the Phase 6.3 deterministic game-script constants -- measured, not
guessed, on the frozen 2019-2023 base seasons only:

  OPP_PACE_ELASTICITY  how much a team's play count moves with its OPPONENT's
                       trailing pace (a team's own pace is already in its own
                       rolling volume -- the opponent's influence is the piece
                       the deterministic path lacked). From
                       log(plays) ~ a + b*log(own_roll) + c*log(opp_roll):
                       c is the elasticity.
  PROE_SPLIT_COEF      does trailing neutral PROE predict this week's pass
                       SHARE beyond trailing pass share + the spread tilt?
                       OLS on residualized pass share; shipped only if |t|>2.

Run:  python3 scripts/fit_game_script.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIT_SEASONS = (2019, 2023)


def main() -> None:
    from nflvalue.candidates import build_week_inputs, games_for_week
    from nflvalue.features import _team_week, load_pbp

    inputs = build_week_inputs()
    tw = inputs.tw
    actual = _team_week(load_pbp())
    d = actual.merge(tw, on=["season", "week", "team"], how="inner")
    d = d[d["season"].between(*FIT_SEASONS)].copy()
    d["plays"] = d["team_pass_att"] + d["team_rush_att"]

    # attach the opponent + spread from schedules
    sched = inputs.schedules
    sched = sched[(sched["game_type"] == "REG") & sched["season"].between(*FIT_SEASONS)]
    rows = []
    for gm in sched.itertuples(index=False):
        rows.append({"season": gm.season, "week": gm.week, "team": gm.home_team,
                     "opp": gm.away_team, "margin": gm.spread_line})
        rows.append({"season": gm.season, "week": gm.week, "team": gm.away_team,
                     "opp": gm.home_team,
                     "margin": -gm.spread_line if pd.notna(gm.spread_line) else np.nan})
    d = d.merge(pd.DataFrame(rows), on=["season", "week", "team"], how="inner")
    opp_tw = tw.rename(columns={"team": "opp", "roll_team_plays": "opp_roll_plays",
                                "roll_team_neutral_proe": "opp_roll_proe"})
    d = d.merge(opp_tw[["season", "week", "opp", "opp_roll_plays", "opp_roll_proe"]],
                on=["season", "week", "opp"], how="left")

    m1 = d.dropna(subset=["plays", "roll_team_plays", "opp_roll_plays"])
    m1 = m1[(m1["plays"] > 30) & (m1["roll_team_plays"] > 30) & (m1["opp_roll_plays"] > 30)]
    X = np.column_stack([np.ones(len(m1)), np.log(m1["roll_team_plays"]),
                         np.log(m1["opp_roll_plays"])])
    y = np.log(m1["plays"])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.sum(resid ** 2) / (len(m1) - 3)
                 * np.linalg.inv(X.T @ X).diagonal())
    print(f"[volume] n={len(m1):,}  log(plays) ~ own^{beta[1]:.3f} x opp^{beta[2]:.3f}")
    print(f"         own t={beta[1]/se[1]:.1f}, opp t={beta[2]/se[2]:.1f}")
    print(f"  -> OPP_PACE_ELASTICITY = {beta[2]:.3f}")
    mult = (m1["opp_roll_plays"] / m1["league_plays_prior"]) ** beta[2]
    print(f"     implied multiplier P5-P95: [{mult.quantile(.05):.4f}, {mult.quantile(.95):.4f}]")

    # ---- split model: pass share ~ trailing pass share + margin + PROE ------ #
    d["pass_share"] = d["team_pass_att"] / d["plays"]
    d["trail_pass_share"] = d["roll_team_pass_att"] / (d["roll_team_pass_att"] + d["roll_team_rush_att"])
    m2 = d.dropna(subset=["pass_share", "trail_pass_share", "margin", "roll_team_neutral_proe"])
    X2 = np.column_stack([np.ones(len(m2)), m2["trail_pass_share"],
                          np.clip(-m2["margin"] / 13.0, -1, 1),      # the existing spread tilt shape
                          m2["roll_team_neutral_proe"] / 100.0])     # pass_oe is in % points
    y2 = m2["pass_share"].to_numpy()
    b2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    r2 = y2 - X2 @ b2
    se2 = np.sqrt(np.sum(r2 ** 2) / (len(m2) - 4) * np.linalg.inv(X2.T @ X2).diagonal())
    labels = ["const", "trail_pass_share", "spread_tilt", "neutral_proe"]
    print(f"\n[split] n={len(m2):,}  pass_share OLS:")
    for l, b, s in zip(labels, b2, se2):
        print(f"         {l:>16}: {b:+.4f} (t={b/s:+.1f})")
    print("  -> PROE term ships only if |t| >= 2; coefficient is per pass_oe/100")


if __name__ == "__main__":
    main()
