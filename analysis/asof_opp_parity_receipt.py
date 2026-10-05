#!/usr/bin/env python3
"""Partial-week live/backtest parity receipt for the as-of opponent factor.

    python analysis/asof_opp_parity_receipt.py [--season 2020 --week 8 --fixture] [--out path.json]

For the target week it enumerates the live path three ways -- wholly unplayed,
after the first completed game, fully played -- holding player/team inputs at
the pre-week cut, and reports per yards row: ``opp_source`` (played / asof /
missing), whether an as-of factor is informed (``opp_roll_games >= 1``) or a
justified neutral (0 prior games -> league prior 1.0), and the maximum absolute
difference between each live factor and the as-played full-history table for
the same (defteam, role).  Exit code 1 if any remaining-game row is ``missing``
or any informed factor disagrees with the as-played table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue.candidates import WeekInputs, enumerate_candidates, load_schedules  # noqa: E402
from nflvalue.features import build_opp_pos_def, build_player_week, build_team_week  # noqa: E402

YARDS = ("rushing_yards", "receiving_yards", "passing_yards")
COL = {"QB": "roll_ypa_allowed_factor", "WR": "roll_ypt_allowed_factor",
       "TE": "roll_ypt_allowed_factor", "RB": "roll_ypc_allowed_factor"}


def classify(df: pd.DataFrame) -> dict:
    y = df[df["market"].isin(YARDS)]
    cls = np.where(y["opp_source"] == "played", "played",
                   np.where(y["opp_source"] == "missing", "missing",
                            np.where(y["opp_roll_games"].fillna(0) >= 1, "asof_informed", "asof_neutral_coldstart")))
    out = pd.Series(cls).value_counts().to_dict()
    out["yards_rows"] = int(len(y))
    out["factor_eq_1"] = int(y["components"].map(lambda c: c["opp_factor"] == 1.0).sum())
    return out


def parity(df: pd.DataFrame, full_played: pd.DataFrame) -> dict:
    y = df[df["market"].isin(YARDS) & (df["opp_source"] != "missing")]
    diffs, n_cmp = [], 0
    for r in y.itertuples(index=False):
        key = (r.defteam, r.pos)
        if key in full_played.index:
            diffs.append(abs(r.components["opp_factor"] - float(full_played.loc[key, COL[r.pos]])))
            n_cmp += 1
    return {"rows_compared": n_cmp, "max_abs_diff": float(max(diffs)) if diffs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2020)
    ap.add_argument("--week", type=int, default=8)
    ap.add_argument("--fixture", action="store_true", help="use tests/fixtures 2019-2020 data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.fixture:
        pbp = pd.read_parquet(os.path.join(ROOT, "tests", "fixtures", "pbp_2019_2020.parquet"))
        sched = pd.read_parquet(os.path.join(ROOT, "tests", "fixtures", "schedules_2019_2020.parquet"))
    else:
        from nflvalue import ingest
        pbp = ingest.load_all_pbp()
        sched = load_schedules()
    S, W = a.season, a.week
    prior = pbp[(pbp["season"] < S) | ((pbp["season"] == S) & (pbp["week"] < W))]
    week_pbp = pbp[(pbp["season"] == S) & (pbp["week"] == W)]
    slate = sched[(sched["season"] == S) & (sched["week"] == W)].sort_values(["gameday", "gametime"])
    first = str(slate.iloc[0]["game_id"])
    pw, tw = build_player_week(prior), build_team_week(prior)
    full = build_opp_pos_def(pbp)
    full_played = full[(full["season"] == S) & (full["week"] == W)].set_index(["defteam", "role"])
    scenarios = {
        "wholly_unplayed": build_opp_pos_def(prior),
        "after_first_completed_game": build_opp_pos_def(pd.concat([prior, week_pbp[week_pbp["game_id"] == first]])),
        "fully_played": full,
    }
    rec = {"commit": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
           "season": S, "week": W, "first_completed_game": first, "scenarios": {}}
    ok = True
    for name, opd in scenarios.items():
        mode = "as_played" if name == "fully_played" else "carry_forward"
        inp = WeekInputs(pw=build_player_week(pbp) if mode == "as_played" else pw, opd=opd,
                         tw=build_team_week(pbp) if mode == "as_played" else tw, schedules=sched.copy())
        df = enumerate_candidates(S, W, inputs=inp, roster_mode=mode)
        y = df[df["market"].isin(YARDS)]
        rest = y[y["game_id"] != first] if name == "after_first_completed_game" else y
        block = {"all_rows": classify(y), "remaining_games": classify(rest), "parity_vs_as_played": parity(y, full_played)}
        miss = y[y["opp_source"] == "missing"]
        block["missing_reasons"] = (miss.groupby(["defteam", "pos"]).size().reset_index(name="n").to_dict("records")
                                    if len(miss) else [])
        neutral = y[(y["opp_source"] == "asof") & (y["opp_roll_games"].fillna(0) == 0)]
        block["neutral_coldstart_keys"] = sorted({(r.defteam, r.pos) for r in neutral.itertuples(index=False)})
        rec["scenarios"][name] = block
        if block["remaining_games"].get("missing", 0) or (block["parity_vs_as_played"]["max_abs_diff"] or 0) > 1e-4:
            ok = False
    rec["required_invariant"] = "adding a completed game never removes as-of factors from unplayed games; every present factor equals the as-played table"
    rec["required_invariant_passes"] = ok
    text = json.dumps(rec, indent=2, default=str)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
