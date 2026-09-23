#!/usr/bin/env python3
"""Bounded reference example on REAL pinned data: 2026 Week 3, ATL@GB (Thu).

    python analysis/role_opportunity_example.py --hist <copy of pinned historical/> --out example.json

Targets = ATL/GB QB/RB/WR/TE on the pinned 2026 week-3 roster snapshot who
recorded an opportunity in 2025 or 2026.  Hyperparameters are fit on seasons
< 2026 only.  Kickoff 2026-09-25T00:15Z is supplied so the post-kickoff guard
is exercised.  Descriptive only: no outcome exists yet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.role_opportunity_eval import load, sha  # noqa: E402
from nflvalue import features, role_opportunity as ro  # noqa: E402

AS_OF = dt.datetime(2026, 9, 23, 1, 0, tzinfo=dt.timezone.utc)
KICKOFF = "2026-09-25T00:15:00Z"
TEAMS = ("ATL", "GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pbp, rosters = load(a.hist)
    pw = features.build_player_week(pbp, rosters=rosters)
    games = ro.player_games_from_player_week(pw)
    hyper = ro.fit_hyperparameters(games, before_season=2026)
    r = rosters[(rosters["season"] == 2026) & (rosters["week"] == 3) & rosters["team"].isin(TEAMS)
                & rosters["position"].isin(["QB", "RB", "WR", "TE"])]
    seen = set(games.loc[games["season"] >= 2025, "player_id"])
    r = r[r["player_id"].isin(seen)]
    targets = pd.DataFrame({"player_id": r["player_id"], "team": r["team"], "position": r["position"],
                            "game_id": "2026_03_ATL_GB", "game_start": KICKOFF})
    src = {"source_id": "pinned fullpipe-20260922 replay-run-54d1ec7 historical/ (nflverse pbp + rosters_weekly)",
           "verified": True, "inputs_sha256": {f: sha(os.path.join(a.hist, f)) for f in
                                               ("pbp_2025.parquet", "pbp_2026.parquet", "rosters_weekly.parquet")}}
    out = ro.forecast_opportunity(games, targets, season=2026, week=3, as_of=AS_OF, hyper=hyper, source=src)
    names = dict(zip(r["player_id"], r["full_name"]))
    keep = sorted(out["players"], key=lambda p: -max(p.get("expected_targets") or 0, p.get("expected_carries") or 0,
                                                    p.get("expected_pass_attempts") or 0))[:10]
    slim = []
    for p in keep:
        s = {k: p[k] for k in ("player_id", "team", "position", "regime", "prior_season_team",
                               "expected_targets", "expected_targets_prior_only", "expected_carries",
                               "expected_carries_prior_only", "expected_pass_attempts",
                               "expected_pass_attempts_prior_only")}
        s["name"] = names.get(p["player_id"])
        for q in ro.OPPORTUNITY_QUANTITIES + ro.EFFICIENCY_QUANTITIES:
            if p.get(q):
                s[q] = p[q]
        slim.append(s)
    ids = {p["player_id"] for p in keep}
    doc = {"label": "REAL pinned-data reference example (descriptive; no outcome yet)", "as_of": AS_OF.isoformat(),
           "kickoff": KICKOFF, "hyper_fit_seasons": hyper["fit_seasons"], "meta": out["meta"],
           "teams": out["teams"], "players": slim,
           "factors_sample": [f for f in out["factors"] if f["entity_id"] in ids][:24],
           "n_factor_records_total": len(out["factors"])}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    for s in slim:
        sh = s.get("target_share") or s.get("carry_share") or s.get("pass_share")
        print(s["name"], s["team"], s["position"], s["regime"], "prior", round(sh["prior_estimate"], 3),
              "cur", None if sh["current_estimate"] is None else round(sh["current_estimate"], 3),
              "games", sh["current_games"], "w", round(sh["w_current"], 2), "post", round(sh["posterior"], 3),
              "exp_tgt", s["expected_targets"] and round(s["expected_targets"], 2),
              "exp_car", s["expected_carries"] and round(s["expected_carries"], 2))


if __name__ == "__main__":
    main()
