"""Fit TD_BLEND_W (Phase 6.2): the anytime-TD blend between the overall-rate
path (trailing volume x trailing TD rate) and the red-zone path (expected
team RZ opportunities x player's trailing RZ share x league TD-per-RZ-opp).

Protocol -- same standard as the other measured constants:
  * fit ONLY on the frozen 2019-2023 base seasons (2024+ stays out-of-sample
    for the checkpoint evals);
  * every ingredient is walk-forward by construction (roll_*/league_* series
    are shift-1 trailing);
  * scored by log-loss and Brier of P(TD >= 1) = 1 - exp(-lambda) against the
    realized anytime-TD outcome, over the candidate-gated population
    (roll_games >= 3, trailing touches >= 2.5 -- the shortlist's own gate);
  * the opponent RZ-defense factor is ablated in the same pass.

Result is hard-coded as projection.TD_BLEND_W with provenance in
docs/decisions_p6.md. Re-run:  python3 scripts/fit_td_blend.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue.projection import RZ_FACTOR_CLIP  # noqa: E402

FIT_SEASONS = (2019, 2023)   # inclusive; the frozen base
MIN_GAMES, MIN_TOUCHES = 3, 2.5


def main() -> None:
    from nflvalue.candidates import build_week_inputs
    inputs = build_week_inputs()
    pw, tw, opd = inputs.pw, inputs.tw, inputs.opd

    d = pw[(pw["season"].between(*FIT_SEASONS))
           & pw["role"].isin(["RB", "WR", "TE"])
           & (pw["roll_games"].fillna(0) >= MIN_GAMES)].copy()
    d = d[(d["roll_carries"].fillna(0) + d["roll_targets"].fillna(0)) >= MIN_TOUCHES]

    d = d.merge(tw, on=["season", "week", "team"], how="left")
    rzf = opd[opd["role"] == "RB"][["season", "week", "defteam", "roll_rz_td_factor"]]
    d = d.merge(rzf, on=["season", "week", "defteam"], how="left")

    d["lam_base"] = (d["roll_carries"].fillna(0) * d["roll_rush_td_rate"].fillna(0)
                     + d["roll_targets"].fillna(0) * d["roll_rec_td_rate"].fillna(0))
    tgt_part = d["roll_team_rz_tgt"] * d["roll_rz_tgt_share"] * d["league_rz_tgt_td_rate"]
    car_part = d["roll_team_rz_car"] * d["roll_rz_carry_share"] * d["league_rz_car_td_rate"]
    d["lam_rz"] = tgt_part.fillna(0) + car_part.fillna(0)
    # the projection's rule: RZ path only when team volume + league rates exist
    # and at least one share is known
    ok = (d["roll_team_rz_tgt"].notna() & d["roll_team_rz_car"].notna()
          & d["league_rz_tgt_td_rate"].notna() & d["league_rz_car_td_rate"].notna()
          & (d["roll_rz_tgt_share"].notna() | d["roll_rz_carry_share"].notna()))
    d.loc[~ok, "lam_rz"] = np.nan
    d["factor"] = d["roll_rz_td_factor"].clip(*RZ_FACTOR_CLIP).fillna(1.0)
    d["y"] = ((d["rush_tds"] + d["rec_tds"]) >= 1).astype(float)

    print(f"n={len(d):,} player-weeks {FIT_SEASONS}, RZ path available on "
          f"{ok.mean():.1%}, base TD rate {d['y'].mean():.3f}")

    def score(w: float, use_factor: bool) -> tuple:
        lam = np.where(d["lam_rz"].notna(),
                       (1 - w) * d["lam_base"] + w * d["lam_rz"], d["lam_base"])
        if use_factor:
            lam = lam * d["factor"].to_numpy()
        p = np.clip(1.0 - np.exp(-np.maximum(lam, 0.0)), 1e-6, 1 - 1e-6)
        y = d["y"].to_numpy()
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        brier = np.mean((p - y) ** 2)
        return round(float(ll), 5), round(float(brier), 5)

    print("\n w    logloss  brier   (opp RZ factor ON)   logloss  brier (factor OFF)")
    best = (None, 1e9)
    for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
        on, off = score(w, True), score(w, False)
        flag = ""
        if on[0] < best[1]:
            best = (w, on[0])
            flag = "  <-"
        print(f" {w:.1f}  {on[0]:.5f} {on[1]:.5f}              {off[0]:.5f} {off[1]:.5f}{flag}")
    print(f"\nbest w={best[0]} (log-loss {best[1]:.5f}, factor ON)")


if __name__ == "__main__":
    main()
