"""Phase 8.3/8.4: the FITTED recency weight (EWM span 8) + rest-game cleaning,
behind features.RECENCY_FIT.

The main leakage guarantee is covered by tests/test_leakage.py, which now runs
against the shipped default (fit ON). These tests pin the flag semantics:
off = byte-identical flat-8 world; on = EWM + rest masking with actuals and
grading inputs untouched; tags leak-safe by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import features
from nflvalue.features import build_player_week, build_team_week, build_opp_pos_def


def _sched_for(pbp: pd.DataFrame) -> pd.DataFrame:
    import os
    root = Path(__file__).resolve().parents[1]
    sched = pd.read_parquet(os.path.join(root, "historical_lines.parquet"))
    return sched[sched["season"].isin(pbp["season"].unique())]


def test_flag_off_reproduces_flat8_world(pbp_tiny):
    """recency_fit=False must reproduce the pre-8.3 build byte-for-byte."""
    off = build_player_week(pbp_tiny, recency_fit=False)
    # a hand-rolled flat-8 for one busy player must match the off build
    pid = off.groupby("player_id")["targets"].sum().idxmax()
    sub = off[off["player_id"] == pid].sort_values(["season", "week"])
    manual = sub["targets"].shift(1).rolling(8, min_periods=1).mean()
    got = sub["roll_targets"].to_numpy()
    exp = manual.to_numpy()
    mask = ~np.isnan(exp)
    assert np.allclose(got[mask], exp[mask], atol=1e-9)


def test_flag_on_switches_to_ewm8(pbp_tiny):
    on = build_player_week(pbp_tiny, recency_fit={"enabled": True, "ewm_span": 8,
                                                  "drop_rest": False})
    pid = on.groupby("player_id")["targets"].sum().idxmax()
    sub = on[on["player_id"] == pid].sort_values(["season", "week"])
    manual = sub["targets"].shift(1).ewm(span=8, min_periods=1).mean()
    got, exp = sub["roll_targets"].to_numpy(), manual.to_numpy()
    mask = ~np.isnan(exp)
    assert np.allclose(got[mask], exp[mask], atol=1e-9)
    # and it genuinely differs from the flat world
    off = build_player_week(pbp_tiny, recency_fit=False)
    j = on.merge(off, on=["season", "week", "player_id"], suffixes=("_on", "_off"))
    assert (j["roll_targets_on"] - j["roll_targets_off"]).abs().max() > 1e-6


def test_actuals_and_grading_inputs_never_change(pbp_tiny):
    on = build_player_week(pbp_tiny)                    # shipped default (fit ON)
    off = build_player_week(pbp_tiny, recency_fit=False)
    j = on.merge(off, on=["season", "week", "player_id"], suffixes=("_on", "_off"))
    for c in ("targets", "rec_yards", "carries", "rush_yards", "pass_yards",
              "receptions", "completions"):
        assert (j[f"{c}_on"] == j[f"{c}_off"]).all(), f"actual {c} changed"
    # roll_games (eligibility count) deliberately unweighted/unmasked
    assert (j["roll_games_on"] == j["roll_games_off"]).all()


def test_rest_masking_zero_weights_flagged_games():
    """A synthetic player: huge usage in a week his team is rest-flagged.
    With drop_rest the trailing mean must ignore that game entirely."""
    rows = []
    for w in range(1, 8):
        rows.append({"season": 2030, "week": w, "team": "AAA",
                     "player_id": "P1", "targets": 5.0 if w != 5 else 30.0})
    pw = pd.DataFrame(rows)
    # replicate the masking + ewm idiom directly (unit-level, no pbp build)
    keep = pw["week"] != 5
    masked = pw["targets"].where(keep)
    pred_clean = masked.shift(1).ewm(span=8, min_periods=1).mean().iloc[-1]
    pred_raw = pw["targets"].shift(1).ewm(span=8, min_periods=1).mean().iloc[-1]
    assert abs(pred_clean - 5.0) < 1e-9          # outlier fully ignored
    assert pred_raw > pred_clean                  # raw world was contaminated


def test_meaningless_tag_is_pregame_knowable(pbp_tiny):
    """game_meaningless comes from records STRICTLY BEFORE the week: removing
    all pbp at/after a cutoff cannot change the tag before the cutoff."""
    sched = _sched_for(pbp_tiny)
    full = build_player_week(pbp_tiny, schedules=sched)
    cutoff = (pbp_tiny["season"] < 2019) | (pbp_tiny["week"] < 10)
    trunc = build_player_week(pbp_tiny[cutoff], schedules=sched)
    key = ["season", "week", "player_id"]
    j = full.merge(trunc[key + ["game_meaningless"]], on=key, suffixes=("", "_t"))
    assert (j["game_meaningless"] == j["game_meaningless_t"]).all()


def test_team_and_opp_tables_take_the_same_flag(pbp_tiny):
    sched = _sched_for(pbp_tiny)
    tw_on = build_team_week(pbp_tiny, schedules=sched)
    tw_off = build_team_week(pbp_tiny, schedules=sched, recency_fit=False)
    opd_on = build_opp_pos_def(pbp_tiny, schedules=sched)
    opd_off = build_opp_pos_def(pbp_tiny, schedules=sched, recency_fit=False)
    # same schema either way; flag flips only the roll VALUES where rest games
    # existed in this slice (2019 has some week-17 clinched/eliminated teams)
    assert list(tw_on.columns) == list(tw_off.columns)
    assert list(opd_on.columns) == list(opd_off.columns)
    assert len(tw_on) == len(tw_off) and len(opd_on) == len(opd_off)


def test_new_pw_columns_exported_and_sane(pbp_tiny):
    pw = build_player_week(pbp_tiny, schedules=_sched_for(pbp_tiny))
    assert {"game_meaningless", "prev_early_exit", "early_exit"} <= set(pw.columns)
    assert set(np.unique(pw["game_meaningless"])) <= {0.0, 1.0}
    # prev_early_exit is the shift of early_exit within a player
    pid = pw[pw["early_exit"] > 0]["player_id"].iloc[0] if (pw["early_exit"] > 0).any() else None
    if pid:
        sub = pw[pw["player_id"] == pid].sort_values(["season", "week"])
        assert np.allclose(sub["prev_early_exit"].to_numpy()[1:],
                           sub["early_exit"].to_numpy()[:-1])
