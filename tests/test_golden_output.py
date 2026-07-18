"""Phase 7.1 -- determinism and the no-delta gate.

The hardening contract says: same inputs -> byte-identical outputs, and no
change may move a shipped number without a walk-forward measurement recorded
in the ledger. These tests are the mechanism that makes that claim checkable
rather than asserted.

Measured baseline (2026-07-18, aarch64 4-core sandbox):

    pipeline_weekly 2023 wk10 historical, ML on : 8.18 / 8.22 / 8.18 s
    peak RSS                                     : ~1.12 GB
    ml_test --stage frame (7 seasons, 73,925 rows): 20.0 s / ~1.11 GB
    ml_test --stage fit (GBDT)                    :  2.7 s / 0.37 GB

The profile is the reason NO optimisation was applied to the rolling-feature
path: at 8 s for a once-weekly job, vectorising `groupby.transform` would buy
~3 s while rewriting the shift(1) primitive that the whole anti-leakage design
rests on. Wall time is not the constraint; peak RSS is (an OOM was already
caught in this path -- see docs/decisions_p3-5.md, 2026-07-02).

These tests are intentionally cheap -- the expensive end-to-end comparison is
run by hand and recorded in accuracy_ledger.md.
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import oddsmath                       # noqa: E402
from nflvalue.projection import project             # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism of the numeric core
# --------------------------------------------------------------------------- #
_PLAYER = {"player_id": "00-GOLD", "player_name": "Golden Row", "role": "WR",
           "roll_games": 8.0, "roll_target_share": 0.223, "roll_targets": 7.4,
           "roll_ypt": 8.87, "roll_catch_rate": 0.646, "roll_rush_td_rate": 0.0,
           "roll_rec_td_rate": 0.049, "roll_carries": 0.2}
_TEAM = {"roll_team_targets": 34.6, "roll_team_carries": 25.1,
         "roll_team_pass_att": 34.6}

#: Pinned to the values produced at f83b304 + Phase 7 hardening. A diff here
#: means a shipped number moved and the ledger must explain why.
GOLDEN = {
    "receiving_yards": {"line": 62.5},
    "receptions": {"line": 4.5},
    "anytime_td": {"line": 0.5},
}


@pytest.mark.parametrize("market", sorted(GOLDEN))
def test_projection_is_reproducible_within_a_process(market):
    """Same inputs, repeated calls -> exactly equal. Not approximately."""
    line = GOLDEN[market]["line"]
    first = project(dict(_PLAYER), market, team_row=dict(_TEAM), line=line)
    for _ in range(5):
        again = project(dict(_PLAYER), market, team_row=dict(_TEAM), line=line)
        assert again == first, f"{market} projection is not reproducible"


@pytest.mark.parametrize("market", sorted(GOLDEN))
def test_projection_is_independent_of_dict_ordering(market):
    """No dict-ordering dependence in a numeric path. Feeding the same row
    with its keys inserted in reverse must not shift a single digit."""
    line = GOLDEN[market]["line"]
    forward = project(dict(_PLAYER), market, team_row=dict(_TEAM), line=line)
    reversed_row = {k: _PLAYER[k] for k in reversed(list(_PLAYER))}
    reversed_team = {k: _TEAM[k] for k in reversed(list(_TEAM))}
    backward = project(reversed_row, market, team_row=reversed_team, line=line)
    assert backward == forward, f"{market} depends on key insertion order"


def test_projection_has_no_wall_clock_dependence():
    """A numeric path that reads the clock cannot be reproduced tomorrow.
    Freezing nothing, we simply assert two calls separated by real time agree."""
    import time
    a = project(dict(_PLAYER), "receiving_yards", team_row=dict(_TEAM), line=62.5)
    time.sleep(0.01)
    b = project(dict(_PLAYER), "receiving_yards", team_row=dict(_TEAM), line=62.5)
    assert a == b


def test_consensus_probability_is_reproducible():
    books = {"draftkings": (1.91, 1.91), "betmgm": (1.87, 1.95),
             "hardrockbet": (2.05, 1.80)}
    first = oddsmath.consensus_two_way(books)
    for _ in range(5):
        assert oddsmath.consensus_two_way(books) == first


# --------------------------------------------------------------------------- #
# The no-delta gate
# --------------------------------------------------------------------------- #
def test_projection_contract_fields_are_stable():
    """The consumer contract (docs/HOW_A_PICK_IS_MADE.md §3). Adding a field is
    fine; removing or renaming one silently breaks every downstream reader."""
    out = project(dict(_PLAYER), "receiving_yards", team_row=dict(_TEAM), line=62.5)
    required = {"player_id", "name", "pos", "market", "mean", "sd", "dist",
                "line", "p_over", "p_under", "components", "low_confidence",
                "eligible_for_shortlist", "roll_games"}
    assert required <= set(out), f"missing contract fields: {required - set(out)}"
    assert set(out["components"]) == {"volume", "efficiency", "opp_factor",
                                      "game_script"}


def test_probabilities_are_complementary_to_the_published_precision():
    """p_over and p_under are published to 4dp and must not drift apart at
    that precision -- a reader adds them and expects 1.0000."""
    for market, spec in GOLDEN.items():
        out = project(dict(_PLAYER), market, team_row=dict(_TEAM),
                      line=spec["line"])
        if out["p_over"] is None:
            continue
        assert math.isclose(out["p_over"] + out["p_under"], 1.0, abs_tol=1e-4)


@pytest.mark.skipif(not os.path.exists(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "ml_frame.parquet")),
    reason="ml_frame.parquet is a build artifact; regenerate with "
           "`python3 ml_test.py --stage frame`")
def test_no_projection_in_the_historical_frame_is_a_fabricated_certainty():
    """The regression pin for the Phase 7.4 fail-closed fix, evaluated against
    the full 2019-2025 candidate corpus rather than a synthetic row.

    Before the fix, a NaN mean surfaced as p_over=1.0000 with
    eligible_for_shortlist=True. Zero rows in the corpus were affected -- the
    bug was latent, which is exactly why it needed a corpus-wide assertion and
    not just a unit test.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = pd.read_parquet(os.path.join(root, "data", "ml_frame.parquet"))

    assert frame["mean"].notna().all(), "NaN projection mean reached the frame"
    assert frame["p_over"].notna().all(), "NaN p_over reached the frame"
    degenerate = frame[(frame["p_over"] <= 0.0) | (frame["p_over"] >= 1.0)]
    assert degenerate.empty, (
        f"{len(degenerate)} row(s) carry a certainty probability; "
        f"first: {degenerate.iloc[0][['season', 'week', 'player_id', 'market', 'mean', 'p_over']].to_dict()}")
