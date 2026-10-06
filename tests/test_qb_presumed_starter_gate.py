"""No confirmed starter: the previous game's roster-QB starter is presumed; other QBs are not bets.

Reproduced defect (2026 Week 4, ATL@NO, Monday 2026-10-05): the Wednesday board carried
pass_attempts cards for T.Tagovailoa (25.2) and C.Rush (19.2) while Michael Penix Jr. -- the
previous game's starter (first pass attempt by a roster QB, 2026_03_ATL_GB) and ESPN depth-chart
QB1 -- started.  No team claim was confirmed before the run, so the confirmed-starter gate
(``starter_not_confirmed``) blocked nothing and the backups' starter-volume forecasts were listed
as research picks.  The gate now presumes the roster-QB proxy starter in that state only, blocks
the other QBs' QB-market rows, and the card builder renders a starter-gated row ``pass`` --
before the synthetic-line ``research`` branch, which is where the ATL@NO cards sat.  No number
changes.
"""
from __future__ import annotations

import copy
import datetime as dt
import json

import pipeline_weekly as pw
from nflvalue import candidates as cand
from nflvalue import pick_cards as pc
from nflvalue import qb_readiness as qr

PENIX, TUA, RUSH = "00-0039917", "00-0036212", "00-0033662"
MARKETS = ("pass_attempts", "passing_yards")
ROWS = [{"player_id": p, "team": "ATL", "market": m} for p in (PENIX, TUA, RUSH) for m in MARKETS] + [
    {"player_id": RUSH, "team": "ATL", "market": "rush_attempts"}]
PRIOR = {"qb_id": PENIX, "name": "Michael Penix Jr.", "game_id": "2026_03_ATL_GB", "season": 2026,
         "week": 3, "basis": qr.PRIOR_BASIS_ROSTER_QB}


def _ctx(state=qr.UNCONFIRMED, prior=PRIOR, qb_id=None, qb_name=None):
    return {"ATL": {"team": "ATL", "state": state, "qb_id": qb_id, "qb_name": qb_name,
                    "prior": copy.deepcopy(prior)}}


def _blocked(gate):
    return {k for k, g in gate.items() if g["blocks_execution"]}


def test_unconfirmed_presumes_previous_roster_qb_starter_and_blocks_the_others():
    gate = cand.confirmed_starter_gate(ROWS, _ctx())
    assert _blocked(gate) == {(TUA, m) for m in MARKETS} | {(RUSH, m) for m in MARKETS}
    assert {gate[(PENIX, m)]["state"] for m in MARKETS} == {"presumed_starter"}
    g = gate[(TUA, "pass_attempts")]
    assert g["state"] == "not_presumed_starter" and g["presumed_qb_id"] == PENIX
    assert "Penix" in g["reason"] and "2026_03_ATL_GB" in g["reason"]
    assert (RUSH, "rush_attempts") not in gate, "non-QB markets are not gated"


def test_unverified_first_passer_proxy_presumes_nothing():
    gate = cand.confirmed_starter_gate(ROWS, _ctx(prior=dict(PRIOR, basis=qr.PRIOR_BASIS_FIRST_PASSER)))
    assert not _blocked(gate)
    assert {g["state"] for g in gate.values()} == {"starter_not_confirmed"}


def test_conflict_and_unknown_still_block_nothing():
    for ctx in (_ctx(state=qr.CONFLICT), _ctx(state=qr.UNKNOWN, prior=None)):
        gate = cand.confirmed_starter_gate(ROWS, ctx)
        assert not _blocked(gate)


def test_a_confirmed_starter_overrides_the_previous_game_proxy():
    gate = cand.confirmed_starter_gate(ROWS, _ctx(state=qr.SOURCED_CHANGED, qb_id=RUSH, qb_name="Cooper Rush"))
    assert _blocked(gate) == {(PENIX, m) for m in MARKETS} | {(TUA, m) for m in MARKETS}
    assert gate[(RUSH, "pass_attempts")]["state"] == "confirmed_starter"


def test_gate_changes_no_number():
    rows = [dict(r, mean=25.25, sd=11.89) for r in ROWS]
    snap, ctx = copy.deepcopy(rows), _ctx()
    before = copy.deepcopy(ctx)
    cand.confirmed_starter_gate(rows, ctx)
    assert rows == snap and ctx == before


def _synthetic_row(pid, player, stamp):
    # the ATL@NO card shape: synthetic line, no quote
    return {"player": player, "player_id": pid, "game_id": "2026_04_ATL_NO", "market": "pass_attempts",
            "side": "under", "line": 25.5, "price": None, "mean": 25.25, "sd": 11.89, "p_side": 0.5653,
            "line_source": "synthetic", "stage_json": json.dumps(stamp)}


def test_atl_no_backup_cards_render_pass_and_the_presumed_starter_stays_research():
    now = dt.datetime(2026, 9, 30, 19, 30, tzinfo=dt.timezone.utc)
    ok = {"availability": {"status": "OK", "eligibility": "eligible", "availability_state": "not_listed"}}
    stamps = {(TUA, "pass_attempts"): copy.deepcopy(ok), (PENIX, "pass_attempts"): copy.deepcopy(ok)}
    before = pc.build_card(_synthetic_row(TUA, "T.Tagovailoa", ok), now)
    assert before["status"] == "research"            # what the Wednesday board showed
    pw._apply_starter_gate(stamps, cand.confirmed_starter_gate(ROWS, _ctx()))
    tua = pc.build_card(_synthetic_row(TUA, "T.Tagovailoa", stamps[(TUA, "pass_attempts")]), now)
    penix = pc.build_card(_synthetic_row(PENIX, "M.Penix", stamps[(PENIX, "pass_attempts")]), now)
    assert tua["status"] == "pass"
    assert any("not_presumed_starter" in r and "Penix" in r for r in tua["status_reasons"])
    assert penix["status"] == "research"
    assert tua["mean"] == before["mean"], "the gate never changes a number"
    a = stamps[(TUA, "pass_attempts")]["availability"]
    assert a["evidence_kind"] == "previous_game_starter_proxy"
    assert a["eligibility_before_starter_gate"] == "eligible" and a["status"] == "OK"
