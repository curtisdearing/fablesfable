"""QB readiness: sourced + clocked + uniquely identified, or explicitly not.
Never writes the primary numeric inputs (qb_continuity / backup_qb_adj)."""
import os

import numpy as np
import pandas as pd
import pytest

from nflvalue import candidates as candmod
from nflvalue import qb_readiness as qr
from nflvalue.advanced_features import build_qb_continuity

AS_OF = "2026-09-23T02:00:00Z"
KO = "2026-09-25T00:15:00Z"


def _pbp():
    rows = []
    for wk, atl_qb in ((1, "Q-PENIX"), (2, "Q-TUA")):
        for i in range(30):
            rows.append({"season": 2026, "week": wk, "game_id": f"g{wk}ATL", "play_id": i,
                         "posteam": "ATL", "pass_attempt": 1, "passer_player_id": atl_qb})
            rows.append({"season": 2026, "week": wk, "game_id": f"g{wk}GB", "play_id": i,
                         "posteam": "GB", "pass_attempt": 1, "passer_player_id": "Q-LOVE"})
    return pd.DataFrame(rows)


ROSTER = pd.DataFrame([
    {"player_id": "Q-PENIX", "full_name": "Michael Penix Jr.", "team": "ATL", "position": "QB"},
    {"player_id": "Q-TUA", "full_name": "Tua Tagovailoa", "team": "ATL", "position": "QB"},
    {"player_id": "Q-LOVE", "full_name": "Jordan Love", "team": "GB", "position": "QB"},
])


def _claim(team, name, pub=AS_OF.replace("02:00", "01:00"), **kw):
    return qr.link_claim_identity({"team": team, "claim_value": name, "claim_kind": "confirmed",
                                   "source": "https://team.example/news", "published_at": pub, **kw}, ROSTER)


def test_prior_realized_starter_is_strictly_before_week():
    pr = qr.prior_realized_starters(_pbp(), 2026, 3)
    assert pr["ATL"]["qb_id"] == "Q-TUA" and pr["ATL"]["week"] == 2
    assert qr.prior_realized_starters(_pbp(), 2026, 2)["ATL"]["qb_id"] == "Q-PENIX"


def test_verified_changed_starter():
    pr = qr.prior_realized_starters(_pbp(), 2026, 3)
    rec = qr.resolve_team("ATL", pr["ATL"], [_claim("ATL", "Michael Penix Jr.")], AS_OF, KO)
    assert (rec["state"], rec["qb_id"]) == (qr.VERIFIED_CHANGED, "Q-PENIX")
    assert rec["numeric_blocked"]["backup_qb_adj"] == [qr.BLOCK_SOURCE, qr.BLOCK_SEMANTICS]


def test_verified_same_starter():
    pr = qr.prior_realized_starters(_pbp(), 2026, 3)
    rec = qr.resolve_team("GB", pr["GB"], [_claim("GB", "Jordan Love")], AS_OF, KO)
    assert (rec["state"], rec["qb_id"]) == (qr.VERIFIED_SAME, "Q-LOVE")


def test_unconfirmed_starter_keeps_prior_as_history_only():
    pr = qr.prior_realized_starters(_pbp(), 2026, 3)
    rec = qr.resolve_team("GB", pr["GB"], [], AS_OF, KO)
    assert rec["state"] == qr.UNCONFIRMED and rec["qb_id"] is None and rec["prior"]["qb_id"] == "Q-LOVE"
    assert qr.resolve_team("NYJ", None, [], AS_OF, KO)["state"] == qr.UNKNOWN


def test_conflicting_claims_are_conflict_not_a_pick():
    pr = qr.prior_realized_starters(_pbp(), 2026, 3)
    rec = qr.resolve_team("ATL", pr["ATL"], [_claim("ATL", "Michael Penix Jr."),
                                             _claim("ATL", "Tua Tagovailoa")], AS_OF, KO)
    assert rec["state"] == qr.CONFLICT and rec["qb_id"] is None


def test_ambiguous_or_unknown_identity_is_rejected():
    dup = pd.concat([ROSTER, pd.DataFrame([{"player_id": "Q-PENIX2", "full_name": "Michael Penix",
                                             "team": "ATL", "position": "QB"}])])
    c = qr.link_claim_identity({"team": "ATL", "claim_value": "Michael Penix Jr.", "claim_kind": "confirmed",
                                "published_at": "2026-09-22T00:00:00Z"}, dup)
    assert c["qb_id"] is None and c["identity"] == qr.REJ_IDENTITY_AMBIG
    none = _claim("ATL", "Kirk Cousins")
    rec = qr.resolve_team("ATL", None, [c, none], AS_OF, KO)
    assert rec["state"] == qr.UNKNOWN
    assert {r["rejected"] for r in rec["rejected_claims"]} == {qr.REJ_IDENTITY_AMBIG, qr.REJ_IDENTITY_NONE}
    # a supplied id must still be a QB on that team's official roster
    assert qr.link_claim_identity({"team": "GB", "qb_id": "Q-PENIX"}, ROSTER)["qb_id"] is None


def test_future_postgame_unclocked_and_other_team_claims_are_rejected():
    pr = qr.prior_realized_starters(_pbp(), 2026, 3)
    claims = [_claim("ATL", "Michael Penix Jr.", pub="2026-09-24T00:00:00Z"),      # after as_of
              _claim("ATL", "Michael Penix Jr.", pub=None),                         # no clock
              _claim("GB", "Jordan Love"),                                          # other team
              {**_claim("ATL", "Michael Penix Jr."), "claim_kind": "reported"}]     # unconfirmed
    rec = qr.resolve_team("ATL", pr["ATL"], claims, AS_OF, KO)
    assert rec["state"] == qr.UNCONFIRMED
    assert [r["rejected"] for r in rec["rejected_claims"]] == [
        qr.REJ_FUTURE, qr.REJ_NO_CLOCK, qr.REJ_TEAM, qr.REJ_NOT_CONFIRMED]
    late = qr.resolve_team("ATL", pr["ATL"], [_claim("ATL", "Michael Penix Jr.", pub="2026-09-25T03:00:00Z")],
                           "2026-09-26T00:00:00Z", KO)
    assert late["rejected_claims"][0]["rejected"] == qr.REJ_POSTGAME
    with pytest.raises(ValueError):
        qr.resolve_team("ATL", pr["ATL"], [], None, KO)


def test_absent_schedule_qb_columns_are_the_root_cause_and_stay_nan():
    sched = pd.DataFrame([{"game_id": "g", "season": 2026, "week": 3, "game_type": "REG",
                           "home_team": "GB", "away_team": "ATL"}])
    assert qr.schedule_qb_coverage(sched) == {2026: 0.0}
    assert build_qb_continuity(_pbp(), sched) == {}        # the silent skip, documented


def test_context_frame_is_market_blind_and_never_touches_primary_inputs():
    pr = qr.prior_realized_starters(_pbp(), 2026, 3)
    recs = {"ATL": qr.resolve_team("ATL", pr["ATL"], [_claim("ATL", "Michael Penix Jr.")], AS_OF, KO),
            "GB": qr.resolve_team("GB", pr["GB"], [], AS_OF, KO)}
    cands = pd.DataFrame([
        {"team": "ATL", "market": "receiving_yards", "mean": 60.0, "sd": 20.0, "line": 55.5,
         "price": -110, "p_over": 0.55, "p_under": 0.45, "dist": "normal", "qb_continuity": np.nan},
        {"team": "GB", "market": "passing_yards", "mean": 240.0, "sd": 50.0, "line": 230.5,
         "price": -115, "p_over": 0.58, "p_under": 0.42, "dist": "normal", "qb_continuity": np.nan}])
    ctx = qr.context_frame(cands, recs, _pbp(), 2026, 3)
    moved = cands.assign(line=[70.5, 260.5], price=[+150, -300], p_over=[0.1, 0.9], mean=[1.0, 999.0])
    pd.testing.assert_frame_equal(ctx, qr.context_frame(moved, recs, _pbp(), 2026, 3))
    assert list(ctx["qb_ready_state"]) == [qr.VERIFIED_CHANGED, qr.UNCONFIRMED]
    assert ctx["qb_ready_trailing_share"].iloc[0] == 0.5 and pd.isna(ctx["qb_ready_trailing_share"].iloc[1])
    assert "qb_continuity" not in ctx.columns and "backup_qb_adj" not in ctx.columns
    # joining the context must not arm the x0.92 stage
    joined = cands.join(ctx)
    out = candmod.apply_backup_qb_adjustment(joined)
    assert out["mean"].tolist() == cands["mean"].tolist() and "backup_qb_adj" not in out.columns


def test_factor_context_claims_are_team_specific_not_a_league_proxy():
    doc = {"news": [{"category": "qb_news", "claim_key": "starter", "team": "ATL", "game_id": "2026_03_ATL_GB",
                     "claim_value": "Michael Penix Jr.", "claim_kind": "confirmed",
                     "source_url": "https://www.atlantafalcons.com/x", "published_at": "2026-09-22T19:00:00Z"},
                    {"category": "def_absence", "claim_key": "atl_practice", "team": "ATL"}]}
    cl = qr.starter_claims_from_factor_context(doc)
    assert len(cl) == 1 and cl[0]["team"] == "ATL"


@pytest.mark.skipif(not os.path.exists("historical/lines_extra.parquet"), reason="pinned fixture absent")
def test_pinned_2024plus_schedule_has_no_qb_ids():
    cov = qr.schedule_qb_coverage(pd.read_parquet("historical/lines_extra.parquet"))
    assert cov and all(v == 0.0 for s, v in cov.items() if s >= 2024)
