"""QB claim provenance: what THIS issuing run retrieved before its decision clock.

Parent review (2026-09-23) reproduced, on ecf80ce, two claims accepted as a starter
comparison although the run could not have used them:
  (a) published before the decision clock but CAPTURED after it;
  (b) no explicit ``claim_kind`` (None was treated as confirmed).
And a first-passer identity is a proxy for the previous starter, never proof.
"""
import pandas as pd

from nflvalue import qb_readiness as qr

AS_OF = "2026-09-23T02:00:00Z"
KO = "2026-09-25T00:15:00Z"
ROSTER = pd.DataFrame([
    {"player_id": "Q-LOVE", "full_name": "Jordan Love", "team": "GB", "position": "QB"},
    {"player_id": "W-TRICK", "full_name": "Trick Receiver", "team": "GB", "position": "WR"},
])
PRIOR = {"qb_id": "Q-LOVE", "game_id": "2026_02_GB_NYJ", "season": 2026, "week": 2,
         "basis": qr.PRIOR_BASIS_ROSTER_QB}


def _claim(**kw):
    base = {"team": "GB", "claim_value": "Jordan Love", "claim_kind": "confirmed",
            "source": "https://team.example/news", "published_at": "2026-09-23T01:00:00Z",
            "fetched_at": "2026-09-23T01:30:00Z"}
    return qr.link_claim_identity({**base, **kw}, ROSTER)


def test_claim_captured_after_the_decision_clock_is_not_what_this_run_knew():
    rec = qr.resolve_team("GB", PRIOR, [_claim(fetched_at="2026-09-23T03:00:00Z")], AS_OF, KO)
    assert rec["state"] == qr.UNCONFIRMED and rec["qb_id"] is None
    rej = rec["rejected_claims"][0]
    assert rej["rejected"] == qr.REJ_CAPTURED_LATE
    # the historical fact is kept, separately: it was public before the clock
    assert rej["published_before_as_of"] is True


def test_claim_without_a_capture_clock_is_not_consumed():
    rec = qr.resolve_team("GB", PRIOR, [_claim(fetched_at=None)], AS_OF, KO)
    assert rec["state"] == qr.UNCONFIRMED
    assert rec["rejected_claims"][0]["rejected"] == qr.REJ_NO_CAPTURE


def test_claim_without_explicit_confirmation_is_rejected():
    rec = qr.resolve_team("GB", PRIOR, [_claim(claim_kind=None)], AS_OF, KO)
    assert rec["state"] == qr.UNCONFIRMED
    assert rec["rejected_claims"][0]["rejected"] == qr.REJ_NOT_CONFIRMED


def test_valid_claim_is_a_sourced_starter_compared_to_a_proxy_not_verified_continuity():
    rec = qr.resolve_team("GB", PRIOR, [_claim()], AS_OF, KO)
    assert rec["state"] == qr.SOURCED_SAME and rec["qb_id"] == "Q-LOVE"
    assert "verified" not in rec["state"]
    assert rec["prior"]["basis"] == qr.PRIOR_BASIS_ROSTER_QB


def _pbp(first_passer):
    rows = [{"season": 2026, "week": 2, "game_id": "2026_02_GB_NYJ", "play_id": 1,
             "posteam": "GB", "pass_attempt": 1, "passer_player_id": first_passer}]
    rows += [{"season": 2026, "week": 2, "game_id": "2026_02_GB_NYJ", "play_id": i,
              "posteam": "GB", "pass_attempt": 1, "passer_player_id": "Q-LOVE"} for i in range(2, 30)]
    return pd.DataFrame(rows)


def test_trick_play_first_passer_is_not_taken_as_the_starter_when_positions_are_known():
    pr = qr.prior_realized_starters(_pbp("W-TRICK"), 2026, 3, positions={"W-TRICK": "WR",
                                                                          "Q-LOVE": "QB"})
    assert pr["GB"]["qb_id"] == "Q-LOVE" and pr["GB"]["basis"] == qr.PRIOR_BASIS_ROSTER_QB


def test_without_positions_the_first_passer_is_labelled_an_unverified_proxy():
    pr = qr.prior_realized_starters(_pbp("W-TRICK"), 2026, 3)
    assert pr["GB"]["qb_id"] == "W-TRICK" and pr["GB"]["basis"] == qr.PRIOR_BASIS_FIRST_PASSER


def test_module_does_not_claim_historical_ids_were_published_after_games():
    assert "published after the" not in (qr.__doc__ or "")
