"""A T-90 post should say what changed, not repeat the week.

The canonical payload is the WHOLE week by design -- that is what stopped a
one-game run from deleting the rest of the product. But Discord is a feed,
and re-posting fifteen unchanged games every time one game hits T-90 buries
the one thing a reader actually needs: this game just changed, here is how.

So the payload stays whole and the MESSAGE is patch-shaped: the patched
game's leans, what got voided, and an explicit line that the rest of the week
is unchanged -- because a reader who sees one game must not conclude the
other fourteen were dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import notify  # noqa: E402

GAMES = ["2025_10_AAA_BBB", "2025_10_CCC_DDD", "2025_10_EEE_FFF"]


def _lean(name: str, **kw):
    base = dict(player_id=f"P_{name}", name=name, pos="WR", team="AAA",
                market="receiving_yards", side="over", line=61.5,
                line_source="synthetic_trailing_mean", mean=68.2, sd=18.0,
                composite=71.4, edge=None, reason="proj 68.2 vs line 61.5 (z=+0.42)",
                risk="counter-case: synthetic reference line (†)")
    base.update(kw)
    return base


def _payload(clock="wed", patched=None, voided=None):
    return {
        "season": 2025, "week": 10, "clock": clock, "as_of": "2025-11-09T17:30:00Z",
        "publish": True, "publish_reasons": [], "mode": "live",
        "games": [{"game_id": g, "matchup": g.split("_", 2)[-1].replace("_", " @ "),
                   "screened_n": 41, "clock": ("t90" if g in (patched or []) else "wed"),
                   "leans": [_lean(f"{g[-3:]} One"), _lean(f"{g[-3:]} Two")]}
                  for g in GAMES],
        "contexts": {},
        **({"patched_game_id": patched[0], "patched_games": list(patched)} if patched else {}),
        **({"voided": voided} if voided is not None else {}),
    }


def _embed_titles(messages):
    return [e["title"] for m in messages for e in m["embeds"]]


# =========================================================================== #
# Wednesday: unchanged -- the whole slate is the news.
# =========================================================================== #
def test_wednesday_still_posts_every_game():
    messages = notify.build_messages(_payload())
    assert len(_embed_titles(messages)) == len(GAMES)
    assert all(any(g.split("_", 2)[-1].replace("_", " @ ") in t
                   for t in _embed_titles(messages)) for g in GAMES)


# =========================================================================== #
# T-90: only the patched game, and honest about the rest.
# =========================================================================== #
def test_t90_posts_only_the_patched_game():
    payload = _payload(clock="t90", patched=[GAMES[0]],
                       voided=[{"player_id": "P_X", "name": "Alpha Wideout",
                                "market": "receiving_yards", "reason": "inactive"}])
    messages = notify.build_messages(payload)
    titles = _embed_titles(messages)
    assert len(titles) == 1, f"a T-90 patch re-posted the whole week: {titles}"
    assert "AAA @ BBB" in titles[0]


def test_t90_header_says_what_changed_and_that_the_week_is_intact():
    payload = _payload(clock="t90", patched=[GAMES[0]],
                       voided=[{"player_id": "P_X", "name": "Alpha Wideout",
                                "market": "receiving_yards", "reason": "inactive"}])
    header = notify.build_messages(payload)[0]["content"]
    assert "T-90" in header
    assert "AAA @ BBB" in header
    assert "Alpha Wideout" in header, "a voided lean is the whole point of the post"
    # a reader seeing one game must not conclude the others were dropped
    assert "unchanged" in header.lower()
    assert "2" in header  # the two untouched games are counted


def test_t90_with_nothing_voided_still_posts_the_refreshed_game():
    payload = _payload(clock="t90", patched=[GAMES[1]], voided=[])
    messages = notify.build_messages(payload)
    assert _embed_titles(messages) == ["CCC @ DDD — top 2 of 41 screened"]
    assert "no lean voided" in messages[0]["content"].lower()


def test_only_this_runs_patch_is_posted_not_every_game_patched_so_far():
    """`patched_games` accumulates across the week (the dashboard wants that);
    the POST is about the game this run just refreshed."""
    payload = _payload(clock="t90", patched=[GAMES[1]], voided=[])
    payload["patched_games"] = [GAMES[0], GAMES[1]]   # an earlier patch happened
    payload["patched_game_id"] = GAMES[1]
    assert _embed_titles(notify.build_messages(payload)) == ["CCC @ DDD — top 2 of 41 screened"]


def test_a_patched_game_missing_from_the_payload_falls_back_to_the_week():
    """Never post nothing. If the patch pointer does not resolve, the whole
    week is the honest fallback."""
    payload = _payload(clock="t90", patched=["2025_10_ZZZ_YYY"], voided=[])
    assert len(_embed_titles(notify.build_messages(payload))) == len(GAMES)


def test_the_footer_survives_the_patch_path():
    payload = _payload(clock="t90", patched=[GAMES[0]], voided=[])
    messages = notify.build_messages(payload)
    for m in messages:
        for e in m["embeds"]:
            assert "1-800-GAMBLER" in e["footer"]["text"]
    assert "Leans, not locks" in json.dumps(messages)


def test_the_publish_gate_still_wins_over_a_patch(monkeypatch):
    """A failed gate posts a notice, never a confident patch."""
    monkeypatch.setattr(notify, "resolve_webhook", lambda *a, **k: "https://example.invalid/h")
    payload = _payload(clock="t90", patched=[GAMES[0]], voided=[])
    payload["publish"] = False
    payload["publish_reasons"] = ["injuries feed stale (52.0h old)"]
    res = notify.post_weekly(payload, cfg={"discord_enabled": True}, dry_run=True)
    body = json.dumps(res["messages"])
    assert "NOT PUBLISHED" in body and "injuries feed stale" in body
    assert res["messages"][0]["embeds"] == []
