"""The weekly document contract: nothing displayed is unexplained.

Every lean that reaches a reader -- in the HTML drop, the markdown, the
canonical JSON, or the ``leans`` forward log -- carries a NON-EMPTY,
DETERMINISTIC rationale, produced by ONE function. An empty Why cell on a
money-adjacent pick hides exactly the pick a reader most needs to distrust,
and two rationale implementations would eventually disagree about the same
lean on two surfaces.

The document also has to keep its honesty furniture through every path:
synthetic reference lines stay visibly distinct from real sportsbook lines,
the screen denominator stays visible, the publish gate still shows, and the
disclaimer and 1-800-GAMBLER never fall off.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import document as docmod  # noqa: E402
from nflvalue import report as rptmod  # noqa: E402
from nflvalue import week_package as wpmod  # noqa: E402
from nflvalue.freshness import stamp_now  # noqa: E402
from tests.test_week_package import (  # noqa: E402,F401
    GAME_1, SEASON, WEEK, _seed_resnapped_lines, _t90_feeds, _wed_feeds,
    env, synthetic_inputs_multi,   # `env` is re-exported as this module's fixture
)

#: the Why cell is the last column of every lean row
WHY_CELL = re.compile(r"<td class='why'>(.*?)</td></tr>", re.S)
#: pre-fix markup: a bare .sub cell in the last column
LEGACY_WHY_CELL = re.compile(r"<td class='sub'>(.*?)</td></tr>", re.S)


def _why_cells(html: str):
    cells = WHY_CELL.findall(html)
    return cells if cells else LEGACY_WHY_CELL.findall(html)


def _n_leans(payload) -> int:
    return sum(len(g["leans"]) for g in payload["games"])


# =========================================================================== #
# G. Every displayed lean has a non-empty deterministic Why.
# =========================================================================== #
def test_every_html_lean_has_a_non_empty_why(env):
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                      inject_feeds=_wed_feeds(now))
    html = Path(res["drop_path"]).read_text()
    cells = _why_cells(html)
    assert len(cells) == _n_leans(res), \
        "every lean row must render a Why cell"
    blank = [i for i, c in enumerate(cells) if not c.strip()]
    assert not blank, f"{len(blank)} of {len(cells)} HTML Why cells are empty"


def test_every_html_lean_has_a_why_after_a_t90_patch(env):
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    res = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now, out_team="AAA"))
    html = Path(res["drop_path"]).read_text()
    cells = _why_cells(html)
    assert len(cells) == _n_leans(res)
    assert all(c.strip() for c in cells), "a T-90 patch left Why cells empty"


def test_every_lean_carries_its_reason_in_json_markdown_and_the_forward_log(env):
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                      inject_feeds=_wed_feeds(now))

    canonical = json.loads((env["tmp"] / "weekly_props.json").read_text())
    for g in canonical["games"]:
        for lean in g["leans"]:
            assert lean.get("reason"), f"{lean.get('name')} has no reason in the payload"

    md = Path(res["md_path"]).read_text()
    rows = [l for l in md.splitlines()
            if l.startswith("| ") and not l.startswith("| Player") and "---" not in l]
    assert len(rows) == _n_leans(res)
    for row in rows:
        why = row.rstrip("|").rsplit("|", 1)[-1].strip()
        assert why, f"empty Why in markdown row: {row}"

    conn = dbmod.connect()
    reasons = dbmod.query_df(conn, "SELECT reason FROM leans WHERE season=? AND week=?",
                             (SEASON, WEEK))["reason"]
    conn.close()
    assert len(reasons) == _n_leans(res)
    assert reasons.notna().all() and (reasons.str.strip() != "").all()


def test_one_rationale_implementation_feeds_every_surface(env):
    """Not two implementations that happen to agree today."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                      inject_feeds=_wed_feeds(now))
    lean = res["games"][0]["leans"][0]
    assert lean["reason"] == wpmod.rationale(lean)
    assert rptmod._one_line_reason(lean) == wpmod.rationale(lean)
    html = Path(res["drop_path"]).read_text()
    assert lean["reason"][:40] in html


def test_rationale_is_deterministic_and_never_empty():
    """Even a lean with no separating component gets a real sentence."""
    bare = {"player_id": "X", "market": "receptions", "side": "over",
            "mean": 4.1, "line": 3.5}
    assert wpmod.rationale(bare) == wpmod.FALLBACK_REASON
    assert wpmod.rationale(bare) == wpmod.rationale(dict(bare))
    rich = {"player_id": "X", "market": "receiving_yards", "side": "over",
            "mean": 68.2, "line": 61.5, "edge": 0.041,
            "components": {"z": 0.42, "model_prob": 0.58, "market_prob": 0.539,
                           "script_sub": 0.71},
            "proj_components": {"opp_factor": 1.08}}
    once, twice = wpmod.rationale(rich), wpmod.rationale(dict(rich))
    assert once == twice and once.strip()
    assert "z=+0.42" in once and "opp-vs-pos 1.08" in once


def test_why_carries_a_principal_risk_without_inventing_prose(env):
    """Concise model rationale PLUS a counter-case -- built only from facts
    already on the lean, and never implying profitability."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                      inject_feeds=_wed_feeds(now))
    for g in res["games"]:
        for lean in g["leans"]:
            assert lean.get("risk"), f"{lean.get('name')} has no counter-case"
    html = Path(res["drop_path"]).read_text()
    assert "counter-case" in html
    # the claim check belongs on the WHY CELLS, not on the page chrome
    for cell in _why_cells(html):
        low = cell.lower()
        for banned in ("guaranteed", "lock", "profit", "sure thing",
                       "can't lose", "expected value", "+ev"):
            assert banned not in low, f"Why cell over-claims ({banned!r}): {cell}"


def test_synthetic_and_real_lines_stay_visibly_distinct(env):
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    _seed_resnapped_lines(env["db_path"], GAME_1, "AAA Wideout", ts=now)
    res = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now))
    html = Path(res["drop_path"]).read_text()
    assert "†" in html, "synthetic lines lost their dagger"
    assert "no_market" in html, "markets with no real price must say so"
    by_id = {g["game_id"]: g for g in res["games"]}
    sources = {l["line_source"] for l in by_id[GAME_1]["leans"]}
    assert "odds_api" in sources
    real = [l for l in by_id[GAME_1]["leans"] if l["line_source"] == "odds_api"][0]
    synth = [l for l in by_id[GAME_1]["leans"]
             if l["line_source"] != "odds_api"]
    assert real.get("edge") is not None
    assert all(l.get("edge") is None for l in synth)


def test_the_document_keeps_its_publish_gate_and_disclaimers(env):
    """A failed gate, the screen denominator and 1-800-GAMBLER survive both
    the Wednesday path and the T-90 patch path."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                      inject_feeds=_wed_feeds(now))
    html = Path(res["drop_path"]).read_text()
    assert "1-800-GAMBLER" in html and "Leans, not locks" in html
    assert "screened" in html and "Not financial advice" in html

    stale = dict(_wed_feeds(now))
    stale["injuries_fetched_at"] = "2020-01-01T00:00:00Z"
    gated = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=stale)
    assert gated["publish"] is False
    assert "NOT PUBLISHED" in Path(gated["drop_path"]).read_text()


def test_render_drop_refuses_to_show_a_lean_with_no_reason():
    """A unit-level guard on the renderer itself: an unexplained lean is a
    bug, and the document must not quietly print a blank cell for it."""
    payload = {"season": SEASON, "week": WEEK, "clock": "wed", "as_of": "now",
               "publish": True,
               "games": [{"game_id": GAME_1, "matchup": "AAA @ BBB", "screened_n": 41,
                          "leans": [{"player_id": "WR_AAA", "name": "AAA Wideout",
                                     "pos": "WR", "team": "AAA",
                                     "market": "receiving_yards", "side": "over",
                                     "line": 61.5, "line_source": "synthetic_trailing_mean",
                                     "mean": 68.2, "composite": 71.4}]}]}
    html = docmod.render_drop(payload)
    cells = _why_cells(html)
    assert len(cells) == 1
    assert cells[0].strip(), "renderer emitted an empty Why cell"


def test_markdown_html_dashboard_and_discord_all_read_one_payload(env, monkeypatch):
    """Requirement 1 in one assertion set: after a T-90 patch, every surface
    shows the same week -- not the one game the patch happened to re-rank."""
    from nflvalue import config as cfgmod
    from nflvalue import notify
    from tests.test_week_package import GAME_IDS, MATCHUPS

    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    res = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now, out_team="AAA"))

    canonical = json.loads((env["tmp"] / "weekly_props.json").read_text())
    assert [g["game_id"] for g in canonical["games"]] == sorted(GAME_IDS)

    week_md = Path(res["week_md_path"]).read_text()
    html = Path(res["drop_path"]).read_text()
    latest = json.loads((env["tmp"] / "latest.json").read_text())
    cfg = dict(cfgmod.load_config())
    cfg["discord_enabled"] = True
    # never touch a real secret: stub the resolver, and stay in dry-run so
    # nothing is ever POSTed from a test
    monkeypatch.setattr(notify, "resolve_webhook", lambda *a, **k: "https://example.invalid/hook")
    discord = notify.post_weekly(res, cfg=cfg, dry_run=True)
    assert discord["status"] == "dry_run", discord
    discord_blob = json.dumps(discord["messages"])

    for away, home in MATCHUPS:
        assert f"{away} @ {home}" in week_md
        assert f"{away} @ {home}" in html
        assert f"{away} @ {home}" in discord_blob
    assert [g["game_id"] for g in latest["weekly_leans"]["games"]] == sorted(GAME_IDS)
    # the same rationale strings, not three stories about one pick
    lean = canonical["games"][0]["leans"][0]
    assert lean["reason"] and lean["reason"][:40] in html
    assert lean["reason"][:40] in week_md
