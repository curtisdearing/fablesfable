"""Weekly Leans output contract: rank-1 treatment, source state, safe language,
and the weekly top-five report link.

Why these are output-contract tests rather than DOM tests
---------------------------------------------------------
The dashboard is a single self-contained page whose lean rows are built by an
inline script from inlined JSON, and CI has no JS runtime (see
``test_dashboard_js.py``, which makes the same choice for the same reason).
So the contract is asserted against the GENERATED ARTEFACT: the script block
that every regenerated page carries, and — for the report link, which is
decided in Python at write time — the inlined payload itself.

The rules under test, all of them honesty rules:

1. The rank-1 lean of each matchup is visibly identified as the top model
   lean. It is the model's own ordering, stated plainly, not a claim about
   the world.
2. Certainty language is banned. "lock", "guaranteed", "proven best bet" are
   the words a research tool with no proven edge must never use to sell a
   pick. The ONE permitted occurrence is the standing disclaimer "Leans, not
   locks." — which exists precisely to deny lockness — so the scan removes
   that exact phrase first and then requires the rest of the page to be
   clean. Banning the substring outright would delete the disclaimer, which
   is the opposite of the intent.
3. Every lean states where its line came from, unmistakably: REAL MARKET only
   when ``line_source == "odds_api"``; SYNTHETIC or NO MARKET otherwise. A
   synthetic reference line rendered without that state reads as a real
   sportsbook price, which is the single most expensive misread this page can
   produce.
4. The link to the weekly top-five report must never be a broken promise: it
   is rendered as a live link only when the report actually exists next to
   the page, and as an explicit unavailable state when it does not.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import dashboard          # noqa: E402

# Two leans in one matchup: rank 1 is real-market, rank 2 is synthetic, so a
# single render exercises both the rank treatment and both source states.
PAYLOAD = {
    "season": 2025, "week": 10, "clock": "wed", "as_of": "2025-11-05T12:00:00Z",
    "publish": True, "publish_reasons": [],
    "games": [{
        "game_id": "2025_10_CLE_BAL", "matchup": "CLE @ BAL",
        "screened": "3 of 41", "screened_n": 41,
        "leans": [
            {"player_id": "00-A1", "name": "M.Andrews", "pos": "TE", "team": "BAL",
             "market": "receiving_yards", "side": "under", "line": 52.5,
             "line_source": "odds_api", "mean": 33.1, "composite": 61.2,
             "edge": 0.055},
            {"player_id": "00-B2", "name": "A.Cooper", "pos": "WR", "team": "CLE",
             "market": "receiving_yards", "side": "over", "line": 44.5,
             "line_source": "synthetic_trailing_mean", "mean": 55.0,
             "composite": 44.0, "edge": None},
            {"player_id": "00-C3", "name": "Z.Flowers", "pos": "WR", "team": "BAL",
             "market": "anytime_td", "side": "over", "line": None,
             "line_source": None, "mean": 0.41, "composite": 39.5, "edge": None},
        ]}],
}

#: Matched as WHOLE WORDS. A substring scan would fire on "display:block" and
#: "script block", which is how a language rule ends up disabled for noise.
BANNED = (r"lock(s|ed|ing)?", r"guarantee(s|d)?", r"proven\s+best\s+bet")
#: The one sanctioned use of a banned word anywhere on the page.
DISCLAIMER = "Leans, not locks."


def _render(tmp_path, payload=None, with_report=False):
    """Generate a dashboard next to an optional reports/ directory, exactly as
    the deployed page sits next to its reports/ folder."""
    if with_report:
        reports = tmp_path / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "latest.html").write_text("<html>week 10 top five</html>")
    out = tmp_path / "dashboard.html"
    dashboard.write_dashboard(payload if payload is not None
                              else {"weekly_leans": PAYLOAD}, str(out))
    return out.read_text()


def _script_block(html: str) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert blocks, "dashboard has no script block"
    return max(blocks, key=len)


def _render_leans_source(html: str) -> str:
    """The body of renderLeans() — so assertions about the lean rows cannot be
    satisfied by an identical string living somewhere else on the page."""
    js = _script_block(html)
    start = js.index("function renderLeans(")
    nxt = js.index("\nfunction ", start + 1)
    return js[start:nxt]


def _inlined_payload(html: str) -> dict:
    m = re.search(r"^const DATA = (.*);$", _script_block(html), flags=re.M)
    assert m, "could not find the inlined DATA payload"
    return json.loads(m.group(1))


# --------------------------------------------------------------------------- #
# 1. Rank-1 treatment
# --------------------------------------------------------------------------- #
def test_rank_one_lean_is_labelled_top_model_lean(tmp_path):
    html = _render(tmp_path)
    assert "Top model lean" in html, (
        "rank 1 of each matchup must be visibly identified as the top model lean")


def test_top_model_lean_is_rendered_inside_the_lean_rows(tmp_path):
    src = _render_leans_source(_render(tmp_path))
    assert "Top model lean" in src, (
        "the label must be produced by renderLeans, not by unrelated copy "
        "elsewhere on the page")


def test_top_model_lean_is_keyed_on_rank_one_only(tmp_path):
    """It marks the FIRST lean of each matchup. A label applied to every row
    distinguishes nothing, and one applied by score threshold would silently
    mark zero or several."""
    src = _render_leans_source(_render(tmp_path))
    assert re.search(r"\(\s*l\s*,\s*i\s*\)|\bindex\b|\bidx\b", src), (
        "renderLeans must map leans with their position to identify rank 1")
    assert re.search(r"(i|idx|index)\s*===?\s*0", src), (
        "rank 1 must be selected by position 0 within the matchup")


def test_top_model_lean_meaning_survives_greyscale(tmp_path):
    """House rule for this page (see the Phase 8.4 CSS block): meaning is never
    carried by colour alone. The label is words, and its row marker must be a
    non-colour cue."""
    html = _render(tmp_path)
    style = re.search(r"<style>(.*?)</style>", html, flags=re.S).group(1)
    rule = re.search(r"\.toplean\b[^{]*\{([^}]*)\}", style)
    assert rule, ".toplean needs its own style rule"
    assert re.search(r"border|outline|font-weight|text-transform", rule.group(1)), (
        "the rank-1 row marker must use a non-colour cue (border/weight), so "
        "the distinction survives a greyscale screenshot")


# --------------------------------------------------------------------------- #
# 2. Source state
# --------------------------------------------------------------------------- #
def test_each_lean_declares_an_unmistakable_source_state(tmp_path):
    src = _render_leans_source(_render(tmp_path))
    for state in ("REAL MARKET", "SYNTHETIC", "NO MARKET"):
        assert state in src, f"lean rows must be able to render {state!r}"


def test_real_market_state_is_gated_on_the_odds_api_line_source(tmp_path):
    """REAL MARKET is a claim about provenance. It may be reachable only from
    line_source == "odds_api"."""
    src = _render_leans_source(_render(tmp_path))
    m = re.search(r'line_source\s*===\s*"odds_api"\s*\?\s*([^:]{0,120}):', src)
    assert m, "the source state must branch on line_source === \"odds_api\""
    assert "REAL MARKET" in m.group(1), (
        "the odds_api branch must be the REAL MARKET branch")
    before = src[:src.index("REAL MARKET")]
    assert 'line_source' in before, (
        "REAL MARKET must never be emitted before the line_source check")


def test_synthetic_dagger_and_screen_count_are_preserved(tmp_path):
    """Requirement: the new badge ADDS to the existing honesty furniture, it
    does not replace it."""
    src = _render_leans_source(_render(tmp_path))
    assert "†" in src, "the synthetic-line dagger must survive"
    assert "of ${g.screened_n} screened" in src or "screened_n" in src, (
        "the 'top N of M screened' denominator must survive")


# --------------------------------------------------------------------------- #
# 3. Safe language
# --------------------------------------------------------------------------- #
def test_page_carries_no_certainty_language(tmp_path):
    html = _render(tmp_path, with_report=True)
    assert DISCLAIMER in html, (
        "the standing 'Leans, not locks.' disclaimer must remain on the page")
    scrubbed = html.replace(DISCLAIMER, "")
    for word in BANNED:
        hits = re.findall(rf"\b{word}\b", scrubbed, flags=re.I)
        assert not hits, (
            f"{hits} appears outside the standing disclaimer; this tool has "
            "not proven edge and must never speak with certainty")


def test_gambling_help_line_survives(tmp_path):
    assert "1-800-GAMBLER" in _render(tmp_path)


# --------------------------------------------------------------------------- #
# 4. The weekly top-five report link
# --------------------------------------------------------------------------- #
def test_report_link_is_offered_when_the_report_exists(tmp_path):
    html = _render(tmp_path, with_report=True)
    payload = _inlined_payload(html)
    rep = payload.get("weekly_report")
    assert rep, "write_dashboard must inline the weekly-report availability"
    assert rep.get("available") is True
    assert rep.get("href") == "reports/latest.html"
    assert "View weekly top-five report" in html


def test_report_link_degrades_to_an_explicit_unavailable_state(tmp_path):
    """No report on disk means no link. A dead href is a broken promise; the
    page must say the report is unavailable instead."""
    html = _render(tmp_path, with_report=False)
    payload = _inlined_payload(html)
    rep = payload.get("weekly_report")
    assert rep, "write_dashboard must inline the weekly-report availability"
    assert rep.get("available") is False
    src = _render_leans_source(html)
    assert re.search(r"available\s*\?", src), (
        "the link must be rendered conditionally on availability")
    assert re.search(r"not (yet )?available|unavailable|no report", src, re.I), (
        "the unavailable branch must say so in words")


def test_availability_is_resolved_next_to_the_written_page(tmp_path):
    """The href is relative to the page, so existence must be checked relative
    to the page — not to the repository root, which would advertise a report
    the deployed page cannot reach."""
    sub = tmp_path / "site"
    sub.mkdir()
    html = _render(sub, with_report=True)
    assert _inlined_payload(html)["weekly_report"]["available"] is True
    other = tmp_path / "empty"
    other.mkdir()
    html2 = _render(other, with_report=False)
    assert _inlined_payload(html2)["weekly_report"]["available"] is False


def test_explicit_weekly_report_payload_is_not_overwritten(tmp_path):
    """A caller that already knows where the report is stays authoritative —
    the same contract every other write_dashboard default follows."""
    html = _render(tmp_path, payload={
        "weekly_leans": PAYLOAD,
        "weekly_report": {"available": True, "href": "reports/props_week_2025_10.html"},
    })
    rep = _inlined_payload(html)["weekly_report"]
    assert rep["href"] == "reports/props_week_2025_10.html"


# --------------------------------------------------------------------------- #
# 5. Escaping + script integrity for the new code
# --------------------------------------------------------------------------- #
def test_new_lean_interpolations_stay_escaped(tmp_path):
    """Every user-derived value in a lean row goes through esc(). A raw
    ${l.something} in the new markup is an injection hole."""
    src = _render_leans_source(_render(tmp_path))
    raw = [m.group(0) for m in re.finditer(r"\$\{\s*l\.[A-Za-z_$][\w$]*\s*\}", src)]
    assert not raw, f"unescaped lean interpolation(s): {raw}"


def test_report_link_href_is_escaped(tmp_path):
    src = _render_leans_source(_render(tmp_path, with_report=True))
    m = re.search(r'href="\$\{([^}]*)\}"', src)
    assert m, "the report link must interpolate its href"
    assert "esc(" in m.group(1), "the report href must be escaped"


def test_hostile_lean_strings_cannot_terminate_the_script_block(tmp_path):
    """The payload is inlined verbatim into a <script> block and unescaped at
    render time by esc(). That is safe for markup — but json.dumps does not
    escape "</script>", so a player or matchup string containing one would
    close the block early and drop the rest of the page into the document as
    live HTML. The inlined JSON must make that impossible."""
    hostile = json.loads(json.dumps(PAYLOAD))
    hostile["games"][0]["leans"][0]["name"] = '</script><img src=x onerror="alert(1)">'
    hostile["games"][0]["matchup"] = '</SCRIPT><script>alert(2)</script>'
    html = _render(tmp_path, payload={"weekly_leans": hostile})
    assert html.count("<script>") == html.count("</script>"), (
        "user-derived text unbalanced the script tags")
    payload_line = re.search(r"^const DATA = (.*);$",
                             _script_block(html), flags=re.M)
    assert payload_line, "the inlined payload must still parse out of the block"
    assert not re.search(r"</\s*script", payload_line.group(1), flags=re.I), (
        "a raw </script> survived into the inlined payload — it can close the "
        "block early and inject live HTML")
    # ...and the data itself must survive the escaping intact.
    data = json.loads(payload_line.group(1))
    assert data["weekly_leans"]["games"][0]["leans"][0]["name"] == \
        '</script><img src=x onerror="alert(1)">'


def test_the_dagger_is_not_attached_to_a_lean_that_has_no_line(tmp_path):
    """The dagger's own legend defines it as "synthetic reference line — the
    player's own trailing mean". A lean with NO line of any kind has no such
    number, so a dagger there labels a value that does not exist. NO MARKET is
    the honest state for that row, and it carries no dagger."""
    src = _render_leans_source(_render(tmp_path))
    m = re.search(r'\$\{[^}]*"†"[^}]*\}', src)
    assert m, "the dagger must be conditionally interpolated"
    assert re.search(r"l\.line\s*!=\s*null", m.group(0)), (
        "the dagger must be gated on a line actually existing, not only on "
        "the line source")


def test_a_missing_line_is_shown_as_a_placeholder_not_an_empty_cell(tmp_path):
    """An empty Line cell reads as a rendering bug. A missing line is a fact
    about the market and is displayed as one."""
    src = _render_leans_source(_render(tmp_path))
    assert re.search(r'l\.line\s*!=\s*null\s*\?[^}]*:\s*"—"', src), (
        "a lean with no line must render an explicit placeholder")
