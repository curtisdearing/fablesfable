"""The Weekly Leans tab, EXECUTED -- not pattern-matched.

The 2026-07-30 lesson written into this repo's memory: a string-presence test
on the dashboard template misses the failure that actually happens, which is a
SyntaxError killing the whole script block so the page renders empty. So this
file runs the real script in Node against a real payload and reads the HTML it
produces.

What it holds the tab to, beyond "it renders":

* After a T-90 patch the payload is the WHOLE week, and its top-level ``clock``
  reads "t90". Without a per-game marker that makes fifteen Wednesday games
  look freshly refreshed 90 minutes before kickoff. Each game therefore shows
  which clock produced IT.
* Every lean shows its Why -- the same rationale + counter-case the markdown,
  the HTML drop and the ``leans`` row carry. A blank Why on a money-adjacent
  page is the failure this column exists to prevent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import dashboard  # noqa: E402

NODE = shutil.which("node") or shutil.which("nodejs")
needs_node = pytest.mark.skipif(not NODE, reason="node not available to execute the dashboard JS")

#: Enough of a DOM for the dashboard's bootstrap to run headlessly. Anything
#: the page touches at load must exist here, or a real SyntaxError would be
#: indistinguishable from a missing shim.
SHIM = """
const __els = {};
function __el(id){
  if(!__els[id]) __els[id] = {id, innerHTML:"", textContent:"", style:{}, dataset:{},
                              classList:{add(){}, remove(){}, toggle(){}, contains(){return false}},
                              onclick:null, appendChild(){}, setAttribute(){}, getAttribute(){return null}};
  return __els[id];
}
globalThis.document = {
  getElementById: __el,
  querySelector: () => __el("_q"),
  querySelectorAll: () => [],
  createElement: () => __el("_c"),
  addEventListener(){},
};
globalThis.window = globalThis;
globalThis.location = {hash: "", reload(){}, href: "about:blank"};
globalThis.history = {replaceState(){}};
globalThis.setInterval = () => 0;
globalThis.setTimeout = () => 0;
globalThis.CSS = {escape: s => String(s)};
globalThis.fetch = () => Promise.reject(new Error("no network in the shim"));
globalThis.__dump = id => __el(id).innerHTML;
"""


def _script_block(html: str) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert blocks, "dashboard has no script block"
    return max(blocks, key=len)


def _render(tmp_path, data: dict, element: str = "leans") -> str:
    """Run the page's real JS in Node and return one element's innerHTML."""
    out = tmp_path / "dash.html"
    dashboard.write_dashboard(data, str(out))
    js = tmp_path / "run.js"
    js.write_text(SHIM + _script_block(out.read_text())
                  + f'\nprocess.stdout.write(__dump({json.dumps(element)}));\n')
    proc = subprocess.run([NODE, str(js)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        f"the dashboard script did not execute (this is the SyntaxError class of bug):\n"
        f"{proc.stderr[-2000:]}")
    return proc.stdout


def _lean(name, **kw):
    base = dict(player_id=f"P_{name}", name=name, pos="WR", team="AAA",
                market="receiving_yards", side="over", line=61.5,
                line_source="synthetic_trailing_mean", mean=68.2, composite=71.4,
                edge=None,
                reason="proj 68.2 vs line 61.5 (z=+0.42); opp-vs-pos 1.08",
                risk="counter-case: synthetic reference line (†) — no sportsbook price")
    base.update(kw)
    return base


def _data(**kw):
    games = [
        {"game_id": "2025_10_AAA_BBB", "matchup": "AAA @ BBB", "screened_n": 41,
         "clock": "t90", "leans": [_lean("Alpha Wideout")]},
        {"game_id": "2025_10_CCC_DDD", "matchup": "CCC @ DDD", "screened_n": 38,
         "clock": "wed", "leans": [_lean("Charlie Wideout", line_source="odds_api",
                                         edge=0.041, line=59.5)]},
    ]
    data = {"mode": "live", "generated_at": "2025-11-09T17:30:00Z",
            "weekly_leans": {"season": 2025, "week": 10, "clock": "t90",
                             "as_of": "2025-11-09T17:30:00Z", "publish": True,
                             "publish_reasons": [], "games": games, "contexts": {},
                             "patched_games": ["2025_10_AAA_BBB"]}}
    data.update(kw)
    return data


# =========================================================================== #
# The page runs at all.
# =========================================================================== #
@needs_node
def test_the_weekly_leans_tab_executes_and_renders(tmp_path):
    html = _render(tmp_path, _data())
    assert "AAA @ BBB" in html and "CCC @ DDD" in html
    assert "1-800-GAMBLER" in html


@needs_node
def test_an_empty_payload_still_executes(tmp_path):
    html = _render(tmp_path, {"mode": "demo"})
    assert "No weekly prop leans" in html


# =========================================================================== #
# Per-game clock: which read is this game on?
# =========================================================================== #
@needs_node
def test_each_game_shows_which_clock_produced_it(tmp_path):
    html = _render(tmp_path, _data())
    patched = html.split("CCC @ DDD")[0]
    untouched = html.split("CCC @ DDD")[1]
    assert "T-90" in patched, "the patched game does not say it was refreshed at T-90"
    assert "Wednesday" in untouched, \
        "an untouched game must not read as freshly refreshed just because the week's clock is t90"


@needs_node
def test_a_wednesday_only_week_marks_every_game_wednesday(tmp_path):
    data = _data()
    for g in data["weekly_leans"]["games"]:
        g["clock"] = "wed"
    data["weekly_leans"]["clock"] = "wed"
    data["weekly_leans"].pop("patched_games", None)
    html = _render(tmp_path, data)
    assert html.count("Wednesday") >= 2
    assert "T-90" not in html


# =========================================================================== #
# Every lean shows its Why.
# =========================================================================== #
@needs_node
def test_every_lean_shows_its_rationale_and_counter_case(tmp_path):
    html = _render(tmp_path, _data())
    assert "Why" in html, "the leans table has no Why column"
    assert "z=+0.42" in html, "the model rationale is missing from the tab"
    assert "counter-case" in html, "the counter-case is missing from the tab"


@needs_node
def test_a_lean_with_no_rationale_is_shown_as_unexplained_not_blank(tmp_path):
    """A blank cell reads as "nothing to say here"; an unexplained pick on a
    money-adjacent page has to say so."""
    data = _data()
    data["weekly_leans"]["games"][0]["leans"][0].pop("reason")
    data["weekly_leans"]["games"][0]["leans"][0].pop("risk")
    html = _render(tmp_path, data)
    normalised = html.replace("'", '"')
    assert '<td class="why"></td>' not in normalised
    assert "unexplained" in html


# =========================================================================== #
# The honesty furniture survives.
# =========================================================================== #
@needs_node
def test_synthetic_and_real_lines_stay_visibly_distinct(tmp_path):
    html = _render(tmp_path, _data())
    assert "†" in html
    assert "no_market" in html
    assert "+4.1%" in html, "a real line must show its edge"


@needs_node
def test_lean_text_is_escaped_not_injected(tmp_path):
    data = _data()
    data["weekly_leans"]["games"][0]["leans"][0]["name"] = "<script>alert(1)</script>"
    data["weekly_leans"]["games"][0]["leans"][0]["risk"] = "counter-case: <img onerror=x>"
    html = _render(tmp_path, data)
    assert "<script>alert(1)</script>" not in html
    assert "<img onerror=x>" not in html
    assert "&lt;script&gt;" in html


@needs_node
def test_a_payload_string_cannot_close_the_script_block(tmp_path):
    """The payload is inlined into <script>. A note or headline containing
    "</script>" would end the block early and blank the WHOLE page -- the same
    invisible-until-deploy failure as a duplicate declaration."""
    data = _data()
    data["weekly_leans"]["contexts"] = {"2025_10_AAA_BBB": {
        "label": "Context only", "mode": "live",
        "entries": [{"player_id": "P", "name": "Alpha",
                     "items": ["manual note: </script><b>pwned</b>"]}]}}
    html = _render(tmp_path, data)          # executing at all is the assertion
    assert "AAA @ BBB" in html
    assert "<b>pwned</b>" not in html
