"""A degraded feed is shown whether or not it blocked the board.

2026-09-02, the first live board of the season: ESPN answered 403 to the
injuries feed (load-bearing -> NOT PUBLISHED, which the page already said)
AND to the news feed (context-only -> the board would have published with
every lean capped at "low" confidence and NOT ONE WORD on the page saying
why). A reader cannot tell a quiet week from a dead feed unless the reason is
printed. Both surfaces -- the dashboard's Weekly Leans panel and the weekly
document itself -- must name every reason the freshness gate recorded.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import dashboard  # noqa: E402
from nflvalue import document   # noqa: E402

NEWS_403 = "news: no/unparseable timestamp -- treated as missing"
INJ_403 = "injuries: no/unparseable timestamp -- treated as missing"


def _payload(publish: bool, reasons):
    return {
        "season": 2026, "week": 1, "clock": "wed", "as_of": "2026-09-02T01:13:36Z",
        "publish": publish, "publish_reasons": list(reasons),
        "games": [{
            "game_id": "2026_01_DAL_PHI", "matchup": "DAL @ PHI", "screened_n": 40,
            "leans": [{"name": "A.Brown", "pos": "WR", "team": "PHI",
                       "market": "receiving_yards", "side": "over", "line": 70.5,
                       "line_source": "odds_api", "mean": 80.0, "composite": 60.0,
                       "edge": 0.04, "confidence": "low"}]}],
    }


def _render_leans_source(html: str) -> str:
    js = max(re.findall(r"<script>(.*?)</script>", html, flags=re.S), key=len)
    start = js.index("function renderLeans(")
    return js[start:js.index("\nfunction ", start + 1)]


def test_dashboard_names_a_context_feed_failure_on_a_published_board(tmp_path):
    out = tmp_path / "dashboard.html"
    dashboard.write_dashboard({"weekly_leans": _payload(True, [NEWS_403])}, str(out))
    src = _render_leans_source(out.read_text())
    assert "Feed warnings" in src
    assert "confidence is capped" in src
    # the reasons reach the panel through the payload, not a hard-coded string
    assert "publish_reasons" in src
    assert "NOT PUBLISHED" in src  # the blocking branch is still there


def test_dashboard_stays_quiet_when_every_feed_landed(tmp_path):
    out = tmp_path / "dashboard.html"
    dashboard.write_dashboard({"weekly_leans": _payload(True, [])}, str(out))
    src = _render_leans_source(out.read_text())
    # the warning is gated on reasons being present, not rendered unconditionally
    assert "reasons.length" in src


def test_document_names_every_gate_reason_published_or_not():
    doc = document.render_drop(_payload(True, [NEWS_403]))
    assert "Feed warnings" in doc and "confidence capped at low" in doc
    assert NEWS_403 in doc
    assert "NOT PUBLISHED" not in doc

    blocked = document.render_drop(_payload(False, [INJ_403, NEWS_403]))
    assert "NOT PUBLISHED" in blocked
    assert "Publish gate failed" in blocked
    assert INJ_403 in blocked and NEWS_403 in blocked

    clean = document.render_drop(_payload(True, []))
    assert "Feed warnings" not in clean and "Publish gate failed" not in clean
