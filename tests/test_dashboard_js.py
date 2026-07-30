"""Dashboard script-block integrity.

Regression for the 2026-07-30 find: Phase 8 added a second top-level
declaration of ``esc`` below the existing ``const esc``.  A duplicate
let/const/function declaration is a SyntaxError that kills the ENTIRE script
block — the regenerated page renders with no tabs, no data, no auto-refresh —
and no test noticed because none examined the script's declarations.  The
committed dashboard.html predated the bug, so the live page kept working
while every future deploy was armed to break.

These tests are pure-Python (no JS runtime needed in CI): they extract the
script block and assert that no top-level identifier is declared twice.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import dashboard


def _script_block(html: str) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert blocks, "dashboard has no script block"
    return max(blocks, key=len)


def _top_level_declarations(js: str):
    """Names declared at column 0 via const/let/var/function (the dashboard's
    own style: top-level declarations are never indented)."""
    names = []
    for m in re.finditer(r"^(?:const|let|var)\s+([A-Za-z_$][\w$]*)", js, flags=re.M):
        names.append(m.group(1))
    for m in re.finditer(r"^function\s+([A-Za-z_$][\w$]*)\s*\(", js, flags=re.M):
        names.append(m.group(1))
    return names


def test_no_duplicate_top_level_declarations(tmp_path):
    out = tmp_path / "dash.html"
    dashboard.write_dashboard({"mode": "demo"}, str(out))
    js = _script_block(out.read_text())
    names = _top_level_declarations(js)
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate top-level declarations kill the script block: {dupes}"


def test_esc_declared_exactly_once(tmp_path):
    out = tmp_path / "dash.html"
    dashboard.write_dashboard({"mode": "demo"}, str(out))
    js = _script_block(out.read_text())
    n = len(re.findall(r"^(?:const|let|var|function)\s+esc\b", js, flags=re.M))
    assert n == 1, f"esc declared {n} times"


def test_esc_escapes_quotes(tmp_path):
    """The surviving esc must be the quote-escaping variant (attribute-safe)."""
    out = tmp_path / "dash.html"
    dashboard.write_dashboard({"mode": "demo"}, str(out))
    js = _script_block(out.read_text())
    decl = re.search(r"^const\s+esc\b.*$", js, flags=re.M)
    assert decl and "&quot;" in decl.group(0)


def test_active_tab_survives_reload_via_hash(tmp_path):
    """Contrarian-review fix: location.reload() every 90s reset the page to
    the first tab, making the longest tab (Honest Record) unreadable on a
    live page. The active tab is persisted in the URL hash and restored."""
    out = tmp_path / "dash.html"
    dashboard.write_dashboard({"mode": "demo"}, str(out))
    js = _script_block(out.read_text())
    assert "history.replaceState" in js
    assert "location.hash" in js
    assert "CSS.escape" in js                     # hash is user-controllable input
