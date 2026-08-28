"""reports/latest.html: the stable target the dashboard's report link points at,
and the staleness rule that keeps that link from becoming a quiet lie.

The dashboard offers "View weekly top-five report" -> reports/latest.html. Two
ways that link can mislead, both closed here:

1. Nothing ever wrote reports/latest.html, so the link was permanently dead.
   Every weekly drop now also lands there.
2. Worse than dead: a report from an EARLIER week sitting at that path while
   the dashboard shows this week's leans. The reader clicks "the weekly
   report" and gets last week's picks with nothing saying so. A sidecar
   records which week the file is for, and the dashboard compares it against
   the week it is itself rendering.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import dashboard                # noqa: E402
from nflvalue import document                 # noqa: E402

PAYLOAD = {
    "season": 2025, "week": 10, "clock": "wed", "as_of": "2025-11-05T12:00:00Z",
    "publish": True,
    "games": [{
        "game_id": "2025_10_CLE_BAL", "matchup": "CLE @ BAL", "screened_n": 41,
        "leans": [
            {"name": "M.Andrews", "pos": "TE", "team": "BAL",
             "market": "receiving_yards", "side": "under", "line": 52.5,
             "line_source": "odds_api", "mean": 33.1, "composite": 61.2,
             "edge": 0.055},
        ]}],
}


def _dirs(tmp_path):
    return str(tmp_path / "drops"), str(tmp_path / "reports")


# --------------------------------------------------------------------------- #
# 1. Every drop also lands at reports/latest.html
# --------------------------------------------------------------------------- #
def test_writing_a_drop_also_writes_reports_latest_html(tmp_path):
    drops, reports = _dirs(tmp_path)
    document.write_drop(PAYLOAD, drops_dir=drops, reports_dir=reports)
    latest = os.path.join(reports, "latest.html")
    assert os.path.exists(latest), (
        "the dashboard links to reports/latest.html; a weekly run must produce it")


def test_latest_html_is_the_same_document_as_the_drop(tmp_path):
    drops, reports = _dirs(tmp_path)
    drop_path = document.write_drop(PAYLOAD, drops_dir=drops, reports_dir=reports)
    with open(drop_path, encoding="utf-8") as fh:
        drop = fh.read()
    with open(os.path.join(reports, "latest.html"), encoding="utf-8") as fh:
        latest = fh.read()
    assert latest == drop, "latest.html must be the week's drop, not a summary of it"
    assert "Leans, not locks" in latest, "the honesty furniture travels with it"


def test_write_drop_still_returns_the_drop_path(tmp_path):
    """Callers store this as result["drop_path"]. The return contract does not
    change just because a second file is now written."""
    drops, reports = _dirs(tmp_path)
    got = document.write_drop(PAYLOAD, drops_dir=drops, reports_dir=reports)
    assert got == os.path.join(drops, "props_week_2025_10.html")


# --------------------------------------------------------------------------- #
# 2. The sidecar that makes staleness detectable
# --------------------------------------------------------------------------- #
def test_a_sidecar_records_which_week_the_report_is_for(tmp_path):
    drops, reports = _dirs(tmp_path)
    document.write_drop(PAYLOAD, drops_dir=drops, reports_dir=reports)
    with open(os.path.join(reports, "latest.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["season"] == 2025 and meta["week"] == 10
    assert meta["clock"] == "wed"
    assert meta["as_of"] == "2025-11-05T12:00:00Z"


def test_the_newest_run_wins(tmp_path):
    """T-90 runs after Wednesday. "latest" means latest."""
    drops, reports = _dirs(tmp_path)
    document.write_drop(PAYLOAD, drops_dir=drops, reports_dir=reports)
    later = dict(PAYLOAD, clock="t90", as_of="2025-11-09T15:30:00Z")
    document.write_drop(later, drops_dir=drops, reports_dir=reports)
    with open(os.path.join(reports, "latest.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["clock"] == "t90"
    assert "t90" in os.listdir(drops)[0] or any("t90" in f for f in os.listdir(drops)), \
        "both drops are kept; only latest.* is overwritten"


def test_a_failed_latest_write_never_costs_us_the_drop(tmp_path):
    """latest.html is a convenience pointer. The drop is the deliverable, and a
    weekly run must not fail because the pointer could not be written."""
    drops, _ = _dirs(tmp_path)
    blocked = tmp_path / "reports-is-a-file"
    blocked.write_text("not a directory")
    drop_path = document.write_drop(PAYLOAD, drops_dir=drops,
                                    reports_dir=str(blocked))
    assert os.path.exists(drop_path), "the drop itself must still be written"


# --------------------------------------------------------------------------- #
# 3. The dashboard reads the sidecar and refuses to present a stale report
#    as the current one
# --------------------------------------------------------------------------- #
def _dash(tmp_path, leans_week=10, sidecar=None):
    if sidecar is not None:
        reports = tmp_path / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "latest.html").write_text("<html>report</html>")
        (reports / "latest.json").write_text(json.dumps(sidecar))
    out = tmp_path / "dashboard.html"
    leans = dict(PAYLOAD, week=leans_week)
    dashboard.write_dashboard({"weekly_leans": leans}, str(out))
    html = out.read_text()
    js = max(re.findall(r"<script>(.*?)</script>", html, flags=re.S), key=len)
    payload = json.loads(re.search(r"^const DATA = (.*);$", js, flags=re.M).group(1))
    return html, payload["weekly_report"]


def test_the_link_names_the_week_the_report_covers(tmp_path):
    _html, rep = _dash(tmp_path, sidecar={"season": 2025, "week": 10, "clock": "wed"})
    assert rep["available"] is True
    assert rep["season"] == 2025 and rep["week"] == 10
    assert rep["stale"] is False


def test_a_report_from_another_week_is_marked_stale(tmp_path):
    """The dashboard is showing week 10; the file on disk is week 9. The link
    may still be offered — an older report is not worthless — but it must not
    be presented as this week's."""
    _html, rep = _dash(tmp_path, leans_week=10,
                       sidecar={"season": 2025, "week": 9, "clock": "wed"})
    assert rep["available"] is True
    assert rep["stale"] is True
    assert rep["week"] == 9


def test_a_report_with_no_sidecar_cannot_claim_to_be_current(tmp_path):
    """No sidecar means we do not know what week the file covers. Unknown is
    not the same as current, and must not be rendered as current."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.html").write_text("<html>report</html>")
    out = tmp_path / "dashboard.html"
    dashboard.write_dashboard({"weekly_leans": PAYLOAD}, str(out))
    html = out.read_text()
    js = max(re.findall(r"<script>(.*?)</script>", html, flags=re.S), key=len)
    rep = json.loads(re.search(r"^const DATA = (.*);$", js, flags=re.M).group(1))["weekly_report"]
    assert rep["available"] is True
    assert rep["season"] is None and rep["week"] is None
    assert rep["stale"] is None, "unknown provenance is None, not False"


def test_a_corrupt_sidecar_does_not_break_the_page(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.html").write_text("<html>report</html>")
    (reports / "latest.json").write_text("{ this is not json")
    out = tmp_path / "dashboard.html"
    dashboard.write_dashboard({"weekly_leans": PAYLOAD}, str(out))
    html = out.read_text()
    assert "Weekly Leans" in html
    js = max(re.findall(r"<script>(.*?)</script>", html, flags=re.S), key=len)
    rep = json.loads(re.search(r"^const DATA = (.*);$", js, flags=re.M).group(1))["weekly_report"]
    assert rep["available"] is True and rep["stale"] is None


def test_the_ui_says_out_of_date_rather_than_hiding_it(tmp_path):
    """Structural: renderLeans must have a branch that speaks the staleness."""
    html, _rep = _dash(tmp_path, leans_week=10,
                       sidecar={"season": 2025, "week": 9, "clock": "wed"})
    js = max(re.findall(r"<script>(.*?)</script>", html, flags=re.S), key=len)
    start = js.index("function renderLeans(")
    src = js[start:js.index("\nfunction ", start + 1)]
    assert re.search(r"\.stale\b", src), "the UI must consult the staleness flag"
    assert re.search(r"not this week|earlier week|out of date|older week", src, re.I), (
        "a stale report must be labelled in words, not silently linked")


def test_end_to_end_a_real_drop_makes_the_dashboard_link_live(tmp_path):
    """The whole point, in one test: run the writer, then generate the page
    beside it, and the link is offered and current."""
    drops = str(tmp_path / "drops")
    reports = str(tmp_path / "reports")
    document.write_drop(PAYLOAD, drops_dir=drops, reports_dir=reports)
    out = tmp_path / "dashboard.html"
    dashboard.write_dashboard({"weekly_leans": PAYLOAD}, str(out))
    js = max(re.findall(r"<script>(.*?)</script>", out.read_text(), flags=re.S), key=len)
    rep = json.loads(re.search(r"^const DATA = (.*);$", js, flags=re.M).group(1))["weekly_report"]
    assert rep["available"] is True and rep["stale"] is False
    assert rep["href"] == "reports/latest.html"


# --------------------------------------------------------------------------- #
# 4. ...and the deployed site must actually serve it
#
# Text assertions, not a YAML parse: PyYAML is not a dependency of this repo
# (requirements.txt is deliberately near-empty) and adding one so a test can
# read a 200-line workflow is a bad trade. importorskip would be worse -- it
# would turn this into a silent skip in CI, the exact failure mode
# FABLESFABLE_STRICT_FIXTURES exists to prevent.
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "live-weekly.yml")


def _pages_step() -> str:
    with open(WORKFLOW, encoding="utf-8") as fh:
        wf = fh.read()
    start = wf.index("Prepare dashboard for GitHub Pages")
    end = wf.index("actions/configure-pages", start)
    return wf[start:end]


def test_the_pages_deploy_publishes_the_report_next_to_the_dashboard():
    """dashboard.html becomes _site/index.html, and it links to
    reports/latest.html RELATIVE TO ITSELF. If the deploy copies only the
    dashboard, that link 404s on the public site for every reader."""
    step = _pages_step()
    assert "reports/latest.html" in step, (
        "the Pages step must publish reports/latest.html")
    assert "_site/reports" in step, (
        "it must land at _site/reports/ so the relative link resolves")


def test_the_published_path_matches_the_href_the_dashboard_renders():
    """The invariant that keeps these two files from drifting apart: the path
    the page asks for and the path the deploy writes are the same string."""
    with open(os.path.join(ROOT, "nflvalue", "dashboard.py"), encoding="utf-8") as fh:
        dash = fh.read()
    hrefs = set(re.findall(r'"href": "([^"]+)"', dash))
    assert hrefs == {"reports/latest.html"}, f"unexpected href(s): {hrefs}"
    step = _pages_step()
    href = hrefs.pop()
    assert f"_site/{os.path.dirname(href)}" in step, (
        "the deploy must publish into the directory the href names")


def test_a_missing_report_does_not_fail_the_deploy():
    """Push-to-main runs a deploy-only heartbeat with no model run, so there is
    often no report at all. That must publish the dashboard, not fail."""
    step = _pages_step()
    assert re.search(r"if \[ -f reports/latest\.html \]", step), (
        "the copy must be guarded on the file existing")


def test_the_sidecar_ships_with_the_report():
    """Without latest.json the deployed page cannot tell which week the report
    covers, and falls back to the weakest of its three states."""
    assert "latest.json" in _pages_step()


def test_the_committed_dashboard_never_promises_a_report_the_repo_lacks():
    """The deploy hole this closes.

    live-weekly.yml commits nothing back, and dashboard.html IS tracked. So a
    push-to-main heartbeat -- which runs no model job -- publishes the
    COMMITTED dashboard.html, and the only reports/ files present are the
    tracked ones. If someone regenerates dashboard.html on a machine where
    reports/latest.html exists locally but does not commit that report, the
    public page ships a link to a 404: the exact broken promise the report
    link was built to avoid.

    Invariant: if the committed page says a report is available, the committed
    report must exist. Vacuous for a dashboard.html generated before
    weekly_report existed, and load-bearing from the first regeneration after.
    """
    dash = os.path.join(ROOT, "dashboard.html")
    if not os.path.exists(dash):
        pytest.skip("no committed dashboard.html in this checkout")
    with open(dash, encoding="utf-8") as fh:
        html = fh.read()
    blocks = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    if not blocks:
        pytest.skip("committed dashboard.html has no script block")
    m = re.search(r"^const DATA = (.*);$", max(blocks, key=len), flags=re.M)
    if not m:
        pytest.skip("committed dashboard.html predates the inlined DATA payload")
    try:
        rep = (json.loads(m.group(1)) or {}).get("weekly_report")
    except ValueError:
        pytest.skip("committed dashboard.html payload is not readable as JSON")
    if not rep or not rep.get("available"):
        return              # nothing promised, nothing to keep
    # Ask git, not the filesystem: a heartbeat deploys what is COMMITTED, and
    # the working tree can be dirtied by any test run that writes a report.
    try:
        tracked = subprocess.run(["git", "ls-files", "reports/latest.html"],
                                 cwd=ROOT, capture_output=True, text=True,
                                 timeout=30).stdout.split()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("git unavailable; cannot inspect committed files")
    assert tracked, (
        "the committed dashboard.html advertises a weekly report, but "
        "reports/latest.html is not committed -- a push-to-main heartbeat "
        "would deploy that page with a dead link. Commit the report next to "
        "the dashboard, or regenerate the dashboard without it.")
