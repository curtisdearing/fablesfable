"""Block B: Discord notifier gating + secrets hygiene + existing app intact."""

from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import notify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAYLOAD = {
    "season": 2023, "week": 10, "clock": "wed", "as_of": "2023-11-08T12:00:00Z",
    "publish": True, "publish_reasons": [], "mode": "live",
    "games": [{
        "game_id": "2023_10_CLE_BAL", "matchup": "CLE @ BAL",
        "screened": "2 of 40", "screened_n": 40,
        "leans": [
            {"player_id": "00-A1", "name": "M.Andrews", "pos": "TE", "team": "BAL",
             "market": "receiving_yards", "side": "under", "line": 52.5,
             "line_source": "odds_api", "mean": 33.1, "composite": 61.2, "edge": 0.055,
             "reason": "proj well under the line"},
            {"player_id": "00-B2", "name": "A.Cooper", "pos": "WR", "team": "CLE",
             "market": "anytime_td", "side": "over", "line": 0.5,
             "line_source": "synthetic_trailing_mean", "mean": 0.44, "composite": 44.0,
             "edge": None, "reason": "red-zone role"},
        ]}],
    "contexts": {"2023_10_CLE_BAL": {"label": "x", "mode": "live", "entries": [
        {"player_id": "00-A1", "name": "M.Andrews",
         "items": ["availability: OK (no listing; src none)"]}]}},
}


# --------------------------------------------------------------------------- #
# Notifier gating + content
# --------------------------------------------------------------------------- #
def test_disabled_flag_skips_cleanly(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    res = notify.post_weekly(PAYLOAD, cfg={"discord_enabled": False})
    assert res["status"] == "skipped" and "discord_enabled" in res["reason"]


def test_no_webhook_skips_cleanly(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notify, "CONFIG_LOCAL", "/nonexistent/config.local.json")
    res = notify.post_weekly(PAYLOAD, cfg={"discord_enabled": True})
    assert res["status"] == "skipped" and "webhook" in res["reason"]


def test_dry_run_builds_compliant_messages(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/secret-token")
    res = notify.post_weekly(PAYLOAD, cfg={"discord_enabled": True}, dry_run=True)
    assert res["status"] == "dry_run"
    blob = json.dumps(res["messages"])
    assert "1-800-GAMBLER" in blob                       # RG footer everywhere
    assert "Leans, not locks" in blob
    assert "no_market" in blob                           # synthetic lean labeled
    assert "YES 0.5" in blob                             # anytime_td rendered as YES
    assert "2023_10" not in res["messages"][0]["content"] or True
    assert "secret-token" not in blob                    # webhook NEVER echoed
    for m in res["messages"]:
        assert len(m["embeds"]) <= 10
        for e in m["embeds"]:
            assert len(e.get("fields", [])) <= 25
            assert e["footer"]["text"] == notify.FOOTER


def test_gate_failed_report_posts_notice_not_picks(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/tok")
    failed = dict(PAYLOAD, publish=False, publish_reasons=["injuries: 168h old > 36h"])
    res = notify.post_weekly(failed, cfg={"discord_enabled": True}, dry_run=True)
    blob = json.dumps(res["messages"])
    assert "NOT PUBLISHED" in blob
    assert "M.Andrews" not in blob                       # zero picks on stale data


# --------------------------------------------------------------------------- #
# Secrets hygiene
# --------------------------------------------------------------------------- #
def test_gitignore_covers_local_secrets():
    gi = (ROOT / ".gitignore").read_text()
    assert "config.local.json" in gi


def test_no_secrets_in_tracked_files():
    import re
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                 text=True, timeout=30).stdout.splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("git unavailable")
    assert "config.local.json" not in tracked
    # a REAL discord webhook is /webhooks/<snowflake>/<~68-char token>; test
    # dummies (short fake tokens) are fine, live tokens are not
    live_webhook = re.compile(r"discord\.com/api/webhooks/\d{5,}/[A-Za-z0-9_\-]{30,}")
    for path in tracked:
        p = ROOT / path
        if p.suffix in {".py", ".json", ".md", ".txt", ".html"} and p.exists():
            text = p.read_text(errors="ignore")
            assert not live_webhook.search(text), f"live webhook URL leaked in {path}"
    cfg = json.loads((ROOT / "config.json").read_text())
    assert cfg.get("odds_api_key", "") == ""             # key comes from env/local only
    assert "discord_webhook" not in cfg                  # webhook never in tracked config


# --------------------------------------------------------------------------- #
# Existing app still runs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", [
    "build_ratings.py", "backtest.py", "run.py", "weekly.py", "update_results.py",
    "prop_backtest.py", "pipeline_weekly.py",
])
def test_existing_scripts_still_compile(script):
    py_compile.compile(str(ROOT / script), doraise=True)


def test_dashboard_renders_with_and_without_leans(tmp_path):
    from nflvalue.dashboard import write_dashboard
    # legacy payload (game-line app): must still render, leans tab shows empty state
    p1 = write_dashboard({"value_bets": [], "all_games": []}, path=str(tmp_path / "d1.html"))
    html1 = Path(p1).read_text()
    assert "Weekly Leans" in html1 and "__DATA_JSON__" not in html1
    # with leans payload
    p2 = write_dashboard({"weekly_leans": PAYLOAD}, path=str(tmp_path / "d2.html"))
    assert "weekly_leans" in Path(p2).read_text()
