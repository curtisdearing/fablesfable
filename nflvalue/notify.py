"""Discord notifier -- flag-gated, personal, unmonetized.

Posts the weekly leans to a PRIVATE Discord webhook as embeds. Hard rules:

  * OFF by default: runs only when config ``discord_enabled`` is true AND a
    webhook URL exists. Missing either -> clean skip (returns skipped status,
    never raises, never blocks the pipeline).
  * The webhook URL is a SECRET: read from the ``DISCORD_WEBHOOK_URL`` env
    var or ``config.local.json`` (gitignored). It is never written to any
    tracked file, any log line, or any returned payload.
  * Never posts when the report's freshness gate said ``publish=false`` --
    stale data does not get a confident post (it CAN post an explicit
    "not published" notice if ``post_gate_notice`` is set).
  * Every post carries the disclaimer + 1-800-GAMBLER footer. Leans, not locks.
  * This module INFORMS. It contains no wagering, no bet-placement, no
    bookmaker session of any kind, and never will.

Discord limits respected: <=10 embeds/message, <=25 fields/embed,
<=6000 chars/message -- games are chunked across messages as needed.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List, Optional

from . import config as cfgmod

FOOTER = ("Leans, not locks — model-ranked research on free data; variance is variance. "
          "Not financial advice. Gambling problem? 1-800-GAMBLER")
CONFIG_LOCAL = os.path.join(cfgmod.ROOT, "config.local.json")
MAX_EMBEDS_PER_MSG = 10
MAX_FIELDS_PER_EMBED = 25


def resolve_webhook() -> Optional[str]:
    """Env var first, then config.local.json (both untracked). Never config.json."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        return url.strip()
    if os.path.exists(CONFIG_LOCAL):
        try:
            with open(CONFIG_LOCAL) as f:
                return (json.load(f).get("discord_webhook") or "").strip() or None
        except Exception:  # noqa: BLE001
            return None
    return None


def _side_label(lean: Dict) -> str:
    return "YES" if lean.get("market") == "anytime_td" else str(lean.get("side", "")).upper()


def _game_embed(game: Dict, context: Optional[Dict]) -> Dict:
    fields = []
    for l in game.get("leans", [])[:MAX_FIELDS_PER_EMBED - 1]:
        edge = (f"{l['edge']*100:+.1f}%" if l.get("edge") is not None else "no_market")
        synth = "" if l.get("line_source") == "odds_api" else "†"
        name = f"{l.get('name')} · {str(l.get('market','')).replace('_',' ')}"
        rank_score = l.get("ml_score") if l.get("ml_score") is not None else l.get("composite")
        value = (f"**{_side_label(l)} {l.get('line')}{synth}** · proj {l.get('mean')} · "
                 f"edge {edge} · score {rank_score}\n{l.get('reason','')[:140]}")
        fields.append({"name": name[:256], "value": value[:1024], "inline": False})
    desc_parts = []
    notes = game.get("notes") or []
    if notes:
        desc_parts.append("\n".join(f"• {n}" for n in notes[:7]))
    if context and context.get("entries"):
        first = context["entries"][0]
        if first["items"]:
            desc_parts.append(f"Context (not scored): {first['name']} — {first['items'][0]}"[:220])
    desc_parts.append("† = synthetic reference line, not a market price")
    return {
        "title": f"{game.get('matchup')} — top {len(game.get('leans', []))} "
                 f"of {game.get('screened_n')} screened",
        "description": "\n".join(desc_parts)[:4096],
        "fields": fields,
    }


def build_messages(report_payload: Dict) -> List[Dict]:
    """Weekly report payload (report.generate output) -> list of webhook bodies."""
    season, week = report_payload.get("season"), report_payload.get("week")
    clock = report_payload.get("clock", "wed")
    header = (f"**NFL Prop Leans — {season} week {week}** (clock: {clock}, "
              f"as of {report_payload.get('as_of')})\n"
              "Personal, unmonetized research post. Leans, not locks.")
    embeds = []
    contexts = report_payload.get("contexts") or {}
    for g in report_payload.get("games", []):
        if g.get("leans"):
            embeds.append(_game_embed(g, contexts.get(g["game_id"])))
    messages = []
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MSG):
        chunk = embeds[i:i + MAX_EMBEDS_PER_MSG]
        for e in chunk:
            e["footer"] = {"text": FOOTER}
        messages.append({
            "content": header if i == 0 else f"(continued — {season} week {week})",
            "embeds": chunk,
        })
    return messages


def post_weekly(report_payload: Dict, cfg: Optional[Dict] = None,
                dry_run: bool = True, post_gate_notice: bool = True,
                opener=None) -> Dict:
    """Post the weekly leans. Returns a status dict; never raises.

    ``dry_run`` DEFAULTS TO True -- safe by default for direct callers. The
    pipeline always passes an explicit value, so real posting behaviour is
    unchanged. ``opener`` is an injectable ``urlopen``-shaped seam for tests;
    it defaults to :func:`urllib.request.urlopen`."""
    cfg = cfg or cfgmod.load_config()
    if not cfg.get("discord_enabled"):
        return {"status": "skipped", "reason": "discord_enabled is false in config.json"}
    webhook = resolve_webhook()
    if not webhook:
        return {"status": "skipped",
                "reason": "no webhook (set DISCORD_WEBHOOK_URL or config.local.json:discord_webhook)"}

    if report_payload.get("publish") is False:
        if not post_gate_notice:
            return {"status": "skipped", "reason": "publish=false and gate notices disabled"}
        messages = [{
            "content": (f"**NFL Prop Leans — {report_payload.get('season')} week "
                        f"{report_payload.get('week')}: NOT PUBLISHED.** Data gate failed: "
                        + "; ".join(report_payload.get("publish_reasons") or ["stale/missing feed"])
                        + f"\n{FOOTER}"),
            "embeds": [],
        }]
    else:
        messages = build_messages(report_payload)

    if dry_run:
        return {"status": "dry_run", "n_messages": len(messages), "messages": messages}

    urlopen = opener or urllib.request.urlopen
    posted = 0
    for body in messages:
        req = urllib.request.Request(
            webhook, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "nfl-value/1.0"})
        # The docstring promises this "never blocks the pipeline" -- a Discord
        # 5xx/timeout must degrade to a status dict, not raise. The webhook URL
        # is NEVER put in the returned payload or any log line.
        try:
            with urlopen(req, timeout=15) as resp:  # noqa: S310 - user-configured webhook
                resp.read()
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "n_messages": posted,
                    "reason": f"discord post failed after {posted} message(s): "
                              f"{type(exc).__name__}"}
        posted += 1
    return {"status": "posted", "n_messages": posted}
