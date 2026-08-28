"""The weekly DROP: a polished, self-contained HTML document per week.

Written to ``drops/props_week_{S}_{W}.html`` by every Wednesday run — opens
in any browser, prints cleanly to PDF, needs no server and no dependencies.
Same honesty furniture as every other surface: leans not locks, screen
counts, synthetic-line daggers, display-only context, 1-800-GAMBLER.
"""

from __future__ import annotations

import html
import os
from typing import Dict, List, Optional

from . import config as cfgmod
from . import week_package as wpmod

DROPS_DIR = os.path.join(cfgmod.ROOT, "drops")

_CSS = """
body{font:15px/1.55 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
 color:#141a24;max-width:880px;margin:24px auto;padding:0 20px;background:#fff}
h1{font-size:26px;margin:0 0 2px} h2{font-size:19px;margin:26px 0 4px;
 border-bottom:2px solid #14532d;padding-bottom:3px}
.sub{color:#5a6472;font-size:13px} .banner{background:#f4f6f8;border-left:4px solid #14532d;
 padding:10px 14px;margin:14px 0;font-size:13.5px}
.notes{background:#fbf7ee;border-left:4px solid #b8860b;padding:8px 14px;margin:8px 0 10px;
 font-size:13.5px} .notes li{margin:2px 0}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 4px}
th{background:#14532d;color:#fff;text-align:left;padding:6px 8px;font-weight:600}
td{padding:6px 8px;border-bottom:1px solid #e3e7ec}
tr:nth-child(even) td{background:#f7f9fa}
.side{font-weight:700} .score{font-weight:700;text-align:right}
.ctx{color:#5a6472;font-size:12.5px;margin:2px 0 0} .dagger{color:#b8860b}
td.why{color:#3c4552;font-size:12.5px;min-width:230px}
.risk{display:block;margin-top:2px;color:#8a5a1a}
.foot{margin-top:28px;padding-top:10px;border-top:1px solid #e3e7ec;
 color:#5a6472;font-size:12.5px}
@media print{body{margin:8px auto} h2{page-break-after:avoid} table{page-break-inside:avoid}}
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _side(lean: Dict) -> str:
    return "YES" if lean.get("market") == "anytime_td" else str(lean.get("side", "")).upper()


def _rank_score(lean: Dict):
    return lean.get("ml_score") if lean.get("ml_score") is not None else lean.get("composite")


def _why(lean: Dict, game: Optional[Dict] = None) -> str:
    """The Why cell: concise model rationale, then the principal risk.

    Both come from :mod:`nflvalue.week_package` -- the same strings the
    markdown table, the canonical JSON and the ``leans`` row carry. The
    canonical payload normally attaches them before rendering; recomputing
    here is the belt-and-braces that makes a BLANK Why cell impossible, since
    an unexplained pick on a money-adjacent page is the failure mode this
    column exists to prevent.

    Nothing here is generated prose and nothing claims profitability: the
    rationale is arithmetic already on the lean, and the counter-case is the
    ledger's own label for the driver arguing against the chosen side.
    """
    reason = (lean.get("reason") or "").strip() or wpmod.rationale(lean)
    risk = (lean.get("risk") or "").strip() or wpmod.principal_risk(lean, game)
    cell = _e(reason)
    if risk:
        cell += f"<span class='risk'>{_e(risk)}</span>"
    return cell


def render_drop(payload: Dict, contexts: Optional[Dict] = None) -> str:
    season, week = payload.get("season"), payload.get("week")
    contexts = contexts or payload.get("contexts") or {}
    parts: List[str] = [
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>NFL Prop Leans — {season} Week {week}</title><style>{_CSS}</style></head><body>",
        f"<h1>NFL Prop Leans — {season} Week {week}</h1>",
        f"<div class='sub'>Generated {_e(payload.get('as_of'))} · clock {_e(payload.get('clock'))}"
        + ("" if payload.get("publish", True) else
           " · <b style='color:#a32d2d'>NOT PUBLISHED — data gate failed</b>") + "</div>",
        "<div class='banner'><b>Leans, not locks.</b> Model-ranked research on free data — "
        "variance is variance and any lean can lose. † marks a synthetic reference line "
        "(the player's own trailing mean), not a market price; edge exists only against real "
        "sportsbook lines. Every game shows its full screen count so the denominator is never "
        "hidden. Not financial advice. Gambling problem? <b>1-800-GAMBLER</b>.</div>",
    ]
    for g in payload.get("games", []):
        parts.append(f"<h2>{_e(g.get('matchup'))} <span class='sub'>· top "
                     f"{len(g.get('leans', []))} of {g.get('screened_n')} screened</span></h2>")
        if g.get("notes"):
            parts.append("<div class='notes'><b>Game notes</b> "
                         "<span class='sub'>(display-only — never scored)</span><ul>"
                         + "".join(f"<li>{_e(n)}</li>" for n in g["notes"]) + "</ul></div>")
        parts.append("<table><tr><th>Player</th><th>Market</th><th>Line</th><th>Side</th>"
                     "<th>Proj</th><th>Edge</th><th>Score</th><th>Why</th></tr>")
        for l in g.get("leans", []):
            dag = "" if l.get("line_source") == "odds_api" else "<span class='dagger'>†</span>"
            edge = (f"{l['edge']*100:+.1f}%" if l.get("edge") is not None
                    else "<span class='sub'>no_market</span>")
            parts.append(
                f"<tr><td><b>{_e(l.get('name'))}</b> <span class='sub'>{_e(l.get('pos'))} · "
                f"{_e(l.get('team'))}</span></td>"
                f"<td>{_e(str(l.get('market', '')).replace('_', ' '))}</td>"
                f"<td>{_e(l.get('line'))}{dag}</td>"
                f"<td class='side'>{_side(l)}</td><td>{_e(l.get('mean'))}</td>"
                f"<td>{edge}</td><td class='score'>{_e(_rank_score(l))}</td>"
                f"<td class='why'>{_why(l, g)}</td></tr>")
        parts.append("</table>")
        ctx = contexts.get(g.get("game_id")) or {}
        items = [f"<b>{_e(e.get('name'))}</b> — {_e(i)}"
                 for e in ctx.get("entries", []) for i in e.get("items", [])
                 if "no context flags" not in i]
        if items:
            parts.append("<div class='ctx'><b>Player context</b> (display-only — never scored): "
                         + " · ".join(items[:12]) + "</div>")
    parts.append("<div class='foot'>Deterministic projections · walk-forward features "
                 "(no leakage) · ML-assisted ordering retrained weekly · context and stories "
                 "are displayed, measured in the evidence ledger, and never move a number "
                 "until they earn it. Leans, not locks — 1-800-GAMBLER.</div></body></html>")
    return "".join(parts)


def write_drop(payload: Dict, contexts: Optional[Dict] = None,
               drops_dir: Optional[str] = None) -> str:
    d = drops_dir or DROPS_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"props_week_{payload.get('season')}_{payload.get('week')}"
                           + ("_t90" if payload.get("clock") == "t90" else "") + ".html")
    with open(path, "w") as f:
        f.write(render_drop(payload, contexts))
    return path
