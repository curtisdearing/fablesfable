"""Weekly props report: markdown for humans, JSON for the dashboard, rows for
the forward log. PROP_SHORTLISTER_SPEC.md §6.

``reports/props_week_{S}_{W}.md`` -- per game: the top-5 leans (player ·
market · line · side · projection · confidence · edge-or-``no_market`` ·
composite · one-line reason · "5 of N screened"), then that game's Context
block, labeled "context only -- not scored". The same rows are:

  * written to ``data/weekly_props.json`` (dashboard Props tab source),
  * upserted into the ``leans`` table (the forward log CLV reads),
  * and the markdown doubles as the RAG context pack (Phase 5).

Every user-facing artifact keeps the honest framing: leans, not locks;
synthetic lines clearly tagged; screen counts shown; 1-800-GAMBLER.

CLI:  python3 -m nflvalue.report --season 2023 --week 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Dict, List, Optional

import pandas as pd

from . import candidates as candmod
from . import config as cfgmod
from . import db as dbmod
from . import shortlist as slmod

REPORTS_DIR = os.path.join(cfgmod.ROOT, "reports")
WEEKLY_PROPS_JSON = os.path.join(cfgmod.DATA_DIR, "weekly_props.json")

DISCLAIMER = (
    "**Leans, not locks.** These are model-ranked research leans on FREE data — "
    "variance is variance, and a lean can lose for no reason at all. Nothing here is "
    "financial advice. If you or someone you know has a gambling problem, call "
    "**1-800-GAMBLER**."
)


def _fmt_edge(lean: Dict) -> str:
    if lean.get("no_market") or lean.get("edge") is None:
        return "`no_market`"
    return f"{lean['edge']*100:+.1f}%"


def _fmt_line(lean: Dict) -> str:
    tag = "†" if lean.get("line_source") == "synthetic_trailing_mean" else ""
    return f"{lean.get('line')}{tag}"


def _one_line_reason(lean: Dict) -> str:
    score_comps = lean.get("components") or {}          # composite breakdown
    proj_comps = lean.get("proj_components") or {}      # projection breakdown
    bits: List[str] = []
    z = score_comps.get("z")
    if z is not None:
        bits.append(f"proj {lean.get('mean')} vs line {lean.get('line')} (z={z:+.2f})")
    if lean.get("edge") is not None:
        bits.append(f"model p {score_comps.get('model_prob')} vs mkt {score_comps.get('market_prob')}")
    opp_factor = proj_comps.get("opp_factor")
    if opp_factor not in (None, 1.0):
        direction = "soft" if (opp_factor > 1.0) == (lean.get("side") == "over") else "tough"
        bits.append(f"opp-vs-pos {opp_factor} ({direction} matchup for this side)")
    gs = score_comps.get("script_sub")
    if gs is not None and abs(gs - 0.5) > 0.15:
        bits.append("game-script fit" if gs > 0.5 else "game-script headwind")
    return "; ".join(bits) if bits else "ranked by composite"


def _side_label(lean: Dict) -> str:
    if lean.get("market") == "anytime_td":
        return "YES"
    return str(lean.get("side", "")).upper()


def _lean_row_md(lean: Dict) -> str:
    return ("| {name} ({pos}, {team}) | {market} | {line} | **{side}** | {mean} | "
            "{conf:.2f} | {edge} | **{comp:.1f}** | {reason} |").format(
        name=lean.get("name"), pos=lean.get("pos"), team=lean.get("team"),
        market=str(lean.get("market")).replace("_", " "),
        line=_fmt_line(lean), side=_side_label(lean),
        mean=lean.get("mean"), conf=lean.get("confidence", 0.0),
        edge=_fmt_edge(lean), comp=lean.get("composite", 0.0),
        reason=_one_line_reason(lean),
    )


def render_markdown(season: int, week: int, games: List[Dict],
                    contexts: Dict[str, Dict], as_of: str, clock: str,
                    publish: bool = True, publish_reasons: Optional[List[str]] = None,
                    line_note: Optional[str] = None) -> str:
    n_real = sum(1 for g in games for l in g["leans"] if l.get("line_source") == "odds_api")
    n_synth = sum(1 for g in games for l in g["leans"] if l.get("line_source") != "odds_api")
    lines = [
        f"# NFL Prop Leans — {season} Week {week}",
        "",
        f"*Generated {as_of} · clock: **{clock}** · "
        f"{'PUBLISHED' if publish else '**NOT PUBLISHED — data gate failed**'}*",
        "",
        DISCLAIMER,
        "",
    ]
    if not publish and publish_reasons:
        lines += ["> **Publish gate failed:** " + "; ".join(publish_reasons), ""]
    lines += [
        f"Lines: {n_real} from live sportsbook pulls, {n_synth} synthetic (†). "
        "† = the player's own trailing mean, floor+0.5 — a modeling reference, NOT a "
        "market price; edge is only computed against real prop prices (`no_market` otherwise)."
        + (f" {line_note}" if line_note else ""),
        "",
        "Every top-5 shows **\"5 of N screened\"** — N is how many candidate props were "
        "actually scored for that game. The bigger the N, the more the top of any ranked "
        "list is partly luck. That number stays visible so we never fool ourselves.",
        "",
    ]
    for g in games:
        lines += [
            f"## {g['matchup']}  ·  top {len(g['leans'])} of {g['screened_n']} screened",
            "",
            "| Player | Market | Line | Side | Proj | Conf | Edge | Composite | Why |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [_lean_row_md(l) for l in g["leans"]]
        ctx = contexts.get(g["game_id"])
        lines.append("")
        if ctx:
            lines.append(f"**Context — {ctx['label']}**")
            lines.append("")
            for e in ctx["entries"]:
                for item in e["items"]:
                    lines.append(f"- **{e['name']}** — {item}")
            lines.append("")
    lines += ["---", "",
              f"*Deterministic model · walk-forward features (no leakage) · context is "
              f"display-only · {DISCLAIMER.split('**')[1]}. 1-800-GAMBLER.*", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def persist_leans(conn, season: int, week: int, clock: str, games: List[Dict],
                  as_of: str, status: str = "active") -> int:
    rows = []
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for g in games:
        for l in g["leans"]:
            prices = l.get("prices") or {}
            rows.append({
                "season": season, "week": week, "clock": clock,
                "game_id": g["game_id"], "player_id": l.get("player_id"),
                "name": l.get("name"), "market": l.get("market"),
                "side": l.get("side"), "line": l.get("line"),
                "line_source": l.get("line_source"),
                "price": prices.get("over") if l.get("side") == "over" else prices.get("under"),
                "book": prices.get("book"),
                "mean": l.get("mean"), "sd": l.get("sd"),
                "p_side": (l.get("p_over") if l.get("side") == "over" else l.get("p_under")),
                "composite": l.get("composite"), "edge": l.get("edge"),
                "confidence_comp": l.get("confidence"), "matchup_comp": l.get("matchup"),
                "screened_n": g.get("screened_n"), "reason": _one_line_reason(l),
                "status": status, "void_reason": None, "as_of": as_of, "created_at": now,
            })
    if not rows:
        return 0
    return dbmod.upsert(conn, "leans", rows,
                        ["season", "week", "clock", "game_id", "player_id", "market"])


def load_manual_notes(conn, season: int, week: int) -> List[Dict]:
    try:
        df = dbmod.query_df(conn, "SELECT * FROM manual_notes WHERE season=? AND week=?",
                            (season, week))
        return df.to_dict("records")
    except Exception:  # noqa: BLE001 -- table may not exist in an old DB
        return []


# --------------------------------------------------------------------------- #
# End-to-end generation
# --------------------------------------------------------------------------- #
def generate(season: int, week: int, inputs: Optional[candmod.WeekInputs] = None,
             prop_lines: Optional[pd.DataFrame] = None,
             synthesis_by_game: Optional[Dict[str, Dict]] = None,
             availability: Optional[Dict[str, Dict]] = None,
             clock: str = "wed", mode: str = "historical",
             publish: bool = True, publish_reasons: Optional[List[str]] = None,
             weights: Optional[Dict] = None, params: Optional[Dict] = None,
             write_files: bool = True, persist: bool = True,
             line_note: Optional[str] = None) -> Dict:
    """Candidates -> composite -> shortlist -> context -> markdown/JSON/DB."""
    cfg = cfgmod.load_config()
    weights = weights or (cfg.get("composite") or {}).get("weights")
    params = params or (cfg.get("composite") or {}).get("params")
    top_n = int((cfg.get("shortlist") or {}).get("top_n", slmod.DEFAULT_TOP_N))
    max_pp = int((cfg.get("shortlist") or {}).get("max_per_player", slmod.DEFAULT_MAX_PER_PLAYER))
    min_usage = (cfg.get("candidates") or {}).get("min_usage")

    inputs = inputs or candmod.build_week_inputs()
    as_of = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cands = candmod.enumerate_candidates(
        season, week, inputs=inputs, min_usage=min_usage, prop_lines=prop_lines,
        roster_mode="as_played" if mode == "historical" else "carry_forward",
    )
    games = slmod.shortlist_week(cands, weights=weights, params=params,
                                 top_n=top_n, max_per_player=max_pp)

    conn = dbmod.connect()
    notes = load_manual_notes(conn, season, week)
    contexts = {}
    for g in games:
        syn = (synthesis_by_game or {}).get(g["game_id"])
        contexts[g["game_id"]] = slmod.build_context_panel(
            g, synthesis_output=syn, manual_notes=notes,
            availability=availability, mode=mode)

    md = render_markdown(season, week, games, contexts, as_of, clock,
                         publish=publish, publish_reasons=publish_reasons,
                         line_note=line_note)

    result = {"season": season, "week": week, "clock": clock, "as_of": as_of,
              "publish": publish, "publish_reasons": publish_reasons or [],
              "mode": mode, "games": games, "contexts": contexts,
              "n_candidates": int(len(cands)), "markdown": md}

    if write_files:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        md_path = os.path.join(REPORTS_DIR, f"props_week_{season}_{week}.md")
        with open(md_path, "w") as f:
            f.write(md)
        result["md_path"] = md_path
        json_payload = {k: v for k, v in result.items() if k != "markdown"}
        cfgmod.save_json(WEEKLY_PROPS_JSON, json_payload)
        result["json_path"] = WEEKLY_PROPS_JSON
    if persist:
        persist_leans(conn, season, week, clock, games, as_of)
    conn.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the weekly props leans report.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--clock", choices=["wed", "t90"], default="wed")
    ap.add_argument("--mode", choices=["historical", "live"], default="historical")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()
    res = generate(args.season, args.week, clock=args.clock, mode=args.mode,
                   persist=not args.no_persist)
    print(f"Wrote {res.get('md_path')}  ({len(res['games'])} games, "
          f"{res['n_candidates']} candidates screened)")


if __name__ == "__main__":
    main()
