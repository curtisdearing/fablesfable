"""Phase 8.4-8.6 -- card payloads, trend series, and the honest record.

Assembles everything the dashboard needs into one JSON artifact. This module
is a COMPOSER: it calls :mod:`explain`, :mod:`evidence` and
:mod:`explain_render` and reshapes their output for display. It computes no new
projection value and re-derives nothing.

Three payloads
--------------
* ``cards``   (8.4) one per published lean: the ledger with display-ready
  magnitude bars, the evidence chips, the counter-case, and the
  "what would change our mind" line.
* ``trends``  (8.5) per pick, the player's own prior-week series with an
  explicit AS-OF BOUNDARY, so a reader can see the model only ever used weeks
  strictly before the one it predicted.
* ``record``  (8.6) the honest performance panel: synthetic-line accuracy
  labelled as synthetic, real-line record where one exists, resolved CLV shown
  as ``n=0`` rather than blank, calibration, and the kill-check verdict in
  plain language.

Display rules that are honesty rules
------------------------------------
* Magnitude is encoded by BAR LENGTH and a NUMERIC LABEL. Colour is redundant,
  never load-bearing -- the card must survive a greyscale screenshot and a
  colourblind reader.
* ``thin`` and ``unproven`` chips are rendered at the same weight as
  ``strong``. A reader who skims past uncertainty is the failure mode.
* A synthetic-line card carries no edge field at all. Not "edge: n/a" -- the
  field is absent, because a rendered field invites a reader to look for a
  number.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

from . import evidence as evmod
from . import explain as explmod
from . import explain_render as rn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS_PATH = os.path.join(ROOT, "data", "explain_cards.json")

#: Bars are drawn from the log contribution, which is the order-independent
#: measure of how much a driver moved the number. Capped so one huge factor
#: cannot flatten every other bar into invisibility.
_BAR_CAP = 0.75


def _bar(contribution: Dict) -> Dict:
    """Bar geometry for one driver.

    Length carries magnitude; the sign is reported by a glyph AND a word, never
    by colour alone, so the card survives greyscale and colourblind readers.

    WHAT THE BAR MEASURES depends on the driver, and getting this wrong
    over-claims badly:

    * A **tilt** has a meaningful neutral of 1.0, so its bar is log(multiplier)
      -- how far it moved the number away from "no effect".
    * A **level** is largely a unit conversion. A 0.568 catch rate turns
      targets into receptions, and its raw log contribution is enormous (-0.57)
      purely because of that conversion. Drawing that as a 75%-long bar would
      tell the reader efficiency was doing three-quarters of the work, when the
      arithmetic is just changing units. So a level's bar measures its
      DEVIATION FROM THE MEASURED REFERENCE, log(value / reference) -- the part
      that is actually an argument. With no reference there is no defensible
      magnitude, so no bar is drawn.
    """
    kind = contribution.get("kind", "tilt")
    mult = contribution.get("multiplier")
    if mult is None:
        return {"pct": 0.0, "glyph": "", "word": "", "basis": "baseline"}

    if kind == "level":
        ref = contribution.get("reference")
        if not ref or ref <= 0 or mult <= 0:
            return {"pct": 0.0, "glyph": "", "word": "",
                    "basis": "no reference — magnitude not claimed"}
        signed = math.log(mult / ref)
        basis = "deviation from the position average"
    else:
        signed = contribution.get("log_contribution")
        if signed is None:
            return {"pct": 0.0, "glyph": "", "word": "", "basis": "unavailable"}
        basis = "distance from no-effect (1.0)"

    magnitude = min(abs(signed), _BAR_CAP) / _BAR_CAP
    return {
        "pct": round(magnitude * 100.0, 1),
        "glyph": "▲" if signed > 0 else ("▼" if signed < 0 else "="),
        "word": "raises" if signed > 0 else ("lowers" if signed < 0 else "no change"),
        "basis": basis,
    }


def build_card(lean: Dict, game: Dict, refs: Optional[Dict] = None,
               trends: Optional[Dict] = None) -> Dict:
    """One pick, fully explained and display-ready."""
    ledger = explmod.build_ledger(lean, refs=refs)
    led = evmod.attach_evidence(ledger.to_dict(), refs)
    case = rn.render_case(led, screened=len(game.get("leans") or []),
                          screened_n=game.get("screened_n"))

    drivers = []
    for c in led["contributions"]:
        ev = c.get("evidence") or {}
        drivers.append({
            "key": c["key"],
            "label": c["label"],
            "kind": c["kind"],
            "direction": c["direction"],
            "multiplier_label": (rn.fmt(c["multiplier"], 3)
                                 if c["multiplier"] is not None else None),
            "value_label": rn.fmt(c["value_after"], 1),
            "unit": c["unit"],
            "delta_label": (rn.fmt(c["delta"], 1) if c.get("delta") is not None
                            else None),
            "bar": _bar(c),
            "reference_label": (rn.fmt(c["reference"], 3)
                                if c.get("reference") is not None else None),
            "evidence": {
                "grade": ev.get("grade"),
                "n_label": (f"n={ev['n']:,}" if ev.get("n") else "n not published"),
                "ci_label": _ci_label(ev),
                "claim": ev.get("claim"),
                "note": ev.get("note"),
                "source": ev.get("source"),
            },
            "provenance": c.get("provenance"),
        })

    card = {
        "player_id": lean.get("player_id"),
        "name": lean.get("name"),
        "pos": lean.get("pos"),
        "team": lean.get("team"),
        "game_id": lean.get("game_id"),
        "matchup": game.get("matchup"),
        "market": lean.get("market"),
        "side": lean.get("side"),
        "line_label": rn.fmt(lean.get("line"), 1),
        "line_source": led["line_source"],
        "is_synthetic_line": led["is_synthetic_line"],
        "projection_label": rn.fmt(led["projected_mean"], 1),
        "screened_label": f"{len(game.get('leans') or [])} of {game.get('screened_n')}",
        "reconciliation": led["reconciliation"],
        "granularity": led["granularity"],
        "drivers": drivers,
        "not_applied": led["not_applied"],
        "prose": case["blocks"],
        "counter_case_count": len([d for d in drivers
                                   if d["direction"] == ("down" if lean.get("side") == "over"
                                                         else "up")]),
        "weakest_grade": _weakest_grade(drivers),
        "measured_zero": led.get("measured_zero"),
        "trend": (trends or {}).get(f"{lean.get('player_id')}|{lean.get('market')}"),
    }

    # A synthetic reference line gets NO edge key at all. An "edge: n/a" field
    # still invites the reader to hunt for a number; absence does not.
    if not led["is_synthetic_line"]:
        edge = ledger.edge(lean)
        if edge is not None:
            card["edge_label"] = rn.pct(edge, 1)
    return card


def _ci_label(ev: Dict) -> str:
    if ev.get("interval"):
        lo, hi = ev["interval"]
        return f"95% CI {rn.fmt(lo, 3)}–{rn.fmt(hi, 3)}"
    if ev.get("interval_status") == "not_recomputed":
        return "CI not recomputed"
    if ev.get("interval_status") == "not_applicable":
        return "observed level, not an effect size"
    return "no interval"


_GRADE_ORDER = {"unproven": 0, "thin": 1, "moderate": 2, "strong": 3}


def _weakest_grade(drivers: List[Dict]) -> str:
    """The card's headline honesty signal: a case is only as strong as its
    weakest load-bearing driver, so that is what gets surfaced at the top
    rather than an average that would wash the weak ones out."""
    grades = [d["evidence"]["grade"] for d in drivers
              if d["evidence"].get("grade") and d["direction"] != "baseline"]
    if not grades:
        return "unproven"
    return min(grades, key=lambda g: _GRADE_ORDER.get(g, 0))


# --------------------------------------------------------------------------- #
# 8.5 Trend series
# --------------------------------------------------------------------------- #
def build_trends(frame, season: int, week: int,
                 keys: Optional[List] = None) -> Dict:
    """Per (player, market) prior-week series, with the as-of boundary.

    Everything returned is STRICTLY BEFORE (season, week). The boundary is
    carried in the payload so the chart can draw it: the point of the trend
    view is to let a reader confirm with their own eyes that the model never
    saw the week it predicted.
    """
    if frame is None or getattr(frame, "empty", True):
        return {}
    prior = frame[(frame["season"] < season)
                  | ((frame["season"] == season) & (frame["week"] < week))]
    if prior.empty:
        return {}
    wanted = set(keys or [])
    out: Dict[str, Dict] = {}
    for (pid, market), grp in prior.groupby(["player_id", "market"]):
        key = f"{pid}|{market}"
        if wanted and key not in wanted:
            continue
        grp = grp.sort_values(["season", "week"]).tail(12)
        out[key] = {
            "as_of_boundary": {"season": int(season), "week": int(week)},
            "note": ("every point is a week STRICTLY BEFORE the predicted "
                     "week; the boundary marks where the model's information "
                     "stops"),
            "points": [
                {"season": int(r.season), "week": int(r.week),
                 "volume": _safe(r.proj_volume),
                 "efficiency": _safe(r.proj_efficiency),
                 "opp_factor": _safe(r.opp_factor)}
                for r in grp.itertuples(index=False)
            ],
        }
    return out


def _safe(x) -> Optional[float]:
    try:
        v = float(x)
        return round(v, 4) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 8.6 The honest record
# --------------------------------------------------------------------------- #
def build_record(conn=None, eval_results: Optional[Dict] = None) -> Dict:
    """The performance panel. Every figure labelled by what it can support.

    The central honesty problem: the only accuracy numbers this project has in
    volume are measured at SYNTHETIC reference lines, and the accuracy protocol
    forbids deriving any profit, ROI, market-edge or CLV claim from them. So
    they are labelled as what they are -- a trend and regression check -- and
    the market scorecard is reported separately, currently at n=0.

    ``n=0`` is displayed as ``n=0``. A blank would read as "nothing to worry
    about"; a zero reads as "no evidence has been collected yet", which is the
    true state.
    """
    record: Dict = {
        "synthetic_line_accuracy": _synthetic_accuracy(eval_results),
        "real_line_record": {
            "n": 0,
            "status": "none_collected",
            "label": "n=0 — no picks have yet been graded against a real "
                     "bookmaker line",
            "caveat": "This, not the synthetic-line figure, is the accuracy "
                      "number that would matter.",
        },
        "clv": {
            "n_resolved": 0,
            "label": "n=0 resolved",
            "status": "insufficient_sample",
            "caveat": "Closing-line value is the only edge proof this project "
                      "accepts. None has been collected.",
        },
        "kill_check": {
            "verdict": "INSUFFICIENT_SAMPLE",
            "plain_english": "Not enough resolved picks to say anything either "
                             "way. This is not a passing grade — it means no "
                             "conclusion has been earned yet.",
        },
        "calibration": {"status": "not_computed", "label": "not computed"},
        "headline": (
            "This tool has NOT demonstrated an edge. Real-closing-line "
            "backtests lose, resolved CLV is n=0, and the kill-check exists to "
            "return NO-GO. Everything below is research, not a track record."),
    }

    if conn is not None:
        try:
            from . import killcheck
            kc = killcheck.report(conn)
            record["kill_check"] = {
                "verdict": kc.get("verdict"),
                "detail": kc.get("detail"),
                "plain_english": _kill_check_plain(kc),
                "leans_logged": kc.get("leans_logged"),
                "min_sample": kc.get("min_sample"),
            }
            record["clv"] = {
                "n_resolved": kc.get("n", 0),
                "label": f"n={kc.get('n', 0)} resolved",
                "lifetime_mean": kc.get("lifetime_mean"),
                "positive_rate": kc.get("positive_rate"),
                "status": ("insufficient_sample"
                           if (kc.get("n") or 0) < (kc.get("min_sample") or 150)
                           else "measured"),
                "caveat": "Closing-line value is the only edge proof this "
                          "project accepts.",
            }
        except Exception as exc:  # noqa: BLE001 -- panel degrades loudly
            record["kill_check"] = {
                "verdict": "UNAVAILABLE",
                "plain_english": f"The kill-check could not be read ({exc}). "
                                 f"Treat this as unknown, not as passing.",
            }
    return record


def _synthetic_accuracy(eval_results: Optional[Dict]) -> Dict:
    base = {
        "status": "unavailable",
        "label": "not available",
        "what_it_is": (
            "Hit rate measured against SYNTHETIC reference lines (the player's "
            "own trailing mean), not bookmaker prices."),
        "what_it_is_not": (
            "It is NOT evidence of profitability. The synthetic line's "
            "over/under split is a known artifact, and no ROI, edge or CLV "
            "claim may be derived from it."),
    }
    if not eval_results:
        return base
    seasons = (eval_results or {}).get("seasons") or {}
    rows = []
    for season, blob in sorted(seasons.items()):
        models = (blob or {}).get("models") or {}
        gbdt = (models.get("gbdt") or {}).get("leans") or {}
        if gbdt.get("n"):
            wins = int(round(float(gbdt["hit_rate"]) * int(gbdt["n"])))
            rows.append({
                "season": season,
                "n": int(gbdt["n"]),
                "hit_rate": gbdt.get("hit_rate"),
                "ci": evmod.wilson_interval(wins, int(gbdt["n"])),
                "label": f"{gbdt['hit_rate']:.1%} at n={int(gbdt['n']):,}",
            })
    if not rows:
        return base
    base.update(status="measured", seasons=rows,
                label=f"{len(rows)} season(s) of walk-forward replay")
    return base


def _kill_check_plain(kc: Dict) -> str:
    verdict = kc.get("verdict")
    n, need = kc.get("n", 0), kc.get("min_sample", 150)
    if verdict == "INSUFFICIENT_SAMPLE":
        return (f"Not enough resolved picks to say anything either way "
                f"({n} of {need}). This is not a passing grade — it means no "
                f"conclusion has been earned yet.")
    if verdict == "NO_GO":
        return ("The pre-committed kill criterion has been met: these picks do "
                "not beat the closing line. The agreed response is to treat "
                "this as a research tool and stop staking.")
    if verdict == "GO":
        return ("Resolved picks have beaten the closing line so far. That is "
                "consistent with an edge, not proof of one, and the sample is "
                "still small relative to what proving an edge requires.")
    return "The kill-check verdict could not be determined. Treat as unknown."


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def build_payload(weekly_props: Dict, refs: Optional[Dict] = None,
                  frame=None, conn=None,
                  eval_results: Optional[Dict] = None) -> Dict:
    """Full explainability payload for a week."""
    refs = refs if refs is not None else evmod.load_reference_levels()
    games = weekly_props.get("games") or []
    season = weekly_props.get("season")
    week = weekly_props.get("week")

    keys = [f"{l.get('player_id')}|{l.get('market')}"
            for g in games for l in (g.get("leans") or [])]
    trends = {}
    if frame is not None and season and week:
        try:
            trends = build_trends(frame, int(season), int(week), keys)
        except Exception as exc:  # noqa: BLE001 -- trends are enhancement
            trends = {"_error": str(exc)}

    cards, failures = [], []
    for game in games:
        for lean in game.get("leans") or []:
            try:
                cards.append(build_card(lean, game, refs=refs, trends=trends))
            except (explmod.LedgerReconciliationError, rn.VocabularyViolation,
                    rn.UntracedSentence) as exc:
                # FAIL VISIBLY. A pick whose ledger will not reconcile, or
                # whose copy breaks the vocabulary contract, must not be
                # silently dropped from the display -- that would hide exactly
                # the pick a reader most needs to distrust.
                failures.append({"player": lean.get("name"),
                                 "market": lean.get("market"),
                                 "error": f"{type(exc).__name__}: {exc}"})

    return {
        "season": season,
        "week": week,
        "as_of": weekly_props.get("as_of"),
        "cards": cards,
        "unexplainable": failures,
        "record": build_record(conn, eval_results),
        "disclaimer": (
            "Leans, not locks. Model-ranked research on free data; variance is "
            "variance. Not financial advice. Gambling problem? 1-800-GAMBLER."),
    }


def write_payload(payload: Dict, path: str = CARDS_PATH) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str, sort_keys=True)
    return path
