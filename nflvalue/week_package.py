"""The canonical finalized weekly payload -- the one object every surface reads.

The initial product is **player-prop research leans** (PROP_SHORTLISTER_SPEC.md
§6), not the separate game-line ``top_bets`` product. It is published on two
clocks (PHASE1_HANDSOFF_DESIGN.md):

* **Wednesday** INITIALIZES the full-week package: every game of the slate.
* **T-90** PATCHES one game. A patch replaces that game and nothing else --
  every untouched Wednesday game, and every game an *earlier* T-90 already
  patched, must survive byte-for-byte. Patches accumulate; they never clobber.

Before this module each surface rebuilt its own view of "the week" from
whatever the current run happened to hold, so a one-game T-90 run produced a
one-game markdown file, a one-game HTML drop and a one-game dashboard payload
-- silently deleting the rest of the slate from the product. Now there is
exactly ONE payload:

    Markdown · HTML drop · dashboard input · Discord input  ->  merge(...)

and one place that decides what "the week" currently is.

Honesty invariants carried through the merge, not bolted onto each renderer:

* ``publish`` is AND-ed, never upgraded. A patch that passes its own freshness
  gate cannot publish a week whose Wednesday gate failed; the reasons union.
* Games are ordered by ``game_id`` -- deterministic across runs and merges.
* Every lean carries a non-empty deterministic ``reason`` (:func:`rationale`),
  computed once here so Markdown, HTML, JSON and the ``leans`` table can never
  disagree about why a lean is on the page.
* A synthetic reference line stays visibly distinct from a real sportsbook
  line at every layer; the risk note says so in words rather than leaving a
  reader to infer a market price that was never pulled.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Rationale -- ONE implementation, shared by every surface
# --------------------------------------------------------------------------- #
#: Never rendered as a rationale. A lean with nothing to say is a bug, and an
#: empty Why cell hides exactly the pick a reader most needs to distrust.
FALLBACK_REASON = "ranked by composite score (no component separated it)"


def rationale(lean: Dict) -> str:
    """The deterministic one-line "why" for a lean.

    Built only from numbers already on the lean -- the projection vs. the
    line, the model/market probability pair (only where a real price exists),
    the opponent-vs-position factor and the game-script sub-score. It invents
    no prose, claims no profitability, and returns the same string for the
    same lean on every run and in every renderer.
    """
    score_comps = lean.get("components") or {}          # composite breakdown
    proj_comps = lean.get("proj_components") or {}      # projection breakdown
    bits: List[str] = []
    z = score_comps.get("z")
    if z is not None:
        bits.append(f"proj {lean.get('mean')} vs line {lean.get('line')} (z={z:+.2f})")
    if lean.get("edge") is not None:
        bits.append(f"model p {score_comps.get('model_prob')} vs mkt "
                    f"{score_comps.get('market_prob')}")
    opp_factor = proj_comps.get("opp_factor")
    if opp_factor not in (None, 1.0):
        direction = "soft" if (opp_factor > 1.0) == (lean.get("side") == "over") else "tough"
        bits.append(f"opp-vs-pos {opp_factor} ({direction} matchup for this side)")
    gs = score_comps.get("script_sub")
    if gs is not None and abs(gs - 0.5) > 0.15:
        bits.append("game-script fit" if gs > 0.5 else "game-script headwind")
    return "; ".join(bits) if bits else FALLBACK_REASON


def principal_risk(lean: Dict, game: Optional[Dict] = None) -> str:
    """The counter-case in one clause: the strongest reason this lean is wrong.

    Sourced from facts already attached to the lean -- never generated prose,
    never a probability of profit. Preference order:

    1. the strongest driver pushing AGAINST the chosen side (explainability
       data, when the ledger produced it);
    2. a synthetic reference line -- there is no market price to be right about;
    3. a market the model itself flags low-confidence;
    4. a thin trailing sample;
    5. the selection denominator, which is always true and always relevant.
    """
    side = lean.get("side")
    against = "down" if side == "over" else "up"
    for d in lean.get("counter_drivers") or []:
        if d.get("direction") == against and d.get("label"):
            return f"counter-case: {d['label']} pushes the other way"

    # The explainability ledger already decomposes this lean into drivers and
    # knows which of them argue against the chosen side. Use its labels rather
    # than writing a second, unverified story about the same number. A lean
    # whose ledger will not build (no projection components) falls through to
    # the facts below -- it never gets an invented counter-case.
    try:
        from .explain import build_ledger
        opposing = build_ledger(lean).opposing(side)
        if opposing:
            worst = max(opposing, key=lambda c: abs(c.log_contribution or 0.0))
            return f"counter-case: {worst.label} pushes the other way"
    except Exception:  # noqa: BLE001 -- no ledger is not a reason to invent one
        pass

    if lean.get("line_source") != "odds_api":
        return ("counter-case: synthetic reference line (†) — no sportsbook price "
                "was pulled, so there is no market to be right about")
    if lean.get("market") == "anytime_td":
        return "counter-case: touchdown markets are high-variance (flagged low-confidence)"
    roll_games = lean.get("roll_games")
    try:
        if roll_games is not None and float(roll_games) < 5:
            return f"counter-case: thin trailing sample ({int(float(roll_games))} games)"
    except (TypeError, ValueError):
        pass
    screened = (game or {}).get("screened_n")
    if screened:
        return (f"counter-case: top of a {screened}-candidate screen — the more that "
                f"is screened, the more the top is partly luck")
    return "counter-case: variance is variance; a lean can lose for no reason at all"


# --------------------------------------------------------------------------- #
# Package shape
# --------------------------------------------------------------------------- #
def _game_id(game: Dict) -> str:
    return str(game.get("game_id") or "")


def order_games(games: List[Dict]) -> List[Dict]:
    """Deterministic slate order. Same input -> same output, every run."""
    return sorted(games or [], key=_game_id)


def stamp_games(games: List[Dict], clock: str, as_of: Optional[str]) -> List[Dict]:
    """Record WHICH clock produced each game, so a merged week can say which
    games are still on the Wednesday read and which a T-90 has refreshed."""
    for g in games or []:
        g["clock"] = clock
        if as_of:
            g["as_of"] = as_of
    return games


def attach_rationale(games: List[Dict]) -> List[Dict]:
    """Give every lean its non-empty deterministic reason + counter-case.

    Done once, on the canonical payload, so the Markdown table, the HTML Why
    cell, the JSON and the ``leans`` row are the same sentence by construction
    rather than by three implementations agreeing.
    """
    for g in games or []:
        for lean in g.get("leans") or []:
            reason = rationale(lean)
            lean["reason"] = reason or FALLBACK_REASON
            lean["risk"] = principal_risk(lean, g)
    return games


def finalize(payload: Dict, clock: Optional[str] = None) -> Dict:
    """Make a raw run result canonical: ordered, clock-stamped, fully reasoned."""
    payload["games"] = order_games(payload.get("games") or [])
    stamp_games(payload["games"], clock or payload.get("clock") or "wed",
                payload.get("as_of"))
    attach_rationale(payload["games"])
    return payload


# --------------------------------------------------------------------------- #
# Merge: a T-90 patch replaces its own games and nothing else
# --------------------------------------------------------------------------- #
def _union(*lists) -> List:
    out: List = []
    for items in lists:
        for x in items or []:
            if x not in out:
                out.append(x)
    return out


def merge(base: Optional[Dict], patch: Dict) -> Dict:
    """Fold a (usually one-game) patch into the standing weekly package.

    Games present in ``patch`` REPLACE their counterparts; every other game --
    Wednesday's, and any game an earlier patch already replaced -- is carried
    through untouched. The result is ordered by ``game_id``.

    ``publish`` is AND-ed and the reasons union: a patch can never publish a
    week whose Wednesday gate failed, and a failed patch gate stops the week.
    """
    patch_games = order_games(patch.get("games") or [])
    if not base or not base.get("games"):
        merged = dict(patch)
        merged["games"] = patch_games
        merged["patched_games"] = [_game_id(g) for g in patch_games]
        return merged

    if (base.get("season"), base.get("week")) != (patch.get("season"), patch.get("week")):
        # A package for a different week is not a base for this patch. Refuse
        # to merge rather than silently publish another week's slate.
        merged = dict(patch)
        merged["games"] = patch_games
        merged["patched_games"] = [_game_id(g) for g in patch_games]
        return merged

    by_id: Dict[str, Dict] = {_game_id(g): g for g in base.get("games") or []}
    for g in patch_games:
        by_id[_game_id(g)] = g

    contexts = dict(base.get("contexts") or {})
    contexts.update(patch.get("contexts") or {})

    voided_by_game = dict(base.get("voided_by_game") or {})
    for gid in [_game_id(g) for g in patch_games]:
        voided_by_game[gid] = [v for v in (patch.get("voided") or [])]

    merged = dict(base)
    merged.update({k: v for k, v in patch.items()
                   if k not in ("games", "contexts", "publish", "publish_reasons",
                                "voided_by_game")})
    merged["games"] = order_games(list(by_id.values()))
    merged["contexts"] = contexts
    merged["voided_by_game"] = voided_by_game
    merged["patched_games"] = _union(base.get("patched_games"),
                                     [_game_id(g) for g in patch_games])
    merged["publish"] = bool(base.get("publish", True)) and bool(patch.get("publish", True))
    merged["publish_reasons"] = _union(base.get("publish_reasons"),
                                       patch.get("publish_reasons"))
    # The screen denominator is a published honesty number, so the merged
    # week's total must be the merged week's total -- not the one-game count
    # the patch happened to carry. Per game, ``screened_n`` is what a reader
    # actually sees ("5 of N"); the week total is their sum by construction.
    screened = [g.get("screened_n") for g in merged["games"]]
    if screened and all(isinstance(n, (int, float)) for n in screened):
        merged["n_candidates"] = int(sum(screened))
    return merged


# --------------------------------------------------------------------------- #
# Real prop lines already snapshotted into the warehouse
# --------------------------------------------------------------------------- #
def latest_line_ts(conn, game_id: str) -> Optional[str]:
    """Timestamp of the freshest ``lines`` snapshot for one game, or None."""
    try:
        row = conn.execute("SELECT MAX(ts) FROM lines WHERE game_id=?",
                           (game_id,)).fetchone()
    except Exception:  # noqa: BLE001 -- an old DB may predate the table
        return None
    return row[0] if row and row[0] else None


def latest_real_prop_lines(conn, game_id: str, players):
    """The freshest REAL sportsbook lines for one game, as a prop-lines frame.

    T-90 is the clock closest to kickoff, so it is where a real price matters
    most -- and by then a resnap has usually already written one into
    ``lines``. Re-enumerating without it would quietly regress the final read
    to a synthetic reference line while the Wednesday run had a real one.

    Returns None when this game has no snapshot (or none of its rows match a
    projected player): the caller then keeps synthetic/``no_market`` labelling
    exactly as it is. A missing market is never blended or guessed.
    """
    ts = latest_line_ts(conn, game_id)
    if not ts:
        return None
    from . import db as dbmod
    from .sources import oddsapi_props as oapmod

    snap = dbmod.query_df(conn, "SELECT * FROM lines WHERE game_id=? AND ts=?",
                          (game_id, ts))
    if snap.empty:
        return None
    pool = players.rename(columns={"player_name": "name"}) \
        if "player_name" in getattr(players, "columns", []) else players
    rows = oapmod.match_player_ids(snap.to_dict("records"), pool)
    frame = oapmod.to_prop_lines_frame(rows)
    if frame is None or frame.empty:
        return None
    frame.attrs["ts"] = ts
    return frame


# --------------------------------------------------------------------------- #
# Persistence of the canonical payload
# --------------------------------------------------------------------------- #
def load(path: str) -> Optional[Dict]:
    """The standing weekly package, or None when this week has none yet."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("games") is not None else None


def load_for_week(path: str, season: int, week: int) -> Optional[Dict]:
    """``load``, but only if the stored package is THIS (season, week)."""
    pkg = load(path)
    if not pkg:
        return None
    try:
        if int(pkg.get("season")) != int(season) or int(pkg.get("week")) != int(week):
            return None
    except (TypeError, ValueError):
        return None
    return pkg


def save(path: str, payload: Dict) -> str:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    body = {k: v for k, v in payload.items() if k != "markdown"}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1, default=str)
    return path
