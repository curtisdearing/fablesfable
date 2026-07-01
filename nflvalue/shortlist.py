"""Per-game top-5 leans + a context panel that CANNOT touch the ranking.

Rank a game's candidates by ``composite.score_candidate``, take the top 5,
and record the selection-honesty denominator: ``screened = "5 of N"`` where N
is the true number of candidates actually scored for that game (premortem:
the more you screen, the more the top of the list is noise -- never hide N).

The context panel (PROP_SHORTLISTER_SPEC.md §4) rides ALONGSIDE the leans:
injury/availability status, synthesis ``context_notes``/``personal_context``,
and ``manual_notes`` rows. It is assembled AFTER scoring, from a ranking that
never saw it -- ``score_candidate`` does not even accept context arguments --
and every panel is labeled "Context only -- not part of the composite score."
tests/test_shortlist.py proves score equality with and without context.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .composite import score_candidate

DEFAULT_TOP_N = 5
DEFAULT_MAX_PER_PLAYER = 2   # correlated markets (rec yds + receptions) pile up otherwise

CONTEXT_LABEL = "Context only — not part of the composite score."


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def rank_game(cands: List[Dict], weights: Optional[Dict] = None,
              params: Optional[Dict] = None, top_n: int = DEFAULT_TOP_N,
              max_per_player: int = DEFAULT_MAX_PER_PLAYER) -> Dict:
    """Rank one game's candidates -> its shortlist.

    Returns::

        {"game_id", "matchup", "screened": "5 of N", "screened_n": N,
         "leans": [candidate + score fields, ...]}   # len <= top_n

    Deterministic: composite desc, then (player_id, market) as an absolute
    tie-break so equal scores can never reorder between runs.
    """
    if not cands:
        return {"game_id": None, "matchup": None, "screened": "0 of 0",
                "screened_n": 0, "leans": []}

    scored = []
    for c in cands:
        s = score_candidate(c, weights=weights, params=params)
        row = dict(c)
        # keep the projection engine's component breakdown under its own key --
        # score_candidate also returns a "components" dict (the score breakdown)
        row["proj_components"] = row.pop("components", None)
        row.update(s)
        scored.append(row)
    n_screened = len(scored)

    scored.sort(key=lambda r: (-r["composite"], str(r["player_id"]), r["market"]))

    leans, per_player = [], {}
    for r in scored:
        pid = r["player_id"]
        if per_player.get(pid, 0) >= max_per_player:
            continue
        leans.append(r)
        per_player[pid] = per_player.get(pid, 0) + 1
        if len(leans) >= top_n:
            break

    top = len(leans)
    return {
        "game_id": cands[0].get("game_id"),
        "matchup": cands[0].get("matchup"),
        "screened": f"{top} of {n_screened}",
        "screened_n": n_screened,
        "leans": leans,
    }


def shortlist_week(candidates_df: pd.DataFrame, weights: Optional[Dict] = None,
                   params: Optional[Dict] = None, top_n: int = DEFAULT_TOP_N,
                   max_per_player: int = DEFAULT_MAX_PER_PLAYER) -> List[Dict]:
    """Rank every game of the week. Games ordered by game_id (deterministic)."""
    out = []
    if candidates_df is None or candidates_df.empty:
        return out
    for game_id, grp in candidates_df.groupby("game_id", sort=True):
        cands = grp.to_dict("records")
        out.append(rank_game(cands, weights=weights, params=params,
                             top_n=top_n, max_per_player=max_per_player))
    return out


# --------------------------------------------------------------------------- #
# Context panel (display-only, by construction)
# --------------------------------------------------------------------------- #
def build_context_panel(game_shortlist: Dict,
                        synthesis_output: Optional[Dict] = None,
                        manual_notes: Optional[List[Dict]] = None,
                        availability: Optional[Dict[str, Dict]] = None,
                        mode: str = "live") -> Dict:
    """Assemble the per-game context block AFTER ranking.

    ``synthesis_output``: a ``nflvalue.synthesis`` OUTPUT dict (§3 contract)
    covering this game's players -- its ``context_notes``, status, divergence
    and reallocation flags are surfaced here, display-only.
    ``manual_notes``: rows from the ``manual_notes`` table for this
    (season, week), already filtered by caller.
    ``availability``: {player_id: {...}} from availability.resolve_statuses.
    ``mode="historical"``: no live feeds exist for a past week -- the panel
    says so honestly instead of pretending.

    Returns {"label", "mode", "entries": [{player_id, name, items: [...]}]}
    and NEVER feeds anything back into scores (the leans are already final
    when this runs).
    """
    entries: List[Dict] = []
    syn_by_pid: Dict[str, Dict] = {}
    for sp in (synthesis_output or {}).get("players", []) or []:
        syn_by_pid.setdefault(sp.get("player_id"), sp)

    notes_by_ref: Dict[str, List[Dict]] = {}
    for n in manual_notes or []:
        notes_by_ref.setdefault(str(n.get("ref")), []).append(n)

    for lean in game_shortlist.get("leans", []):
        pid, name = lean.get("player_id"), lean.get("name")
        items: List[str] = []

        if availability and pid in availability:
            a = availability[pid]
            items.append(f"availability: {a.get('status')} ({a.get('status_raw') or 'no listing'}; "
                         f"src {a.get('source')})")
        sp = syn_by_pid.get(pid)
        if sp:
            if sp.get("status") and sp["status"] != "OK":
                items.append(f"synthesis status: {sp['status']}")
            if sp.get("divergence_flag"):
                items.append("fantasy cross-check divergence (see synthesis flags)")
            if sp.get("needs_reallocation"):
                items.append("usage reallocation pending (teammate ruled out)")
            for note in sp.get("context_notes", []):
                items.append(f"note ({note.get('source')}): {note.get('text')}")
        for n in notes_by_ref.get(str(pid), []) + notes_by_ref.get(str(lean.get("team")), []):
            items.append(f"manual note [{n.get('tag')}]: {n.get('note')}")

        if not items:
            items.append("no context flags" if mode == "live"
                         else "historical run — live injury/news feeds not applicable")
        entries.append({"player_id": pid, "name": name, "items": items})

    return {"label": CONTEXT_LABEL, "mode": mode, "entries": entries}
