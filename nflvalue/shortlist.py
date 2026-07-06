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

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .composite import score_candidate
from . import correlation as corrmod

DEFAULT_TOP_N = 5
DEFAULT_MAX_PER_PLAYER = 2   # correlated markets (rec yds + receptions) pile up otherwise
DEFAULT_CORR_DISCOUNT_STRENGTH = 1.0   # how much of the shrunk rho to apply as a discount

CONTEXT_LABEL = "Context only — not part of the composite score."
SGP_LABEL = ("SGP joint estimate — informational only. Built from the model's own "
            "probabilities and the Phase 7.5 measured correlation (Gaussian copula); "
            "it is NOT a synthetic-line edge and prices nothing until a real "
            "same-game-parlay market exists.")


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def _max_positive_rho(cand: Dict, selected: List[Dict],
                      corr: "corrmod.CorrelationStructure",
                      as_of_season: Optional[int]) -> Tuple[float, Optional[Dict]]:
    """Highest POSITIVE shrunk rho between ``cand`` and any already-selected
    lean in this game (0.0 / None if nothing correlates, or every measured
    pair is negative/diversifying -- those are left alone, per 7.5)."""
    best, best_with = 0.0, None
    for s in selected:
        rho = corr.rho_for(cand.get("pos"), cand.get("market"), cand.get("player_id"), cand.get("team"),
                           s.get("pos"), s.get("market"), s.get("player_id"), s.get("team"),
                           as_of_season=as_of_season)
        if rho > best:
            best, best_with = rho, {"player_id": s.get("player_id"), "name": s.get("name"),
                                    "market": s.get("market"), "rho": round(rho, 4)}
    return best, best_with


def rank_game(cands: List[Dict], weights: Optional[Dict] = None,
              params: Optional[Dict] = None, top_n: int = DEFAULT_TOP_N,
              max_per_player: int = DEFAULT_MAX_PER_PLAYER,
              corr: Optional["corrmod.CorrelationStructure"] = None,
              as_of_season: Optional[int] = None,
              corr_discount_strength: float = DEFAULT_CORR_DISCOUNT_STRENGTH) -> Dict:
    """Rank one game's candidates -> its shortlist.

    Returns::

        {"game_id", "matchup", "screened": "5 of N", "screened_n": N,
         "leans": [candidate + score fields, ...]}   # len <= top_n

    Deterministic: composite desc, then (player_id, market) as an absolute
    tie-break so equal scores can never reorder between runs.

    ``corr`` (Phase 7.6, optional): a ``correlation.CorrelationStructure``. When
    given, selection becomes an MMR-style greedy walk -- at each slot, pick the
    REMAINING candidate with the highest *discounted* score, where the discount
    is driven by the largest POSITIVE shrunk rho vs any lean already selected
    in this game (same-player pairs ~0.76, same-team QB+pass-catcher ~0.30).
    A near-duplicate leg that would have filled a slot on raw composite alone
    can lose it to a lower-composite but genuinely independent candidate
    further down the list -- "a top-5 isn't secretly five bets on one game
    outcome." Negative/diversifying rho is never penalized. ``corr=None``
    (the default) reproduces the exact pre-7.6 selection, byte-for-byte.
    ``as_of_season``: pass the backtest season for a strict walk-forward rho
    (only seasons < as_of_season inform it); omit it live for the production
    (all-history) value, per the artifact's documented interface.
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

    # ML ranking mode (flag-gated upstream): candidates arrive stamped with
    # ``ml_score`` (100 x the classifier's side probability). Ordering uses it;
    # the deterministic composite is still computed and displayed so every
    # lean stays explainable. Absent the stamp, ranking is pure composite.
    use_ml = all(r.get("ml_score") is not None for r in scored) and bool(scored)
    rank_key = (lambda r: (-r["ml_score"], str(r["player_id"]), r["market"])) if use_ml \
        else (lambda r: (-r["composite"], str(r["player_id"]), r["market"]))
    scored.sort(key=rank_key)

    if corr is None:
        # Pre-7.6 selection -- unchanged, byte-identical.
        leans, per_player = [], {}
        for r in scored:
            pid = r["player_id"]
            if per_player.get(pid, 0) >= max_per_player:
                continue
            leans.append(r)
            per_player[pid] = per_player.get(pid, 0) + 1
            if len(leans) >= top_n:
                break
    else:
        base_key = (lambda r: float(r["ml_score"])) if use_ml else (lambda r: float(r["composite"]))
        leans, per_player = [], {}
        remaining = list(scored)   # already sorted; scan preserves tie order
        while remaining and len(leans) < top_n:
            best_i, best_eff, best_disc, best_with = None, None, 0.0, None
            for i, r in enumerate(remaining):
                if per_player.get(r["player_id"], 0) >= max_per_player:
                    continue
                rho, rho_with = _max_positive_rho(r, leans, corr, as_of_season)
                disc = corrmod.redundancy_discount(rho, corr_discount_strength)
                eff = base_key(r) * (1.0 - disc)
                if best_eff is None or eff > best_eff:
                    best_i, best_eff, best_disc, best_with = i, eff, disc, rho_with
            if best_i is None:
                break   # every remaining candidate is per-player capped out
            r = remaining.pop(best_i)
            r["corr_discount"] = round(best_disc, 4)
            r["corr_with"] = best_with
            leans.append(r)
            per_player[r["player_id"]] = per_player.get(r["player_id"], 0) + 1
        for r in leans:
            r.setdefault("corr_discount", 0.0)
            r.setdefault("corr_with", None)

    top = len(leans)
    return {
        "game_id": cands[0].get("game_id"),
        "matchup": cands[0].get("matchup"),
        "screened": f"{top} of {n_screened}",
        "screened_n": n_screened,
        "leans": leans,
        # the FULL scored pool (every candidate, evaluated): consumed by the
        # post-projection selector (nflvalue/selector.py) so best-picks
        # selection can only ever happen AFTER everything was scored; callers
        # that don't need it (report.generate pops it before persisting)
        "scored_pool": scored,
    }


def shortlist_week(candidates_df: pd.DataFrame, weights: Optional[Dict] = None,
                   params: Optional[Dict] = None, top_n: int = DEFAULT_TOP_N,
                   max_per_player: int = DEFAULT_MAX_PER_PLAYER,
                   corr: Optional["corrmod.CorrelationStructure"] = None,
                   as_of_season: Optional[int] = None,
                   corr_discount_strength: float = DEFAULT_CORR_DISCOUNT_STRENGTH) -> List[Dict]:
    """Rank every game of the week. Games ordered by game_id (deterministic).
    ``corr``/``as_of_season``/``corr_discount_strength``: see ``rank_game``."""
    out = []
    if candidates_df is None or candidates_df.empty:
        return out
    for game_id, grp in candidates_df.groupby("game_id", sort=True):
        cands = grp.to_dict("records")
        out.append(rank_game(cands, weights=weights, params=params,
                             top_n=top_n, max_per_player=max_per_player,
                             corr=corr, as_of_season=as_of_season,
                             corr_discount_strength=corr_discount_strength))
    return out


# --------------------------------------------------------------------------- #
# Optional SGP joint-probability readout (Phase 7.6 / 7.5's narrow green-light)
# --------------------------------------------------------------------------- #
def sgp_readouts(game_shortlist: Dict, corr: Optional["corrmod.CorrelationStructure"],
                 as_of_season: Optional[int] = None) -> List[Dict]:
    """Optional, clearly-labeled Same-Game-Parlay joint-probability estimate
    for pairs of SELECTED leans whose type cleared 7.5's REAL correlation bar.

    Uses each leg's own model probability (never a synthetic-line "edge") and
    a Gaussian copula on the measured shrunk rho (``correlation.sgp_joint_prob``).
    Computed AFTER selection -- display-only, feeds nothing back into ranking.
    Returns [] if ``corr`` is None, no real-type pair is present among the
    selected leans, or either leg's model probability is unavailable."""
    if corr is None:
        return []
    leans = game_shortlist.get("leans", [])
    out: List[Dict] = []
    seen = set()
    for i in range(len(leans)):
        for j in range(i + 1, len(leans)):
            a, b = leans[i], leans[j]
            rho = corr.rho_for(a.get("pos"), a.get("market"), a.get("player_id"), a.get("team"),
                               b.get("pos"), b.get("market"), b.get("player_id"), b.get("team"),
                               as_of_season=as_of_season)
            if rho == 0.0:
                continue   # unmeasured/NOISE type -- nothing honest to price
            key = tuple(sorted([f"{a.get('player_id')}:{a.get('market')}",
                                f"{b.get('player_id')}:{b.get('market')}"]))
            if key in seen:
                continue
            seen.add(key)
            pa = (a.get("components") or {}).get("model_prob")
            pb = (b.get("components") or {}).get("model_prob")
            if pa is None or pb is None:
                continue
            joint = corrmod.sgp_joint_prob(pa, a.get("side"), pb, b.get("side"), rho)
            if joint is None:
                continue
            out.append({
                "leg_a": {"player_id": a.get("player_id"), "name": a.get("name"),
                         "market": a.get("market"), "side": a.get("side")},
                "leg_b": {"player_id": b.get("player_id"), "name": b.get("name"),
                         "market": b.get("market"), "side": b.get("side")},
                "rho": round(float(rho), 4),
                "independent_joint_prob": round(float(pa) * float(pb), 6),
                "copula_joint_prob": joint,
                "label": SGP_LABEL,
            })
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

        # deterministic context facts (birthday/revenge/defensive outs,
        # contract year, O-line health, wind, QB continuity) -- computed from
        # fact tables, no news needed
        from .advanced_features import panel_items as adv_panel_items
        from .chemistry import panel_items as chem_panel_items
        from .ftn_features import panel_items as ftn_panel_items
        from .context_features import panel_items
        items.extend(panel_items(lean))
        items.extend(adv_panel_items(lean))
        items.extend(chem_panel_items(lean))
        items.extend(ftn_panel_items(lean))

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
