"""Composite 0-100 prop score: edge + confidence + matchup. Pure math, no I/O.

PROP_SHORTLISTER_SPEC.md §3, with the premortem guardrails wired into the
score itself:

  edge        model P(side) minus the de-vigged implied probability of that
              side's price -- ONLY when a real prop line+price was pulled
              (Phase 3). Without prices the component is dropped, the weights
              renormalize over confidence+matchup, and the result is tagged
              ``no_market`` -- graceful degradation, clearly labeled.
  confidence  |projection - line| in SD units (z), capped and scaled to 0-1.
              Tighter, higher-conviction distributions score higher.
  matchup     opponent-vs-position factor, team pace, and game-script fit,
              each expressed DIRECTIONALLY for the chosen side (a soft
              defense helps an OVER; a slow, run-leaning script helps an
              UNDER of a pass market).

``composite = 100 * (w_e*edge + w_c*conf + w_m*matchup) / (w_e+w_c+w_m)``

Design notes (encoding the build prompts' requirements):
  * EDGE IS THE DOMINANT TERM once real lines exist (default weights
    0.5/0.3/0.2): "best" means best value vs the line, not highest projection.
  * The CALIBRATION GATE (Phase 3 hard rule): edges are only computed/trusted
    when ``calibration_passed`` is True (config "composite" section). The
    Phase 1B calibration fix reviewed at Checkpoint 1B-a is the basis for the
    default True; flipping it to False forces every candidate to no_market
    pricing behavior (confidence+matchup only) without touching callers.
  * CONTEXT CARRIES ZERO WEIGHT. This function does not even accept context/
    news arguments -- context is attached later, display-only, by
    ``shortlist.py``. A test asserts score equality with/without context.
  * Deterministic: same candidate dict -> same score, always.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from . import oddsmath

DEFAULT_WEIGHTS = {"edge": 0.5, "confidence": 0.3, "matchup": 0.2}
DEFAULT_PARAMS = {
    "z_cap": 2.0,             # |z| at/above this = full confidence component
    "edge_cap": 0.10,         # a 10-point probability edge = full edge component
    "edge_floor": 0.0,        # negative edges clamp to 0 (they also flip side first)
    "opp_factor_span": 0.30,  # +/-30% vs league avg spans the matchup sub-score
    "pace_span": 8.0,         # +/-8 plays vs league avg spans the pace sub-score
    "script_span": 0.12,      # game_script_multipliers max tilt
    "calibration_passed": True,   # Phase 1B Checkpoint 1B-a calibration fix reviewed
    "low_confidence_mult": 0.8,   # spec §2: TDs are high-variance -- "include last"
}

# Markets quoted one-sided by books (anytime TD = "Yes" only). The model may
# think a TD is UNLIKELY, but "no TD" isn't a purchasable lean -- ranking such
# unders would fill the top-5 with degenerate, untradeable picks.
YES_ONLY_MARKETS = {"anytime_td"}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _direction(side: str) -> float:
    return 1.0 if side == "over" else -1.0


def _devig_probs(prices: Dict) -> Optional[Dict[str, float]]:
    """{'over': decimal, 'under': decimal} -> fair {'over': p, 'under': p}."""
    over, under = prices.get("over"), prices.get("under")
    if not over or not under:
        return None
    try:
        over_d, under_d = float(over), float(under)
    except (TypeError, ValueError):
        return None
    if over_d <= 1.0 or under_d <= 1.0:
        return None
    p_over, p_under = oddsmath.devig_multiplicative([over_d, under_d])
    return {"over": p_over, "under": p_under}


def score_candidate(cand: Dict, weights: Optional[Dict[str, float]] = None,
                    params: Optional[Dict] = None) -> Dict:
    """Score one candidate (a row from ``candidates.enumerate_candidates``).

    Returns::

        {composite, side, no_market, edge, confidence, matchup,
         components: {edge_raw, market_prob?, model_prob, z, opp_sub, pace_sub,
                      script_sub, weights_used}}

    ``edge`` is None (and ``no_market`` True) when no two-sided price exists
    or the calibration gate is closed.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    prm = {**DEFAULT_PARAMS, **(params or {})}

    mean = float(cand["mean"])
    sd = max(float(cand["sd"]), 1e-6)
    line = cand.get("line")
    p_over = cand.get("p_over")
    p_under = cand.get("p_under")

    # ---- side + market comparison ---------------------------------------- #
    yes_only = cand.get("market") in YES_ONLY_MARKETS
    fair = None
    prices = cand.get("prices") or None
    if prices and prm["calibration_passed"]:
        fair = _devig_probs(prices)

    if yes_only:
        side = "over"  # rendered as YES; the only side a book quotes
        edge_raw = (float(p_over) - fair["over"]) if (fair is not None and p_over is not None) else None
        market_prob = fair["over"] if fair is not None else None
    elif fair is not None and p_over is not None:
        edge_over = float(p_over) - fair["over"]
        edge_under = float(p_under) - fair["under"]
        side = "over" if edge_over >= edge_under else "under"
        edge_raw = max(edge_over, edge_under)
        market_prob = fair[side]
    else:
        side = "over" if (p_over is not None and float(p_over) >= 0.5) else "under"
        edge_raw, market_prob = None, None

    model_prob = float(p_over if side == "over" else p_under) if p_over is not None else None

    # ---- components ------------------------------------------------------- #
    no_market = edge_raw is None
    edge_comp = None
    if not no_market:
        edge_comp = _clip01(max(edge_raw, prm["edge_floor"]) / prm["edge_cap"])

    z = (mean - float(line)) / sd if line is not None else 0.0
    conf_comp = _clip01(min(abs(z), prm["z_cap"]) / prm["z_cap"])
    # no confidence credit for a side the model itself puts under 50% --
    # distance-from-line means nothing if the lean points the wrong way
    # (edge can still carry a market-mispricing signal on such a side)
    if model_prob is not None and model_prob < 0.5:
        conf_comp = 0.0

    comps = cand.get("components") or {}
    opp_factor = float(comps.get("opp_factor", 1.0) or 1.0)
    d = _direction(side)
    opp_sub = _clip01(0.5 + d * (opp_factor - 1.0) / prm["opp_factor_span"] * 0.5)

    gs = float(comps.get("game_script", 1.0) or 1.0)
    script_sub = _clip01(0.5 + d * (gs - 1.0) / prm["script_span"] * 0.5)

    pace_sub = 0.5  # neutral unless team volume context is present
    team_volume = cand.get("team_plays_vs_league")
    if team_volume is not None and not (isinstance(team_volume, float) and math.isnan(team_volume)):
        pace_sub = _clip01(0.5 + d * float(team_volume) / prm["pace_span"] * 0.5)

    matchup_comp = (opp_sub + script_sub + pace_sub) / 3.0

    # ---- weighted blend (renormalize when edge is unavailable) ------------- #
    if no_market:
        active = {"confidence": w["confidence"], "matchup": w["matchup"]}
        total = sum(active.values())
        composite = 100.0 * (w["confidence"] * conf_comp + w["matchup"] * matchup_comp) / total
    else:
        total = w["edge"] + w["confidence"] + w["matchup"]
        composite = 100.0 * (w["edge"] * edge_comp + w["confidence"] * conf_comp
                             + w["matchup"] * matchup_comp) / total

    if cand.get("low_confidence"):
        composite *= float(prm["low_confidence_mult"])

    return {
        "composite": round(composite, 2),
        "side": side,
        "no_market": no_market,
        "edge": round(edge_raw, 4) if edge_raw is not None else None,
        "confidence": round(conf_comp, 4),
        "matchup": round(matchup_comp, 4),
        "components": {
            "edge_raw": round(edge_raw, 4) if edge_raw is not None else None,
            "edge_component": round(edge_comp, 4) if edge_comp is not None else None,
            "market_prob": round(market_prob, 4) if market_prob is not None else None,
            "model_prob": round(model_prob, 4) if model_prob is not None else None,
            "z": round(z, 3),
            "opp_sub": round(opp_sub, 4),
            "script_sub": round(script_sub, 4),
            "pace_sub": round(pace_sub, 4),
            "weights_used": ({"confidence": w["confidence"], "matchup": w["matchup"]}
                             if no_market else dict(w)),
            "calibration_gate": bool(prm["calibration_passed"]),
        },
    }
