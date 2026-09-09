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
from . import prop_decision

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
    """Fair market probabilities for a prop quote.

    Prefers the CROSS-BOOK consensus (sharp-weighted, computed at snapshot
    time by oddsapi_props.to_prop_lines_frame) — a single soft book's vig
    shape can't masquerade as fair value. Falls back to de-vigging the
    two-sided quote when no consensus was carried (older rows, tests)."""
    cp = prices.get("consensus_p_over")
    if cp is not None:
        try:
            cpf = float(cp)
            if 0.0 < cpf < 1.0:
                return {"over": cpf, "under": 1.0 - cpf}
        except (TypeError, ValueError):
            pass
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
    # Never use an ML ranker score as a decision probability. Re-derive from
    # the displayed distribution; drift between the stored p_over and the
    # distribution is a hard killcheck for ACTION (edge/EV/Kelly), never
    # silently recomputed into an actionable number.
    probs = prop_decision.side_probabilities(cand)
    p_over, p_under, p_push = probs["p_over"], probs["p_under"], probs["p_push"]
    coherent = prop_decision.probability_coherent(cand, p_over)
    # conditional-on-no-push side probabilities: the de-vigged two-way market
    # is conditional on no push too, and Kelly/EV per unit at risk use them
    p_over_c = p_under_c = None
    if p_over is not None:
        no_push = max(1.0 - float(p_push or 0.0), 1e-9)
        p_over_c, p_under_c = float(p_over) / no_push, float(p_under) / no_push

    # ---- side + market comparison ---------------------------------------- #
    yes_only = cand.get("market") in YES_ONLY_MARKETS
    fair = None
    prices = cand.get("prices") or None
    n_books = prop_decision.valid_book_count((prices or {}).get("n_books"))
    enough_books = n_books is not None and n_books >= 2
    side_price_ok = bool(prices) and (
        prop_decision.valid_price(prices.get("over")) if yes_only
        else (prop_decision.valid_price(prices.get("over"))
              and prop_decision.valid_price(prices.get("under"))))
    actionable = bool(prices and side_price_ok and enough_books and coherent
                      and prm["calibration_passed"] and p_over_c is not None)
    if actionable:
        fair = _devig_probs(prices)

    if yes_only:
        side = "over"  # rendered as YES; the only side a book quotes
        if fair is not None and p_over_c is not None:
            edge_raw = float(p_over_c) - fair["over"]
            market_prob = fair["over"]
        elif actionable and prices.get("over"):
            # one-sided market: no de-vig possible; compare against the RAW
            # implied probability (vig included -> conservative, edge understated)
            market_prob = oddsmath.implied_prob(float(prices["over"]))
            edge_raw = float(p_over_c) - market_prob
        else:
            edge_raw, market_prob = None, None
    elif fair is not None and p_over_c is not None:
        edge_over = float(p_over_c) - fair["over"]
        edge_under = float(p_under_c) - fair["under"]
        side = "over" if edge_over >= edge_under else "under"
        edge_raw = max(edge_over, edge_under)
        market_prob = fair[side]
    elif p_over_c is not None:
        side = "over" if float(p_over_c) >= 0.5 else "under"
        edge_raw, market_prob = None, None
    else:
        # no distribution probability could be derived: the projection's own
        # direction (mean vs line) picks the side; nothing here is actionable
        side = "over" if (line is None or mean >= float(line)) else "under"
        edge_raw, market_prob = None, None

    model_prob = (float(p_over_c if side == "over" else p_under_c)
                  if p_over_c is not None else None)

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

    # EPA-allowed dimension (evaluation catch: computed since the context-
    # features build but never scored) -- a defense bleeding EPA to this
    # position is a soft matchup beyond raw yards-per-play
    subs = [opp_sub, script_sub, pace_sub]
    epa_f = cand.get("opp_epa_factor")
    if epa_f is not None and not (isinstance(epa_f, float) and math.isnan(epa_f)):
        subs.append(_clip01(0.5 + d * (float(epa_f) - 1.0) / 0.15 * 0.5))
    matchup_comp = sum(subs) / len(subs)

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

    # -- learning-loop multipliers (both absent/1.0 unless the pipeline set
    # them from walk-forward evidence; context_mult additionally requires the
    # tag to be user-promoted in config after clearing the evidence bar) ----- #
    reliability_mult = cand.get("reliability_mult")
    if reliability_mult is not None:
        composite *= float(reliability_mult)
    context_mult = cand.get("context_mult")
    if context_mult is not None:
        composite *= float(context_mult)

    if not no_market:
        market_state = "REAL_MARKET"
    elif not prices:
        market_state = "NO_MARKET"
    elif not side_price_ok:
        market_state = "INVALID_PRICE"
    elif not enough_books:
        market_state = "ONE_BOOK_CONTEXT_ONLY"
    elif not coherent or p_over_c is None:
        market_state = "PROBABILITY_KILLCHECK"
    elif not prm["calibration_passed"]:
        market_state = "CALIBRATION_GATE_CLOSED"
    else:
        market_state = "NO_MARKET"
    side_price = (prices or {}).get("over" if side == "over" else "under")
    can_stake = (not no_market and model_prob is not None
                 and prop_decision.valid_price(side_price))
    # Is this edge bigger than our ignorance about SD? `market_residual_sd`
    # fits ONE residual SD per market for every player, so a quarterback's
    # passing-yards SD is a league-wide number (97.375 on 2026 Week 1, the
    # same for D.Maye and S.Darnold). Differencing a probability built on that
    # against a real book can manufacture a few points of "edge" out of a
    # quantity never estimated for this player. Report the swing next to the
    # edge so a reader -- and the shortlist -- can tell the two apart.
    sd_swing = prop_decision.sd_stress(cand)
    edge_survives = prop_decision.edge_survives_sd_uncertainty(edge_raw, sd_swing)
    return {
        "composite": round(composite, 2),
        "side": side,
        "no_market": no_market,
        "market_state": market_state,
        "edge": round(edge_raw, 4) if edge_raw is not None else None,
        "confidence": round(conf_comp, 4),
        "matchup": round(matchup_comp, 4),
        "components": {
            "edge_raw": round(edge_raw, 4) if edge_raw is not None else None,
            "edge_component": round(edge_comp, 4) if edge_comp is not None else None,
            "market_prob": round(market_prob, 4) if market_prob is not None else None,
            "model_prob": round(model_prob, 4) if model_prob is not None else None,
            "model_prob_source": "mean_sd_line_distribution",
            "probability_coherent": bool(coherent),
            # SD provenance and stress. `sd_scope: "market"` says out loud
            # that this SD was not estimated for this player.
            "sd_scope": cand.get("sd_scope") or "market",
            "sd_source": cand.get("sd_source"),
            "sd_stress_fraction": prop_decision.SD_STRESS_FRACTION,
            "sd_prob_swing": round(sd_swing, 4) if sd_swing is not None else None,
            "edge_survives_sd_uncertainty": edge_survives,
            "z": round(z, 3),
            "opp_sub": round(opp_sub, 4),
            "script_sub": round(script_sub, 4),
            "pace_sub": round(pace_sub, 4),
            "weights_used": ({"confidence": w["confidence"], "matchup": w["matchup"]}
                             if no_market else dict(w)),
            "calibration_gate": bool(prm["calibration_passed"]),
            "p_push": (round(float(p_push), 4) if p_push is not None else None),
            # ev_best_price is CONDITIONAL on settlement (no push); a push
            # returns the stake, so the expected return per ORIGINAL stake is
            # (1 - p_push) x that.  Kelly is correct on the conditional
            # probability; the per-stake return is the smaller number.
            "ev_best_price": (round(model_prob * float(side_price) - 1.0, 4)
                              if can_stake else None),
            "ev_per_stake": (round((1.0 - float(p_push or 0.0)) * (model_prob * float(side_price) - 1.0), 4)
                             if can_stake else None),
            "ev_basis": {"ev_best_price": "conditional_on_settlement",
                         "ev_per_stake": "per_original_stake"},
            "kelly_fraction": (round(oddsmath.kelly_fraction(model_prob, float(side_price)), 4)
                               if can_stake else None),
            "n_books": n_books if prices else None,
            "reliability_mult": (round(float(reliability_mult), 4)
                                 if reliability_mult is not None else None),
            "context_mult": (round(float(context_mult), 4)
                             if context_mult is not None else None),
            "bias_mult": cand.get("bias_mult"),
        },
    }
