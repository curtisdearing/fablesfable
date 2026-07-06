"""Phase 7.7 — advisory stake sizing (bankroll / Kelly). ADVISORY ONLY.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ This module RECOMMENDS stake sizes. It NEVER places a bet, moves      │
    │ money, or initiates a transfer. Output is advice a human may act on,  │
    │ ignore, or override. Nothing here is a guarantee of profit; every     │
    │ input probability is model-estimated and, until real CLV accrues,     │
    │ graded at SYNTHETIC lines. 1-800-GAMBLER.                             │
    └─────────────────────────────────────────────────────────────────────┘

The rule turns a lean's calibrated edge into a fraction of bankroll, sized to
survive the Phase-6.8 variance envelope. It is deliberately conservative,
because bet sizing is where a good model still goes broke:

  1. edge         = calibrated P(side) − de-vigged market prob; <= 0 -> stake 0.
  2. shrink       the edge toward the market prior (S_EDGE): the market is
                  efficient and our P is an estimate -- size as if the edge is
                  smaller than it looks. p_s = market_prob + S_EDGE*edge.
  3. full Kelly   on the SHRUNK prob at the real price: f* = (b*p_s-(1-p_s))/b.
  4. fractional   f = KAPPA * f*  (quarter-Kelly; absorbs parameter +
                  correlation uncertainty, matches 6.8).
  5. correlation  divide by (1 + sum of positive shrunk rho to the other leans
                  in the same game, from the 7.5 artifact) -- two correlated
                  leans are not two independent edges. Negative (hedging) rho
                  gets no bonus: conservative.
  6. per-bet cap  min(f, CAP_BET).
  7. portfolio    if the stakes on a slate sum past MAX_SLATE, scale them all
                  down proportionally; then a global DD_SCALE (<=1) lets the
                  6.8 Monte Carlo pin the fraction to a drawdown tolerance.

Deterministic: same inputs -> same recommendation, always. Pure: no DB, no
network, no file writes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

DISCLAIMER = ("ADVISORY ONLY — recommended sizes, not instructions. This tool "
              "never places a bet or moves money. Probabilities are model "
              "estimates graded at synthetic lines until real CLV accrues. "
              "1-800-GAMBLER.")


@dataclass(frozen=True)
class StakeConfig:
    s_edge: float = 0.5          # estimation-error edge shrink (toward market)
    kappa: float = 0.25          # fractional-Kelly multiple (quarter-Kelly)
    cap_bet: float = 0.02        # max fraction of bankroll on any one lean
    max_slate: float = 0.10      # max total fraction staked across a slate
    dd_scale: float = 1.0        # global scale (<=1) tuned by 6.8 MC to a DD cap
    unit_pct: float = 0.01       # 1 "unit" = 1% of bankroll (6.8 convention)


def _finite(x) -> Optional[float]:
    """Coerce to a finite float, or None if missing / NaN / inf / unparseable."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _kelly_full(p: float, b: float) -> float:
    """Full-Kelly fraction for prob p at net decimal odds b (= decimal-1)."""
    if b <= 0:
        return 0.0
    return (b * p - (1.0 - p)) / b


def _price_b(lean: Dict) -> Optional[float]:
    """Net fractional odds from the lean's price (decimal) for its side."""
    d = lean.get("price")
    if d is None:
        pr = lean.get("prices") or {}
        d = pr.get(lean.get("side"))
    d = _finite(d)
    if d is None:
        return None
    return d - 1.0 if d > 1.0 else None


def _corr_penalty(lean: Dict, others: List[Dict], corr, as_of_season) -> float:
    """Sum of positive shrunk correlations to the other leans in the same game."""
    if corr is None:
        return 0.0
    pen = 0.0
    for o in others:
        if o is lean or o.get("game_id") != lean.get("game_id"):
            continue
        rho = corr.rho_for(
            lean.get("pos", "NA"), lean.get("market"), lean.get("player_id"), lean.get("team"),
            o.get("pos", "NA"), o.get("market"), o.get("player_id"), o.get("team"),
            as_of_season=as_of_season)
        if rho > 0:
            pen += rho
    return pen


def recommend_stakes(leans: List[Dict], bankroll: float = 100.0,
                     corr=None, config: Optional[StakeConfig] = None,
                     as_of_season: Optional[int] = None) -> Dict:
    """Advisory stake sizes for a slate of leans. Returns a dict with per-lean
    recommendations and a slate risk readout. Places NO bets.

    Each lean dict needs: ``p`` (calibrated model prob of the chosen side),
    ``market_prob`` (de-vigged fair prob), ``price`` (decimal) or ``prices``,
    plus ``game_id``/``player_id``/``market``/``pos``/``team`` for correlation.
    """
    cfg = config or StakeConfig()

    # Advisory-safe bankroll: a non-finite or non-positive bankroll cannot fund a
    # bet, so we advise from a zero bankroll -> zero stake amounts, never negative.
    bankroll_ok = _finite(bankroll)
    bankroll = bankroll_ok if (bankroll_ok is not None and bankroll_ok > 0) else 0.0

    # dd_scale is a global de-risking dial the 6.8 MC pins to a drawdown tolerance.
    # By convention it is in (0, 1]; clamp defensively so a stray >1 (or negative /
    # non-finite) value can never breach the per-bet or slate caps or flip a sign.
    dd_scale = _finite(cfg.dd_scale)
    dd_scale = 1.0 if dd_scale is None else min(max(dd_scale, 0.0), 1.0)

    recs = []
    for ln in leans:
        p = _finite(ln.get("p"))
        mp = _finite(ln.get("market_prob"))
        b = _price_b(ln)
        rec = {**{k: ln.get(k) for k in ("game_id", "player_id", "market", "side", "pos", "team")},
               "stake_frac": 0.0, "stake_units": 0.0, "stake_amount": 0.0,
               "edge": None, "edge_shrunk": None, "kelly_full": 0.0,
               "corr_penalty": 0.0, "reason": ""}
        if p is None or mp is None or b is None:
            rec["reason"] = "missing / non-finite p / market_prob / price"
            recs.append(rec); continue
        # Probabilities must be genuine probabilities. Anything outside [0,1] is a
        # corrupt input; conservatively refuse to size it rather than guess a clamp.
        if not (0.0 <= p <= 1.0) or not (0.0 <= mp <= 1.0):
            rec["reason"] = "p / market_prob outside [0,1] -> unstakeable"
            recs.append(rec); continue
        edge = float(p) - float(mp)
        rec["edge"] = round(edge, 5)
        if edge <= 0:
            rec["reason"] = "no positive edge -> no stake"
            recs.append(rec); continue
        edge_s = cfg.s_edge * edge
        p_s = float(mp) + edge_s
        f_full = max(_kelly_full(p_s, b), 0.0)
        f = cfg.kappa * f_full
        pen = _corr_penalty(ln, leans, corr, as_of_season)
        f_corr = f / (1.0 + pen)
        f_capped = min(f_corr, cfg.cap_bet)
        rec.update({"edge_shrunk": round(edge_s, 5), "kelly_full": round(f_full, 5),
                    "corr_penalty": round(pen, 4), "stake_frac": f_capped,
                    "capped": bool(f_corr > cfg.cap_bet),
                    "reason": "ok"})
        recs.append(rec)

    # portfolio (slate) cap: scale all down if the total exceeds max_slate
    total = sum(r["stake_frac"] for r in recs)
    slate_scale = min(1.0, cfg.max_slate / total) if total > cfg.max_slate else 1.0
    for r in recs:
        f = r["stake_frac"] * slate_scale * dd_scale
        r["stake_frac"] = round(f, 6)
        r["stake_units"] = round(f / cfg.unit_pct, 3)      # 1u = 1% bankroll
        r["stake_amount"] = round(f * bankroll, 2)

    staked = [r for r in recs if r["stake_frac"] > 0]
    readout = {
        "n_leans": len(recs), "n_staked": len(staked),
        "total_exposure_frac": round(sum(r["stake_frac"] for r in recs), 5),
        "total_exposure_units": round(sum(r["stake_units"] for r in recs), 2),
        "largest_stake_units": round(max((r["stake_units"] for r in recs), default=0.0), 3),
        "slate_scaled": slate_scale < 1.0,
        "config": cfg.__dict__, "bankroll": bankroll,
    }
    return {"recommendations": recs, "readout": readout, "disclaimer": DISCLAIMER}
