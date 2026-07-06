"""Same-game prop correlation structure (Phase 7.5) — classify + consume.

This module owns the *shared* logic: how a pair of props is classified into a
correlation TYPE, the Fisher-z shrinkage, and the read-side accessor that 7.6
(selection) and 7.7 (staking) use. The heavy measurement (build residuals,
pool pairs, write the artifact) lives in ``scripts/fit_correlation.py``, which
imports from here so the classification is identical on both sides.

Correlation is measured on STANDARDIZED RESIDUALS ``(actual − proj mean)/proj
sd`` within a game, then shrunk toward zero so thin/noisy pairs don't invent
structure. A pair TYPE is ``relationship | posA.familyA ~ posB.familyB`` with
``relationship`` ∈ {sameplayer, sameteam, opponent} and the two side-keys sorted
so the type is order-independent. Cross-player pairs are restricted to one
market per family (``CROSS_MARKETS``) so a single player can't be double-counted;
same-player pairs keep every market (yards ↔ attempts).

The accessor returns the SHRUNK ρ for a type, and **0.0 for any type the 7.5
audit judged NOISE or that isn't in the artifact** — so a consumer that asks
about two uncorrelated leans is told, correctly, zero.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from . import config as cfgmod

ART_PATH = os.path.join(cfgmod.DATA_DIR, "correlation_structure.json")

POSITIONS = ("QB", "RB", "WR", "TE")
FAMILY = {"passing_yards": "pass", "pass_attempts": "pass",
          "receiving_yards": "rec", "receptions": "rec",
          "rushing_yards": "rush", "rush_attempts": "rush",
          "anytime_td": "td"}
# one market per family for CROSS-player pairs (no double-counting a player)
CROSS_MARKETS = {"passing_yards", "receiving_yards", "rushing_yards", "anytime_td"}


def relationship(player_i: str, team_i: str, player_j: str, team_j: str) -> str:
    if player_i == player_j:
        return "sameplayer"
    return "sameteam" if team_i == team_j else "opponent"


def classify_pair(pos_i: str, market_i: str, player_i: str, team_i: str,
                  pos_j: str, market_j: str, player_j: str, team_j: str) -> Optional[str]:
    """Canonical, order-independent pair-type key, or None if the pair is not a
    measured type (e.g. a cross-player pair on a volume market)."""
    if market_i not in FAMILY or market_j not in FAMILY:
        return None
    rel = relationship(player_i, team_i, player_j, team_j)
    if rel != "sameplayer" and (market_i not in CROSS_MARKETS or market_j not in CROSS_MARKETS):
        return None
    lo, hi = sorted([f"{pos_i}.{FAMILY[market_i]}", f"{pos_j}.{FAMILY[market_j]}"])
    return f"{rel}|{lo}~{hi}"


def eb_fisher_z_shrink(rhos: Dict[str, float], ns: Dict[str, int]) -> Tuple[Dict[str, float], float]:
    """Empirical-Bayes shrink toward 0 in Fisher-z space. Returns (shrunk, tau2).

    z = atanh(ρ), SE² = 1/(n−3); τ² (between-type signal variance) is estimated
    across types; each z is pulled toward 0 by τ²/(τ²+SE²). A noisy small-n type
    (large SE²) collapses to ~0; a well-measured type barely moves."""
    keys = [k for k in rhos if ns[k] > 3 and np.isfinite(rhos[k])]
    if not keys:
        return {k: 0.0 for k in rhos}, 0.0
    z = {k: float(np.arctanh(np.clip(rhos[k], -0.999, 0.999))) for k in keys}
    se2 = {k: 1.0 / (ns[k] - 3) for k in keys}
    zbar = float(np.mean([z[k] for k in keys]))
    tau2 = max(0.0, float(np.mean([(z[k] - zbar) ** 2 for k in keys]))
               - float(np.mean([se2[k] for k in keys])))
    out = {}
    for k in rhos:
        if k not in z:
            out[k] = 0.0
        else:
            factor = tau2 / (tau2 + se2[k]) if (tau2 + se2[k]) > 0 else 0.0
            out[k] = float(np.tanh(z[k] * factor))
    return out, tau2


class CorrelationStructure:
    """Read-side accessor over ``data/correlation_structure.json``."""

    def __init__(self, payload: Dict):
        self.payload = payload
        self.pair_types: Dict[str, Dict] = payload.get("pair_types", {})
        self.walk_forward: Dict[str, Dict] = payload.get("walk_forward", {})

    @classmethod
    def load(cls, path: str = ART_PATH) -> "CorrelationStructure":
        return cls(cfgmod.load_json(path, {}) or {})

    def rho(self, ptype: Optional[str], as_of_season: Optional[int] = None) -> float:
        """Shrunk ρ for a pair type. **0.0 if the type is unknown or NOISE.**

        ``as_of_season`` gives the strict walk-forward value (estimated only from
        seasons < as_of) for a backtest; without it, the production (all-history)
        shrunk ρ is returned. NOISE types return 0.0 either way — a consumer is
        never handed structure the audit called noise."""
        if not ptype:
            return 0.0
        if as_of_season is not None:
            # Strict walk-forward: the slice encodes a PRIOR-ONLY real/noise
            # verdict (built from seasons < as_of only). Gate ONLY on presence
            # here -- never on the full-history verdict, which is contaminated
            # by seasons >= as_of and would leak the future inclusion decision.
            wf = self.walk_forward.get(str(as_of_season), {})
            if ptype not in wf:
                return 0.0          # not real / not enough prior-season data yet
            return float(wf[ptype])
        info = self.pair_types.get(ptype)
        if info is None or info.get("verdict") != "real":
            return 0.0
        return float(info.get("rho_shrunk", 0.0))

    def rho_for(self, pos_i: str, market_i: str, player_i: str, team_i: str,
                pos_j: str, market_j: str, player_j: str, team_j: str,
                as_of_season: Optional[int] = None) -> float:
        """Classify two props and return their shrunk correlation (0.0 if the
        pair isn't a measured/real type)."""
        pt = classify_pair(pos_i, market_i, player_i, team_i,
                           pos_j, market_j, player_j, team_j)
        return self.rho(pt, as_of_season)

    def real_types(self) -> Dict[str, float]:
        return {k: v["rho_shrunk"] for k, v in self.pair_types.items()
                if v.get("verdict") == "real"}


# --------------------------------------------------------------------------- #
# Phase 7.6 -- consumption: selection discount + optional SGP joint readout
# --------------------------------------------------------------------------- #
def redundancy_discount(rho: float, strength: float = 1.0) -> float:
    """POSITIVE-only discount factor in [0, 0.95] for correlation-aware
    selection (7.6).

    Only a POSITIVE rho discounts -- 7.5's read is explicit that negative
    (diversifying) pairs like QB-pass vs RB-rush or opposing RBs *help*, not
    hurt, so they are left alone (no bonus either; still just their own
    composite). A discount is never total: even a near-duplicate leg
    (same-player pair, rho~0.76) keeps a residual, so it can still fill an
    otherwise-empty slot rather than being hard-banned."""
    return float(min(0.95, max(0.0, float(rho)) * float(strength)))


def sgp_joint_prob(p_i: float, side_i: str, p_j: float, side_j: str,
                   rho: float) -> Optional[float]:
    """Gaussian-copula joint hit probability for two legs.

    Each leg's own model probability of hitting ITS selected side (never a
    synthetic-line "edge") is mapped to a standard-normal cut point, and the
    two legs' underlying standardized residuals are treated as bivariate
    normal with the 7.5-measured shrunk ``rho`` (side-agnostic; the
    correlation was measured directly on raw standardized residuals, so the
    sign/orientation of each leg's OWN cut point carries the side
    information). Returns the correct joint-probability quadrant via
    inclusion-exclusion on the bivariate normal CDF.

    Returns ``None`` for degenerate inputs (p at/near 0 or 1) rather than a
    meaningless number. ``rho == 0`` (unmeasured/NOISE pair, per
    ``CorrelationStructure.rho``) falls back to the independence product --
    callers should generally skip those pairs rather than display a copula
    readout that's identical to assuming independence."""
    try:
        p_i, p_j = float(p_i), float(p_j)
    except (TypeError, ValueError):
        return None
    eps = 1e-6
    if not (eps < p_i < 1 - eps) or not (eps < p_j < 1 - eps):
        return None
    if side_i not in ("over", "under") or side_j not in ("over", "under"):
        return None
    if abs(rho) < 1e-9:
        return round(p_i * p_j, 6)

    from scipy.stats import multivariate_normal, norm

    r = max(-0.999, min(0.999, float(rho)))
    # cut point s.t. P(hit) = p in the residual's own orientation: an "over"
    # hits when the residual exceeds its threshold, "under" when it's below
    a = norm.ppf(1 - p_i) if side_i == "over" else norm.ppf(p_i)
    b = norm.ppf(1 - p_j) if side_j == "over" else norm.ppf(p_j)
    phi2 = float(multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]]).cdf([a, b]))
    phi_a, phi_b = float(norm.cdf(a)), float(norm.cdf(b))
    if side_i == "over" and side_j == "over":
        joint = 1.0 - phi_a - phi_b + phi2
    elif side_i == "over" and side_j == "under":
        joint = phi_b - phi2
    elif side_i == "under" and side_j == "over":
        joint = phi_a - phi2
    else:
        joint = phi2
    return round(float(min(1.0, max(0.0, joint))), 6)
