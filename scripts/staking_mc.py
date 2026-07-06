#!/usr/bin/env python3
"""Phase 7.7 — bankroll Monte Carlo for the advisory staking rule.

Reuses the Phase-6.8 realism framing: run the ACTUAL sizing rule
(nflvalue.staking) over a simulated season at PLAUSIBLE REAL-LINE edges
(52-58%), NOT the synthetic 66-68% (which compounds to fiction — 6.8 makes this
point; we honor it). Reports, per true hit rate and per strategy: median ending
bankroll (growth), p95 max drawdown, P(ruin), P(halve).

The season has within-game CORRELATED legs (a Gaussian-copula equicorrelation
model using the Phase-7.5 residual ρ as the latent correlation), so the sizing
rule's correlation adjustment is actually exercised: correlated legs share a
common game factor, which fattens drawdowns unless sizing accounts for it.

Strategies compared:
  flat      1u flat per bet (non-compounding) — the 6.8 baseline.
  qkelly    plain quarter-Kelly on the raw edge (no shrink, corr, or caps).
  shrunk    the shipped rule: edge-shrink x quarter-Kelly x correlation x caps.

ADVISORY ONLY. No bet is ever placed. Synthetic-line caveat on the inputs;
these are what PLAUSIBLE real-line edges would imply, not a profit claim.

Run: python3 scripts/staking_mc.py   # writes reports/staking_mc.md + data/staking_mc.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import staking as st                       # noqa: E402
from nflvalue.correlation import CorrelationStructure    # noqa: E402

UNIT_WIN = 100.0 / 110.0            # -110 net decimal odds (b)
PRICE = 1.0 + UNIT_WIN
MARKET_PROB = 0.5238               # de-vigged fair prob at -110 breakeven
BANKROLL0 = 100.0
WEEKS = 17
GAMES_PER_WEEK = 6                 # each a correlated QB+WR pair
SINGLES_PER_WEEK = 6               # independent singles
B_PATHS = 2000
TRUE_HITS = (0.52, 0.5238, 0.54, 0.55, 0.56, 0.58)
RNG = np.random.default_rng(20260707)

# the correlated pair we simulate; ρ from the 7.5 artifact if present, else 0.30
_cs = CorrelationStructure.load()
RHO_PAIR = _cs.rho("sameteam|QB.pass~WR.rec") or 0.30


def _slate(p: float):
    """One week's leans + a group id per lean (correlated legs share a group)."""
    leans, groups = [], []
    gi = 0
    for _ in range(GAMES_PER_WEEK):
        g = f"cg{gi}"
        leans.append(dict(game_id=g, player_id=f"qb{gi}", market="passing_yards",
                          pos="QB", team="A", side="over", p=p, market_prob=MARKET_PROB, price=PRICE))
        leans.append(dict(game_id=g, player_id=f"wr{gi}", market="receiving_yards",
                          pos="WR", team="A", side="over", p=p, market_prob=MARKET_PROB, price=PRICE))
        groups += [gi, gi]
        gi += 1
    for _ in range(SINGLES_PER_WEEK):
        leans.append(dict(game_id=f"sg{gi}", player_id=f"s{gi}", market="rushing_yards",
                          pos="RB", team="A", side="over", p=p, market_prob=MARKET_PROB, price=PRICE))
        groups.append(gi)
        gi += 1
    return leans, np.array(groups)


def _fracs(leans, strategy):
    if strategy == "flat":
        return None
    if strategy == "qkelly":            # plain quarter-Kelly, no shrink/corr/caps
        cfg = st.StakeConfig(s_edge=1.0, kappa=0.25, cap_bet=1.0, max_slate=1e9)
        rec = st.recommend_stakes(leans, BANKROLL0, corr=None, config=cfg)
    else:                                # 'shrunk' — the shipped rule
        rec = st.recommend_stakes(leans, BANKROLL0, corr=_cs,
                                  config=st.StakeConfig(), as_of_season=None)
    return np.array([r["stake_frac"] for r in rec["recommendations"]])


def _draw_outcomes(groups: np.ndarray, p: float) -> np.ndarray:
    """Correlated Bernoulli via a latent-Gaussian equicorrelation copula: legs
    sharing a group share a common factor with corr RHO_PAIR, thresholded so
    each leg wins with probability p (P(latent > Phi^{-1}(1-p)) = p)."""
    from scipy.special import ndtri            # inverse standard-normal CDF
    thr = ndtri(1.0 - p)
    n = len(groups)
    z = RNG.standard_normal(n)
    uniq = {g: RNG.standard_normal() for g in sorted(set(groups.tolist()))}
    common = np.array([uniq[g] for g in groups])
    r = RHO_PAIR
    latent = np.sqrt(r) * common + np.sqrt(1 - r) * z      # ~ N(0,1)
    return (latent > thr).astype(float)                    # 1 = win (over hits)


STRATEGIES = ("flat", "qkelly", "shrunk")


def _summ(ends: list, dds: list) -> dict:
    ends = np.array(ends)
    n_ruin = int(np.sum(ends <= 0.20 * BANKROLL0))         # ruin = lost >=80%
    n_halve = int(np.sum(ends <= 0.50 * BANKROLL0))
    return {
        "median_end": round(float(np.median(ends)), 1),
        "p5_end": round(float(np.quantile(ends, 0.05)), 1),
        "p95_end": round(float(np.quantile(ends, 0.95)), 1),
        "p95_max_drawdown_pct": round(float(np.quantile(dds, 0.95)) * 100, 1),
        "p_ruin": round(n_ruin / B_PATHS, 4),
        "p_halve": round(n_halve / B_PATHS, 4),
    }


def simulate_all(p: float) -> dict:
    """PAIRED cross-strategy comparison via common random numbers: for each
    (path, week) the SAME correlated outcome draws (and within-week order)
    drive all three strategies' bankroll updates, so drawdown/median are
    compared on identical outcome streams rather than independent draws.
    Stake fractions are bankroll-independent and computed once per strategy."""
    slate_leans, groups = _slate(p)
    frac = {s: _fracs(slate_leans, s) for s in STRATEGIES}
    ends = {s: [] for s in STRATEGIES}
    dds = {s: [] for s in STRATEGIES}
    for _ in range(B_PATHS):
        bank = {s: BANKROLL0 for s in STRATEGIES}
        peak = {s: BANKROLL0 for s in STRATEGIES}
        maxdd = {s: 0.0 for s in STRATEGIES}
        ruined = {s: False for s in STRATEGIES}
        for _wk in range(WEEKS):
            outc = _draw_outcomes(groups, p)               # drawn ONCE, shared
            order = RNG.permutation(len(outc))             # shared within-week order
            for i in order:
                won = outc[i] == 1
                for s in STRATEGIES:
                    if ruined[s]:
                        continue
                    stake = 1.0 if s == "flat" else frac[s][i] * bank[s]
                    bank[s] += stake * UNIT_WIN if won else -stake
                    peak[s] = max(peak[s], bank[s])
                    if peak[s] > 0:
                        maxdd[s] = max(maxdd[s], (peak[s] - bank[s]) / peak[s])
                    if bank[s] <= 0:
                        ruined[s] = True
        for s in STRATEGIES:
            ends[s].append(max(bank[s], 0.0))
            dds[s].append(maxdd[s])
    return {s: _summ(ends[s], dds[s]) for s in STRATEGIES}


def main() -> None:
    results = {}
    for p in TRUE_HITS:
        results[f"{p:.4f}"] = simulate_all(p)
    payload = {"note": "Advisory staking MC at PLAUSIBLE real-line edges; synthetic-line "
                       "caveat on inputs. No bet is ever placed.",
               "rho_pair_used": round(RHO_PAIR, 4), "bankroll0": BANKROLL0,
               "bets_per_season": WEEKS * (2 * GAMES_PER_WEEK + SINGLES_PER_WEEK),
               "B_paths": B_PATHS, "config": st.StakeConfig().__dict__, "results": results}
    with open(os.path.join(ROOT, "data", "staking_mc.json"), "w") as fh:
        json.dump(payload, fh, indent=1)

    L = ["# Phase 7.7 — advisory staking, bankroll Monte Carlo", "",
         "**ADVISORY ONLY — no bet is ever placed.** Sizes come from "
         "`nflvalue.staking`; this MC shows what they imply for the bankroll at "
         f"**plausible real-line edges (52-58%), NOT** the synthetic 66-68%. "
         f"Start {BANKROLL0:.0f}u, ~{WEEKS*(2*GAMES_PER_WEEK+SINGLES_PER_WEEK)} bets/season, "
         f"within-game legs correlated at ρ={RHO_PAIR:.2f} (7.5). Ruin = lost ≥80%. "
         "Synthetic-line caveat on every input. 1-800-GAMBLER.", "",
         "`shrunk` = shipped rule (edge-shrink × ¼-Kelly × correlation × caps); "
         "`qkelly` = plain ¼-Kelly on the raw edge; `flat` = 1u non-compounding.", "",
         "| true hit | strategy | median end | p5 | p95 | p95 max DD | P(ruin) | P(halve) |",
         "|---|---|---|---|---|---|---|---|"]
    for p in TRUE_HITS:
        for s in ("flat", "qkelly", "shrunk"):
            r = results[f"{p:.4f}"][s]
            L.append(f"| {p:.2%} | {s} | {r['median_end']} | {r['p5_end']} | {r['p95_end']} "
                     f"| {r['p95_max_drawdown_pct']}% | {r['p_ruin']} | {r['p_halve']} |")
        L.append("| | | | | | | | |")
    L += ["", "Reading it: at plausible edges the shipped `shrunk` rule grows the "
          "bankroll far slower than raw quarter-Kelly but with a much smaller p95 "
          "drawdown and near-zero ruin — the point of shrinking an estimated edge. "
          "At 52.38% (breakeven) every strategy drifts flat-to-down; there is no "
          "sizing that manufactures an edge that isn't there.", ""]
    with open(os.path.join(ROOT, "reports", "staking_mc.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
