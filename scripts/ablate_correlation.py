#!/usr/bin/env python3
"""Phase 7.6 — CHECKPOINT C ablation: does correlation-aware selection change
the OOS hit rate / units at the frozen policy (config "shortlist": top_n=5,
max_per_player=2)?

Runs ``lean_backtest.run`` twice per season -- BASELINE (corr_aware=False,
today's shipped selection) and CORR-AWARE (corr_aware=True, the 7.6 MMR-style
greedy discount using the 7.5 walk-forward rho artifact) -- over the same
walk-forward replay, and diffs:
  * overall / top-1 hit rate and units (flat 1u/lean, -110 juice)
  * how many leans actually changed (a near-duplicate leg swapped out for a
    more independent one) and a few EXAMPLE SLIPS showing it concretely
  * an "effective independent bets" measure: for every selected slip, sum of
    (1 - max positive pairwise rho with earlier-selected legs in the slip) --
    a crude but honest way to say "5 leans is really ~N independent bets"

Honesty (Done-when criterion): report the numbers as they land. Correlation-
aware selection ships (config "correlation.enabled") only if it beats or ties
baseline hit-rate/units, OR is neutral on both while measurably diversifying
(fewer near-duplicate pairs surviving into the top-5) -- otherwise the honest
call is to flip "correlation.enabled" to false and keep the module built but
dormant, exactly like 7.2's pruned-features/ensemble capabilities.

Run: python3 scripts/ablate_correlation.py --seasons 2022 2023 2024 2025
Writes: reports/correlation_ablation.md, data/correlation_ablation.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import lean_backtest as lb                              # noqa: E402
from nflvalue import config as cfgmod                    # noqa: E402
from nflvalue.correlation import ART_PATH, CorrelationStructure  # noqa: E402


def _effective_bets(leans: pd.DataFrame, corr: CorrelationStructure) -> Dict:
    """For every (season, week, game_id) slip actually selected, sum
    (1 - max positive pairwise rho vs earlier-selected legs) per leg -- an
    honest "how many independent bets did this top-5 really contain" number.
    Uses the SAME walk-forward-season rho the selection itself used."""
    if leans.empty:
        return {"n_slips": 0, "avg_leans_per_slip": None, "avg_effective_per_slip": None}
    totals = []
    for (season, wk, gid), grp in leans.groupby(["season", "week", "game_id"]):
        rows = grp.to_dict("records")
        selected: List[Dict] = []
        eff = 0.0
        for r in rows:
            best = 0.0
            for s in selected:
                rho = corr.rho_for(r.get("pos"), r["market"], r["player_id"], r.get("team"),
                                   s.get("pos"), s["market"], s["player_id"], s.get("team"),
                                   as_of_season=int(season))
                best = max(best, rho)
            eff += (1.0 - max(0.0, best))
            selected.append(r)
        totals.append((len(rows), eff))
    n_leans = sum(t[0] for t in totals)
    n_eff = sum(t[1] for t in totals)
    return {
        "n_slips": len(totals),
        "avg_leans_per_slip": round(n_leans / len(totals), 3),
        "avg_effective_per_slip": round(n_eff / len(totals), 3),
    }


def _example_slips(base_leans: pd.DataFrame, aware_leans: pd.DataFrame, n: int = 4) -> List[Dict]:
    """Games where the two policies picked a DIFFERENT set of (player, market)
    legs -- concrete before/after evidence of de-correlation (or lack of it)."""
    out = []
    if base_leans.empty or aware_leans.empty:
        return out
    keys = sorted(set(zip(base_leans.season, base_leans.week, base_leans.game_id)) &
                 set(zip(aware_leans.season, aware_leans.week, aware_leans.game_id)))
    for season, wk, gid in keys:
        b = base_leans[(base_leans.season == season) & (base_leans.week == wk)
                       & (base_leans.game_id == gid)]
        a = aware_leans[(aware_leans.season == season) & (aware_leans.week == wk)
                        & (aware_leans.game_id == gid)]
        b_set = set(zip(b.player_id, b.market))
        a_set = set(zip(a.player_id, a.market))
        if b_set == a_set:
            continue
        out.append({
            "season": int(season), "week": int(wk), "game_id": gid,
            "matchup": b["matchup"].iloc[0] if len(b) else a["matchup"].iloc[0],
            "baseline": [f"{r.name} {r.market} {r.side} (composite {r.composite})"
                        for r in b.itertuples(index=False)],
            "corr_aware": [f"{r.name} {r.market} {r.side} (composite {r.composite}"
                          + (f", discounted {r.corr_discount:.0%} vs {r.corr_with}"
                             if r.corr_discount else "") + ")"
                          for r in a.itertuples(index=False)],
        })
        if len(out) >= n:
            break
    return out


def run_ablation(seasons: List[int], inputs=None) -> Dict:
    if inputs is None:
        from nflvalue.candidates import build_week_inputs
        inputs = build_week_inputs()

    if not os.path.exists(ART_PATH):
        print(f"[ablate] correlation artifact missing ({ART_PATH}) -- nothing to ablate. "
              "Run scripts/fit_correlation.py first.")
        return {}
    corr = CorrelationStructure.load()

    per_season = {}
    all_base_leans, all_aware_leans = [], []
    for season in seasons:
        base = lb.run(season, inputs, write_files=False, corr_aware=False)
        aware = lb.run(season, inputs, write_files=False, corr_aware=True)
        all_base_leans.append(base["leans"])
        all_aware_leans.append(aware["leans"])
        bo, ao = base["report"]["leans"]["overall"], aware["report"]["leans"]["overall"]
        bt1, at1 = base["report"]["leans"]["top1_per_game"], aware["report"]["leans"]["top1_per_game"]
        per_season[season] = {
            "baseline": {"overall": bo, "top1": bt1},
            "corr_aware": {"overall": ao, "top1": at1},
            "delta_hit_rate": round((ao["hit_rate"] or 0) - (bo["hit_rate"] or 0), 4),
            "delta_units": round((ao["units"] or 0) - (bo["units"] or 0), 2),
        }
        print(f"[ablate] {season}: baseline hit={bo['hit_rate']:.1%} units={bo['units']:+.1f}u  "
              f"|  corr-aware hit={ao['hit_rate']:.1%} units={ao['units']:+.1f}u  "
              f"|  Δhit={per_season[season]['delta_hit_rate']:+.4f} Δunits={per_season[season]['delta_units']:+.2f}")

    base_all = pd.concat(all_base_leans, ignore_index=True) if all_base_leans else pd.DataFrame()
    aware_all = pd.concat(all_aware_leans, ignore_index=True) if all_aware_leans else pd.DataFrame()

    pooled_base = lb._rate(base_all) if len(base_all) else {}
    pooled_aware = lb._rate(aware_all) if len(aware_all) else {}
    pooled_base_top1 = lb._rate(base_all[base_all["rank"] == 1]) if len(base_all) else {}
    pooled_aware_top1 = lb._rate(aware_all[aware_all["rank"] == 1]) if len(aware_all) else {}

    eff_base = _effective_bets(base_all, corr)
    eff_aware = _effective_bets(aware_all, corr)
    n_diff_legs = int((aware_all["corr_discount"] > 0).sum()) if len(aware_all) else 0
    examples = _example_slips(base_all, aware_all, n=6)

    result = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seasons": seasons,
        "per_season": per_season,
        "pooled": {
            "baseline": {"overall": pooled_base, "top1": pooled_base_top1,
                        "effective_bets": eff_base},
            "corr_aware": {"overall": pooled_aware, "top1": pooled_aware_top1,
                          "effective_bets": eff_aware, "n_discounted_legs": n_diff_legs},
        },
        "example_slips": examples,
    }
    return result


def render_md(result: Dict) -> str:
    if not result:
        return "# Correlation ablation\n\nNo correlation artifact -- nothing run.\n"
    p = result["pooled"]
    bo, ao = p["baseline"]["overall"], p["corr_aware"]["overall"]
    bt1, at1 = p["baseline"]["top1"], p["corr_aware"]["top1"]
    lines = [
        f"# Correlation-aware selection ablation — seasons {result['seasons'][0]}–{result['seasons'][-1]}",
        "",
        "**Leans, not locks.** Directional grading at synthetic trailing-mean lines "
        "(not price-beating/profit). Units at flat 1u/lean, standard -110 juice.",
        "",
        "## Pooled result",
        "",
        "| | n | hit rate | units |",
        "|---|---|---|---|",
        f"| Baseline (shipped) | {bo['n']} | {bo['hit_rate']:.1%} | {bo['units']:+.1f}u |",
        f"| Correlation-aware | {ao['n']} | {ao['hit_rate']:.1%} | {ao['units']:+.1f}u |",
        "",
        "| Top-1 per game | n | hit rate | units |",
        "|---|---|---|---|",
        f"| Baseline | {bt1['n']} | {bt1['hit_rate']:.1%} | {bt1['units']:+.1f}u |",
        f"| Correlation-aware | {at1['n']} | {at1['hit_rate']:.1%} | {at1['units']:+.1f}u |",
        "",
        "## Diversification",
        "",
        f"- Baseline: avg {p['baseline']['effective_bets']['avg_leans_per_slip']} leans/slip, "
        f"~{p['baseline']['effective_bets']['avg_effective_per_slip']} effective independent bets/slip "
        "(1 minus max positive pairwise rho with each earlier-selected leg in the same slip).",
        f"- Correlation-aware: avg {p['corr_aware']['effective_bets']['avg_leans_per_slip']} leans/slip, "
        f"~{p['corr_aware']['effective_bets']['avg_effective_per_slip']} effective independent bets/slip; "
        f"{p['corr_aware']['n_discounted_legs']} selected legs carried a nonzero correlation discount.",
        "",
        "## Per-season",
        "",
        "| Season | Baseline hit | Baseline units | Corr-aware hit | Corr-aware units | Δhit | Δunits |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, r in result["per_season"].items():
        bo_s, ao_s = r["baseline"]["overall"], r["corr_aware"]["overall"]
        lines.append(f"| {s} | {bo_s['hit_rate']:.1%} | {bo_s['units']:+.1f}u | "
                     f"{ao_s['hit_rate']:.1%} | {ao_s['units']:+.1f}u | "
                     f"{r['delta_hit_rate']:+.2%} | {r['delta_units']:+.2f}u |")
    lines += ["", "## Example slips (baseline vs correlation-aware, same game)", ""]
    if not result["example_slips"]:
        lines.append("No games differed between the two policies over these seasons/weeks.")
    for ex in result["example_slips"]:
        lines += [f"**{ex['matchup']}** — {ex['season']} week {ex['week']}", "",
                 "Baseline:"]
        lines += [f"- {b}" for b in ex["baseline"]]
        lines += ["", "Correlation-aware:"]
        lines += [f"- {a}" for a in ex["corr_aware"]]
        lines += [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    args = ap.parse_args()
    result = run_ablation(args.seasons)
    os.makedirs("reports", exist_ok=True)
    cfgmod.save_json(os.path.join(cfgmod.DATA_DIR, "correlation_ablation.json"), result)
    with open(os.path.join("reports", "correlation_ablation.md"), "w") as f:
        f.write(render_md(result))
    print("\nWrote reports/correlation_ablation.md + data/correlation_ablation.json")


if __name__ == "__main__":
    main()
