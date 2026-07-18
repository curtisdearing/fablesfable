#!/usr/bin/env python3
"""Lever #3 band recalibration with a held-out forward test.

Rebuilds the graded game-line replay across every walk-forward season in
backtest_games.json, writes the full pool to data/weekly.json (the product's
band-calibration source), and runs the registered honesty check:

    TRAIN band accuracies on 2021-2024, FREEZE them, then apply those frozen
    bands to 2025 and measure the REALIZED accuracy of every tier/band the
    ranker emits. If a band's out-of-sample 2025 accuracy collapses relative
    to its 2021-2024 training accuracy, the tier label was overfit.

Grading matches nflvalue.top_bets.calibrate_bands exactly (pushes / ties are
excluded from the denominator).

Run:  PYTHONPATH=. python3 analysis/lever3_holdout_recal.py
"""
import json, os
import weekly
from nflvalue import config, top_bets as tb

TRAIN = {2021, 2022, 2023, 2024}
VERIFY = 2025


def build_all_seasons(sims=5000):
    games = config.load_json(os.path.join(config.DATA_DIR, "backtest_games.json"), None)
    seasons = sorted({g["season"] for g in games})
    weeks = []
    for s in seasons:
        d = weekly.build_historical(season=s, sims=sims)
        for w in d["weeks"]:
            weeks.append({**w, "season": s, "week": f"{s}-{w['week']}"})
    return {"mode": "historical-multiseason", "season": f"{seasons[0]}-{seasons[-1]}",
            "seasons_available": seasons, "sims": sims, "weeks": weeks,
            "note": "Full walk-forward replay for Lever #3 band calibration."}


def subset(weekly_obj, seasons):
    return {**weekly_obj, "weeks": [w for w in weekly_obj["weeks"] if w["season"] in seasons]}


def _result_win(g, market):
    """Realized win (1/0) or None (push/tie -> excluded), matching calibrate_bands."""
    if market == "spread":
        r = g.get("ats_result");  return 1 if r == "W" else (0 if r == "L" else None)
    if market == "total":
        r = g.get("total_result");  return 1 if r == "W" else (0 if r == "L" else None)
    if market == "moneyline":
        sc = g.get("su_correct");  return None if sc is None else (1 if sc else 0)
    return None


def grade_holdout(verify_weekly, frozen_bands):
    """Apply frozen bands to the held-out season; measure realized tier/band acc."""
    from collections import defaultdict
    tier_hits = defaultdict(lambda: [0, 0])   # tier -> [wins, n]
    band_hits = defaultdict(lambda: [0, 0])   # band -> [wins, n]
    for wk in verify_weekly["weeks"]:
        for g in wk["games"]:
            if not g.get("settled"):
                continue
            for bet in tb.rank_game(g, frozen_bands):
                w = _result_win(g, bet["market"])
                if w is None:
                    continue
                tier_hits[bet["tier"]][0] += w; tier_hits[bet["tier"]][1] += 1
                band_hits[bet["band"]][0] += w; band_hits[bet["band"]][1] += 1
    return tier_hits, band_hits


def _fmt_bands(bands):
    for b, r in bands.items():
        acc, lb = r["accuracy"], r["accuracy_lb"]
        if acc is None:
            print(f"    {b:20s} n={r['n']:4d}  acc=None")
        else:
            print(f"    {b:20s} n={r['n']:4d}  acc={acc*100:5.1f}%  lb={lb*100:5.1f}%")


def main():
    full = build_all_seasons()
    config.save_json(os.path.join(config.DATA_DIR, "weekly.json"), full)
    n = sum(len(w["games"]) for w in full["weeks"])
    print(f"Wrote data/weekly.json: {len(full['weeks'])} weeks, {n} games, "
          f"seasons {full['seasons_available']}")

    train_bands = tb.calibrate_bands(subset(full, TRAIN))
    print(f"\n=== TRAIN bands (2021-2024) ===")
    _fmt_bands(train_bands)

    verify = subset(full, {VERIFY})
    tier_hits, band_hits = grade_holdout(verify, train_bands)

    print(f"\n=== HELD-OUT 2025 realized accuracy (frozen 2021-24 bands applied) ===")
    print("  by tier:")
    for tier in ("best", "value"):
        w, nn = tier_hits.get(tier, [0, 0])
        if nn: print(f"    {tier:6s} {w}/{nn} = {w/nn*100:5.1f}%")
    print("  by band (only bands the ranker actually emitted in 2025):")
    for b in sorted(band_hits):
        w, nn = band_hits[b]
        trained = train_bands.get(b, {})
        ta = trained.get("accuracy")
        tlb = trained.get("accuracy_lb")
        tstr = f"train {ta*100:.1f}% (lb {tlb*100:.1f}%, n={trained.get('n')})" if ta is not None else "train n/a"
        print(f"    {b:20s} 2025: {w}/{nn} = {w/nn*100:5.1f}%   [{tstr}]")

    # Full-pool bands (what the product ships as the calibration table)
    pool_bands = tb.calibrate_bands(full)
    print(f"\n=== FULL-POOL bands (2019-2025, shipped calibration) ===")
    _fmt_bands(pool_bands)
    json.dump({"train_2021_2024": train_bands, "pool_2019_2025": pool_bands,
               "holdout_2025_by_tier": {k: v for k, v in tier_hits.items()},
               "holdout_2025_by_band": {k: v for k, v in band_hits.items()}},
              open("/tmp/lever3_holdout.json", "w"), indent=2)

    # Regenerate the product top_bets.json from the full pool
    out = tb.build_top_bets(full)
    json.dump(out, open(os.path.join(config.DATA_DIR, "top_bets.json"), "w"), indent=1, default=str)
    best = sum(1 for wk in out["weeks"] for g in wk["games"] for x in g["bets"] if x["tier"] == "best")
    val = sum(1 for wk in out["weeks"] for g in wk["games"] for x in g["bets"] if x["tier"] == "value")
    print(f"\nRegenerated data/top_bets.json: {best} BEST / {val} VALUE over {len(out['weeks'])} weeks")


if __name__ == "__main__":
    main()
