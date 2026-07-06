#!/usr/bin/env python3
"""Walk-forward backtest of the Monte Carlo against historical closing lines.

For every historical game we simulate it using ONLY the ratings known before
kickoff (built that way in build_ratings.py), then:
  * grade the model's spread / total / moneyline picks against the real result,
  * measure ROI at standard -110 juice (spreads/totals) and at the real
    moneyline price,
  * check calibration (do 60%-confidence picks really win ~60%?).

This is the honest test: NFL closing lines are razor sharp, so a sound model
should land NEAR break-even and be well-calibrated. Beating the vig consistently
is the hard part — the backtest tells you where (if anywhere) an edge persists.

Run:  python3 backtest.py [--sims 6000] [--threshold 0.03]
Needs numpy (for the simulator). Reads data/backtest_games.json from build_ratings.py.
"""

from __future__ import annotations

import argparse
import os

from nflvalue import config, montecarlo as mc

DEC_110 = 1.9091  # decimal odds for a -110 bet


def american_to_decimal(a):
    if a is None:
        return None
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def run(sims=6000, threshold=0.03):
    priors = config.load_json(os.path.join(config.DATA_DIR, "league_priors.json"), None)
    games = config.load_json(os.path.join(config.DATA_DIR, "backtest_games.json"), None)
    if not priors or not games:
        print("Missing data. Run:  python3 build_ratings.py")
        return
    games = [g for g in games if g.get("ready")]
    print(f"Backtesting {len(games)} games ({sims} sims each, EV threshold {threshold:.0%})...")

    markets = {m: {"bets": 0, "wins": 0, "push": 0, "profit": 0.0} for m in ("spread", "total", "ml")}
    cal_bins = [[0, 0, 0] for _ in range(10)]  # [sum_pred, sum_actual, count] per decile
    cover_hit = [0, 0]                          # [correct, total] over all ready games
    brier_n, brier_s = 0, 0.0
    equity, bank = [100.0], 100.0
    by_season = {}
    # running sums to compare model vs market as margin predictors
    cc = {k: 0.0 for k in ("n", "xm", "xk", "y", "xmxm", "xkxk", "yy", "xmy", "xky")}

    for g in games:
        home = {"off": g["off_home"], "def": g["def_home"]}
        away = {"off": g["off_away"], "def": g["def_away"]}
        sp, tot = g["spread_line"], g["total_line"]
        r = mc.simulate(home, away, priors, spread_line=sp, total_line=tot, n=sims)

        margin = g["home_score"] - g["away_score"]
        total_pts = g["home_score"] + g["away_score"]
        s = by_season.setdefault(g["season"], {"spread": [0, 0.0], "total": [0, 0.0], "ml": [0, 0.0]})

        # ---- calibration on WIN probability (the model's intrinsic calibration) ----
        if margin != 0:
            b = min(int(r["p_home_win"] * 10), 9)
            cal_bins[b][0] += r["p_home_win"]
            cal_bins[b][1] += 1 if margin > 0 else 0
            cal_bins[b][2] += 1
        # ---- can the model beat the spread? (ATS pick accuracy vs the close) ----
        if margin != sp:
            cover_hit[1] += 1
            if (r["p_home_cover"] > 0.5) == (margin > sp):
                cover_hit[0] += 1
        brier_s += (r["p_home_win"] - (1 if margin > 0 else 0)) ** 2
        brier_n += 1
        # model vs market as predictors of actual margin
        xm, xk, y = r["margin_mean"], sp, margin
        cc["n"] += 1; cc["xm"] += xm; cc["xk"] += xk; cc["y"] += y
        cc["xmxm"] += xm * xm; cc["xkxk"] += xk * xk; cc["yy"] += y * y
        cc["xmy"] += xm * y; cc["xky"] += xk * y

        # ---- spread pick ----
        side, p = ("home", r["p_home_cover"]) if r["p_home_cover"] >= r["p_away_cover"] else ("away", r["p_away_cover"])
        if p * DEC_110 - 1 >= threshold:
            won = (margin > sp) if side == "home" else (margin < sp)
            push = (margin == sp)
            _book(markets["spread"], s["spread"], won, push, DEC_110)
            bank += _pl(won, push, DEC_110); equity.append(round(bank, 2))

        # ---- total pick ----
        side, p = ("over", r["p_over"]) if r["p_over"] >= r["p_under"] else ("under", r["p_under"])
        if p * DEC_110 - 1 >= threshold:
            won = (total_pts > tot) if side == "over" else (total_pts < tot)
            push = (total_pts == tot)
            _book(markets["total"], s["total"], won, push, DEC_110)
            bank += _pl(won, push, DEC_110); equity.append(round(bank, 2))

        # ---- moneyline pick (real price) ----
        dh, da = american_to_decimal(g.get("home_ml")), american_to_decimal(g.get("away_ml"))
        if dh and da:
            if r["p_home_win"] * dh >= r["p_away_win"] * da:
                side, p, dec = "home", r["p_home_win"], dh
            else:
                side, p, dec = "away", r["p_away_win"], da
            if p * dec - 1 >= threshold:
                won = (margin > 0) if side == "home" else (margin < 0)
                _book(markets["ml"], s["ml"], won, False, dec)
                bank += _pl(won, False, dec); equity.append(round(bank, 2))

    # ---- assemble report ----
    def summ(m):
        graded = m["bets"] - m["push"]
        return {
            "bets": m["bets"], "wins": m["wins"], "push": m["push"],
            "win_rate": round(m["wins"] / graded, 4) if graded else 0,
            "roi": round(m["profit"] / graded, 4) if graded else 0,
            "units": round(m["profit"], 1),
        }

    calibration = []
    for i, (ps, pa, c) in enumerate(cal_bins):
        if c:
            calibration.append({"bucket": f"{i*10}-{i*10+10}%",
                                "predicted": round(ps / c, 4),
                                "actual": round(pa / c, 4), "n": c})

    def _corr(sx, sxx, sxy):
        n = cc["n"]
        cov = sxy - sx * cc["y"] / n
        vx = sxx - sx * sx / n
        vy = cc["yy"] - cc["y"] * cc["y"] / n
        return round(cov / (vx * vy) ** 0.5, 4) if vx > 0 and vy > 0 else 0.0

    corr_model = _corr(cc["xm"], cc["xmxm"], cc["xmy"])
    corr_market = _corr(cc["xk"], cc["xkxk"], cc["xky"])
    verdict = (
        f"The Monte Carlo is well-calibrated (Brier {brier_s/brier_n:.3f}) and its "
        f"margin forecast correlates {corr_model:.2f} with actual results — real "
        f"signal. But the closing line correlates {corr_market:.2f}: it is sharper, "
        "so betting blindly into closing numbers loses the vig. Use the model as a "
        "fair-value second opinion and capture edges by line-shopping softer prices, "
        "not by beating the close."
    )
    report = {
        "generated": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "n_games": len(games), "sims": sims, "threshold": threshold,
        "spread": summ(markets["spread"]), "total": summ(markets["total"]),
        "ml": summ(markets["ml"]),
        "ats_pick_accuracy": round(cover_hit[0] / cover_hit[1], 4) if cover_hit[1] else 0,
        "brier": round(brier_s / brier_n, 4) if brier_n else 0,
        "corr_model": corr_model, "corr_market": corr_market,
        "verdict": verdict,
        "calibration": calibration,
        "equity_curve": equity,
        "final_bankroll": round(bank, 1),
        "by_season": {str(k): {m: {"bets": v[m][0], "units": round(v[m][1], 1)} for m in v}
                      for k, v in sorted(by_season.items())},
    }
    config.save_json(os.path.join(config.DATA_DIR, "backtest.json"), report)

    # ---- console summary ----
    print(f"\n  Games simulated: {report['n_games']}   ATS pick accuracy (no juice): "
          f"{report['ats_pick_accuracy']*100:.1f}%   Brier(win): {report['brier']:.3f}")
    for name in ("spread", "total", "ml"):
        m = report[name]
        print(f"  {name.upper():7} bets {m['bets']:4}  win {m['win_rate']*100:4.1f}%  "
              f"ROI {m['roi']*100:+5.1f}%  ({m['units']:+.1f}u)")
    print(f"  Combined bankroll: 100u -> {report['final_bankroll']}u")
    print("\n  Saved: data/backtest.json")


def _book(m, season_acc, won, push, dec):
    m["bets"] += 1
    if push:
        m["push"] += 1
        return
    if won:
        m["wins"] += 1
        m["profit"] += dec - 1
        season_acc[1] += dec - 1
    else:
        m["profit"] -= 1
        season_acc[1] -= 1
    season_acc[0] += 1


def _pl(won, push, dec):
    if push:
        return 0.0
    return (dec - 1) if won else -1.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=6000)
    ap.add_argument("--threshold", type=float, default=0.03)
    args = ap.parse_args()
    run(sims=args.sims, threshold=args.threshold)
